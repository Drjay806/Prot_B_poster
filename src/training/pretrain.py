from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.models.compgcn import CompGCN
from src.data.graph_builder import build_annotation_matrix
from src.utils.losses import ranking_loss, cosine_loss, manifold_alignment_loss
from src.utils.schedulers import WarmupCosineScheduler
from src.utils.logger import TrainingLogger

# Max positive pairs sampled per epoch for loss computation.
# 50k × 256-dim float16 ≈ 25 MB — safe on T4 alongside the GCN activation graph.
_MAX_PAIRS_EPOCH = 50_000


def pretrain(
    encoder: CompGCN,
    train_data: HeteroData,
    val_data: HeteroData,
    cfg: dict,
    device: str = "cuda",
    logger: Optional[TrainingLogger] = None,
) -> CompGCN:
    """
    Phase 1: Pre-train CompGCN with four simultaneous losses.

    The encoder is run ONCE per epoch (not once per mini-batch) so we
    only materialise the full-graph activation graph once before the
    backward pass.  A random sample of up to _MAX_PAIRS_EPOCH pairs is
    used for the loss each epoch.
    """
    pt_cfg = cfg["pretrain"]
    epochs     = pt_cfg["epochs"]
    lr         = pt_cfg["lr"]
    wd         = pt_cfg.get("weight_decay", 1e-4)
    num_neg    = pt_cfg.get("num_negatives", 5)
    margin     = pt_cfg.get("ranking_margin", 0.5)
    eval_every = pt_cfg.get("eval_every", 10)
    log_every  = pt_cfg.get("log_every_steps", 1)   # 1 = every epoch (no inner loop)
    warmup     = pt_cfg.get("warmup_epochs", 10)

    target_type = cfg["data"]["target_type"]

    encoder.set_dropout("pretrain")
    encoder.train()

    loss_weights = nn.Parameter(torch.ones(4, device=device) / 4.0)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + [loss_weights], lr=lr, weight_decay=wd
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup, total_epochs=epochs)

    print("Building annotation index ...")
    protein_indices, go_indices, _, _ = build_annotation_matrix(train_data, target_type)
    protein_indices = protein_indices.to(device)
    go_indices      = go_indices.to(device)
    num_go          = train_data[target_type].x.shape[0]
    n_pairs         = len(protein_indices)

    print(f"Starting pre-training: {epochs} epochs, {n_pairs:,} positive pairs "
          f"(sampling {min(n_pairs, _MAX_PAIRS_EPOCH):,}/epoch)")

    scaler = torch.amp.GradScaler('cuda')
    global_step = 0

    for epoch in range(1, epochs + 1):
        scheduler.step(epoch - 1)
        optimizer.zero_grad()

        # ── Single full-graph forward pass ──────────────────────────────────────
        with torch.amp.autocast('cuda'):
            protein_embs, go_embs, _ = encoder(train_data)

            # Sample pairs for this epoch
            sample_sz = min(n_pairs, _MAX_PAIRS_EPOCH)
            perm      = torch.randperm(n_pairs, device=device)[:sample_sz]
            b_prot    = protein_indices[perm]
            b_go      = go_indices[perm]

            pos_p = protein_embs[b_prot]   # [S, D]
            pos_g = go_embs[b_go]           # [S, D]

            neg_go_idx = torch.randint(0, num_go, (sample_sz * num_neg,), device=device)
            neg_g = go_embs[neg_go_idx].view(sample_sz, num_neg, -1)   # [S, neg, D]

            l_mse  = F.mse_loss(pos_p, pos_g)
            l_cos  = cosine_loss(pos_p, pos_g)
            pos_sc = (pos_p * pos_g).sum(-1, keepdim=True)              # [S, 1]
            neg_sc = (pos_p.unsqueeze(1) * neg_g).sum(-1)               # [S, neg]
            l_rank = ranking_loss(
                pos_sc.expand_as(neg_sc).reshape(-1),
                neg_sc.reshape(-1),
                margin=margin,
            )
            l_mmd = manifold_alignment_loss(pos_p, pos_g)

            losses = torch.stack([l_mse, l_cos, l_rank, l_mmd])
            w = F.softmax(loss_weights, dim=0)
            total_loss = (w * losses).sum()

        # ── Single backward + step ──────────────────────────────────────────────
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        vals = {
            "total":   total_loss.item(),
            "mse":     l_mse.item(),
            "cosine":  l_cos.item(),
            "ranking": l_rank.item(),
            "mmd":     l_mmd.item(),
        }

        with torch.no_grad():
            p_norm = protein_embs.norm(dim=-1).mean().item()
            g_norm = go_embs.norm(dim=-1).mean().item()

        # Validation
        val_cos = None
        if epoch % eval_every == 0 or epoch == epochs:
            val_cos = _val_cosine(encoder, val_data, target_type, cfg, device)
            if logger:
                logger.log({"val/cosine_similarity": val_cos}, step=global_step, phase="pretrain")

        if logger:
            metrics = {f"loss/{k}": v for k, v in vals.items()}
            metrics["embed/protein_norm_mean"] = p_norm
            metrics["embed/go_norm_mean"]      = g_norm
            logger.log(metrics, step=global_step, phase="pretrain")
            logger.log_pretrain_epoch(epoch, epochs, vals, val_cos)
            if epoch % log_every == 0:
                logger.check_health(global_step)
        else:
            parts   = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
            val_str = f"  val_cos={val_cos:.4f}" if val_cos is not None else ""
            print(f"[Pretrain {epoch}/{epochs}] {parts}{val_str}")

    return encoder


@torch.no_grad()
def _val_cosine(encoder: CompGCN, val_data: HeteroData, target_type: str,
                cfg: dict, device: str) -> float:
    encoder.eval()
    with torch.amp.autocast('cuda'):
        protein_embs, go_embs, _ = encoder(val_data)

    from src.data.graph_builder import build_annotation_matrix
    row, col, _, _ = build_annotation_matrix(val_data, target_type)
    p_emb = protein_embs[row.to(device)]
    g_emb = go_embs[col.to(device)]
    score = F.cosine_similarity(p_emb, g_emb, dim=-1).mean().item()
    encoder.train()
    return score
