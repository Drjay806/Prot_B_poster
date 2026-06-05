from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.models.compgcn import CompGCN
from src.data.graph_builder import build_dense_annotation_matrix
from src.utils.losses import ranking_loss, cosine_loss, manifold_alignment_loss
from src.utils.schedulers import WarmupCosineScheduler
from src.utils.logger import TrainingLogger


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
    Returns the encoder with updated weights.
    """
    pt_cfg = cfg["pretrain"]
    epochs = pt_cfg["epochs"]
    batch_size = pt_cfg["batch_size"]
    lr = pt_cfg["lr"]
    wd = pt_cfg.get("weight_decay", 1e-4)
    num_neg = pt_cfg.get("num_negatives", 5)
    margin = pt_cfg.get("ranking_margin", 0.5)
    eval_every = pt_cfg.get("eval_every", 10)
    log_every = pt_cfg.get("log_every_steps", 50)
    warmup = pt_cfg.get("warmup_epochs", 10)

    target_type = cfg["data"]["target_type"]

    encoder.set_dropout("pretrain")
    encoder.train()

    # Learnable loss weights (one per loss term)
    loss_weights = nn.Parameter(torch.ones(4, device=device) / 4.0)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + [loss_weights], lr=lr, weight_decay=wd
    )
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup, total_epochs=epochs)

    # Precompute annotation matrix to sample positives/negatives
    print("Building annotation matrix ...")
    ann = build_dense_annotation_matrix(train_data, target_type)   # [N_p, N_go] on same device as data
    protein_indices, go_indices = ann.nonzero(as_tuple=True)
    protein_indices = protein_indices.to(device)
    go_indices = go_indices.to(device)
    num_go = train_data[target_type].x.shape[0]

    print(f"Starting pre-training: {epochs} epochs, {len(protein_indices):,} positive pairs")

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    for epoch in range(1, epochs + 1):
        scheduler.step(epoch - 1)
        perm = torch.randperm(len(protein_indices), device=device)
        protein_indices_shuf = protein_indices[perm]
        go_indices_shuf = go_indices[perm]

        epoch_losses: Dict[str, list] = {k: [] for k in ["total", "mse", "cosine", "ranking", "mmd"]}
        num_batches = 0

        for start in range(0, len(protein_indices_shuf), batch_size):
            end = min(start + batch_size, len(protein_indices_shuf))
            b_prot_idx = protein_indices_shuf[start:end]
            b_go_idx = go_indices_shuf[start:end]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                protein_embs, go_embs, _ = encoder(train_data)

                # Positive embeddings
                pos_p_emb = protein_embs[b_prot_idx]   # [B, D]
                pos_g_emb = go_embs[b_go_idx]           # [B, D]

                # Negative GO samples (random, not in protein's annotations)
                neg_go_idx = torch.randint(0, num_go, (len(b_prot_idx) * num_neg,), device=device)
                neg_g_emb = go_embs[neg_go_idx].view(len(b_prot_idx), num_neg, -1)  # [B, neg, D]

                # L1: MSE between protein and its positive GO embedding
                l_mse = F.mse_loss(pos_p_emb, pos_g_emb)

                # L2: Cosine similarity loss
                l_cos = cosine_loss(pos_p_emb, pos_g_emb)

                # L3: Ranking loss — positive GO should score higher than each negative
                pos_score = (pos_p_emb * pos_g_emb).sum(-1, keepdim=True)   # [B, 1]
                neg_scores = (pos_p_emb.unsqueeze(1) * neg_g_emb).sum(-1)   # [B, neg]
                l_rank = ranking_loss(
                    pos_score.expand_as(neg_scores).reshape(-1),
                    neg_scores.reshape(-1),
                    margin=margin,
                )

                # L4: Manifold alignment — protein and GO distributions should overlap
                l_mmd = manifold_alignment_loss(pos_p_emb, pos_g_emb)

                losses = torch.stack([l_mse, l_cos, l_rank, l_mmd])
                w = F.softmax(loss_weights, dim=0)
                total_loss = (w * losses).sum()

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            num_batches += 1

            vals = {
                "total": total_loss.item(),
                "mse": l_mse.item(),
                "cosine": l_cos.item(),
                "ranking": l_rank.item(),
                "mmd": l_mmd.item(),
            }
            for k, v in vals.items():
                epoch_losses[k].append(v)

            # Embedding health monitoring
            with torch.no_grad():
                p_norm = protein_embs.norm(dim=-1).mean().item()
                g_norm = go_embs.norm(dim=-1).mean().item()

            if logger and global_step % log_every == 0:
                metrics = {f"loss/{k}": v for k, v in vals.items()}
                metrics["embed/protein_norm_mean"] = p_norm
                metrics["embed/go_norm_mean"] = g_norm
                logger.log(metrics, step=global_step, phase="pretrain")
                logger.check_health(global_step)

        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}

        # Validation every eval_every epochs
        val_cos = None
        if epoch % eval_every == 0 or epoch == epochs:
            val_cos = _val_cosine(encoder, val_data, target_type, cfg, device)
            if logger:
                logger.log({"val/cosine_similarity": val_cos}, step=global_step, phase="pretrain")

        if logger:
            logger.log_pretrain_epoch(epoch, epochs, avg, val_cos)
        else:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            val_str = f"  val_cos={val_cos:.4f}" if val_cos else ""
            print(f"[Pretrain {epoch}/{epochs}] {parts}{val_str}")

    return encoder


@torch.no_grad()
def _val_cosine(encoder: CompGCN, val_data: HeteroData, target_type: str, cfg: dict, device: str) -> float:
    encoder.eval()
    with torch.cuda.amp.autocast():
        protein_embs, go_embs, _ = encoder(val_data)

    from src.data.graph_builder import build_annotation_matrix
    row, col, _, _ = build_annotation_matrix(val_data, target_type)
    p_emb = protein_embs[row.to(device)]
    g_emb = go_embs[col.to(device)]
    score = F.cosine_similarity(p_emb, g_emb, dim=-1).mean().item()
    encoder.train()
    return score
