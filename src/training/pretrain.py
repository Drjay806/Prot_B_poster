from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.data.graph_builder import build_annotation_matrix
from src.utils.losses import infonce_loss, cosine_loss, ranking_loss
from src.utils.schedulers import WarmupCosineScheduler
from src.utils.logger import TrainingLogger

# InfoNCE batch: [S x S] similarity matrix = S^2 float32 entries.
# At S=4096: 67 MB on-GPU, safe alongside the encoder activation graph on T4.
_MAX_PAIRS_EPOCH = 4_096


def pretrain(
    encoder: CompGCN,
    train_data: HeteroData,
    val_data: HeteroData,
    cfg: dict,
    device: str = "cuda",
    logger: Optional[TrainingLogger] = None,
) -> CompGCN:
    """
    Phase 1 (v2): Pre-train CompGCN with InfoNCE + cosine + DistMult link prediction.

    ── What changed from v1 and why ──────────────────────────────────────────────

    REMOVED — MSE loss
        MSE penalises differences in L2 magnitude, pulling the protein embedding
        toward the exact numerical value of the GO embedding.  InfoNCE operates on
        the unit sphere (after F.normalize), so L2 magnitude is irrelevant.  MSE
        was actively fighting InfoNCE and slowing learning.

    REMOVED — MMD manifold alignment loss
        MMD is O(n^2) in subsampled pairs.  It aligns the *global distribution* of
        protein and GO embeddings but provides zero signal about WHICH protein goes
        with WHICH GO term.  Fmax is a per-protein ranking metric, not a distribution
        alignment metric.  MMD was contributing noise, not signal.

    REMOVED — learnable loss weights (softmax over nn.Parameter)
        With 4 losses of very different magnitudes and objectives, the softmax
        converged to nearly uniform weights — it learned nothing.  Fixed weights
        are more predictable and easier to tune.

    ADDED — InfoNCE contrastive loss  (Chen et al. 2020, SimCLR)
        For every protein in the batch, all other GO terms in that same batch
        become negatives automatically.  At batch size 4096 this is 4095 negatives
        per protein vs 5 in the old ranking loss — an 800x increase in negative
        coverage.  InfoNCE directly optimises the cross-batch ranking that Fmax
        measures at evaluation time.  This is the dominant loss (weight=1.0).

        Literature precedent for protein function: ProtST (Xu et al. 2023),
        DeepGraphGO (You et al. 2021), and TALE (Guo et al. 2022) all use
        InfoNCE-style contrastive objectives for GO function prediction and report
        Fmax in the 0.45–0.60 range on BP — far above our v1 baseline of 0.097.

    ADDED — DistMult link-prediction loss
        The has_function relation embedding in CompGCN was NEVER given a gradient
        signal in v1.  It remained random noise.  This meant Phase 2's DistMult
        scoring was completely random, and Phase 3's hierarchy penalty (which uses
        the same relation embedding) was always 1.000 — a constant with zero
        gradient.  Adding (protein * relation * GO).sum() > margin over random GO
        negatives trains the relation embedding to encode "protein has function"
        semantics.  After Phase 1 the relation embedding will be meaningful, and
        Phase 2 DistMult scores will show real separation (not flat at ~0).

    KEPT — Cosine alignment (weight=0.1)
        Light regulariser keeping protein embedding direction aligned with its true
        GO terms.  Without it, InfoNCE would optimise relative rankings but the
        encoder might drift so that absolute directions are meaningless (matters
        for Phase 2 anchor loss and Phase 3 hierarchy penalty).

    ── Expected outcome ──────────────────────────────────────────────────────────
    val_cos will stay high (~0.95–0.97) — cosine loss is still present.
    val_Fmax (now printed every eval_every epochs) should reach 0.30–0.45
    by epoch 50, up from 0.097.  This is now printed during pretrain so you
    can see the progress in real time without running Cell 11.
    """
    pt_cfg      = cfg["pretrain"]
    epochs      = pt_cfg["epochs"]
    lr          = pt_cfg["lr"]
    wd          = pt_cfg.get("weight_decay", 1e-4)
    num_neg     = pt_cfg.get("num_negatives", 32)         # only for DistMult LP negatives
    temperature = pt_cfg.get("infonce_temperature", 0.07)
    dm_weight   = pt_cfg.get("distmult_weight", 0.5)
    eval_every  = pt_cfg.get("eval_every", 10)
    warmup      = pt_cfg.get("warmup_epochs", 10)

    target_type = cfg["data"]["target_type"]

    encoder.set_dropout("pretrain")
    encoder.train()

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=wd)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup, total_epochs=epochs)

    print("Building annotation index ...")
    protein_indices, go_indices, _, _ = build_annotation_matrix(train_data, target_type)
    protein_indices = protein_indices.to(device)
    go_indices      = go_indices.to(device)
    num_go          = train_data[target_type].x.shape[0]
    n_pairs         = len(protein_indices)

    # Get has_function relation index once before training loop
    from src.training.adversarial import _get_has_function_rel_idx
    rel_idx = _get_has_function_rel_idx(encoder)

    print(f"Starting pre-training: {epochs} epochs, {n_pairs:,} positive pairs")
    print(f"  InfoNCE batch={min(n_pairs, _MAX_PAIRS_EPOCH):,}  temp={temperature}  dm_neg={num_neg}")
    print(f"  Loss weights: 1.0*InfoNCE  0.1*cosine  {dm_weight}*DistMult-LP")

    global_step = 0

    for epoch in range(1, epochs + 1):
        scheduler.step(epoch - 1)
        optimizer.zero_grad()

        # Single full-graph forward (float32 — FFT in CompGCN is incompatible with float16)
        protein_embs, go_embs, rel_embs = encoder(train_data)
        rel_vec = rel_embs[rel_idx]   # [D] — has_function relation embedding

        # Sample a batch of positive (protein, GO) pairs for this epoch
        sample_sz = min(n_pairs, _MAX_PAIRS_EPOCH)
        perm      = torch.randperm(n_pairs, device=device)[:sample_sz]
        b_prot    = protein_indices[perm]
        b_go      = go_indices[perm]

        pos_p = protein_embs[b_prot]   # [S, D]
        pos_g = go_embs[b_go]           # [S, D]

        # ── InfoNCE: every other GO term in this batch is a negative ───────────
        # Cross-entropy on [S, S] cosine similarity matrix forces the model to rank
        # protein i's true GO term (diagonal) above all other S-1 GO terms in batch.
        l_infonce = infonce_loss(pos_p, pos_g, temperature=temperature)

        # ── Cosine alignment: keep protein direction aligned with true GO ───────
        l_cos = cosine_loss(pos_p, pos_g)

        # ── DistMult link prediction: trains the has_function relation embedding ─
        # DistMult score: (protein * relation * GO).sum().
        # Positive pairs > random GO negatives by margin=1.0.
        # No learnable parameters in DistMult itself — gradient flows through
        # encoder's relation embedding, giving it its first real learning signal.
        r           = rel_vec.unsqueeze(0)                                   # [1, D]
        pos_dm      = (pos_p * r * pos_g).sum(-1)                           # [S]
        neg_go_idx  = torch.randint(0, num_go, (sample_sz * num_neg,), device=device)
        neg_g_flat  = go_embs[neg_go_idx]                                   # [S*neg, D]
        pos_p_rep   = pos_p.repeat_interleave(num_neg, dim=0)               # [S*neg, D]
        r_rep       = r.expand(sample_sz * num_neg, -1)                     # [S*neg, D]
        neg_dm      = (pos_p_rep * r_rep * neg_g_flat).sum(-1)             # [S*neg]
        pos_dm_rep  = pos_dm.repeat_interleave(num_neg)                     # [S*neg]
        l_dm        = ranking_loss(pos_dm_rep, neg_dm, margin=1.0)

        # Fixed weights — InfoNCE dominates, cosine and DistMult support it
        total_loss = 1.0 * l_infonce + 0.1 * l_cos + dm_weight * l_dm

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()

        global_step += 1

        vals = {
            "total":    total_loss.item(),
            "infonce":  l_infonce.item(),
            "cosine":   l_cos.item(),
            "distmult": l_dm.item(),
        }

        with torch.no_grad():
            p_norm = protein_embs.norm(dim=-1).mean().item()
            g_norm = go_embs.norm(dim=-1).mean().item()

        val_cos  = None
        val_fmax = None
        if epoch % eval_every == 0 or epoch == epochs:
            val_cos  = _val_cosine(encoder, val_data, target_type, device)
            val_fmax = _val_fmax(encoder, val_data, target_type, device)
            if logger:
                logger.log({
                    "val/cosine_similarity": val_cos,
                    "val/fmax_bp": val_fmax,
                }, step=global_step, phase="pretrain")

        if logger:
            metrics = {f"loss/{k}": v for k, v in vals.items()}
            metrics["embed/protein_norm_mean"] = p_norm
            metrics["embed/go_norm_mean"]      = g_norm
            logger.log(metrics, step=global_step, phase="pretrain")
        else:
            parts    = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
            val_str  = f"  val_cos={val_cos:.4f}" if val_cos is not None else ""
            fmax_str = f"  val_Fmax={val_fmax:.4f}" if val_fmax is not None else ""
            print(f"[Pretrain {epoch}/{epochs}] {parts}{val_str}{fmax_str}")

    return encoder


@torch.no_grad()
def _val_cosine(encoder: CompGCN, val_data: HeteroData, target_type: str, device: str) -> float:
    """Mean cosine similarity between proteins and their true GO terms on the val set."""
    encoder.eval()
    protein_embs, go_embs, _ = encoder(val_data)
    from src.data.graph_builder import build_annotation_matrix
    row, col, _, _ = build_annotation_matrix(val_data, target_type)
    p_emb = protein_embs[row.to(device)]
    g_emb = go_embs[col.to(device)]
    score = F.cosine_similarity(p_emb, g_emb, dim=-1).mean().item()
    encoder.train()
    return score


@torch.no_grad()
def _val_fmax(
    encoder: CompGCN,
    val_data: HeteroData,
    target_type: str,
    device: str,
    n_sample: int = 2000,
    t_steps: int = 50,
) -> float:
    """
    Fast approximate Fmax on a random sample of val proteins.

    Uses cosine similarity (matches what InfoNCE actually optimises) and sweeps
    thresholds across the actual score range — not a fixed [0,1] — so the threshold
    can meaningfully separate true from false annotations even when all scores
    cluster in 0.85–0.97 (the 'GO embedding compression' problem from v1).
    """
    from src.data.graph_builder import build_annotation_matrix
    encoder.eval()
    protein_embs, go_embs, _ = encoder(val_data)
    row, col, n_p, n_go = build_annotation_matrix(val_data, target_type)
    row_cpu, col_cpu = row.cpu(), col.cpu()

    unique_prots = row_cpu.unique()
    if len(unique_prots) > n_sample:
        unique_prots = unique_prots[torch.randperm(len(unique_prots))[:n_sample]]
    n_s = len(unique_prots)

    prot_map = {p.item(): i for i, p in enumerate(unique_prots)}
    mask  = torch.isin(row_cpu, unique_prots)
    s_row = torch.tensor([prot_map[p.item()] for p in row_cpu[mask]], dtype=torch.long)
    s_col = col_cpu[mask]
    true_mat = torch.zeros(n_s, n_go, dtype=torch.float32)
    true_mat[s_row, s_col] = 1.0

    p_norm = F.normalize(protein_embs[unique_prots.to(device)].float(), dim=-1)
    g_norm = F.normalize(go_embs.float(), dim=-1)
    scores = (p_norm @ g_norm.t()).cpu()   # [n_s, n_go]

    score_min = scores.min().item()
    score_max = scores.max().item()
    true_pos  = true_mat.sum(dim=1).clamp(min=1e-8)
    best_fmax = 0.0

    for i in range(t_steps + 1):
        t    = score_min + i * (score_max - score_min) / t_steps
        pred = (scores >= t).float()
        tp   = (pred * true_mat).sum(dim=1)
        pp   = pred.sum(dim=1).clamp(min=1e-8)
        prec = (tp / pp).mean().item()
        rec  = (tp / true_pos).mean().item()
        if prec + rec > 0:
            best_fmax = max(best_fmax, 2 * prec * rec / (prec + rec))

    encoder.train()
    return best_fmax
