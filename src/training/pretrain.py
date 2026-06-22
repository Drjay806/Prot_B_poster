from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.data.graph_builder import build_annotation_matrix
from src.utils.losses import infonce_loss, cosine_loss, ranking_loss
from src.utils.schedulers import WarmupCosineScheduler
from src.utils.logger import TrainingLogger

# InfoNCE batch: [S x S] similarity matrix = S^2 float32 entries.
# At S=4096: 67 MB on-GPU — too large on T4 once the 3-layer CompGCN forward
# is also live.  Default 1024 (4 MB sim matrix); configurable via pretrain.infonce_batch.
_MAX_PAIRS_EPOCH = 1_024


def pretrain(
    encoder: CompGCN,
    train_data: HeteroData,
    val_data: HeteroData,
    cfg: dict,
    device: str = "cuda",
    logger: Optional[TrainingLogger] = None,
    direct_parents: Optional[Dict[int, List[int]]] = None,
) -> CompGCN:
    """
    Phase 1 (v3): Pre-train CompGCN with:
      1. Symmetric InfoNCE with false-negative masking   (weight 1.0)
      2. Cosine alignment                                 (weight 0.1)
      3. ComplEx link-prediction loss                     (weight dm_weight)
      4. Order-consistency hierarchy loss [optional]      (weight oc_weight)

    ── Change log from v2 ────────────────────────────────────────────────────────

    FIXED — InfoNCE is now symmetric (L_p2g + L_g2p averaged)
        Previously only protein→GO direction.  Adding GO→protein doubles gradient
        signal per batch at zero extra memory — same [S,S] matrix transposed.

    FIXED — False-negative masking in InfoNCE
        A protein often has 10–50 true GO annotations.  When two batch samples
        share a GO term, the "negative" pair is a real positive — treating it as
        negative penalises correct predictions.  We pre-build a [S,S] boolean
        mask (prot_to_gos_tensor lookup) and set sim[i,j]=-inf for known off-diagonal
        positives before the softmax.  This stops ~0.18% of "negatives" per batch
        from being false penalties.

    FIXED — ComplEx formula for LP loss (was DistMult)
        The original inline code used `(p * r * g).sum()` (DistMult: symmetric).
        has_function is NOT symmetric.  The correct four-term ComplEx formula:
            Re(p * r * conj(g)) = p_re*r_re*g_re + p_re*r_im*g_im
                                 + p_im*r_re*g_im - p_im*r_im*g_re
        This trains the relation embedding to encode the asymmetric one-to-many
        structure of protein→GO annotations.

    ADDED — Order-consistency hierarchy loss (optional, requires direct_parents)
        For each (protein, child_GO) annotation in the batch, the child's direct
        parents in the GO hierarchy should score at LEAST as high as the child.
        Hinge penalty: max(0, score_child − score_parent) for each (child, parent) pair.
        This teaches the embedding space that more general GO terms always outrank
        their specific children — a structural requirement of GO.  Requires passing
        direct_parents=build_direct_parents(train_data) from the notebook.

    ── Pass direct_parents for order-consistency ─────────────────────────────────
    In the notebook (Cell 7 or Cell 8 header):
        from src.data.go_hierarchy import build_direct_parents
        direct_parents = build_direct_parents(train_data)
    Then: encoder = pretrain(..., direct_parents=direct_parents)

    ── Expected outcome ──────────────────────────────────────────────────────────
    val_Fmax (printed every eval_every epochs) should reach 0.35–0.50 by epoch 50.
    The symmetry fix and false-neg masking together add roughly +0.05–0.08 Fmax
    over v2 because the model stops being punished for correctly-placed predictions.
    """
    pt_cfg   = cfg["pretrain"]
    epochs   = pt_cfg["epochs"]
    lr       = pt_cfg["lr"]
    wd       = pt_cfg.get("weight_decay", 1e-4)
    num_neg  = pt_cfg.get("num_negatives", 32)
    temperature = pt_cfg.get("infonce_temperature", 0.07)
    dm_weight   = pt_cfg.get("distmult_weight", 0.5)
    oc_weight   = pt_cfg.get("order_consistency_weight", 0.3)
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

    # Build per-protein GO lookup (on CPU) for false-negative masking.
    # prot_to_gos_tensor[p] = 1-D long tensor of GO indices annotated to protein p.
    print("Building false-negative mask lookup ...")
    _prot_go_lists: Dict[int, List[int]] = {}
    for p_i, g_i in zip(protein_indices.cpu().tolist(), go_indices.cpu().tolist()):
        if p_i not in _prot_go_lists:
            _prot_go_lists[p_i] = []
        _prot_go_lists[p_i].append(g_i)
    prot_to_gos_tensor = {
        p: torch.tensor(gos, dtype=torch.long)
        for p, gos in _prot_go_lists.items()
    }
    del _prot_go_lists

    from src.training.adversarial import _get_has_function_rel_idx
    rel_idx = _get_has_function_rel_idx(encoder)

    infonce_batch = pt_cfg.get("infonce_batch", _MAX_PAIRS_EPOCH)

    use_oc = (direct_parents is not None)
    oc_str = f"  {oc_weight}*OrderConsistency" if use_oc else ""
    print(f"Starting pre-training: {epochs} epochs, {n_pairs:,} positive pairs")
    print(f"  InfoNCE batch={min(n_pairs, infonce_batch):,}  temp={temperature}  dm_neg={num_neg}")
    print(f"  Loss weights: 1.0*InfoNCE(sym+mask)  0.1*cosine  {dm_weight}*ComplEx-LP{oc_str}")

    # Print node type sizes so users can spot any unexpectedly large node types (e.g. ChEMBL)
    print("  Node type sizes in graph:")
    for ntype in train_data.node_types:
        if hasattr(train_data[ntype], "x") and train_data[ntype].x is not None:
            print(f"    {ntype}: {train_data[ntype].x.shape[0]:,} nodes")
        elif hasattr(train_data[ntype], "num_nodes"):
            print(f"    {ntype}: {train_data[ntype].num_nodes:,} nodes (no features)")

    global_step = 0
    half = encoder.output_dim // 2  # ComplEx splits dim into real / imaginary halves

    for epoch in range(1, epochs + 1):
        torch.cuda.empty_cache()   # flush fragmented cache before each encoder forward
        scheduler.step(epoch - 1)
        optimizer.zero_grad()

        protein_embs, go_embs, rel_embs = encoder(train_data)
        rel_vec = rel_embs[rel_idx]   # [D]

        sample_sz = min(n_pairs, infonce_batch)
        perm      = torch.randperm(n_pairs, device=device)[:sample_sz]
        b_prot    = protein_indices[perm]
        b_go      = go_indices[perm]

        pos_p = protein_embs[b_prot]   # [S, D]
        pos_g = go_embs[b_go]           # [S, D]

        # ── False-negative mask: [S, S] bool, True = known off-diagonal positive ──
        neg_mask = _build_false_neg_mask(b_prot, b_go, prot_to_gos_tensor, device)

        # ── 1. Symmetric InfoNCE with false-negative masking ─────────────────────
        l_infonce = infonce_loss(pos_p, pos_g, temperature=temperature, neg_mask=neg_mask)

        # ── 2. Cosine alignment ────────────────────────────────────────────────────
        l_cos = cosine_loss(pos_p, pos_g)

        # ── 3. ComplEx link-prediction loss ───────────────────────────────────────
        # Trains the has_function relation embedding with the correct asymmetric
        # ComplEx formula instead of the old symmetric DistMult product.
        # Positive (protein, has_function, GO) pairs score above random GO negatives.
        r = rel_vec.unsqueeze(0)           # [1, D]

        # ComplEx positive scores
        p_re, p_im   = pos_p[:, :half], pos_p[:, half:]       # [S, D/2] each
        r_re, r_im   = r[:, :half],     r[:, half:]            # [1, D/2] each
        g_re, g_im   = pos_g[:, :half], pos_g[:, half:]        # [S, D/2] each
        pos_dm = (
              p_re * r_re * g_re
            + p_re * r_im * g_im
            + p_im * r_re * g_im
            - p_im * r_im * g_re
        ).sum(-1)   # [S]

        # Random GO negative scores
        neg_go_idx = torch.randint(0, num_go, (sample_sz * num_neg,), device=device)
        neg_g_flat  = go_embs[neg_go_idx]                                  # [S*neg, D]
        pos_p_rep   = pos_p.repeat_interleave(num_neg, dim=0)              # [S*neg, D]
        r_rep       = r.expand(sample_sz * num_neg, -1)                    # [S*neg, D]

        ng_re, ng_im  = neg_g_flat[:, :half], neg_g_flat[:, half:]
        prp_re, prp_im = pos_p_rep[:, :half], pos_p_rep[:, half:]
        r_rp_re, r_rp_im = r_rep[:, :half], r_rep[:, half:]
        neg_dm = (
              prp_re * r_rp_re * ng_re
            + prp_re * r_rp_im * ng_im
            + prp_im * r_rp_re * ng_im
            - prp_im * r_rp_im * ng_re
        ).sum(-1)   # [S*neg]

        pos_dm_rep = pos_dm.repeat_interleave(num_neg)    # [S*neg]
        l_dm = ranking_loss(pos_dm_rep, neg_dm, margin=1.0)

        total_loss = 1.0 * l_infonce + 0.1 * l_cos + dm_weight * l_dm

        # ── 4. Order-consistency hierarchy loss (optional) ────────────────────────
        # For each (protein, child_GO) pair in batch, the direct parents of child_GO
        # must score at least as high as the child under ComplEx.
        # Vectorised: collect all (protein_row, child_go, parent_go) triples, then
        # do a single batched ComplEx forward — no Python loops over scores.
        l_oc = torch.tensor(0.0, device=device)
        if use_oc:
            l_oc = _order_consistency_loss(
                pos_p, rel_vec, go_embs, b_go, direct_parents, half, device
            )
            total_loss = total_loss + oc_weight * l_oc

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
        if use_oc:
            vals["order_cons"] = l_oc.item()

        with torch.no_grad():
            p_norm_val = protein_embs.norm(dim=-1).mean().item()
            g_norm_val = go_embs.norm(dim=-1).mean().item()

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
            metrics["embed/protein_norm_mean"] = p_norm_val
            metrics["embed/go_norm_mean"]      = g_norm_val
            logger.log(metrics, step=global_step, phase="pretrain")
        else:
            parts    = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
            val_str  = f"  val_cos={val_cos:.4f}" if val_cos is not None else ""
            fmax_str = f"  val_Fmax={val_fmax:.4f}" if val_fmax is not None else ""
            print(f"[Pretrain {epoch}/{epochs}] {parts}{val_str}{fmax_str}")

    return encoder


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_false_neg_mask(
    b_prot: torch.Tensor,
    b_go: torch.Tensor,
    prot_to_gos_tensor: Dict[int, torch.Tensor],
    device: str,
) -> torch.Tensor:
    """
    Returns a [S, S] bool tensor where entry (i, j) is True when protein b_prot[i]
    has b_go[j] as a known annotation AND i ≠ j.

    These are 'false negatives' in the InfoNCE batch — pairs that look like negatives
    (off-diagonal) but are actually true positives.  We set them to -inf in the
    similarity matrix so the loss doesn't penalise them.

    Implementation: for each protein in the batch, we call torch.isin (vectorised)
    to check which batch GO terms are in that protein's annotation set.  One call
    per unique protein in the batch — fast even at S=4096.
    """
    S        = b_prot.size(0)
    b_go_cpu = b_go.cpu()
    mask     = torch.zeros(S, S, dtype=torch.bool)

    for i, prot_i in enumerate(b_prot.cpu().tolist()):
        known = prot_to_gos_tensor.get(prot_i)
        if known is None:
            continue
        col_matches = torch.isin(b_go_cpu, known)  # [S] — which batch GOs are annotated
        col_matches[i] = False                       # diagonal stays positive
        mask[i] = col_matches

    return mask.to(device)


def _order_consistency_loss(
    pos_p: torch.Tensor,
    rel_vec: torch.Tensor,
    go_embs: torch.Tensor,
    b_go: torch.Tensor,
    direct_parents: Dict[int, List[int]],
    half: int,
    device: str,
) -> torch.Tensor:
    """
    Order-consistency: for each annotated (protein, child_GO) pair, every direct
    parent of child_GO must score at least as high as the child under ComplEx.

    Hinge loss: max(0, score_child − score_parent)
    Averaged over all (protein, child, parent) triples in the batch.

    WHY THIS HELPS:
        The GO hierarchy says "if a protein has function X, it also has the more
        general function Y (X's parent)."  Enforcing score(parent) ≥ score(child)
        embeds this structural requirement directly into the encoder's ComplEx scores.
        Without it, the model may rank a specific GO term above a general one even
        when both are annotated — violating the hierarchy and hurting Fmax after
        label propagation.

    Uses a fully-vectorised ComplEx computation: Python loop only collects indices
    (O(batch_size)), then a single batched ComplEx forward on [N_triples, D] tensors.
    """
    b_go_list   = b_go.cpu().tolist()
    row_indices = []   # which row of pos_p to use
    child_gos   = []   # GO term index of the child
    parent_gos  = []   # GO term index of the parent

    for i, go_idx in enumerate(b_go_list):
        parents = direct_parents.get(go_idx, [])
        for parent_idx in parents[:4]:   # cap at 4 direct parents per GO term
            row_indices.append(i)
            child_gos.append(go_idx)
            parent_gos.append(parent_idx)

    if not row_indices:
        return torch.tensor(0.0, device=device)

    rows_t    = torch.tensor(row_indices, device=device)
    child_t   = torch.tensor(child_gos,  device=device)
    parent_t  = torch.tensor(parent_gos, device=device)

    p_batch  = pos_p[rows_t]       # [N, D]
    g_child  = go_embs[child_t]    # [N, D]
    g_parent = go_embs[parent_t]   # [N, D]
    N        = len(row_indices)
    r_rep    = rel_vec.unsqueeze(0).expand(N, -1)  # [N, D]

    def cx(p, r, g):
        p_re, p_im = p[:, :half], p[:, half:]
        r_re, r_im = r[:, :half], r[:, half:]
        g_re, g_im = g[:, :half], g[:, half:]
        return (
              p_re * r_re * g_re
            + p_re * r_im * g_im
            + p_im * r_re * g_im
            - p_im * r_im * g_re
        ).sum(-1)  # [N]

    score_child  = cx(p_batch, r_rep, g_child)    # [N]
    score_parent = cx(p_batch, r_rep, g_parent)   # [N]

    return F.relu(score_child - score_parent).mean()


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
    Fast approximate Fmax on a random sample of val proteins (cosine similarity).
    Sweeps thresholds over the actual score range so the threshold can separate
    true from false annotations even when scores cluster in a narrow band.
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
