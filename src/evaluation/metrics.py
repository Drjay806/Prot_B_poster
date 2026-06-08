from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.models.generator import Generator
from src.models.distmult import DistMult
from src.data.graph_builder import build_annotation_matrix
from src.data.go_hierarchy import build_propagation_edges
from src.training.adversarial import _get_has_function_rel_idx

# Max proteins to materialise at once — [1000 x 27855] = 27M floats = 108 MB, safe on T4
_SCORE_CHUNK = 1000
# Proteins sampled for AUROC/AUPR (full flattened matrix is too large for sklearn)
_AUC_SAMPLE  = 8000


def evaluate_all(
    encoder: CompGCN,
    generator: Generator,
    distmult: DistMult,
    data: HeteroData,
    ancestor_table: Dict[int, FrozenSet[int]],
    target_type: str,
    cfg: dict,
    device: str = "cuda",
    baseline_fmax: float = 0.7489,
) -> Dict[str, float]:
    """
    Full evaluation matching ProtHGT's protocol.
    Streams proteins in chunks so we never hold a [N_p x N_go] tensor in memory.
    """
    eval_cfg   = cfg.get("evaluation", {})
    t_steps    = eval_cfg.get("threshold_steps", 100)
    propagate  = eval_cfg.get("propagate_hierarchy", True)
    chunk_size = eval_cfg.get("score_chunk_size", _SCORE_CHUNK)

    encoder.eval(); generator.eval()

    print("Encoding graph ...")
    with torch.no_grad():
        protein_embs, go_embs, rel_embs = encoder(data)
    rel_idx = _get_has_function_rel_idx(encoder)
    rel_vec  = rel_embs[rel_idx].detach()
    protein_embs = protein_embs.detach()
    go_embs      = go_embs.detach()

    # Sparse ground-truth annotations
    row, col, n_p, n_go = build_annotation_matrix(data, target_type)
    row_cpu, col_cpu = row.cpu(), col.cpu()

    # Propagation edges (topological order) — used instead of ancestor_table for memory safety
    prop_edges: List[Tuple[int, int]] = []
    if propagate:
        print("Loading propagation edges ...")
        prop_edges = build_propagation_edges(data, target_type)

    # ── Streaming Fmax ───────────────────────────────────────────────────────────
    # Pre-normalise for cosine scoring — encoder was trained with cosine/dot objective;
    # DistMult relation vector was never trained, so cosine similarity is the correct scorer.
    p_norm = torch.nn.functional.normalize(protein_embs.float(), dim=-1)   # [N_p, D]
    g_norm = torch.nn.functional.normalize(go_embs.float(), dim=-1)        # [N_go, D]

    # Discover actual score range with a small probe so thresholds cover it properly
    with torch.no_grad():
        probe = (p_norm[:min(500, n_p)] @ g_norm.t()).cpu()
    score_min = float(probe.min())
    score_max = float(probe.max())
    del probe

    print(f"Computing Fmax (streaming over {n_p:,} proteins in chunks of {chunk_size}) ...")
    print(f"  Score range probe: [{score_min:.3f}, {score_max:.3f}] — thresholds span this range")
    thresholds = [score_min + i * (score_max - score_min) / t_steps for i in range(t_steps + 1)]
    sum_prec   = np.zeros(t_steps + 1)
    sum_rec    = np.zeros(t_steps + 1)
    n_counted  = 0

    for prot_start in range(0, n_p, chunk_size):
        prot_end  = min(prot_start + chunk_size, n_p)
        chunk_len = prot_end - prot_start

        # Cosine similarity [chunk, N_go] on GPU, move to CPU immediately
        with torch.no_grad():
            chunk_scores = (p_norm[prot_start:prot_end] @ g_norm.t()).cpu()

        # True labels for this chunk (sparse → dense in-chunk only)
        chunk_mask = (row_cpu >= prot_start) & (row_cpu < prot_end)
        c_row = row_cpu[chunk_mask] - prot_start
        c_col = col_cpu[chunk_mask]
        chunk_true = torch.zeros(chunk_len, n_go, dtype=torch.float32)
        chunk_true[c_row, c_col] = 1.0

        if propagate and prop_edges:
            _propagate_chunk(chunk_scores, prop_edges)
            _propagate_chunk(chunk_true, prop_edges)
        chunk_true = (chunk_true > 0.5).float()

        # Skip proteins with no annotations in this split
        has_annot = chunk_true.any(dim=1)
        if not has_annot.any():
            continue
        cs = chunk_scores[has_annot]
        ct = chunk_true[has_annot]
        ct_sum = ct.sum(dim=1).clamp(min=1e-8)
        n_counted += has_annot.sum().item()

        for t_idx, t in enumerate(thresholds):
            pred     = (cs >= t).float()
            tp       = (pred * ct).sum(dim=1)
            pred_pos = pred.sum(dim=1).clamp(min=1e-8)
            sum_prec[t_idx] += (tp / pred_pos).sum().item()
            sum_rec [t_idx] += (tp / ct_sum).sum().item()

    best_f1, best_t = 0.0, 0.0
    for t_idx, t in enumerate(thresholds):
        p = sum_prec[t_idx] / max(n_counted, 1)
        r = sum_rec [t_idx] / max(n_counted, 1)
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, best_t = f1, t

    # ── AUROC / AUPR on a random protein sample ──────────────────────────────────
    print(f"Computing AUROC / AUPR (sample of {_AUC_SAMPLE:,} proteins) ...")
    unique_prots = row_cpu.unique()
    if len(unique_prots) > _AUC_SAMPLE:
        sel = unique_prots[torch.randperm(len(unique_prots))[:_AUC_SAMPLE]]
    else:
        sel = unique_prots

    prot_map  = {p.item(): i for i, p in enumerate(sel)}
    auc_mask  = torch.isin(row_cpu, sel)
    auc_row   = torch.tensor([prot_map[p.item()] for p in row_cpu[auc_mask]], dtype=torch.long)
    auc_col   = col_cpu[auc_mask]
    auc_true  = torch.zeros(len(sel), n_go, dtype=torch.float32)
    auc_true[auc_row, auc_col] = 1.0

    with torch.no_grad():
        auc_scores = (p_norm[sel.to(device)] @ g_norm.t()).cpu()

    if propagate and prop_edges:
        _propagate_chunk(auc_scores, prop_edges)
        _propagate_chunk(auc_true, prop_edges)
    auc_true = (auc_true > 0.5).float()

    col_has_pos   = auc_true.sum(dim=0) > 0
    y_true_flat   = auc_true[:, col_has_pos].numpy().ravel()
    y_score_flat  = auc_scores[:, col_has_pos].numpy().ravel()
    auroc = roc_auc_score(y_true_flat, y_score_flat)       if y_true_flat.sum() > 0 else 0.0
    aupr  = average_precision_score(y_true_flat, y_score_flat) if y_true_flat.sum() > 0 else 0.0

    # ── Micro / Macro F1 at best threshold ───────────────────────────────────────
    print("Computing F1 ...")
    pred_bin = (auc_scores >= best_t).numpy().astype(int)
    true_bin = auc_true.numpy().astype(int)
    micro_f1 = f1_score(true_bin.ravel(), pred_bin.ravel(), zero_division=0)
    per_go   = [
        f1_score(true_bin[:, g], pred_bin[:, g], zero_division=0)
        for g in range(n_go) if true_bin[:, g].sum() > 0
    ]
    macro_f1 = float(np.mean(per_go)) if per_go else 0.0

    results = {
        "fmax": best_f1,
        "best_threshold": best_t,
        "auroc": auroc,
        "aupr": aupr,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }
    _print_results(results, baseline_fmax)
    return results


def _propagate_chunk(chunk: torch.Tensor, prop_edges: List[Tuple[int, int]]) -> None:
    """In-place propagation over a [N_proteins_in_chunk, N_go] tensor."""
    for child_idx, parent_idx in prop_edges:
        torch.maximum(chunk[:, parent_idx], chunk[:, child_idx], out=chunk[:, parent_idx])


def _print_results(results: Dict[str, float], baseline_fmax: float) -> None:
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    delta = results["fmax"] - baseline_fmax
    sign  = "+" if delta >= 0 else ""
    print(f"  Fmax (BP):    {results['fmax']:.4f}  [{sign}{delta:.4f} vs ProtHGT {baseline_fmax}]")
    print(f"  AUROC:        {results['auroc']:.4f}")
    print(f"  AUPR:         {results['aupr']:.4f}")
    print(f"  Micro-F1:     {results['micro_f1']:.4f}")
    print(f"  Macro-F1:     {results['macro_f1']:.4f}")
    print(f"  Threshold:    {results['best_threshold']:.4f}")
    print("=" * 55 + "\n")
