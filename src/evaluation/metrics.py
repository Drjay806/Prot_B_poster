from typing import Dict, FrozenSet, Optional

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.models.generator import Generator
from src.models.distmult import DistMult
from src.data.graph_builder import build_annotation_matrix
from src.data.go_hierarchy import propagate_predictions
from src.training.adversarial import _get_has_function_rel_idx


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
    Returns dict with fmax, auroc, aupr, micro_f1, macro_f1.
    """
    eval_cfg = cfg.get("evaluation", {})
    threshold_steps = eval_cfg.get("threshold_steps", 100)
    propagate = eval_cfg.get("propagate_hierarchy", True)
    chunk_size = eval_cfg.get("score_chunk_size", 1000)

    encoder.eval(); generator.eval()

    print("Building score matrix ...")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            protein_embs, go_embs, rel_embs = encoder(data)
        rel_idx = _get_has_function_rel_idx(encoder)
        rel_vec = rel_embs[rel_idx]

        score_matrix = distmult.score_all_chunked(protein_embs, rel_vec, go_embs, chunk_size)
        score_matrix = torch.sigmoid(score_matrix)   # [N_p, N_go] on CPU

    print("Building ground truth matrix ...")
    row, col, n_p, n_go = build_annotation_matrix(data, target_type)
    true_matrix = torch.zeros(n_p, n_go, dtype=torch.float32)
    true_matrix[row.cpu(), col.cpu()] = 1.0

    if propagate:
        print("Propagating predictions up GO hierarchy ...")
        score_matrix = propagate_predictions(score_matrix, ancestor_table)
        true_matrix = propagate_predictions(true_matrix, ancestor_table)
        true_matrix = (true_matrix > 0.5).float()

    print("Computing Fmax ...")
    fmax, best_threshold = compute_fmax(score_matrix, true_matrix, threshold_steps)

    print("Computing AUROC, AUPR ...")
    y_true_flat = true_matrix.numpy().ravel()
    y_score_flat = score_matrix.numpy().ravel()

    # Remove GO terms with no positives (undefined AUC)
    col_has_pos = true_matrix.sum(dim=0) > 0
    y_true_filt = true_matrix[:, col_has_pos].numpy().ravel()
    y_score_filt = score_matrix[:, col_has_pos].numpy().ravel()

    auroc = roc_auc_score(y_true_filt, y_score_filt) if y_true_filt.sum() > 0 else 0.0
    aupr = average_precision_score(y_true_filt, y_score_filt) if y_true_filt.sum() > 0 else 0.0

    print("Computing Micro/Macro F1 ...")
    pred_binary = (score_matrix >= best_threshold).numpy()
    true_binary = true_matrix.numpy().astype(int)
    micro_f1 = f1_score(true_binary.ravel(), pred_binary.ravel(), zero_division=0)
    # Macro F1: per-GO-term, then average (skip terms with no true positives)
    per_go_f1 = []
    for g in range(n_go):
        if true_binary[:, g].sum() > 0:
            per_go_f1.append(f1_score(true_binary[:, g], pred_binary[:, g], zero_division=0))
    macro_f1 = float(np.mean(per_go_f1)) if per_go_f1 else 0.0

    results = {
        "fmax": fmax,
        "best_threshold": best_threshold,
        "auroc": auroc,
        "aupr": aupr,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }

    _print_results(results, baseline_fmax)
    return results


def compute_fmax(
    pred_matrix: torch.Tensor,
    true_matrix: torch.Tensor,
    threshold_steps: int = 100,
) -> tuple:
    """
    Protein-centric Fmax: sweep thresholds, compute per-protein P and R, average, then max F1.
    Returns (fmax_score, best_threshold).
    """
    best_f1 = 0.0
    best_t = 0.0

    for t_int in range(threshold_steps + 1):
        t = t_int / threshold_steps
        pred = (pred_matrix >= t).float()

        tp = (pred * true_matrix).sum(dim=1)              # [N_p]
        pred_pos = pred.sum(dim=1).clamp(min=1e-8)
        true_pos = true_matrix.sum(dim=1).clamp(min=1e-8)

        precision = (tp / pred_pos).mean().item()
        recall = (tp / true_pos).mean().item()

        denom = precision + recall
        if denom > 0:
            f1 = 2 * precision * recall / denom
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

    return best_f1, best_t


def _print_results(results: Dict[str, float], baseline_fmax: float):
    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    fmax = results["fmax"]
    delta = fmax - baseline_fmax
    sign = "+" if delta >= 0 else ""
    print(f"  Fmax (BP):    {fmax:.4f}  [{sign}{delta:.4f} vs ProtHGT {baseline_fmax}]")
    print(f"  AUROC:        {results['auroc']:.4f}")
    print(f"  AUPR:         {results['aupr']:.4f}")
    print(f"  Micro-F1:     {results['micro_f1']:.4f}")
    print(f"  Macro-F1:     {results['macro_f1']:.4f}")
    print(f"  Threshold:    {results['best_threshold']:.4f}")
    print("=" * 55 + "\n")
