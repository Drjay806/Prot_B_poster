"""
Smin metric — CAFA semantic distance protocol (Clark & Radivojac 2011).

Smin measures how wrong predictions are, weighted by GO term specificity via
information content (IC).  Unlike Fmax (which treats every GO term equally),
Smin penalises missing a rare, specific term much more than missing a common,
broad one.  Lower Smin is better; Smin = 0 means every protein's predicted set
exactly matches the propagated true set.

Formula (per protein i, threshold τ):
    ru(i,τ) = Σ_{g ∈ true_i \ pred_i(τ)}  IC(g)   — remaining uncertainty
    mi(i,τ) = Σ_{g ∈ pred_i(τ) \ true_i}  IC(g)   — misinformation
    S(i,τ)  = √(ru² + mi²)

Smin = min_τ  mean_i  S(i,τ)         [over proteins with ≥1 annotation]

IC is computed from TRAINING annotations after label propagation:
    IC(t) = −log₂( freq_train(t) / N_train_proteins )
    freq_train(t) = # proteins annotated with t (after propagating annotations up DAG)
    Terms with freq=0 in training get IC=0 (no penalty for predicting them).
"""

import math
from typing import List, Optional, Tuple

import torch
import numpy as np
from torch_geometric.data import HeteroData

from src.data.graph_builder import build_annotation_matrix


def compute_information_content(
    train_data: HeteroData,
    target_type: str,
    prop_edges: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Compute IC vector [N_go] from training-set annotation frequencies.

    Annotations are propagated up the GO DAG before counting so that a protein
    annotated with a specific child term also counts toward all its ancestors —
    matching how ProtHGT and CAFA compute IC.

    Returns float32 tensor on CPU.  IC[g] = 0 for GO terms with no training
    annotations (neither directly nor via propagation).
    """
    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)

    # Build a dense [N_p, N_go] annotation matrix and propagate up hierarchy
    # Use float32 for accumulation but bool is enough after propagation
    # Memory: N_p × N_go × 4 bytes.  For N_p=19k, N_go=28k → ~2 GB — too large.
    # Instead: count per-term frequency in sparse form, then propagate the counts.

    # freq[g] = # proteins annotated with g (sparse accumulation)
    freq = torch.zeros(n_go, dtype=torch.float32)
    # Count: each protein contributes 1 to each of its annotated GO terms
    # Use unique to count distinct proteins per GO term
    for go_idx in range(n_go):
        # This inner loop would be too slow for 28k GO terms — use bincount instead
        pass

    # Efficient approach: for each (protein, GO) pair in the annotation list,
    # create a binary protein-indicator and propagate, then count unique proteins.
    # Since propagation changes which terms a protein is annotated with, we need
    # to propagate the PROTEIN ANNOTATIONS (row-wise) rather than the term counts.

    # Build per-protein GO sets by propagating up the hierarchy:
    # 1. Start with direct annotation sets
    # 2. For each (child, parent) edge in topological order: add parent to protein's set
    # We do this as a [N_go] indicator per protein — but N_p × N_go is too large dense.
    # Instead: propagate sparsely using prop_edges.

    # Sparse propagation:
    #   For each edge (child, parent) in topo order:
    #     All proteins that have 'child' in their annotation set also get 'parent'.
    # Represented as: protein_has[g] = sorted list of protein indices annotated with g

    # Build initial mapping: go_idx → set of protein indices
    go_to_proteins: List[set] = [set() for _ in range(n_go)]
    row_list = row.tolist()
    col_list = col.tolist()
    for p, g in zip(row_list, col_list):
        go_to_proteins[g].add(p)

    # Propagate: if protein annotated with child → also annotated with parent
    for child_idx, parent_idx in prop_edges:
        go_to_proteins[parent_idx].update(go_to_proteins[child_idx])

    # Count propagated annotations per GO term
    freq = torch.tensor([len(s) for s in go_to_proteins], dtype=torch.float32)

    # IC = -log2(freq / N_p); IC=0 for unobserved terms
    ic = torch.zeros(n_go, dtype=torch.float32)
    nonzero = freq > 0
    ic[nonzero] = -torch.log2(freq[nonzero] / n_p)

    print(f"  IC computed: {nonzero.sum().item():,}/{n_go:,} GO terms have annotations")
    print(f"  IC range: [{ic[nonzero].min().item():.2f}, {ic[nonzero].max().item():.2f}]")

    return ic


def smin_from_score_matrix(
    score_matrix: torch.Tensor,     # [N_p, N_go] raw scores (CPU)
    true_matrix:  torch.Tensor,     # [N_p, N_go] binary 0/1 (CPU), already propagated
    ic_vec:       torch.Tensor,     # [N_go] information content (CPU)
    t_steps:      int = 100,
) -> Tuple[float, float]:
    """
    Compute Smin by sweeping thresholds over score_matrix.

    Requires score_matrix and true_matrix to be fully materialised on CPU.
    Use smin_streaming() if score_matrix does not fit in RAM.

    Returns: (smin_value, best_threshold_for_smin)
    """
    n_p, n_go = score_matrix.shape

    has_annot = true_matrix.any(dim=1)
    if not has_annot.any():
        return 0.0, 0.0

    sm = score_matrix[has_annot]  # [M, N_go]
    tm = true_matrix[has_annot]   # [M, N_go] float
    ic = ic_vec.unsqueeze(0)       # [1, N_go]

    score_min = float(sm.min())
    score_max = float(sm.max())

    best_s = math.inf
    best_t = score_min

    for i in range(t_steps + 1):
        t    = score_min + i * (score_max - score_min) / t_steps
        pred = (sm >= t).float()   # [M, N_go]

        tp   = pred * tm           # [M, N_go]
        fn   = tm - tp             # [M, N_go] — true but not predicted
        fp   = pred - tp           # [M, N_go] — predicted but not true

        ru   = (fn * ic).sum(dim=1)  # [M]
        mi   = (fp * ic).sum(dim=1)  # [M]
        s    = (ru ** 2 + mi ** 2).sqrt().mean().item()

        if s < best_s:
            best_s = s
            best_t = t

    return best_s, best_t


class SminAccumulator:
    """
    Streaming Smin accumulator for chunked score matrices.

    Usage:
        acc = SminAccumulator(ic_vec, t_steps=100, score_min=s_min, score_max=s_max)
        for prot_chunk_scores, prot_chunk_true in chunks:
            acc.update(prot_chunk_scores, prot_chunk_true)
        smin_val, best_t = acc.finalize()
    """

    def __init__(
        self,
        ic_vec:    torch.Tensor,   # [N_go] on CPU
        t_steps:   int,
        score_min: float,
        score_max: float,
    ):
        self.ic_vec    = ic_vec
        self.t_steps   = t_steps
        self.score_min = score_min
        self.score_max = score_max
        self.thresholds = [
            score_min + i * (score_max - score_min) / t_steps
            for i in range(t_steps + 1)
        ]
        # sum_s[t_idx] = sum of S(i, τ) over all proteins
        self.sum_s   = np.zeros(t_steps + 1, dtype=np.float64)
        self.n_valid = 0   # proteins with ≥1 annotation

    def update(
        self,
        chunk_scores: torch.Tensor,  # [C, N_go] CPU
        chunk_true:   torch.Tensor,  # [C, N_go] binary float CPU (already propagated)
    ) -> None:
        has_annot = chunk_true.any(dim=1)
        if not has_annot.any():
            return

        sm = chunk_scores[has_annot]  # [M, N_go]
        tm = chunk_true[has_annot]    # [M, N_go]
        ic = self.ic_vec.unsqueeze(0) # [1, N_go]
        M  = sm.size(0)
        self.n_valid += M

        for t_idx, t in enumerate(self.thresholds):
            pred = (sm >= t).float()
            tp   = pred * tm
            fn   = tm - tp
            fp   = pred - tp
            ru   = (fn * ic).sum(dim=1)  # [M]
            mi   = (fp * ic).sum(dim=1)  # [M]
            s    = (ru ** 2 + mi ** 2).sqrt()
            self.sum_s[t_idx] += s.sum().item()

    def finalize(self) -> Tuple[float, float]:
        """Returns (smin, best_threshold_for_smin)."""
        if self.n_valid == 0:
            return 0.0, 0.0
        mean_s = self.sum_s / self.n_valid
        best_idx = int(np.argmin(mean_s))
        return float(mean_s[best_idx]), self.thresholds[best_idx]
