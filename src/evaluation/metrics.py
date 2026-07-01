"""
Evaluation harness — CAFA protocol.

Metrics reported (per ontology, after label propagation):
    Fmax   — primary; protein-centric max F1 over threshold sweep
    Smin   — primary; minimum semantic distance (IC-weighted, lower is better)
    AUPR   — secondary; area under precision-recall curve
    F1     — at Fmax threshold; for alignment with ProtHGT comparison table
    MCC    — at Fmax threshold; Matthews Correlation Coefficient

Scoring modes (select via `mode` parameter in evaluate_all):
    "encoder"   — ComplEx score(protein, has_function, GO)  [default, matches training]
    "critic"    — WGAN critic score(protein, GO) for all GO terms
    "ensemble"  — α·ComplEx + (1−α)·critic, α tuned on val by tune_ensemble_alpha()

All metrics are applied AFTER label propagation up the GO DAG (true-path rule).
The same threshold grid and propagation protocol are used for every scoring mode
and for the ProtHGT comparison row, ensuring apples-to-apples comparison.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.models.generator import Generator
from src.models.distmult import DistMult
from src.data.graph_builder import build_annotation_matrix
from src.data.go_hierarchy import build_propagation_edges
from src.training.adversarial import _get_has_function_rel_idx
from src.evaluation.smin import SminAccumulator, compute_information_content

# Max proteins per ComplEx/critic scoring chunk — [1000 x 27855] ≈ 108 MB on T4
_SCORE_CHUNK = 1000
# Proteins sampled for AUROC/AUPR/MCC (full matrix is too large for sklearn)
_AUC_SAMPLE  = 8000

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_all(
    encoder:       CompGCN,
    generator:     Generator,
    distmult:      DistMult,
    data:          HeteroData,
    ancestor_table: Dict[int, FrozenSet[int]],
    target_type:   str,
    cfg:           dict,
    device:        str = "cuda",
    baseline_fmax: Optional[float] = None,
    mode:          str = "encoder",
    ensemble_alpha: float = 0.7,
    ic_vec:        Optional[torch.Tensor] = None,
    discriminator  = None,
    calibration_head = None,
) -> Dict[str, float]:
    """
    Full evaluation on a single ontology.

    Parameters
    ----------
    mode : "encoder" | "critic" | "ensemble"
        Scoring function for the protein→GO score matrix.
    ensemble_alpha : float
        ComplEx weight in ensemble mode (1-alpha = critic weight).
        Ignored for other modes.  Tune with tune_ensemble_alpha().
    ic_vec : [N_go] tensor or None
        Pre-computed information content from training data (for Smin).
        Pass None to skip Smin (it will be reported as -1.0).
        Compute with: smin.compute_information_content(train_data, target_type, prop_edges)
    discriminator : Discriminator or None
        Required when mode is "critic" or "ensemble".
    """
    eval_cfg   = cfg.get("evaluation", {})
    t_steps    = eval_cfg.get("threshold_steps", 100)
    propagate  = eval_cfg.get("propagate_hierarchy", True)
    chunk_size = eval_cfg.get("score_chunk_size", _SCORE_CHUNK)
    gen_weight = eval_cfg.get("gen_weight", 0.0)   # generator path weight (set 0 for ablation)
    p_chunk    = eval_cfg.get("protein_chunk_size", 64)
    g_chunk    = eval_cfg.get("go_chunk_size", 512)

    if mode in ("critic", "ensemble") and discriminator is None:
        raise ValueError(f"mode='{mode}' requires discriminator to be passed")
    if mode == "calibration" and calibration_head is None:
        raise ValueError("mode='calibration' requires calibration_head to be passed")

    encoder.eval()
    if generator is not None:
        generator.eval()

    print(f"Encoding graph  [mode={mode}] ...")
    with torch.no_grad():
        protein_embs, go_embs, rel_embs = encoder(data)
    rel_idx      = _get_has_function_rel_idx(encoder)
    rel_vec      = rel_embs[rel_idx].detach()
    protein_embs = protein_embs.detach()
    go_embs      = go_embs.detach()

    g_norm = F.normalize(go_embs.float(), dim=-1)   # for generator cosine path

    row, col, n_p, n_go = build_annotation_matrix(data, target_type)
    row_cpu, col_cpu = row.cpu(), col.cpu()

    prop_edges: List[Tuple[int, int]] = []
    if propagate:
        print("Loading propagation edges ...")
        prop_edges = build_propagation_edges(data, target_type)

    # ── Precompute critic/calibration scores [N_p, N_go] if needed ───────────
    if mode in ("critic", "ensemble"):
        print(f"Computing critic scores for {n_p:,} proteins × {n_go:,} GO terms ...")
        critic_scores_full = critic_score_all(
            discriminator, protein_embs, go_embs, device, p_chunk, g_chunk
        )  # [N_p, N_go] CPU
    elif mode == "calibration":
        print(f"Computing calibration scores for {n_p:,} proteins × {n_go:,} GO terms ...")
        critic_scores_full = critic_score_all(
            calibration_head, protein_embs, go_embs, device, p_chunk, g_chunk
        )  # [N_p, N_go] CPU — reuses same chunked infrastructure

    # ── Precompute generator embeddings [N_p, D] if gen_weight > 0 ────────────
    gen_p_norm = None
    if gen_weight > 0 and generator is not None:
        print(f"Computing generator embeddings (noise=0) ...")
        gen_chunks = []
        for gs in range(0, n_p, 512):
            ge = min(gs + 512, n_p)
            with torch.no_grad():
                chunk_p   = protein_embs[gs:ge].to(device)
                rel_chunk = rel_vec.unsqueeze(0).expand(ge - gs, -1)
                noise     = torch.zeros(ge - gs, generator.noise_dim, device=device)
                fake_go   = generator(chunk_p, rel_chunk, noise)
            gen_chunks.append(F.normalize(fake_go.float(), dim=-1).cpu())
        gen_p_norm = torch.cat(gen_chunks, dim=0)

    # ── Probe score range for threshold grid ──────────────────────────────────
    probe_end = min(500, n_p)
    with torch.no_grad():
        probe = _build_chunk_scores(
            protein_embs[:probe_end], go_embs, rel_vec, g_norm,
            gen_p_norm[:probe_end] if gen_p_norm is not None else None,
            critic_scores_full[:probe_end].to(device) if mode in ("critic","ensemble") else None,
            distmult, gen_weight, mode, ensemble_alpha, device
        ).cpu()

    score_min = float(probe.min())
    score_max = float(probe.max())
    del probe

    print(f"Computing Fmax+Smin [{mode}] streaming {n_p:,} proteins ...")
    print(f"  Score range: [{score_min:.3f}, {score_max:.3f}]")

    thresholds = [score_min + i * (score_max - score_min) / t_steps for i in range(t_steps + 1)]
    sum_prec = np.zeros(t_steps + 1)
    sum_rec  = np.zeros(t_steps + 1)
    n_counted = 0

    smin_acc = SminAccumulator(ic_vec, t_steps, score_min, score_max) if ic_vec is not None else None

    for prot_start in range(0, n_p, chunk_size):
        prot_end  = min(prot_start + chunk_size, n_p)
        chunk_len = prot_end - prot_start

        with torch.no_grad():
            critic_chunk = (
                critic_scores_full[prot_start:prot_end].to(device)
                if mode in ("critic", "ensemble", "calibration") else None
            )
            chunk_scores = _build_chunk_scores(
                protein_embs[prot_start:prot_end], go_embs, rel_vec, g_norm,
                gen_p_norm[prot_start:prot_end] if gen_p_norm is not None else None,
                critic_chunk,
                distmult, gen_weight, mode, ensemble_alpha, device
            ).cpu()

        chunk_mask = (row_cpu >= prot_start) & (row_cpu < prot_end)
        c_row = row_cpu[chunk_mask] - prot_start
        c_col = col_cpu[chunk_mask]
        chunk_true = torch.zeros(chunk_len, n_go, dtype=torch.float32)
        chunk_true[c_row, c_col] = 1.0

        if propagate and prop_edges:
            _propagate_chunk(chunk_scores, prop_edges)
            _propagate_chunk(chunk_true,  prop_edges)
        chunk_true = (chunk_true > 0.5).float()

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

        if smin_acc is not None:
            smin_acc.update(chunk_scores, chunk_true)

    best_f1, best_t = 0.0, 0.0
    for t_idx, t in enumerate(thresholds):
        p = sum_prec[t_idx] / max(n_counted, 1)
        r = sum_rec [t_idx] / max(n_counted, 1)
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1, best_t = f1, t

    smin_val = smin_acc.finalize()[0] if smin_acc is not None else -1.0

    # ── AUROC / AUPR / F1 / MCC on a random protein sample ───────────────────
    print(f"Computing AUROC / AUPR / MCC (sample of {_AUC_SAMPLE:,} proteins) ...")
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
        critic_auc = (
            critic_scores_full[sel].to(device)
            if mode in ("critic", "ensemble", "calibration") else None
        )
        auc_scores = _build_chunk_scores(
            protein_embs[sel.to(device)], go_embs, rel_vec, g_norm,
            gen_p_norm[sel] if gen_p_norm is not None else None,
            critic_auc,
            distmult, gen_weight, mode, ensemble_alpha, device
        ).cpu()

    if propagate and prop_edges:
        _propagate_chunk(auc_scores, prop_edges)
        _propagate_chunk(auc_true,   prop_edges)
    auc_true = (auc_true > 0.5).float()

    col_has_pos  = auc_true.sum(dim=0) > 0
    y_true_flat  = auc_true[:, col_has_pos].numpy().ravel()
    y_score_flat = auc_scores[:, col_has_pos].numpy().ravel()
    auroc = roc_auc_score(y_true_flat, y_score_flat)        if y_true_flat.sum() > 0 else 0.0
    aupr  = average_precision_score(y_true_flat, y_score_flat) if y_true_flat.sum() > 0 else 0.0

    pred_bin_auc = (auc_scores >= best_t).numpy().astype(int)
    true_bin_auc = auc_true.numpy().astype(int)
    micro_f1 = f1_score(true_bin_auc.ravel(), pred_bin_auc.ravel(), zero_division=0)
    per_go   = [
        f1_score(true_bin_auc[:, g], pred_bin_auc[:, g], zero_division=0)
        for g in range(n_go) if true_bin_auc[:, g].sum() > 0
    ]
    macro_f1 = float(np.mean(per_go)) if per_go else 0.0
    mcc_val  = float(matthews_corrcoef(true_bin_auc.ravel(), pred_bin_auc.ravel()))

    print("Computing Hit@k / MRR (filtered ranking, KGC protocol) ...")
    rank_metrics = compute_ranking_metrics(auc_scores, auc_true)

    results = {
        "fmax":           best_f1,
        "smin":           smin_val,
        "best_threshold": best_t,
        "auroc":          auroc,
        "aupr":           aupr,
        "micro_f1":       micro_f1,
        "macro_f1":       macro_f1,
        "mcc":            mcc_val,
        "mode":           mode,
        **rank_metrics,
    }
    _print_results(results, baseline_fmax, target_type)
    return results


def evaluate_all_ontologies(
    encoder:       CompGCN,
    generator:     Generator,
    distmult:      DistMult,
    discriminator,
    splits:        Dict[str, Tuple[HeteroData, HeteroData, HeteroData]],
    ic_vecs:       Dict[str, torch.Tensor],
    cfg:           dict,
    device:        str = "cuda",
    mode:          str = "encoder",
    ensemble_alpha: float = 0.7,
    baselines:     Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run evaluate_all for all three ontologies in one call.

    Parameters
    ----------
    splits : {"bp": (train, val, test), "mf": ..., "cc": ...}
    ic_vecs : {"bp": ic_tensor, "mf": ..., "cc": ...}
              Each precomputed with compute_information_content(train_data, target_type, prop_edges)
    baselines : {"bp": 0.7489, "mf": ..., "cc": ...}  — ProtHGT reference Fmax per ontology

    Returns {"bp": {metric_dict}, "mf": {...}, "cc": {...}}
    """
    from src.data.loader import ONTOLOGY_TO_NODE

    baselines = baselines or {}
    results   = {}

    for ontology, target_type in ONTOLOGY_TO_NODE.items():
        if ontology not in splits:
            print(f"Skipping {ontology} (not in splits dict)")
            continue

        _, _, test_data = splits[ontology]
        ancestor_table = _build_ancestor_table_for(test_data, target_type)

        print(f"\n{'='*60}")
        print(f"  Ontology: {ontology.upper()}  ({target_type})")
        print(f"{'='*60}")

        results[ontology] = evaluate_all(
            encoder=encoder,
            generator=generator,
            distmult=distmult,
            data=test_data,
            ancestor_table=ancestor_table,
            target_type=target_type,
            cfg=cfg,
            device=device,
            baseline_fmax=baselines.get(ontology),
            mode=mode,
            ensemble_alpha=ensemble_alpha,
            ic_vec=ic_vecs.get(ontology),
            discriminator=discriminator,
        )

    return results


def tune_ensemble_alpha(
    encoder:       CompGCN,
    discriminator,
    distmult:      DistMult,
    val_data:      HeteroData,
    target_type:   str,
    cfg:           dict,
    device:        str = "cuda",
    n_sample:      int = 2000,
    alpha_steps:   int = 10,
) -> float:
    """
    Sweep alpha in {0.0, 0.1, ..., 1.0} on a val sample and return the alpha
    with the highest fast approximate Fmax.

    alpha = ComplEx weight;  (1 - alpha) = critic weight.
    Uses cosine-proxy Fmax (fast) rather than full ComplEx scoring.
    """
    from src.data.graph_builder import build_annotation_matrix

    encoder.eval()
    with torch.no_grad():
        protein_embs, go_embs, rel_embs = encoder(val_data)
    rel_idx      = _get_has_function_rel_idx(encoder)
    rel_vec      = rel_embs[rel_idx].detach()
    protein_embs = protein_embs.detach()
    go_embs      = go_embs.detach()

    row, col, n_p, n_go = build_annotation_matrix(val_data, target_type)
    row_cpu, col_cpu    = row.cpu(), col.cpu()

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

    sel = unique_prots.to(device)

    # Encoder ComplEx scores
    with torch.no_grad():
        enc_scores = distmult.score_all(
            protein_embs[sel].float(), rel_vec.float(), go_embs.float()
        ).cpu()   # [n_s, N_go]

    # Critic scores (chunked because n_s × N_go is large)
    p_chunk = cfg.get("evaluation", {}).get("protein_chunk_size", 64)
    g_chunk = cfg.get("evaluation", {}).get("go_chunk_size", 512)
    crit_scores = critic_score_all(
        discriminator, protein_embs[sel], go_embs, device, p_chunk, g_chunk
    )  # [n_s, N_go] CPU

    # Normalise both to same range before blending
    enc_min, enc_max = enc_scores.min(), enc_scores.max()
    crit_min, crit_max = crit_scores.min(), crit_scores.max()
    enc_norm  = (enc_scores  - enc_min)  / (enc_max  - enc_min  + 1e-8)
    crit_norm = (crit_scores - crit_min) / (crit_max - crit_min + 1e-8)

    best_alpha, best_fmax = 0.7, 0.0
    true_pos = true_mat.sum(dim=1).clamp(min=1e-8)

    for step in range(alpha_steps + 1):
        alpha  = step / alpha_steps
        scores = alpha * enc_norm + (1.0 - alpha) * crit_norm

        s_min  = scores.min().item()
        s_max  = scores.max().item()
        fmax_a = 0.0

        for i in range(51):
            t    = s_min + i * (s_max - s_min) / 50
            pred = (scores >= t).float()
            tp   = (pred * true_mat).sum(dim=1)
            pp   = pred.sum(dim=1).clamp(min=1e-8)
            prec = (tp / pp).mean().item()
            rec  = (tp / true_pos).mean().item()
            if prec + rec > 0:
                fmax_a = max(fmax_a, 2 * prec * rec / (prec + rec))

        print(f"  alpha={alpha:.1f} → val_Fmax={fmax_a:.4f}")
        if fmax_a > best_fmax:
            best_fmax  = fmax_a
            best_alpha = alpha

    print(f"Best ensemble alpha: {best_alpha:.1f}  (val_Fmax={best_fmax:.4f})")
    return best_alpha


def compute_ranking_metrics(
    scores:    torch.Tensor,           # [N_p, N_go] CPU
    true_mat:  torch.Tensor,           # [N_p, N_go] CPU, 0/1
    k_values:  Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, float]:
    """
    Filtered Hit@k and MRR — standard knowledge-graph-completion protocol
    (Bordes et al. 2013).  For each protein's true GO terms, rank against all
    GO terms with the protein's OTHER true positives removed from the
    candidate pool, so a protein's own correct multi-label annotations don't
    suppress each other's rank.

    This is a different lens than Fmax: Fmax is the CAFA protein-function
    metric (hierarchy-aware, threshold-based F1). Hit@k/MRR are the metrics
    used to evaluate ComplEx/DistMult/TransE-style link predictors on
    standard KGC benchmarks (FB15k-237, WN18RR). Reporting both makes clear
    which literature each number should be compared against.
    """
    n_p = scores.size(0)
    rank_chunks = []
    for i in range(n_p):
        true_mask = true_mat[i] > 0.5
        n_true = int(true_mask.sum().item())
        if n_true == 0:
            continue
        true_scores  = scores[i][true_mask]     # [n_true]
        false_scores = scores[i][~true_mask]    # [N_go - n_true]
        ranks = (false_scores.unsqueeze(0) > true_scores.unsqueeze(1)).sum(dim=1) + 1
        rank_chunks.append(ranks)

    if not rank_chunks:
        out = {f"hit@{k}": 0.0 for k in k_values}
        out["mrr"] = 0.0
        return out

    ranks = torch.cat(rank_chunks).float()
    out = {f"hit@{k}": (ranks <= k).float().mean().item() for k in k_values}
    out["mrr"] = (1.0 / ranks).mean().item()
    return out


def critic_score_all(
    discriminator,
    protein_embs: torch.Tensor,   # [N_p, D]
    go_embs:      torch.Tensor,   # [N_go, D]
    device:       str,
    protein_chunk: int = 64,
    go_chunk:      int = 512,
) -> torch.Tensor:
    """
    Score every (protein, GO) pair with the WGAN critic.
    Returns [N_p, N_go] float32 tensor on CPU.

    Memory: protein_chunk × go_chunk pairs materialised at once.
    C=64, G=512 → 32,768 pairs × 512 dims × 4 bytes × 2 inputs ≈ 134 MB peak.
    Reduce protein_chunk if T4 OOM occurs.
    """
    discriminator.eval()
    n_p  = protein_embs.size(0)
    n_go = go_embs.size(0)
    result = torch.zeros(n_p, n_go, dtype=torch.float32)

    with torch.no_grad():
        for ps in range(0, n_p, protein_chunk):
            pe      = min(ps + protein_chunk, n_p)
            C       = pe - ps
            p_chunk = protein_embs[ps:pe].to(device)   # [C, D]

            for gs in range(0, n_go, go_chunk):
                ge      = min(gs + go_chunk, n_go)
                G       = ge - gs
                g_chunk = go_embs[gs:ge].to(device)   # [G, D]

                # Build [C×G, D] pairs
                p_rep = p_chunk.unsqueeze(1).expand(-1, G, -1).reshape(C * G, -1)
                g_rep = g_chunk.unsqueeze(0).expand(C, -1, -1).reshape(C * G, -1)

                scores = discriminator.score(p_rep, g_rep)   # [C*G]
                result[ps:pe, gs:ge] = scores.reshape(C, G).cpu()

    return result   # [N_p, N_go] CPU


def print_ablation_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    ontologies:  Optional[List[str]] = None,
) -> None:
    """
    Print a markdown ablation table.

    Parameters
    ----------
    all_results : {condition_name: {ontology: {metric: value}}}
        e.g. {"Baseline": {"bp": {...}, "mf": {...}, "cc": {...}}, ...}
    ontologies : list of ontology keys to print (default: all present)
    """
    if ontologies is None:
        all_onts = set()
        for cond in all_results.values():
            all_onts.update(cond.keys())
        ontologies = sorted(all_onts)

    metrics = ["fmax", "smin", "aupr", "micro_f1", "mcc", "hit@10", "mrr"]
    header_metrics = ["Fmax↑", "Smin↓", "AUPR↑", "F1↑", "MCC↑", "Hit@10↑", "MRR↑"]

    for ont in ontologies:
        print(f"\n## {ont.upper()}")
        # Header
        cols = ["Condition"] + header_metrics
        print("| " + " | ".join(cols) + " |")
        print("| " + " | ".join(["---"] * len(cols)) + " |")
        # Rows
        for condition, per_ont in all_results.items():
            if ont not in per_ont:
                continue
            m = per_ont[ont]
            vals = [condition]
            for metric in metrics:
                v = m.get(metric, float("nan"))
                if metric == "smin":
                    vals.append(f"{v:.4f}" if v >= 0 else "—")
                else:
                    vals.append(f"{v:.4f}" if not (v != v) else "—")
            print("| " + " | ".join(vals) + " |")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_chunk_scores(
    protein_chunk: torch.Tensor,          # [C, D] on device
    go_embs:       torch.Tensor,          # [N_go, D] on device
    rel_vec:       torch.Tensor,          # [D] on device
    g_norm:        torch.Tensor,          # [N_go, D] normalised, on device
    gen_chunk:     Optional[torch.Tensor],# [C, D] normalised generator output, CPU
    critic_chunk:  Optional[torch.Tensor],# [C, N_go] precomputed critic scores
    distmult:      DistMult,
    gen_weight:    float,
    mode:          str,
    ensemble_alpha: float,
    device:        str,
) -> torch.Tensor:
    """Build [C, N_go] score tensor for a protein chunk. Returned on CPU."""
    if mode == "encoder":
        enc_scores = distmult.score_all(
            protein_chunk.float(), rel_vec.float(), go_embs.float()
        ).cpu()   # [C, N_go]
        if gen_weight > 0 and gen_chunk is not None:
            gen_scores = (gen_chunk.to(device) @ g_norm.t()).cpu()
            return (1.0 - gen_weight) * enc_scores + gen_weight * gen_scores
        return enc_scores

    elif mode in ("critic", "calibration"):
        # precomputed [C, N_go] — same path for critic and calibration head
        return critic_chunk.cpu() if critic_chunk.device.type != "cpu" else critic_chunk

    elif mode == "ensemble":
        enc_scores  = distmult.score_all(
            protein_chunk.float(), rel_vec.float(), go_embs.float()
        ).cpu()
        crit_scores = critic_chunk.cpu() if critic_chunk.device.type != "cpu" else critic_chunk

        # Normalise to [0,1] before blending so different score ranges don't dominate
        def _norm01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        return ensemble_alpha * _norm01(enc_scores) + (1.0 - ensemble_alpha) * _norm01(crit_scores)

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Choose 'encoder', 'critic', or 'ensemble'.")


def _propagate_chunk(chunk: torch.Tensor, prop_edges: List[Tuple[int, int]]) -> None:
    """In-place propagation over a [C, N_go] tensor."""
    for child_idx, parent_idx in prop_edges:
        torch.maximum(chunk[:, parent_idx], chunk[:, child_idx], out=chunk[:, parent_idx])


def _build_ancestor_table_for(data: HeteroData, target_type: str) -> Dict[int, FrozenSet[int]]:
    """Build ancestor table on the fly for a given data split and target type."""
    from src.data.go_hierarchy import build_ancestor_table
    # Use a per-target_type cache path so BP, MF, CC don't collide
    cache = f"/tmp/ancestor_table_{target_type}.pkl"
    return build_ancestor_table(data, target_type=target_type, cache_path=cache)


def _print_results(
    results:       Dict[str, float],
    baseline_fmax: Optional[float],
    target_type:   str,
) -> None:
    smin_str = f"{results['smin']:.4f}" if results.get("smin", -1) >= 0 else "n/a (ic_vec not provided)"
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS  [{target_type}  mode={results.get('mode','?')}]")
    print(f"{'='*60}")
    if baseline_fmax is not None:
        delta = results["fmax"] - baseline_fmax
        sign  = "+" if delta >= 0 else ""
        print(f"  Fmax:     {results['fmax']:.4f}  [{sign}{delta:.4f} vs baseline {baseline_fmax}]")
    else:
        print(f"  Fmax:     {results['fmax']:.4f}")
    print(f"  Smin:     {smin_str}")
    print(f"  AUPR:     {results['aupr']:.4f}")
    print(f"  AUROC:    {results['auroc']:.4f}")
    print(f"  Micro-F1: {results['micro_f1']:.4f}")
    print(f"  Macro-F1: {results['macro_f1']:.4f}")
    print(f"  MCC:      {results['mcc']:.4f}")
    print(f"  Threshold:{results['best_threshold']:.4f}")
    if "mrr" in results:
        print(f"  -- KGC ranking metrics (filtered protocol) --")
        print(f"  Hit@1:    {results.get('hit@1', float('nan')):.4f}")
        print(f"  Hit@3:    {results.get('hit@3', float('nan')):.4f}")
        print(f"  Hit@10:   {results.get('hit@10', float('nan')):.4f}")
        print(f"  MRR:      {results['mrr']:.4f}")
    print(f"{'='*60}\n")
