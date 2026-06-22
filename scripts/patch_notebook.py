"""Insert/update evaluation cells in train_colab.ipynb."""
import json
import sys

NB_PATH = "notebooks/train_colab.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ── Cell A: Split disjointness check ──────────────────────────────────────────
CELL_A = """\
# Cell 6b: Verify Split Disjointness
# Confirms test proteins were never seen during training.
# Cold-start split: test protein IDs must not appear in train annotation edges.
# Raises ValueError if leakage is detected (evaluation numbers would be inflated).
from src.data.graph_builder import validate_split_disjointness

print("Checking train/test protein leakage (cold-start protein-level split)...")
for ont, ttype in [("bp", "GO_term_P"), ("mf", "GO_term_F"), ("cc", "GO_term_C")]:
    try:
        validate_split_disjointness(train_data, test_data, ttype)
    except KeyError:
        print(f"  {ttype}: no annotations found in this split (skipping)")
print("Split validation complete.")
"""

# ── Cell B: Pre-compute IC vectors ────────────────────────────────────────────
CELL_B = """\
# Cell 6c: Pre-compute Information Content Vectors (required for Smin metric)
# IC[t] = -log2(freq_train(t) / N_proteins), computed after label propagation.
# One call per ontology; results stored in ic_vecs dict keyed by ontology shortname.
from src.evaluation.smin import compute_information_content
from src.data.go_hierarchy import build_propagation_edges

ic_vecs = {}
for ont, ttype in [("bp", "GO_term_P"), ("mf", "GO_term_F"), ("cc", "GO_term_C")]:
    try:
        pe = build_propagation_edges(train_data, target_type=ttype, cache=True)
        ic = compute_information_content(train_data, ttype, pe)
        ic_vecs[ont] = ic
        print(f"  {ont} ({ttype}): IC ready ({(ic > 0).sum().item()} terms with annotations)")
    except Exception as e:
        print(f"  {ont}: skipped -- {e}")
print("IC computation complete.")
"""

# ── Updated Cell 14: single-ontology final evaluation ─────────────────────────
CELL_14_NEW = """\
# Cell 11: Test Evaluation (current ontology, all scoring modes)
# Condition 1 = encoder-only (run right after Phase 1, before Phase 2).
# Conditions 2-4 = run Cell 11b after Phase 2 completes.
import json, os
from src.evaluation.metrics import evaluate_all

PUBLISHED_FMAX = {
    "GO_term_P": 0.7489,   # ProtHGT-ESM2 BP (as reported in their paper)
    "GO_term_F": None,     # fill from ProtHGT paper once known
    "GO_term_C": None,
}

print("\\n=== CONDITION 1: Encoder-only (ComplEx scorer) ===")
results_baseline = evaluate_all(
    encoder=encoder,
    generator=generator,
    distmult=distmult,
    data=test_data,
    ancestor_table=ancestor_table,
    target_type=target_type,
    cfg=cfg,
    device=DEVICE,
    baseline_fmax=PUBLISHED_FMAX.get(target_type),
    mode="encoder",
    ic_vec=ic_vecs.get(cfg["data"]["ontology"]),
    discriminator=discriminator,
)

results_path = os.path.join(CHECKPOINT_DIR, "results_baseline.json")
with open(results_path, "w") as f:
    json.dump(results_baseline, f, indent=2)
print(f"Saved -> {results_path}")
logger.close()
"""

# ── Cell C: Full ablation table ───────────────────────────────────────────────
CELL_C = """\
# Cell 11b: Full Ablation Table (4 conditions x BP/MF/CC)
# Run AFTER Phase 2 completes (discriminator required for critic/ensemble modes).
# tune_ensemble_alpha() sweeps alpha on val data and returns the best value.
import json, os
from src.evaluation.metrics import evaluate_all, tune_ensemble_alpha, print_ablation_table
from src.data.loader import load_prothgt_splits

ONTOLOGY_TO_NODE = {"bp": "GO_term_P", "mf": "GO_term_F", "cc": "GO_term_C"}

# -- Tune ensemble alpha on validation set (ComplEx vs critic weight) -----------
print("Tuning ensemble alpha on val set ...")
best_alpha = tune_ensemble_alpha(
    encoder=encoder,
    discriminator=discriminator,
    distmult=distmult,
    val_data=val_data,
    target_type=target_type,
    cfg=cfg,
    device=DEVICE,
)
print(f"Best alpha (ComplEx weight): {best_alpha:.2f}")
cfg["evaluation"]["ensemble_alpha"] = best_alpha

# -- Load splits for all three ontologies --------------------------------------
all_splits = {}
for ont in ["bp", "mf", "cc"]:
    try:
        sp = load_prothgt_splits(cfg["data"]["drive_root"], ontology=ont, device="cpu")
        all_splits[ont] = (sp.train, sp.val, sp.test)
        print(f"  Loaded {ont}")
    except Exception as e:
        print(f"  {ont} skipped: {e}")

# -- Build ancestor tables and IC vecs for each ontology -----------------------
from src.data.go_hierarchy import build_ancestor_table, build_propagation_edges
from src.evaluation.smin import compute_information_content

anc_tables, ic_all = {}, {}
for ont, ttype in ONTOLOGY_TO_NODE.items():
    if ont not in all_splits:
        continue
    train_d, _, _ = all_splits[ont]
    cache_p = f"/tmp/ancestor_table_{ttype}.pkl"
    anc_tables[ont] = build_ancestor_table(train_d, target_type=ttype, cache=True,
                                           cache_path=cache_p)
    try:
        pe = build_propagation_edges(train_d, target_type=ttype, cache=True)
        ic_all[ont] = compute_information_content(train_d, ttype, pe)
    except Exception as e:
        print(f"  IC {ont}: {e}")

# -- Run 4 ablation conditions -------------------------------------------------
conditions = [
    ("Baseline (encoder only)",      "encoder",  None),
    ("+ Adversarial (encoder)",      "encoder",  None),
    ("+ Adversarial (critic)",       "critic",   None),
    ("+ Adversarial (ensemble)",     "ensemble", best_alpha),
]

ablation_results = {}
for cond_name, mode, alpha in conditions:
    ablation_results[cond_name] = {}
    cfg["evaluation"]["mode"] = mode
    if alpha is not None:
        cfg["evaluation"]["ensemble_alpha"] = alpha

    for ont, ttype in ONTOLOGY_TO_NODE.items():
        if ont not in all_splits:
            continue
        _, _, test_d = all_splits[ont]
        print(f"\\n{cond_name} | {ont.upper()} ...")
        ablation_results[cond_name][ont] = evaluate_all(
            encoder=encoder, generator=generator, distmult=distmult,
            data=test_d, ancestor_table=anc_tables[ont],
            target_type=ttype, cfg=cfg, device=DEVICE,
            mode=mode,
            ensemble_alpha=alpha if alpha is not None else cfg["evaluation"]["ensemble_alpha"],
            ic_vec=ic_all.get(ont),
            discriminator=discriminator,
        )

# -- Print and save ------------------------------------------------------------
print("\\n" + "=" * 70)
print("  ABLATION TABLE")
print("=" * 70)
print_ablation_table(ablation_results)

with open(os.path.join(CHECKPOINT_DIR, "ablation_results.json"), "w") as f:
    json.dump(ablation_results, f, indent=2)
print(f"\\nSaved -> {CHECKPOINT_DIR}/ablation_results.json")
"""

# ── Cell D: ProtHGT re-run ────────────────────────────────────────────────────
CELL_D = """\
# Cell 11c: ProtHGT Re-run Through Our Metric Harness
# Run ProtHGT inference on the same .pt splits, then evaluate through identical
# propagation/threshold code so the comparison is apples-to-apples.
#
# Step 1 (run once): Clone ProtHGT and install deps
# !git clone https://github.com/HanwenXuTHU/ProtHGT /content/ProtHGT
# %cd /content/ProtHGT && pip install -r requirements.txt -q && %cd /content/Prot_B_poster
#
# Step 2: Run their inference script and save scores as [N_test_p, N_go] tensors
# !python /content/ProtHGT/eval.py \\
#     --data_root /content/drive/MyDrive/Poster\\ code/prothgt/knowledge_graphs/ \\
#     --output_dir /content/drive/MyDrive/Poster\\ code/checkpoints/ \\
#     --ontology bp
#
# Step 3 (this cell): Load saved scores and evaluate
import os, json, torch
from src.data.go_hierarchy import build_propagation_edges
from src.data.graph_builder import build_annotation_matrix
import numpy as np

PUBLISHED = {"bp": 0.7489, "mf": None, "cc": None}

prothgt_row = {}
for ont, ttype in [("bp", "GO_term_P"), ("mf", "GO_term_F"), ("cc", "GO_term_C")]:
    score_file = os.path.join(CHECKPOINT_DIR, f"prothgt_scores_{ont}.pt")
    if not os.path.exists(score_file):
        print(f"  {ont}: {score_file} not found -- skipping")
        continue
    if ont not in all_splits:
        print(f"  {ont}: split not loaded -- skipping")
        continue

    prothgt_scores = torch.load(score_file)   # [N_test_p, N_go]
    _, _, test_d = all_splits[ont]

    pe  = build_propagation_edges(test_d, target_type=ttype, cache=True)
    row, col, n_p, n_go = build_annotation_matrix(test_d, ttype)
    true_mat = torch.zeros(n_p, n_go, dtype=torch.float32)
    true_mat[row.cpu(), col.cpu()] = 1.0

    # Propagate predictions and labels up the GO DAG
    for child_i, parent_i in pe:
        prothgt_scores[:, parent_i] = torch.max(prothgt_scores[:, parent_i],
                                                 prothgt_scores[:, child_i])
        true_mat[:, parent_i] = torch.max(true_mat[:, parent_i],
                                           true_mat[:, child_i])
    true_mat = (true_mat > 0.5).float()

    has_annot = true_mat.any(dim=1)
    sm = prothgt_scores[has_annot]
    tm = true_mat[has_annot]
    ct_sum = tm.sum(1).clamp(min=1e-8)

    s_min_v, s_max_v = sm.min().item(), sm.max().item()
    t_steps = cfg.get("evaluation", {}).get("threshold_steps", 100)
    best_f1, best_t = 0.0, s_min_v
    for i in range(t_steps + 1):
        t    = s_min_v + i * (s_max_v - s_min_v) / t_steps
        pred = (sm >= t).float()
        tp   = (pred * tm).sum(1)
        prec = (tp / pred.sum(1).clamp(1e-8)).mean().item()
        rec  = (tp / ct_sum).mean().item()
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_t = f1, t

    prothgt_row[ont] = {"fmax": best_f1, "best_threshold": best_t}
    print(f"  ProtHGT-ESM2 (our harness) | {ont.upper()} | Fmax={best_f1:.4f}")

# Add to ablation table as reference rows
ablation_results["ProtHGT-ESM2 (our harness)"] = prothgt_row
ablation_results["ProtHGT-ESM2 (as reported)"] = {
    ont: {"fmax": v} for ont, v in PUBLISHED.items() if v is not None
}

from src.evaluation.metrics import print_ablation_table
print("\\nFull ablation table with ProtHGT reference rows:")
print_ablation_table(ablation_results)

with open(os.path.join(CHECKPOINT_DIR, "ablation_results_final.json"), "w") as f:
    json.dump(ablation_results, f, indent=2)
print(f"Saved -> {CHECKPOINT_DIR}/ablation_results_final.json")
"""

# ── Patch the notebook ─────────────────────────────────────────────────────────
cells = nb["cells"]

# Find cell index 6 (0-based) -- the Load Data cell
# Identify it by its source content
target_snippet = "load_prothgt_splits"
cell6_idx = next(
    i for i, c in enumerate(cells)
    if target_snippet in "".join(c.get("source", []))
)
print(f"Found data-load cell at index {cell6_idx}")

# Insert Cell B then Cell A after cell6_idx (insert in reverse order so indices stay valid)
cells.insert(cell6_idx + 1, make_code_cell(CELL_B))
cells.insert(cell6_idx + 1, make_code_cell(CELL_A))

# Find the original final-evaluation cell (index 14 originally, now shifted by 2)
eval_snippet = "evaluate_all"
final_eval_idx = next(
    i for i, c in enumerate(cells)
    if eval_snippet in "".join(c.get("source", []))
    and "baseline_fmax=0.7489" in "".join(c.get("source", []))
)
print(f"Found old final-evaluation cell at index {final_eval_idx}")

# Replace it
cells[final_eval_idx] = make_code_cell(CELL_14_NEW)

# Insert Cell C and D after it
cells.insert(final_eval_idx + 1, make_code_cell(CELL_D))
cells.insert(final_eval_idx + 1, make_code_cell(CELL_C))

nb["cells"] = cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nDone. Notebook now has {len(cells)} cells.")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))[:80].encode("ascii", "replace").decode()
    print(f"  {i:2d}: {src}")
