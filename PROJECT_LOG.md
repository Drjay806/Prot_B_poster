# Project Log — Prot_B_poster (GAN-RL Protein Function Prediction)

**Purpose of this file**: a complete, honest history of this project — what was built, every major path taken (including abandoned ones and why), every significant numeric result (including raw pasted output, not just summaries), every bug found and fixed, and the current open questions. Written so any person or LLM picking this up cold has full context without re-deriving it from chat history or git archaeology.

**Repo**: `Drjay806/Prot_B_poster` · **Data/runs**: Google Drive + Google Colab (GPU) · **Output target**: an academic poster/paper comparing this model against ProtHGT-ESM2.

**⚠️ Known unresolved inconsistency — read before citing any ProtHGT number**: this project's own history contains a self-correction that may not have been fully carried forward. See Section 5.2 and Section 9, item 1. Short version: at one point it was established that ProtHGT's cited Fmax=0.7489 comes from the **DeepHGAT benchmark** (a 2021/2023 temporal split, Human/Mouse/Arabidopsis only), *not* CAFA3 (2016/2017, broader scope) — but later planning in this same project referred to 0.7489 as a CAFA3 number. This needs to be re-checked against ProtHGT's paper (Table 3) before anything citing 0.7489 or "CAFA3" goes into the final paper.

---

## 1. Project Overview

**Goal**: predict Gene Ontology Biological Process (BP) annotations for proteins — i.e., given a protein, score how likely it is to carry each of ~27,855 possible GO-BP function labels — using a heterogeneous knowledge graph, encoded with CompGCN, with a GAN and an RL stage layered on top to push past what a plain encoder + scoring head achieves.

**Baseline being compared against**: **ProtHGT-ESM2**, a published heterogeneous-graph-transformer (HGTConv-based) model that scores (protein, GO term) pairs with an MLP head, trained on the same underlying knowledge graph with ESM2 protein embeddings.

### 1.1 Dataset provenance (verbatim from the project's own paper draft)

> "We repurpose the comprehensive heterogeneous dataset assembled by E. Ulusoy and T. Doğan. The knowledge graph (KG) they created integrates datasets on proteins, pathways, diseases, phenotypes, and drugs, using CROssBAR (Doğan et al. 2021) to unify human proteins. They incorporated proteins from 29 organisms using OMA orthology and kept only high-quality UniProtKB/Swiss-Prot annotations. They extended the dataset with protein domains from InterPro, GO terms from the Gene Ontology Consortium (GOA), and EC numbers from Expasy ENZYME. To prevent data leakage, they grouped proteins with high sequence similarity using UniRef90 and selected one representative from each cluster."

**Node counts** (from the paper draft's Table 1 — this is ProtHGT's own published dataset, inherited wholesale, not built by this project):

| Node type | Count |
|---|---|
| Protein | 326,783 |
| Compound | 133,104 |
| GO term | 42,885 |
| Domain | 10,229 |
| Drug | 6,179 |
| Disease | 5,694 |
| EC number | 4,627 |
| Pathway | 3,993 |
| **Total** | **542,465** |

**Edge counts**:

| Edge type | Count |
|---|---|
| Protein–Biological process | 1,000,952 |
| Protein–Molecular function | 926,844 |
| Protein–Cellular component | 672,139 |
| Protein–Domain | 461,668 |
| Protein–EC number | 128,242 |
| Protein–Compound | 153,764 |
| Orthology | 125,372 |
| GO term–GO term | 83,831 |
| Protein–protein interaction | 82,524 |
| Protein–Pathway | 51,692 |
| Protein–Phenotype | 38,239 |
| Disease–Phenotype | 24,259 |
| Protein–Drug | 22,718 |
| Protein–Disease | 12,817 |
| Domain–GO term | 5,791 |
| Disease–Pathway | 1,682 |
| Disease–Drug | 298 |
| **Total** | **542,465** |

**Note on "9 entity types / 17 relation types / 14 databases" claim** (raised in a paper-draft paragraph reviewed in this conversation): the 9-entity/17-relation counts check out against the table above and the runtime relation list this pipeline logs. The specific **"14 curated biomedical databases" figure was never verified anywhere in this project's history** — don't cite it without sourcing it from the Ulusoy/Doğan paper directly.

**Important data-provenance fact confirmed by direct investigation** (not an assumption): there is **no graph-construction code anywhere in this repository**. `src/data/loader.py` only `torch.load()`s three fixed `.pt` files; `src/data/graph_builder.py` only reads/queries an already-built graph; `src/data/go_hierarchy.py` only parses hierarchy out of existing `function_function` edges. `requirements.txt` lists `obonet` (a `.obo`-parsing library) as a dependency, but **it is never imported or used anywhere in the codebase** — dead weight, not a working path. The actual `.pt` graph files (`prothgt-esm2-{train,val,test}-graph.pt`) are external artifacts released by ProtHGT's own authors and are treated as black-box downloads. **This project has never built a knowledge graph from raw source data — it has always consumed ProtHGT's pre-built graph wholesale.**

**CompGCN encoding, from the paper draft**: "Because the raw representations vary widely in size and source (the protein ESM-2 vector is 1280 numbers, a GO term's anc2vec vector is 200 numbers, a domain's dom2vec vector is only 50 numbers), we standardize the initial embeddings to a 256-dimensional vector before the structural representation model... Message Composition: each layer composes source entity embeddings h_u and relation embeddings h_r using circular correlation: φ(h_u,h_r) = h_u ⋆ h_r. Aggregation & Refinement: composed representations are projected through relation-direction-specific transformation matrices (W_in, W_out, W_self) and aggregated over local neighborhoods with residual connections and layer normalization, mapping all entities and relations into a shared d=256 embedding space."

---

## 2. Pipeline Architecture

Code lives under `src/`; the whole pipeline is driven from `notebooks/train_colab.ipynb` (61 cells) on Colab GPU (originally T4-class, later a confirmed 80GB GPU — config scaled up accordingly, see `configs/full_graph.yaml`).

### Phase 1 — CompGCN pretraining (`src/training/pretrain.py`, `src/models/compgcn.py`)
Self-supervised encoder pretraining over the full heterogeneous graph. Current (v3) loss combination:
1. Symmetric InfoNCE with false-negative masking (weight 1.0)
2. Cosine alignment (weight 0.1)
3. ComplEx link-prediction loss (weight `dm_weight`)
4. Optional order-consistency hierarchy loss (weight `oc_weight`)

This replaced an earlier MSE/MMD-based Phase 1 (commit `1b51c7a`, 2026-06-08) that didn't work well enough, and an even earlier version restricted to only 2 edge types before the encoder was opened to the full graph (`ab22a88`, same day).

### Phase 2 — Adversarial training (`src/training/adversarial.py`, `src/models/generator.py`, `src/models/discriminator.py`)
- **Generator**: maps `[protein_emb(256) | relation_emb(256) | noise(64)]` → a 256-dim candidate GO embedding, via 4 FC layers with LayerNorm + LeakyReLU(0.2), no final activation.
- **Critic** (aliased `Discriminator` in code): WGAN critic using **spectral normalization** on every linear layer (not gradient penalty — found more stable, commit `69c4e75`). Scores are unbounded reals; high = compatible protein–function pair. Trained on real annotated pairs, generator-produced fakes, and tiered hard-negative samples.
- Critic updates `n_critic` times per generator update (later Optuna-searched).
- Encoder is **unfrozen** during Phase 2 (`f568322`) but its update is delayed until epoch 20 with LR=1e-5 to avoid destabilizing the Phase 1 representation (`e95d2c2`).
- Progress monitored via `_quick_fmax_complex()` (ComplEx-scored, matches `evaluate_all`'s official metric).

### Phase 3 — RL fine-tuning of the Generator (`src/training/rl_trainer.py`, `src/models/reward.py`)
- **Direct generator fine-tuning** (current default, `train_rl()`): loss = `w_sem*(1-cosine(fake,true)) + w_hier*(1-cosine(fake,ancestor_avg))` — this replaced an earlier, broken REINFORCE implementation (`dbaac7b`, 2026-06-08).
- **Genuine policy-gradient REINFORCE** (added back `f0c96af`, 2026-09-13) with a composite, partly-learnable reward:
  `R = w1*R_struct + w2*R_adv + w3(t)*R_sem − lambda_hier*P_hier`
  (`w1,w2,w3` learnable/softplus-positive; `w3` ramps 0→1 over a curriculum warmup.) `R_struct` from DistMult, `R_adv` from the Critic, `R_sem` from cosine-to-true, `P_hier` penalizes hierarchy violations.

### Phase 4 — Calibration head (`src/models/calibration_head.py`, `src/training/train_calibration.py`)
Optional score-recalibration head, added `2026-06-30` (`f1f88f6`). **Failed outright** (Fmax=0.0138) and was abandoned — see Section 4.3.

### Evaluation (`src/evaluation/metrics.py`, `src/evaluation/smin.py`)
Computes Fmax, Smin, AUROC, AUPR, Micro-F1, Macro-F1, MCC, Hit@1/3/10, MRR under a "filtered protocol." `_AUC_SAMPLE = 8000` — **AUROC/AUPR/MCC/Micro-F1/Macro-F1/Hit@k/MRR are all computed on an 8,000-protein random sample, not the full test population; only Fmax and Smin use the complete population.** `evaluate_all()` is the single source of truth reused everywhere, including the ProtHGT comparison.

### Node/edge-type structural contract (confirmed by direct code investigation, useful for any future graph work)
- Node types present: `"Protein"` and up to three GO-term types — `"GO_term_P"` (BP), `"GO_term_F"` (MF), `"GO_term_C"` (CC) (`ONTOLOGY_TO_NODE` in `loader.py`).
- `Protein.x`: `[N,1280]` (ESM2, asserted). `GO_term_*.x`: `[N,200]` (anc2vec, asserted). Node types without raw features (e.g. `Drug`) get an `nn.Embedding` fallback in `compgcn.py`.
- **Landmine, already solved once**: ProtHGT's raw `Protein` node sets `.num_nodes` to the full UniProt DB size (~13.7M) while `.x` only covers the ~261k featured proteins — code must always prefer `x.shape[0]` over `.num_nodes`.
- Required edge: `("Protein","protein_function","GO_term_*")`. Hierarchy edge: `("GO_term_*","function_function","GO_term_*")`, read as `(child, parent)` pairs.
- **No accession/GO-ID mapping attribute exists anywhere on the `HeteroData` object or in any loader code** — everything is pure positional integer indexing. Any external ID mapping must be built and kept separately.
- Train/val/test are **transductive at the node level** — same protein/GO node space shared across all three splits; only the supervision *edges* differ per split. `validate_split_disjointness` checks edge-level, not node-level, disjointness.
- **No date/timestamp/cutoff/temporal metadata exists anywhere in this codebase** — confirmed via exhaustive regex search across `src/` and the notebook. The 80/10/10 split is purely protein/edge-level and transductive; there is no notion of "when an annotation was added" anywhere in the current pipeline.

---

## 3. Timeline of Major Milestones

| Date | Milestone |
|---|---|
| 2026-06-05 | Initial CompGCN pipeline working end-to-end; wave of OOM fixes (20GB→9GB on T4), dtype/autocast crash fixes, early GAN explosion fixed with L2 normalization + LayerNorm |
| 2026-06-07 | Phase 2 overhaul: WGAN-GP with tiered negative sampling; critic dominance fixed via spectral norm; major bug found — **wrong relation index was causing val_Fmax≈0** |
| 2026-06-08 | Phase 1 rewritten (InfoNCE + ComplEx LP); Phase 3 rewritten (direct generator fine-tuning replaces a broken REINFORCE attempt); `evaluate_all` fixed to use cosine similarity instead of an untrained DistMult head |
| 2026-06-21–23 | Full Phase 1 loss rewrite (symmetric InfoNCE, false-negative masking, ComplEx LP, order-consistency, ancestor-similarity hierarchy); CAFA evaluation harness added (Smin, multi-ontology ablation); long sequence of T4-memory fixes |
| 2026-06-24–30 | Encoder unfrozen during Phase 2 with delayed/low-LR update; Hit@k/MRR + shallow ComplEx/DistMult baselines added; first ProtHGT head-to-head evaluation + calibration head added |
| 2026-07-01 | Calibration-mode fix in streaming evaluation loop |
| 2026-09-06–08 | **Full-knowledge-graph retrain path added**; leakage-aware Fmax evaluation added; GPU upgrade to confirmed 80GB card, config scaled up; full-graph encoder wired into Phase 2 onward; 100-epoch full Phase 2 run completed |
| 2026-09-09–11 | Multiple correctness bugs found/fixed in Phase 2 progress metric and a baseline-comparison cell (see Section 6); `freeze_encoder` option added to isolate the GAN's standalone value |
| ~2026-09 (mid, exact date uncertain) | **Discrepancy found and corrected**: ProtHGT's cited Fmax=0.7489 traced to Table 3 of their paper — it's the **DeepHGAT benchmark** result (time-based split, train = annotations before 2021-09-01, test = curated 2021-09-01 to 2023-04-01, restricted to Human/Mouse/Arabidopsis), *not* the paper's own 80/10/10 cold-start split, and **not CAFA3**. No ProtHGT number exists anywhere on the exact split this project's data loads. ⚠️ See Section 9, item 1 — this correction may have been lost/re-conflated later in the project. |
| 2026-09-13–15 | No-adversarial ablation added (isolating the Generator's adversarial term); genuine REINFORCE Phase 3 added; OFAT weight-sensitivity search (adv→dm→anchor), first tuned config: `adv=0.3, dm=0.1, anchor=1.5` (test Fmax=0.7085 at gen_weight=0.85) |
| 2026-09-16–17 | Test-set confirmation of OFAT-tuned run; ablations re-run at tuned weights |
| 2026-09-20–22 | **Optuna joint hyperparameter search** (25 trials, 7 parameters, composite objective 0.5×Fmax+0.5×MCC) converging on `adv=0.815, dm=0.256, anchor=1.053, lr_generator=2.85e-4, lr_critic=2.47e-5, n_critic=2, hard_neg_temperature=0.883` (test Fmax=0.7032, MCC=0.5762 at gen_weight=0.85); separate RL/REINFORCE hyperparameter search run (15 trials, best Fmax=0.6878/MCC=0.5462 — did not exceed the pre-RL baseline); leakage check rebuilt/triple-verified; detailed test-split data overview added |
| 2026-09-27 | Leakage check, data-only setup, test-split overview cells restored after being lost; no-adversarial ablation re-isolated; single-fix RL v2 replaced with the proper hyperparameter search |
| 2026-10-05 | **`HGTConvPure`**: pure-PyTorch reimplementation of PyG 2.2.0's `HGTConv`, replacing the slow old-PyG/conda ProtHGT inference workflow |
| (this conversation, Oct 2026) | ProtHGT full-candidate evaluation run and diagnosed (Fmax=0.2035, not a bug — negative-sampling/calibration mismatch); restricted-candidate-list experiment tried and found worse; a CAFA3-style temporal-split investigation and build plan were produced; de-prioritized in favor of the already-completed comparison; this log file created |

---

## 4. Key Results

All numbers use `evaluate_all()` unless noted — the same evaluation code for every model/configuration.

### 4.1 Final / headline numbers

| Model | Config | Fmax | Smin | AUROC | AUPR | Micro-F1 | Macro-F1 | MCC |
|---|---|---|---|---|---|---|---|---|
| **ProtHGT-ESM2** | released checkpoint, ProtHGT's own 80/10/10 split, full ~27,855-term candidate space | 0.2035 | 212.1923 | 0.8007 | 0.0858 | 0.1961 | 0.0031 | 0.1949 |
| **Our model** | Optuna-tuned adversarial, `gen_weight=0.85`, test set | 0.7032 | 159.9146 | 0.9896 | 0.6422 | 0.5766 | — | 0.5762 |
| Our model (alt) | Optuna-tuned adversarial, `gen_weight=1.00`, test set | 0.7185–0.7207 | 162.5535–172.5642 | 0.9883–0.9892 | 0.5916–0.6324 | 0.4292–0.5313 | 0.1359–0.1439 | 0.4655–0.5393 |

> **Open question**: a further RL/REINFORCE stage runs on top of the Optuna-tuned adversarial model. Validation-set post-RL sweeps were found (`[VAL, post-RL/Optuna] gen_weight=0.85: Fmax=0.7053`), but no confirmed **test**-set post-RL number was located. Best RL search result overall (Fmax=0.6878/MCC=0.5462, "near but not exceeding the pre-RL baseline of 0.7044/0.5591") suggests the RL stage, as searched so far, does **not** improve on the pre-RL Optuna-tuned model — the headline number is very likely the pre-RL Optuna-tuned-adversarial result in 4.1, not a post-RL number. Confirm before finalizing.

### 4.2 Main Findings — ablation summary (with causal reasoning, from the paper-draft "Main Findings" writeup)

**Critic-as-scorer**: using the trained Critic directly as a ranking score never beats the raw encoder.
- Critic alone: Fmax=0.2080. Critic blended with ComplEx: Fmax=0.5433. Raw encoder alone: Fmax=0.5462.
- *Reasoning*: the Critic was trained only to distinguish real-vs-fake pairs (a local, pairwise task), not to produce a globally well-calibrated ranking across the whole GO-term space, which is what Fmax needs.

**Generator's blended score (`gen_weight`)**: large, consistent gains, reproduced on two independently-trained models.
- OFAT: 0.546 → 0.709. Optuna: 0.546 → 0.703–0.719.
- *Reasoning*: the Generator learns a specialized, task-tuned nonlinear transform of the embeddings (via anchor/dm losses), outperforming the generic hand-designed ComplEx bilinear score on the same embeddings.

**Adversarial term, isolated (adv=0.815 vs adv=0.0, everything else fixed)**: Fmax barely moves (±0.01); MCC/Micro-F1 collapse without it, worst at gen_weight=1.0 (MCC drop 0.147, Micro-F1 0.56→0.35).
- *Reasoning*: Fmax picks the single best threshold per run and tolerates an unbalanced score distribution; MCC/Micro-F1 use a fixed threshold and directly penalize poor calibration.

**gen_weight=0.85 vs 1.0**: 1.0 wins on raw Fmax (0.719 vs 0.703); 0.85 wins on MCC, Micro-F1, Smin.
- *Reasoning*: relying entirely on the Generator trades calibration/balance for a small ranking gain.

**OFAT / Optuna searches**: both found working configs, but **neither isolates any single term's causal contribution** — they vary parameters jointly. The 18-trial-table honest read shows no clean monotonic trend between `adv` weight and score (high-adv trials scored both best and worst) — a structural limitation of joint search, not a finding about any one term.

**Data leakage / split methodology**: corrected analysis showed **0% true protein-level overlap** between train/val/test (exactly 80.0/10.0/10.0% by protein count) — a clean cold-start split. The original false-positive leakage signal came from comparing raw `(protein_idx, GO_idx)` tuples across splits that each use independent local index spaces — coincidental index collisions, not real shared identity.

**Calibration head (Phase 4)**: failed outright, Fmax=0.0138, abandoned. Trained only on frozen Phase 1 embeddings (never benefiting from Phase 2's richer representations); never load-bearing for the paper's ranking metrics anyway.

**Best-of-K differentiable refinement**: only ever run on an old, pre-full-graph pipeline version (Fmax≈0.13) — never tested on the current architecture. Status: "not evaluated," not "found to not work."

**REINFORCE, original attempt**: catastrophic failure, Fmax dropped from ~0.72 to ~0.37 at gen_weight=1.0. Root cause (diagnosed via code, not guessed): `curriculum_warmup_epochs` ≈ total run length, so the semantic/hierarchy reward term (the only one rewarding *correctness*, not just plausibility) was suppressed for nearly the entire run.

**REINFORCE, post-fix search (15 trials)**: best trial (`policy_sigma=0.467`) reached Fmax=0.6878/MCC=0.5462 — close to but still below the pre-RL baseline (0.7044/0.5591). *Reasoning*: Optuna's TPE sampler spends its first ~10 trials on pure random exploration by default, leaving only ~5 trials of guided exploitation — search may not have converged; alternatively, RL fine-tuning on top of an already loss-optimized Generator may have a genuinely low ceiling.

### 4.3 Optuna hyperparameter search (adversarial stage, 25 trials) — raw trial data

Searched: `adv`, `dm` (domain-mixing/ComplEx margin weight), `anchor`, generator LR, critic LR, `n_critic`, hard-negative temperature. Composite objective: 0.5×Fmax + 0.5×MCC.

Representative raw trial outputs (`[GO_term_P mode=encoder]` blocks, test/val sweep at `gw=0.85` unless noted):

```
Trial 0:  adv=— dm=— anchor=— (baseline config)         Fmax=0.7011  MCC=0.5308  (trial value: 0.615)
Trial 1:  adv=0.866 dm=0.301 anchor=2.153 lr_g=1.10e-05  Fmax=0.6635  MCC=0.5134
Trial 2:  adv=0.182 dm=0.092 anchor=0.982 lr_g=1.12e-04  Fmax=0.7062  MCC=0.5176
Trial 3:  adv=0.139 dm=0.146 anchor=1.162 lr_g=8.17e-05  Fmax=0.6981  MCC=0.5227
Trial 4:  adv=0.592 dm=0.023 anchor=1.862 lr_g=2.19e-05  Fmax=0.6863  MCC=0.5237
Trial 5:  adv=0.808 dm=0.152 anchor=0.383 lr_g=2.34e-04  Fmax=0.6913  MCC=0.5783
Trial 6:  adv=0.034 dm=0.455 anchor=0.850 lr_g=2.11e-04  Fmax=0.6949  MCC=0.5514
Trial 7:  adv=— (full config)                             Fmax=0.7102  MCC=0.5432  (trial value: 0.626)
Trial 8:  adv=0.088 dm=0.098 anchor=0.231 lr_g=4.47e-05  Fmax=0.6907  MCC=0.5669
...
Trial 17: adv=0.676 dm=0.248 anchor=1.428 lr_g=1.63e-04  Fmax=0.7049  MCC=0.5635
Trial 18: adv=0.744 dm=0.352 anchor=1.230 lr_g=5.95e-04  Fmax=0.7047  MCC=0.5750
Trial 19: adv=0.827 dm=0.188 anchor=0.929 lr_g=2.06e-04  Fmax=0.6976  MCC=0.5750
Trial 20: adv=0.792 dm=0.220 anchor=1.212 lr_g=1.91e-04  Fmax=0.7042  MCC=0.5641  (trial value: 0.63)
Trial 21: adv=0.752 dm=0.237 anchor=1.361 lr_g=1.25e-04  Fmax=0.6968  MCC=0.5715
Trial 22: adv=0.981 dm=0.109 anchor=0.596 lr_g=2.61e-04  Fmax=0.6526  MCC=0.5547  (trial value: 0.60)
Trial 23: adv=0.981 dm=0.109 anchor=0.596 lr_g=2.61e-04  Fmax=0.6978  MCC=0.5627
Trial 24: adv=0.792 dm=0.220 anchor=1.212 lr_g=1.91e-04  Fmax=0.7032  MCC=0.5697

BEST (study-level): lr_critic=2.465e-05, n_critic=2, hard_neg_temperature=0.883
  -> Fmax=0.7042  MCC=0.5888  MicroF1=0.5894
Total trials in study: 25.  Total wall time this session: 475.9 min.
```

**Final paper sentence (confirmed numbers)**:
> "...then Optuna/TPE jointly over all 7 parameters (25 trials, composite objective), converging on adv=0.815, dm=0.256, anchor=1.053, lr_generator=2.85×10⁻⁴, lr_critic=2.47×10⁻⁵, n_critic=2, hard_neg_temperature=0.883 (test Fmax=0.7032, MCC=0.5762 at gen_weight=0.85)."

### 4.4 `gen_weight` sweep — raw VAL and TEST blocks

VAL sweep:
```
gen_weight=0.00  Fmax=0.5462  Smin=204.9062  AUROC=0.9761  MCC=0.3860  MicroF1=0.3763
gen_weight=0.30  Fmax=0.6078  Smin=190.5133  AUROC=0.9852  MCC=0.4702  MicroF1=0.4692
gen_weight=0.50  Fmax=0.6460  Smin=178.3187  AUROC=0.9879  MCC=0.5295  MicroF1=0.5301
gen_weight=0.70  Fmax=0.6813  Smin=167.7374  AUROC=0.9891  MCC=0.5656  MicroF1=0.5661
gen_weight=0.85  Fmax=0.7044  Smin=160.5815  AUROC=0.9897  MCC=0.5591  MicroF1=0.5571
gen_weight=1.00  Fmax=0.7190  Smin=161.2907  AUROC=0.9896  MCC=0.5607  MicroF1=0.5586
```
TEST sweep:
```
gen_weight=0.00  Fmax=0.5501  Smin=203.2621  AUROC=0.9769  MCC=0.3882  MicroF1=0.3751
gen_weight=0.85  Fmax=0.7032  Smin=159.9146  AUROC=0.9896  MCC=0.5762  MicroF1=0.5766
gen_weight=1.00  Fmax=0.7185  Smin=162.5535  AUROC=0.9892  MCC=0.5393  MicroF1=0.5313
```

### 4.5 No-adversarial ablation — exact numbers (paper Section 5.3.4)

Retrained Generator with adversarial weight forced to 0, all other Optuna-tuned parameters held fixed (`dm=0.256, anchor=1.053`, learning rates, `n_critic`, hard-neg temperature unchanged). Compared across the full `gen_weight` sweep against the adversarial-on model:

| gen_weight | Fmax (adv=0) | Fmax (adv=0.815) | MCC (adv=0) | MCC (adv=0.815) | Micro-F1 (adv=0) | Micro-F1 (adv=0.815) |
|---|---|---|---|---|---|---|
| 0.00 | 0.5462 | 0.5462 | 0.3842 | 0.3860 | 0.3731 | 0.3763 |
| 0.85 | 0.7053 | 0.7044 | 0.5179 | 0.5591 | 0.5019 | 0.5571 |
| 1.00 | 0.7174 | 0.7190 | **0.4135** | **0.5607** | **0.3452** | **0.5586** |

**Conclusion (verbatim from the paper draft, Section 5.3.4)**: *"Fmax proved largely insensitive to the manipulation, differing by no more than 0.013 at any blend weight and even marginally favoring the adversarial-off condition at gen_weight=0.85 (0.7053 vs. 0.7044). MCC, by contrast, showed a clear and widening gap: from a negligible difference at gen_weight=0.00... to a substantial 0.147 deficit at gen_weight=1.00 (0.5607 vs. 0.4135)... Micro-F1 followed the same pattern, dropping from 0.5586 to 0.3452 under the same condition."* Interpretation: the adversarial term regularizes the Generator's output distribution toward separability/consistency, not toward a single higher-scoring answer — invisible to Fmax, substantial under fixed-threshold metrics.

---

## 5. The ProtHGT Comparison Saga

### 5.1 Why it was hard in the first place
ProtHGT's released, usable artifact is a PyTorch Geometric `HGTConv`-based checkpoint. Running it required an old PyG + old-torch conda environment (CPU-only in practice) — slow and awkward alongside this repo's modern-PyTorch pipeline.

### 5.2 ⚠️ Critical correction found in this project's own history — the 0.7489 number
At one point mid-project, a discrepancy was explicitly found and documented: *"I traced it [0.7489] in Table 3 — it's ProtHGT-ESM2's Fmax on the BP sub-ontology of the DeepHGAT benchmark. This is a completely different evaluation setup than I'd assumed... DeepHGAT is a time-based split, not the same 80/10/10 cold-start protein split... 'all functional annotations available for three species (Human, Mouse, and Arabidopsis) prior to 2021-09-01 were used for training, while annotations curated between 2021-09-01 and 2023-04-01 formed the test set.' Separately, Section 2.3 describes their main KG dataset split as 80/10/10 at the protein level, cold-start — matching your project's own verified split ratios almost exactly... There is currently no published ProtHGT number measured on the exact split your project actually uses."*

**This means**: 0.7489 is a **DeepHGAT-benchmark number** (2021/2023 dates, 3 species only), not a CAFA3 number (which would be 2016/2017 dates, broader species scope). **However**, later planning in this same project (the Tier 1–4 comparison-rigor framework, and the approved CAFA3-acquisition plan) referred to 0.7489 as coming from "CAFA3." **This is an unresolved internal inconsistency.** Possible explanations, not yet checked: (a) the paper reports a *separate* CAFA3-specific number distinct from the DeepHGAT number, and later planning correctly targeted that different number, or (b) the project re-conflated DeepHGAT and CAFA3 after this correction was made. **Action needed before writing anything citing 0.7489 or "CAFA3" in the final paper: re-open ProtHGT's paper, Table 3, and confirm exactly which benchmark(s) they report BP Fmax for, and which number is which.**

One more fact surfaced during this same check: ProtHGT's leakage prevention includes **UniRef90 sequence-similarity clustering (90% identity threshold)**, applied during *construction of the dataset itself* (Section 2.1 of their paper, before the 80/10/10 split description) — see Section 5.6 below for the full resolution of this point.

### 5.3 `HGTConvPure`: the GPU-fast reimplementation
From-scratch, pure-PyTorch reimplementation of PyG 2.2.0's `HGTConv` (no `torch_geometric`/`torch_scatter`/`torch_sparse` dependency), module names and math matching the real layer exactly. Committed `0494c4f` (2026-10-05).

**Correctness verification**:
1. Standalone unit test (construction, forward-pass shapes/finiteness, state_dict key-naming match, strict round-trip load with 0 missing/0 unexpected keys, CPU/GPU execution) — all passed.
2. **Decisive test**: ran the real PyG `HGTConv` in the old conda environment on actual ProtHGT data, saved output embeddings, compared against `HGTConvPure`'s output. **Cosine similarity = 1.000000** across every node, differences only at floating-point noise level (1e-6 to 2e-5).

### 5.4 The "is this a bug?" investigation
Running ProtHGT (via `HGTConvPure`) on its own released split, graded with this repo's evaluation code: **Fmax=0.2035 despite AUROC=0.8007**. Diagnosis (not assumption, confirmed via 5.3's embeddings comparison): not a bug. ProtHGT's MLP scoring head was trained with only **100 negatives per positive** (per their paper); at test time it's scored against the full ~27,855-term candidate space — ~278× larger than what it was calibrated for. Symptoms matched exactly: scores saturated at extremes (best threshold=0.997), Macro-F1≈0.0031 (failing almost every individual term), while AUROC (ranking-only, calibration-insensitive) stayed healthy at 0.80.

**Scope limit, stated honestly**: this is a confirmed property of ProtHGT's *training* methodology. Whether it also degrades ProtHGT's own *published* headline number is **not verified** — unknown whether their own benchmark submission (DeepHGAT or CAFA3, per 5.2) scored exhaustively against the full candidate space or a restricted subset.

### 5.5 The restricted-candidate-list dead end
Hypothesis: a smaller, more "realistic" candidate list (every GO term true for *someone* in the test set) might be fairer to ProtHGT. **Result: made things worse.** Fmax dropped 0.2035→0.0243, AUROC dropped 0.80→0.65 — restricting removed many "easy" true negatives, leaving a harder, more confusable pool. Decision: use the full-candidate result as the honest, final number; cite this failed experiment as additional evidence the gap reflects genuine calibration differences, not an artifact of evaluation choices.

### 5.6 UniRef90 — resolved, not a gap
Question raised: does this project need to independently run UniRef90 (90% sequence-identity) clustering to match ProtHGT's own leakage-prevention rigor? **Answer: no.** UniRef90 clustering is applied by ProtHGT's own authors *during construction of the dataset itself* (their paper's Section 2.1, before the split description) — Table 1's protein count (326,783) is already *after* this deduplication. Since this project loads ProtHGT's own released graph and split wholesale, the deduplication was already done before the files ever arrived here. **One small discrepancy worth disclosing, not fixing**: this project's own loaded protein count across train+val+test (261,373+32,669+32,675 = 326,717) differs from the paper's reported 326,783 by 66 proteins (~0.02%) — likely a dataset-version/snapshot difference between their GitHub release and the paper's description, not evidence of a problem. Suggested paper wording: *"Sequence-similarity-based leakage control (UniRef90 clustering at 90% identity) was applied by the original authors during construction of the underlying dataset, prior to the train/validation/test split; we did not independently re-verify this deduplication, and our loaded protein count (326,717) differs marginally from the count reported in the original publication (326,783), a discrepancy of under 0.1% likely attributable to dataset versioning."*

### 5.7 Honest-framing decision
Explicit instruction carried through the project: **"we shouldn't say ours is better, this one wasn't trained for that particular task."** Agreed framing: report numbers under identical split + identical evaluation code (fully legitimate), attribute the gap honestly to both real architectural differences (ablation-supported, Section 4.2) and the negative-sampling/calibration mismatch (Section 5.4) — not to a blanket "our architecture is better" claim.

**⚠️ Conflict with existing paper draft**: the Conclusion draft already written for this paper (Section 10.3 below) states *"our model performs substantially better... We attribute this advantage to two properties of our architecture"* with **no mention of the calibration-mismatch confound**. This directly contradicts the honest-framing decision above and was independently flagged as a problem later in this same conversation when a similar paragraph was reviewed. **The existing Conclusion draft needs revision before use** — add the calibration-mismatch caveat (template given in Section 10.3's note).

---

## 6. Bugs Found & Fixed

| Bug | Root cause | Fix | Commit |
|---|---|---|---|
| val_Fmax≈0 in Phase 2 | Wrong relation index selected (needed both "protein" AND "function" in the name) | Corrected relation-index selection | `903620f`, `3137fdd` |
| GAN explosion (DistMult scores diverging) | No normalization constraint on generator output / discriminator inputs | L2-normalize outputs/inputs; final LayerNorm | `75ae760`, `5bcd51e` |
| GAN training collapse | Discriminator too strong relative to generator | Weakened discriminator, later restored with spectral norm for stability | `ec38fb6`, `96694d5` |
| `evaluate_all` giving wrong signal | Used an untrained DistMult head instead of cosine similarity | Switched to cosine similarity | `ba9c0f6` |
| REINFORCE (first attempt) broken | — (replaced wholesale, root cause not separately isolated at the time) | Replaced with direct generator fine-tuning (cosine + hierarchy loss) | `dbaac7b` |
| OOM on T4 (repeated) | ~20GB+ memory footprint from full-graph CompGCN forward pass | Edge chunking, gradient checkpointing, `hidden_dim` 256→128 (restored once on 80GB GPU), dropped redundant `rev_*` edges, CPU-resident raw features | multiple, 2026-06-05 to 06-23 |
| `_quick_fmax_complex` reading ~0.14 low vs `evaluate_all` | Flat 51-point threshold grid, too coarse | Fine-grained threshold resolution in top 10% of score range | `41cab1c`, `367378c` |
| Cell 8L comparing two different Fmax scorers | Baseline (0.3470) was a fit-test artifact, not a real comparison; cell wasn't actually testing the trained full-graph encoder | Corrected baseline + cell logic | `7e1e8f5`, `a5fff5f` |
| "Leakage check" cell never tested the trained encoder | Notebook reused plain variable names (`encoder`, `cfg`, `CHECKPOINT_DIR`) meaning different things at different points; a bad Fmax=0.1007/0.1602 "ComplEx encoder-mode" number propagated into a second cell's own calibration baseline before being caught | Rebuilt and triple-verified leakage check for val+test | `e6236a7` (restored after being lost: `20fad94`) |
| Phase 1→Phase 2 checkpoint contamination | Phase 2 inadvertently started from a stale/wrong Phase 1 checkpoint | Eliminated contamination, rebuilt Phase 2 flow cleanly | `e0507e3` |
| False-positive data-leakage signal | Raw `(protein_idx, GO_idx)` overlap checks compared indices across splits using independent local index spaces — coincidental collisions | Corrected to protein-ID-level check; confirmed 0% true overlap | (see Section 4.2) |
| REINFORCE catastrophic failure (Fmax 0.72→0.37) | `curriculum_warmup_epochs` ≈ total run length, suppressing the semantic/hierarchy reward term for nearly the whole run | Fixed curriculum timing, then ran a 15-trial hyperparameter search on top | `f0c96af` + later search |
| ProtHGT Fmax=0.2035 "looks like a bug" scare | Not a bug — see Section 5.4 | N/A (diagnosis, not a code fix) | — |

---

## 7. Paths Explored But Not Taken

- **Tier 2** of the comparison-rigor framework (emailing the DeepHGAT authors for their unreleased benchmark data) — **explicitly ruled out by the user**.
- **Restricted-candidate-list grading** (Section 5.5) — tried, made results worse, abandoned.
- **Tier 4** (retrain both models under fully identical conditions from scratch) — flagged as too large, not pursued.
- **CAFA3 full retrain** (Tier 3 — Section 8) — plan written and approved, data source identified, but paused/de-prioritized against a poster deadline.
- **CAFA5-on-Kaggle** — investigated as an easier data source, ruled out: different, non-comparable benchmark from CAFA3.
- **ProtHGT's own GitHub repo as a CAFA3 data source** — checked twice (full file-tree listing both times), confirmed absent.
- **Best-of-K differentiable refinement on the current architecture** — implemented but only ever evaluated on an old, pre-full-graph pipeline version; status is "untested on current pipeline," not "abandoned."
- **Isolated ablation of the tiered negative-sampling design** (vs. uniform random) — not done; `hard_neg_temperature` was only included in the joint hyperparameter search, confounded with other parameters.
- **Isolated ablation of the RL hierarchy-consistency term** (`lambda_hier`) — not done; bundled into the combined reward.
- **ProtHGT-TAPE / ProtHGT-ProtT5 comparison** — not done; only ProtHGT-ESM2 was compared (the architecturally fair choice, matching this project's own protein embeddings), but the original ProtHGT paper shows both TAPE and ProtT5 variants outperform ESM2 on every metric — so this project is not benchmarked against ProtHGT's single best-reported configuration.
- **Statistical significance testing** (paired bootstrap, permutation tests) between configurations — not done; all comparisons are point estimates. No repeated/multi-seed runs either — only an informally observed noise floor of ~0.004–0.006 Fmax from comparing nominally similar runs.

---

## 8. CAFA3 Comparison Path — Status

**⚠️ Re-read Section 5.2 first** — the original motivation for this path ("get a number comparable to ProtHGT's published 0.7489") rests on an assumption (0.7489 = CAFA3) that was explicitly corrected once (0.7489 = DeepHGAT) and may have been re-introduced later without re-checking. Independent of that specific number, CAFA3 remains valuable as a **standard, independently-meaningful community benchmark** (BLAST/Naive baselines, dozens of published competitor numbers) — but confirm what you're actually trying to match before investing more effort here.

**What was done**:
1. Confirmed CAFA3's official raw data lives on Synapse (login-gated).
2. Found a **Zenodo-hosted CAFA3-derived dataset** (DOI `10.5281/zenodo.7409660`, CC-BY 4.0, no login) as a lower-friction alternative.
3. **Known caveat**: this Zenodo dataset is a cleaned derivative ("functions presented in at least 50 occurrences... duplicated proteins removed"), not the byte-identical official release. Date-cutoff alignment with the real CAFA3 boundary was never independently verified.
4. A full 6-milestone plan was written and approved (`C:\Users\dysonj\.claude\plans\wondrous-singing-peacock.md`): accession-mapping recon → raw data acquisition → graph construction (`src/data/cafa3_builder.py`, not written) → smoke-test pretraining → full Phase 1+2 training → dual evaluation (internal + official `CAFA-evaluator` tool).
5. Milestone 0 found that ProtHGT's own released `.pt` files carry an `id_mapping` (261,373 protein accessions, 27,855 GO IDs) never exposed by this repo's own loader — meaning CAFA3 embeddings could potentially reuse ProtHGT's precomputed ESM2/anc2vec vectors.
6. A follow-up investigation (full report preserved in Section 8.1 below) **confirmed definitively** that this repo has zero graph-construction code of any kind, and zero date/temporal metadata anywhere — a CAFA3 graph would need to be built entirely from scratch using an external builder, with dates/cutoffs tracked completely outside the existing codebase.

**Current recommendation**: **do not pursue this for the poster.** Requires a full new data pipeline + full retrain (multi-day/week). A legitimate, verified, honestly-framed comparison (Section 5) already exists. Treat as one-line "future work." **Preserved here in case priorities change.**

### 8.1 Structural requirements for a future CAFA3 build (from direct investigation, not guesswork)
- No files exist yet: `src/data/cafa3_builder.py`, `src/data/cafa3_raw.py`, `configs/cafa3.yaml`, `src/evaluation/cafa_export.py` are all planned, none written.
- Minimum graph shape needed: `Protein.x [N,1280]`, target GO-type `.x [N,200]`, a `("Protein","protein_function",target_type)` edge, ideally a `(target_type,"function_function",target_type)` hierarchy edge for the order-consistency loss and tiered negative sampler. Everything else is optional/additive — `compgcn.py`'s edge handling is fully generic.
- Train/val/test should most likely **share identical node tensors** and differ only in which `protein_function` edges are included as supervision, by date — this matches the shape `build_annotation_matrix`/`validate_split_disjointness` already expect, so it's the path of least resistance (rather than building fully separate node spaces per split).
- No accession/GO-ID mapping attribute should be added to the `HeteroData` object itself (nothing downstream reads one) — keep that mapping in separate external files.
- `Phase 1 (pretrain())` and `Phase 2 (train_adversarial())` both need only `(encoder, train_data, val_data, cfg, ...)` plus, for Phase 2, an `ancestor_table` built via `go_hierarchy.build_ancestor_table()` off the new graph's `function_function` edges. Both phases are structurally agnostic otherwise — "should just work" once the graph contract is satisfied.
- Expected Drive layout mirrors the existing ProtHGT convention: a root folder, a subdirectory with the three `.pt` graph files, and a separate raw-source-file area (format/layout entirely your own choice, since no convention exists in-repo for this).

---

## 9. Current Status / Open Questions

1. **⚠️ Highest priority**: resolve the DeepHGAT-vs-CAFA3 conflation (Section 5.2) before writing anything in the paper that cites "0.7489" or "CAFA3." Re-check ProtHGT's paper Table 3 directly.
2. **Which exact "our model" number is the paper's headline figure?** Likely the pre-RL Optuna-tuned-adversarial result (Section 4.1) since the RL search never exceeded it — but confirm no better post-RL test number exists.
3. **CAFA3 path**: paused by recommendation, not technically blocked — Section 8 has the exact resume point.
4. **"14 curated biomedical databases" claim**: unverified anywhere in this project's history — don't cite without sourcing from Ulusoy & Doğan directly.
5. **Existing Conclusion draft (Section 10.3) needs revision**: currently claims blanket superiority without the calibration-mismatch caveat, contradicting this project's own honest-framing decision (Section 5.7). A similar paragraph was independently flagged as a problem later in the same conversation — the same fix applies to the formal draft.
6. Whether ProtHGT's own published headline number suffers the same full-candidate-space calibration collapse observed here (Section 5.4) is unverified either way.
7. **TPE/REINFORCE search may be under-converged** — only ~5 of 15 trials were genuine guided exploitation (TPE's default ~10-trial random-exploration phase ate most of the budget); a longer search might still beat the pre-RL baseline.

---

## 10. Paper-Drafted Content (verbatim, for reuse/revision)

### 10.1 Limitations (comprehensive version, from the paper draft's working notes)

**Computational**
- Only the BP sub-ontology was evaluated — not MF or CC — due to compute/time cost.
- No temporal benchmark evaluation (CAFA3/DeepHGAT) performed — only the internal cold-start 80/10/10 split (and see Section 5.2's caveat about which benchmark 0.7489 actually is).
- The RL hyperparameter search used short 20-epoch trials for ranking — the true answer depends on the 40-epoch confirmation run.
- `gen_weight` was swept only at fixed points (0.0, 0.3, 0.5, 0.7, 0.85, 1.0), not a finer continuous sweep.
- AUROC/AUPR/MCC/Micro-F1/Macro-F1 are computed on an 8,000-protein sample, not the full population (Fmax/Smin use the full population).
- No repeated/multi-seed runs — only an informal noise floor (~0.004–0.006 Fmax).
- ProtHGT's checkpoint had to be run on CPU only historically (old torch/PyG constraints) — superseded by `HGTConvPure` (Section 5.3), which now runs on GPU.

**Methodological / scope**
- No isolated ablation of the negative-sampling design (tiered/curriculum vs. uniform random).
- No isolated ablation of the RL hierarchy-consistency term (`lambda_hier`).
- Best-of-K refinement never tested on the current pipeline.
- Calibration head never re-tuned after its one failed attempt.
- No genuine DeepHGAT-benchmark retraining performed — would require sourcing that benchmark's own temporal-split data independently.
- Only compared against ProtHGT-ESM2, not ProtHGT-TAPE or ProtHGT-ProtT5 (both outperform ESM2 in ProtHGT's own paper).
- Sequence-similarity leakage control (UniRef90) was applied by ProtHGT's authors at dataset-construction time, inherited automatically — not independently re-verified by this project (see Section 5.6 for exact wording to use).
- No statistical significance testing between configurations.

### 10.1b Section 5.3.1 — Does the Critic add value as a scorer (final drafted text)

> To test whether the trained Critic contributes useful signal on its own, we froze the Phase 1 encoder (Fmax=0.5462 on validation) and trained the Generator-Critic pair for 100 epochs with the encoder held fixed. We then scored validation proteins three ways using the same trained checkpoint: (1) the raw encoder/ComplEx score alone, (2) the Critic's score used directly as a ranker, and (3) a tuned ensemble blend of the two (alpha=0.90, favoring the encoder). Results: encoder=0.5462, critic=0.2080, ensemble=0.5433. Neither the Critic alone nor the best blend improved on the raw encoder baseline, indicating that the Critic's judgment does not transfer into a useful standalone scoring signal.

### 10.1c Section 5.3.2 — The Generator's learned embedding improves ranking (final drafted text)

> While the Critic does not help as a direct scorer (5.3.1), the Generator's own output does. At inference time we blend the raw ComplEx score with the Generator's cosine-similarity score to a candidate GO embedding via a weight gen_weight in [0,1], where 0 uses only ComplEx and 1 uses only the Generator. Sweeping this weight on both our hand-tuned (OFAT) and Optuna-tuned models shows a large, consistent improvement as gen_weight increases: gw=0.00 Fmax=0.5462 → gw=0.85 Fmax=0.7044/0.7032 → gw=1.00 Fmax=0.7190/0.7185. This result reproduces across two independently trained models, indicating the Generator has learned a transformation of the encoder's embeddings that ranks GO terms substantially better than the hand-designed ComplEx scoring formula applied to those same embeddings.

### 10.2 Section 5.3.3 — Loss weight and hyperparameter search (final drafted text)

> The Generator's training objective combines three weighted terms — an adversarial term (adv), a ComplEx-style margin-ranking term (dm), and a cosine-anchor term (anchor) — alongside training hyperparameters governing the learning dynamics of both the Generator and the Critic. We searched this space in two stages. First, a one-factor-at-a-time (OFAT) sweep over the three loss weights, with training hyperparameters held at their defaults, identified adv=0.3, dm=0.1, anchor=1.5 as the best-performing configuration (test Fmax=0.7085 at gen_weight=0.85). Second, a Bayesian search (Optuna, Tree-structured Parzen Estimator) jointly optimized all seven parameters — the three loss weights together with the Generator and Critic learning rates, the number of Critic updates per Generator step (n_critic), and the hard-negative sampling temperature — over 25 trials against a composite objective (0.5×Fmax + 0.5×MCC). This search converged on adv=0.815, dm=0.256, anchor=1.053, lr_generator=2.85×10⁻⁴, lr_critic=2.47×10⁻⁵, n_critic=2, and hard_neg_temperature=0.883, yielding test Fmax=0.7032 and MCC=0.5762 at gen_weight=0.85. Because both searches vary multiple parameters simultaneously, neither can isolate the causal contribution of any individual term; this is addressed directly through a controlled, single-variable ablation in Section 5.3.4.

### 10.3 Section 5.3.4 — Isolating the adversarial term (final drafted text)

> The searches described in 5.3.3 tune the adversarial weight jointly with several other parameters and therefore cannot, on their own, establish its individual contribution. To isolate it directly, we retrained the Generator with the adversarial weight set to zero, holding every other Optuna-tuned parameter fixed at its optimized value (dm=0.256, anchor=1.053, learning rates, n_critic, and negative-sampling temperature unchanged), and compared the resulting model against the original adversarial-on configuration across the full gen_weight sweep. Fmax proved largely insensitive to the manipulation, differing by no more than 0.013 at any blend weight and even marginally favoring the adversarial-off condition at gen_weight=0.85 (0.7053 vs. 0.7044). MCC, by contrast, showed a clear and widening gap: from a negligible difference at gen_weight=0.00 (0.3860 vs. 0.3842, as expected when the Generator contributes nothing to the score) to a substantial 0.147 deficit at gen_weight=1.00 (0.5607 vs. 0.4135), where the final prediction relies entirely on the Generator's output. Micro-F1 followed the same pattern, dropping from 0.5586 to 0.3452 under the same condition. This asymmetry indicates that the adversarial term's contribution is largely invisible to Fmax — a best-of-all-thresholds metric that tolerates a poorly calibrated score distribution — while being substantial under metrics computed at a fixed decision threshold, which directly penalize that same miscalibration. We interpret this as evidence that adversarial training regularizes the Generator's output distribution toward greater separability and consistency, rather than toward any single higher-scoring answer, and that this property becomes more consequential the more heavily the final prediction depends on the Generator rather than the base encoder score.

### 10.4 Section 4 — Conclusion (full draft, ⚠️ NEEDS REVISION before use — see Section 9 item 5)

> In this work, we introduced a GAN-based Generator-Critic architecture built on top of a CompGCN knowledge-graph encoder, formulating protein function prediction as a knowledge graph completion task. Our approach integrates a Wasserstein GAN training regime, a hyperparameter search conducted with Optuna, and a series of targeted ablations designed to isolate the specific contribution of each architectural component, rather than relying solely on aggregate performance improvements to justify design choices.
>
> Our final model achieves Fmax and Smin scores comparable to ProtHGT-ESM2 (Fmax: **[ProtHGT Fmax]** vs. **0.7032**; Smin: **[ProtHGT Smin]** vs. **159.91**, at gen_weight=0.85) on the identical 80/10/10 protein-level cold-start split. However, when evaluated holistically across classification-balance metrics (MCC, Micro-F1) and knowledge-graph-completion ranking metrics (Hit@k, MRR), our model performs substantially better (**[ProtHGT MCC/Micro-F1/Hit@k/MRR]** vs. **0.5762 / 0.5766 / [ours]**). We attribute this advantage to two properties of our architecture: first, the Generator learns to produce a single embedding per protein anchored to its true annotation, capturing the overall distribution of the annotation space rather than scoring each candidate pair independently; second, adversarial training measurably improves the consistency and separability of the Generator's output distribution, even though the Critic itself is not a usable standalone scorer.
>
> [... ablation-results paragraph, calibration-head/best-of-K/REINFORCE honest-negative-results paragraph, Limitations, Future Work — see Sections 4.2, 10.1 above for the full content these paragraphs draw from ...]
>
> Taken together, our results suggest that adversarial training's value in this setting is real but easily missed if only top-line ranking accuracy is reported — a finding we believe merits attention beyond this specific architecture, as evaluation practices in protein function prediction continue to standardize around a small set of headline metrics.

**⚠️ Note on the bracketed placeholders**: this draft was written *before* the ProtHGT full-candidate evaluation (Fmax=0.2035, Section 4.1) was completed, so `[ProtHGT Fmax]` etc. are now known: **ProtHGT Fmax=0.2035, Smin=212.1923, MCC=0.1949, Micro-F1=0.1961**. Filling these in directly would make the gap look enormous (0.70 vs 0.20) rather than "comparable" as the first paragraph's framing implies — **the first paragraph's wording ("comparable... scores") no longer matches the data and needs rewriting**, consistent with Section 9 item 5 and the suggested rewrite given earlier in this project's history: lead with the actual gap, then explain it using both the ablation evidence (legitimate architectural claim) and the calibration-mismatch caveat (Section 5.4) — not a blanket superiority claim.

### 10.5 Section 4 — Conclusion, shortened version (~300 words, also drafted, same unresolved framing issue)

> We introduced a GAN-based Generator-Critic architecture built on a CompGCN knowledge-graph encoder, framing protein function prediction as knowledge graph completion. Our final model, tuned via Optuna and evaluated through a series of isolated ablations, achieves Fmax and Smin comparable to ProtHGT-ESM2 (**[ProtHGT Fmax/Smin]** vs. **0.7032 / 159.91**) on the identical 80/10/10 split, while performing substantially better on classification-balance and ranking metrics (MCC, Micro-F1, Hit@k, MRR: **[ProtHGT values]** vs. **0.5762 / 0.5766 / [ours]**). As discussed above, this advantage traces to the Generator's learned representation of the annotation space and to the regularizing effect of adversarial training — a contribution invisible to top-line Fmax but measurable in calibration-sensitive metrics. Not every direction we explored succeeded: a calibration head and a best-of-K refinement strategy did not yield usable results, and reinforcement learning, after a diagnosed and corrected implementation issue, reached **[final RL confirmation result]** without clearly exceeding the pre-RL baseline.
>
> **Limitations.** This study is restricted to the Biological Process sub-ontology and does not include temporal-benchmark evaluation (CAFA3, DeepHGAT); our ProtHGT comparison uses their own released 80/10/10 split rather than the separately-trained model behind their most widely cited headline number. Several of our hyperparameter searches vary multiple parameters jointly and cannot isolate individual factors on their own — causal claims here rest only on single-variable ablations. Sampling-based metrics use an 8,000-protein subset rather than the full population, no multi-seed runs were conducted, and we did not apply sequence-similarity-based leakage control (e.g., UniRef90 clustering) as ProtHGT does.
>
> **Future Work.** Natural extensions include evaluating Molecular Function and Cellular Component sub-ontologies; conducting a genuine temporal-benchmark comparison following DeepHGAT's protocol; isolating the negative-sampling strategy and the reinforcement-learning hierarchy term, both currently confounded with other tuned parameters; adopting sequence-similarity-based leakage control; and repeating key comparisons across multiple seeds with formal significance testing.
>
> Overall, our results suggest that adversarial training's value in this setting is real but easily missed if only top-line ranking accuracy is reported — a finding we believe is relevant beyond this specific architecture as evaluation practices in protein function prediction continue to standardize around a small set of headline metrics.

**⚠️ Same two issues apply here as in 10.4**: (1) the "comparable" framing in paragraph 1 needs rewriting once the actual ProtHGT numbers are filled in (0.70 vs 0.20 is not "comparable"); (2) the Limitations paragraph's line "we did not apply sequence-similarity-based leakage control... as ProtHGT does" is **factually superseded by Section 5.6** — ProtHGT applies UniRef90 at dataset-construction time and this project inherits that deduplication automatically; use the corrected wording from Section 5.6 instead ("we did not independently re-verify this deduplication," not "we did not apply it").

---

## 11. Metrics Glossary

**Honesty note**: an earlier version of this project is described (in a prior conversation summary) as having produced a plain-language metrics glossary with original-paper citations. That content could not be located verbatim in either session transcript after a direct search — it may have existed only in a part of a conversation that was summarized without being preserved. Rather than guess at what it said, this is a **freshly written replacement**, built to match exactly what this repo's own code computes (`src/evaluation/metrics.py`, `src/evaluation/smin.py`).

### Fmax
The main headline metric throughout this project. At every possible decision threshold `t`, compute precision and recall per-protein, average them across all proteins, combine into an F1 score, then take the **best F1 across all thresholds**:

$$F_{\max} = \max_{t} \left\{ \frac{2 \cdot pr(t) \cdot rc(t)}{pr(t) + rc(t)} \right\}$$

**Plain language**: imagine sliding a "how confident do I need to be before I'll call this a real prediction" dial from 0 to 1, checking precision/recall at every setting, and reporting your best score. This is why a poorly-calibrated model (like ProtHGT scored against the full candidate space, Section 5.4) can still get a *reasonable* Fmax if there's *some* threshold where it does okay — but collapses hard if its scores are bunched at the extremes with no usable middle ground.
**Origin**: introduced for CAFA (Critical Assessment of Function Annotation) in Radivojac et al. 2013, *"A large-scale evaluation of computational protein function prediction,"* Nature Methods.

### Smin
A semantic-distance metric using the Gene Ontology's structure and each term's **information content** (how specific/rare a term is — "metabolic process" is less informative than "regulation of mitochondrial fission during apoptosis"). It penalizes a wrong prediction more if it's a confusing, semantically-distant guess than if it's a near-miss in the GO hierarchy.

$$S_{\min} = \min_t \sqrt{ru(t)^2 + mi(t)^2}$$

where `ru` = remaining uncertainty (true terms missed, weighted by information content) and `mi` = misinformation (wrong terms predicted, weighted by information content).
**Plain language**: Fmax treats every GO term as equally important; Smin instead penalizes you more for missing or wrongly predicting a *specific, informative* term than a vague, common one. **Lower is better** (unlike every other metric here) — it's an error measure, not a score.
**Origin**: Clark & Radivojac 2013, *"Information-theoretic evaluation of predicted ontological annotations,"* Bioinformatics (ISMB proceedings).

### MCC (Matthews Correlation Coefficient)
A single number from -1 (total disagreement) to +1 (perfect prediction), 0 = random, computed from the confusion matrix (true/false positives/negatives) **at one fixed threshold** — unlike Fmax, which gets to pick its best threshold.

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Plain language**: this is why ProtHGT's MCC (0.1949) looks so much worse relative to its AUROC (0.80) than you'd expect — MCC is computed at ONE threshold and punishes a model that hasn't found a sensible operating point, exactly the calibration problem diagnosed in Section 5.4.
**Origin**: Matthews 1975, *"Comparison of the predicted and observed secondary structure of T4 phage lysozyme,"* Biochimica et Biophysica Acta.

### Micro-F1 vs. Macro-F1
Both are F1 (precision/recall harmonic mean) at a fixed threshold, but averaged differently:
- **Micro-F1**: pool every (protein, GO-term) prediction across the whole test set into one giant confusion matrix, then compute F1 once. Dominated by common terms (they have more examples).
- **Macro-F1**: compute F1 separately for *each* GO term, then average those scores equally across all ~27,855 terms — a term with 3 examples counts exactly as much as one with 30,000.
**Plain language**: ProtHGT's Macro-F1 of 0.0031 (essentially zero) versus a less-catastrophic Micro-F1 of 0.1961 tells you it's failing on almost *every individual term*, especially rare ones, while doing comparatively less badly on the handful of very common terms that dominate the micro-averaged pool.

### AUROC (Area Under the ROC Curve)
Plots true-positive rate vs. false-positive rate across every threshold, then takes the area underneath. 0.5 = random guessing, 1.0 = perfect ranking. **Only cares about relative ranking of scores, not their absolute calibration** — this is precisely why ProtHGT can score a respectable 0.80 here while collapsing on Fmax/MCC (Section 5.4): it's still putting true pairs above false ones most of the time, it just hasn't found a sensible absolute cutoff.

### AUPR (Area Under the Precision-Recall Curve)
Like AUROC, but plots precision vs. recall instead. More sensitive than AUROC when the true positives are a tiny fraction of all possible pairs (true here: out of ~27,855 candidate GO terms per protein, only a handful are actually correct) — this is why AUPR dropped much further than AUROC for ProtHGT (0.0858 vs. 0.8007): AUPR reacts much more harshly to a flood of false positives than AUROC does.

### Hit@k and MRR (Mean Reciprocal Rank)
Borrowed from knowledge-graph-completion literature (not originally a protein-function metric): for each true (protein, GO-term) pair, rank all candidate terms by score and ask "was the true term in the top k?" (**Hit@k**, e.g. Hit@1/3/10) or "what's 1/rank of the true term, averaged across all test pairs?" (**MRR** — a true term at rank 1 contributes 1.0, at rank 10 contributes 0.1, etc.).
**Plain language**: these answer "if I handed you the model's top-k guesses, how often would the real answer be in there?" — a more forgiving, ranking-based view than Fmax's pick-one-threshold-and-grade-everything approach. **Origin** (commonly cited source for this evaluation style): Bordes et al. 2013, *"Translating Embeddings for Modeling Multi-relational Data,"* NeurIPS (the TransE paper).

### Quick reference table

| Metric | Range | Higher/lower better | Threshold-dependent? | What it's most sensitive to |
|---|---|---|---|---|
| Fmax | 0–1 | Higher | No (picks best) | Overall precision/recall balance |
| Smin | 0–∞ | **Lower** | No (picks best) | Semantic severity of errors |
| MCC | -1–1 | Higher | Yes (fixed) | Calibration at one operating point |
| Micro-F1 | 0–1 | Higher | Yes (fixed) | Performance on common terms |
| Macro-F1 | 0–1 | Higher | Yes (fixed) | Performance on rare terms |
| AUROC | 0–1 | Higher | No (integrates over all) | Relative ranking quality |
| AUPR | 0–1 | Higher | No (integrates over all) | Ranking quality under class imbalance |
| Hit@k | 0–1 | Higher | No (top-k) | "Is the answer in my shortlist?" |
| MRR | 0–1 | Higher | No (rank-based) | Average rank of the true answer |

---

## Appendix A — Full Commit Log

See `git log --all --date=short --pretty=format:"%ad | %h | %s"` for the authoritative, current list. Spans `75ff58c` (2026-06-05) through `0494c4f` (2026-10-05), ~100 commits.

## Appendix B — File Map

```
src/
  data/
    loader.py            — loads pre-built ProtHGT .pt graph splits (no accession/GO-ID mapping exposed; no construction code)
    graph_builder.py      — reads/queries an already-loaded graph (annotation matrix, leakage checks, disjointness validation)
    go_hierarchy.py        — parses GO hierarchy OUT OF existing function_function edges (does not parse raw .obo files despite obonet being a listed, unused dependency)
    negative_sampler.py    — tiered negative sampling for the Critic
  models/
    compgcn.py              — CompGCN encoder (Phase 1)
    generator.py            — GAN Generator (Phase 2/3)
    discriminator.py        — WGAN Critic, spectral-norm (Phase 2)
    distmult.py              — DistMult scoring (link prediction signal)
    reward.py                — composite RL reward module (Phase 3 REINFORCE)
    calibration_head.py      — Phase 4 score recalibration (abandoned, failed)
    shallow_kge.py           — shallow ComplEx/DistMult baselines
  training/
    pretrain.py              — Phase 1
    adversarial.py           — Phase 2 (WGAN-GP)
    rl_trainer.py             — Phase 3 (direct fine-tune + REINFORCE)
    train_calibration.py      — Phase 4
  evaluation/
    metrics.py                — Fmax, AUROC, AUPR, Micro/Macro-F1, MCC, Hit@k, MRR (8,000-protein sample except Fmax/Smin)
    smin.py                     — Smin computation
  utils/
    losses.py, schedulers.py, logger.py, seed.py
configs/
  default.yaml, full_graph.yaml   — full_graph.yaml sets gnn_edge_types=[] (use every edge type)
notebooks/
  train_colab.ipynb            — 61-cell main Colab pipeline
scripts/
  patch_notebook.py             — the only script in scripts/; a notebook-cell patcher, unrelated to data construction
```

---

*This document should be updated as the project progresses — particularly Section 9's open questions, Section 8 if the CAFA3 path is resumed, and Section 10.4 once the Conclusion is actually revised.*
