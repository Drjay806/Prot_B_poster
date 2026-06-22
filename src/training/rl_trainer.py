import os
from typing import Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.models.generator import Generator
from src.models.distmult import DistMult
from src.models.reward import RewardModule
from src.data.graph_builder import build_annotation_matrix
from src.training.adversarial import _get_has_function_rel_idx, _quick_fmax
from src.utils.logger import TrainingLogger


def train_rl(
    encoder: CompGCN,
    generator: Generator,
    distmult: DistMult,
    reward_module: RewardModule,
    train_data: HeteroData,
    val_data: HeteroData,
    ancestor_table: Dict[int, FrozenSet[int]],
    cfg: dict,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    logger: Optional[TrainingLogger] = None,
) -> Tuple[CompGCN, Generator]:
    """
    Phase 3: Generator fine-tuning with semantic + ancestor-similarity loss.

    Loss per batch:
        L = w_sem  * (1 - cosine(fake_g, true_g))       [semantic]
          + w_hier * (1 - cosine(fake_g, ancestor_g))    [hierarchy, avg over K sampled ancestors]

    WHY ANCESTOR-SIMILARITY REPLACES k_ancestors PENALTY:
        The old hierarchy penalty:
            1. Snapped fake_g to the nearest real GO term (non-differentiable step).
            2. Looked up that GO term's ancestors.
            3. Checked if those ancestors were in the protein's top-k cosine ranking.
        Problems:
            • k=100 was needed to avoid penalty=1.0 always — but 100 still left
              many proteins fully penalised because their ancestors are sparse.
            • The penalty was a detached constant (zero gradient on fake_g).
            • The check was about the PROTEIN's neighbourhood, not the GENERATOR OUTPUT.

        The new ancestor-similarity loss:
            1. For each (protein, true_GO) pair, sample K ancestors of true_GO.
            2. Compute cosine(fake_g, ancestor_emb) directly — fully differentiable.
            3. Penalty = 1 − mean cosine across sampled ancestors.
        Benefits:
            • Gradient flows directly through fake_g into the generator.
            • The generator learns to land near ALL ancestors, not just the true GO.
            • This encodes "if you predict a specific function, you should also be
              near the more general functions above it in the hierarchy."
            • No snapping, no constant penalty — smooth loss from epoch 1.

    Curriculum: w_sem ramps linearly from 0 → 1 over warmup_epochs.
    w_hier is always active at lambda_hier strength.
    Encoder is frozen (computed once per epoch). Only generator is updated.
    """
    rl_cfg     = cfg["rl"]
    epochs     = rl_cfg["epochs"]
    batch_size = rl_cfg["batch_size"]
    k_samples  = rl_cfg.get("num_samples", 5)
    grad_clip  = rl_cfg.get("grad_clip", 1.0)
    eval_every = rl_cfg.get("eval_every", 3)
    log_every  = rl_cfg.get("log_every_steps", 10)
    lr_gen     = rl_cfg["lr_generator"]

    warmup_epochs = cfg["reward"].get("curriculum_warmup_epochs", 30)
    lambda_hier   = cfg["reward"].get("lambda_hier", 0.5)
    anc_k         = cfg["reward"].get("ancestor_sim_k", 8)

    target_type = cfg["data"]["target_type"]

    generator.train()
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr_gen)

    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)
    row, col = row.to(device), col.to(device)

    rel_idx = _get_has_function_rel_idx(encoder)

    print(f"Starting RL training: {epochs} epochs, batch={batch_size}, k_samples={k_samples}")
    print(f"  Hierarchy loss: ancestor-similarity (K={anc_k} ancestors sampled per GO)")

    best_fmax   = 0.0
    global_step = 0

    for epoch in range(1, epochs + 1):

        # Curriculum: semantic weight ramps from 0 → 1 over warmup_epochs
        w_sem = min(epoch / max(warmup_epochs, 1), 1.0)

        # ── Encoder forward: once per epoch, no gradient ──────────────────────
        encoder.eval()
        with torch.no_grad():
            protein_embs, go_embs, rel_embs = encoder(train_data)
        protein_embs = protein_embs.detach()
        go_embs      = go_embs.detach()
        rel_vec      = rel_embs[rel_idx].detach()

        perm = torch.randperm(len(row), device=device)
        row_s, col_s = row[perm], col[perm]

        epoch_sem   = []
        epoch_hier  = []
        epoch_gnorm = []

        for start in range(0, len(row_s), batch_size):
            end        = min(start + batch_size, len(row_s))
            b_prot_idx = row_s[start:end]
            b_go_idx   = col_s[start:end]
            B          = end - start

            opt_gen.zero_grad()

            pos_p  = protein_embs[b_prot_idx]   # [B, D]
            true_g = go_embs[b_go_idx]           # [B, D]

            # Sample k candidates; take the one most similar to the true GO.
            candidates = generator.sample(pos_p, rel_vec, k=k_samples)   # [B, k, D]
            with torch.no_grad():
                true_g_exp = true_g.unsqueeze(1).expand_as(candidates)
                sem_per_k  = F.cosine_similarity(candidates, true_g_exp, dim=-1)  # [B, k]
                best_k     = sem_per_k.argmax(dim=1)
            fake_g = candidates[torch.arange(B, device=device), best_k]   # [B, D]

            # ── Semantic loss ─────────────────────────────────────────────────
            sem_loss = (1.0 - F.cosine_similarity(fake_g, true_g)).mean()

            # ── Ancestor-similarity hierarchy loss ────────────────────────────
            # For each protein, fake_g should be similar to all ancestors of true_GO.
            # Fully differentiable: cosine(fake_g, ancestor_emb) has grad through fake_g.
            hier_loss = _ancestor_sim_loss(
                fake_g, b_go_idx, go_embs, ancestor_table, anc_k, device
            )

            # Combined loss
            loss = w_sem * sem_loss + lambda_hier * hier_loss

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_gen.step()

            global_step += 1

            sem_val   = 1.0 - sem_loss.item()
            hier_val  = hier_loss.item()
            gnorm_val = gnorm if isinstance(gnorm, float) else gnorm.item()

            epoch_sem.append(sem_val)
            epoch_hier.append(hier_val)
            epoch_gnorm.append(gnorm_val)

            if logger and global_step % log_every == 0:
                logger.log({
                    "reward/semantic":          sem_val,
                    "reward/hierarchy_loss":    hier_val,
                    "grad/gen_norm":            gnorm_val,
                    "curriculum/w3":            w_sem,
                }, step=global_step, phase="rl")

        avg_sem   = sum(epoch_sem)   / max(len(epoch_sem),   1)
        avg_hier  = sum(epoch_hier)  / max(len(epoch_hier),  1)
        avg_gnorm = sum(epoch_gnorm) / max(len(epoch_gnorm), 1)

        avg = {
            "reward/total_mean":     avg_sem - lambda_hier * avg_hier,
            "reward/structural":     0.0,
            "reward/adversarial":    0.0,
            "reward/semantic":       avg_sem,
            "reward/hierarchy_loss": avg_hier,
            "grad/gen_norm":         avg_gnorm,
            "curriculum/w3":         w_sem,
        }

        is_best = False
        if epoch % eval_every == 0 or epoch == epochs:
            val_fmax = _quick_fmax(encoder, generator, distmult, rel_idx, val_data, target_type, cfg, device)
            avg["val/fmax_bp"] = val_fmax
            if logger:
                logger.log({"val/fmax_bp": val_fmax}, step=global_step, phase="rl")
            if val_fmax > best_fmax:
                best_fmax = val_fmax
                is_best   = True
                if checkpoint_dir:
                    _save_checkpoint(encoder, generator, epoch, val_fmax, checkpoint_dir)

        if logger:
            logger.log_rl_epoch(epoch, epochs, avg, is_best=is_best)
        else:
            fmax_str = f"  val_Fmax={avg['val/fmax_bp']:.4f}" if "val/fmax_bp" in avg else ""
            best_str = " *** NEW BEST" if is_best else ""
            print(
                f"[RL Epoch {epoch}/{epochs}] "
                f"sem={avg_sem:.3f}  anc_sim={1.0-avg_hier:.3f}  "
                f"w_sem={w_sem:.2f}  grad_norm={avg_gnorm:.3f}"
                f"{fmax_str}{best_str}"
            )

    print(f"RL training complete. Best val Fmax: {best_fmax:.4f}")
    return encoder, generator


def _ancestor_sim_loss(
    fake_g: torch.Tensor,
    b_go_idx: torch.Tensor,
    go_embs: torch.Tensor,
    ancestor_table: Dict[int, FrozenSet[int]],
    anc_k: int,
    device: str,
) -> torch.Tensor:
    """
    Compute 1 - mean cosine(fake_g_i, ancestor_of_true_go_i) for all proteins.

    WHY:
        The generator's output fake_g should not only be near the true GO term
        but also near that term's ancestors (parent, grandparent, etc.).
        GO hierarchy requires: if a protein has function X, it also has all
        more-general functions above X.  Pulling fake_g toward ancestors makes
        the prediction semantically consistent with the ontology even before
        the final label propagation step.

    DIFFERENTIABLE:
        cosine_similarity(fake_g, ancestor_emb) has a gradient through fake_g.
        No snapping, no constant penalty — smooth signal from epoch 1.

    Implementation: builds index tensors in a Python loop (O(batch_size)),
    then does a single vectorised cosine_similarity call on [N_total, D].
    """
    all_fake   = []   # [K_i, D] repeated fake_g for each ancestor
    all_anc    = []   # [K_i, D] ancestor embeddings

    for b in range(fake_g.size(0)):
        go_idx    = b_go_idx[b].item()
        ancestors = list(ancestor_table.get(go_idx, frozenset()))
        if not ancestors:
            continue

        k_use     = min(anc_k, len(ancestors))
        # Take a deterministic prefix rather than random sample for reproducibility
        anc_idx   = ancestors[:k_use]
        anc_emb   = go_embs[torch.tensor(anc_idx, device=device)]   # [K, D]

        all_fake.append(fake_g[b:b+1].expand(k_use, -1))  # [K, D]
        all_anc.append(anc_emb)

    if not all_fake:
        return torch.tensor(0.0, device=device)

    fake_cat = torch.cat(all_fake, dim=0)  # [total_K, D]
    anc_cat  = torch.cat(all_anc,  dim=0)  # [total_K, D]

    # 1 - cosine_similarity so loss decreases as fake_g moves toward ancestors
    return (1.0 - F.cosine_similarity(fake_cat, anc_cat, dim=-1)).mean()


def _save_checkpoint(encoder, generator, epoch: int, fmax: float, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"rl_best_epoch{epoch}_fmax{fmax:.4f}.pt")
    torch.save({
        "epoch":     epoch,
        "fmax":      fmax,
        "encoder":   encoder.state_dict(),
        "generator": generator.state_dict(),
    }, path)
    print(f"  Saved best checkpoint → {path}")
