import os
from typing import Dict, FrozenSet, Optional, Tuple

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
    Phase 3: Generator fine-tuning with semantic + hierarchy-aware reward.

    Replaces REINFORCE (which required a trained DistMult policy) with direct
    gradient descent on a composite loss:
        L = w_sem * (1 - cosine(fake_g, true_g))
          + w_hier * hierarchy_penalty
          + w_div  * -cosine_variance   (diversity across k samples)

    Curriculum linearly increases w_sem from 0 → 1 over warmup_epochs so the
    generator first stabilises near the GO embedding manifold before being pushed
    toward specific annotations.

    Encoder is frozen (computed once per epoch); only generator is updated.
    Returns updated (encoder, generator).
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
    k_anc         = cfg["reward"].get("k_ancestors", 20)

    target_type = cfg["data"]["target_type"]

    generator.train()
    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr_gen)

    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)
    row, col = row.to(device), col.to(device)

    rel_idx = _get_has_function_rel_idx(encoder)

    print(f"Starting RL training: {epochs} epochs, batch={batch_size}, k_samples={k_samples}")

    best_fmax   = 0.0
    global_step = 0

    for epoch in range(1, epochs + 1):

        # Curriculum weight: sem contribution ramps from 0 → 1 over warmup_epochs
        w_sem = min(epoch / max(warmup_epochs, 1), 1.0)

        # ── Encoder forward: ONCE per epoch, no gradient ──────────────────────
        encoder.eval()
        with torch.no_grad():
            protein_embs, go_embs, rel_embs = encoder(train_data)
        protein_embs = protein_embs.detach()
        go_embs      = go_embs.detach()
        rel_vec      = rel_embs[rel_idx].detach()

        # Pre-normalise GO embeddings once for hierarchy penalty scoring
        go_norms = F.normalize(go_embs, dim=-1)   # [N_go, D]

        perm = torch.randperm(len(row), device=device)
        row_s, col_s = row[perm], col[perm]

        epoch_sem  = []
        epoch_hier = []
        epoch_gnorm = []

        for start in range(0, len(row_s), batch_size):
            end        = min(start + batch_size, len(row_s))
            b_prot_idx = row_s[start:end]
            b_go_idx   = col_s[start:end]
            B          = end - start

            opt_gen.zero_grad()

            pos_p  = protein_embs[b_prot_idx]   # [B, D]
            true_g = go_embs[b_go_idx]           # [B, D]

            # Sample k candidates and take the one most similar to true GO.
            # Using semantic similarity (not DistMult) to select best candidate
            # avoids a random policy corrupting the gradient direction.
            candidates = generator.sample(pos_p, rel_vec, k=k_samples)   # [B, k, D]
            with torch.no_grad():
                true_g_exp = true_g.unsqueeze(1).expand_as(candidates)    # [B, k, D]
                sem_per_k  = F.cosine_similarity(candidates, true_g_exp, dim=-1)  # [B, k]
                best_k     = sem_per_k.argmax(dim=1)                      # [B]
            fake_g = candidates[torch.arange(B, device=device), best_k]   # [B, D]

            # ── Semantic loss: push generator toward true GO direction ─────────
            sem_loss = (1.0 - F.cosine_similarity(fake_g, true_g)).mean()

            # ── Hierarchy penalty: fake_g's nearest GO term should have its
            #    ancestors covered in the protein's top-k cosine neighbourhood ──
            with torch.no_grad():
                fake_norm         = F.normalize(fake_g.detach(), dim=-1)
                cos_to_vocab      = fake_norm @ go_norms.t()               # [B, N_go]
                predicted_go_idxs = cos_to_vocab.argmax(dim=1)            # [B]

                # Top-k GO terms per protein by cosine (encoder already trained)
                p_norm    = F.normalize(pos_p, dim=-1)
                top_k_idx = (p_norm @ go_norms.t()).topk(k_anc, dim=-1).indices  # [B, k]

            hier_penalties = []
            for b in range(B):
                go_idx    = predicted_go_idxs[b].item()
                ancestors = ancestor_table.get(go_idx, frozenset())
                if not ancestors:
                    hier_penalties.append(0.0)
                    continue
                top_k_set = set(top_k_idx[b].tolist())
                missing   = len(ancestors - top_k_set) / len(ancestors)
                hier_penalties.append(missing)
            hier_penalty = torch.tensor(hier_penalties, device=device).mean()

            # Combined loss — sem weight ramps up with curriculum
            loss = w_sem * sem_loss + lambda_hier * hier_penalty

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_gen.step()

            global_step += 1

            sem_val  = 1.0 - sem_loss.item()   # cosine similarity (higher = better)
            hier_val = hier_penalty.item()
            gnorm_val = gnorm if isinstance(gnorm, float) else gnorm.item()

            epoch_sem.append(sem_val)
            epoch_hier.append(hier_val)
            epoch_gnorm.append(gnorm_val)

            if logger and global_step % log_every == 0:
                logger.log({
                    "reward/semantic":          sem_val,
                    "reward/hierarchy_penalty": hier_val,
                    "grad/gen_norm":            gnorm_val,
                    "curriculum/w3":            w_sem,
                }, step=global_step, phase="rl")

        avg_sem  = sum(epoch_sem)  / max(len(epoch_sem),  1)
        avg_hier = sum(epoch_hier) / max(len(epoch_hier), 1)
        avg_gnorm = sum(epoch_gnorm) / max(len(epoch_gnorm), 1)

        avg = {
            "reward/total_mean":        avg_sem - lambda_hier * avg_hier,
            "reward/structural":        0.0,
            "reward/adversarial":       0.0,
            "reward/semantic":          avg_sem,
            "reward/hierarchy_penalty": avg_hier,
            "grad/gen_norm":            avg_gnorm,
            "curriculum/w3":            w_sem,
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
                f"sem={avg_sem:.3f}  hier_pen={avg_hier:.3f}  "
                f"w_sem={w_sem:.2f}  grad_norm={avg_gnorm:.3f}"
                f"{fmax_str}{best_str}"
            )

    print(f"RL training complete. Best val Fmax: {best_fmax:.4f}")
    return encoder, generator


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
