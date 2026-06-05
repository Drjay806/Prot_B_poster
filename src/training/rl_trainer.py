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
    Phase 3: REINFORCE-based RL fine-tuning with GO hierarchy constraint reward.
    Returns updated (encoder, generator).
    """
    rl_cfg = cfg["rl"]
    epochs = rl_cfg["epochs"]
    batch_size = rl_cfg["batch_size"]
    k_samples = rl_cfg.get("num_samples", 5)
    grad_clip = rl_cfg.get("grad_clip", 1.0)
    eval_every = rl_cfg.get("eval_every", 3)
    log_every = rl_cfg.get("log_every_steps", 10)
    lr_gen = rl_cfg["lr_generator"]
    lr_enc = rl_cfg["lr_encoder"]

    target_type = cfg["data"]["target_type"]

    encoder.set_dropout("adversarial")
    encoder.train(); generator.train()

    opt_gen = torch.optim.Adam(generator.parameters(), lr=lr_gen)
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=lr_enc)

    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)
    row, col = row.to(device), col.to(device)

    rel_idx = _get_has_function_rel_idx(encoder)

    print(f"Starting RL training: {epochs} epochs, batch={batch_size}, k_samples={k_samples}")

    best_fmax = 0.0
    global_step = 0

    for epoch in range(1, epochs + 1):
        reward_module.set_epoch(epoch)
        _, _, w3 = reward_module.get_weights()
        curriculum_w3 = reward_module._curriculum_multiplier

        perm = torch.randperm(len(row), device=device)
        row_s, col_s = row[perm], col[perm]

        epoch_metrics: Dict[str, list] = {k: [] for k in [
            "reward/total_mean", "reward/structural", "reward/adversarial",
            "reward/semantic", "reward/hierarchy_penalty",
            "grad/gen_norm",
        ]}

        for start in range(0, len(row_s), batch_size):
            end = min(start + batch_size, len(row_s))
            b_prot_idx = row_s[start:end]
            b_go_idx = col_s[start:end]

            opt_gen.zero_grad()
            opt_enc.zero_grad()

            # Encoder in float32 (no autocast) — avoids LayerNorm/FFT dtype mismatches
            protein_embs, go_embs, rel_embs = encoder(train_data)
            rel_vec = rel_embs[rel_idx]

            pos_p = protein_embs[b_prot_idx]   # [B, D]
            true_g = go_embs[b_go_idx]          # [B, D]

            # Sample k candidates per protein
            candidates = generator.sample(pos_p, rel_vec, k=k_samples)   # [B, k, D]

            # Score each candidate with DistMult: [B, k]
            r_vec = rel_vec.unsqueeze(0).expand(len(b_prot_idx), -1)  # [B, D]
            dm_scores = distmult(
                pos_p.unsqueeze(1).expand_as(candidates),
                r_vec.unsqueeze(1).expand_as(candidates),
                candidates,
            )   # [B, k]

            # Select top-1 candidate per protein
            best_k = dm_scores.argmax(dim=1)               # [B]
            fake_g = candidates[torch.arange(len(b_prot_idx)), best_k]   # [B, D]

            # Get predicted GO term index by scoring fake_g against all GO embs
            with torch.no_grad():
                score_mat = distmult.score_all(pos_p.detach(), rel_vec.detach(), go_embs.detach())
                predicted_go_idxs = score_mat.argmax(dim=1)

            # Compute composite reward
            reward, reward_breakdown = reward_module.compute(
                pos_p, rel_vec, fake_g, true_g,
                go_embs.detach(), predicted_go_idxs, ancestor_table,
            )   # [B]

            # REINFORCE: log-prob from softmax over k DistMult scores
            log_probs = F.log_softmax(dm_scores, dim=1)   # [B, k]
            selected_log_probs = log_probs[torch.arange(len(b_prot_idx)), best_k]   # [B]

            # Baseline: mean reward across batch (reduces variance)
            baseline = reward.detach().mean()
            advantage = reward.detach() - baseline

            loss = -(advantage * selected_log_probs).mean()

            loss.backward()
            gen_grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
            opt_gen.step()
            opt_enc.step()

            global_step += 1

            step_m = {
                "reward/total_mean": reward.detach().mean().item(),
                "grad/gen_norm": gen_grad_norm if isinstance(gen_grad_norm, float) else gen_grad_norm.item(),
                **{k: v for k, v in reward_breakdown.items() if k.startswith("reward/")},
            }
            for k, v in step_m.items():
                epoch_metrics.setdefault(k, []).append(v)

            if logger and global_step % log_every == 0:
                metrics_to_log = {**step_m, "curriculum/w3": curriculum_w3 * w3.item()}
                metrics_to_log.update({k: v for k, v in reward_breakdown.items() if k.startswith("curriculum/")})
                logger.log(metrics_to_log, step=global_step, phase="rl")
                logger.check_health(global_step)

        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_metrics.items()}
        avg["curriculum/w3"] = curriculum_w3

        is_best = False
        if epoch % eval_every == 0 or epoch == epochs:
            val_fmax = _quick_fmax(encoder, generator, distmult, rel_idx, val_data, target_type, cfg, device)
            avg["val/fmax_bp"] = val_fmax
            if logger:
                logger.log({"val/fmax_bp": val_fmax}, step=global_step, phase="rl")
            if val_fmax > best_fmax:
                best_fmax = val_fmax
                is_best = True
                if checkpoint_dir:
                    _save_checkpoint(encoder, generator, epoch, val_fmax, checkpoint_dir)

        if logger:
            logger.log_rl_epoch(epoch, epochs, avg, is_best=is_best)
        else:
            r = avg["reward/total_mean"]
            fmax_str = f"  val_Fmax={avg.get('val/fmax_bp', 0):.4f}" if "val/fmax_bp" in avg else ""
            best_str = " *** NEW BEST" if is_best else ""
            print(f"[RL {epoch}/{epochs}] R={r:.3f} w3={curriculum_w3:.3f}{fmax_str}{best_str}")

    print(f"RL training complete. Best val Fmax: {best_fmax:.4f}")
    return encoder, generator


def _save_checkpoint(encoder, generator, epoch: int, fmax: float, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"rl_best_epoch{epoch}_fmax{fmax:.4f}.pt")
    torch.save({
        "epoch": epoch,
        "fmax": fmax,
        "encoder": encoder.state_dict(),
        "generator": generator.state_dict(),
    }, path)
    print(f"  Saved best checkpoint → {path}")
