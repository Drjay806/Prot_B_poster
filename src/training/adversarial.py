import os
from typing import Dict, FrozenSet, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.models.generator import Generator
from src.models.discriminator import Discriminator   # = Critic alias
from src.models.distmult import DistMult
from src.data.graph_builder import build_annotation_matrix
from src.data.negative_sampler import NegativeSampler
from src.utils.logger import TrainingLogger


# ──────────────────────────────────────────────────────────────────────────────
# Gradient penalty
# ──────────────────────────────────────────────────────────────────────────────

def _gradient_penalty(
    critic:      nn.Module,
    real_p:      Tensor,   # [B, D_prot]
    real_go:     Tensor,   # [B, D_go]
    fake_go:     Tensor,   # [B, D_go]
    lambda_gp:   float,
    device:      str,
) -> Tensor:
    """
    WGAN-GP gradient penalty (Gulrajani et al., NeurIPS 2017).
    Interpolates between real and generated GO embeddings, penalises
    ||∇_x̂ D(protein, x̂)||₂ deviating from 1.
    """
    B = real_go.size(0)
    alpha = torch.rand(B, 1, device=device)
    x_hat = (alpha * real_go + (1 - alpha) * fake_go.detach()).requires_grad_(True)

    score = critic(real_p, x_hat)   # [B, 1]
    ones  = torch.ones_like(score)
    grads = torch.autograd.grad(
        outputs=score, inputs=x_hat,
        grad_outputs=ones,
        create_graph=True, retain_graph=True,
    )[0]                             # [B, D_go]

    gp = lambda_gp * ((grads.norm(2, dim=-1) - 1) ** 2).mean()
    return gp


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train_adversarial(
    encoder:       CompGCN,
    generator:     Generator,
    discriminator: Discriminator,     # Critic instance
    distmult:      DistMult,
    train_data:    HeteroData,
    val_data:      HeteroData,
    ancestor_table: Dict[int, FrozenSet[int]],
    cfg:           dict,
    device:        str = "cuda",
    logger:        Optional[TrainingLogger] = None,
) -> Tuple[CompGCN, Generator, Discriminator]:
    """
    Phase 2: WGAN-GP adversarial training.

    Critic is updated n_critic times per generator update.
    Three-way critic loss:
        real annotated pairs (high score)
        generator fake pairs  (medium)
        hard vocabulary negatives (low)

    No mixed-precision (autocast removed) — encoder produces float32 throughout.
    Returns updated (encoder, generator, discriminator/critic).
    """
    adv_cfg    = cfg["adversarial"]
    epochs     = adv_cfg["epochs"]
    batch_size = adv_cfg["batch_size"]
    n_critic   = adv_cfg.get("n_critic", 5)
    lambda_gp  = adv_cfg.get("lambda_gp", 10.0)
    grad_clip  = adv_cfg.get("grad_clip", 1.0)
    eval_every = adv_cfg.get("eval_every", 5)
    log_every  = adv_cfg.get("log_every_steps", 25)
    beta1      = adv_cfg.get("adam_beta1", 0.0)
    beta2      = adv_cfg.get("adam_beta2", 0.9)
    hard_neg_temperature = adv_cfg.get("negative_sampling", {}).get("hard_neg_temperature", 0.5)

    target_type = cfg["data"]["target_type"]

    encoder.set_dropout("adversarial")
    encoder.train(); generator.train(); discriminator.train()

    # WGAN-GP requires Adam(β1=0, β2=0.9) — other betas destabilise Wasserstein
    opt_enc  = torch.optim.Adam(encoder.parameters(),       lr=adv_cfg["lr_encoder"],     betas=(beta1, beta2))
    opt_gen  = torch.optim.Adam(generator.parameters(),     lr=adv_cfg["lr_generator"],   betas=(beta1, beta2))
    opt_crit = torch.optim.Adam(discriminator.parameters(), lr=adv_cfg.get("lr_critic", adv_cfg["lr_discriminator"]), betas=(beta1, beta2))

    # Precompute positive edges for sampling
    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)
    row_cpu = row.cpu()
    col_cpu = col.cpu()
    row, col = row.to(device), col.to(device)

    # Identify the has_function relation index for DistMult scoring
    rel_idx = _get_has_function_rel_idx(encoder)

    # Build tiered negative sampler once — pools are computed from GO depth
    neg_sampler = NegativeSampler(
        n_go=n_go,
        protein_indices=row_cpu,
        go_indices=col_cpu,
        ancestor_table=ancestor_table,
    )

    print(f"Starting adversarial training: {epochs} epochs, {len(row):,} positive pairs")
    print(f"  WGAN-GP | n_critic={n_critic} | lambda_gp={lambda_gp} | beta1={beta1} beta2={beta2}")

    global_step = 0
    best_fmax   = 0.0

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(row), device=device)
        row_s, col_s   = row[perm], col[perm]
        row_s_cpu      = row_s.cpu()

        epoch_metrics: Dict[str, list] = {k: [] for k in [
            "loss/critic", "loss/gen", "loss/gp",
            "reward/distmult_mean",
            "scores/real", "scores/fake", "scores/hard",
            "disc/real_acc", "disc/fake_acc",
        ]}

        # Encoder called ONCE per epoch — frozen during adversarial phase.
        encoder.eval()
        with torch.no_grad():
            protein_embs, go_embs, rel_embs = encoder(train_data)
        encoder.train()
        protein_embs = protein_embs.detach()
        go_embs      = go_embs.detach()
        rel_vec      = rel_embs[rel_idx].detach()

        # Curriculum negative sampling ratios
        ratios = NegativeSampler.curriculum_ratios(epoch, epochs)

        for start in range(0, len(row_s), batch_size):
            end = min(start + batch_size, len(row_s))
            b_prot_idx     = row_s[start:end]
            b_go_idx       = col_s[start:end]
            b_prot_idx_cpu = row_s_cpu[start:end]

            pos_p = protein_embs[b_prot_idx]   # [B, D]
            pos_g = go_embs[b_go_idx]           # [B, D]

            # Sample hard vocabulary negatives (tiered, self-adversarial)
            neg_go_idx = neg_sampler.sample(
                b_prot_idx=b_prot_idx_cpu,
                go_embs=go_embs,
                protein_embs=protein_embs,
                rel_vec=rel_vec,
                device=device,
                ratios=ratios,
                temperature=hard_neg_temperature,
            )
            hard_g = go_embs[neg_go_idx]       # [B, D]

            # ── Critic updates (n_critic times) ──────────────────────────────
            for _ in range(n_critic):
                opt_crit.zero_grad()

                noise  = torch.randn(len(b_prot_idx), generator.noise_dim, device=device)
                fake_g = generator(pos_p, rel_vec, noise).detach()

                score_real = discriminator.score(pos_p, pos_g)     # [B]
                score_fake = discriminator.score(pos_p, fake_g)    # [B]
                score_hard = discriminator.score(pos_p, hard_g)    # [B]

                # Three-way Wasserstein loss: maximise real, minimise fake + hard
                w_dist   = score_real.mean() - 0.5 * score_fake.mean() - 0.5 * score_hard.mean()
                gp       = _gradient_penalty(discriminator, pos_p, pos_g, fake_g, lambda_gp, device)
                loss_crit = -w_dist + gp

                loss_crit.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                opt_crit.step()

            # ── Generator update (1 time per n_critic critic steps) ───────────
            opt_gen.zero_grad()
            noise  = torch.randn(len(b_prot_idx), generator.noise_dim, device=device)
            fake_g = generator(pos_p, rel_vec, noise)

            # Maximise critic score on generated pairs + DistMult structural reward
            dm_scores = distmult(pos_p, rel_vec.unsqueeze(0).expand_as(pos_p), fake_g)
            loss_gen  = -discriminator.score(pos_p, fake_g).mean() - 0.5 * dm_scores.mean()

            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            opt_gen.step()

            global_step += 1

            # Monitoring (no_grad)
            with torch.no_grad():
                ra = discriminator.accuracy(pos_p, pos_g,   is_real=True)
                fa = discriminator.accuracy(pos_p, fake_g,  is_real=False)

            step_metrics = {
                "loss/critic":          loss_crit.item(),
                "loss/gen":             loss_gen.item(),
                "loss/gp":              gp.item(),
                "reward/distmult_mean": dm_scores.detach().mean().item(),
                "scores/real":          score_real.detach().mean().item(),
                "scores/fake":          score_fake.detach().mean().item(),
                "scores/hard":          score_hard.detach().mean().item(),
                "disc/real_acc":        ra,
                "disc/fake_acc":        fa,
            }
            for k, v in step_metrics.items():
                epoch_metrics[k].append(v)

            if logger and global_step % log_every == 0:
                logger.log(step_metrics, step=global_step, phase="adversarial")
                logger.check_health(global_step)

        # Epoch-level averages
        avg = {k: sum(v) / max(len(v), 1) for k, v in epoch_metrics.items()}

        if epoch % eval_every == 0 or epoch == epochs:
            val_fmax = _quick_fmax(encoder, generator, distmult, rel_idx, val_data, target_type, cfg, device)
            avg["val/fmax_bp"] = val_fmax
            if logger:
                logger.log({"val/fmax_bp": val_fmax}, step=global_step, phase="adversarial")
            if val_fmax > best_fmax:
                best_fmax = val_fmax

        if logger:
            logger.log_adv_epoch(epoch, epochs, avg)
        else:
            w_dist_approx = avg["scores/real"] - 0.5 * avg["scores/fake"] - 0.5 * avg["scores/hard"]
            fmax_str      = f"  val_Fmax={avg['val/fmax_bp']:.4f}" if "val/fmax_bp" in avg else ""
            print(
                f"[Adv {epoch}/{epochs}] "
                f"W_dist={w_dist_approx:.3f}  G_loss={avg['loss/gen']:.3f}  "
                f"scores(real={avg['scores/real']:.2f} fake={avg['scores/fake']:.2f} hard={avg['scores/hard']:.2f})  "
                f"DistMult={avg['reward/distmult_mean']:.3f}  "
                f"acc(real={avg['disc/real_acc']*100:.0f}% fake={avg['disc/fake_acc']*100:.0f}%)"
                f"{fmax_str}"
            )

    return encoder, generator, discriminator


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def _get_has_function_rel_idx(encoder: CompGCN) -> int:
    """Find the relation index for 'has_function' (or the first protein→GO relation)."""
    for rel, idx in encoder.rel_name_to_idx.items():
        if "function" in rel.lower() or "annotation" in rel.lower():
            return idx
    return 0


@torch.no_grad()
def _quick_fmax(
    encoder:     CompGCN,
    generator:   Generator,
    distmult:    DistMult,
    rel_idx:     int,
    val_data:    HeteroData,
    target_type: str,
    cfg:         dict,
    device:      str,
    n_sample:    int = 5000,
) -> float:
    """
    Fast approximate Fmax on a random sample of val proteins.
    Avoids materialising a [N_p x N_go] dense matrix (would be ~26 GB).
    """
    from src.data.graph_builder import build_annotation_matrix
    encoder.eval(); generator.eval()

    protein_embs, go_embs, rel_embs = encoder(val_data)
    rel_vec = rel_embs[rel_idx]

    row, col, n_p, n_go = build_annotation_matrix(val_data, target_type)
    row_cpu, col_cpu = row.cpu(), col.cpu()

    unique_prots = row_cpu.unique()
    if len(unique_prots) > n_sample:
        perm = torch.randperm(len(unique_prots))[:n_sample]
        unique_prots = unique_prots[perm]

    prot_map = {p.item(): i for i, p in enumerate(unique_prots)}
    mask  = torch.isin(row_cpu, unique_prots)
    s_row = torch.tensor([prot_map[p.item()] for p in row_cpu[mask]], dtype=torch.long)
    s_col = col_cpu[mask]

    n_sample_actual = len(unique_prots)
    true_mat = torch.zeros(n_sample_actual, n_go, dtype=torch.float32)
    true_mat[s_row, s_col] = 1.0

    sampled_embs = protein_embs[unique_prots.to(device)]
    scores = torch.sigmoid(distmult.score_all(sampled_embs, rel_vec, go_embs)).cpu()

    pred    = (scores >= 0.5).float()
    tp      = (pred * true_mat).sum(dim=1)
    pred_pos = pred.sum(dim=1).clamp(min=1e-8)
    true_pos = true_mat.sum(dim=1).clamp(min=1e-8)
    prec    = (tp / pred_pos).mean()
    rec     = (tp / true_pos).mean()
    fmax    = (2 * prec * rec / (prec + rec + 1e-8)).item()

    encoder.train(); generator.train()
    return fmax
