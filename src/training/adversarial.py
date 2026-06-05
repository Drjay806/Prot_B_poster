import os
from typing import Dict, FrozenSet, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from tqdm import tqdm

from src.models.compgcn import CompGCN
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.distmult import DistMult
from src.data.graph_builder import build_annotation_matrix
from src.utils.logger import TrainingLogger


def train_adversarial(
    encoder: CompGCN,
    generator: Generator,
    discriminator: Discriminator,
    distmult: DistMult,
    train_data: HeteroData,
    val_data: HeteroData,
    ancestor_table: Dict[int, FrozenSet[int]],
    cfg: dict,
    device: str = "cuda",
    logger: Optional[TrainingLogger] = None,
) -> Tuple[CompGCN, Generator, Discriminator]:
    """
    Phase 2: Adversarial training of generator vs discriminator.
    Discriminator is updated 2x per generator update.
    Returns updated (encoder, generator, discriminator).
    """
    adv_cfg = cfg["adversarial"]
    epochs = adv_cfg["epochs"]
    batch_size = adv_cfg["batch_size"]
    d_steps = adv_cfg.get("discriminator_steps", 2)
    grad_clip = adv_cfg.get("grad_clip", 1.0)
    eval_every = adv_cfg.get("eval_every", 5)
    log_every = adv_cfg.get("log_every_steps", 25)

    target_type = cfg["data"]["target_type"]

    encoder.set_dropout("adversarial")
    encoder.train(); generator.train(); discriminator.train()

    # Separate optimizers per component
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=adv_cfg["lr_encoder"], betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(generator.parameters(), lr=adv_cfg["lr_generator"], betas=(0.5, 0.999))
    opt_dis = torch.optim.Adam(discriminator.parameters(), lr=adv_cfg["lr_discriminator"], betas=(0.5, 0.999))

    scaler = torch.amp.GradScaler('cuda')

    # Precompute positive edges for sampling
    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)
    row, col = row.to(device), col.to(device)

    # Identify the has_function relation index for DistMult scoring
    rel_idx = _get_has_function_rel_idx(encoder)

    print(f"Starting adversarial training: {epochs} epochs, {len(row):,} positive pairs")

    global_step = 0
    best_fmax = 0.0

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(row), device=device)
        row_s, col_s = row[perm], col[perm]

        epoch_metrics: Dict[str, list] = {
            k: [] for k in ["loss/disc_real", "loss/disc_fake", "loss/disc_total",
                             "loss/gen", "reward/distmult_mean",
                             "disc/real_acc", "disc/fake_acc", "embed/gen_output_norm"]
        }

        # Encoder called ONCE per epoch — same strategy as pretrain.
        # Encoder is frozen during adversarial phase; fine-tuned in RL instead.
        encoder.eval()
        with torch.no_grad():
            protein_embs, go_embs, rel_embs = encoder(train_data)
        encoder.train()
        protein_embs = protein_embs.detach()
        go_embs      = go_embs.detach()
        rel_vec      = rel_embs[rel_idx].detach()

        for start in range(0, len(row_s), batch_size):
            end = min(start + batch_size, len(row_s))
            b_prot_idx = row_s[start:end]
            b_go_idx   = col_s[start:end]

            pos_p = protein_embs[b_prot_idx]
            pos_g = go_embs[b_go_idx]

            # ── Discriminator updates (d_steps times) ──────────────────────────────
            for _ in range(d_steps):
                opt_dis.zero_grad()
                with torch.amp.autocast('cuda'):
                    noise = torch.randn(len(b_prot_idx), generator.noise_dim, device=device)
                    fake_g = generator(pos_p, rel_vec, noise)
                    loss_real = discriminator.loss_real(pos_p, pos_g)
                    loss_fake = discriminator.loss_fake(pos_p, fake_g.detach())
                    loss_d = loss_real + loss_fake

                scaler.scale(loss_d).backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                scaler.step(opt_dis)
                scaler.update()

            # ── Generator update (1 time) ───────────────────────────────────────────
            opt_gen.zero_grad()
            with torch.amp.autocast('cuda'):
                noise = torch.randn(len(b_prot_idx), generator.noise_dim, device=device)
                fake_g = generator(pos_p, rel_vec, noise)

                loss_adv = discriminator.adversarial_loss_for_generator(pos_p, fake_g)

                dm_scores = distmult(pos_p, rel_vec.unsqueeze(0).expand_as(pos_p), fake_g)
                loss_struct = -dm_scores.mean()

                loss_g = loss_adv + 0.5 * loss_struct

            scaler.scale(loss_g).backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            scaler.step(opt_gen)
            scaler.update()

            global_step += 1

            # Monitoring
            with torch.no_grad():
                ra = discriminator.accuracy(pos_p.detach(), pos_g.detach(), is_real=True)
                fa = discriminator.accuracy(pos_p.detach(), fake_g.detach(), is_real=False)
                gen_norm = fake_g.detach().norm(dim=-1).mean().item()
                dm_mean = dm_scores.detach().mean().item()

            step_metrics = {
                "loss/disc_real": loss_real.item(),
                "loss/disc_fake": loss_fake.item(),
                "loss/disc_total": (loss_real + loss_fake).item(),
                "loss/gen": loss_adv.item(),
                "reward/distmult_mean": dm_mean,
                "disc/real_acc": ra,
                "disc/fake_acc": fa,
                "embed/gen_output_norm": gen_norm,
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
            d = avg["loss/disc_total"]
            g = avg["loss/gen"]
            fmax_str = f"  val_Fmax={avg.get('val/fmax_bp', 0):.4f}" if "val/fmax_bp" in avg else ""
            print(f"[Adv {epoch}/{epochs}] D={d:.3f} G={g:.3f}{fmax_str}")

    return encoder, generator, discriminator


def _get_has_function_rel_idx(encoder: CompGCN) -> int:
    """Find the relation index for 'has_function' (or the first protein→GO relation)."""
    for rel, idx in encoder.rel_name_to_idx.items():
        if "function" in rel.lower() or "annotation" in rel.lower():
            return idx
    # Fallback: return index 0
    return 0


@torch.no_grad()
def _quick_fmax(
    encoder: CompGCN,
    generator: Generator,
    distmult: DistMult,
    rel_idx: int,
    val_data: HeteroData,
    target_type: str,
    cfg: dict,
    device: str,
    n_sample: int = 5000,
) -> float:
    """
    Fast approximate Fmax on a random sample of val proteins.
    Avoids materialising a [N_p x N_go] dense matrix (would be ~26 GB).
    """
    from src.data.graph_builder import build_annotation_matrix
    encoder.eval(); generator.eval()

    # Encoder in float32 (no autocast) — same reason as training path
    protein_embs, go_embs, rel_embs = encoder(val_data)
    rel_vec = rel_embs[rel_idx]

    row, col, n_p, n_go = build_annotation_matrix(val_data, target_type)
    row_cpu, col_cpu = row.cpu(), col.cpu()

    # Sample up to n_sample proteins that actually have annotations
    unique_prots = row_cpu.unique()
    if len(unique_prots) > n_sample:
        perm = torch.randperm(len(unique_prots))[:n_sample]
        unique_prots = unique_prots[perm]

    # Remap to contiguous indices
    prot_map = {p.item(): i for i, p in enumerate(unique_prots)}
    mask = torch.isin(row_cpu, unique_prots)
    s_row = torch.tensor([prot_map[p.item()] for p in row_cpu[mask]], dtype=torch.long)
    s_col = col_cpu[mask]

    n_sample_actual = len(unique_prots)
    true_mat = torch.zeros(n_sample_actual, n_go, dtype=torch.float32)
    true_mat[s_row, s_col] = 1.0

    # Score only the sampled proteins — [n_sample, N_go], safe on T4
    sampled_embs = protein_embs[unique_prots.to(device)]
    scores = torch.sigmoid(distmult.score_all(sampled_embs, rel_vec, go_embs)).cpu()

    # Single-threshold F1 at 0.5 as a fast proxy (good enough for early-stop signal)
    pred = (scores >= 0.5).float()
    tp = (pred * true_mat).sum(dim=1)
    pred_pos = pred.sum(dim=1).clamp(min=1e-8)
    true_pos = true_mat.sum(dim=1).clamp(min=1e-8)
    prec = (tp / pred_pos).mean()
    rec  = (tp / true_pos).mean()
    fmax = (2 * prec * rec / (prec + rec + 1e-8)).item()

    encoder.train(); generator.train()
    return fmax
