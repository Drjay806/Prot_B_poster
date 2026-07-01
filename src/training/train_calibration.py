"""
Phase 4: Train CalibrationHead on frozen Phase 1 encoder embeddings.

Why this improves Fmax:
    ComplEx scores rank correctly (AUROC 0.97) but the score distribution is
    crowded — true and false positives occupy overlapping ranges so any threshold
    cuts badly. A sigmoid MLP trained with weighted BCE on true/false annotations
    produces calibrated probabilities in (0,1) where true positives cluster near
    1.0 and negatives near 0.0, giving the Fmax threshold a clean decision surface.

Training protocol (mirrors ProtHGT's MLP head):
    - Encoder frozen (Phase 1 weights, best Fmax 0.3477)
    - Embeddings cached once — no encoder forward pass per epoch
    - Positives: actual protein-GO annotation edges from train_data
    - Negatives: neg_ratio random GO terms per positive
    - Loss: BCELoss with manual pos_weight = neg_ratio
    - Adam LR 1e-3, 50 epochs, eval on val every 5 epochs
    - Best val-Fmax checkpoint saved to CHECKPOINT_DIR/calibration_head_best.pt
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.models.compgcn import CompGCN
from src.models.calibration_head import CalibrationHead
from src.data.graph_builder import build_annotation_matrix


def train_calibration_head(
    encoder:       CompGCN,
    train_data:    HeteroData,
    val_data:      HeteroData,
    target_type:   str,
    cfg:           dict,
    device:        str  = "cuda",
    epochs:        int  = 50,
    lr:            float = 1e-3,
    neg_ratio:     int  = 100,
    batch_size:    int  = 512,
    eval_every:    int  = 5,
    checkpoint_dir: Optional[str] = None,
) -> CalibrationHead:
    """
    Train and return CalibrationHead (best val-Fmax checkpoint).
    Encoder is frozen throughout; its parameters are restored to requires_grad
    after this function returns so the rest of the pipeline is unaffected.
    """
    embed_dim = cfg["encoder"]["output_dim"]
    head = CalibrationHead(embed_dim=embed_dim).to(device)

    # Freeze encoder — cache all embeddings up front (fits in T4 VRAM easily)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print("Caching encoder embeddings (frozen) ...")
    with torch.no_grad():
        protein_embs, go_embs, _ = encoder(train_data)
    protein_embs = protein_embs.detach()
    go_embs      = go_embs.detach()

    row, col, n_p, n_go = build_annotation_matrix(train_data, target_type)
    row, col = row.to(device), col.to(device)
    n_pos = len(row)

    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)

    print(f"Training CalibrationHead: {epochs} epochs, {n_pos:,} positives, "
          f"neg_ratio={neg_ratio}, embed_dim={embed_dim}")

    best_fmax, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_pos, device=device)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n_pos, batch_size):
            b_idx  = perm[start:start + batch_size]
            b_row  = row[b_idx]
            b_col  = col[b_idx]
            B      = len(b_idx)

            p_emb = protein_embs[b_row].to(device)   # [B, D]
            g_pos = go_embs[b_col].to(device)         # [B, D]

            # Random negative GO terms
            neg_col = torch.randint(0, n_go, (B, neg_ratio), device=device)
            g_neg   = go_embs[neg_col.reshape(-1)].to(device).reshape(B * neg_ratio, -1)

            # Positive scores
            pos_prob = head(p_emb, g_pos)   # [B]

            # Negative scores — expand protein embs over neg_ratio negatives
            p_exp    = p_emb.unsqueeze(1).expand(-1, neg_ratio, -1).reshape(B * neg_ratio, -1)
            neg_prob = head(p_exp, g_neg)    # [B * neg_ratio]

            # Weighted BCE: pos_weight balances extreme class imbalance
            # Equivalent to BCEWithLogitsLoss(pos_weight=neg_ratio)
            pos_loss = -neg_ratio * torch.log(pos_prob.clamp(min=1e-7)).mean()
            neg_loss = -torch.log((1 - neg_prob).clamp(min=1e-7)).mean()
            loss     = pos_loss + neg_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)

        if epoch % eval_every == 0 or epoch == epochs:
            val_fmax = _quick_fmax_head(head, protein_embs, go_embs,
                                        val_data, target_type, device)
            marker = ""
            if val_fmax > best_fmax:
                best_fmax  = val_fmax
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
                marker     = "  *** NEW BEST"
                if checkpoint_dir:
                    torch.save(best_state,
                               os.path.join(checkpoint_dir, "calibration_head_best.pt"))
            print(f"[CalibHead {epoch:3d}/{epochs}] loss={avg_loss:.4f}  "
                  f"val_Fmax={val_fmax:.4f}{marker}")
        else:
            print(f"[CalibHead {epoch:3d}/{epochs}] loss={avg_loss:.4f}")

    # Restore best weights
    if best_state is not None:
        head.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"\nDone. Best val Fmax: {best_fmax:.4f}")

    # Restore encoder gradients
    for p in encoder.parameters():
        p.requires_grad_(True)

    return head


def _quick_fmax_head(
    head:          CalibrationHead,
    protein_embs:  torch.Tensor,
    go_embs:       torch.Tensor,
    val_data:      HeteroData,
    target_type:   str,
    device:        str,
    n_sample:      int = 2000,
    go_chunk:      int = 512,
) -> float:
    """Fast approximate Fmax for calibration head on a protein sample."""
    head.eval()

    row, col, n_p, n_go = build_annotation_matrix(val_data, target_type)
    row_cpu, col_cpu    = row.cpu(), col.cpu()

    unique_prots = row_cpu.unique()
    if len(unique_prots) > n_sample:
        unique_prots = unique_prots[torch.randperm(len(unique_prots))[:n_sample]]
    n_s = len(unique_prots)

    prot_map = {p.item(): i for i, p in enumerate(unique_prots)}
    mask     = torch.isin(row_cpu, unique_prots)
    s_row    = torch.tensor([prot_map[p.item()] for p in row_cpu[mask]], dtype=torch.long)
    s_col    = col_cpu[mask]
    true_mat = torch.zeros(n_s, n_go)
    true_mat[s_row, s_col] = 1.0

    p_embs = protein_embs[unique_prots.to(device)]   # [n_s, D]
    scores = torch.zeros(n_s, n_go)

    with torch.no_grad():
        for gs in range(0, n_go, go_chunk):
            ge = min(gs + go_chunk, n_go)
            G  = ge - gs
            g_chunk_emb = go_embs[gs:ge].to(device)   # [G, D]

            p_tile = p_embs.unsqueeze(1).expand(-1, G, -1).reshape(n_s * G, -1)
            g_tile = g_chunk_emb.unsqueeze(0).expand(n_s, -1, -1).reshape(n_s * G, -1)
            scores[:, gs:ge] = head(p_tile, g_tile).reshape(n_s, G).cpu()

    s_min, s_max = scores.min().item(), scores.max().item()
    best_f1      = 0.0
    true_pos     = true_mat.sum(dim=1).clamp(min=1e-8)

    for i in range(51):
        t    = s_min + i * (s_max - s_min) / 50
        pred = (scores >= t).float()
        tp   = (pred * true_mat).sum(dim=1)
        pp   = pred.sum(dim=1).clamp(min=1e-8)
        p_   = (tp / pp).mean().item()
        r_   = (tp / true_pos).mean().item()
        if p_ + r_ > 0:
            best_f1 = max(best_f1, 2 * p_ * r_ / (p_ + r_))

    head.train()
    return best_f1
