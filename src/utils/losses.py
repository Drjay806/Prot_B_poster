import torch
import torch.nn.functional as F


def infonce_loss(protein_embs: torch.Tensor, go_embs: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss (NT-Xent, SimCLR — Chen et al. 2020).

    For a batch of B (protein, GO) pairs, every other GO term in the batch
    becomes a negative for each protein.  This gives B-1 negatives per sample
    automatically — 4,095 negatives at batch=4096, vs 5 in the old ranking loss.

    The model must learn to rank the true GO term above ALL other GO terms
    seen in the batch, which is exactly what Fmax measures at evaluation time.

    Temperature 0.07 is the SimCLR default and works well for unit-sphere geometry.
    """
    p_norm = F.normalize(protein_embs, dim=-1)   # [B, D]
    g_norm = F.normalize(go_embs, dim=-1)         # [B, D]
    sim = (p_norm @ g_norm.t()) / temperature      # [B, B]  — scaled cosine matrix
    labels = torch.arange(sim.size(0), device=sim.device)
    # Protein i must rank GO i highest across all GO terms in the batch
    return F.cross_entropy(sim, labels)


def ranking_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Margin-based ranking loss: pushes positive scores above negatives by `margin`."""
    return F.relu(margin - pos_score + neg_score).mean()


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity between pred and target."""
    return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()


def manifold_alignment_loss(protein_embs: torch.Tensor, go_embs: torch.Tensor, num_kernels: int = 5) -> torch.Tensor:
    """Maximum Mean Discrepancy (MMD) with RBF kernels to align protein and GO manifolds."""
    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # [n, m, d]
        dist_sq = (diff ** 2).sum(dim=-1)                # [n, m]
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    # Subsample to at most 512 per side to keep cost manageable
    n = min(protein_embs.size(0), 512)
    m = min(go_embs.size(0), 512)
    px = protein_embs[torch.randperm(protein_embs.size(0), device=protein_embs.device)[:n]]
    gy = go_embs[torch.randperm(go_embs.size(0), device=go_embs.device)[:m]]

    sigmas = [0.1, 0.5, 1.0, 5.0, 10.0][:num_kernels]
    mmd = torch.tensor(0.0, device=protein_embs.device)
    for sigma in sigmas:
        kxx = rbf_kernel(px, px, sigma).mean()
        kyy = rbf_kernel(gy, gy, sigma).mean()
        kxy = rbf_kernel(px, gy, sigma).mean()
        mmd = mmd + kxx + kyy - 2.0 * kxy
    return mmd / len(sigmas)


def label_smooth_bce(logit: torch.Tensor, is_real: bool, smooth_real: float = 0.9, smooth_fake: float = 0.1) -> torch.Tensor:
    """BCE loss with label smoothing. is_real=True uses smooth_real target, else smooth_fake."""
    target_val = smooth_real if is_real else smooth_fake
    target = torch.full_like(logit, target_val)
    return F.binary_cross_entropy_with_logits(logit, target)
