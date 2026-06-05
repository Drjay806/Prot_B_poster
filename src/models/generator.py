import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    """
    GAN generator that maps (protein_emb, relation_emb, noise) → candidate GO embedding.

    Architecture:
        [protein(256) | relation(256) | noise(64)] → 576
        FC1: 576 → 512, LayerNorm, LeakyReLU(0.2)
        FC2: 512 → 512, LayerNorm, LeakyReLU(0.2)
        FC3: 512 → 256, LayerNorm, LeakyReLU(0.2)
        FC4: 256 → 256   (no activation — output lives in GO embedding space)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        gen_cfg = cfg["generator"]
        protein_dim = gen_cfg.get("protein_dim", 256)
        relation_dim = gen_cfg.get("relation_dim", 256)
        self.noise_dim = gen_cfg.get("noise_dim", 64)
        output_dim = gen_cfg.get("output_dim", 256)
        hidden_dims = gen_cfg.get("hidden_dims", [512, 512, 256])
        neg_slope = gen_cfg.get("negative_slope", 0.2)

        in_dim = protein_dim + relation_dim + self.noise_dim
        dims = [in_dim] + hidden_dims + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.LeakyReLU(neg_slope))
        # Final LayerNorm keeps output on same scale as real GO embeddings
        # and prevents DistMult score explosion during adversarial training.
        layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        protein_emb: Tensor,          # [B, 256]
        relation_emb: Tensor,         # [B, 256] or [256]
        noise: Tensor | None = None,  # [B, 64] — sampled externally or None for deterministic
    ) -> Tensor:
        """Returns [B, 256] candidate GO embedding."""
        B = protein_emb.size(0)
        device = protein_emb.device

        if relation_emb.dim() == 1:
            relation_emb = relation_emb.unsqueeze(0).expand(B, -1)

        if noise is None:
            noise = torch.zeros(B, self.noise_dim, device=device)

        x = torch.cat([protein_emb, relation_emb, noise], dim=-1)
        return self.net(x)

    def sample(
        self,
        protein_emb: Tensor,         # [B, 256]
        relation_emb: Tensor,        # [256] or [B, 256]
        k: int = 1,
    ) -> Tensor:
        """
        Sample k candidate GO embeddings per protein by running k forward passes with different noise.
        Returns [B, k, 256].
        """
        B = protein_emb.size(0)
        candidates = []
        for _ in range(k):
            noise = torch.randn(B, self.noise_dim, device=protein_emb.device)
            candidates.append(self.forward(protein_emb, relation_emb, noise))
        return torch.stack(candidates, dim=1)   # [B, k, 256]

    def deterministic(self, protein_emb: Tensor, relation_emb: Tensor) -> Tensor:
        """Single deterministic forward (noise=0). Used during evaluation."""
        return self.forward(protein_emb, relation_emb, noise=None)
