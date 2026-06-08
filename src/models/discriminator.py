import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Critic(nn.Module):
    """
    WGAN-GP critic for protein-function compatibility scoring.

    Replaces the binary BCE discriminator. Outputs unbounded real scores —
    high values indicate compatible (protein, GO) pairs, low values indicate
    incompatible. Both inputs are L2-normalised before concatenation so the
    critic learns directional protein-function relationships rather than
    responding to embedding magnitude differences.

    Trained with three input types:
      - Real      : annotated (protein, GO) pairs from the knowledge graph
      - Fake      : generator-produced GO embeddings
      - Hard-neg  : plausible but unannotated GO terms sampled from the vocabulary
    """

    def __init__(self, cfg: dict):
        super().__init__()
        disc_cfg    = cfg["discriminator"]
        neg_slope   = disc_cfg.get("negative_slope", 0.2)
        hidden_dims = disc_cfg.get("hidden_dims", [512, 256])
        in_dim      = disc_cfg.get("input_dim", 512)

        dims   = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.LeakyReLU(neg_slope))
        self.net = nn.Sequential(*layers)

    def forward(self, protein_emb: Tensor, go_emb: Tensor) -> Tensor:
        """Returns [B, 1] unbounded compatibility scores."""
        x = torch.cat([
            F.normalize(protein_emb.float(), dim=-1),
            F.normalize(go_emb.float(), dim=-1),
        ], dim=-1)
        return self.net(x)

    def score(self, protein_emb: Tensor, go_emb: Tensor) -> Tensor:
        """Returns [B] scalar scores (unbounded real values)."""
        return self.forward(protein_emb, go_emb).squeeze(-1)

    def real_probability(self, protein_emb: Tensor, go_emb: Tensor) -> Tensor:
        """[B] probability in [0,1]. Sigmoid of score. Used by RL reward module."""
        return torch.sigmoid(self.score(protein_emb, go_emb))

    def accuracy(self, protein_emb: Tensor, go_emb: Tensor, is_real: bool) -> float:
        """Fraction classified correctly: score > 0 treated as real prediction."""
        with torch.no_grad():
            pred_real = self.score(protein_emb, go_emb) > 0
            correct   = pred_real if is_real else ~pred_real
            return correct.float().mean().item()


# Backward-compatible alias — notebook code using Discriminator still works.
Discriminator = Critic
