"""
CalibrationHead: MLP classifier on frozen CompGCN embeddings.

The GAN discriminator and this head share identical architecture:
    concat(protein_emb [D], go_emb [D]) → MLP → scalar

The difference is the training objective:
    Discriminator: WGAN adversarial loss (real GO vs generator fake GO)
    CalibrationHead: weighted BCE on true protein-GO annotations

ComplEx scores are excellent for ranking (AUROC 0.97) but uncalibrated for
Fmax thresholding. This head directly predicts P(annotation is true) using
the same frozen encoder embeddings, giving calibrated probabilities that the
Fmax threshold sweep can use cleanly — without retraining the encoder or GAN.

The `.score()` method matches the Discriminator.score() interface exactly, so
evaluate_all's critic_score_all infrastructure works unchanged with this head
passed as the discriminator argument under mode="calibration".
"""

import torch
import torch.nn as nn
from typing import Sequence


class CalibrationHead(nn.Module):

    def __init__(
        self,
        embed_dim:   int = 256,
        hidden_dims: Sequence[int] = (256, 128),
        dropout:     float = 0.1,
    ):
        super().__init__()
        in_dim = embed_dim * 2   # concat(protein_emb, go_emb)
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        protein_emb: torch.Tensor,   # [..., D]
        go_emb:      torch.Tensor,   # [..., D]
    ) -> torch.Tensor:
        """Returns [...] sigmoid probabilities in (0, 1)."""
        return self.net(torch.cat([protein_emb, go_emb], dim=-1)).squeeze(-1)

    def score(
        self,
        protein_emb: torch.Tensor,
        go_emb:      torch.Tensor,
    ) -> torch.Tensor:
        """Alias for forward — matches Discriminator.score() interface."""
        return self.forward(protein_emb, go_emb)
