import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 used in forward
from torch import Tensor


class Discriminator(nn.Module):
    """
    GAN discriminator that scores (protein, GO_term) pairs as real or fake.

    Architecture:
        [protein(256) | go(256)] → 512
        FC1: 512 → 512, LayerNorm, LeakyReLU(0.2)
        FC2: 512 → 256, LayerNorm, LeakyReLU(0.2)
        FC3: 256 → 1    (raw logit)

    Uses label-smoothed BCE: real target=0.9, fake target=0.1.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        disc_cfg = cfg["discriminator"]
        self.smooth_real = disc_cfg.get("label_smooth_real", 0.9)
        self.smooth_fake = disc_cfg.get("label_smooth_fake", 0.1)
        neg_slope = disc_cfg.get("negative_slope", 0.2)

        hidden_dims = disc_cfg.get("hidden_dims", [512, 256])
        in_dim = disc_cfg.get("input_dim", 512)   # protein(256) + go(256)

        dims = [in_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.LeakyReLU(neg_slope))
        self.net = nn.Sequential(*layers)

    def forward(self, protein_emb: Tensor, go_emb: Tensor) -> Tensor:
        """Returns [B, 1] raw logits (before sigmoid)."""
        # Normalize both inputs so discrimination is based on direction, not magnitude.
        x = torch.cat([F.normalize(protein_emb, dim=-1), F.normalize(go_emb, dim=-1)], dim=-1)
        return self.net(x)

    def loss_real(self, protein_emb: Tensor, go_emb: Tensor) -> Tensor:
        """Discriminator loss for real (protein, GO) pairs."""
        logit = self.forward(protein_emb, go_emb)
        target = torch.full_like(logit, self.smooth_real)
        return F.binary_cross_entropy_with_logits(logit, target)

    def loss_fake(self, protein_emb: Tensor, fake_go_emb: Tensor) -> Tensor:
        """Discriminator loss for generator-produced fake pairs."""
        logit = self.forward(protein_emb, fake_go_emb.detach())
        target = torch.full_like(logit, self.smooth_fake)
        return F.binary_cross_entropy_with_logits(logit, target)

    def adversarial_loss_for_generator(self, protein_emb: Tensor, fake_go_emb: Tensor) -> Tensor:
        """Generator adversarial loss: wants D to output ~1 (real) for its outputs."""
        logit = self.forward(protein_emb, fake_go_emb)  # no detach — gradient flows to G
        target = torch.full_like(logit, self.smooth_real)
        return F.binary_cross_entropy_with_logits(logit, target)

    def real_probability(self, protein_emb: Tensor, go_emb: Tensor) -> Tensor:
        """Returns [B] probability that each pair is real (used as adversarial reward)."""
        logit = self.forward(protein_emb, go_emb)
        return torch.sigmoid(logit).squeeze(-1)

    def accuracy(self, protein_emb: Tensor, go_emb: Tensor, is_real: bool) -> float:
        """Fraction of correct classifications (for monitoring)."""
        with torch.no_grad():
            prob = self.real_probability(protein_emb, go_emb)
            pred_real = (prob > 0.5)
            correct = pred_real if is_real else ~pred_real
            return correct.float().mean().item()
