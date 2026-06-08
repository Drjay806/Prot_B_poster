from typing import Dict, FrozenSet, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.distmult import DistMult
from src.models.discriminator import Discriminator
from src.utils.schedulers import CurriculumScheduler


class RewardModule(nn.Module):
    """
    Composite reward for RL training:
        R = w1*R_struct + w2*R_adv + w3(t)*R_sem - lambda_hier*P_hier

    w1, w2, w3 are learnable parameters constrained positive via softplus.
    w3 is scaled by a curriculum multiplier that ramps from 0→1 over warmup_epochs.
    """

    def __init__(self, cfg: dict, distmult: DistMult, discriminator: Discriminator):
        super().__init__()
        reward_cfg = cfg["reward"]
        self.lambda_hier = reward_cfg.get("lambda_hier", 0.5)
        self.k_ancestors = reward_cfg.get("k_ancestors", 20)

        self.distmult = distmult
        self.discriminator = discriminator

        # Raw (unconstrained) learnable weights — pass through softplus to ensure positivity
        self._w1 = nn.Parameter(torch.tensor(1.0 / 3.0))
        self._w2 = nn.Parameter(torch.tensor(1.0 / 3.0))
        self._w3 = nn.Parameter(torch.tensor(1.0 / 3.0))

        self.curriculum = CurriculumScheduler(
            warmup_epochs=reward_cfg.get("curriculum_warmup_epochs", 30)
        )
        self._curriculum_multiplier: float = 0.0

    def set_epoch(self, epoch: int):
        """Call at the start of each RL epoch to advance the curriculum."""
        self._curriculum_multiplier = self.curriculum.get_multiplier(epoch)

    def get_weights(self):
        w1 = F.softplus(self._w1)
        w2 = F.softplus(self._w2)
        w3 = F.softplus(self._w3) * self._curriculum_multiplier
        return w1, w2, w3

    def compute(
        self,
        protein_emb: Tensor,                           # [B, D]
        relation_emb: Tensor,                          # [D]
        fake_go_emb: Tensor,                           # [B, D]  — generator output
        true_go_emb: Optional[Tensor],                 # [B, D]  — ground truth GO emb (may be None)
        all_go_embs: Tensor,                           # [N_go, D]
        predicted_go_idxs: Optional[Tensor],           # [B]  — index of top-1 predicted GO term
        ancestor_table: Dict[int, FrozenSet[int]],
    ) -> Tensor:
        """
        Compute per-sample composite reward. Returns [B] tensor.
        All operations that don't need gradients for the reward signal are wrapped in no_grad.
        """
        w1, w2, w3 = self.get_weights()

        # Structural reward: DistMult score of (protein, relation, fake_go), mapped to [0,1]
        r_struct = torch.sigmoid(
            self.distmult(protein_emb, relation_emb.unsqueeze(0).expand_as(protein_emb), fake_go_emb)
        )   # [B]

        # Adversarial reward: discriminator's probability that the fake pair is real
        with torch.no_grad():
            r_adv = self.discriminator.real_probability(protein_emb, fake_go_emb)   # [B]

        # Semantic reward: cosine similarity between generated and true GO embedding
        if true_go_emb is not None:
            r_sem = F.cosine_similarity(fake_go_emb, true_go_emb, dim=-1)   # [B]
        else:
            r_sem = torch.zeros(protein_emb.size(0), device=protein_emb.device)

        # Hierarchy penalty (no gradient — pure scalar penalty)
        if predicted_go_idxs is not None and ancestor_table:
            p_hier = self._hierarchy_penalty(
                protein_emb, relation_emb, all_go_embs, predicted_go_idxs, ancestor_table
            )
        else:
            p_hier = torch.zeros_like(r_struct)

        reward = w1 * r_struct + w2 * r_adv + w3 * r_sem - self.lambda_hier * p_hier
        return reward, {
            "reward/structural": r_struct.mean().item(),
            "reward/adversarial": r_adv.mean().item(),
            "reward/semantic": r_sem.mean().item(),
            "reward/hierarchy_penalty": p_hier.mean().item(),
            "curriculum/w1": w1.item(),
            "curriculum/w2": w2.item(),
            "curriculum/w3": w3.item(),
        }

    @torch.no_grad()
    def _hierarchy_penalty(
        self,
        protein_emb: Tensor,
        relation_emb: Tensor,
        all_go_embs: Tensor,
        predicted_go_idxs: Tensor,
        ancestor_table: Dict[int, FrozenSet[int]],
    ) -> Tensor:
        """
        For each protein, compute the fraction of the predicted GO term's ancestors
        that are NOT in the top-k scored GO terms.
        Returns [B] penalty tensor in [0, 1].
        """
        B = protein_emb.size(0)
        # Score all GO terms per protein: [B, N_go]
        scaled = protein_emb * relation_emb.unsqueeze(0)
        scores = scaled @ all_go_embs.t()

        k = min(self.k_ancestors, all_go_embs.size(0))
        top_k_indices = scores.topk(k, dim=-1).indices   # [B, k]

        penalties = torch.zeros(B, device=protein_emb.device)
        for b in range(B):
            go_idx = predicted_go_idxs[b].item()
            ancestors = ancestor_table.get(go_idx, frozenset())
            if not ancestors:
                continue
            top_k_set = set(top_k_indices[b].tolist())
            missing = len(ancestors - top_k_set)
            penalties[b] = missing / len(ancestors)

        return penalties
