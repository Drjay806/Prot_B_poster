from typing import Dict, FrozenSet, Set, Tuple

import torch
from torch import Tensor


class NegativeSampler:
    """
    Tiered negative sampler for WGAN-GP adversarial protein function training.

    GO terms are assigned to difficulty tiers based on their depth in the GO
    hierarchy (number of ancestors as a proxy for specificity):

      Easy   (~33% shallowest): General terms — obviously wrong for specific functions.
      Medium (~33% mid-depth) : Same namespace, different functional area.
      Hard   (~33% deepest)   : Specific terms close to true annotations in the
                                ontology, weighted by DistMult score using
                                self-adversarial sampling (Sun et al., RotatE, ICLR 2019).

    A curriculum linearly shifts sampling ratios from easy-heavy at epoch 1
    toward hard-heavy by epoch T//2, matching training progression.
    """

    def __init__(
        self,
        n_go:            int,
        protein_indices: Tensor,
        go_indices:      Tensor,
        ancestor_table:  Dict[int, FrozenSet[int]],
    ):
        self.n_go          = n_go
        self.ancestor_table = ancestor_table

        # Build protein → annotated GO set for exclusion filtering
        self.protein_to_go: Dict[int, Set[int]] = {}
        for p, g in zip(protein_indices.tolist(), go_indices.tolist()):
            self.protein_to_go.setdefault(int(p), set()).add(int(g))

        # Depth = number of ancestors — more ancestors = more specific = harder negative
        depths        = [len(ancestor_table.get(g, frozenset())) for g in range(n_go)]
        sorted_depths = sorted(depths)
        t1 = sorted_depths[int(0.33 * n_go)]
        t2 = sorted_depths[int(0.67 * n_go)]

        self.easy_pool   = torch.tensor(
            [g for g, d in enumerate(depths) if d <= t1],       dtype=torch.long)
        self.medium_pool = torch.tensor(
            [g for g, d in enumerate(depths) if t1 < d <= t2],  dtype=torch.long)
        self.hard_pool   = torch.tensor(
            [g for g, d in enumerate(depths) if d > t2],        dtype=torch.long)

        print(f"[NegativeSampler] tiers — "
              f"easy: {len(self.easy_pool):,}  "
              f"medium: {len(self.medium_pool):,}  "
              f"hard: {len(self.hard_pool):,}")

    @staticmethod
    def curriculum_ratios(epoch: int, total_epochs: int) -> Tuple[float, float, float]:
        """
        Linearly shift easy:medium:hard ratios over first half of training.
          Epoch 1    → (0.50, 0.40, 0.10)
          Epoch T//2 → (0.30, 0.40, 0.30)
        """
        progress = min(epoch / max(total_epochs * 0.5, 1), 1.0)
        easy_r   = 0.50 - 0.20 * progress
        hard_r   = 0.10 + 0.20 * progress
        medium_r = 1.0 - easy_r - hard_r
        return (float(easy_r), float(medium_r), float(hard_r))

    def sample(
        self,
        b_prot_idx:   Tensor,       # [B] protein indices — CPU tensor
        go_embs:      Tensor,       # [N_go, D] on device
        protein_embs: Tensor,       # [N_prot, D] on device
        rel_vec:      Tensor,       # [D] on device
        device:       str,
        ratios:       Tuple[float, float, float] = (0.30, 0.40, 0.30),
        temperature:  float = 0.5,
    ) -> Tensor:
        """
        Sample one negative GO index per protein. Returns [B] long tensor on device.
        """
        B  = b_prot_idx.size(0)
        rv = torch.rand(B)

        use_easy   = rv < ratios[0]
        use_medium = (rv >= ratios[0]) & (rv < ratios[0] + ratios[1])
        use_hard   = rv >= ratios[0] + ratios[1]

        result = torch.zeros(B, dtype=torch.long)

        # ── Easy: random from shallow GO terms ─────────────────────────────
        n_easy = int(use_easy.sum())
        if n_easy and len(self.easy_pool) > 0:
            idx = torch.randint(len(self.easy_pool), (n_easy,))
            result[use_easy] = self.easy_pool[idx]

        # ── Medium: random from mid-depth GO terms ──────────────────────────
        n_medium = int(use_medium.sum())
        if n_medium and len(self.medium_pool) > 0:
            idx = torch.randint(len(self.medium_pool), (n_medium,))
            result[use_medium] = self.medium_pool[idx]

        # ── Hard: self-adversarial DistMult-weighted sampling ───────────────
        n_hard = int(use_hard.sum())
        if n_hard and len(self.hard_pool) > 0:
            hard_mask   = use_hard.nonzero(as_tuple=True)[0]          # CPU
            hard_p_idx  = b_prot_idx[hard_mask]                       # CPU
            p_embs_hard = protein_embs[hard_p_idx.to(device)]         # [n_hard, D]
            pool_dev    = self.hard_pool.to(device)
            h_embs      = go_embs[pool_dev]                           # [N_hard_pool, D]

            # DistMult scores: [n_hard, N_hard_pool]
            scaled  = p_embs_hard * rel_vec.unsqueeze(0)
            scores  = (scaled @ h_embs.t()).float()
            weights = torch.softmax(temperature * scores, dim=-1)

            # Sample one per protein, proportional to score
            chosen          = torch.multinomial(weights.cpu(), 1).squeeze(1)  # [n_hard]
            result[use_hard] = self.hard_pool[chosen]

        result = result.to(device)

        # ── Filter: replace accidentally-sampled true annotations ───────────
        for i in range(B):
            p        = int(b_prot_idx[i])
            true_set = self.protein_to_go.get(p, set())
            if int(result[i]) in true_set and len(self.easy_pool) > 0:
                fallback = torch.randint(len(self.easy_pool), (1,))
                result[i] = self.easy_pool[fallback].to(device)

        return result
