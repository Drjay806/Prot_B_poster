import torch
import torch.nn as nn


class DistMult(nn.Module):
    """
    ComplEx triple scorer for knowledge graph completion.

    Replaces the original DistMult scorer.

    Why ComplEx instead of DistMult:
        DistMult treats every relation as symmetric — score(A, r, B) always equals
        score(B, r, A).  Protein function is NOT symmetric ("has_function" goes one
        direction) and is many-to-many (many proteins share the same GO term; one
        protein has many GO terms).  DistMult cannot represent either property.

        TransE would be even worse: it requires all proteins with the same GO term
        to map to the same point in space, which is impossible for many-to-many.

        ComplEx splits each embedding into a real half and an imaginary half (like a
        complex number with a real part and an imaginary part).  The scoring formula
        uses the complex-valued inner product, which naturally handles asymmetric and
        many-to-many relations.  It is mathematically proven to represent any relation
        pattern that DistMult cannot.  Same hidden_dim, same interface, better math.

    Scoring formula (Trouillon et al. 2016):
        score(p, r, g) = Re( sum_k  p_k * r_k * conj(g_k) )
                       = sum_k ( p_re * r_re * g_re
                               + p_re * r_im * g_im
                               + p_im * r_re * g_im
                               - p_im * r_im * g_re )

    The embedding dim D is split: first D//2 dims = real part, last D//2 = imaginary.
    All existing callers pass D-dimensional vectors and receive scalar scores —
    the interface is unchanged.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.half       = hidden_dim // 2

    def _split(self, x: torch.Tensor):
        """Split [..., D] into real [..., D//2] and imaginary [..., D//2] parts."""
        return x[..., :self.half], x[..., self.half:]

    def forward(
        self,
        protein_emb:  torch.Tensor,   # [..., D]
        relation_emb: torch.Tensor,   # [..., D] or [D] (broadcast)
        go_emb:       torch.Tensor,   # [..., D]
    ) -> torch.Tensor:
        """Returns [...] scalar scores."""
        p_re, p_im = self._split(protein_emb)
        r_re, r_im = self._split(relation_emb)
        g_re, g_im = self._split(go_emb)

        # Re( p * r * conj(g) ) — four terms, last one negative
        score = (
              p_re * r_re * g_re
            + p_re * r_im * g_im
            + p_im * r_re * g_im
            - p_im * r_im * g_re
        ).sum(dim=-1)
        return score

    def score_all(
        self,
        protein_emb:  torch.Tensor,   # [N_p, D]
        relation_emb: torch.Tensor,   # [D]
        go_embs:      torch.Tensor,   # [N_g, D]
    ) -> torch.Tensor:
        """
        Score every (protein, relation, GO_term) combination.
        Returns [N_p, N_g] score matrix.
        """
        p_re, p_im = self._split(protein_emb)                # [N_p, D//2]
        r_re, r_im = self._split(relation_emb)               # [D//2]
        g_re, g_im = self._split(go_embs)                    # [N_g, D//2]

        # Scale proteins by relation once, then matrix multiply against GO halves
        pr_re = p_re * r_re.unsqueeze(0) - p_im * r_im.unsqueeze(0)   # [N_p, D//2]
        pr_im = p_re * r_im.unsqueeze(0) + p_im * r_re.unsqueeze(0)   # [N_p, D//2]

        # Re( (pr_re + i*pr_im) * conj(g_re + i*g_im) )
        # = pr_re @ g_re.T + pr_im @ g_im.T
        return pr_re @ g_re.t() + pr_im @ g_im.t()           # [N_p, N_g]

    def score_all_chunked(
        self,
        protein_emb:  torch.Tensor,   # [N_p, D]
        relation_emb: torch.Tensor,   # [D]
        go_embs:      torch.Tensor,   # [N_g, D]
        chunk_size:   int = 1000,
    ) -> torch.Tensor:
        """Memory-safe chunked version of score_all. Returns [N_p, N_g]."""
        parts = []
        for start in range(0, protein_emb.size(0), chunk_size):
            chunk = protein_emb[start:start + chunk_size]
            parts.append(self.score_all(chunk, relation_emb, go_embs).cpu())
        return torch.cat(parts, dim=0)
