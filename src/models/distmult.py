import torch
import torch.nn as nn


class DistMult(nn.Module):
    """
    DistMult triple scorer for knowledge graph completion.
    Scores (protein, has_function, GO_term) triples via element-wise product + sum.
    Higher score = more plausible triple in the knowledge graph.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        protein_emb: torch.Tensor,   # [..., D]
        relation_emb: torch.Tensor,  # [..., D] or [D] (broadcast)
        go_emb: torch.Tensor,        # [..., D]
    ) -> torch.Tensor:
        """Returns [...] scalar scores."""
        return (protein_emb * relation_emb * go_emb).sum(dim=-1)

    def score_all(
        self,
        protein_emb: torch.Tensor,   # [N_p, D]
        relation_emb: torch.Tensor,  # [D]
        go_embs: torch.Tensor,       # [N_g, D]
    ) -> torch.Tensor:
        """
        Score every (protein, relation, GO_term) combination efficiently.
        Returns [N_p, N_g] score matrix.
        """
        # [N_p, D] * [D] → [N_p, D], then [N_p, D] @ [N_g, D]^T → [N_p, N_g]
        scaled_protein = protein_emb * relation_emb.unsqueeze(0)   # [N_p, D]
        return scaled_protein @ go_embs.t()                          # [N_p, N_g]

    def score_all_chunked(
        self,
        protein_emb: torch.Tensor,   # [N_p, D]
        relation_emb: torch.Tensor,  # [D]
        go_embs: torch.Tensor,       # [N_g, D]
        chunk_size: int = 1000,
    ) -> torch.Tensor:
        """Memory-safe chunked version of score_all. Returns [N_p, N_g]."""
        parts = []
        for start in range(0, protein_emb.size(0), chunk_size):
            chunk = protein_emb[start:start + chunk_size]
            parts.append(self.score_all(chunk, relation_emb, go_embs).cpu())
        return torch.cat(parts, dim=0)
