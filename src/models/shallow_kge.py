"""
Shallow knowledge-graph-embedding baseline — no graph convolution at all.

Plain lookup-table embeddings for proteins and GO terms, trained directly on
protein->GO triples with negative sampling. This isolates how much the
CompGCN's message passing actually contributes versus a standard ComplEx or
DistMult model with the same scoring function and embedding dimension,
trained shallowly the way the KGC literature usually does it (TransE/DistMult/
ComplEx on FB15k-237, WN18RR, etc.).

`ShallowLookupEncoder.forward(data)` deliberately matches `CompGCN.forward`'s
return signature `(protein_embs, go_embs, rel_embs)` and exposes the same
`rel_name_to_idx` attribute, so it is a drop-in replacement for `encoder` in
`evaluate_all` — no changes needed to the evaluation harness.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.data.graph_builder import build_annotation_matrix


class ShallowLookupEncoder(nn.Module):
    def __init__(self, n_proteins: int, n_go: int, hidden_dim: int, target_type: str):
        super().__init__()
        self.protein_emb = nn.Embedding(n_proteins, hidden_dim)
        self.go_emb      = nn.Embedding(n_go, hidden_dim)
        self.rel_emb     = nn.Embedding(1, hidden_dim)
        nn.init.xavier_uniform_(self.protein_emb.weight)
        nn.init.xavier_uniform_(self.go_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # Satisfies _get_has_function_rel_idx's "protein" + "function" match.
        self.rel_name_to_idx = {"protein_function": 0}
        self.target_type = target_type

    def forward(self, data: HeteroData, target_type=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.protein_emb.weight, self.go_emb.weight, self.rel_emb.weight


def train_shallow_kge(
    data:        HeteroData,
    target_type: str,
    scorer:      nn.Module,          # DistMult (=ComplEx) or TrueDistMult instance
    hidden_dim:  int,
    device:      str = "cuda",
    epochs:      int = 50,
    lr:          float = 1.0e-3,
    batch_size:  int = 1024,
    num_negatives: int = 32,
    weight_decay: float = 1.0e-5,
    log_every:   int = 10,
) -> ShallowLookupEncoder:
    """
    Trains a shallow embedding-table encoder against `scorer` using
    logistic (BCE-with-logits-style) loss with in-batch random negative
    sampling — the standard shallow KGC training recipe.

    Returns the trained encoder; pass it straight into evaluate_all(encoder=...)
    alongside `scorer` as the `distmult` argument and `generator=None`.
    """
    row, col, n_p, n_go = build_annotation_matrix(data, target_type)
    row, col = row.to(device), col.to(device)

    encoder = ShallowLookupEncoder(n_p, n_go, hidden_dim, target_type).to(device)
    scorer  = scorer.to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    n_edges = len(row)
    print(f"Training shallow {scorer.__class__.__name__} baseline: "
          f"{epochs} epochs, {n_edges:,} positive pairs, no graph convolution")

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n_edges, device=device)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n_edges, batch_size):
            b_idx = perm[start:start + batch_size]
            b_row, b_col = row[b_idx], col[b_idx]
            B = len(b_idx)

            p_emb = encoder.protein_emb(b_row)                              # [B, D]
            g_emb = encoder.go_emb(b_col)                                   # [B, D]
            r_emb = encoder.rel_emb.weight[0].unsqueeze(0).expand(B, -1)    # [B, D]

            pos_score = scorer(p_emb, r_emb, g_emb)                         # [B]

            neg_col   = torch.randint(0, n_go, (B, num_negatives), device=device)
            neg_g_emb = encoder.go_emb(neg_col)                             # [B, K, D]
            neg_r_emb = r_emb.unsqueeze(1).expand(-1, num_negatives, -1)
            neg_p_emb = p_emb.unsqueeze(1).expand(-1, num_negatives, -1)
            neg_score = scorer(neg_p_emb, neg_r_emb, neg_g_emb)             # [B, K]

            loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        if epoch % log_every == 0 or epoch == epochs:
            print(f"  [Shallow {scorer.__class__.__name__}] epoch {epoch}/{epochs}  "
                  f"loss={total_loss / max(n_batches, 1):.4f}")

    return encoder
