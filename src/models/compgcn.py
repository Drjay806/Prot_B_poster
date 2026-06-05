from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

# Process at most this many edges per chunk to cap peak VRAM usage.
# At 50k edges, float16, dim=512: ~100 MB per chunk — safe on T4.
_EDGE_CHUNK = 50_000


def ccorr(a: Tensor, b: Tensor) -> Tensor:
    """Circular correlation: ifft(conj(fft(a)) * fft(b)).real"""
    fa = torch.fft.rfft(a, dim=-1)
    fb = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(fa.conj() * fb, n=a.size(-1), dim=-1)


class CompGCNLayer(nn.Module):
    """
    Single CompGCN layer operating on a heterogeneous graph.
    Edges are processed in chunks of _EDGE_CHUNK to bound peak VRAM.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_O = nn.Linear(in_dim, out_dim, bias=False)
        self.W_I = nn.Linear(in_dim, out_dim, bias=False)
        self.W_S = nn.Linear(in_dim, out_dim, bias=False)
        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W_O.weight)
        nn.init.xavier_uniform_(self.W_I.weight)
        nn.init.xavier_uniform_(self.W_S.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)

    def forward(
        self,
        node_embs: Dict[str, Tensor],
        rel_embs: Tensor,
        edge_types: List[Tuple[str, int, str]],
        edge_indices: List[Tensor],
        num_nodes: Dict[str, int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        agg: Dict[str, Tensor] = {}
        device = next(iter(node_embs.values())).device

        for (src_type, rel_idx, dst_type), edge_index in zip(edge_types, edge_indices):
            n_edges = edge_index.size(1)
            if n_edges == 0:
                continue

            e_rel = rel_embs[rel_idx]  # [D] — one vector for the whole relation type

            for chunk_s in range(0, n_edges, _EDGE_CHUNK):
                chunk_e = min(chunk_s + _EDGE_CHUNK, n_edges)
                src_idx = edge_index[0, chunk_s:chunk_e]
                dst_idx = edge_index[1, chunk_s:chunk_e]

                e_src = node_embs[src_type][src_idx]                              # [chunk, D]
                e_rel_exp = e_rel.unsqueeze(0).expand(len(src_idx), -1)           # [chunk, D]

                msg_out = self.W_O(ccorr(e_src, e_rel_exp))                       # [chunk, out_dim]
                msg_in  = self.W_I(ccorr(e_src, e_rel_exp))                       # [chunk, out_dim]

                # agg buffers derive dtype from msg so scatter_add_ never has a type mismatch
                if dst_type not in agg:
                    agg[dst_type] = torch.zeros(num_nodes[dst_type], self.out_dim,
                                                device=device, dtype=msg_out.dtype)
                if src_type not in agg:
                    agg[src_type] = torch.zeros(num_nodes[src_type], self.out_dim,
                                                device=device, dtype=msg_in.dtype)

                agg[dst_type].scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, self.out_dim), msg_out)
                agg[src_type].scatter_add_(0, src_idx.unsqueeze(1).expand(-1, self.out_dim), msg_in)

        new_node_embs: Dict[str, Tensor] = {}
        for ntype, h in node_embs.items():
            agg_msg = agg.get(ntype, torch.zeros(h.size(0), self.out_dim, device=h.device, dtype=h.dtype))
            self_msg = self.W_S(h)
            combined = self.norm(self.dropout(agg_msg) + self_msg)
            new_node_embs[ntype] = F.leaky_relu(combined, negative_slope=0.2)

        return new_node_embs, self.W_rel(rel_embs)


class CompGCN(nn.Module):
    """
    3-layer CompGCN encoder with circular correlation composition and residual connections.
    Produces 256-dim embeddings for every protein and GO term node.

    Memory controls:
    - Edge processing is chunked (_EDGE_CHUNK edges at a time).
    - Only edge types listed in `encoder.gnn_edge_types` config are used for message
      passing.  An empty list means "use all edge types" (original behaviour, risky on
      large graphs).  Recommended: ["protein_function", "function_function"].
    """

    def __init__(self, data: HeteroData, cfg: dict, target_type: Optional[str] = None):
        super().__init__()
        enc_cfg = cfg["encoder"]
        self.hidden_dim = enc_cfg["hidden_dim"]
        self.output_dim = enc_cfg["output_dim"]
        self.num_layers = enc_cfg["num_layers"]
        self.dropout_pretrain = enc_cfg.get("dropout_pretrain", 0.3)
        self.dropout_adv = enc_cfg.get("dropout_adv", 0.1)
        self._current_dropout = self.dropout_pretrain

        # Optional whitelist of relation names to include in message passing.
        # Filtering to protein_function + function_function reduces edge count from
        # potentially 50M+ to ~1M, keeping activation memory within T4 budget.
        _wl: List[str] = enc_cfg.get("gnn_edge_types", [])
        self.edge_type_whitelist: Optional[Set[str]] = set(_wl) if _wl else None

        self.node_types = list(data.node_types)
        self.edge_types_raw = list(data.edge_types)

        rel_names: Dict[str, int] = {}
        for _, rel, _ in self.edge_types_raw:
            if rel not in rel_names:
                rel_names[rel] = len(rel_names)
        self.rel_name_to_idx = rel_names
        num_relations = len(rel_names)

        self.input_projs = nn.ModuleDict()
        for ntype in self.node_types:
            node_store = data[ntype]
            if hasattr(node_store, "x") and node_store.x is not None:
                in_dim = node_store.x.shape[-1]
                self.input_projs[ntype] = nn.Sequential(
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                )
            else:
                n = node_store.num_nodes
                self.input_projs[ntype] = nn.Embedding(n, self.hidden_dim)

        self.target_type: str = target_type or cfg.get("data", {}).get("target_type", "GO_term_P")

        self.rel_emb = nn.Embedding(num_relations, self.hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight.unsqueeze(0))

        dims = [self.hidden_dim] * self.num_layers + [self.output_dim]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(CompGCNLayer(dims[i], dims[i + 1],
                                            dropout=self._current_dropout))

        self.skip_projs = nn.ModuleList()
        for i in range(self.num_layers):
            in_d, out_d = dims[i], dims[i + 1]
            self.skip_projs.append(nn.Linear(in_d, out_d, bias=False) if in_d != out_d else nn.Identity())

        self.residual_norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(self.num_layers)])

    def set_dropout(self, mode: str):
        rate = self.dropout_pretrain if mode == "pretrain" else self.dropout_adv
        for layer in self.layers:
            layer.dropout.p = rate

    def _get_input_embeddings(self, data: HeteroData) -> Dict[str, Tensor]:
        node_embs: Dict[str, Tensor] = {}
        for ntype, proj in self.input_projs.items():
            if isinstance(proj, nn.Embedding):
                idx = torch.arange(data[ntype].num_nodes, device=next(self.parameters()).device)
                node_embs[ntype] = proj(idx)
            else:
                node_embs[ntype] = proj(data[ntype].x)
        return node_embs

    def _build_edge_info(self, data: HeteroData):
        edge_types_indexed = []
        edge_indices = []
        for src_type, rel, dst_type in self.edge_types_raw:
            if (src_type, rel, dst_type) not in data.edge_types:
                continue
            if self.edge_type_whitelist is not None and rel not in self.edge_type_whitelist:
                continue
            ei = data[(src_type, rel, dst_type)].edge_index
            rel_idx = self.rel_name_to_idx[rel]
            edge_types_indexed.append((src_type, rel_idx, dst_type))
            edge_indices.append(ei)
        return edge_types_indexed, edge_indices

    def forward(self, data: HeteroData, target_type: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            protein_embs: [N_protein, output_dim]
            go_embs:      [N_go, output_dim]
            rel_embs:     [num_relations, output_dim]
        """
        ttype = target_type or self.target_type
        node_embs = self._get_input_embeddings(data)
        rel_embs = self.rel_emb.weight

        edge_types_indexed, edge_indices = self._build_edge_info(data)
        num_nodes = {ntype: data[ntype].num_nodes for ntype in self.node_types}

        for layer, skip_proj, res_norm in zip(self.layers, self.skip_projs, self.residual_norms):
            new_node_embs, rel_embs = layer(node_embs, rel_embs, edge_types_indexed, edge_indices, num_nodes)
            for ntype in new_node_embs:
                skip = skip_proj(node_embs[ntype])
                new_node_embs[ntype] = res_norm(new_node_embs[ntype] + skip)
            node_embs = new_node_embs

        protein_embs = node_embs["Protein"]

        if ttype not in node_embs:
            raise RuntimeError(
                f"Target type '{ttype}' not found. Available: {list(node_embs.keys())}"
            )
        go_embs = node_embs[ttype]

        return protein_embs, go_embs, rel_embs
