from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData


def ccorr(a: Tensor, b: Tensor) -> Tensor:
    """Circular correlation: ifft(conj(fft(a)) * fft(b)).real"""
    fa = torch.fft.rfft(a, dim=-1)
    fb = torch.fft.rfft(b, dim=-1)
    return torch.fft.irfft(fa.conj() * fb, n=a.size(-1), dim=-1)


class CompGCNLayer(nn.Module):
    """
    Single CompGCN layer operating on a heterogeneous graph.
    Updates every node type by aggregating composed (neighbour, relation) messages.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Three projection matrices: outgoing edges (W_O), incoming edges (W_I), self (W_S)
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
        node_embs: Dict[str, Tensor],              # node_type → [N, in_dim]
        rel_embs: Tensor,                           # [num_relations, in_dim]
        edge_types: List[Tuple[str, int, str]],    # (src_type, rel_idx, dst_type)
        edge_indices: List[Tensor],                # per edge_type: [2, E]
        num_nodes: Dict[str, int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """Returns updated node_embs and updated relation embeddings."""
        agg: Dict[str, Tensor] = {}

        for (src_type, rel_idx, dst_type), edge_index in zip(edge_types, edge_indices):
            if edge_index.size(1) == 0:
                continue

            src_idx = edge_index[0]   # [E]
            dst_idx = edge_index[1]   # [E]

            e_src = node_embs[src_type][src_idx]   # [E, D]
            e_rel = rel_embs[rel_idx].unsqueeze(0).expand(e_src.size(0), -1)   # [E, D]

            # Outgoing direction: src → dst
            msg_out = self.W_O(ccorr(e_src, e_rel))   # [E, out_dim]
            # Incoming direction: treat as inverse relation
            msg_in = self.W_I(ccorr(e_src, e_rel))    # [E, out_dim]

            # agg buffers must match msg dtype — inside autocast both are float16;
            # scatter_add_ requires self.dtype == src.dtype, so derive from msg_out.
            n_dst = num_nodes[dst_type]
            if dst_type not in agg:
                agg[dst_type] = torch.zeros(n_dst, self.out_dim, device=e_src.device, dtype=msg_out.dtype)
            agg[dst_type].scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, self.out_dim), msg_out)

            n_src = num_nodes[src_type]
            if src_type not in agg:
                agg[src_type] = torch.zeros(n_src, self.out_dim, device=e_src.device, dtype=msg_in.dtype)
            agg[src_type].scatter_add_(0, src_idx.unsqueeze(1).expand(-1, self.out_dim), msg_in)

        # Combine aggregated messages with self-loop
        new_node_embs: Dict[str, Tensor] = {}
        for ntype, h in node_embs.items():
            agg_msg = agg.get(ntype, torch.zeros(h.size(0), self.out_dim, device=h.device, dtype=h.dtype))
            self_msg = self.W_S(h)
            combined = self.norm(self.dropout(agg_msg) + self_msg)
            new_node_embs[ntype] = F.leaky_relu(combined, negative_slope=0.2)

        # Update relation embeddings
        new_rel_embs = self.W_rel(rel_embs)

        return new_node_embs, new_rel_embs


class CompGCN(nn.Module):
    """
    3-layer CompGCN encoder with circular correlation composition and residual connections.
    Produces 256-dim embeddings for every protein and GO term node.
    """

    def __init__(self, data: HeteroData, cfg: dict, target_type: Optional[str] = None):
        super().__init__()
        enc_cfg = cfg["encoder"]
        self.hidden_dim = enc_cfg["hidden_dim"]        # 512
        self.output_dim = enc_cfg["output_dim"]        # 256
        self.num_layers = enc_cfg["num_layers"]        # 3
        self.dropout_pretrain = enc_cfg.get("dropout_pretrain", 0.3)
        self.dropout_adv = enc_cfg.get("dropout_adv", 0.1)
        self._current_dropout = self.dropout_pretrain

        # Collect metadata
        self.node_types = list(data.node_types)
        self.edge_types_raw = list(data.edge_types)

        # Build relation index: unique relation strings
        rel_names: Dict[str, int] = {}
        for _, rel, _ in self.edge_types_raw:
            if rel not in rel_names:
                rel_names[rel] = len(rel_names)
        self.rel_name_to_idx = rel_names
        num_relations = len(rel_names)

        # Input projection per node type — every node type in the real ProtHGT data has features
        # at varying dims: Protein=1280(ESM2), GO_term_*=200, Domain=50, Disease=100,
        # HPO=160, Drug=768, Compound=768, Pathway=200, kegg_Pathway=200, EC_number=256
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

        # Store the target GO type so forward() returns the right GO embeddings
        self.target_type: str = target_type or cfg.get("data", {}).get("target_type", "GO_term_P")

        # Relation embeddings
        self.rel_emb = nn.Embedding(num_relations, self.hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight.unsqueeze(0))

        # GCN layers
        dims = [self.hidden_dim] * self.num_layers + [self.output_dim]
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_d = dims[i]
            out_d = dims[i + 1]
            self.layers.append(CompGCNLayer(in_d, out_d, num_relations, dropout=self._current_dropout))

        # Skip projections for residual connections (layers 2 and 3 where dim may change)
        self.skip_projs = nn.ModuleList()
        for i in range(self.num_layers):
            in_d = dims[i]
            out_d = dims[i + 1]
            if in_d != out_d:
                self.skip_projs.append(nn.Linear(in_d, out_d, bias=False))
            else:
                self.skip_projs.append(nn.Identity())

        self.residual_norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(self.num_layers)])

    def set_dropout(self, mode: str):
        """Switch dropout between 'pretrain' (0.3) and 'adversarial' (0.1)."""
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
            ei = data[(src_type, rel, dst_type)].edge_index
            rel_idx = self.rel_name_to_idx[rel]
            edge_types_indexed.append((src_type, rel_idx, dst_type))
            edge_indices.append(ei)
        return edge_types_indexed, edge_indices

    def forward(self, data: HeteroData, target_type: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            protein_embs: [N_protein, output_dim]
            go_embs: [N_go, output_dim]  — embeddings for `target_type` (default: self.target_type)
            rel_embs: [num_relations, output_dim]
        """
        ttype = target_type or self.target_type
        node_embs = self._get_input_embeddings(data)
        rel_embs = self.rel_emb.weight    # [R, hidden_dim]

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
            raise RuntimeError(f"Target type '{ttype}' not found in node embeddings. Available: {list(node_embs.keys())}")
        go_embs = node_embs[ttype]

        return protein_embs, go_embs, rel_embs
