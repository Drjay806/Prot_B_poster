from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _ck
from torch_geometric.data import HeteroData

_EDGE_CHUNK = 5_000   # T4 has 14.6 GB; 50k chunks created ~250 MB FFT temp per chunk


def ccorr(a: Tensor, b: Tensor) -> Tensor:
    fa = torch.fft.rfft(a.float(), dim=-1)
    fb = torch.fft.rfft(b.float(), dim=-1)
    return torch.fft.irfft(fa.conj() * fb, n=a.size(-1), dim=-1)


class CompGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.W_O   = nn.Linear(in_dim, out_dim, bias=False)
        self.W_I   = nn.Linear(in_dim, out_dim, bias=False)
        self.W_S   = nn.Linear(in_dim, out_dim, bias=False)
        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        for w in [self.W_O, self.W_I, self.W_S, self.W_rel]:
            nn.init.xavier_uniform_(w.weight)

    def forward(
        self,
        node_embs: Dict[str, Tensor],
        rel_embs:  Tensor,
        edge_types: List[Tuple[str, int, str]],
        edge_indices: List[Tensor],
        num_nodes: Dict[str, int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        device = next(iter(node_embs.values())).device
        agg: Dict[str, Tensor] = {}

        for (src_type, rel_idx, dst_type), edge_index in zip(edge_types, edge_indices):
            n_edges = edge_index.size(1)
            if n_edges == 0:
                continue

            # Precompute rfft of the relation vector once per relation type.
            # Everything stays float32 — no autocast inside the GCN.
            e_rel     = rel_embs[rel_idx].float()
            fb_single = torch.fft.rfft(e_rel.unsqueeze(0), dim=-1)   # [1, D/2+1]

            for chunk_s in range(0, n_edges, _EDGE_CHUNK):
                chunk_e = min(chunk_s + _EDGE_CHUNK, n_edges)
                # Edge indices may be CPU tensors (graph kept on CPU); move to device
                src_idx = edge_index[0, chunk_s:chunk_e].to(device)
                dst_idx = edge_index[1, chunk_s:chunk_e].to(device)

                e_src    = node_embs[src_type][src_idx].float()          # [chunk, D]
                fa       = torch.fft.rfft(e_src, dim=-1)                 # [chunk, D/2+1]
                composed = torch.fft.irfft(fa.conj() * fb_single,
                                           n=self.in_dim, dim=-1)        # [chunk, D]
                del e_src, fa   # free FFT intermediates before linear layers

                # Keep everything float32 to avoid scatter dtype mismatches
                msg_out = self.W_O(composed)   # [chunk, out_dim] float32
                msg_in  = self.W_I(composed)   # [chunk, out_dim] float32
                del composed   # free before scatter

                if dst_type not in agg:
                    agg[dst_type] = torch.zeros(
                        num_nodes[dst_type], self.out_dim, device=device, dtype=torch.float32)
                if src_type not in agg:
                    agg[src_type] = torch.zeros(
                        num_nodes[src_type], self.out_dim, device=device, dtype=torch.float32)

                agg[dst_type].scatter_add_(
                    0, dst_idx.unsqueeze(1).expand(-1, self.out_dim), msg_out)
                agg[src_type].scatter_add_(
                    0, src_idx.unsqueeze(1).expand(-1, self.out_dim), msg_in)
                del msg_out, msg_in, src_idx, dst_idx   # free before next chunk

        new_node_embs: Dict[str, Tensor] = {}
        for ntype, h in node_embs.items():
            h_f      = h.float()
            agg_msg  = agg.get(ntype, torch.zeros(
                h_f.size(0), self.out_dim, device=device, dtype=torch.float32))
            self_msg = self.W_S(h_f)
            combined = self.norm(self.dropout(agg_msg) + self_msg)
            new_node_embs[ntype] = F.leaky_relu(combined, negative_slope=0.2)

        return new_node_embs, self.W_rel(rel_embs.float())


class CompGCN(nn.Module):
    """
    2-layer CompGCN encoder.

    Memory strategy:
    - Graph data stays on CPU; only ~150 MB of relevant node features are moved
      to GPU per forward pass.
    - GCN runs entirely in float32 (no autocast) to avoid dtype mismatches from
      mixed-precision LayerNorm / FFT interactions.
    - Each layer is gradient-checkpointed during training, so peak activation
      memory = one layer (~6.6 GB) instead of both layers simultaneously (13.2 GB).
    - Only edge types listed in encoder.gnn_edge_types are used for message passing.
    """

    def __init__(self, data: HeteroData, cfg: dict, target_type: Optional[str] = None):
        super().__init__()
        enc_cfg          = cfg["encoder"]
        self.hidden_dim  = enc_cfg["hidden_dim"]
        self.output_dim  = enc_cfg["output_dim"]
        self.num_layers  = enc_cfg["num_layers"]
        self.dropout_pretrain = enc_cfg.get("dropout_pretrain", 0.3)
        self.dropout_adv      = enc_cfg.get("dropout_adv", 0.1)

        _wl: List[str] = enc_cfg.get("gnn_edge_types", [])
        self.edge_type_whitelist: Optional[Set[str]] = set(_wl) if _wl else None

        self.node_types      = list(data.node_types)
        self.edge_types_raw  = list(data.edge_types)

        rel_names: Dict[str, int] = {}
        for _, rel, _ in self.edge_types_raw:
            if rel not in rel_names:
                rel_names[rel] = len(rel_names)
        self.rel_name_to_idx = rel_names

        # Only build projections for node types that appear in whitelisted edges.
        # This avoids loading Drug/Disease/HPO/... features (hundreds of MB) onto GPU.
        self._relevant_types: Set[str] = self._compute_relevant_types()

        self.input_projs = nn.ModuleDict()
        for ntype in self.node_types:
            if ntype not in self._relevant_types:
                continue
            node_store = data[ntype]
            if hasattr(node_store, "x") and node_store.x is not None:
                in_dim = node_store.x.shape[-1]
                self.input_projs[ntype] = nn.Sequential(
                    nn.Linear(in_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.ReLU(),
                )
            else:
                self.input_projs[ntype] = nn.Embedding(node_store.num_nodes, self.hidden_dim)

        self.target_type: str = target_type or cfg.get("data", {}).get("target_type", "GO_term_P")

        self.rel_emb = nn.Embedding(len(rel_names), self.hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight.unsqueeze(0))

        dims = [self.hidden_dim] * self.num_layers + [self.output_dim]
        self.layers = nn.ModuleList([
            CompGCNLayer(dims[i], dims[i + 1], dropout=self.dropout_pretrain)
            for i in range(self.num_layers)
        ])
        self.skip_projs = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False)
            if dims[i] != dims[i + 1] else nn.Identity()
            for i in range(self.num_layers)
        ])
        self.residual_norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(self.num_layers)
        ])

    def _compute_relevant_types(self) -> Set[str]:
        relevant: Set[str] = set()
        for src_type, rel, dst_type in self.edge_types_raw:
            if self.edge_type_whitelist is None or rel in self.edge_type_whitelist:
                relevant.add(src_type)
                relevant.add(dst_type)
        return relevant

    def set_dropout(self, mode: str):
        rate = self.dropout_pretrain if mode == "pretrain" else self.dropout_adv
        for layer in self.layers:
            layer.dropout.p = rate

    def _get_input_embeddings(self, data: HeteroData) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        node_embs: Dict[str, Tensor] = {}
        for ntype, proj in self.input_projs.items():
            if isinstance(proj, nn.Embedding):
                idx = torch.arange(data[ntype].num_nodes, device=device)
                node_embs[ntype] = proj(idx)
            else:
                x_cpu = data[ntype].x   # stays on CPU — never permanently moved to GPU
                if self.training:
                    # Gradient-checkpoint the projection so the raw feature tensor (e.g.
                    # 261k×1280 = 1.31 GB for Protein) is saved on CPU, not GPU.
                    # Without this, autograd holds a GPU copy alive for the weight gradient
                    # of the Linear layer, filling the T4 before any GCN chunk can run.
                    # During backward, _ck recomputes proj(x_cpu.to(device)) briefly.
                    _p, _d = proj, device
                    node_embs[ntype] = _ck(lambda xc, p=_p, d=_d: p(xc.to(d)),
                                           x_cpu, use_reentrant=False)
                else:
                    node_embs[ntype] = proj(x_cpu.to(device))
        return node_embs

    def _build_edge_info(self, data: HeteroData):
        edge_types_indexed, edge_indices = [], []
        for src_type, rel, dst_type in self.edge_types_raw:
            if (src_type, rel, dst_type) not in data.edge_types:
                continue
            if self.edge_type_whitelist is not None and rel not in self.edge_type_whitelist:
                continue
            if src_type not in self._relevant_types or dst_type not in self._relevant_types:
                continue
            # Keep edge index on CPU — layer forward moves chunks to GPU per iteration
            ei = data[(src_type, rel, dst_type)].edge_index.cpu()
            rel_idx = self.rel_name_to_idx[rel]
            edge_types_indexed.append((src_type, rel_idx, dst_type))
            edge_indices.append(ei)
        return edge_types_indexed, edge_indices

    def _apply_layer(self, layer, skip_proj, res_norm,
                     node_embs, ntypes, rel_embs,
                     edge_types_indexed, edge_indices, num_nodes):
        """Runs one GCN layer + residual. Used directly or via checkpoint."""
        new_ne, new_rel = layer(node_embs, rel_embs,
                                edge_types_indexed, edge_indices, num_nodes)
        for ntype in new_ne:
            skip = skip_proj(node_embs[ntype].float())
            new_ne[ntype] = res_norm(new_ne[ntype] + skip)
        return new_ne, new_rel

    def _apply_layer_ck(self, layer, skip_proj, res_norm,
                        node_embs, ntypes, rel_embs,
                        edge_types_indexed, edge_indices, num_nodes):
        """Gradient-checkpointed version: stores only layer inputs, recomputes on backward."""
        node_embs_list = [node_embs[nt] for nt in ntypes]

        def _fn(rel_in, *ne_in):
            ne = dict(zip(ntypes, ne_in))
            new_ne, new_rel = layer(ne, rel_in,
                                    edge_types_indexed, edge_indices, num_nodes)
            for ntype in new_ne:
                skip = skip_proj(ne[ntype].float())
                new_ne[ntype] = res_norm(new_ne[ntype] + skip)
            return (new_rel,) + tuple(new_ne[nt] for nt in ntypes)

        result   = _ck(_fn, rel_embs, *node_embs_list, use_reentrant=False)
        new_rel  = result[0]
        new_ne   = dict(zip(ntypes, result[1:]))
        return new_ne, new_rel

    def forward(self, data: HeteroData,
                target_type: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        ttype      = target_type or self.target_type
        node_embs  = self._get_input_embeddings(data)
        rel_embs   = self.rel_emb.weight.float()
        ntypes     = list(node_embs.keys())

        edge_types_indexed, edge_indices = self._build_edge_info(data)
        num_nodes = {ntype: data[ntype].num_nodes for ntype in self._relevant_types
                     if ntype in self.node_types}

        for layer, skip_proj, res_norm in zip(self.layers, self.skip_projs, self.residual_norms):
            if self.training:
                node_embs, rel_embs = self._apply_layer_ck(
                    layer, skip_proj, res_norm,
                    node_embs, ntypes, rel_embs,
                    edge_types_indexed, edge_indices, num_nodes)
            else:
                node_embs, rel_embs = self._apply_layer(
                    layer, skip_proj, res_norm,
                    node_embs, ntypes, rel_embs,
                    edge_types_indexed, edge_indices, num_nodes)

        protein_embs = node_embs["Protein"]
        if ttype not in node_embs:
            raise RuntimeError(
                f"Target type '{ttype}' not found. Available: {list(node_embs.keys())}")
        go_embs = node_embs[ttype]
        return protein_embs, go_embs, rel_embs
