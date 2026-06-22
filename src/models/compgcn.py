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

        _d = lambda: torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[MEM] CompGCNLayer.forward entry: {_d():.3f} GB")
        print(f"[MEM]   node_embs: {[(k, tuple(v.shape), f'{v.numel()*4/1e6:.0f}MB') for k,v in node_embs.items()]}")
        print(f"[MEM]   num_nodes: {num_nodes}")

        for (src_type, rel_idx, dst_type), edge_index in zip(edge_types, edge_indices):
            n_edges = edge_index.size(1)
            if n_edges == 0:
                continue

            # Precompute rfft of the relation vector once per relation type.
            # Everything stays float32 — no autocast inside the GCN.
            e_rel     = rel_embs[rel_idx].float()
            fb_single = torch.fft.rfft(e_rel.unsqueeze(0), dim=-1)   # [1, D/2+1]

            print(f"[MEM]   rel ({src_type}→{dst_type}) n_edges={n_edges}: {_d():.3f} GB after fb_single")

            for chunk_s in range(0, n_edges, _EDGE_CHUNK):
                chunk_e = min(chunk_s + _EDGE_CHUNK, n_edges)
                # Edge indices may be CPU tensors (graph kept on CPU); move to device
                src_idx = edge_index[0, chunk_s:chunk_e].to(device)
                dst_idx = edge_index[1, chunk_s:chunk_e].to(device)

                e_src    = node_embs[src_type][src_idx].float()          # [chunk, D]
                fa       = torch.fft.rfft(e_src, dim=-1)                 # [chunk, D/2+1]
                print(f"[MEM]     chunk {chunk_s}-{chunk_e}: after rfft={_d():.3f} GB  e_src={tuple(e_src.shape)} fa={tuple(fa.shape)}")
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
                # Project raw features without grad to avoid moving 261k×1280=1.3 GB
                # ESM2 features to GPU permanently.  The projection output (268 MB for
                # Protein) goes on GPU; raw features are freed immediately via del+cache.
                with torch.no_grad():
                    x = data[ntype].x.to(device)
                    emb = proj(x)
                    del x
                node_embs[ntype] = emb
        torch.cuda.empty_cache()   # flush freed raw-feature blocks before GCN forward
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
        _d = lambda: torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[MEM] _apply_layer_ck before _ck: {_d():.3f} GB  (node_embs_list total={sum(t.numel()*4 for t in node_embs_list)/1e6:.0f} MB)")

        def _fn(rel_in, *ne_in):
            print(f"[MEM] inside _fn (checkpoint body): {_d():.3f} GB")
            ne = dict(zip(ntypes, ne_in))
            new_ne, new_rel = layer(ne, rel_in,
                                    edge_types_indexed, edge_indices, num_nodes)
            for ntype in new_ne:
                skip = skip_proj(ne[ntype].float())
                new_ne[ntype] = res_norm(new_ne[ntype] + skip)
            return (new_rel,) + tuple(new_ne[nt] for nt in ntypes)

        # use_reentrant=True runs _fn inside torch.no_grad() during the forward
        # pass, so PyTorch never builds an autograd tape for the thousands of
        # operations inside _fn (18 edge types × millions of edges = many chunks).
        # use_reentrant=False builds an internal computation trace proportional to
        # the number of operations — for our GCN this grows to 14 GB before the
        # first chunk completes.  With use_reentrant=True, peak memory = live
        # tensors only (~330 MB agg + 10 MB chunk temps).  During backward the
        # checkpoint re-runs _fn with enable_grad to recompute intermediates and
        # accumulate gradients into the GCN weight leaf parameters.
        result   = _ck(_fn, rel_embs, *node_embs_list, use_reentrant=True)
        new_rel  = result[0]
        new_ne   = dict(zip(ntypes, result[1:]))
        return new_ne, new_rel

    def forward(self, data: HeteroData,
                target_type: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
        _d = lambda: torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"[MEM] CompGCN.forward entry: {_d():.3f} GB")
        ttype      = target_type or self.target_type
        node_embs  = self._get_input_embeddings(data)
        print(f"[MEM] after _get_input_embeddings: {_d():.3f} GB")
        rel_embs   = self.rel_emb.weight.float()
        ntypes     = list(node_embs.keys())

        edge_types_indexed, edge_indices = self._build_edge_info(data)
        print(f"[MEM] after _build_edge_info: {_d():.3f} GB  ({len(edge_types_indexed)} edge type groups)")
        # Use x.shape[0] when available — ProtHGT data sets _num_nodes to the raw
        # UniProt database size (~13.7M) while x only covers the 261k featured proteins.
        # data[ntype].num_nodes returns _num_nodes (explicit metadata wins over x.shape[0]
        # in PyG), so using it here causes agg["Protein"] = zeros(13.7M, 256) = 14 GB.
        num_nodes = {}
        for _nt in self._relevant_types:
            if _nt not in self.node_types:
                continue
            _store = data[_nt]
            if hasattr(_store, 'x') and _store.x is not None:
                num_nodes[_nt] = _store.x.shape[0]
            else:
                num_nodes[_nt] = _store.num_nodes

        for _li, (layer, skip_proj, res_norm) in enumerate(zip(self.layers, self.skip_projs, self.residual_norms)):
            print(f"[MEM] before layer {_li}: {_d():.3f} GB")
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
