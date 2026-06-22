from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


def get_all_node_types(data: HeteroData) -> List[str]:
    return list(data.node_types)


def get_all_relation_types(data: HeteroData) -> List[Tuple[str, str, str]]:
    return list(data.edge_types)


def get_relation_name_list(data: HeteroData) -> List[str]:
    """Return unique relation (middle) strings in edge type triples."""
    seen = {}
    names = []
    for src, rel, dst in data.edge_types:
        if rel not in seen:
            seen[rel] = True
            names.append(rel)
    return names


def build_relation_embedding(data: HeteroData, hidden_dim: int = 256) -> nn.Embedding:
    """
    Create a learnable relation embedding table (one vector per unique relation name).
    Initialised with Xavier uniform. The embedding index for each relation is its
    position in `get_relation_name_list(data)`.
    """
    rel_names = get_relation_name_list(data)
    emb = nn.Embedding(len(rel_names), hidden_dim)
    nn.init.xavier_uniform_(emb.weight.unsqueeze(0))
    return emb


def build_annotation_matrix(
    data: HeteroData,
    target_type: str,
    src_type: str = "Protein",
    relation: str = "protein_function",
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Extract positive annotation edges from a split graph.

    Returns:
        row_idx: [E] protein indices
        col_idx: [E] GO term indices
        num_proteins: int
        num_go: int
    """
    edge_key = (src_type, relation, target_type)
    # Try to find the edge type regardless of exact relation string name
    if edge_key not in data.edge_types:
        for et in data.edge_types:
            if et[0] == src_type and et[2] == target_type:
                edge_key = et
                break
        else:
            raise KeyError(
                f"Could not find a protein→GO edge in the graph. "
                f"Available edge types: {data.edge_types}"
            )

    edge_store = data[edge_key]
    if hasattr(edge_store, "edge_label_index") and edge_store.edge_label_index is not None:
        row = edge_store.edge_label_index[0]
        col = edge_store.edge_label_index[1]
        if hasattr(edge_store, "edge_label"):
            mask = edge_store.edge_label.bool()
            row, col = row[mask], col[mask]
    else:
        row, col = edge_store.edge_index[0], edge_store.edge_index[1]

    num_proteins = data[src_type].x.shape[0]
    num_go = data[target_type].x.shape[0]
    return row, col, num_proteins, num_go


def build_dense_annotation_matrix(
    data: HeteroData,
    target_type: str,
    src_type: str = "Protein",
    relation: str = "protein_function",
) -> torch.Tensor:
    """Returns a dense [num_proteins, num_go] float tensor of ground-truth annotations."""
    row, col, n_p, n_g = build_annotation_matrix(data, target_type, src_type, relation)
    mat = torch.zeros(n_p, n_g, dtype=torch.float32, device=row.device)
    mat[row, col] = 1.0
    return mat


def get_node_counts(data: HeteroData) -> Dict[str, int]:
    return {ntype: data[ntype].x.shape[0] for ntype in data.node_types if hasattr(data[ntype], "x") and data[ntype].x is not None}


def validate_split_disjointness(
    train_data: HeteroData,
    test_data:  HeteroData,
    target_type: str,
    src_type:   str = "Protein",
    relation:   str = "protein_function",
) -> bool:
    """
    Confirm that no test protein appears in training protein→GO annotation edges.

    WHY THIS MATTERS:
        ProtHGT uses a cold-start protein-level split: test proteins must be
        entirely absent from training annotations.  If test proteins appear in
        train edges, the encoder has seen their labels — Fmax numbers are inflated
        and the comparison with ProtHGT's published numbers is invalid.

    Raises ValueError listing the overlap size if leakage is detected.
    Prints "split OK" if the split is clean.
    Returns True on success (so it can be used in assertions).
    """
    train_row, _, _, _ = build_annotation_matrix(train_data, target_type, src_type, relation)
    test_row,  _, _, _ = build_annotation_matrix(test_data,  target_type, src_type, relation)

    train_prots = set(train_row.cpu().tolist())
    test_prots  = set(test_row.cpu().tolist())
    overlap     = train_prots & test_prots

    if overlap:
        raise ValueError(
            f"Split leakage detected for {target_type}: "
            f"{len(overlap):,} test proteins also appear in train annotations "
            f"(e.g. indices {sorted(overlap)[:5]}...). "
            f"Evaluation numbers would be inflated."
        )

    print(f"  {target_type}: split OK — "
          f"{len(train_prots):,} train proteins / {len(test_prots):,} test proteins / 0 overlap")
    return True
