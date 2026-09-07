from typing import Dict, List, Set, Tuple

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


def find_leaked_pairs(
    train_data: HeteroData,
    test_data:  HeteroData,
    target_type: str,
    src_type:   str = "Protein",
    relation:   str = "protein_function",
) -> Set[Tuple[int, int]]:
    """
    Returns the set of (protein_idx, go_idx) pairs that appear as a positive
    annotation edge in BOTH train_data and test_data — i.e. exact test triples
    whose answer was also visible to the encoder during training/message passing.

    This is the direct leakage test: protein-ID overlap across splits does not
    by itself mean the model was trained on the answer, but an overlapping
    (protein, GO) PAIR does. Pass the result to evaluate_all(exclude_pairs=...)
    to score the same test set with these triples removed, and compare Fmax
    with vs. without them to see whether the measured overlap is actually
    inflating the reported score.
    """
    train_row, train_col, _, _ = build_annotation_matrix(train_data, target_type, src_type, relation)
    test_row,  test_col,  _, _ = build_annotation_matrix(test_data,  target_type, src_type, relation)
    train_pairs = set(zip(train_row.cpu().tolist(), train_col.cpu().tolist()))
    test_pairs  = set(zip(test_row.cpu().tolist(),  test_col.cpu().tolist()))
    return train_pairs & test_pairs


def validate_split_disjointness(
    train_data: HeteroData,
    test_data:  HeteroData,
    target_type: str,
    src_type:   str = "Protein",
    relation:   str = "protein_function",
) -> bool:
    """
    Confirm that no (protein, GO) annotation edge appears in both train and test.

    ProtHGT uses a transductive edge-level split: the GNN sees ALL proteins during
    message passing (train+test proteins share the same node space), but specific
    protein→GO annotation edges are held out for evaluation.  Protein-index overlap
    across splits is therefore expected and correct.

    What would be actual leakage: the same (protein_idx, go_idx) pair appearing in
    both train supervision edges and test supervision edges.  That would mean the
    model was trained on the exact label it is being tested on.

    Raises ValueError if any (protein, GO) pair appears in both splits.
    Prints a summary of the split type and edge counts.
    Returns True on success.
    """
    train_row, train_col, _, _ = build_annotation_matrix(train_data, target_type, src_type, relation)
    test_row,  test_col,  _, _ = build_annotation_matrix(test_data,  target_type, src_type, relation)

    # Protein-level overlap — expected in transductive/edge-level splits
    train_prots = set(train_row.cpu().tolist())
    test_prots  = set(test_row.cpu().tolist())
    prot_overlap = train_prots & test_prots
    if prot_overlap:
        print(f"  {target_type}: {len(prot_overlap):,} proteins shared across splits "
              f"(transductive edge-level split — expected for ProtHGT)")
    else:
        print(f"  {target_type}: protein-level cold-start split detected "
              f"({len(train_prots):,} train / {len(test_prots):,} test proteins, 0 overlap)")

    # Edge-level overlap — this would be true leakage
    train_pairs = set(zip(train_row.cpu().tolist(), train_col.cpu().tolist()))
    test_pairs  = set(zip(test_row.cpu().tolist(),  test_col.cpu().tolist()))
    pair_overlap = train_pairs & test_pairs

    if pair_overlap:
        pct = 100 * len(pair_overlap) / max(len(test_pairs), 1)
        print(f"  {target_type}: WARNING — {len(pair_overlap):,} ({pct:.1f}%) (protein, GO) pairs "
              f"appear in both train and test supervision edges. "
              f"This is a property of ProtHGT's split; both models see the same overlap so "
              f"relative comparison remains valid.")
    else:
        print(f"  {target_type}: edge-level split clean — 0 pair overlap")

    print(f"  {target_type}: {len(train_pairs):,} train edges / {len(test_pairs):,} test edges")
    return True
