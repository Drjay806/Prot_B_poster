import os
import pickle
from collections import defaultdict, deque
from typing import Dict, FrozenSet, Optional

from torch_geometric.data import HeteroData

CACHE_PATH = "/tmp/ancestor_table.pkl"

# In ProtHGT, GO hierarchy edges use the relation name "function_function"
HIERARCHY_RELATIONS = {"function_function"}


def build_ancestor_table(
    data: HeteroData,
    target_type: str = "GO_term_P",
    cache: bool = True,
    cache_path: str = CACHE_PATH,
) -> Dict[int, FrozenSet[int]]:
    """
    Build a mapping from each GO term index to the frozenset of all its transitive ancestors.

    The ancestor set is derived from the hierarchy edges inside the ProtHGT graph
    (e.g. `("GO_term_P", "is_a", "GO_term_P")`).

    Results are cached to `cache_path` so repeated Colab cell runs are fast.
    """
    if cache and os.path.exists(cache_path):
        print(f"Loading ancestor table from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    child_to_parents: Dict[int, list] = defaultdict(list)
    num_go = data[target_type].x.shape[0]
    found_edges = False

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        if src_type == target_type and dst_type == target_type and rel in HIERARCHY_RELATIONS:
            edge_index = data[edge_type].edge_index
            for child, parent in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                child_to_parents[child].append(parent)
            found_edges = True

    if not found_edges:
        # Fallback: try any self-loop edge type on the GO node type
        for edge_type in data.edge_types:
            src_type, rel, dst_type = edge_type
            if src_type == target_type and dst_type == target_type:
                edge_index = data[edge_type].edge_index
                for child, parent in zip(edge_index[0].tolist(), edge_index[1].tolist()):
                    child_to_parents[child].append(parent)
                found_edges = True
                print(f"  Using fallback hierarchy relation: {edge_type}")

    print(f"Building ancestor table for {num_go:,} GO terms ({target_type}) ...")

    ancestor_table: Dict[int, FrozenSet[int]] = {}
    for node in range(num_go):
        visited: set = set()
        queue: deque = deque(child_to_parents.get(node, []))
        while queue:
            parent = queue.popleft()
            if parent not in visited:
                visited.add(parent)
                queue.extend(child_to_parents.get(parent, []))
        ancestor_table[node] = frozenset(visited)

    print(f"Done. Avg ancestors per GO term: {sum(len(v) for v in ancestor_table.values()) / max(num_go,1):.1f}")

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(ancestor_table, f)
        print(f"Ancestor table cached to {cache_path}")

    return ancestor_table


def propagate_predictions(
    pred_matrix,          # [num_proteins, num_go] float tensor
    ancestor_table: Dict[int, FrozenSet[int]],
) -> object:             # returns same tensor type
    """
    Propagate prediction scores up the GO hierarchy (true-path rule).
    For every GO term g, every ancestor of g receives score = max(ancestor_score, g_score).
    This runs in-place on a clone to avoid modifying the original.
    """
    import torch
    prop = pred_matrix.clone()
    for go_idx, ancestors in ancestor_table.items():
        if not ancestors:
            continue
        anc_list = list(ancestors)
        # Broadcast: each ancestor takes the max of its own score and this child's score
        prop[:, anc_list] = torch.max(prop[:, anc_list], pred_matrix[:, go_idx].unsqueeze(1))
    return prop
