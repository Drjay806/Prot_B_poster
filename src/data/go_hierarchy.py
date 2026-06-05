import os
import pickle
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from torch_geometric.data import HeteroData

CACHE_PATH       = "/tmp/ancestor_table.pkl"
PROP_CACHE_PATH  = "/tmp/prop_edges_{}.pkl"

# In ProtHGT, GO hierarchy edges use the relation name "function_function"
HIERARCHY_RELATIONS = {"function_function"}


def build_ancestor_table(
    data: HeteroData,
    target_type: str = "GO_term_P",
    cache: bool = True,
    cache_path: str = CACHE_PATH,
) -> Dict[int, FrozenSet[int]]:
    """
    Build a mapping from each GO term to the frozenset of ALL its transitive ancestors.
    Used by the RL hierarchy penalty (needs the full ancestor set per GO term).
    Cached to disk so repeated Colab cell re-runs are instant.
    """
    if cache and os.path.exists(cache_path):
        print(f"Loading ancestor table from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    child_to_parents: Dict[int, list] = defaultdict(list)
    num_go = data[target_type].x.shape[0]

    _load_hierarchy_edges(data, target_type, child_to_parents)

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

    avg = sum(len(v) for v in ancestor_table.values()) / max(num_go, 1)
    print(f"Done. Avg ancestors per GO term: {avg:.1f}")

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(ancestor_table, f)
        print(f"Ancestor table cached to {cache_path}")

    return ancestor_table


def build_propagation_edges(
    data: HeteroData,
    target_type: str = "GO_term_P",
    cache: bool = True,
) -> List[Tuple[int, int]]:
    """
    Build a list of (child_idx, parent_idx) DIRECT hierarchy edges sorted in
    topological order (all children of a node appear before the node itself).

    This is used by propagate_predictions so it can do one efficient pass over
    direct edges instead of iterating over the full transitive ancestor sets,
    which would require huge intermediate tensors.
    """
    cache_path = PROP_CACHE_PATH.format(target_type)
    if cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    child_to_parents: Dict[int, list] = defaultdict(list)
    parent_to_children: Dict[int, list] = defaultdict(list)

    _load_hierarchy_edges(data, target_type, child_to_parents, build_reverse=True,
                          reverse_dict=parent_to_children)
    all_nodes: set = set(child_to_parents.keys()) | set(parent_to_children.keys())

    # Kahn's topological sort — children first (leaves start, roots finish)
    # child_count[node] = number of its children not yet visited
    child_count: Dict[int, int] = defaultdict(int)
    for parent, children in parent_to_children.items():
        child_count[parent] = len(children)

    queue: deque = deque(sorted(n for n in all_nodes if child_count[n] == 0))
    topo_order: List[int] = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for parent in child_to_parents.get(node, []):
            child_count[parent] -= 1
            if child_count[parent] == 0:
                queue.append(parent)

    node_rank = {n: i for i, n in enumerate(topo_order)}

    # Collect all direct (child, parent) pairs and sort by child's topo rank
    edges: List[Tuple[int, int]] = []
    for child, parents in child_to_parents.items():
        for parent in parents:
            edges.append((child, parent))
    edges.sort(key=lambda e: node_rank.get(e[0], 0))

    print(f"Built {len(edges):,} propagation edges in topological order ({target_type})")

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(edges, f)

    return edges


def _load_hierarchy_edges(
    data: HeteroData,
    target_type: str,
    child_to_parents: Dict[int, list],
    build_reverse: bool = False,
    reverse_dict: Optional[Dict[int, list]] = None,
) -> None:
    found = False
    for src, rel, dst in data.edge_types:
        if src == target_type and dst == target_type and rel in HIERARCHY_RELATIONS:
            ei = data[(src, rel, dst)].edge_index
            for child, parent in zip(ei[0].tolist(), ei[1].tolist()):
                if child != parent:
                    child_to_parents[child].append(parent)
                    if build_reverse and reverse_dict is not None:
                        reverse_dict[parent].append(child)
            found = True
    if not found:
        # Fallback: any self-loop edge on the GO node type
        for src, rel, dst in data.edge_types:
            if src == target_type and dst == target_type:
                ei = data[(src, rel, dst)].edge_index
                for child, parent in zip(ei[0].tolist(), ei[1].tolist()):
                    if child != parent:
                        child_to_parents[child].append(parent)
                        if build_reverse and reverse_dict is not None:
                            reverse_dict[parent].append(child)
                print(f"  Hierarchy fallback: using relation '{rel}'")
                found = True
                break


def propagate_predictions(
    pred_matrix,                           # [N_proteins, N_go] float tensor (CPU)
    prop_edges: List[Tuple[int, int]],     # (child, parent) in topological order
):
    """
    Propagate scores up the GO hierarchy using direct parent edges in topological order.
    One pass over the ~64k edges is sufficient because topological order guarantees
    all descendants are processed before ancestors.

    Memory cost: only [N_proteins] vectors at a time — no [N_p × N_go] intermediates.
    """
    import torch
    prop = pred_matrix.clone()
    for child_idx, parent_idx in prop_edges:
        torch.maximum(prop[:, parent_idx], prop[:, child_idx], out=prop[:, parent_idx])
    return prop
