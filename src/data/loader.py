import os
from typing import NamedTuple, Optional

import torch
from torch_geometric.data import HeteroData


ONTOLOGY_TO_NODE = {
    "bp": "GO_term_P",
    "mf": "GO_term_F",
    "cc": "GO_term_C",
}

ESM2_SUBDIR = "alternative_protein_embeddings/esm2"

SPLIT_FILES = {
    "train": "prothgt-esm2-train-graph.pt",
    "val":   "prothgt-esm2-val-graph.pt",
    "test":  "prothgt-esm2-test-graph.pt",
}


class ProtHGTSplits(NamedTuple):
    train: HeteroData
    val: HeteroData
    test: HeteroData
    target_type: str


def load_prothgt_splits(
    drive_root: str,
    ontology: str = "bp",
    device: str = "cpu",
) -> ProtHGTSplits:
    """
    Load the three ProtHGT ESM2 split .pt files from Google Drive.

    Args:
        drive_root: Path to the knowledge_graphs/ folder in Drive, e.g.
            "/content/drive/MyDrive/Poster code/prothgt/knowledge_graphs/"
        ontology: "bp", "mf", or "cc"
        device: "cuda" or "cpu"

    Returns:
        ProtHGTSplits named tuple with train/val/test HeteroData objects and target node type.
    """
    target_type = ONTOLOGY_TO_NODE[ontology.lower()]
    esm2_root = os.path.join(drive_root, ESM2_SUBDIR)

    splits = {}
    for split_name, filename in SPLIT_FILES.items():
        path = os.path.join(esm2_root, filename)
        if not os.path.exists(path):
            if split_name == "test":
                print(f"WARNING: Test split not found at {path} — evaluation will use val split as proxy.")
                splits["test"] = splits.get("val")
                continue
            raise FileNotFoundError(
                f"Expected {path}\n"
                f"Check that drive_root points to the knowledge_graphs/ folder and the ESM2 files are present.\n"
                f"Files should be named: prothgt-esm2-train-graph.pt, prothgt-esm2-val-graph.pt, prothgt-esm2-test-graph.pt"
            )
        print(f"Loading {split_name} split from {path} ...")
        data: HeteroData = torch.load(path, map_location=device, weights_only=False)
        _validate_split(data, split_name, target_type)
        splits[split_name] = data

    print(
        f"Loaded ProtHGT ESM2 splits  —  "
        f"proteins: {splits['train']['Protein'].x.shape[0]:,}  "
        f"GO terms ({ontology.upper()}): {splits['train'][target_type].x.shape[0]:,}"
    )
    return ProtHGTSplits(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        target_type=target_type,
    )


def _validate_split(data: HeteroData, split_name: str, target_type: str):
    assert "Protein" in data.node_types, f"{split_name}: 'Protein' node type missing"
    assert target_type in data.node_types, f"{split_name}: '{target_type}' node type missing"

    protein_dim = data["Protein"].x.shape[-1]
    go_dim = data[target_type].x.shape[-1]
    assert protein_dim == 1280, f"{split_name}: Expected protein dim 1280, got {protein_dim} (are you using the ESM2 files?)"
    assert go_dim == 200, f"{split_name}: Expected GO dim 200 (anc2vec), got {go_dim}"
