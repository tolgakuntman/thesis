"""
scripts/compute_perrepo_scaler.py

Compute per-repo mean/std for continuous numeric graph features from the built
training graphs in outputs/final_graph_ready/graphs.

This is intentionally training-only for repo_split / temporal_split hygiene:
held-out repos must not contribute their own normalization statistics.

All graph tensors are already globally normalized at build time where
applicable. The loader then applies:
    z_repo = (z_global - repo_mean_of_z) / repo_std_of_z

Only continuous numeric features are included. Binary indicators and one-hot
change-type features are excluded because per-repo z-scoring them is not a
meaningful intervention.

Usage:
    conda run -n thesis python scripts/compute_perrepo_scaler.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "outputs" / "final_graph_ready" / "graphs"
SPLIT_INDEX = ROOT / "outputs" / "final_graph_ready" / "split_index.csv"
OUT = ROOT / "outputs" / "final_graph_ready" / "perrepo_function_scaler.json"
MIN_SAMPLES = 10

GROUPS: dict[str, dict[str, object]] = {
    "commit_node": {
        "kind": "node",
        "target": "commit",
        "idxs": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
    "fn_node": {
        "kind": "node",
        "target": "function",
        "idxs": [0, 1, 2, 3, 4],
    },
    "file_node": {
        "kind": "node",
        "target": "file",
        "idxs": [0, 1, 2],
    },
    "hunk_node": {
        "kind": "node",
        "target": "hunk",
        "idxs": [1, 2],
    },
    "dev_node": {
        "kind": "node",
        "target": "developer",
        "idxs": [0, 1, 2, 3, 4],
    },
    "issue_node": {
        "kind": "node",
        "target": "issue",
        "idxs": [0, 1, 2, 3],
    },
    "pr_node": {
        "kind": "node",
        "target": "pull_request",
        "idxs": [0, 1, 2, 3],
    },
    "tag_node": {
        "kind": "node",
        "target": "release_tag",
        "idxs": [0, 1, 2, 3],
    },
    "commit_file_edge": {
        "kind": "edge",
        "target": ("commit", "modifies_file", "file"),
        "idxs": [0, 1, 2, 3],
    },
    "commit_fn_edge": {
        "kind": "edge",
        "target": ("commit", "modifies_func", "function"),
        "idxs": [0, 1, 2, 3, 4, 5],
    },
    "author_edge": {
        "kind": "edge",
        "target": ("commit", "authored_by", "developer"),
        "idxs": [0, 1],
    },
    "committer_edge": {
        "kind": "edge",
        "target": ("commit", "committed_by", "developer"),
        "idxs": [0, 1],
    },
    "owns_edge": {
        "kind": "edge",
        "target": ("developer", "owns", "file"),
        "idxs": [0, 1, 2],
    },
    "issue_edge": {
        "kind": "edge",
        "target": ("commit", "has_issue", "issue"),
        "idxs": [0, 1],
    },
    "pr_edge": {
        "kind": "edge",
        "target": ("commit", "has_pr", "pull_request"),
        "idxs": [0, 1],
    },
    "tag_edge": {
        "kind": "edge",
        "target": ("commit", "has_release", "release_tag"),
        "idxs": [0],
    },
}


def _extract_group(data, spec: dict[str, object]) -> np.ndarray:
    idxs = spec["idxs"]
    idx_t = torch.tensor(idxs, dtype=torch.long)
    if spec["kind"] == "node":
        tensor = data[spec["target"]].x
    else:
        tensor = getattr(data[spec["target"]], "edge_attr", None)
    if tensor is None or tensor.numel() == 0 or tensor.size(1) <= max(idxs):
        return np.zeros((0, len(idxs)), dtype=np.float32)
    return tensor[:, idx_t].detach().cpu().float().numpy()


def main() -> None:
    split = pd.read_csv(SPLIT_INDEX, usecols=["hash", "repo_url", "repo_split"])
    train = split[split["repo_split"] == "train"].copy()
    print(f"Train split: {len(train):,} commits across {train['repo_url'].nunique()} repos")

    sums: dict[str, dict[str, np.ndarray]] = {}
    sumsqs: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, dict[str, int]] = {}
    skipped = 0

    for row in train.itertuples(index=False):
        graph_path = GRAPHS_DIR / f"{row.hash}.pt"
        if not graph_path.exists():
            skipped += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data = torch.load(graph_path, weights_only=False)
        repo = str(row.repo_url)
        for group, spec in GROUPS.items():
            arr = _extract_group(data, spec)
            if arr.size == 0:
                continue
            finite = np.isfinite(arr).all(axis=1)
            arr = arr[finite]
            if arr.size == 0:
                continue
            sums.setdefault(group, {})
            sumsqs.setdefault(group, {})
            counts.setdefault(group, {})
            if repo not in sums[group]:
                sums[group][repo] = np.zeros(arr.shape[1], dtype=np.float64)
                sumsqs[group][repo] = np.zeros(arr.shape[1], dtype=np.float64)
                counts[group][repo] = 0
            sums[group][repo] += arr.sum(axis=0, dtype=np.float64)
            sumsqs[group][repo] += np.square(arr, dtype=np.float64).sum(axis=0, dtype=np.float64)
            counts[group][repo] += int(arr.shape[0])

    result: dict[str, dict[str, dict[str, object]]] = {}
    for group in GROUPS:
        result[group] = {}
        for repo, n in counts.get(group, {}).items():
            if n < MIN_SAMPLES:
                continue
            mean = sums[group][repo] / n
            var = np.maximum(sumsqs[group][repo] / n - np.square(mean), 1e-12)
            std = np.sqrt(var)
            result[group][repo] = {
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
                "count": int(n),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Skipped missing graphs: {skipped}")
    for group in GROUPS:
        print(f"{group:16s} {len(result.get(group, {})):4d} repos")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
