"""
scripts/compute_perrepo_scaler_v2.py

Compute per-repo mean/std for continuous numeric graph features from
training graphs built by scripts/build_graphs_v2.py.

Updated feature indices for graph_ready_v2 node/edge schemas:

  commit_node  (14 dims):  idxs [2,3,4,5,6,7,8,9,10,12,13]
                            (DMM×3, tz×2, sin/cos×4, repo_commits_90d, repo_active_authors_90d)
                            excluding binary at 0,1,11
  fn_node      (776 dims): idxs [0,1,2,3,4,5,6,7]  (8 normalized numeric; skip 768-dim embedding)
  file_node      (3 dims): idxs [0,1,2]
  hunk_node    (770 dims): idxs [0,1]               (2 numeric; skip 768-dim embedding)
  dev_node       (9 dims): idxs [0,1,2,3,4,5,6,7,8]
  issue_node     (4 dims): idxs [0,1,2,3]
  pr_node        (4 dims): idxs [0,1,2,3]
  tag_node       (4 dims): idxs [0,1,2,3]
  commit→file   (4 dims):  idxs [0,1,2,3]
  commit→func  (11 dims):  idxs [0,1,2,3,4,5]  (continuous; skip one-hot at 6-10)
  author edge   (3 dims):  idxs [0,1]           (skip binary at 2)
  committer edge(3 dims):  idxs [0,1]
  owns edge     (3 dims):  idxs [0,1,2]
  issue edge    (3 dims):  idxs [0,1]           (skip binary at 2)
  pr edge       (3 dims):  idxs [0,1]
  tag edge      (1 dim):   idxs [0]

Usage:
    python scripts/compute_perrepo_scaler_v2.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT       = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "outputs" / "graph_ready_v2" / "graphs"
SPLIT_IDX  = ROOT / "outputs" / "graph_ready_v2" / "split_index.csv"
OUT        = ROOT / "outputs" / "graph_ready_v2" / "perrepo_scaler_v2.json"
MIN_SAMPLES = 10

GROUPS: dict[str, dict] = {
    "commit_node": {
        "kind": "node", "target": "commit",
        "idxs": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13],
    },
    "fn_node": {
        "kind": "node", "target": "function",
        "idxs": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "file_node": {
        "kind": "node", "target": "file",
        "idxs": [0, 1, 2],
    },
    "hunk_node": {
        "kind": "node", "target": "hunk",
        "idxs": [0, 1],
    },
    "dev_node": {
        "kind": "node", "target": "developer",
        "idxs": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    },
    "issue_node": {
        "kind": "node", "target": "issue",
        "idxs": [0, 1, 2, 3],
    },
    "pr_node": {
        "kind": "node", "target": "pull_request",
        "idxs": [0, 1, 2, 3],
    },
    "tag_node": {
        "kind": "node", "target": "release_tag",
        "idxs": [0, 1, 2, 3],
    },
    "commit_file_edge": {
        "kind": "edge", "target": ("commit", "modifies_file", "file"),
        "idxs": [0, 1, 2, 3],
    },
    "commit_fn_edge": {
        "kind": "edge", "target": ("commit", "modifies_func", "function"),
        "idxs": [0, 1, 2, 3, 4, 5],  # continuous only; skip fct_* one-hot at 6-10
    },
    "author_edge": {
        "kind": "edge", "target": ("commit", "authored_by", "developer"),
        "idxs": [0, 1],  # skip binary dev_is_new_contributor at 2
    },
    "committer_edge": {
        "kind": "edge", "target": ("commit", "committed_by", "developer"),
        "idxs": [0, 1],
    },
    "owns_edge": {
        "kind": "edge", "target": ("developer", "owns", "file"),
        "idxs": [0, 1, 2],
    },
    "issue_edge": {
        "kind": "edge", "target": ("commit", "has_issue", "issue"),
        "idxs": [0, 1],  # skip binary has_issue_pr_gap at 2
    },
    "pr_edge": {
        "kind": "edge", "target": ("commit", "has_pr", "pull_request"),
        "idxs": [0, 1],
    },
    "tag_edge": {
        "kind": "edge", "target": ("commit", "has_release", "release_tag"),
        "idxs": [0],
    },
}


def _extract(data, spec: dict) -> np.ndarray:
    idxs  = spec["idxs"]
    idx_t = torch.tensor(idxs, dtype=torch.long)
    if spec["kind"] == "node":
        tensor = data[spec["target"]].x
    else:
        tensor = getattr(data[spec["target"]], "edge_attr", None)
    if tensor is None or tensor.numel() == 0 or tensor.size(1) <= max(idxs):
        return np.zeros((0, len(idxs)), dtype=np.float32)
    return tensor[:, idx_t].detach().cpu().float().numpy()


def main() -> None:
    split = pd.read_csv(SPLIT_IDX, usecols=["hash", "repo_url", "repo_split"])
    train = split[split["repo_split"] == "train"].copy()
    print(f"Train: {len(train):,} commits, {train['repo_url'].nunique()} repos")

    sums: dict[str, dict[str, np.ndarray]]   = {}
    sumsqs: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, dict[str, int]]        = {}
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
            arr = _extract(data, spec)
            if arr.size == 0:
                continue
            finite = np.isfinite(arr).all(axis=1)
            arr = arr[finite]
            if arr.size == 0:
                continue
            sums.setdefault(group,   {}); sumsqs.setdefault(group, {}); counts.setdefault(group, {})
            if repo not in sums[group]:
                sums[group][repo]   = np.zeros(arr.shape[1], dtype=np.float64)
                sumsqs[group][repo] = np.zeros(arr.shape[1], dtype=np.float64)
                counts[group][repo] = 0
            sums[group][repo]   += arr.sum(axis=0).astype(np.float64)
            sumsqs[group][repo] += np.square(arr.astype(np.float64)).sum(axis=0)
            counts[group][repo] += int(arr.shape[0])

    result: dict[str, dict] = {}
    for group in GROUPS:
        result[group] = {}
        for repo, n in counts.get(group, {}).items():
            if n < MIN_SAMPLES:
                continue
            mean = sums[group][repo] / n
            var  = np.maximum(sumsqs[group][repo] / n - np.square(mean), 1e-12)
            std  = np.sqrt(var)
            result[group][repo] = {
                "mean":  mean.astype(np.float32).tolist(),
                "std":   std.astype(np.float32).tolist(),
                "count": int(n),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Skipped (missing graphs): {skipped}")
    for group in GROUPS:
        print(f"  {group:20s}  {len(result.get(group,{})):4d} repos")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
