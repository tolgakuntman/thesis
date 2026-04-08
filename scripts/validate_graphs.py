"""
Random validation check on built graphs in data_new/graph_ready/graphs/
Samples 10 VCC, 10 FC, 10 normal commits and validates each graph.
"""

import os
import sys
import random
import torch
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/Users/tolgakuntman/Desktop/groupt/master thesis/thesis")
GRAPH_DIR = BASE / "data_new/graph_ready/graphs"
COMMIT_INFO = BASE / "data/graph_data/commit_info_full.csv"

random.seed(42)

# ─── expected shapes ───────────────────────────────────────────────────────────
EXPECTED_SHAPES = {
    "commit":       {"feat_dim": 777, "min_nodes": 1, "max_nodes": 1},
    "file":         {"feat_dim": 10,  "min_nodes": 1, "max_nodes": None},
    "function":     {"feat_dim": 10,  "min_nodes": 0, "max_nodes": None},
    "developer":    {"feat_dim": 6,   "min_nodes": 0, "max_nodes": None},
    "issue":        {"feat_dim": 5,   "min_nodes": 0, "max_nodes": 1},
    "pull_request": {"feat_dim": 5,   "min_nodes": 0, "max_nodes": 1},
    "release_tag":  {"feat_dim": 4,   "min_nodes": 0, "max_nodes": 2},
}

# ─── load commit info ──────────────────────────────────────────────────────────
print("Loading commit_info_full.csv ...")
df = pd.read_csv(COMMIT_INFO, usecols=["hash", "commit_type"])

# Only keep hashes that have a corresponding .pt file
available = set(p.stem for p in GRAPH_DIR.glob("*.pt"))
df = df[df["hash"].isin(available)].copy()
print(f"  Total rows in CSV: {len(pd.read_csv(COMMIT_INFO))}  |  Graphs on disk: {len(available)}  |  Matched: {len(df)}")

vccs    = df[df.commit_type == "VCC"]["hash"].tolist()
fcs     = df[df.commit_type == "FC"]["hash"].tolist()
normals = df[df.commit_type == "normal"]["hash"].tolist()

print(f"  VCC: {len(vccs)}  FC: {len(fcs)}  normal: {len(normals)}")

n_sample = 10
sampled_vccs    = random.sample(vccs,    min(n_sample, len(vccs)))
sampled_fcs     = random.sample(fcs,     min(n_sample, len(fcs)))
sampled_normals = random.sample(normals, min(n_sample, len(normals)))

sample = (
    [(h, "VCC",    1) for h in sampled_vccs]
  + [(h, "FC",     0) for h in sampled_fcs]
  + [(h, "normal", 0) for h in sampled_normals]
)

print(f"\nSampled {len(sample)} graphs ({len(sampled_vccs)} VCC, {len(sampled_fcs)} FC, {len(sampled_normals)} normal)\n")
print("=" * 90)

# ─── per-type stats accumulators ──────────────────────────────────────────────
stats = {"VCC": [], "FC": [], "normal": []}
failures = []

def check_graph(hash_, commit_type, expected_label):
    pt_path = GRAPH_DIR / f"{hash_}.pt"
    issues = []

    # 1. Load
    try:
        data = torch.load(pt_path, weights_only=False)
    except Exception as e:
        return {"hash": hash_, "type": commit_type, "fatal": f"LOAD FAILED: {e}", "stats": None}

    # 2. PyG validate
    try:
        data.validate(raise_on_error=True)
    except Exception as e:
        issues.append(f"data.validate() FAILED: {e}")

    # 3. Node shape checks
    node_counts = {}
    for ntype, spec in EXPECTED_SHAPES.items():
        if ntype not in data.node_types:
            if spec["min_nodes"] > 0:
                issues.append(f"MISSING node type: {ntype} (min_nodes={spec['min_nodes']})")
            node_counts[ntype] = 0
            continue

        x = data[ntype].x
        if x is None:
            issues.append(f"{ntype}.x is None")
            node_counts[ntype] = 0
            continue

        n_nodes, feat_dim = x.shape
        node_counts[ntype] = n_nodes

        # feature dim
        if feat_dim != spec["feat_dim"]:
            issues.append(f"{ntype}: feat_dim={feat_dim}, expected {spec['feat_dim']}")

        # node count bounds
        if spec["min_nodes"] is not None and n_nodes < spec["min_nodes"]:
            issues.append(f"{ntype}: n_nodes={n_nodes} < min={spec['min_nodes']}")
        if spec["max_nodes"] is not None and n_nodes > spec["max_nodes"]:
            issues.append(f"{ntype}: n_nodes={n_nodes} > max={spec['max_nodes']}")

        # dtype
        if x.dtype != torch.float32:
            issues.append(f"{ntype}.x dtype={x.dtype}, expected float32")

        # NaN / Inf (only commit node flagged as error; others as warning)
        if ntype == "commit":
            if torch.isnan(x).any():
                issues.append(f"commit.x contains NaN values")
            if torch.isinf(x).any():
                issues.append(f"commit.x contains Inf values")

    # 4. Edge checks
    total_edges = 0
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        if edge_index is None:
            issues.append(f"edge {edge_type}: edge_index is None")
            continue

        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            issues.append(f"edge {edge_type}: bad shape {edge_index.shape}")
            continue

        n_edges = edge_index.shape[1]
        total_edges += n_edges

        if n_edges == 0:
            continue  # empty edge list is fine

        n_src = node_counts.get(src_type, 0)
        n_dst = node_counts.get(dst_type, 0)

        if n_src == 0:
            issues.append(f"edge {edge_type}: src type '{src_type}' has 0 nodes but {n_edges} edges")
        elif edge_index[0].max().item() >= n_src:
            issues.append(f"edge {edge_type}: src idx out-of-bounds (max={edge_index[0].max().item()} >= n_src={n_src})")

        if n_dst == 0:
            issues.append(f"edge {edge_type}: dst type '{dst_type}' has 0 nodes but {n_edges} edges")
        elif edge_index[1].max().item() >= n_dst:
            issues.append(f"edge {edge_type}: dst idx out-of-bounds (max={edge_index[1].max().item()} >= n_dst={n_dst})")

    # 5. Label check
    if hasattr(data, "y") and data.y is not None:
        actual_label = int(data.y.item()) if data.y.numel() == 1 else int(data.y[0].item())
        if actual_label != expected_label:
            issues.append(f"LABEL MISMATCH: y={actual_label}, expected={expected_label} (commit_type={commit_type})")
    else:
        issues.append("No label (data.y is missing or None)")

    # 6. repo_url check
    repo_url = None
    if hasattr(data["commit"], "repo_url"):
        repo_url = data["commit"].repo_url
        if not repo_url:
            issues.append("commit.repo_url is empty/falsy")
    else:
        repo_url = "MISSING"
        # Not flagging as error here; just reporting

    return {
        "hash": hash_,
        "type": commit_type,
        "issues": issues,
        "stats": {
            "n_files":   node_counts.get("file", 0),
            "n_funcs":   node_counts.get("function", 0),
            "n_devs":    node_counts.get("developer", 0),
            "n_tags":    node_counts.get("release_tag", 0),
            "n_edges":   total_edges,
            "commit_feat_dim": node_counts.get("commit_feat_dim", None),
        },
        "repo_url": repo_url,
        "label_ok": len([i for i in issues if "LABEL" in i]) == 0,
    }

# ─── run checks ───────────────────────────────────────────────────────────────
results = []
for hash_, ctype, exp_label in sample:
    r = check_graph(hash_, ctype, exp_label)
    results.append(r)
    ok_str = "OK" if not r.get("issues") and not r.get("fatal") else "FAIL"
    print(f"  [{ok_str}] {ctype:6s} {hash_[:12]}...", end="")
    if r.get("fatal"):
        print(f"  FATAL: {r['fatal']}")
    elif r.get("issues"):
        print(f"  {len(r['issues'])} issue(s)")
        for iss in r["issues"]:
            print(f"           -> {iss}")
    else:
        s = r["stats"]
        repo = r.get("repo_url", "?")
        print(f"  files={s['n_files']} funcs={s['n_funcs']} devs={s['n_devs']} tags={s['n_tags']} edges={s['n_edges']}  repo_url={repr(repo)}")

# ─── summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY TABLE  (per commit_type  |  n_files / n_funcs / n_devs / n_tags / n_edges_total)")
print("=" * 90)
print(f"{'type':>8} | {'metric':>10} | {'min':>8} | {'max':>8} | {'avg':>8}")
print("-" * 60)

for ctype in ["VCC", "FC", "normal"]:
    type_results = [r for r in results if r["type"] == ctype and r.get("stats")]
    if not type_results:
        print(f"  {ctype}: no valid results")
        continue
    for key in ["n_files", "n_funcs", "n_devs", "n_tags", "n_edges"]:
        vals = [r["stats"][key] for r in type_results]
        print(f"{ctype:>8} | {key:>10} | {min(vals):>8} | {max(vals):>8} | {sum(vals)/len(vals):>8.1f}")
    print("-" * 60)

# ─── failures summary ─────────────────────────────────────────────────────────
all_failed = [r for r in results if r.get("issues") or r.get("fatal")]
print(f"\n{'=' * 90}")
if not all_failed:
    print("ALL 30 GRAPHS PASSED ALL CHECKS.")
else:
    print(f"FAILURES: {len(all_failed)} graph(s) had issues:")
    for r in all_failed:
        print(f"\n  Hash: {r['hash']}  Type: {r['type']}")
        if r.get("fatal"):
            print(f"    FATAL: {r['fatal']}")
        for iss in r.get("issues", []):
            print(f"    -> {iss}")

# ─── repo_url spot check ──────────────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("repo_url SPOT CHECK (first 5 graphs):")
for r in results[:5]:
    print(f"  {r['hash'][:12]}... ({r['type']:6s}): repo_url = {repr(r.get('repo_url', 'NOT CHECKED'))}")

print("\nDone.")
