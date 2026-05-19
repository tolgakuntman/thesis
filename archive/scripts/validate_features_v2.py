"""
scripts/validate_features_v2.py

Validate node and edge feature values across a sample of built graphs.
Checks dimensions, ranges, binary constraints, embedding norms, and one-hot validity.
"""
from __future__ import annotations
import warnings, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = ROOT / "outputs" / "graph_ready_v2" / "graphs"

N_SAMPLE = 500
SEED = 42
random.seed(SEED)

# Expected dims
EXPECTED_NODE_DIMS = {
    "commit": 14,
    "function": 776,
    "file": 3,
    "hunk": 770,
    "developer": 9,
    "issue": 4,
    "pull_request": 4,
    "release_tag": 4,
}
EXPECTED_EDGE_DIMS = {
    ("commit", "modifies_file",  "file"):         4,
    ("commit", "modifies_func",  "function"):    11,
    ("commit", "authored_by",    "developer"):    3,
    ("commit", "committed_by",   "developer"):    3,
    ("developer", "owns",        "file"):         3,
    ("commit", "has_issue",      "issue"):        3,
    ("commit", "has_pr",         "pull_request"): 3,
    ("commit", "has_release",    "release_tag"):  1,
}

COMMIT_FEAT_NAMES = [
    "in_main_branch",          # 0  binary
    "is_merge",                # 1  binary
    "dmm_unit_complexity",     # 2  log1p
    "dmm_unit_interfacing",    # 3  log1p
    "dmm_unit_size",           # 4  log1p
    "tz_author_norm",          # 5  [-1,1]
    "tz_committer_norm",       # 6  [-1,1]
    "hour_sin",                # 7  [-1,1]
    "hour_cos",                # 8  [-1,1]
    "dow_sin",                 # 9  [-1,1]
    "dow_cos",                 # 10 [-1,1]
    "has_sdlc_data",           # 11 binary
    "repo_commits_90d",        # 12 log1p
    "repo_active_authors_90d", # 13 log1p
]

FN_FEAT_NAMES = [
    "num_lines_of_code",   # 0
    "complexity",          # 1
    "token_count",         # 2
    "length",              # 3
    "top_nesting_level",   # 4
    "loc_before",          # 5
    "complexity_before",   # 6
    "tokens_before",       # 7
    # 8..775: code embedding (768-dim)
]

DEV_FEAT_NAMES = [
    "repo_total_commits_before",         # 0 log1p
    "repo_active_weeks_before",          # 1 log1p
    "repo_tenure_days",                  # 2 log1p
    "repo_commits_as_committer_before",  # 3 log1p
    "recent_commits_90d",                # 4 log1p
    "time_since_last_commit_days",       # 5 log1p
    "experience_percentile_in_repo",     # 6 [0,1]
    "cross_repo_commits_before",         # 7 log1p
    "num_repos_contributed_before",      # 8 log1p
]

FN_EDGE_FEAT_NAMES = [
    "loc_before",         # 0
    "complexity_before",  # 1
    "tokens_before",      # 2
    "delta_loc",          # 3
    "delta_complexity",   # 4
    "delta_tokens",       # 5
    "fct_MODIFY",         # 6 one-hot
    "fct_ADD",            # 7 one-hot
    "fct_DELETE",         # 8 one-hot
    "fct_RENAME",         # 9 one-hot
    "fct_REFACTOR",       # 10 one-hot
]


def stats(arr: np.ndarray, name: str) -> str:
    if arr.size == 0:
        return f"  {name}: (empty)"
    mn, mx = arr.min(), arr.max()
    mean, std = arr.mean(), arr.std()
    zeros_pct = (arr == 0).mean() * 100
    return f"  {name}: min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={std:.4f} zeros={zeros_pct:.1f}%"


def check_binary(arr: np.ndarray, name: str) -> str:
    unique = np.unique(arr)
    ok = all(v in (0.0, 1.0) for v in unique)
    status = "OK" if ok else f"FAIL (found {unique[:5]})"
    return f"  {name}: binary {status}"


def check_range(arr: np.ndarray, name: str, lo: float, hi: float) -> str:
    ok = (arr >= lo - 1e-5).all() and (arr <= hi + 1e-5).all()
    status = "OK" if ok else f"FAIL min={arr.min():.4f} max={arr.max():.4f}"
    return f"  {name}: range [{lo},{hi}] {status}"


def main():
    all_pts = sorted(GRAPHS_DIR.glob("*.pt"))
    if not all_pts:
        raise SystemExit(f"No graphs in {GRAPHS_DIR}")

    sample_pts = random.sample(all_pts, min(N_SAMPLE, len(all_pts)))
    print(f"Validating {len(sample_pts)} graphs from {GRAPHS_DIR}\n")

    # Accumulators: per-node-type per-dim
    node_data: dict[str, list[np.ndarray]] = {}
    edge_data: dict[tuple, list[np.ndarray]] = {}
    dim_errors: list[str] = []
    graphs_loaded = 0

    for p in sample_pts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = torch.load(p, weights_only=False)
        graphs_loaded += 1

        # Node dims
        for ntype, expected_dim in EXPECTED_NODE_DIMS.items():
            if ntype not in g.node_types:
                continue
            x = g[ntype].x
            if x.shape[1] != expected_dim:
                dim_errors.append(f"{p.stem} {ntype}: got {x.shape[1]}, expected {expected_dim}")
            arr = x.detach().cpu().float().numpy()
            node_data.setdefault(ntype, []).append(arr)

        # Edge dims
        for et, expected_dim in EXPECTED_EDGE_DIMS.items():
            if et not in g.edge_types:
                continue
            ea = getattr(g[et], "edge_attr", None)
            if ea is None or ea.numel() == 0:
                continue
            if ea.shape[1] != expected_dim:
                dim_errors.append(f"{p.stem} {et}: got {ea.shape[1]}, expected {expected_dim}")
            arr = ea.detach().cpu().float().numpy()
            edge_data.setdefault(et, []).append(arr)

    print(f"Loaded {graphs_loaded} graphs\n")

    # ── Dim errors ────────────────────────────────────────────────────────────
    if dim_errors:
        print(f"DIMENSION ERRORS ({len(dim_errors)}):")
        for e in dim_errors[:20]:
            print(f"  {e}")
    else:
        print("Dimension check: ALL OK\n")

    # ── Commit node (14 dims) ─────────────────────────────────────────────────
    if "commit" in node_data:
        arr = np.vstack(node_data["commit"])  # (N, 14)
        print(f"=== commit node ({arr.shape[0]} rows) ===")
        print(check_binary(arr[:, 0], "in_main_branch"))
        print(check_binary(arr[:, 1], "is_merge"))
        for i in [2, 3, 4]:
            print(check_range(arr[:, i], COMMIT_FEAT_NAMES[i], 0.0, 100.0))
            print(stats(arr[:, i], COMMIT_FEAT_NAMES[i]))
        for i in [5, 6]:
            print(check_range(arr[:, i], COMMIT_FEAT_NAMES[i], -1.2, 1.2))
        for i in [7, 8, 9, 10]:
            print(check_range(arr[:, i], COMMIT_FEAT_NAMES[i], -1.0, 1.0))
        print(check_binary(arr[:, 11], "has_sdlc_data"))
        for i in [12, 13]:
            print(stats(arr[:, i], COMMIT_FEAT_NAMES[i]))
        print()

    # ── Function node (776 dims) ──────────────────────────────────────────────
    if "function" in node_data:
        arr = np.vstack(node_data["function"])
        print(f"=== function node ({arr.shape[0]} rows, {arr.shape[1]} dims) ===")
        for i, name in enumerate(FN_FEAT_NAMES):
            print(stats(arr[:, i], name))
        # Embedding L2 norm
        emb = arr[:, 8:]
        norms = np.linalg.norm(emb, axis=1)
        zero_embs = (norms < 1e-6).mean() * 100
        nonzero_norms = norms[norms > 1e-6]
        if nonzero_norms.size > 0:
            print(f"  embedding L2 norm (non-zero): mean={nonzero_norms.mean():.4f} std={nonzero_norms.std():.4f}")
        print(f"  zero embeddings: {zero_embs:.1f}%")
        print()

    # ── File node (3 dims) ───────────────────────────────────────────────────
    if "file" in node_data:
        arr = np.vstack(node_data["file"])
        print(f"=== file node ({arr.shape[0]} rows) ===")
        for i, name in enumerate(["num_lines_added", "num_lines_deleted", "complexity"]):
            print(stats(arr[:, i], name))
        print()

    # ── Hunk node (770 dims) ─────────────────────────────────────────────────
    if "hunk" in node_data:
        arr = np.vstack(node_data["hunk"])
        print(f"=== hunk node ({arr.shape[0]} rows, {arr.shape[1]} dims) ===")
        for i, name in enumerate(["complexity", "token_count"]):
            print(stats(arr[:, i], name))
        emb = arr[:, 2:]
        norms = np.linalg.norm(emb, axis=1)
        zero_embs = (norms < 1e-6).mean() * 100
        nonzero_norms = norms[norms > 1e-6]
        if nonzero_norms.size > 0:
            print(f"  embedding L2 norm (non-zero): mean={nonzero_norms.mean():.4f} std={nonzero_norms.std():.4f}")
        print(f"  zero embeddings: {zero_embs:.1f}%")
        print()

    # ── Developer node (9 dims) ──────────────────────────────────────────────
    if "developer" in node_data:
        arr = np.vstack(node_data["developer"])
        print(f"=== developer node ({arr.shape[0]} rows) ===")
        for i in [0, 1, 2, 3, 4, 5, 7, 8]:  # log1p ones
            print(stats(arr[:, i], DEV_FEAT_NAMES[i]))
        print(check_range(arr[:, 6], "experience_percentile", 0.0, 1.0))
        print(stats(arr[:, 6], "experience_percentile"))
        print()

    # ── SDLC nodes (issue, pull_request, release_tag) ────────────────────────
    for ntype in ["issue", "pull_request", "release_tag"]:
        if ntype in node_data:
            arr = np.vstack(node_data[ntype])
            print(f"=== {ntype} node ({arr.shape[0]} rows) ===")
            for i in range(arr.shape[1]):
                print(stats(arr[:, i], f"dim_{i}"))
            print()

    # ── commit->file edge (4 dims) ───────────────────────────────────────────
    et = ("commit", "modifies_file", "file")
    if et in edge_data:
        arr = np.vstack(edge_data[et])
        print(f"=== commit->file edge ({arr.shape[0]} rows) ===")
        for i, name in enumerate(["lines_added", "lines_deleted", "complexity", "file_ownership_stats_dim3"]):
            print(stats(arr[:, i], name))
        print()

    # ── commit->function edge (11 dims) ─────────────────────────────────────
    et = ("commit", "modifies_func", "function")
    if et in edge_data:
        arr = np.vstack(edge_data[et])
        print(f"=== commit->function edge ({arr.shape[0]} rows) ===")
        for i, name in enumerate(FN_EDGE_FEAT_NAMES):
            print(stats(arr[:, i], name))
        # fct one-hot sum should be 1
        fct = arr[:, 6:11]
        fct_sum = fct.sum(axis=1)
        bad = (np.abs(fct_sum - 1.0) > 1e-5).mean() * 100
        print(f"  fct one-hot sum=1: {100-bad:.1f}% OK ({bad:.1f}% bad)")
        print()

    # ── author/committer edges (3 dims) ──────────────────────────────────────
    for et_name, et in [("authored_by", ("commit","authored_by","developer")),
                         ("committed_by", ("commit","committed_by","developer"))]:
        if et in edge_data:
            arr = np.vstack(edge_data[et])
            print(f"=== {et_name} edge ({arr.shape[0]} rows) ===")
            for i, name in enumerate(["dev_experience_days", "dev_commits_before", "dev_is_new_contributor"]):
                if i == 2:
                    print(check_binary(arr[:, i], name))
                else:
                    print(stats(arr[:, i], name))
            print()

    # ── developer->file ownership edge (3 dims) ──────────────────────────────
    et = ("developer", "owns", "file")
    if et in edge_data:
        arr = np.vstack(edge_data[et])
        print(f"=== developer->file owns edge ({arr.shape[0]} rows) ===")
        print(check_range(arr[:, 0], "ownership_ratio", 0.0, 1.0))
        print(stats(arr[:, 0], "ownership_ratio"))
        print(check_range(arr[:, 1], "lines_owned_ratio", 0.0, 1.0))
        print(stats(arr[:, 1], "lines_owned_ratio"))
        print(check_range(arr[:, 2], "log1p_edits", 0.0, 1e9))
        print(stats(arr[:, 2], "log1p_edits"))
        print()

    # ── issue/pr edges (3 dims) ──────────────────────────────────────────────
    for et_name, et in [("has_issue", ("commit","has_issue","issue")),
                         ("has_pr", ("commit","has_pr","pull_request"))]:
        if et in edge_data:
            arr = np.vstack(edge_data[et])
            print(f"=== {et_name} edge ({arr.shape[0]} rows) ===")
            for i, name in enumerate(["ratio_0", "ratio_1", "has_gap_binary"]):
                if i == 2:
                    print(check_binary(arr[:, i], name))
                else:
                    print(stats(arr[:, i], name))
            print()

    # ── release_tag edge (1 dim) ─────────────────────────────────────────────
    et = ("commit", "has_release", "release_tag")
    if et in edge_data:
        arr = np.vstack(edge_data[et])
        print(f"=== has_release edge ({arr.shape[0]} rows) ===")
        print(stats(arr[:, 0], "activity_since_last_tag"))
        print()

    print("=== DONE ===")


if __name__ == "__main__":
    main()
