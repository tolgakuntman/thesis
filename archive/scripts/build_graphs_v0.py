"""
scripts/build_graphs.py

Batch-builds per-commit HeteroData graphs from pre-normalized graph_ready tables.
Each graph is saved as data_new/graph_ready/graphs/{hash}.pt.

Outputs:
  data_new/graph_ready/graphs/{hash}.pt      — one HeteroData per commit
  data_new/graph_ready/build_manifest.csv    — hash, label, n_nodes, n_edges, build_time_ms
  data_new/graph_ready/failed_commits.jsonl  — {hash, reason} per failed/skipped commit

Usage:
  conda activate thesis
  python scripts/build_graphs.py             # full build (~29,826 commits)
  python scripts/build_graphs.py --limit 10  # test run
  python scripts/build_graphs.py --hashes a1b2c3 d4e5f6   # specific commits
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
GR   = ROOT / "data_new" / "graph_ready"
GD   = ROOT / "data" / "graph_data"

OUT_DIR  = GR / "graphs"
MANIFEST = GR / "build_manifest.csv"
FAILED   = GR / "failed_commits.jsonl"

# ── feature column definitions ─────────────────────────────────────────────────
COMMIT_NUM_COLS = [
    "author_timezone", "committer_timezone",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]  # 6-dim: timezone behaviour + cyclical time-of-commit
# Removed DMM (dmm_unit_size/complexity/interfacing) and churn
# (num_lines_deleted/added/changed/files_changed) — both encode commit size
# and act as a large-commit detector rather than a vulnerability predictor.

FUNC_FEAT_COLS = [
    "num_lines_of_code", "complexity", "token_count", "length", "top_nesting_level",
    "fct_add", "fct_modify", "fct_refactor", "fct_delete", "fct_rename",
]  # 10-dim

FILE_CODE_COLS = [
    "num_lines_of_code", "complexity", "token_count",
]  # 3-dim: static code quality metrics
# Removed num_lines_added/deleted/num_method_changed — before_change=True rows
# (VCC canonical) have 0 for all three, creating systematic leakage.

OWN_COLS = ["n_owners", "max_own_ratio", "hhi", "total_lines_norm"]  # 4-dim

DEV_FEAT_COLS = [
    "total_commits", "active_weeks", "commits_as_committer",
    "total_issues", "total_pull_requests", "is_github_user",
]  # 6-dim

ISSUE_COLS = [
    "issue_close_rate_180d",
    "pr_to_issue_open_ratio_90d",
]  # 2-dim
# Removed issue_open_at_anchor (88% VCC=0), issue_age_median, has_issue_pr_gap —
# all encode CVE discovery timeline: VCC made before discovery, FC made to fix it.

PR_COLS = [
    "pr_merge_or_close_rate_180d",
    "pr_to_issue_open_ratio_90d",
    "has_release_pressure_180d",
]  # 3-dim
# Removed pr_count (82% VCC=0) and pr_age_median — same CVE-timeline leakage.

# release_tag: 4-dim = [time_since_last_tag, days_to_next_tag, release_cycle_position, is_prev_flag]


# ── helpers ────────────────────────────────────────────────────────────────────

def canonical_fn_filter(fn_rows: pd.DataFrame, commit_type: str) -> pd.DataFrame:
    """VCC → before_change=True (fallback to False for ADD/RENAME). FC/normal → before_change=False."""
    if commit_type == "VCC":
        bc_true = fn_rows[fn_rows["before_change"] == True]
        return bc_true if len(bc_true) > 0 else fn_rows[fn_rows["before_change"] == False]
    return fn_rows[fn_rows["before_change"] == False]


def _safe_feat(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract cols from df as float32, filling missing cols with 0."""
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for i, c in enumerate(cols):
        if c in df.columns:
            out[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
    return out


# ── table loader ───────────────────────────────────────────────────────────────

def load_tables() -> dict:
    """Load all graph_ready tables into memory. Returns dict of index structures."""
    print("Loading tables into memory...")
    t0 = time.time()
    T = {}

    # ── commit info (normalized numeric features) ──────────────────────────
    ci = pd.read_csv(GR / "commit_info_normalized.csv")
    ci = ci.drop_duplicates(subset=["hash"]).set_index("hash")
    T["commit_info"] = ci  # DataFrame indexed by hash

    # ── commit message embeddings ──────────────────────────────────────────
    T["msg_emb"] = np.load(GR / "commit_msg_embeddings.npy", mmap_mode="r")
    msg_idx_df = pd.read_csv(GR / "commit_msg_index.csv")
    T["msg_idx"] = dict(zip(msg_idx_df["hash"], range(len(msg_idx_df))))

    # ── function features ──────────────────────────────────────────────────
    fn = pd.read_csv(GR / "function_features_normalized.csv")
    T["fn_by_hash"] = dict(tuple(fn.groupby("hash")))

    # ── function code embedding index (key-based, NOT positional) ─────────
    fn_code_idx_df = pd.read_csv(GR / "function_code_index.csv")
    fn_code_idx_df["_key"] = list(zip(
        fn_code_idx_df["hash"], fn_code_idx_df["name"], fn_code_idx_df["filename"]
    ))
    T["fn_code_idx"] = dict(zip(fn_code_idx_df["_key"], range(len(fn_code_idx_df))))
    T["fn_code_emb"] = np.load(GR / "function_code_embeddings.npy", mmap_mode="r")  # [269760, 768]

    # ── file features ──────────────────────────────────────────────────────
    fi = pd.read_csv(GR / "file_features_normalized.csv")
    T["fi_by_hash"] = dict(tuple(fi.groupby("hash")))

    # ── ownership stats (pre-aggregated, 90-day window) ───────────────────
    own = pd.read_csv(GR / "ownership_stats_90d.csv")
    T["own_by_hash"] = dict(tuple(own.groupby("commit_hash")))

    # ── developer features ─────────────────────────────────────────────────
    dev = pd.read_csv(GR / "developer_features_normalized.csv")
    dev = dev.drop_duplicates(subset=["dev_id"]).set_index("dev_id")
    T["developer_info"] = dev

    # ── commit → developer mapping ─────────────────────────────────────────
    ca = pd.read_csv(GD / "commit_author_full.csv")
    # keep both author and committer roles
    T["author_by_hash"] = ca.groupby("commit_hash")["dev_id"].apply(list).to_dict()

    # ── commit-level SDLC features + label ────────────────────────────────
    clf = pd.read_csv(GR / "commit_level_features_normalized.csv")
    clf = clf.drop_duplicates(subset=["hash"]).set_index("hash")
    T["clf"] = clf

    # ── release tag window ─────────────────────────────────────────────────
    tag = pd.read_csv(GR / "commit_tag_window.csv")
    tag = tag.drop_duplicates(subset=["hash"]).set_index("hash")
    T["tag"] = tag

    # ── commit type + repo_url (from raw commit_info_full) ─────────────────
    ci_full = pd.read_csv(
        GD / "commit_info_full.csv",
        usecols=["hash", "commit_type", "repo_url"],
    )
    ci_full = ci_full.drop_duplicates(subset=["hash"]).set_index("hash")
    T["ci_full"] = ci_full

    print(f"  Tables loaded in {time.time()-t0:.1f}s")
    return T


# ── single-commit graph builder ────────────────────────────────────────────────

def build_graph(h: str, commit_type: str, T: dict) -> HeteroData:
    """
    Build a HeteroData graph for commit `h`.
    Raises ValueError with a reason string if the graph cannot be built (orphan guard).
    """
    data = HeteroData()

    # ── commit node [1, 777] ───────────────────────────────────────────────
    if h not in T["commit_info"].index:
        raise ValueError("no_commit_info")

    ci_row = T["commit_info"].loc[h]
    commit_num = _safe_feat(ci_row.to_frame().T, COMMIT_NUM_COLS)  # [1, 13]

    emb_row = T["msg_idx"].get(h)
    if emb_row is not None:
        msg_vec = T["msg_emb"][emb_row].reshape(1, -1).astype(np.float32)  # [1, 768]
    else:
        msg_vec = np.zeros((1, 768), dtype=np.float32)

    data["commit"].x = torch.tensor(
        np.concatenate([commit_num, msg_vec], axis=1), dtype=torch.float
    )  # [1, 774]  (6 numeric + 768 msg_emb)

    # ── file nodes [N_files, 10] ───────────────────────────────────────────
    fi_rows = T["fi_by_hash"].get(h, pd.DataFrame())
    if len(fi_rows) == 0:
        raise ValueError("orphan_no_file_info")

    own_rows = T["own_by_hash"].get(h, pd.DataFrame())

    if len(own_rows) > 0:
        fi_merged = fi_rows.merge(
            own_rows[["file_path"] + OWN_COLS],
            left_on="filename", right_on="file_path",
            how="left",
        )
    else:
        fi_merged = fi_rows.copy()

    for c in OWN_COLS:
        if c not in fi_merged.columns:
            fi_merged[c] = 0.0
    fi_merged[OWN_COLS] = fi_merged[OWN_COLS].fillna(0.0)

    file_feat = _safe_feat(fi_merged, FILE_CODE_COLS + OWN_COLS)
    data["file"].x = torch.tensor(file_feat, dtype=torch.float)  # [N_files, 7] (3 code + 4 own)
    N_files = data["file"].x.size(0)

    # ── function nodes [N_funcs, 778] (10 numeric + 768 code embedding) ────
    fn_rows_all = T["fn_by_hash"].get(h, pd.DataFrame())
    fn_rows = canonical_fn_filter(fn_rows_all, commit_type) if len(fn_rows_all) > 0 else fn_rows_all

    FN_DIM = len(FUNC_FEAT_COLS) + 768  # 778

    if len(fn_rows) > 0:
        fn_numeric = _safe_feat(fn_rows, FUNC_FEAT_COLS)  # [N, 10]

        # look up code embedding for each function row by (hash, name, filename)
        fn_code_parts = []
        for _, row in fn_rows.iterrows():
            key = (row["hash"], row["name"], row["filename"])
            idx = T["fn_code_idx"].get(key)
            if idx is not None:
                fn_code_parts.append(T["fn_code_emb"][idx].astype(np.float32))
            else:
                fn_code_parts.append(np.zeros(768, dtype=np.float32))
        fn_code = np.stack(fn_code_parts)  # [N, 768]

        fn_feat = np.concatenate([fn_numeric, fn_code], axis=1)  # [N, 778]
        data["function"].x = torch.tensor(fn_feat, dtype=torch.float)
    else:
        data["function"].x = torch.zeros(0, FN_DIM, dtype=torch.float)
    N_funcs = data["function"].x.size(0)

    # ── developer nodes [N_devs, 6] ────────────────────────────────────────
    dev_ids = list(dict.fromkeys(T["author_by_hash"].get(h, [])))  # dedup, preserve order
    valid_devs = [d for d in dev_ids if d in T["developer_info"].index]

    if valid_devs:
        dev_feat = _safe_feat(T["developer_info"].loc[valid_devs], DEV_FEAT_COLS)
        data["developer"].x = torch.tensor(dev_feat, dtype=torch.float)
    else:
        data["developer"].x = torch.zeros(0, len(DEV_FEAT_COLS), dtype=torch.float)
    N_devs = data["developer"].x.size(0)

    # ── issue + PR nodes [1, 5] each ───────────────────────────────────────
    if h in T["clf"].index:
        clf_row = T["clf"].loc[h]
        data["issue"].x = torch.tensor(
            _safe_feat(clf_row.to_frame().T, ISSUE_COLS), dtype=torch.float
        )  # [1, 2]
        data["pull_request"].x = torch.tensor(
            _safe_feat(clf_row.to_frame().T, PR_COLS), dtype=torch.float
        )  # [1, 3]
    else:
        data["issue"].x = torch.zeros(0, len(ISSUE_COLS), dtype=torch.float)
        data["pull_request"].x = torch.zeros(0, len(PR_COLS), dtype=torch.float)

    has_issue = data["issue"].x.size(0) > 0
    has_pr    = data["pull_request"].x.size(0) > 0

    # ── release_tag nodes [0–2, 4] ─────────────────────────────────────────
    tag_feats = []
    if h in T["tag"].index:
        tag_row = T["tag"].loc[h]

        def _tag_float(val, default=0.0):
            """Convert tag value to float, replacing NaN/None/non-finite with default."""
            try:
                x = float(val)
                return default if (x != x or not np.isfinite(x)) else x
            except (TypeError, ValueError):
                return default

        prev_days = _tag_float(tag_row.get("time_since_last_tag", 0))
        next_days = _tag_float(tag_row.get("days_to_next_tag", 0))
        rcp       = _tag_float(tag_row.get("release_cycle_position", 0))

        if tag_row.get("has_prev_tag", False):
            tag_feats.append([prev_days, 0.0, rcp, 1.0])  # is_prev=1
        if tag_row.get("has_next_tag", False):
            tag_feats.append([0.0, next_days, rcp, 0.0])  # is_prev=0

    if tag_feats:
        data["release_tag"].x = torch.tensor(tag_feats, dtype=torch.float)
    else:
        data["release_tag"].x = torch.zeros(0, 4, dtype=torch.float)
    N_tags = data["release_tag"].x.size(0)

    # ── edges ──────────────────────────────────────────────────────────────

    # commit ↔ file
    file_dst = torch.arange(N_files, dtype=torch.long)
    commit_src = torch.zeros(N_files, dtype=torch.long)
    data["commit", "modifies_file", "file"].edge_index   = torch.stack([commit_src, file_dst])
    data["file",   "in_commit",     "commit"].edge_index = torch.stack([file_dst, commit_src])

    # file ↔ function (via filename match)
    if N_funcs > 0:
        file_names = fi_merged["filename"].tolist()
        fn_filenames = fn_rows["filename"].tolist()

        f_src, f_dst = [], []  # file_idx → func_idx
        for fi_idx, fn_fname in enumerate(fn_filenames):
            try:
                file_idx = file_names.index(fn_fname)
                f_src.append(file_idx)
                f_dst.append(fi_idx)
            except ValueError:
                pass  # function references a file not in this commit's file list

        if f_src:
            src_t = torch.tensor(f_src, dtype=torch.long)
            dst_t = torch.tensor(f_dst, dtype=torch.long)
            data["file",     "contains", "function"].edge_index = torch.stack([src_t, dst_t])
            data["function", "in_file",  "file"].edge_index     = torch.stack([dst_t, src_t])
        else:
            data["file",     "contains", "function"].edge_index = torch.zeros(2, 0, dtype=torch.long)
            data["function", "in_file",  "file"].edge_index     = torch.zeros(2, 0, dtype=torch.long)

        # commit ↔ function (direct)
        func_dst = torch.arange(N_funcs, dtype=torch.long)
        comm_src = torch.zeros(N_funcs, dtype=torch.long)
        data["commit",   "modifies_func", "function"].edge_index = torch.stack([comm_src, func_dst])
        data["function", "in_commit_fn",  "commit"].edge_index   = torch.stack([func_dst, comm_src])

    # commit ↔ developer
    if N_devs > 0:
        dev_dst = torch.arange(N_devs, dtype=torch.long)
        comm_src = torch.zeros(N_devs, dtype=torch.long)
        data["commit",    "authored_by", "developer"].edge_index = torch.stack([comm_src, dev_dst])
        data["developer", "authored",    "commit"].edge_index    = torch.stack([dev_dst, comm_src])

    # commit ↔ issue
    if has_issue:
        data["commit", "has_issue",         "issue"].edge_index  = torch.tensor([[0], [0]], dtype=torch.long)
        data["issue",  "linked_to_commit",  "commit"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # commit ↔ pull_request
    if has_pr:
        data["commit",       "has_pr",           "pull_request"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        data["pull_request", "linked_to_commit", "commit"].edge_index       = torch.tensor([[0], [0]], dtype=torch.long)

    # commit ↔ release_tag
    if N_tags > 0:
        tag_dst  = torch.arange(N_tags, dtype=torch.long)
        comm_src = torch.zeros(N_tags, dtype=torch.long)
        data["commit",      "has_release", "release_tag"].edge_index = torch.stack([comm_src, tag_dst])
        data["release_tag", "release_of",  "commit"].edge_index      = torch.stack([tag_dst, comm_src])

    return data


# ── main build loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build per-commit HeteroData graphs")
    parser.add_argument("--limit",   type=int,   default=None, help="Build only first N commits (for testing)")
    parser.add_argument("--hashes",  nargs="+",  default=None, help="Build specific commit hashes only")
    parser.add_argument("--overwrite", action="store_true",    help="Re-build and overwrite existing .pt files")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    T = load_tables()

    # ── determine build list ───────────────────────────────────────────────
    ci_full = T["ci_full"]

    # only commits with a known commit_type
    buildable = ci_full[ci_full["commit_type"].isin(["VCC", "FC", "normal"])].copy()

    # apply label: VCC=1, FC/normal=0
    buildable["y"] = (buildable["commit_type"] == "VCC").astype(int)

    if args.hashes:
        buildable = buildable.loc[buildable.index.intersection(args.hashes)]
    if args.limit:
        buildable = buildable.head(args.limit)

    print(f"\nBuild target: {len(buildable):,} commits  "
          f"(VCC={int((buildable['y']==1).sum())}, neg={int((buildable['y']==0).sum())})")

    # ── build loop ─────────────────────────────────────────────────────────
    manifest_rows = []
    failed_file = open(FAILED, "w")

    n_ok = n_skip = n_fail = 0

    for h, row in tqdm(buildable.iterrows(), total=len(buildable), desc="Building graphs"):
        out_path = OUT_DIR / f"{h}.pt"

        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        t_start = time.time()
        try:
            data = build_graph(h, row["commit_type"], T)
            data.y = torch.tensor([int(row["y"])], dtype=torch.long)

            # store repo_url for dataset splitting
            data["commit"].repo_url = row.get("repo_url", "")

            torch.save(data, out_path)

            elapsed_ms = int((time.time() - t_start) * 1000)
            n_nodes = sum(data[nt].x.size(0) for nt in data.node_types)
            n_edges = sum(
                data[et].edge_index.size(1)
                for et in data.edge_types
                if hasattr(data[et], "edge_index")
            )
            manifest_rows.append({
                "hash":         h,
                "commit_type":  row["commit_type"],
                "label":        int(row["y"]),
                "repo_url":     row.get("repo_url", ""),
                "n_nodes":      n_nodes,
                "n_edges":      n_edges,
                "build_time_ms": elapsed_ms,
            })
            n_ok += 1

        except ValueError as e:
            failed_file.write(json.dumps({"hash": h, "reason": str(e)}) + "\n")
            n_fail += 1
        except Exception as e:
            failed_file.write(json.dumps({"hash": h, "reason": f"exception:{type(e).__name__}:{e}"}) + "\n")
            n_fail += 1

    failed_file.close()

    # ── save manifest ──────────────────────────────────────────────────────
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        write_header = not MANIFEST.exists() or args.overwrite
        manifest_df.to_csv(MANIFEST, index=False, mode="w" if write_header else "a", header=write_header)

    print(f"\nDone — built: {n_ok:,}  skipped: {n_skip:,}  failed: {n_fail:,}")
    if manifest_rows:
        df = pd.DataFrame(manifest_rows)
        print(f"  label=1 (VCC): {int((df['label']==1).sum()):,}")
        print(f"  label=0 (neg): {int((df['label']==0).sum()):,}")
        print(f"  avg nodes/graph: {df['n_nodes'].mean():.1f}")
        print(f"  avg edges/graph: {df['n_edges'].mean():.1f}")
        print(f"  avg build time: {df['build_time_ms'].mean():.1f}ms")


if __name__ == "__main__":
    main()
