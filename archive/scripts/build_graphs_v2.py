"""
scripts/build_graphs_v2.py

Build per-commit HeteroData graphs from the graph_ready_sampling_v2 dataset.

Reads from:  ICVul_pp/graph_ready_sampling_v2/
Writes to:   outputs/graph_ready_v2/graphs/{hash}.pt

Node types and feature dimensions
----------------------------------
  commit      :  14  (core: in_main_branch, merge, dmm×3, tz×2, hour_sin/cos, dow_sin/cos,
                       SDLC: has_sdlc_data, repo_commits_last_90d, repo_active_authors_90d)
  function    : 776  (8 normalized numeric + 768-dim GraphCodeBERT embedding)
  file        :   3  (num_lines_added, num_lines_deleted, complexity — normalized)
  hunk        : 770  (2 normalized numeric + 768-dim GraphCodeBERT embedding)
  developer   :   9  (contextual per-commit per-developer features, log1p-transformed)
  issue       :   4  (SDLC aggregates — already normalized)
  pull_request:   4  (SDLC aggregates — already normalized)
  release_tag :   4  (SDLC aggregates — already normalized)

Edge attribute dimensions (matching src/model.py EDGE_ATTR_DIMS)
-----------------------------------------------------------------
  commit→file      : 4   (n_owners, max_own_ratio, ownership_hhi, total_lines_norm)
  commit→function  : 11  (loc_before, complexity_before, tokens_before,
                          delta_loc, delta_complexity, delta_tokens,
                          fct_modify, fct_add, fct_delete, fct_rename, fct_refactor)
  commit→developer : 3   (dev_experience_days, dev_commits_before, dev_is_new_contributor)
  developer→file   : 3   (ownership_ratio, lines_owned/total_lines, log1p(edits))
  commit→issue     : 3   (pr_to_issue_open_ratio_90d, issue_to_pr_closed_ratio_90d, has_issue_pr_gap)
  commit→pr        : 3   (same)
  commit→release   : 1   (activity_since_last_tag)

Usage
-----
  python scripts/build_graphs_v2.py
  python scripts/build_graphs_v2.py --limit 100          # smoke test
  python scripts/build_graphs_v2.py --overwrite          # rebuild existing
  python scripts/build_graphs_v2.py --allow_commit_only  # include no-code commits
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[1]              # thesis/
DATA_ROOT  = ROOT.parent / "ICVul_pp" / "graph_ready_sampling_v2"
FEAT_DIR   = DATA_ROOT / "features"
ENC_CODE   = DATA_ROOT / "encodings" / "code"
OUT_ROOT   = ROOT / "outputs" / "graph_ready_v2"
OUT_DIR    = OUT_ROOT / "graphs"
MANIFEST   = OUT_ROOT / "build_manifest.csv"
FAILED     = OUT_ROOT / "failed_commits.jsonl"

OWNERSHIP_THRESHOLD = 0.05   # min ownership_ratio for developer→file edge

# ── Feature column lists ───────────────────────────────────────────────────────

# Commit node: 14 dims
#   in_main_branch [0], merge [1]  — binary (no transform)
#   dmm×3 [2-4]                   — log1p (NaN→0 first)
#   author_timezone [5],           — divide by 43200 (seconds→fraction)
#   committer_timezone [6]
#   hour_sin [7], hour_cos [8]     — computed from author_date
#   dow_sin [9], dow_cos [10]      — computed from author_date
#   has_sdlc_data [11]             — binary, from SDLC features (already normalized)
#   repo_commits_last_90d [12]     — already normalized
#   repo_active_authors_90d [13]   — already normalized
SDLC_COMMIT_EXTRA = ["has_sdlc_data", "repo_commits_last_90d", "repo_active_authors_90d"]

ISSUE_FEAT_COLS = ["issue_open_90d", "issue_age_median", "issues_closed_last_90d",
                   "issue_open_velocity_90d"]
PR_FEAT_COLS    = ["pr_count_90d", "pr_age_median", "pr_closed_last_90d",
                   "pr_open_velocity_90d"]
TAG_FEAT_COLS   = ["days_since_prev_tag", "tags_last_365d", "avg_release_cadence_days",
                   "days_since_prev_tag_norm"]
AUTHOR_EDGE_FEAT_COLS  = ["dev_experience_days", "dev_commits_before", "dev_is_new_contributor"]
ISSUE_EDGE_FEAT_COLS   = ["pr_to_issue_open_ratio_90d", "issue_to_pr_closed_ratio_90d", "has_issue_pr_gap"]
TAG_EDGE_FEAT_COLS     = ["activity_since_last_tag"]

# Function node: 8 numeric (already normalized) + 768 embedding
FN_FEAT_COLS = ["num_lines_of_code", "complexity", "token_count", "length",
                "top_nesting_level", "loc_before", "complexity_before", "tokens_before"]

# File node: 3 numeric (already normalized)
FILE_FEAT_COLS = ["num_lines_added", "num_lines_deleted", "complexity"]

# Hunk node: 2 numeric (already normalized) + 768 embedding
HUNK_FEAT_COLS = ["complexity", "token_count"]

# Developer node: 9 contextual features (apply log1p at build time)
DEV_FEAT_COLS = [
    "repo_total_commits_before",      # 0
    "repo_active_weeks_before",        # 1
    "repo_tenure_days",                # 2
    "repo_commits_as_committer_before",# 3
    "recent_commits_90d",              # 4
    "time_since_last_commit_days",     # 5  (NaN→0)
    "experience_percentile_in_repo",   # 6  ([0,1] — no log1p)
    "cross_repo_commits_before",       # 7
    "num_repos_contributed_before",    # 8
]
DEV_LOG1P_IDXS = [0, 1, 2, 3, 4, 5, 7, 8]  # indices to apply log1p (skip 6=percentile)

# function_change_type → fct_* one-hot slot
FCT_MAP = {"MODIFY": 0, "ADD": 1, "DELETE": 2, "RENAME": 3, "REFACTOR": 4}


# ── Utilities ──────────────────────────────────────────────────────────────────

def safe_values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for i, col in enumerate(cols):
        if col in df.columns:
            out[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return out


def empty_edge(data: HeteroData, src: str, rel: str, dst: str) -> None:
    data[src, rel, dst].edge_index = torch.zeros((2, 0), dtype=torch.long)


def connect_commit(
    data: HeteroData,
    dst_type: str,
    rel_fwd: str,
    rel_rev: str,
    edge_attr: np.ndarray | None = None,
) -> None:
    n_dst = data[dst_type].x.size(0)
    if n_dst == 0:
        empty_edge(data, "commit", rel_fwd, dst_type)
        empty_edge(data, dst_type, rel_rev, "commit")
        return
    src = torch.zeros(n_dst, dtype=torch.long)
    dst = torch.arange(n_dst, dtype=torch.long)
    data["commit", rel_fwd, dst_type].edge_index = torch.stack([src, dst])
    data[dst_type, rel_rev, "commit"].edge_index = torch.stack([dst, src])
    if edge_attr is not None:
        ea = torch.tensor(edge_attr, dtype=torch.float32)
        data["commit", rel_fwd, dst_type].edge_attr = ea
        data[dst_type, rel_rev, "commit"].edge_attr = ea


def build_bipartite_edges(
    data: HeteroData,
    src_type: str,
    rel_fwd: str,
    dst_type: str,
    src_idx: list[int],
    dst_idx: list[int],
    rel_rev: str | None = None,
    edge_attr: list | np.ndarray | None = None,
) -> None:
    if src_idx:
        src = torch.tensor(src_idx, dtype=torch.long)
        dst = torch.tensor(dst_idx, dtype=torch.long)
        data[src_type, rel_fwd, dst_type].edge_index = torch.stack([src, dst])
        if rel_rev is not None:
            data[dst_type, rel_rev, src_type].edge_index = torch.stack([dst, src])
        if edge_attr is not None:
            ea = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float32)
            data[src_type, rel_fwd, dst_type].edge_attr = ea
            if rel_rev is not None:
                data[dst_type, rel_rev, src_type].edge_attr = ea
    else:
        empty_edge(data, src_type, rel_fwd, dst_type)
        if rel_rev is not None:
            empty_edge(data, dst_type, rel_rev, src_type)


def time_sinusoids(author_date: str) -> tuple[float, float, float, float]:
    """Return (hour_sin, hour_cos, dow_sin, dow_cos). Returns zeros on parse failure."""
    try:
        dt = pd.to_datetime(author_date, utc=True)
        h = dt.hour + dt.minute / 60.0
        d = dt.day_of_week  # 0=Monday … 6=Sunday
        return (
            float(np.sin(2 * np.pi * h / 24.0)),
            float(np.cos(2 * np.pi * h / 24.0)),
            float(np.sin(2 * np.pi * d / 7.0)),
            float(np.cos(2 * np.pi * d / 7.0)),
        )
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


# ── Table loading ──────────────────────────────────────────────────────────────

def load_tables() -> dict[str, object]:
    print("Loading commit_info …")
    commit_info = pd.read_csv(
        DATA_ROOT / "commit_info.csv",
        usecols=["hash", "commit_label", "repo_url", "author_date",
                 "author_timezone", "committer_timezone",
                 "in_main_branch", "merge",
                 "dmm_unit_size", "dmm_unit_complexity", "dmm_unit_interfacing"],
        low_memory=False,
    ).drop_duplicates("hash").set_index("hash")

    print("Loading commit SDLC features …")
    cf = pd.read_csv(FEAT_DIR / "final_commit_features_normalized_final.csv",
                     low_memory=False).drop_duplicates("hash").set_index("hash")

    print("Loading function features + change types …")
    fn_feat = pd.read_csv(FEAT_DIR / "function_numeric_features_normalized.csv",
                          low_memory=False)
    fn_types = pd.read_csv(
        DATA_ROOT / "function_info.csv",
        usecols=["hash", "name", "filename", "function_change_type"],
        low_memory=False,
    )
    fn_df = fn_feat.merge(fn_types, on=["hash", "name", "filename"], how="left")

    print("Loading file features …")
    file_df = pd.read_csv(FEAT_DIR / "file_numeric_features_normalized.csv",
                          low_memory=False)

    print("Loading hunk features …")
    hunk_df = pd.read_csv(FEAT_DIR / "hunk_numeric_features_normalized.csv",
                          low_memory=False)

    print("Loading developer info …")
    di_df = pd.read_csv(DATA_ROOT / "developer_info.csv", low_memory=False)
    di_df["dev_email"] = di_df["dev_email"].astype(str).str.strip().str.lower()
    di_df = di_df.dropna(subset=["dev_email"]).copy()
    # Deduplicate by (commit_hash, dev_email) — keep first occurrence
    di_df = di_df.drop_duplicates(subset=["commit_hash", "dev_email"])

    print("Loading commit_author …")
    ca_df = pd.read_csv(DATA_ROOT / "commit_author.csv", low_memory=False)
    ca_df["dev_id"] = ca_df["dev_id"].astype(str).str.strip().str.lower()
    ca_df["role"]   = ca_df["role"].astype(str).str.strip().str.lower()

    print("Loading ownership (window_days=90) …")
    own_df = pd.read_csv(
        DATA_ROOT / "ownership.csv",
        usecols=["commit_hash", "file_path", "dev_email", "ownership_ratio",
                 "lines_owned", "edits_in_window", "total_lines", "window_days"],
        low_memory=False,
    )
    own_df = own_df[own_df["window_days"] == 90].copy()
    own_df["dev_email"] = own_df["dev_email"].astype(str).str.strip().str.lower()

    print("Loading function code embeddings …")
    fn_emb_matrix = np.load(ENC_CODE / "function_code_embeddings.npy", mmap_mode="r")
    fn_emb_index = pd.read_csv(ENC_CODE / "function_code_full_index.csv", low_memory=False)
    fn_emb_index["_key"] = list(zip(
        fn_emb_index["hash"].astype(str),
        fn_emb_index["name"].astype(str),
        fn_emb_index["filename"].astype(str),
    ))
    fn_emb_lookup: dict[tuple, int] = dict(zip(fn_emb_index["_key"], fn_emb_index["emb_idx"]))

    print("Loading hunk code embeddings …")
    hk_emb_matrix = np.load(ENC_CODE / "hunk_code_embeddings.npy", mmap_mode="r")
    hk_emb_index = pd.read_csv(ENC_CODE / "hunk_code_index.csv", low_memory=False)
    # Hunk embeddings are positional: row i in index → row i in embedding matrix
    hk_emb_index["_key"] = list(zip(
        hk_emb_index["hash"].astype(str),
        hk_emb_index["name"].astype(str),
        hk_emb_index["filename"].astype(str),
    ))
    hk_emb_lookup: dict[tuple, int] = {key: i for i, key in enumerate(hk_emb_index["_key"])}

    print("Pre-grouping by commit hash …")
    fn_by_hash   = dict(tuple(fn_df.groupby("hash",   sort=False)))
    file_by_hash = dict(tuple(file_df.groupby("hash", sort=False)))
    hunk_by_hash = dict(tuple(hunk_df.groupby("hash", sort=False)))
    di_by_hash   = dict(tuple(di_df.groupby("commit_hash", sort=False)))
    ca_by_hash   = dict(tuple(ca_df.groupby("commit_hash", sort=False)))
    own_by_hash  = dict(tuple(own_df.groupby("commit_hash", sort=False)))

    print("Tables loaded.")
    return {
        "commit_info":      commit_info,
        "cf":               cf,
        "fn_by_hash":       fn_by_hash,
        "file_by_hash":     file_by_hash,
        "hunk_by_hash":     hunk_by_hash,
        "di_by_hash":       di_by_hash,
        "ca_by_hash":       ca_by_hash,
        "own_by_hash":      own_by_hash,
        "fn_emb_matrix":    fn_emb_matrix,
        "fn_emb_lookup":    fn_emb_lookup,
        "hk_emb_matrix":    hk_emb_matrix,
        "hk_emb_lookup":    hk_emb_lookup,
    }


# ── Per-commit graph builder ───────────────────────────────────────────────────

def build_graph(
    commit_hash: str,
    label: int,
    tables: dict[str, object],
    allow_commit_only: bool,
) -> HeteroData:
    data = HeteroData()
    ci       = tables["commit_info"]
    cf       = tables["cf"]

    if commit_hash not in ci.index:
        raise ValueError("missing_commit_info")
    if commit_hash not in cf.index:
        raise ValueError("missing_commit_features")

    ci_row = ci.loc[commit_hash]
    cf_row = cf.loc[[commit_hash]]

    # ── Commit node ────────────────────────────────────────────────────────────
    # Binary features
    in_main  = float(pd.to_numeric(ci_row.get("in_main_branch", 0), errors="coerce") or 0)
    merge    = float(pd.to_numeric(ci_row.get("merge",          0), errors="coerce") or 0)
    # DMM: log1p (NaN→0)
    dmm_size  = float(np.log1p(max(0, pd.to_numeric(ci_row.get("dmm_unit_size",        np.nan), errors="coerce") or 0)))
    dmm_cmplx = float(np.log1p(max(0, pd.to_numeric(ci_row.get("dmm_unit_complexity",  np.nan), errors="coerce") or 0)))
    dmm_iface = float(np.log1p(max(0, pd.to_numeric(ci_row.get("dmm_unit_interfacing", np.nan), errors="coerce") or 0)))
    # Timezone: seconds → normalized to [-1, 1] by dividing by 43200
    tz_author = float((pd.to_numeric(ci_row.get("author_timezone",    0), errors="coerce") or 0) / 43200.0)
    tz_commit = float((pd.to_numeric(ci_row.get("committer_timezone", 0), errors="coerce") or 0) / 43200.0)
    # Sinusoidal time encoding
    hour_sin, hour_cos, dow_sin, dow_cos = time_sinusoids(str(ci_row.get("author_date", "")))
    # SDLC extras (already normalized)
    sdlc_extra = safe_values(cf_row.reset_index(drop=True), SDLC_COMMIT_EXTRA)  # (1, 3)

    commit_num = np.array([[
        in_main, merge, dmm_size, dmm_cmplx, dmm_iface,
        tz_author, tz_commit, hour_sin, hour_cos, dow_sin, dow_cos,
    ]], dtype=np.float32)  # (1, 11)
    commit_x = np.concatenate([commit_num, sdlc_extra], axis=1)  # (1, 14)
    data["commit"].x = torch.tensor(commit_x, dtype=torch.float32)

    # ── Code nodes ─────────────────────────────────────────────────────────────
    fn_rows   = tables["fn_by_hash"].get(commit_hash, pd.DataFrame())
    file_rows = tables["file_by_hash"].get(commit_hash, pd.DataFrame())
    hunk_rows = tables["hunk_by_hash"].get(commit_hash, pd.DataFrame())

    has_code = len(fn_rows) > 0 or len(file_rows) > 0 or len(hunk_rows) > 0
    if not has_code and not allow_commit_only:
        raise ValueError("no_code_rows")

    # Function nodes: 8 numeric + 768 embedding
    fn_emb_matrix  = tables["fn_emb_matrix"]
    fn_emb_lookup  = tables["fn_emb_lookup"]
    EMB_DIM = fn_emb_matrix.shape[1]  # 768

    if len(fn_rows) > 0:
        fn_num = safe_values(fn_rows.reset_index(drop=True), FN_FEAT_COLS)  # (N, 8)
        fn_num[:, 5:8] = 0.0
        fn_emb = np.zeros((len(fn_rows), EMB_DIM), dtype=np.float32)
        for local_i, row in enumerate(fn_rows.itertuples(index=False)):
            key = (str(row.hash), str(row.name), str(row.filename))
            emb_idx = fn_emb_lookup.get(key)
            if emb_idx is not None:
                fn_emb[local_i] = fn_emb_matrix[emb_idx]
        data["function"].x = torch.tensor(
            np.concatenate([fn_num, fn_emb], axis=1), dtype=torch.float32
        )
        # Function edge features: 11 dims
        fn_edge = _build_fn_edge_feats(fn_rows)
    else:
        data["function"].x = torch.zeros((0, len(FN_FEAT_COLS) + EMB_DIM), dtype=torch.float32)
        fn_edge = np.zeros((0, 11), dtype=np.float32)

    # File nodes: 3 numeric
    if len(file_rows) > 0:
        file_num = safe_values(file_rows.reset_index(drop=True), FILE_FEAT_COLS)  # (M, 3)
        data["file"].x = torch.tensor(file_num, dtype=torch.float32)
    else:
        data["file"].x = torch.zeros((0, len(FILE_FEAT_COLS)), dtype=torch.float32)

    # Hunk nodes: 2 numeric + 768 embedding
    hk_emb_matrix = tables["hk_emb_matrix"]
    hk_emb_lookup = tables["hk_emb_lookup"]
    if len(hunk_rows) > 0:
        hk_num = safe_values(hunk_rows.reset_index(drop=True), HUNK_FEAT_COLS)  # (K, 2)
        hk_emb = np.zeros((len(hunk_rows), EMB_DIM), dtype=np.float32)
        for local_i, row in enumerate(hunk_rows.itertuples(index=False)):
            key = (str(row.hash), str(row.name), str(row.filename))
            row_idx = hk_emb_lookup.get(key)
            if row_idx is not None:
                hk_emb[local_i] = hk_emb_matrix[row_idx]
        data["hunk"].x = torch.tensor(
            np.concatenate([hk_num, hk_emb], axis=1), dtype=torch.float32
        )
    else:
        data["hunk"].x = torch.zeros((0, len(HUNK_FEAT_COLS) + EMB_DIM), dtype=torch.float32)

    # ── Developer nodes ────────────────────────────────────────────────────────
    ca_rows  = tables["ca_by_hash"].get(commit_hash, pd.DataFrame())
    own_rows = tables["own_by_hash"].get(commit_hash, pd.DataFrame())
    di_rows  = tables["di_by_hash"].get(commit_hash, pd.DataFrame())

    # Build email-keyed dev index lookup
    di_by_email: dict[str, pd.Series] = {}
    if len(di_rows) > 0:
        for _, dr in di_rows.iterrows():
            email = str(dr.get("dev_email", "")).strip().lower()
            if email and email not in di_by_email:
                di_by_email[email] = dr

    # Collect authors, committers, owners in order (no duplicates)
    author_emails:    list[str] = []
    committer_emails: list[str] = []
    if len(ca_rows) > 0:
        for _, cr in ca_rows.iterrows():
            email = str(cr.get("dev_id", "")).strip().lower()
            if not email:
                continue
            role = str(cr.get("role", "")).strip().lower()
            if role == "author":
                author_emails.append(email)
            elif role == "committer":
                committer_emails.append(email)

    owner_emails: list[str] = []
    if len(own_rows) > 0:
        own_over_thresh = own_rows[
            pd.to_numeric(own_rows["ownership_ratio"], errors="coerce").fillna(0.0) >= OWNERSHIP_THRESHOLD
        ]
        owner_emails = own_over_thresh["dev_email"].dropna().unique().tolist()

    dev_emails = list(dict.fromkeys(author_emails + committer_emails + owner_emails))

    if dev_emails:
        dev_feats = []
        for email in dev_emails:
            dr = di_by_email.get(email)
            if dr is not None:
                raw = np.array([float(pd.to_numeric(dr.get(c, 0), errors="coerce") or 0)
                                for c in DEV_FEAT_COLS], dtype=np.float32)
            else:
                raw = np.zeros(len(DEV_FEAT_COLS), dtype=np.float32)
            # log1p on count/days cols, skip percentile
            for idx in DEV_LOG1P_IDXS:
                raw[idx] = float(np.log1p(max(0.0, raw[idx])))
            dev_feats.append(raw)
        data["developer"].x = torch.tensor(
            np.vstack(dev_feats), dtype=torch.float32
        )
    else:
        data["developer"].x = torch.zeros((0, len(DEV_FEAT_COLS)), dtype=torch.float32)
    dev_idx_by_email = {email: i for i, email in enumerate(dev_emails)}

    # ── SDLC aggregate nodes (1 per commit) ────────────────────────────────────
    cf_reset = cf_row.reset_index(drop=True)
    data["issue"].x        = torch.tensor(safe_values(cf_reset, ISSUE_FEAT_COLS),  dtype=torch.float32)
    data["pull_request"].x = torch.tensor(safe_values(cf_reset, PR_FEAT_COLS),     dtype=torch.float32)
    data["release_tag"].x  = torch.tensor(safe_values(cf_reset, TAG_FEAT_COLS),    dtype=torch.float32)

    # ── commit→file edges with ownership stats ─────────────────────────────────
    if len(file_rows) > 0:
        own_stats = _build_file_ownership_stats(file_rows, own_rows)
        connect_commit(data, "file", "modifies_file", "in_commit", edge_attr=own_stats)
    else:
        connect_commit(data, "file", "modifies_file", "in_commit")

    # ── commit→function edges with delta + categorical attrs ───────────────────
    connect_commit(data, "function", "modifies_func", "in_commit_fn",
                   edge_attr=fn_edge if len(fn_rows) > 0 else None)

    # ── commit→hunk edges (no attr) ────────────────────────────────────────────
    connect_commit(data, "hunk", "modifies_hunk", "in_commit_hunk")

    # ── commit→SDLC aggregate edges ────────────────────────────────────────────
    issue_edge_attr = np.tile(safe_values(cf_reset, ISSUE_EDGE_FEAT_COLS),
                              (max(data["issue"].x.size(0), 1), 1))
    connect_commit(data, "issue", "has_issue", "linked_to_commit", edge_attr=issue_edge_attr)

    pr_edge_attr = np.tile(safe_values(cf_reset, ISSUE_EDGE_FEAT_COLS),
                           (max(data["pull_request"].x.size(0), 1), 1))
    connect_commit(data, "pull_request", "has_pr", "linked_to_commit", edge_attr=pr_edge_attr)

    tag_edge_attr = np.tile(safe_values(cf_reset, TAG_EDGE_FEAT_COLS),
                            (max(data["release_tag"].x.size(0), 1), 1))
    connect_commit(data, "release_tag", "has_release", "release_of", edge_attr=tag_edge_attr)

    # ── file→function edges (file contains function) ──────────────────────────
    if len(file_rows) > 0 and len(fn_rows) > 0:
        file_name_to_idx = {
            str(r.filename): local_i
            for local_i, r in enumerate(file_rows.itertuples(index=False))
        }
        ff_src: list[int] = []
        ff_dst: list[int] = []
        for local_fn_i, fr in enumerate(fn_rows.itertuples(index=False)):
            fi = file_name_to_idx.get(str(fr.filename))
            if fi is not None:
                ff_src.append(fi)
                ff_dst.append(local_fn_i)
        build_bipartite_edges(data, "file", "contains", "function", ff_src, ff_dst, "in_file")
    else:
        empty_edge(data, "file", "contains", "function")
        empty_edge(data, "function", "in_file", "file")

    # ── commit→developer edges (author, committer) ────────────────────────────
    author_edge_attr = safe_values(cf_reset, AUTHOR_EDGE_FEAT_COLS)  # (1, 3)

    a_src, a_dst = [], []
    for email in author_emails:
        if email in dev_idx_by_email:
            a_src.append(0)
            a_dst.append(dev_idx_by_email[email])
    build_bipartite_edges(
        data, "commit", "authored_by", "developer", a_src, a_dst, "authored",
        edge_attr=np.tile(author_edge_attr, (len(a_src), 1)) if a_src else None,
    )

    c_src, c_dst = [], []
    for email in committer_emails:
        if email in dev_idx_by_email:
            c_src.append(0)
            c_dst.append(dev_idx_by_email[email])
    build_bipartite_edges(
        data, "commit", "committed_by", "developer", c_src, c_dst, "committed",
        edge_attr=np.tile(author_edge_attr, (len(c_src), 1)) if c_src else None,
    )

    # ── developer→file ownership edges ────────────────────────────────────────
    if len(file_rows) > 0 and len(own_rows) > 0 and dev_idx_by_email:
        file_path_to_idx = {}
        for local_fi, fr in enumerate(file_rows.itertuples(index=False)):
            fname = str(fr.filename)
            file_path_to_idx[fname] = local_fi

        own_src: list[int] = []
        own_dst: list[int] = []
        own_attr: list[list[float]] = []
        seen_pairs: set = set()
        for row in own_rows.itertuples(index=False):
            dev_idx = dev_idx_by_email.get(str(row.dev_email).strip().lower())
            # file_path in ownership may be full path; match on basename or exact
            file_path = str(getattr(row, "file_path", ""))
            file_idx = file_path_to_idx.get(file_path)
            if file_idx is None:
                # try basename match
                file_idx = file_path_to_idx.get(Path(file_path).name)
            if dev_idx is None or file_idx is None:
                continue
            pair = (dev_idx, file_idx)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            own_ratio   = float(pd.to_numeric(getattr(row, "ownership_ratio",   0.0), errors="coerce") or 0.0)
            lines_owned = float(pd.to_numeric(getattr(row, "lines_owned",        0.0), errors="coerce") or 0.0)
            total_lines = float(pd.to_numeric(getattr(row, "total_lines",        1.0), errors="coerce") or 1.0)
            edits       = float(pd.to_numeric(getattr(row, "edits_in_window",    0.0), errors="coerce") or 0.0)
            own_src.append(dev_idx)
            own_dst.append(file_idx)
            own_attr.append([own_ratio, lines_owned / max(total_lines, 1.0), float(np.log1p(edits))])
        build_bipartite_edges(data, "developer", "owns", "file", own_src, own_dst, "owned_by",
                              edge_attr=own_attr)
    else:
        empty_edge(data, "developer", "owns", "file")
        empty_edge(data, "file", "owned_by", "developer")

    # ── Label + metadata ───────────────────────────────────────────────────────
    data.y = torch.tensor([int(label)], dtype=torch.long)
    data["commit"].hash     = commit_hash
    data["commit"].repo_url = str(ci_row.get("repo_url", ""))
    data["commit"].author_date = str(ci_row.get("author_date", ""))
    return data


def _build_fn_edge_feats(fn_rows: pd.DataFrame) -> np.ndarray:
    """Build 11-dim edge feature matrix for commit→function edges."""
    n = len(fn_rows)
    out = np.zeros((n, 11), dtype=np.float32)
    fn_reset = fn_rows.reset_index(drop=True)
    # Dims 0-5 are intentionally kept at zero after audit. The original
    # pre-change and delta metrics showed overly strong label correlation.
    # Dims 6-10: fct_* one-hot from function_change_type
    if "function_change_type" in fn_reset.columns:
        for i, fct in enumerate(fn_reset["function_change_type"].fillna("MODIFY")):
            slot = FCT_MAP.get(str(fct).upper(), 0)  # default to MODIFY
            out[i, 6 + slot] = 1.0
    return out


def _build_file_ownership_stats(
    file_rows: pd.DataFrame, own_rows: pd.DataFrame
) -> np.ndarray:
    """Build 4-dim ownership stats per file: n_owners, max_own_ratio, ownership_hhi, total_lines_norm."""
    n = len(file_rows)
    stats = np.zeros((n, 4), dtype=np.float32)
    if len(own_rows) == 0:
        return stats
    # Build filename→local_idx map
    fname_to_idx: dict[str, int] = {}
    for local_i, fr in enumerate(file_rows.itertuples(index=False)):
        fname_to_idx[str(fr.filename)] = local_i

    for fpath, grp in own_rows.groupby("file_path", sort=False):
        # Try exact match, then basename
        fi = fname_to_idx.get(str(fpath))
        if fi is None:
            fi = fname_to_idx.get(Path(str(fpath)).name)
        if fi is None:
            continue
        ratios      = pd.to_numeric(grp["ownership_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
        total_lines = pd.to_numeric(grp["total_lines"],     errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if ratios.size == 0:
            continue
        stats[fi, 0] = float(np.log1p((ratios >= OWNERSHIP_THRESHOLD).sum()))
        stats[fi, 1] = float(ratios.max(initial=0.0))
        stats[fi, 2] = float(np.square(ratios).sum())  # HHI
        stats[fi, 3] = float(np.log1p(total_lines.max(initial=0.0)))
    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build HeteroData graphs from graph_ready_sampling_v2")
    parser.add_argument("--limit",    type=int, default=None, help="Build only the first N commits")
    parser.add_argument("--hashes",   nargs="+", default=None, help="Build specific commits")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing .pt files")
    parser.add_argument("--allow_commit_only", action="store_true",
                        help="Build graphs even when no file/function/hunk rows exist")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()

    commit_info = tables["commit_info"].copy()
    commit_info["label"] = (commit_info["commit_label"] == "VCC").astype(int)

    if args.hashes:
        commits = commit_info[commit_info.index.isin(args.hashes)].copy()
    else:
        commits = commit_info.copy()
    if args.limit is not None:
        commits = commits.head(args.limit).copy()

    n_ok, n_skip, n_fail = 0, 0, 0
    manifest_rows: list[dict] = []

    with open(FAILED, "w", encoding="utf-8") as fail_f:
        for commit_hash, row in tqdm(commits.iterrows(), total=len(commits),
                                     desc="Building v2 graphs"):
            out_path = OUT_DIR / f"{commit_hash}.pt"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue
            t0 = time.time()
            try:
                g = build_graph(commit_hash, int(row["label"]), tables,
                                allow_commit_only=args.allow_commit_only)
                torch.save(g, out_path)
                n_nodes = sum(g[nt].x.size(0) for nt in g.node_types)
                n_edges = sum(
                    g[et].edge_index.size(1) for et in g.edge_types
                    if hasattr(g[et], "edge_index")
                )
                manifest_rows.append({
                    "hash":         commit_hash,
                    "commit_label": row.get("commit_label", ""),
                    "label":        int(row["label"]),
                    "repo_url":     str(row.get("repo_url", "")),
                    "n_nodes":      n_nodes,
                    "n_edges":      n_edges,
                    "build_ms":     int((time.time() - t0) * 1000),
                })
                n_ok += 1
            except ValueError as exc:
                fail_f.write(json.dumps({"hash": commit_hash, "reason": str(exc)}) + "\n")
                n_fail += 1
            except Exception as exc:
                fail_f.write(json.dumps(
                    {"hash": commit_hash, "reason": f"{type(exc).__name__}: {exc}"}
                ) + "\n")
                n_fail += 1

    if manifest_rows:
        mdf = pd.DataFrame(manifest_rows)
        mdf.to_csv(MANIFEST, index=False)
        print(f"\nbuilt={n_ok}  skipped={n_skip}  failed={n_fail}")
        print(f"avg_nodes = {mdf['n_nodes'].mean():.1f}")
        print(f"avg_edges = {mdf['n_edges'].mean():.1f}")
        lc = mdf["label"].value_counts().to_dict()
        print(f"labels   VCC={lc.get(1,0)}  neg={lc.get(0,0)}")
    else:
        print(f"built={n_ok}  skipped={n_skip}  failed={n_fail}")

    print(f"Manifest -> {MANIFEST}")
    print(f"Graphs   -> {OUT_DIR}")


if __name__ == "__main__":
    main()
