from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm

OWNERSHIP_THRESHOLD = 0.05   # Bird et al. (2011) minor-contributor cutoff

# Known automation / bot identities — used to flag authored_by / committed_by edges
BOT_EMAILS: frozenset[str] = frozenset({
    "gardener@tensorflow.org",
})


def _is_bot(email: str) -> float:
    """Return 1.0 if the email belongs to a known automation bot, else 0.0."""
    return 1.0 if email.strip().lower() in BOT_EMAILS else 0.0

# Columns to exclude when auto-detecting raw numeric features
EXCLUDE_PATTERNS = [
    "hash", "repo", "url", "id", "number", "node_id", "fc_hash", "vcc_hash",
    "matched_anchor", "window_since", "window_until", "commit_datetime",
    "created_at", "closed_at", "merged_at", "updated_at", "title", "body",
]

COMMIT_FEAT_COLS = [
    "num_lines_deleted", "num_lines_added", "num_lines_changed",
    "num_files_changed", "dmm_unit_size", "dmm_unit_complexity",
    "dmm_unit_interfacing",
]
FILE_CODE_FEAT_COLS = [
    "num_lines_added", "num_lines_deleted", "num_method_changed",
    "num_lines_of_code", "complexity", "token_count",
]
FUNC_FEAT_COLS = [
    "num_lines_of_code", "complexity", "token_count", "length", "top_nesting_level",
]
DEV_FEAT_COLS = [
    "total_commits", "active_weeks", "total_issues", "total_pull_requests",
]
# Engineered commit-level features for SDLC nodes (mode 2/3)
ENGINEERED_ISSUE_FEATS  = ["issue_open_30d", "issue_open_90d", "issue_open_180d", "issue_age_median", "issue_close_rate_180d"]
ENGINEERED_PR_FEATS     = ["pr_count_30d", "pr_count_90d", "pr_count_180d", "pr_age_median", "pr_merge_or_close_rate_180d"]
ENGINEERED_TAG_FEATS    = ["days_since_prev_tag", "tags_last_180d"]
ENGINEERED_C2I_EDGE     = ["issue_open_90d"]
ENGINEERED_C2PR_EDGE    = ["pr_count_90d"]
ENGINEERED_C2TAG_EDGE   = ["release_cycle_position"]
ENGINEERED_I2PR_EDGE    = ["pr_to_issue_open_ratio_90d", "has_issue_pr_gap"]
ENGINEERED_TAG2X_EDGE   = ["activity_since_last_tag"]

CHANGE_TYPE_CATS = ["ADD", "MODIFY", "DELETE", "RENAME", "COPY", "UNKNOWN"]
CHANGE_TYPE_MAP  = {ct: i for i, ct in enumerate(CHANGE_TYPE_CATS)}

FUNC_CHANGE_TYPE_CATS = ["MODIFY", "ADD", "DELETE", "RENAME", "REFACTOR"]
FUNC_CHANGE_TYPE_MAP  = {ct: i for i, ct in enumerate(FUNC_CHANGE_TYPE_CATS)}

# Engineered features that are counts / durations and need log1p normalisation.
# Rate features (0-1) and binary flags are left as-is.
SDLC_LOG1P_COLS = {
    "issue_open_30d", "issue_open_90d", "issue_open_180d",
    "issue_age_median",
    "pr_count_30d", "pr_count_90d", "pr_count_180d",
    "pr_age_median",
    "days_since_prev_tag", "days_to_next_tag",
    "tags_last_180d",
    "activity_since_last_tag",
}


def build_graph(
    commit_hash: str,
    tables: dict[str, pd.DataFrame],
    mode: int = 1,
    include_files: bool = True,
    include_functions: bool = True,
    include_developers: bool = True,
    include_issues: bool = True,
    include_prs: bool = True,
    include_tags: bool = True,
    ownership_window_days: int = 90,
    ownership_threshold: float = OWNERSHIP_THRESHOLD,
    max_issues: Optional[int] = None,
    max_prs: Optional[int] = None,
    max_tags: Optional[int] = None,
) -> HeteroData:
    """
    Build a HeteroData graph for *commit_hash*.

    Parameters
    ----------
    commit_hash          : target commit SHA
    tables               : dict returned by load_all_tables()
    mode                 : 1 = raw/multiple, 2 = engineered/single, 3 = hybrid
    include_*            : toggle entire node types on/off
    ownership_window_days: window snapshot for developer→file edges (30|90|180)
    ownership_threshold  : min ownership_ratio to materialise a developer node

    Returns
    -------
    HeteroData
    """
    data = HeteroData()

    commit_row = tables["commit_info"][
        tables["commit_info"]["hash"] == commit_hash
    ].reset_index(drop=True)
    if commit_row.empty:
        raise ValueError(f"Commit {commit_hash!r} not found in commit_info.")

    files = tables["file_info"][
        tables["file_info"]["hash"] == commit_hash
    ].reset_index(drop=True)
    # Only keep function rows whose filename has a matching file node.
    # When file_info has no rows for this commit (e.g. clone-failure commits),
    # exclude all functions to prevent unreachable orphan nodes — functions with
    # no file connection carry no message-passing signal in the GNN.
    _known_files = set(files["filename"].tolist())
    # A commit can appear in both FC and VCC roles across different CVEs, so
    # function_info may contain rows with both commit_label values for the same
    # hash.  Restrict to the role this commit actually plays (commit_type from
    # commit_info) so the graph only carries the correct before_change snapshot.
    _commit_type = str(commit_row["commit_type"].values[0]) if not commit_row.empty else "FC"
    _fn_mask = (
        (tables["function_info"]["hash"] == commit_hash) &
        (tables["function_info"]["commit_label"] == _commit_type)
    )
    if _known_files:
        _fn_mask = _fn_mask & tables["function_info"]["filename"].isin(_known_files)
    else:
        # No file nodes for this commit: all functions would be orphans.
        _fn_mask = pd.Series(False, index=tables["function_info"].index)
    funcs = tables["function_info"][_fn_mask].reset_index(drop=True)

    issues = _filter_sdlc(tables.get("issue_info"), commit_hash)
    prs    = _filter_sdlc(tables.get("pr_info"), commit_hash)
    tags   = _filter_sdlc(tables.get("release_tag_info"), commit_hash)

    # Apply per-type limits (most relevant for mode 1 where each entity = one node)
    if max_issues is not None:
        issues = issues.iloc[:max_issues].reset_index(drop=True)
    if max_prs is not None:
        prs = prs.iloc[:max_prs].reset_index(drop=True)
    if max_tags is not None:
        tags = tags.iloc[:max_tags].reset_index(drop=True)

    # Mode 4: narrow tags to the 1 before + 1 after the commit date
    if mode == 4:
        commit_date = pd.to_datetime(
            commit_row["author_date"].values[0], utc=True
        )
        tags = _select_window_tags(tags, commit_date)

    commit_features_row = pd.DataFrame()
    if "commit_features" in tables and tables["commit_features"] is not None:
        commit_features_row = tables["commit_features"][
            tables["commit_features"]["hash"] == commit_hash
        ]

    add_commit_node(data, commit_row, mode, commit_features_row)

    if include_files:
        add_file_nodes(
            data, files,
            tables.get("ownership_window"),
            tables.get("commit_author"),
            commit_hash, ownership_window_days,
        )

    if include_functions and include_files:
        add_function_nodes(data, funcs)

    if include_developers and include_files:
        add_developer_nodes(
            data, commit_hash, files,
            tables.get("ownership_window"),
            tables.get("developer_info"),
            tables.get("commit_author"),
            ownership_window_days, ownership_threshold,
            commit_row=commit_row,
        )

    if include_issues:
        add_issue_nodes(data, issues, mode, commit_features_row)

    if include_prs:
        add_pr_nodes(data, prs, mode, commit_features_row)

    if include_tags:
        add_tag_nodes(data, tags, mode, commit_features_row)

    if include_files:
        add_commit_file_edges(data, commit_row, files)

    if include_functions and include_files:
        add_file_function_edges(data, files, funcs)
        add_function_comod_edges(data, files, funcs)

    if include_developers and include_files:
        add_developer_edges(
            data, commit_hash, files,
            tables.get("ownership_window"),
            tables.get("commit_author"),
            ownership_window_days, ownership_threshold,
        )

    if include_issues:
        add_commit_issue_edges(data, mode, commit_features_row)

    if include_prs:
        add_commit_pr_edges(data, mode, commit_features_row)

    if include_tags:
        add_commit_tag_edges(data, tags, mode, commit_features_row)

    if include_issues or include_prs or include_tags:
        add_sdlc_cross_edges(data, issues, prs, mode,
                             include_issues, include_prs, include_tags,
                             commit_features_row)
        add_temporal_edges(data, issues, prs, tags, mode,
                           include_issues, include_prs, include_tags)

    return data


def add_commit_node(
    data: HeteroData,
    commit_row: pd.DataFrame,
    mode: int = 1,
    commit_features_row: pd.DataFrame = pd.DataFrame(),
) -> None:
    """
    commit node features:
      modes 1/2/3 : 7-dim  (COMMIT_FEAT_COLS — DMM metrics + line/file counts)
      mode 4      : 9-dim  (7 base + log1p(time_since_last_tag) + log1p(tags_last_180d))
                    Engineered tag context is attached here instead of on tag nodes,
                    since in mode 4 tags carry only raw distance features.

    Normalisation: first 4 cols are raw counts (lines/files changed) spanning
    several orders of magnitude — log1p is applied.  DMM cols [4:7] are already
    in [0, 1] and are left as-is.
    """
    base = commit_row[COMMIT_FEAT_COLS].fillna(0).values.astype("float32")
    # log1p-normalise the raw count columns (indices 0-3); DMM [4:7] stay in [0,1]
    base[:, :4] = np.log1p(base[:, :4])

    if mode == 4 and not commit_features_row.empty:
        tag_ctx = np.array([
            _engineered_feats(commit_features_row, ["days_since_prev_tag"]),
            _engineered_feats(commit_features_row, ["tags_last_180d"]),
        ], dtype="float32").reshape(1, 2)   # already log1p'd by _engineered_feats
        data["commit"].x = torch.tensor(
            np.concatenate([base, tag_ctx], axis=1), dtype=torch.float32
        )
    else:
        data["commit"].x = torch.tensor(base, dtype=torch.float32)


def add_file_nodes(
    data: HeteroData,
    files: pd.DataFrame,
    ownership_window: Optional[pd.DataFrame],
    commit_author: Optional[pd.DataFrame],
    commit_hash: str,
    ownership_window_days: int = 90,
) -> None:
    """
    file nodes — 10 features:
      [0:6]  code metrics  (FILE_CODE_FEAT_COLS, log1p-normalised)
      [6:10] ownership distribution (log1p(num_owners), HHI, max_ratio, committer_ratio)

    All FILE_CODE_FEAT_COLS are raw counts (lines, complexity, tokens) — log1p is
    applied so that large files don't dominate gradient magnitude.
    """
    code_feats = np.log1p(files[FILE_CODE_FEAT_COLS].fillna(0).values.astype("float32"))
    own_stats  = _compute_file_ownership_stats(
        files, ownership_window, commit_author, commit_hash, ownership_window_days
    )
    data["file"].x = torch.tensor(
        np.concatenate([code_feats, own_stats], axis=1), dtype=torch.float32
    )


def add_function_nodes(data: HeteroData, funcs: pd.DataFrame) -> None:
    """function nodes — 10 features:
      [0:5] FUNC_FEAT_COLS (loc, complexity, tokens, length, nesting) — log1p normalised
      [5:10] function_change_type one-hot (MODIFY, ADD, DELETE, RENAME, REFACTOR)
    """
    n = len(funcs)
    numeric = np.log1p(funcs[FUNC_FEAT_COLS].fillna(0).values.astype("float32")) if n > 0 else np.zeros((0, len(FUNC_FEAT_COLS)), dtype="float32")
    onehot = np.zeros((n, len(FUNC_CHANGE_TYPE_CATS)), dtype="float32")
    if n > 0 and "function_change_type" in funcs.columns:
        for i, ct in enumerate(funcs["function_change_type"].fillna("MODIFY")):
            idx = FUNC_CHANGE_TYPE_MAP.get(str(ct).strip().upper(), 0)
            onehot[i, idx] = 1.0
    data["function"].x = torch.tensor(np.concatenate([numeric, onehot], axis=1), dtype=torch.float32)


def add_developer_nodes(
    data: HeteroData,
    commit_hash: str,
    files: pd.DataFrame,
    ownership_window: Optional[pd.DataFrame],
    developer_info: Optional[pd.DataFrame],
    commit_author: Optional[pd.DataFrame],
    ownership_window_days: int = 90,
    ownership_threshold: float = OWNERSHIP_THRESHOLD,
    commit_row: Optional[pd.DataFrame] = None,
) -> None:
    """
    developer nodes — 4 features (log1p normalised).
    Inclusion rule: ownership_ratio >= threshold OR commit author (Bird et al. 2011).
    Stores email_to_dev_idx on data for use by add_developer_edges().
    """
    author_emails, committer_emails, own_rows = _get_developer_data(
        commit_hash, ownership_window, commit_author,
        commit_row=commit_row,
        ownership_window_days=ownership_window_days,
        ownership_threshold=ownership_threshold,
    )
    all_commit_emails = list(dict.fromkeys(author_emails + committer_emails))

    all_dev_emails   = list(dict.fromkeys(
        all_commit_emails + (own_rows["_email"].tolist() if not own_rows.empty else [])
    ))
    email_to_dev_idx = {e: i for i, e in enumerate(all_dev_emails)}

    n_devs    = len(all_dev_emails)
    dev_feats = np.zeros((n_devs, len(DEV_FEAT_COLS)), dtype=np.float32)

    if developer_info is not None and n_devs > 0:
        di = developer_info.copy()
        di["_email"] = di["dev_id"].str.strip().str.lower()
        # Keep one row per email — if duplicates exist, take the one with highest total_commits
        if "total_commits" in di.columns:
            di = di.sort_values("total_commits", ascending=False)
        di = di.drop_duplicates(subset=["_email"])
        di = di.set_index("_email")
        for email, idx in email_to_dev_idx.items():
            if email in di.index:
                vals = di.loc[email, DEV_FEAT_COLS].fillna(0).values.astype("float64")
                dev_feats[idx] = np.log1p(vals)

    data["developer"].x = torch.tensor(dev_feats, dtype=torch.float32)
    # Stash for edge builder (avoids recomputing)
    data["developer"]._email_to_idx     = email_to_dev_idx
    data["developer"]._author_emails    = author_emails
    data["developer"]._committer_emails = committer_emails
    data["developer"]._own_rows         = own_rows


def add_issue_nodes(
    data: HeteroData,
    issues: pd.DataFrame,
    mode: int,
    commit_features_row: pd.DataFrame,
) -> None:
    """
    issue nodes.
    mode 1: one node per issue, raw numeric features
    mode 2/3: one aggregate node, engineered features from commit_features
    """
    if issues.empty:
        n_feats = len(ENGINEERED_ISSUE_FEATS) if mode in (2, 3) else 1
        data["issue"].x = torch.zeros((0, n_feats), dtype=torch.float32)
        return

    if mode == 1:
        feats, _ = _raw_numeric_features(issues)
        data["issue"].x = torch.tensor(feats, dtype=torch.float32)
    else:  # mode 2 or 3 — single aggregate node
        feats = _engineered_feats(commit_features_row, ENGINEERED_ISSUE_FEATS)
        data["issue"].x = torch.tensor(feats.reshape(1, -1), dtype=torch.float32)


def add_pr_nodes(
    data: HeteroData,
    prs: pd.DataFrame,
    mode: int,
    commit_features_row: pd.DataFrame,
) -> None:
    """
    pull_request nodes.
    mode 1: one node per PR, raw numeric features
    mode 2/3: one aggregate node, engineered features
    """
    if prs.empty:
        n_feats = len(ENGINEERED_PR_FEATS) if mode in (2, 3) else 1
        data["pull_request"].x = torch.zeros((0, n_feats), dtype=torch.float32)
        return

    if mode == 1:
        feats, _ = _raw_numeric_features(prs)
        data["pull_request"].x = torch.tensor(feats, dtype=torch.float32)
    else:
        feats = _engineered_feats(commit_features_row, ENGINEERED_PR_FEATS)
        data["pull_request"].x = torch.tensor(feats.reshape(1, -1), dtype=torch.float32)


def add_tag_nodes(
    data: HeteroData,
    tags: pd.DataFrame,
    mode: int,
    commit_features_row: pd.DataFrame,
) -> None:
    """
    release_tag nodes.
    mode 1/3: one node per tag, raw numeric features
    mode 2:   one aggregate node, engineered features
    mode 4:   up to 2 nodes (prev_tag + next_tag), 7-dim via parse_tag_features().
              Source: commit_features_row (prev_tag_name, next_tag_name,
              days_since_prev_tag, days_to_next_tag) — covers ALL commits
              including those with no rows in release_tag_info_v4.
    """
    TAG_FEAT_KEYS = ["major", "minor", "patch", "is_prerelease", "is_hotfix", "is_major_bump", "days_distance"]

    if mode == 4:
        rows = []
        if not commit_features_row.empty:
            cfr = commit_features_row.iloc[0]
            prev_name = cfr.get("prev_tag_name", None)
            next_name = cfr.get("next_tag_name", None)
            prev_days = cfr.get("days_since_prev_tag", None)
            next_days = cfr.get("days_to_next_tag", None)
            # For normal commits, prev_tag_name may be NaN even when days_since_prev_tag>0
            # (tag names not captured in the simplified normal-commit SDLC pipeline).
            # Fall back to a placeholder name so we still get a node with temporal distance.
            _prev_name_eff = prev_name if pd.notna(prev_name) else (
                "v0.0" if (pd.notna(prev_days) and float(prev_days) > 0) else None
            )
            _next_name_eff = next_name if pd.notna(next_name) else (
                "v0.0" if (pd.notna(next_days) and float(next_days) > 0) else None
            )
            prev_parsed = parse_tag_features(
                _prev_name_eff,
                -float(prev_days) if pd.notna(prev_days) else 0.0,  # negative = before
            )
            next_parsed = parse_tag_features(
                _next_name_eff,
                float(next_days) if pd.notna(next_days) else 0.0,   # positive = after
            )
            if prev_parsed is not None:
                rows.append([prev_parsed[k] for k in TAG_FEAT_KEYS])
            if next_parsed is not None:
                rows.append([next_parsed[k] for k in TAG_FEAT_KEYS])
        if rows:
            data["release_tag"].x = torch.tensor(np.array(rows, dtype="float32"), dtype=torch.float32)
        else:
            data["release_tag"].x = torch.zeros((0, len(TAG_FEAT_KEYS)), dtype=torch.float32)
        return

    if tags.empty:
        n_feats = len(ENGINEERED_TAG_FEATS) if mode == 2 else 1
        data["release_tag"].x = torch.zeros((0, n_feats), dtype=torch.float32)
        return

    if mode == 2:
        feats = _engineered_feats(commit_features_row, ENGINEERED_TAG_FEATS)
        data["release_tag"].x = torch.tensor(feats.reshape(1, -1), dtype=torch.float32)
    else:  # mode 1 or 3 — multiple raw nodes
        feats, _ = _raw_numeric_features(tags)
        data["release_tag"].x = torch.tensor(feats, dtype=torch.float32)


def add_commit_file_edges(
    data: HeteroData,
    commit_row: pd.DataFrame,
    files: pd.DataFrame,
) -> None:
    """
    (commit, has, file) — 10 features:
    change-type one-hot [6] + lines_add_ratio + lines_del_ratio + method_ratio + path_changed
    """
    n = len(files)
    src = torch.zeros(n, dtype=torch.long)
    dst = torch.arange(n, dtype=torch.long)

    total_add  = max(commit_row["num_lines_added"].fillna(0).values[0], 1)
    total_del  = max(commit_row["num_lines_deleted"].fillna(0).values[0], 1)
    total_meth = max(files["num_method_changed"].fillna(0).sum(), 1)

    feats = []
    for _, frow in files.iterrows():
        ct = str(frow.get("change_type", "UNKNOWN")).upper()
        oh = [0.0] * 6
        oh[CHANGE_TYPE_MAP.get(ct, 5)] = 1.0
        old_p = str(frow.get("old_path", ""))
        new_p = str(frow.get("new_path", ""))
        feats.append(oh + [
            frow["num_lines_added"]   / total_add,
            frow["num_lines_deleted"] / total_del,
            frow.get("num_method_changed", 0) / total_meth,
            float(bool(old_p) and bool(new_p) and old_p != new_p),
        ])

    data["commit", "has", "file"].edge_index = torch.stack([src, dst])
    data["commit", "has", "file"].edge_attr  = torch.tensor(feats, dtype=torch.float32)


def add_file_function_edges(
    data: HeteroData,
    files: pd.DataFrame,
    funcs: pd.DataFrame,
) -> None:
    """
    (file, has, function) — 7 features:
    loc_frac, complexity_ratio, token_ratio, position, num_params, before_change, is_hunk
    is_hunk=1.0 for diff-window hunk fallback nodes, 0.0 for Lizard-parsed functions.
    """
    src, dst, feats = [], [], []
    if len(funcs) > 0:
        fn_to_file = {row["filename"]: fi for fi, row in files.iterrows()}
        for fni, fnrow in funcs.iterrows():
            fi = fn_to_file.get(fnrow["filename"])
            if fi is None:
                continue
            frow = files.iloc[fi]
            n_fn = len(funcs[funcs["filename"] == fnrow["filename"]])
            pos  = list(funcs[funcs["filename"] == fnrow["filename"]].index).index(fni)
            loc_frac         = min(fnrow["num_lines_of_code"] / max(frow["num_lines_of_code"], 1), 1.0)
            # Ratios can exceed 1.0 when the file-level metric is 0 (e.g. header
            # files where Lizard reports complexity=0 at file scope).  Clamp at 5.0
            # to keep the feature bounded while preserving relative ordering.
            complexity_ratio = min(fnrow["complexity"]  / max(frow["complexity"], 1),  5.0)
            token_ratio      = min(fnrow["token_count"] / max(frow["token_count"], 1), 5.0)
            position         = pos / max(n_fn - 1, 1)
            try:
                params = len(fnrow.get("parameters", []) or [])
            except TypeError:
                params = 0
            before_change = float(fnrow.get("before_change", False))
            is_hunk       = float(fnrow.get("is_hunk", False))
            src.append(fi); dst.append(fni)
            feats.append([loc_frac, complexity_ratio, token_ratio,
                          position, float(np.log1p(params)), before_change, is_hunk])

    if src:
        data["file", "has", "function"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["file", "has", "function"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["file", "has", "function"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["file", "has", "function"].edge_attr  = torch.zeros((0, 7), dtype=torch.float32)


def add_function_comod_edges(
    data: HeteroData,
    files: pd.DataFrame,
    funcs: pd.DataFrame,
) -> None:
    """
    (function, co_modified, function) — 2 features: proximity, complexity_similarity
    Bidirectional. Only between functions in the same file.
    """
    src, dst, feats = [], [], []
    if len(funcs) > 0:
        for _, frow in files.iterrows():
            fn_in_file = funcs[funcs["filename"] == frow["filename"]].reset_index()
            if len(fn_in_file) < 2:
                continue
            for (ia, row_a), (ib, row_b) in combinations(fn_in_file.iterrows(), 2):
                ga = fn_in_file.loc[ia, "index"]
                gb = fn_in_file.loc[ib, "index"]
                gap      = abs(row_a.get("start_line", 0) - row_b.get("start_line", 0))
                prox     = 1.0 / (1.0 + gap)
                ca, cb   = row_a.get("complexity", 0) or 0, row_b.get("complexity", 0) or 0
                cplx_sim = 1.0 - abs(ca - cb) / max(ca + cb, 1)
                for s, d in [(ga, gb), (gb, ga)]:
                    src.append(s); dst.append(d); feats.append([prox, cplx_sim])

    if src:
        data["function", "co_modified", "function"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["function", "co_modified", "function"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["function", "co_modified", "function"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["function", "co_modified", "function"].edge_attr  = torch.zeros((0, 2), dtype=torch.float32)


def add_developer_edges(
    data: HeteroData,
    commit_hash: str,
    files: pd.DataFrame,
    ownership_window: Optional[pd.DataFrame],
    commit_author: Optional[pd.DataFrame],
    ownership_window_days: int = 90,
    ownership_threshold: float = OWNERSHIP_THRESHOLD,
) -> None:
    """
    (commit, authored_by,  developer) — 1 feature: is_bot (1=known automation bot)
    (commit, committed_by, developer) — 1 feature: is_bot
    (developer, owns, file)           — 3 features: ownership_ratio, norm_lines, log_edits
    Reads stashed data from add_developer_nodes() if available.
    """
    # Retrieve stashed data (set by add_developer_nodes).
    # Use data["developer"] directly — data.get() in PyG's HeteroData may return a
    # fresh default object rather than the NodeStorage, causing hasattr to miss the stash.
    if hasattr(data["developer"], "_email_to_idx"):
        email_to_dev_idx  = data["developer"]._email_to_idx
        author_emails     = data["developer"]._author_emails
        committer_emails  = data["developer"]._committer_emails
        own_rows          = data["developer"]._own_rows
    else:
        author_emails, committer_emails, own_rows = _get_developer_data(
            commit_hash, ownership_window, commit_author,
            ownership_window_days=ownership_window_days,
            ownership_threshold=ownership_threshold,
        )
        all_commit_emails = list(dict.fromkeys(author_emails + committer_emails))
        all_dev_emails    = list(dict.fromkeys(
            all_commit_emails + (own_rows["_email"].tolist() if not own_rows.empty else [])
        ))
        email_to_dev_idx  = {e: i for i, e in enumerate(all_dev_emails)}

    def _build_c2d(emails: list[str]):
        src, dst, attr = [], [], []
        for email in emails:
            if email in email_to_dev_idx:
                src.append(0)
                dst.append(email_to_dev_idx[email])
                attr.append([_is_bot(email)])
        return src, dst, attr

    for et, emails in [
        ("authored_by",  author_emails),
        ("committed_by", committer_emails),
    ]:
        src, dst, attr = _build_c2d(emails)
        if src:
            data["commit", et, "developer"].edge_index = torch.tensor([src, dst], dtype=torch.long)
            data["commit", et, "developer"].edge_attr  = torch.tensor(attr, dtype=torch.float32)
        else:
            data["commit", et, "developer"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["commit", et, "developer"].edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    new_path_to_fi = {
        str(frow["new_path"]): fi
        for fi, frow in files.iterrows()
        if str(frow.get("new_path", ""))
    }
    d2f_src, d2f_dst, d2f_feats = [], [], []
    if not own_rows.empty:
        for _, orow in own_rows.iterrows():
            email    = str(orow["_email"])
            file_idx = new_path_to_fi.get(str(orow.get("file_path", "")))
            if email not in email_to_dev_idx or file_idx is None:
                continue
            own_r  = float(orow.get("ownership_ratio", 0) or 0)
            tot_l  = float(orow.get("total_lines", 1) or 1)
            own_l  = float(orow.get("lines_owned", 0) or 0)
            edits  = float(orow.get("edits_in_window", 0) or 0)
            d2f_src.append(email_to_dev_idx[email])
            d2f_dst.append(file_idx)
            d2f_feats.append([own_r, own_l / max(tot_l, 1), np.log1p(edits)])

    if d2f_src:
        data["developer", "owns", "file"].edge_index = torch.tensor([d2f_src, d2f_dst], dtype=torch.long)
        data["developer", "owns", "file"].edge_attr  = torch.tensor(d2f_feats, dtype=torch.float32)
    else:
        data["developer", "owns", "file"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["developer", "owns", "file"].edge_attr  = torch.zeros((0, 3), dtype=torch.float32)


def add_commit_issue_edges(
    data: HeteroData,
    mode: int,
    commit_features_row: pd.DataFrame,
) -> None:
    """
    (commit, linked_issue, issue) and reverse.
    mode 1: one edge per issue (no attr)
    mode 2/3: one edge to single aggregate node, attr = engineered edge features
    """
    n = data["issue"].x.shape[0]
    if n == 0:
        _empty_edge(data, "commit", "linked_issue", "issue")
        _empty_edge(data, "issue", "issue_of", "commit")
        return

    ei = torch.stack([torch.zeros(n, dtype=torch.long), torch.arange(n, dtype=torch.long)])
    data["commit", "linked_issue", "issue"].edge_index = ei
    data["issue", "issue_of", "commit"].edge_index     = ei.flip(0)

    if mode in (2, 3) and not commit_features_row.empty:
        ea = _engineered_edge_attr(commit_features_row, ENGINEERED_C2I_EDGE, n)
        data["commit", "linked_issue", "issue"].edge_attr = ea
        data["issue", "issue_of", "commit"].edge_attr     = ea


def add_commit_pr_edges(
    data: HeteroData,
    mode: int,
    commit_features_row: pd.DataFrame,
) -> None:
    """(commit, linked_pr, pull_request) and reverse."""
    n = data["pull_request"].x.shape[0]
    if n == 0:
        _empty_edge(data, "commit", "linked_pr", "pull_request")
        _empty_edge(data, "pull_request", "pr_of", "commit")
        return

    ei = torch.stack([torch.zeros(n, dtype=torch.long), torch.arange(n, dtype=torch.long)])
    data["commit", "linked_pr", "pull_request"].edge_index = ei
    data["pull_request", "pr_of", "commit"].edge_index     = ei.flip(0)

    if mode in (2, 3) and not commit_features_row.empty:
        ea = _engineered_edge_attr(commit_features_row, ENGINEERED_C2PR_EDGE, n)
        data["commit", "linked_pr", "pull_request"].edge_attr = ea
        data["pull_request", "pr_of", "commit"].edge_attr     = ea


def add_commit_tag_edges(
    data: HeteroData,
    tags: pd.DataFrame,
    mode: int,
    commit_features_row: pd.DataFrame,
) -> None:
    """
    (commit, has_release_tag, release_tag) and reverse.
    edge attr:
      mode 2/3 : release_cycle_position  (1-dim)
      mode 4   : no edge attr (features are in the tag nodes themselves)
    """
    n = data["release_tag"].x.shape[0]
    if n == 0:
        _empty_edge(data, "commit", "has_release_tag", "release_tag")
        _empty_edge(data, "release_tag", "tag_of", "commit")
        return

    ei = torch.stack([torch.zeros(n, dtype=torch.long), torch.arange(n, dtype=torch.long)])
    data["commit", "has_release_tag", "release_tag"].edge_index = ei
    data["release_tag", "tag_of", "commit"].edge_index          = ei.flip(0)

    if mode in (2, 3) and not commit_features_row.empty:
        ea = _engineered_edge_attr(commit_features_row, ENGINEERED_C2TAG_EDGE, n)
        data["commit", "has_release_tag", "release_tag"].edge_attr = ea
        data["release_tag", "tag_of", "commit"].edge_attr          = ea
    # mode 4: no edge attr needed — distance info is embedded in tag node features


def add_sdlc_cross_edges(
    data: HeteroData,
    issues: pd.DataFrame,
    prs: pd.DataFrame,
    mode: int,
    include_issues: bool,
    include_prs: bool,
    include_tags: bool,
    commit_features_row: pd.DataFrame = pd.DataFrame(),
) -> None:
    """
    PR ↔ issue, tag → PR, tag → issue cross edges.
    mode 1: attempt structural PR→issue matching; multiple edges
    mode 2/3: single aggregate node per type → single edges

    Edge attrs (mode 2/3):
      PR ↔ issue  : pr_to_issue_open_ratio_90d, has_issue_pr_gap
      tag → PR/issue: activity_since_last_tag
    """
    n_issues = data["issue"].x.shape[0] if include_issues else 0
    n_prs    = data["pull_request"].x.shape[0] if include_prs else 0
    n_tags   = data["release_tag"].x.shape[0] if include_tags else 0

    # PR ↔ issue
    if include_prs and include_issues and n_prs > 0 and n_issues > 0:
        if mode == 1:
            ei = _match_pr_issue(prs, issues)
        else:
            ei = torch.zeros((2, 1), dtype=torch.long)  # single node → single node
        if ei.shape[1] > 0:
            data["pull_request", "references_issue", "issue"].edge_index = ei
            data["issue", "referenced_by_pr", "pull_request"].edge_index = ei.flip(0)
            if mode in (2, 3) and not commit_features_row.empty:
                ea = _engineered_edge_attr(commit_features_row, ENGINEERED_I2PR_EDGE, ei.shape[1])
                data["pull_request", "references_issue", "issue"].edge_attr = ea
                data["issue", "referenced_by_pr", "pull_request"].edge_attr = ea

    # tag → PR  (mode 4: moved to commit→PR via tag context)
    if include_tags and include_prs and n_tags > 0 and n_prs > 0:
        if mode == 4:
            # tag→PR replaced by commit→PR with activity_since_last_tag as edge attr
            ei = torch.zeros((2, 1), dtype=torch.long)  # commit_0 → pr_0
            data["commit", "tag_context_pr", "pull_request"].edge_index = ei
            data["pull_request", "tag_context_of", "commit"].edge_index = ei.flip(0)
            if not commit_features_row.empty:
                ea = _engineered_edge_attr(commit_features_row, ENGINEERED_TAG2X_EDGE, 1)
                data["commit", "tag_context_pr", "pull_request"].edge_attr = ea
                data["pull_request", "tag_context_of", "commit"].edge_attr = ea
        else:
            if mode in (1, 3):
                ei = torch.stack([torch.arange(n_tags, dtype=torch.long),
                                  torch.zeros(n_tags, dtype=torch.long)])
            else:
                ei = torch.zeros((2, 1), dtype=torch.long)
            data["release_tag", "affects_pr", "pull_request"].edge_index = ei
            data["pull_request", "in_release", "release_tag"].edge_index = ei.flip(0)
            if mode in (2, 3) and not commit_features_row.empty:
                ea = _engineered_edge_attr(commit_features_row, ENGINEERED_TAG2X_EDGE, ei.shape[1])
                data["release_tag", "affects_pr", "pull_request"].edge_attr = ea
                data["pull_request", "in_release", "release_tag"].edge_attr = ea

    # tag → issue  (mode 4: moved to commit→issue via tag context)
    if include_tags and include_issues and n_tags > 0 and n_issues > 0:
        if mode == 4:
            ei = torch.zeros((2, 1), dtype=torch.long)  # commit_0 → issue_0
            data["commit", "tag_context_issue", "issue"].edge_index = ei
            data["issue", "tag_context_of", "commit"].edge_index    = ei.flip(0)
            if not commit_features_row.empty:
                ea = _engineered_edge_attr(commit_features_row, ENGINEERED_TAG2X_EDGE, 1)
                data["commit", "tag_context_issue", "issue"].edge_attr = ea
                data["issue", "tag_context_of", "commit"].edge_attr    = ea
        else:
            if mode in (1, 3):
                ei = torch.stack([torch.arange(n_tags, dtype=torch.long),
                                  torch.zeros(n_tags, dtype=torch.long)])
            else:
                ei = torch.zeros((2, 1), dtype=torch.long)
            data["release_tag", "affects_issue", "issue"].edge_index = ei
            data["issue", "in_release", "release_tag"].edge_index    = ei.flip(0)
            if mode in (2, 3) and not commit_features_row.empty:
                ea = _engineered_edge_attr(commit_features_row, ENGINEERED_TAG2X_EDGE, ei.shape[1])
                data["release_tag", "affects_issue", "issue"].edge_attr = ea
                data["issue", "in_release", "release_tag"].edge_attr    = ea


def add_temporal_edges(
    data: HeteroData,
    issues: pd.DataFrame,
    prs: pd.DataFrame,
    tags: pd.DataFrame,
    mode: int,
    include_issues: bool,
    include_prs: bool,
    include_tags: bool,
) -> None:
    """
    Temporal chain edges (entity_i → entity_{i+1} sorted by timestamp).
    mode 1:   issues, PRs, tags all get temporal chains
    mode 2:   none (single aggregate nodes)
    mode 3:   tags only
    mode 4:   single before→after edge between the two window tags
    """
    if mode == 2:
        return

    if mode in (1,) and include_issues:
        ei = _temporal_chain(issues, ["created_at", "issue_created_at"])
        if ei.shape[1] > 0:
            data["issue", "next_issue", "issue"].edge_index = ei
            data["issue", "prev_issue", "issue"].edge_index = ei.flip(0)

    if mode in (1,) and include_prs:
        ei = _temporal_chain(prs, ["created_at", "pr_created_at"])
        if ei.shape[1] > 0:
            data["pull_request", "next_pr", "pull_request"].edge_index = ei
            data["pull_request", "prev_pr", "pull_request"].edge_index = ei.flip(0)

    if include_tags and data["release_tag"].x.shape[0] > 1:
        if mode == 4:
            # before tag (index 0) → after tag (index 1) — fixed order from _select_window_tags
            n_tags = data["release_tag"].x.shape[0]
            if n_tags == 2:
                ei = torch.tensor([[0], [1]], dtype=torch.long)
                data["release_tag", "next_tag", "release_tag"].edge_index = ei
                data["release_tag", "prev_tag", "release_tag"].edge_index = ei.flip(0)
        else:
            ei = _temporal_chain(tags, ["created_at", "tagged_at", "published_at"])
            if ei.shape[1] > 0:
                data["release_tag", "next_tag", "release_tag"].edge_index = ei
                data["release_tag", "prev_tag", "release_tag"].edge_index = ei.flip(0)


def load_all_tables(
    base_path: str = "..",
    commit_hashes: Optional[list[str]] = None,
    use_full: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load all required DataFrames from base_path (thesis root directory).

    use_full=True (default): loads *_full.csv tables that include VCC + FC + normal commits.
    use_full=False: legacy behaviour loading VCC/FC-only tables (for demos/visualisation).

    Function/hunk data loading priority (use_full=True):
      1. function_info_full.csv (data_new/processed_v4_fixed/) — 269k rows, all commit types
      2. Falls back to v4_merged → v3 if not found

    Canonical filter (per DATA_README.md):
      - FC/normal: before_change=False  (post-change / unrelated state)
      - VCC+ADD:   before_change=False  (newly added function IS the vulnerability)
      - VCC+other: before_change=True   (pre-change snapshot = vulnerable state)
    v3 note: in v3, VCC+ADD rows are stored with before_change=True (different convention
             than v4). The filter adapts automatically based on which source is loaded.

    Loading rules:
      - file_info  : apply defensive change_type normalisation; skip large text cols (diff/code)
      - function_info : canonical filter above; dedup on [hash, commit_label, name, filename, start_line, before_change]
      - hunk_info  : merged unconditionally; is_hunk=True on hunk rows
      - commit_features : merges old VCC/FC schema (28-col) with normal schema (16-col),
                          aligning column names (time_since_last_tag → days_since_prev_tag)
    """
    import os
    p = base_path
    g = os.path.join(p, "data", "graph_data")   # canonical graph-ready tables

    tables: dict[str, Optional[pd.DataFrame]] = {}
    _hashes_set: Optional[set] = set(commit_hashes) if commit_hashes is not None else None

    def _load(path: str, key: str) -> None:
        if os.path.exists(path):
            tables[key] = pd.read_csv(path)
        else:
            print(f"  [warning] {key} not found at {path}")
            tables[key] = None

    def _load_sdlc(path: str, key: str, drop_cols: Optional[list[str]] = None) -> None:
        """Load an SDLC table, optionally dropping large unused columns and
        pre-filtering to only the rows matching the requested commit hashes.
        Pre-filtering cuts memory from O(all CVEs) to O(requested commits)."""
        if not os.path.exists(path):
            print(f"  [warning] {key} not found at {path}")
            tables[key] = None
            return
        usecols = None
        if drop_cols:
            all_cols = pd.read_csv(path, nrows=0).columns.tolist()
            usecols = [c for c in all_cols if c not in drop_cols]
        df = pd.read_csv(path, usecols=usecols)
        if _hashes_set is not None:
            mask = pd.Series(False, index=df.index)
            for col in ("fc_hash", "vcc_hash"):
                if col in df.columns:
                    mask |= df[col].isin(_hashes_set)
            df = df[mask].reset_index(drop=True)
        tables[key] = df

    # ── commit-level ──────────────────────────────────────────────────────────
    _ci_full = os.path.join(g, "commit_info_full.csv")
    _ci_base = os.path.join(g, "commit_info.csv")
    _load(_ci_full if (use_full and os.path.exists(_ci_full)) else _ci_base, "commit_info")

    # ── file_info ─────────────────────────────────────────────────────────────
    # use_full: load file_info_full.csv (VCC+FC+normal, ~124k rows) from data_new/.
    #   Skip large text columns (diff, diff_parsed, code_after, code_before) to
    #   avoid loading 9+ GB into memory.
    # legacy: load file_info_new.csv (v3, VCC+FC only) with v2 gap-fill.
    _FI_USECOLS = [
        "hash", "commit_label", "filename", "old_path", "new_path", "change_type",
        "num_lines_added", "num_lines_deleted", "num_method_changed",
        "num_lines_of_code", "complexity", "token_count",
    ]
    fi_full_path = os.path.join(p, "data_new/processed_v4_fixed/file_info_full.csv")
    fi_v3_path   = os.path.join(g, "file_info_new.csv")

    if use_full and os.path.exists(fi_full_path):
        _all_cols = pd.read_csv(fi_full_path, nrows=0).columns.tolist()
        _usecols  = [c for c in _FI_USECOLS if c in _all_cols]
        fi_df = pd.read_csv(fi_full_path, usecols=_usecols, low_memory=False)
        fi_df["change_type"] = (
            fi_df["change_type"]
            .str.replace("ModificationType.", "", regex=False)
            .str.replace("MODIFIED", "MODIFY", regex=False)
        )
        fi_df = fi_df.drop_duplicates(subset=["hash", "commit_label", "filename"])
        tables["file_info"] = fi_df.reset_index(drop=True)
        print(f"  [file_info] loaded full: {len(fi_df)} rows")
    elif os.path.exists(fi_v3_path):
        fi_df = pd.read_csv(fi_v3_path)
        fi_df["change_type"] = (
            fi_df["change_type"]
            .str.replace("ModificationType.", "", regex=False)
            .str.replace("MODIFIED", "MODIFY", regex=False)
        )
        fi_df = fi_df.drop_duplicates(subset=["hash", "commit_label", "filename"])

        # v2 fallback: supplement VCC commits absent from v3 file_info
        v2_fi_path = os.path.join(p, "data/processed_v2/file_info_v2_merged.csv")
        if os.path.exists(v2_fi_path):
            v2_fi = pd.read_csv(v2_fi_path)
            v2_fi["change_type"] = (
                v2_fi["change_type"]
                .str.replace("ModificationType.", "", regex=False)
                .str.replace("MODIFIED", "MODIFY", regex=False)
            )
            hashes_in_fi = set(fi_df["hash"].unique())
            v2_fi_gap = v2_fi[
                (v2_fi["commit_label"] == "VCC") &
                (~v2_fi["hash"].isin(hashes_in_fi))
            ].copy()
            if len(v2_fi_gap) > 0:
                shared = [c for c in fi_df.columns if c in v2_fi_gap.columns]
                fi_df = pd.concat([fi_df, v2_fi_gap[shared]], ignore_index=True)
                print(f"  [v2 fallback] added {len(v2_fi_gap)} file rows from v2 for {v2_fi_gap['hash'].nunique()} gap commits")

        tables["file_info"] = fi_df.reset_index(drop=True)
    else:
        print(f"  [warning] file_info not found at {fi_full_path} or {fi_v3_path}")
        tables["file_info"] = None

    # ── function_info + hunk_info ─────────────────────────────────────────────
    # use_full: prefer function_info_full.csv (269k rows, all commit types)
    fn_full_path = os.path.join(p, "data_new/processed_v4_fixed/function_info_full.csv")
    v4_fn_path   = os.path.join(p, "data_new/processed_v4_fixed/function_info_v4_merged.csv")
    v3_fn_path   = os.path.join(g, "function_info_new.csv")
    v4_hunk_path = os.path.join(p, "data_new/processed_v4_fixed/hunk_info_v4_merged.csv")
    v3_hunk_path = os.path.join(p, "data_new/processed_v3/hunk_info_v3_merged.csv")

    # Select primary source
    if use_full and os.path.exists(fn_full_path):
        fn_path   = fn_full_path
        hunk_path = v4_hunk_path if os.path.exists(v4_hunk_path) else v3_hunk_path
        is_v4     = True
        print("  [function_info] loading full (VCC+FC+normal)")
    elif os.path.exists(v4_fn_path):
        fn_path   = v4_fn_path
        hunk_path = v4_hunk_path if os.path.exists(v4_hunk_path) else v3_hunk_path
        is_v4     = True
        print("  [function_info] loading v4")
    elif os.path.exists(v3_fn_path):
        fn_path   = v3_fn_path
        hunk_path = v3_hunk_path
        is_v4     = False
        print("  [function_info] v4 not found, falling back to v3")
    else:
        fn_path = None
        is_v4   = False

    if fn_path and os.path.exists(fn_path):
        fn_df = pd.read_csv(fn_path)

        # Canonical filter:
        #   FC:        before_change=False  (after-fix / patched state)
        #   VCC+ADD:   before_change=False  (the newly-added function IS the vulnerability)
        #              In v4, ADD functions at VCC commits have only before_change=False.
        #              In v3, ADD functions at VCC commits have before_change=True (different
        #              convention) — handled by the VCC+other branch below.
        #   VCC+other: before_change=True   (pre-change snapshot = vulnerable state)
        if is_v4 and "function_change_type" in fn_df.columns:
            _vcc_add_mask = (
                (fn_df["commit_label"] == "VCC") &
                (fn_df["function_change_type"] == "ADD") &
                (fn_df["before_change"] == False)
            )
            # VCC+RENAME+before_change=False captures the renamed-to function name.
            # Currently 0 such rows exist in v4; they are synthesised below from v2.
            _vcc_rename_mask = (
                (fn_df["commit_label"] == "VCC") &
                (fn_df["function_change_type"] == "RENAME") &
                (fn_df["before_change"] == False)
            )
            fn_canon = fn_df[
                ((fn_df["commit_label"] == "FC")     & (fn_df["before_change"] == False)) |
                ((fn_df["commit_label"] == "normal") & (fn_df["before_change"] == False)) |
                ((fn_df["commit_label"] == "VCC")    & (fn_df["before_change"] == True)) |
                _vcc_add_mask |
                _vcc_rename_mask
            ].drop_duplicates(
                subset=["hash", "commit_label", "name", "filename", "start_line", "before_change"]
            )
        else:
            # v3: VCC+ADD stored with before_change=True; old filter is correct
            fn_canon = fn_df[
                ((fn_df["commit_label"] == "FC")  & (fn_df["before_change"] == False)) |
                ((fn_df["commit_label"] == "VCC") & (fn_df["before_change"] == True))
            ].drop_duplicates(
                subset=["hash", "commit_label", "name", "filename", "start_line", "before_change"]
            )

        fn_canon = fn_canon.assign(is_hunk=False)

        # ── RENAME supplement: synthesise the renamed-to node from v2 data ───
        # For VCC+RENAME commits, v4 only stores the old name (before_change=True).
        # If v2 has a row for the same commit+file with a different name, that is
        # the renamed-to function.  We synthesise a before_change=False row so
        # fn_evolution can chain old-name → new-name within the same commit.
        if is_v4:
            v2_fn_path_pre = os.path.join(p, "data/processed_v2/function_info_v2_merged.csv")
            if os.path.exists(v2_fn_path_pre):
                _v2_sup = pd.read_csv(v2_fn_path_pre)
                _rename_src = fn_canon[
                    (fn_canon["commit_label"] == "VCC") &
                    (fn_canon["function_change_type"] == "RENAME") &
                    (fn_canon["before_change"] == True)
                ]
                _syn_rows = []
                for _, rrow in _rename_src.iterrows():
                    # Skip if a before_change=False row already exists for this commit+file
                    _existing = fn_canon[
                        (fn_canon["hash"] == rrow["hash"]) &
                        (fn_canon["filename"] == rrow["filename"]) &
                        (fn_canon["before_change"] == False)
                    ]
                    if len(_existing) > 0:
                        continue
                    # Look for a single v2 row in the same commit+file with a different name
                    _v2_cands = _v2_sup[
                        (_v2_sup["hash"] == rrow["hash"]) &
                        (_v2_sup["filename"] == rrow["filename"]) &
                        (_v2_sup["name"] != rrow["name"])
                    ]
                    if len(_v2_cands) == 1:
                        _new = rrow.copy()
                        _new["name"] = _v2_cands.iloc[0]["name"]
                        _new["before_change"] = False
                        for _col in ["num_lines_of_code", "complexity", "token_count", "length", "top_nesting_level"]:
                            if _col in _v2_cands.columns:
                                _new[_col] = _v2_cands.iloc[0][_col]
                        _syn_rows.append(_new)
                if _syn_rows:
                    _syn_df = pd.DataFrame(_syn_rows)
                    fn_canon = pd.concat([fn_canon, _syn_df], ignore_index=True)
                    print(f"  [RENAME supplement] added {len(_syn_df)} renamed-to rows for "
                          f"{_syn_df['hash'].nunique()} VCC RENAME commits")

        if os.path.exists(hunk_path):
            hunk_df = pd.read_csv(hunk_path)
            hunk_df = hunk_df.drop_duplicates(
                subset=["hash", "commit_label", "name", "filename", "start_line", "before_change"]
            ).assign(is_hunk=True)
            shared_cols = [c for c in fn_canon.columns if c in hunk_df.columns]
            fn_merged = pd.concat(
                [fn_canon, hunk_df[shared_cols]], ignore_index=True
            )
        else:
            fn_merged = fn_canon.reset_index(drop=True)

        # ── v2 fallback: supplement VCC commits absent from v4/v3 ─────────────
        # Some VCC commits have manually-inserted rows in the old v2 file but
        # were not carried over into v3/v4.  Add them back.
        # function_change_type is inferred from before/after row presence:
        #   only before_change=True  → DELETE
        #   only before_change=False → ADD
        #   both                     → MODIFY
        v2_fn_path = os.path.join(p, "data/processed_v2/function_info_v2_merged.csv")
        if os.path.exists(v2_fn_path):
            v2_df = pd.read_csv(v2_fn_path)
            vcc_hashes_in_merged = set(
                fn_merged.loc[fn_merged["commit_label"] == "VCC", "hash"].unique()
            )
            gap_mask = (v2_df["commit_label"] == "VCC") & (~v2_df["hash"].isin(vcc_hashes_in_merged))
            if gap_mask.any():
                def _infer_change_type(grp: pd.DataFrame) -> str:
                    bc_vals = set(grp["before_change"].tolist())
                    if bc_vals == {True}:
                        return "DELETE"
                    elif bc_vals == {False}:
                        return "ADD"
                    return "MODIFY"

                ct_map = (
                    v2_df[gap_mask]
                    .groupby(["hash", "name"])
                    .apply(_infer_change_type)
                    .rename("function_change_type")
                    .reset_index()
                )
                # v2 convention: keep before_change=True for VCC canonical rows
                v2_vcc = v2_df[gap_mask & (v2_df["before_change"] == True)].copy()
                v2_vcc = v2_vcc.merge(ct_map, on=["hash", "name"], how="left")
                v2_vcc["function_change_type"] = v2_vcc["function_change_type"].fillna("MODIFY")
                v2_vcc["is_hunk"] = False
                shared = [c for c in fn_merged.columns if c in v2_vcc.columns]
                fn_merged = pd.concat(
                    [fn_merged, v2_vcc[shared]], ignore_index=True
                )
                print(f"  [v2 fallback] added {len(v2_vcc)} VCC function rows from v2 for {v2_vcc['hash'].nunique()} gap commits")

        tables["function_info"] = fn_merged
        print(f"  [function_info] {len(fn_merged)} canonical rows loaded (is_v4={is_v4})")
    else:
        print(f"  [warning] function_info not found at any known path")
        tables["function_info"] = None

    # ── ownership ─────────────────────────────────────────────────────────────
    _ow_full = os.path.join(g, "ownership_window_full.csv")
    _ow_base = os.path.join(g, "ownership_window.csv")
    _load(_ow_full if (use_full and os.path.exists(_ow_full)) else _ow_base, "ownership_window")

    # ── developer / author ────────────────────────────────────────────────────
    dev_full     = os.path.join(g, "developer_info_full.csv")
    dev_primary  = os.path.join(p, "data/processed/developer_info_commits.csv")
    dev_fallback = os.path.join(g, "developer_info_clean.csv")
    if use_full and os.path.exists(dev_full):
        _load(dev_full, "developer_info")
    else:
        _load(dev_primary if os.path.exists(dev_primary) else dev_fallback, "developer_info")

    ca_full     = os.path.join(g, "commit_author_full.csv")
    ca_primary  = os.path.join(p, "data/processed/commit_author.csv")
    ca_fallback = os.path.join(g, "commit_author_clean.csv")
    if use_full and os.path.exists(ca_full):
        _load(ca_full, "commit_author")
    else:
        _load(ca_primary if os.path.exists(ca_primary) else ca_fallback, "commit_author")

    # ── commit-level SDLC features ────────────────────────────────────────────
    # The old VCC/FC features file (28 cols) has full tag info (prev/next tag names,
    # days_since_prev_tag, days_to_next_tag, release_cycle_position).
    # The full features file (16 cols) covers all commit types but uses a simplified
    # schema: time_since_last_tag (= days_since_prev_tag), no days_to_next_tag/tag names.
    # Strategy: merge both — prefer old file for VCC/FC (full schema), supplement with
    # full file for normal commits after renaming time_since_last_tag → days_since_prev_tag.
    _cf_old  = os.path.join(p, "data/sdlc_features/final_commit_level_features.csv")
    _cf_full = os.path.join(g, "final_commit_level_features_full.csv")

    if use_full and os.path.exists(_cf_full):
        cf_full = pd.read_csv(_cf_full)
        # Align column name used by normal-commit extraction script → graph_builder name
        if "time_since_last_tag" in cf_full.columns and "days_since_prev_tag" not in cf_full.columns:
            cf_full = cf_full.rename(columns={"time_since_last_tag": "days_since_prev_tag"})
        if os.path.exists(_cf_old):
            cf_old = pd.read_csv(_cf_old)
            # Supplement: add normal-commit rows that are absent from the old VCC/FC file
            normal_hashes = set(cf_full["hash"]) - set(cf_old["hash"])
            cf_new_only   = cf_full[cf_full["hash"].isin(normal_hashes)]
            tables["commit_features"] = pd.concat(
                [cf_old, cf_new_only], ignore_index=True
            )
        else:
            tables["commit_features"] = cf_full
        print(f"  [commit_features] {len(tables['commit_features'])} rows loaded (full merge)")
    elif os.path.exists(_cf_old):
        _load(_cf_old, "commit_features")
    else:
        print(f"  [warning] commit_features not found")
        tables["commit_features"] = None
    # Large SDLC tables: drop text columns never used as GNN features, and
    # pre-filter to requested commits when commit_hashes is provided.
    # issue_info: drop opened_by_dev_id (identifier, unused), labels (text, unused)
    # pr_info:    drop pr_url, title, opened_by_dev_id, closed_by_dev_id (all unused)
    _load_sdlc(os.path.join(p, "data/sdlc_filtered/issue_info_v4.csv"),
               "issue_info",
               drop_cols=["opened_by_dev_id", "labels"])
    _load_sdlc(os.path.join(p, "data/sdlc_filtered/pull_request_info_v4.csv"),
               "pr_info",
               drop_cols=["pr_url", "title", "opened_by_dev_id", "closed_by_dev_id"])
    _load_sdlc(os.path.join(p, "data/sdlc_filtered/release_tag_info_v4.csv"),
               "release_tag_info")
    _load(os.path.join(p, "data/processed/cve_fc_vcc_mapping.csv"),              "vcc_fc_mapping")

    return tables


def get_repo_commits(
    tables: dict[str, pd.DataFrame],
    repo_url: str,
) -> list[str]:
    """
    Return all commit hashes for *repo_url*, sorted by author_date ascending.

    Parameters
    ----------
    tables   : dict returned by load_all_tables()
    repo_url : repository URL as stored in commit_info (e.g.
               'https://github.com/tensorflow/tensorflow')

    Returns
    -------
    List of commit hashes (strings), chronologically ordered.
    """
    ci = tables.get("commit_info", pd.DataFrame())
    if ci.empty or "repo_url" not in ci.columns:
        return []
    subset = ci[ci["repo_url"] == repo_url].copy()
    subset["_ts"] = pd.to_datetime(subset["author_date"], utc=True, errors="coerce")
    subset = subset.sort_values("_ts")
    return subset["hash"].tolist()


def parse_tag_features(tag_name, days_distance: float) -> Optional[dict]:
    """
    Parse a release tag name string into structured node features (7-dim).
    Returns None if tag_name is missing/NaN — caller should skip adding the node.
    """
    import re as _re
    if tag_name is None or (isinstance(tag_name, float) and np.isnan(tag_name)):
        return None
    name = str(tag_name).lower().strip()
    m = _re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', name)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if m else 0
    patch = int(m.group(3)) if m and m.group(3) else 0
    is_prerelease = int(any(x in name for x in ['alpha', 'beta', 'rc', 'pre', 'dev']))
    is_hotfix     = int(any(x in name for x in ['hotfix', 'fix', 'patch']))
    is_major_bump = int(minor == 0 and patch == 0 and major > 0)
    return {
        "major": major,
        "minor": minor,
        "patch": patch,
        "is_prerelease": is_prerelease,
        "is_hotfix": is_hotfix,
        "is_major_bump": is_major_bump,
        "days_distance": float(np.log1p(abs(days_distance))) * (1.0 if days_distance >= 0 else -1.0),
    }


def _select_window_tags(
    tags: pd.DataFrame,
    commit_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Select at most 2 tags that bracket the commit (mode 4):
      - 1 tag immediately BEFORE the commit  (most recent tag with created_at < commit_date)
      - 1 tag immediately AFTER  the commit  (earliest  tag with created_at >= commit_date)

    Adds two columns to the returned DataFrame:
      _log_abs_days : log1p(|days between tag and commit|)  — magnitude of distance
      _is_before    : 1.0 if tag is before commit, 0.0 if after
    """
    if tags.empty:
        return tags

    t = tags.copy()
    t["_created_at"]  = pd.to_datetime(t["created_at"], errors="coerce", utc=True)
    t                 = t.dropna(subset=["_created_at"])
    if t.empty:
        return t

    t["_days"]        = (commit_date - t["_created_at"]).dt.days  # +ve = before, -ve = after
    t["_is_before"]   = (t["_days"] > 0).astype(float)
    t["_log_abs_days"] = np.log1p(t["_days"].abs())

    before = t[t["_days"] > 0].nsmallest(1, "_days")   # closest before
    after  = t[t["_days"] <= 0].nlargest(1, "_days")   # closest after  (least negative)

    result = pd.concat([before, after], ignore_index=True)
    return result


def _filter_sdlc(df: Optional[pd.DataFrame], commit_hash: str) -> pd.DataFrame:
    """
    Filter issue/PR/tag table to rows linked to commit_hash, then deduplicate
    by entity identity.

    A VCC can be linked to multiple FC hashes (one CVE → many fix commits),
    causing the same tag/issue/PR to appear once per FC row. We deduplicate
    by dropping join-key columns (fc_hash, vcc_hash, matched_anchor) before
    calling drop_duplicates(), so the same entity is kept only once regardless
    of which CVE/FC pair it was matched through.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for col in ("fc_hash", "vcc_hash"):
        if col in df.columns:
            mask |= (df[col] == commit_hash)
    filtered = df[mask].copy()
    # Deduplicate on entity attributes only, ignoring join-key columns
    join_keys = {"fc_hash", "vcc_hash", "matched_anchor", "window_since", "window_until"}
    dedup_cols = [c for c in filtered.columns if c not in join_keys]
    return filtered.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def _raw_numeric_features(
    df: pd.DataFrame,
    exclude: list[str] = EXCLUDE_PATTERNS,
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric columns excluding identifier-like patterns."""
    if df.empty:
        return np.zeros((0, 1), dtype="float32"), []
    num_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    cols = [c for c in num_cols
            if not any(p in c.lower() for p in exclude)]
    if not cols:
        return np.ones((len(df), 1), dtype="float32"), ["_placeholder"]
    return df[cols].fillna(0).values.astype("float32"), cols


def _engineered_feats(
    commit_features_row: pd.DataFrame,
    feat_cols: list[str],
) -> np.ndarray:
    """
    Extract engineered scalar features from the commit-level feature row.
    Count / duration columns (SDLC_LOG1P_COLS) are log1p-normalised;
    rate and binary columns are left as-is.
    """
    if commit_features_row.empty:
        return np.zeros(len(feat_cols), dtype="float32")
    available = [c for c in feat_cols if c in commit_features_row.columns]
    raw = commit_features_row[available].fillna(0).values[0].astype("float64")
    # Apply log1p to count/duration columns
    for i, c in enumerate(available):
        if c in SDLC_LOG1P_COLS:
            raw[i] = np.log1p(raw[i])
    vals = raw.astype("float32")
    if len(available) < len(feat_cols):
        out = np.zeros(len(feat_cols), dtype="float32")
        for i, c in enumerate(feat_cols):
            if c in available:
                out[i] = vals[available.index(c)]
        return out
    return vals


def _engineered_edge_attr(
    commit_features_row: pd.DataFrame,
    feat_cols: list[str],
    n_edges: int,
) -> torch.Tensor:
    """Repeat engineered edge features for each edge."""
    vals = _engineered_feats(commit_features_row, feat_cols)
    return torch.tensor(
        np.tile(vals, (n_edges, 1)), dtype=torch.float32
    )


def _empty_edge(data: HeteroData, src_t: str, rel: str, dst_t: str) -> None:
    data[src_t, rel, dst_t].edge_index = torch.zeros((2, 0), dtype=torch.long)


def _match_pr_issue(
    prs: pd.DataFrame,
    issues: pd.DataFrame,
) -> torch.Tensor:
    """Try to build PR→issue edges via direct column match or #NNN regex."""
    issue_id_col = next((c for c in ("issue_id", "id", "number") if c in issues.columns), None)
    if issue_id_col is None:
        return torch.zeros((2, 0), dtype=torch.long)

    issue_id_to_idx = {row[issue_id_col]: i for i, row in issues.iterrows()}
    src, dst = [], []

    link_cols = [c for c in ("issue_id", "linked_issue_id", "closing_issue_id") if c in prs.columns]
    for pr_idx, pr_row in prs.iterrows():
        for lc in link_cols:
            iid = pr_row[lc]
            if pd.notna(iid) and iid in issue_id_to_idx:
                src.append(pr_idx); dst.append(issue_id_to_idx[iid])
        if not src:  # fallback: regex
            text = " ".join(str(pr_row.get(c, "")) for c in ("body", "title") if c in prs.columns)
            for m in re.findall(r"#(\d+)", text):
                num = int(m)
                if num in issue_id_to_idx:
                    src.append(pr_idx); dst.append(issue_id_to_idx[num])

    if src:
        return torch.tensor([src, dst], dtype=torch.long)
    return torch.zeros((2, 0), dtype=torch.long)


def _temporal_chain(
    df: pd.DataFrame,
    ts_candidates: list[str],
) -> torch.Tensor:
    """Build index[i]→index[i+1] chain sorted by the first available timestamp col."""
    for col in ts_candidates:
        if col in df.columns:
            tmp = df.copy()
            tmp["_ts"] = pd.to_datetime(tmp[col], errors="coerce", utc=True)
            valid = tmp.dropna(subset=["_ts"]).sort_values("_ts")
            if len(valid) > 1:
                idxs = valid.index.tolist()
                return torch.tensor([idxs[:-1], idxs[1:]], dtype=torch.long)
            break
    return torch.zeros((2, 0), dtype=torch.long)


def _compute_file_ownership_stats(
    files: pd.DataFrame,
    ownership_window: Optional[pd.DataFrame],
    commit_author: Optional[pd.DataFrame],
    commit_hash: str,
    ownership_window_days: int,
) -> np.ndarray:
    """Per-file ownership distribution: [log1p(n_owners), HHI, max_ratio, committer_ratio]."""
    n_files = len(files)
    stats   = np.zeros((n_files, 4), dtype=np.float32)
    if ownership_window is None:
        return stats

    ow = ownership_window[ownership_window["commit_hash"] == commit_hash].copy()
    if ow.empty:
        return stats

    avail       = ow["window_days"].unique()
    chosen      = ownership_window_days if ownership_window_days in avail else \
                  avail[np.argmin(np.abs(avail - ownership_window_days))]
    ow          = ow[ow["window_days"] == chosen].copy()
    email_col   = "dev_email" if "dev_email" in ow.columns else "dev_id"
    ow["_email"] = ow[email_col].str.strip().str.lower()

    author_emails: set[str] = set()
    if commit_author is not None:
        ca = commit_author[
            (commit_author["commit_hash"] == commit_hash) &
            (commit_author["role"].str.lower() == "author")
        ]
        author_emails = set(ca["dev_id"].str.strip().str.lower())

    new_path_to_idx = {
        str(frow["new_path"]): fi
        for fi, frow in files.iterrows() if str(frow.get("new_path", ""))
    }
    for fpath, grp in ow.groupby("file_path"):
        fi = new_path_to_idx.get(str(fpath))
        if fi is None:
            continue
        ratios = grp["ownership_ratio"].fillna(0).values.astype("float64")
        committer_ratio = 0.0
        for _, orow in grp.iterrows():
            if str(orow["_email"]) in author_emails:
                committer_ratio = float(orow["ownership_ratio"])
                break
        stats[fi] = [
            np.log1p(len(grp)),
            float(np.sum(ratios ** 2)),
            float(ratios.max()) if len(ratios) else 0.0,
            committer_ratio,
        ]
    return stats


def _get_developer_data(
    commit_hash: str,
    ownership_window: Optional[pd.DataFrame],
    commit_author: Optional[pd.DataFrame],
    ownership_window_days: int = 90,
    ownership_threshold: float = OWNERSHIP_THRESHOLD,
    commit_row: Optional[pd.DataFrame] = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Shared logic for collecting and filtering developer data.

    Returns
    -------
    author_emails    : emails that appear as ``role == "author"`` for this commit
    committer_emails : emails that appear as ``role == "committer"`` for this commit
    own_rows         : filtered ownership_window rows (Bird et al. threshold)
    """
    author_emails:    list[str] = []
    committer_emails: list[str] = []

    if commit_author is not None:
        ca = commit_author[commit_author["commit_hash"] == commit_hash]
        for _, row in ca.iterrows():
            email = str(row["dev_id"]).strip().lower()
            role  = str(row.get("role", "")).lower()
            if role == "author":
                author_emails.append(email)
            else:
                committer_emails.append(email)

    # Fallback: if commit_author has no "committer" row for this commit (which
    # happens when author == committer — the table omits the redundant row),
    # read the committer email directly from commit_info.
    if not committer_emails and commit_row is not None and not commit_row.empty:
        committer_raw = str(commit_row["committer"].values[0]).strip().lower()
        if committer_raw and committer_raw != "nan":
            committer_emails.append(committer_raw)

    own_rows = pd.DataFrame()
    if ownership_window is not None:
        ow = ownership_window[ownership_window["commit_hash"] == commit_hash].copy()
        if not ow.empty:
            avail  = ow["window_days"].unique()
            chosen = ownership_window_days if ownership_window_days in avail else \
                     avail[np.argmin(np.abs(avail - ownership_window_days))]
            ow     = ow[ow["window_days"] == chosen].copy()
            ecol   = "dev_email" if "dev_email" in ow.columns else "dev_id"
            ow["_email"] = ow[ecol].str.strip().str.lower()
            all_author_set = set(author_emails) | set(committer_emails)
            own_rows   = ow[
                (ow["ownership_ratio"] >= ownership_threshold) |  # 5% threshold (Bird et al.)
                ow["_email"].isin(all_author_set)
            ].copy()

    return author_emails, committer_emails, own_rows


# ═════════════════════════════════════════════════════════════════════════════
# Multi-commit graph construction
# ═════════════════════════════════════════════════════════════════════════════

def build_multi_commit_graph(
    commit_hashes: list[str],
    tables: dict[str, pd.DataFrame],
    mode: int = 4,
    include_files: bool = True,
    include_functions: bool = True,
    include_developers: bool = True,
    include_issues: bool = True,
    include_prs: bool = True,
    include_tags: bool = True,
    ownership_window_days: int = 90,
    ownership_threshold: float = OWNERSHIP_THRESHOLD,
    max_issues: Optional[int] = None,
    max_prs: Optional[int] = None,
    max_tags: Optional[int] = None,
    include_parent_edges: bool = True,
    include_temporal_edges: bool = True,
) -> HeteroData:
    """
    Build a multi-commit heterogeneous graph from a list of commit hashes.

    Node deduplication strategy (hybrid):
      - commit        : one node per commit hash (always unique)
      - file          : snapshot per commit (one node per commit-file pair)
      - function      : snapshot per commit (one node per commit-function pair)
      - developer     : unified across all commits (one node per unique email)
      - issue/PR/tag  : snapshot per commit (same as single-commit graph)

    Cross-commit edges added:
      (file, file_evolution, file)
          same filename — forward-only, adjacent temporal snapshots (older→newer)
          attr: [delta_complexity, delta_loc, delta_token_count, log1p(time_gap_days)]
      (function, fn_evolution, function)
          same (filename, name) — forward-only, adjacent temporal snapshots (older→newer)
          attr: [delta_complexity, delta_loc, log1p(time_gap_days)]
      (commit, parent_of, commit)
          git DAG parent→child — both commits must be in commit_hashes
          attr: [log1p(time_delta_days), is_merge]
      (commit, precedes, commit)
          chronological chain: each commit → its immediate temporal successor
          attr: [log1p(delta_days)]

    All intra-commit edge types from build_graph() are preserved with adjusted
    global node indices. Developer edges are remapped to unified developer indices.

    Parameters
    ----------
    commit_hashes          : ordered list of commit SHAs to include
    tables                 : dict returned by load_all_tables()
    mode                   : graph mode passed to build_graph() for each commit
    include_parent_edges   : add (commit, parent_of, commit) edges
    include_temporal_edges : add (commit, precedes, commit) chronological chain edges

    Returns
    -------
    HeteroData with global node indices and all intra/cross-commit edges.
    Metadata stored as:
      data._commit_hashes   — list of commit hashes in node order
      data._commit_idx_map  — dict mapping hash → commit node index
    """
    if not commit_hashes:
        raise ValueError("commit_hashes must be non-empty.")

    # ── Step 1: per-commit subgraphs ──────────────────────────────────────────
    subgraphs: list[HeteroData] = []
    _show_progress = len(commit_hashes) >= 10
    for h in tqdm(commit_hashes, desc="Building subgraphs", unit="commit",
                  disable=not _show_progress):
        try:
            g = build_graph(
                h, tables, mode=mode,
                include_files=include_files,
                include_functions=include_functions,
                include_developers=include_developers,
                include_issues=include_issues,
                include_prs=include_prs,
                include_tags=include_tags,
                ownership_window_days=ownership_window_days,
                ownership_threshold=ownership_threshold,
                max_issues=max_issues,
                max_prs=max_prs,
                max_tags=max_tags,
            )
        except ValueError:
            g = HeteroData()
        subgraphs.append(g)

    # ── Step 2: cumulative offsets for snapshot node types ────────────────────
    SNAPSHOT_TYPES = ["commit", "file", "function", "issue", "pull_request", "release_tag"]
    offsets: dict[str, list[int]] = {nt: [0] for nt in SNAPSHOT_TYPES}
    for g in subgraphs:
        for nt in SNAPSHOT_TYPES:
            prev = offsets[nt][-1]
            n = g[nt].x.shape[0] if nt in g.node_types else 0
            offsets[nt].append(prev + n)

    # ── Step 3: unified developer nodes ───────────────────────────────────────
    all_emails: list[str] = []
    seen_emails: set[str] = set()
    for g in subgraphs:
        dev = g["developer"] if "developer" in g.node_types else None
        if dev is not None and hasattr(dev, "_email_to_idx"):
            for email in dev._email_to_idx:
                if email not in seen_emails:
                    all_emails.append(email)
                    seen_emails.add(email)
    global_email_to_idx: dict[str, int] = {e: i for i, e in enumerate(all_emails)}

    n_devs = len(all_emails)
    dev_feats = np.zeros((max(n_devs, 0), len(DEV_FEAT_COLS)), dtype=np.float32)
    developer_info = tables.get("developer_info")
    if developer_info is not None and n_devs > 0:
        di = developer_info.copy()
        di["_email"] = di["dev_id"].str.strip().str.lower()
        di = di.set_index("_email")
        for email, idx in global_email_to_idx.items():
            if email in di.index:
                vals = di.loc[email, DEV_FEAT_COLS].fillna(0).values.astype("float64")
                dev_feats[idx] = np.log1p(vals)

    # ── Step 4: merge HeteroData — snapshot node features ────────────────────
    data = HeteroData()

    for nt in SNAPSHOT_TYPES:
        tensors = [g[nt].x for g in subgraphs if nt in g.node_types]
        if tensors:
            # Pad to max feature dim in case some commits lack optional features (e.g. tag context)
            max_dim = max(t.shape[1] for t in tensors)
            padded = []
            for t in tensors:
                if t.shape[1] < max_dim:
                    pad = torch.zeros(t.shape[0], max_dim - t.shape[1], dtype=torch.float32)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            data[nt].x = torch.cat(padded, dim=0)
        else:
            fdim = next((g[nt].x.shape[1] for g in subgraphs if nt in g.node_types), 1)
            data[nt].x = torch.zeros((0, fdim), dtype=torch.float32)

    # Mean-impute developer nodes that have no row in developer_info (all-zero).
    # Without imputation ~1/3 of developer nodes act as dead-ends in message
    # passing because their feature vector is the zero vector.
    if n_devs > 0:
        known_mask = dev_feats.sum(axis=1) > 0
        if known_mask.any() and not known_mask.all():
            dev_mean = dev_feats[known_mask].mean(axis=0)
            dev_feats[~known_mask] = dev_mean

    data["developer"].x = torch.tensor(dev_feats, dtype=torch.float32)
    data["developer"]._email_to_idx = global_email_to_idx

    # ── Step 5: merge intra-commit snapshot edges (non-developer) ─────────────
    SNAPSHOT_EDGE_TYPES = [
        ("commit", "has", "file"),
        ("file", "has", "function"),
        ("function", "co_modified", "function"),
        ("commit", "linked_issue", "issue"),
        ("issue", "issue_of", "commit"),
        ("commit", "linked_pr", "pull_request"),
        ("pull_request", "pr_of", "commit"),
        ("commit", "has_release_tag", "release_tag"),
        ("release_tag", "tag_of", "commit"),
        ("pull_request", "references_issue", "issue"),
        ("issue", "referenced_by_pr", "pull_request"),
        ("release_tag", "next_tag", "release_tag"),
        ("release_tag", "prev_tag", "release_tag"),
        ("issue", "next_issue", "issue"),
        ("issue", "prev_issue", "issue"),
        ("pull_request", "next_pr", "pull_request"),
        ("pull_request", "prev_pr", "pull_request"),
        ("commit", "tag_context_pr", "pull_request"),
        ("pull_request", "tag_context_of", "commit"),
        ("commit", "tag_context_issue", "issue"),
        ("issue", "tag_context_of", "commit"),
        ("release_tag", "affects_pr", "pull_request"),
        ("pull_request", "in_release", "release_tag"),
        ("release_tag", "affects_issue", "issue"),
        ("issue", "in_release", "release_tag"),
    ]

    for et in SNAPSHOT_EDGE_TYPES:
        src_t, rel, dst_t = et
        all_ei, all_ea = [], []
        for i, g in enumerate(subgraphs):
            if et not in g.edge_types:
                continue
            ei = g[et].edge_index
            if ei.shape[1] == 0:
                continue
            src_off = offsets[src_t][i]
            dst_off = offsets[dst_t][i]
            all_ei.append(ei + torch.tensor([[src_off], [dst_off]], dtype=torch.long))
            ea = g[et].get("edge_attr")
            if ea is not None:
                all_ea.append(ea)

        if all_ei:
            data[et].edge_index = torch.cat(all_ei, dim=1)
            if len(all_ea) == len(all_ei):   # all non-empty subgraphs have attr
                data[et].edge_attr = torch.cat(all_ea, dim=0)
        else:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Step 6: merge developer edges with global index remapping ─────────────
    # Accumulate authored_by and committed_by separately; owns is shared.
    c2d_buckets: dict[str, tuple[list, list, list]] = {
        "authored_by":  ([], [], []),
        "committed_by": ([], [], []),
    }
    d2f_src, d2f_dst, d2f_attr = [], [], []

    for i, g in enumerate(subgraphs):
        dev = g["developer"] if "developer" in g.node_types else None
        if dev is None or not hasattr(dev, "_email_to_idx"):
            continue
        local_idx_to_email = {v: k for k, v in dev._email_to_idx.items()}

        def _remap(local_idx: int) -> int:
            email = local_idx_to_email.get(local_idx)
            return global_email_to_idx.get(email, -1) if email else -1

        for rel in ("authored_by", "committed_by"):
            c2d_et = ("commit", rel, "developer")
            if c2d_et not in g.edge_types:
                continue
            ei = g[c2d_et].edge_index
            ea = g[c2d_et].get("edge_attr")
            src_list, dst_list, attr_list = c2d_buckets[rel]
            for k in range(ei.shape[1]):
                gci = offsets["commit"][i] + ei[0, k].item()
                gdi = _remap(ei[1, k].item())
                if gdi < 0:
                    continue
                src_list.append(gci)
                dst_list.append(gdi)
                if ea is not None:
                    attr_list.append(ea[k])

        d2f_et = ("developer", "owns", "file")
        if d2f_et in g.edge_types:
            ei = g[d2f_et].edge_index
            ea = g[d2f_et].get("edge_attr")
            for k in range(ei.shape[1]):
                gdi = _remap(ei[0, k].item())
                gfi = offsets["file"][i] + ei[1, k].item()
                if gdi < 0:
                    continue
                d2f_src.append(gdi)
                d2f_dst.append(gfi)
                if ea is not None:
                    d2f_attr.append(ea[k])

    for rel, (src_list, dst_list, attr_list) in c2d_buckets.items():
        if src_list:
            data["commit", rel, "developer"].edge_index = torch.tensor(
                [src_list, dst_list], dtype=torch.long)
            if attr_list:
                data["commit", rel, "developer"].edge_attr = torch.stack(attr_list)
        else:
            data["commit", rel, "developer"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["commit", rel, "developer"].edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    if d2f_src:
        data["developer", "owns", "file"].edge_index = torch.tensor(
            [d2f_src, d2f_dst], dtype=torch.long)
        if d2f_attr:
            data["developer", "owns", "file"].edge_attr = torch.stack(d2f_attr)
    else:
        data["developer", "owns", "file"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ── Step 7: cross-commit edges ────────────────────────────────────────────
    commit_info = tables.get("commit_info", pd.DataFrame())
    commit_dates: dict[str, pd.Timestamp] = {}
    if not commit_info.empty:
        for h in commit_hashes:
            row = commit_info[commit_info["hash"] == h]
            if not row.empty:
                commit_dates[h] = pd.to_datetime(row["author_date"].values[0], utc=True)

    commit_idx_map = {h: i for i, h in enumerate(commit_hashes)}

    _add_file_evolution_edges(data, commit_hashes, tables, offsets, commit_dates)
    _add_fn_evolution_edges(data, commit_hashes, tables, offsets, commit_dates)

    if include_parent_edges:
        _add_parent_of_edges(data, tables, commit_idx_map, commit_dates)

    if include_temporal_edges:
        _add_temporal_commit_edges(data, commit_hashes, commit_idx_map, commit_dates)

    # ── Metadata ──────────────────────────────────────────────────────────────
    data._commit_hashes  = commit_hashes
    data._commit_idx_map = commit_idx_map

    return data


def _add_file_evolution_edges(
    data: HeteroData,
    commit_hashes: list[str],
    tables: dict[str, pd.DataFrame],
    offsets: dict[str, list[int]],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (file, file_evolution, file) — bidirectional edges between temporally
    adjacent snapshot file nodes with the same filename.

    Only consecutive pairs (sorted by author_date) are connected, avoiding the
    O(N²) density of all-pairs combinations while preserving the temporal chain.
    Edges are forward-only (older → newer) to respect temporal causality.

    Edge attr (4-dim): [delta_complexity, delta_loc, delta_token_count, log1p(time_gap_days)]
    Delta is signed: newer - older (positive = growth, negative = shrinkage).
    """
    file_info = tables.get("file_info")
    if file_info is None:
        _empty_edge(data, "file", "file_evolution", "file")
        data["file", "file_evolution", "file"].edge_attr = torch.zeros((0, 4), dtype=torch.float32)
        return

    # Build registry: one entry per (commit, file)
    registry: list[dict] = []
    for i, h in enumerate(commit_hashes):
        files = file_info[file_info["hash"] == h].reset_index(drop=True)
        for fi, frow in files.iterrows():
            registry.append({
                "global_idx":  offsets["file"][i] + fi,
                "filename":    str(frow.get("filename", "") or frow.get("new_path", "")),
                "complexity":  float(frow.get("complexity", 0) or 0),
                "loc":         float(frow.get("num_lines_of_code", 0) or 0),
                "token_count": float(frow.get("token_count", 0) or 0),
                "commit_hash": h,
                "commit_i":    i,
            })

    by_filename: dict[str, list[dict]] = defaultdict(list)
    for r in registry:
        by_filename[r["filename"]].append(r)

    _epoch = pd.Timestamp("1970-01-01", tz="UTC")

    src, dst, feats = [], [], []
    for _, entries in by_filename.items():
        if len(entries) < 2:
            continue
        # Sort by author_date so we only connect adjacent temporal snapshots
        entries_sorted = sorted(
            entries,
            key=lambda e: commit_dates.get(e["commit_hash"], _epoch),
        )
        for a, b in zip(entries_sorted, entries_sorted[1:]):
            if a["commit_i"] == b["commit_i"]:
                continue
            d_cplx = b["complexity"]  - a["complexity"]
            d_loc  = b["loc"]         - a["loc"]
            d_tok  = b["token_count"] - a["token_count"]
            da = commit_dates.get(a["commit_hash"])
            db = commit_dates.get(b["commit_hash"])
            tgap = float(np.log1p(abs((db - da).days))) if da and db else 0.0
            # Signed log1p normalisation: preserves direction, compresses magnitude.
            src.append(a["global_idx"]); dst.append(b["global_idx"])
            feats.append([
                float(np.sign(d_cplx) * np.log1p(abs(d_cplx))),
                float(np.sign(d_loc)  * np.log1p(abs(d_loc))),
                float(np.sign(d_tok)  * np.log1p(abs(d_tok))),
                tgap,
            ])

    # --- Fallback: bridge orphaned VCC file nodes to their FC via CVE mapping ---
    # Handles cases where the file was renamed/moved between VCC and FC commits
    # (e.g. CVE-2021-29524: conv_grad_ops.cc → conv_grad_shape_utils.cc).
    vcc_fc_map_df = tables.get("vcc_fc_mapping")
    if vcc_fc_map_df is not None and not vcc_fc_map_df.empty:
        vcc_to_fc = _parse_vcc_to_fc(vcc_fc_map_df)
        hash_to_ci = {h: i for i, h in enumerate(commit_hashes)}
        src_set = set(src)  # global file indices already with an outgoing edge

        for i, h in enumerate(commit_hashes):
            fc_h = vcc_to_fc.get(h)
            if fc_h is None or fc_h not in hash_to_ci:
                continue
            fc_i = hash_to_ci[fc_h]

            vcc_files = file_info[file_info["hash"] == h].reset_index(drop=True)
            fc_files  = file_info[file_info["hash"] == fc_h].reset_index(drop=True)
            if vcc_files.empty or fc_files.empty:
                continue

            # Pre-compute FC filename set.
            # Only bridge VCC files whose filename is completely absent from the
            # FC's file set — this is the true file-rename/move case
            # (e.g. conv_grad_ops.cc → conv_grad_shape_utils.cc in CVE-2021-29524).
            # If the VCC filename exists in the FC, either it was already
            # name-matched (in src_set) or it belongs to a different CVE role.
            fc_filenames_set = set(
                str(r.get("filename", "") or r.get("new_path", ""))
                for _, r in fc_files.iterrows()
            )

            for fi_idx, frow in vcc_files.iterrows():
                g_idx = offsets["file"][i] + fi_idx
                if g_idx in src_set:
                    continue  # already has an outgoing file_evolution edge
                vcc_filename = str(frow.get("filename", "") or frow.get("new_path", ""))
                if vcc_filename in fc_filenames_set:
                    continue  # same filename exists in FC — not a file rename
                for fj_idx, fcrow in fc_files.iterrows():
                    fc_g_idx = offsets["file"][fc_i] + fj_idx
                    d_cplx = float(fcrow.get("complexity", 0) or 0) - float(frow.get("complexity", 0) or 0)
                    d_loc  = float(fcrow.get("num_lines_of_code", 0) or 0) - float(frow.get("num_lines_of_code", 0) or 0)
                    d_tok  = float(fcrow.get("token_count", 0) or 0) - float(frow.get("token_count", 0) or 0)
                    da = commit_dates.get(h)
                    db = commit_dates.get(fc_h)
                    tgap = float(np.log1p(abs((db - da).days))) if da and db else 0.0
                    src.append(g_idx)
                    dst.append(fc_g_idx)
                    feats.append([
                        float(np.sign(d_cplx) * np.log1p(abs(d_cplx))),
                        float(np.sign(d_loc)  * np.log1p(abs(d_loc))),
                        float(np.sign(d_tok)  * np.log1p(abs(d_tok))),
                        tgap,
                    ])

    if src:
        data["file", "file_evolution", "file"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["file", "file_evolution", "file"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["file", "file_evolution", "file"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["file", "file_evolution", "file"].edge_attr  = torch.zeros((0, 4), dtype=torch.float32)


def _parse_vcc_to_fc(vcc_fc_map_df: pd.DataFrame) -> dict[str, str]:
    """Parse vcc_fc_mapping DataFrame → {vcc_hash: fc_hash}.

    The ``vcc_hash`` column is stored as a stringified Python list, e.g.
    ``"['abc123', 'def456']"``.  This helper parses that format and returns
    a flat mapping from every VCC hash to its corresponding FC hash.
    Duplicate rows are deduplicated (same mapping, just keep last write).
    """
    import ast

    result: dict[str, str] = {}
    for _, row in vcc_fc_map_df.iterrows():
        fc = str(row.get("fc_hash", "")).strip()
        if not fc:
            continue
        vcc_raw = str(row.get("vcc_hash", "[]"))
        try:
            vccs = ast.literal_eval(vcc_raw)
        except Exception:
            vccs = []
        for v in vccs:
            v = str(v).strip()
            if v:
                result[v] = fc
    return result


def _fn_change_type_onehot(ct: str) -> list[float]:
    """Return a 5-dim one-hot for the function_change_type string."""
    idx = FUNC_CHANGE_TYPE_MAP.get(str(ct).strip().upper(), 0)
    oh = [0.0] * len(FUNC_CHANGE_TYPE_CATS)
    oh[idx] = 1.0
    return oh


def _add_fn_evolution_edges(
    data: HeteroData,
    commit_hashes: list[str],
    tables: dict[str, pd.DataFrame],
    offsets: dict[str, list[int]],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (function, fn_evolution, function) — forward-only edges between temporally
    adjacent snapshot function nodes with the same qualified function name.

    Edge attr (8-dim):
      [0:5]  src function_change_type one-hot (MODIFY/ADD/DELETE/RENAME/REFACTOR)
      [5]    sign*log1p(d_complexity)
      [6]    sign*log1p(d_loc)
      [7]    log1p(time_gap_days)

    Main pass: keys on function name — connects same-named functions across commits.

    RENAME fallback (pass 2): VCC functions with change_type=RENAME that still have
    no outgoing evolution edge are bridged to all functions in their paired FC commit
    (per vcc_fc_mapping) that share the same source filename.  This handles the case
    where a function is renamed at VCC or FC time so the names don't match but the
    file-level CVE mapping still links them.  d_complexity and d_loc are set to 0
    for these fallback edges (no name match → no comparable metric delta).

    Keyed on function name only (not filename) so that functions which move
    between files across commits (e.g. extracted into a utility header) are
    still correctly linked. Names are typically fully qualified (e.g.
    'tensorflow::FooOp::Compute'), making cross-file false positives unlikely.

    Only consecutive pairs (sorted by author_date) are connected, avoiding the
    O(N²) density of all-pairs combinations while preserving the temporal chain.
    Edges are forward-only (older → newer) to respect temporal causality.

    Fallback pass: VCC function nodes that have no outgoing fn_evolution edge
    (i.e. the function was renamed between VCC and FC) are bridged to *all*
    function nodes of their paired FC commit using the vcc_fc_mapping table.
    This handles the CVE-2021-29524 rename case and similar situations.

    """
    function_info = tables.get("function_info")
    if function_info is None:
        _empty_edge(data, "function", "fn_evolution", "function")
        data["function", "fn_evolution", "function"].edge_attr = torch.zeros((0, 8), dtype=torch.float32)
        return

    commit_info = tables.get("commit_info", pd.DataFrame())
    # Build hash → commit_type lookup so we can apply the same per-role filter
    # that build_graph uses.  Dual-role commits (FC in one CVE, VCC in another)
    # have both commit_label='FC' and commit_label='VCC' rows in function_info;
    # we must only count the rows that were actually added as graph nodes.
    _commit_type_map: dict[str, str] = {}
    if not commit_info.empty:
        for h in commit_hashes:
            row = commit_info[commit_info["hash"] == h]
            if not row.empty:
                _commit_type_map[h] = str(row["commit_type"].values[0])

    # Pre-build known_files per commit from file_info so the registry applies
    # exactly the same filename guard that build_graph uses.  Without this,
    # functions with filenames absent from file_info inflate the per-commit
    # count and cause global_idx to overflow into the next commit's function
    # nodes — producing edges that point to wrong (possibly earlier) nodes,
    # which appear as backward-in-time arrows in the visualisation.
    file_info = tables.get("file_info")
    _known_files_per_commit: dict[str, set] = {}
    if file_info is not None:
        for h in commit_hashes:
            rows = file_info[file_info["hash"] == h]
            _known_files_per_commit[h] = set(rows["filename"].tolist()) if not rows.empty else set()

    registry: list[dict] = []
    for i, h in enumerate(commit_hashes):
        _ctype      = _commit_type_map.get(h, "FC")
        _known_files = _known_files_per_commit.get(h, set())
        fn_mask = (
            (function_info["hash"] == h) &
            (function_info["commit_label"] == _ctype)
        )
        # Apply the same filename filter as build_graph so that global_idx
        # aligns with the actual node positions in the merged graph.
        if _known_files:
            fn_mask = fn_mask & function_info["filename"].isin(_known_files)
        else:
            # No file nodes for this commit: exclude all functions (mirrors build_graph).
            fn_mask = pd.Series(False, index=function_info.index)
        funcs = function_info[fn_mask].reset_index(drop=True)
        for fi, frow in funcs.iterrows():
            ct = str(frow.get("function_change_type", "MODIFY") or "MODIFY").strip().upper()
            registry.append({
                "global_idx":   offsets["function"][i] + fi,
                "key":          str(frow.get("name", "")).strip(),
                "filename":     str(frow.get("filename", "") or ""),
                "change_type":  ct,
                "complexity":   float(frow.get("complexity", 0) or 0),
                "loc":          float(frow.get("num_lines_of_code", 0) or 0),
                "commit_hash":  h,
                "commit_type":  _ctype,
                "commit_i":     i,
                "before_change": bool(frow.get("before_change", True)),
            })

    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in registry:
        by_key[r["key"]].append(r)

    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    _attr_dim = 5 + 3  # 5 change-type one-hot + [d_cplx, d_loc, tgap]

    src, dst, feats = [], [], []
    for _, entries in by_key.items():
        if len(entries) < 2:
            continue
        # Sort by author_date so we only connect adjacent temporal snapshots
        entries_sorted = sorted(
            entries,
            key=lambda e: commit_dates.get(e["commit_hash"], _epoch),
        )
        for a, b in zip(entries_sorted, entries_sorted[1:]):
            if a["commit_i"] == b["commit_i"]:
                continue
            d_cplx = b["complexity"] - a["complexity"]
            d_loc  = b["loc"]        - a["loc"]
            da = commit_dates.get(a["commit_hash"])
            db = commit_dates.get(b["commit_hash"])
            tgap = float(np.log1p(abs((db - da).days))) if da and db else 0.0
            src.append(a["global_idx"]); dst.append(b["global_idx"])
            feats.append(
                _fn_change_type_onehot(a["change_type"]) + [
                    float(np.sign(d_cplx) * np.log1p(abs(d_cplx))),
                    float(np.sign(d_loc)  * np.log1p(abs(d_loc))),
                    tgap,
                ]
            )

    # ── Within-commit RENAME edges: old-name → new-name (tgap=0) ──────────────
    # When a VCC commit renames a function, we have two nodes in the same commit:
    #   before_change=True  → old name (the node that "disappears")
    #   before_change=False → new name (the node that "appears")
    # Connect them with an fn_evolution RENAME edge so the chain is unbroken.
    _rename_old: dict[tuple, dict] = {}  # (commit_hash, filename) → old-name entry
    _rename_new: dict[tuple, dict] = {}  # (commit_hash, filename) → new-name entry
    for r in registry:
        if r["change_type"] != "RENAME":
            continue
        key = (r["commit_hash"], r["filename"])
        if r["before_change"]:
            _rename_old[key] = r
        else:
            _rename_new[key] = r
    for key, old_r in _rename_old.items():
        new_r = _rename_new.get(key)
        if new_r is None:
            continue
        d_cplx = new_r["complexity"] - old_r["complexity"]
        d_loc  = new_r["loc"]        - old_r["loc"]
        src.append(old_r["global_idx"]); dst.append(new_r["global_idx"])
        feats.append(
            _fn_change_type_onehot("RENAME") + [
                float(np.sign(d_cplx) * np.log1p(abs(d_cplx))),
                float(np.sign(d_loc)  * np.log1p(abs(d_loc))),
                0.0,  # tgap=0: same commit
            ]
        )

    # ── RENAME fallback: bridge VCC RENAME functions that have no evolution edge ──
    # A VCC function with change_type=RENAME changed name between VCC and FC time,
    # so the name-based matching above found no partner.  We explicitly bridge it
    # to functions in the same file of the paired FC commit using the CVE mapping.
    # REFACTOR is intentionally excluded: REFACTOR implies a major cross-file
    # restructuring where file-level evolution already provides the connectivity.
    # Note: after the within-commit pass above, the new-name node may still be
    # orphaned (no outgoing edge).  The fallback below checks src_set AFTER the
    # within-commit edges are added, so new-name nodes are included.
    vcc_fc_map_df = tables.get("vcc_fc_mapping")
    if vcc_fc_map_df is not None and not vcc_fc_map_df.empty:
        vcc_to_fc = _parse_vcc_to_fc(vcc_fc_map_df)
        src_set   = set(src)  # global function indices already with an outgoing edge
        hash_to_ci = {h: i for i, h in enumerate(commit_hashes)}

        # Index FC function registry entries by (commit_hash, filename)
        fc_by_hash_file: dict[tuple, list[dict]] = defaultdict(list)
        for r in registry:
            if r["commit_type"] == "FC":
                fc_by_hash_file[(r["commit_hash"], r["filename"])].append(r)

        for r in registry:
            if r["commit_type"] != "VCC":
                continue
            if r["change_type"] != "RENAME":
                continue
            if r["global_idx"] in src_set:
                continue  # already bridged by name-match or within-commit RENAME edge

            fc_h = vcc_to_fc.get(r["commit_hash"])
            if fc_h is None or fc_h not in hash_to_ci:
                continue

            fc_candidates = fc_by_hash_file.get((fc_h, r["filename"]), [])
            if not fc_candidates:
                continue

            da = commit_dates.get(r["commit_hash"])
            db = commit_dates.get(fc_h)
            tgap = float(np.log1p(abs((db - da).days))) if da and db else 0.0

            for fc_r in fc_candidates:
                src.append(r["global_idx"]); dst.append(fc_r["global_idx"])
                # d_cplx and d_loc are 0 (no same-name match to compute deltas)
                feats.append(
                    _fn_change_type_onehot("RENAME") + [0.0, 0.0, tgap]
                )

    if src:
        data["function", "fn_evolution", "function"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["function", "fn_evolution", "function"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["function", "fn_evolution", "function"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["function", "fn_evolution", "function"].edge_attr  = torch.zeros((0, _attr_dim), dtype=torch.float32)


def _add_parent_of_edges(
    data: HeteroData,
    tables: dict[str, pd.DataFrame],
    commit_idx_map: dict[str, int],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (commit, parent_of, commit) — git DAG edges.
    Only materialised when BOTH parent and child are in commit_hashes.

    Edge attr (2-dim): [log1p(time_delta_days), is_merge]
    Direction: parent → child (temporal forward direction).
    """
    commit_info = tables.get("commit_info", pd.DataFrame())
    if commit_info.empty:
        _empty_edge(data, "commit", "parent_of", "commit")
        data["commit", "parent_of", "commit"].edge_attr = torch.zeros((0, 2), dtype=torch.float32)
        return

    src, dst, feats = [], [], []
    for h, child_idx in commit_idx_map.items():
        row = commit_info[commit_info["hash"] == h]
        if row.empty:
            continue
        parents_raw = row["parents"].values[0]
        try:
            parents = eval(str(parents_raw)) if isinstance(parents_raw, str) else []
        except Exception:
            parents = []
        if not isinstance(parents, list):
            parents = [parents]
        is_merge = float(len(parents) > 1)
        for parent_hash in parents:
            parent_hash = str(parent_hash).strip()
            if parent_hash not in commit_idx_map:
                continue
            parent_idx = commit_idx_map[parent_hash]
            dp = commit_dates.get(parent_hash)
            dc = commit_dates.get(h)
            time_delta = abs((dc - dp).days) if dp and dc else 0.0
            src.append(parent_idx)
            dst.append(child_idx)
            feats.append([float(np.log1p(time_delta)), is_merge])

    if src:
        data["commit", "parent_of", "commit"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["commit", "parent_of", "commit"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["commit", "parent_of", "commit"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["commit", "parent_of", "commit"].edge_attr  = torch.zeros((0, 2), dtype=torch.float32)


def _add_temporal_commit_edges(
    data: HeteroData,
    commit_hashes: list[str],
    commit_idx_map: dict[str, int],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (commit, precedes, commit) — chronological chain over all commits in the graph.

    Commits are sorted by author_date and each is connected to its immediate
    temporal successor. This replaces the label-leaking VCC→FC 'fixes' edge:
    the temporal chain is fully constructible at inference time without any
    knowledge of commit labels.

    Direction: earlier commit → later commit.
    Edge attr (1-dim): [log1p(delta_days)]
    """
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")

    # Sort commits by author_date
    sorted_hashes = sorted(
        commit_hashes,
        key=lambda h: commit_dates.get(h, _epoch),
    )

    src, dst, feats = [], [], []
    for a_hash, b_hash in zip(sorted_hashes, sorted_hashes[1:]):
        a_idx = commit_idx_map[a_hash]
        b_idx = commit_idx_map[b_hash]
        da = commit_dates.get(a_hash)
        db = commit_dates.get(b_hash)
        days = abs((db - da).days) if da and db else 0.0
        src.append(a_idx)
        dst.append(b_idx)
        feats.append([float(np.log1p(days))])

    if src:
        data["commit", "precedes", "commit"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["commit", "precedes", "commit"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["commit", "precedes", "commit"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["commit", "precedes", "commit"].edge_attr  = torch.zeros((0, 1), dtype=torch.float32)


if __name__ == "__main__":
    import os
    base   = os.path.join(os.path.dirname(__file__), "..")
    target = "a5a51ad3a1200e2e5ef46c140bab717422e41ca2"

    print("Loading tables …")
    tables = load_all_tables(base)

    for mode in (1, 2, 3):
        print(f"\n{'='*60}")
        print(f"  MODE {mode}")
        print(f"{'='*60}")
        g = build_graph(target, tables, mode=mode)
        print(g)
        print(f"  file feat_dim  : {g['file'].x.shape[1]}")
        print(f"  dev nodes      : {g['developer'].x.shape[0]}  "
              f"(owns edges: {g['developer','owns','file'].edge_index.shape[1]})")
        if g['issue'].x.shape[0] > 0:
            print(f"  issue nodes    : {g['issue'].x.shape[0]}  feat_dim={g['issue'].x.shape[1]}")
        if g['pull_request'].x.shape[0] > 0:
            print(f"  pr nodes       : {g['pull_request'].x.shape[0]}  feat_dim={g['pull_request'].x.shape[1]}")
        if g['release_tag'].x.shape[0] > 0:
            print(f"  tag nodes      : {g['release_tag'].x.shape[0]}  feat_dim={g['release_tag'].x.shape[1]}")

    print("\n✓ All modes passed.")
