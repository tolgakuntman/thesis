from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

OWNERSHIP_THRESHOLD = 0.05   # Bird et al. (2011) minor-contributor cutoff

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
ENGINEERED_ISSUE_FEATS  = ["issue_age_median", "issue_close_rate_180d"]
ENGINEERED_PR_FEATS     = ["pr_age_median", "pr_merge_or_close_rate_180d"]
ENGINEERED_TAG_FEATS    = ["time_since_last_tag", "tags_last_180d"]
ENGINEERED_C2I_EDGE     = ["issue_open_at_anchor"]
ENGINEERED_C2PR_EDGE    = ["pr_count"]
ENGINEERED_C2TAG_EDGE   = ["has_release_pressure_180d"]
ENGINEERED_I2PR_EDGE    = ["pr_to_issue_open_ratio_90d", "has_issue_pr_gap"]
ENGINEERED_TAG2X_EDGE   = ["activity_since_last_tag"]

CHANGE_TYPE_CATS = ["ADD", "MODIFY", "DELETE", "RENAME", "COPY", "UNKNOWN"]
CHANGE_TYPE_MAP  = {ct: i for i, ct in enumerate(CHANGE_TYPE_CATS)}

# Engineered features that are counts / durations and need log1p normalisation.
# Rate features (0-1) and binary flags are left as-is.
SDLC_LOG1P_COLS = {
    "issue_age_median",        # days — can be 1000+
    "pr_age_median",           # days
    "time_since_last_tag",     # days
    "issue_open_at_anchor",    # count of open issues
    "pr_count",                # count of linked PRs
    "tags_last_180d",          # count of tags
    "activity_since_last_tag", # count of events
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
    funcs = tables["function_info"][
        tables["function_info"]["hash"] == commit_hash
    ].reset_index(drop=True)

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
    """
    base = commit_row[COMMIT_FEAT_COLS].fillna(0).values.astype("float32")

    if mode == 4 and not commit_features_row.empty:
        tag_ctx = np.array([
            _engineered_feats(commit_features_row, ["time_since_last_tag"]),
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
      [0:6]  code metrics  (FILE_CODE_FEAT_COLS)
      [6:10] ownership distribution (log1p(num_owners), HHI, max_ratio, committer_ratio)
    """
    code_feats = files[FILE_CODE_FEAT_COLS].fillna(0).values.astype("float32")
    own_stats  = _compute_file_ownership_stats(
        files, ownership_window, commit_author, commit_hash, ownership_window_days
    )
    data["file"].x = torch.tensor(
        np.concatenate([code_feats, own_stats], axis=1), dtype=torch.float32
    )


def add_function_nodes(data: HeteroData, funcs: pd.DataFrame) -> None:
    """function nodes — 5 features."""
    if len(funcs) > 0:
        data["function"].x = torch.tensor(
            funcs[FUNC_FEAT_COLS].fillna(0).values.astype("float32")
        )
    else:
        data["function"].x = torch.zeros((0, len(FUNC_FEAT_COLS)), dtype=torch.float32)


def add_developer_nodes(
    data: HeteroData,
    commit_hash: str,
    files: pd.DataFrame,
    ownership_window: Optional[pd.DataFrame],
    developer_info: Optional[pd.DataFrame],
    commit_author: Optional[pd.DataFrame],
    ownership_window_days: int = 90,
    ownership_threshold: float = OWNERSHIP_THRESHOLD,
) -> None:
    """
    developer nodes — 4 features (log1p normalised).
    Inclusion rule: ownership_ratio >= threshold OR commit author (Bird et al. 2011).
    Stores email_to_dev_idx on data for use by add_developer_edges().
    """
    authored_emails, author_roles, own_rows = _get_developer_data(
        commit_hash, ownership_window, commit_author,
        ownership_window_days, ownership_threshold,
    )

    all_dev_emails   = list(dict.fromkeys(
        authored_emails + (own_rows["_email"].tolist() if not own_rows.empty else [])
    ))
    email_to_dev_idx = {e: i for i, e in enumerate(all_dev_emails)}

    n_devs    = len(all_dev_emails)
    dev_feats = np.zeros((n_devs, len(DEV_FEAT_COLS)), dtype=np.float32)

    if developer_info is not None and n_devs > 0:
        di = developer_info.copy()
        di["_email"] = di["dev_id"].str.strip().str.lower()
        di = di.set_index("_email")
        for email, idx in email_to_dev_idx.items():
            if email in di.index:
                vals = di.loc[email, DEV_FEAT_COLS].fillna(0).values.astype("float64")
                dev_feats[idx] = np.log1p(vals)

    data["developer"].x = torch.tensor(dev_feats, dtype=torch.float32)
    # Stash for edge builder (avoids recomputing)
    data["developer"]._email_to_idx  = email_to_dev_idx
    data["developer"]._authored_emails = authored_emails
    data["developer"]._author_roles    = author_roles
    data["developer"]._own_rows        = own_rows


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
    mode 4:   up to 2 nodes (tag before + tag after commit).
              Features: [log1p(abs_days_to_commit), is_before_flag]
              Tags must have been pre-filtered by _select_window_tags().
    """
    if tags.empty:
        n_feats = len(ENGINEERED_TAG_FEATS) if mode == 2 else (2 if mode == 4 else 1)
        data["release_tag"].x = torch.zeros((0, n_feats), dtype=torch.float32)
        return

    if mode == 2:
        feats = _engineered_feats(commit_features_row, ENGINEERED_TAG_FEATS)
        data["release_tag"].x = torch.tensor(feats.reshape(1, -1), dtype=torch.float32)
    elif mode == 4:
        # _select_window_tags added _log_abs_days and _is_before columns
        feats = tags[["_log_abs_days", "_is_before"]].fillna(0).values.astype("float32")
        data["release_tag"].x = torch.tensor(feats, dtype=torch.float32)
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
    (file, has, function) — 6 features:
    loc_frac, complexity_ratio, token_ratio, position, num_params, before_change
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
            loc_frac         = fnrow["num_lines_of_code"] / max(frow["num_lines_of_code"], 1)
            complexity_ratio = fnrow["complexity"]         / max(frow["complexity"], 1)
            token_ratio      = fnrow["token_count"]        / max(frow["token_count"], 1)
            position         = pos / max(n_fn - 1, 1)
            try:
                params = len(fnrow.get("parameters", []) or [])
            except TypeError:
                params = 0
            before_change = float(fnrow.get("before_change", False))
            src.append(fi); dst.append(fni)
            feats.append([loc_frac, complexity_ratio, token_ratio,
                          position, float(params), before_change])

    if src:
        data["file", "has", "function"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["file", "has", "function"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["file", "has", "function"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["file", "has", "function"].edge_attr  = torch.zeros((0, 6), dtype=torch.float32)


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
    (commit, authored_by, developer) — 1 feature: role (1=author, 0=committer)
    (developer, owns, file)          — 3 features: ownership_ratio, norm_lines, log_edits
    Reads stashed data from add_developer_nodes() if available.
    """
    # Retrieve stashed data (set by add_developer_nodes)
    if hasattr(data.get("developer", HeteroData()), "_email_to_idx"):
        email_to_dev_idx = data["developer"]._email_to_idx
        authored_emails  = data["developer"]._authored_emails
        author_roles     = data["developer"]._author_roles
        own_rows         = data["developer"]._own_rows
    else:
        authored_emails, author_roles, own_rows = _get_developer_data(
            commit_hash, ownership_window, commit_author,
            ownership_window_days, ownership_threshold,
        )
        all_dev_emails   = list(dict.fromkeys(
            authored_emails + (own_rows["_email"].tolist() if not own_rows.empty else [])
        ))
        email_to_dev_idx = {e: i for i, e in enumerate(all_dev_emails)}

    c2d_src, c2d_dst, c2d_role = [], [], []
    for email, role in zip(authored_emails, author_roles):
        if email in email_to_dev_idx:
            c2d_src.append(0)
            c2d_dst.append(email_to_dev_idx[email])
            c2d_role.append(float(role))

    if c2d_src:
        data["commit", "authored_by", "developer"].edge_index = torch.tensor([c2d_src, c2d_dst], dtype=torch.long)
        data["commit", "authored_by", "developer"].edge_attr  = torch.tensor([[r] for r in c2d_role], dtype=torch.float32)
    else:
        data["commit", "authored_by", "developer"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["commit", "authored_by", "developer"].edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

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
      mode 2/3 : has_release_pressure_180d  (1-dim)
      mode 4   : [log1p(abs_days_to_commit), is_before_flag]  (2-dim)
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
    elif mode == 4 and "_log_abs_days" in tags.columns:
        ea = torch.tensor(
            tags[["_log_abs_days", "_is_before"]].fillna(0).values.astype("float32")
        )
        data["commit", "has_release_tag", "release_tag"].edge_attr = ea
        data["release_tag", "tag_of", "commit"].edge_attr          = ea


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


def load_all_tables(base_path: str = "..") -> dict[str, pd.DataFrame]:
    """Load all required DataFrames from base_path (thesis root directory)."""
    import os
    p = base_path
    g = os.path.join(p, "data", "graph_data")   # symlinked graph data

    tables: dict[str, Optional[pd.DataFrame]] = {}

    def _load(path: str, key: str) -> None:
        if os.path.exists(path):
            tables[key] = pd.read_csv(path)
        else:
            print(f"  [warning] {key} not found at {path}")
            tables[key] = None

    _load(os.path.join(p, "data/processed/commit_info.csv"),          "commit_info")
    _load(os.path.join(p, "data_new/processed/file_info_new.csv"),    "file_info")
    _load(os.path.join(p, "data_new/processed/function_info_new.csv"),"function_info")
    _load(os.path.join(p, "data_new/processed/ownership_window.csv"), "ownership_window")
    _load(os.path.join(p, "data/_cleaned/developer_info_clean.csv"),  "developer_info")
    _load(os.path.join(p, "data/_cleaned/commit_author_clean.csv"),   "commit_author")
    _load(os.path.join(g, "final_commit_level_features.csv"),         "commit_features")
    _load(os.path.join(g, "issue_info_v3.csv"),                       "issue_info")
    _load(os.path.join(g, "pull_request_info_v3.csv"),                "pr_info")
    _load(os.path.join(g, "release_tag_info_v3.csv"),                 "release_tag_info")
    _load(os.path.join(p, "data/processed/cve_fc_vcc_mapping.csv"),   "vcc_fc_mapping")

    return tables


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
    ownership_window_days: int,
    ownership_threshold: float,
) -> tuple[list[str], list[int], pd.DataFrame]:
    """Shared logic for collecting and filtering developer data."""
    authored_emails: list[str] = []
    author_roles:    list[int] = []

    if commit_author is not None:
        ca = commit_author[commit_author["commit_hash"] == commit_hash]
        for _, row in ca.iterrows():
            authored_emails.append(str(row["dev_id"]).strip().lower())
            author_roles.append(1 if str(row.get("role", "")).lower() == "author" else 0)

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
            author_set = set(authored_emails)
            own_rows   = ow[
                (ow["ownership_ratio"] >= ownership_threshold) |  # 5% threshold (Bird et al.)
                ow["_email"].isin(author_set)
            ].copy()

    return authored_emails, author_roles, own_rows


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
    include_vcc_fc_edges: bool = True,
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
      (file, same_entity, file)
          same filename modified in different commits
          attr: [delta_complexity, delta_loc, delta_token_count, log1p(time_gap_days)]
      (function, same_entity, function)
          same (filename, name) in different commits
          attr: [delta_complexity, delta_loc, log1p(time_gap_days)]
      (commit, parent_of, commit)
          git DAG parent→child — both commits must be in commit_hashes
          attr: [log1p(time_delta_days), is_merge]
      (commit, fixes, commit)
          VCC→FC pairs from cve_fc_vcc_mapping — both must be in commit_hashes
          attr: [log1p(days_vcc_to_fc)]

    All intra-commit edge types from build_graph() are preserved with adjusted
    global node indices. Developer edges are remapped to unified developer indices.

    Parameters
    ----------
    commit_hashes        : ordered list of commit SHAs to include
    tables               : dict returned by load_all_tables()
    mode                 : graph mode passed to build_graph() for each commit
    include_parent_edges : add (commit, parent_of, commit) edges
    include_vcc_fc_edges : add (commit, fixes, commit) edges

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
    for h in commit_hashes:
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
    c2d_src, c2d_dst, c2d_attr = [], [], []
    d2f_src, d2f_dst, d2f_attr = [], [], []

    for i, g in enumerate(subgraphs):
        dev = g["developer"] if "developer" in g.node_types else None
        if dev is None or not hasattr(dev, "_email_to_idx"):
            continue
        local_idx_to_email = {v: k for k, v in dev._email_to_idx.items()}

        def _remap(local_idx: int) -> int:
            email = local_idx_to_email.get(local_idx)
            return global_email_to_idx.get(email, -1) if email else -1

        c2d_et = ("commit", "authored_by", "developer")
        if c2d_et in g.edge_types:
            ei = g[c2d_et].edge_index
            ea = g[c2d_et].get("edge_attr")
            for k in range(ei.shape[1]):
                gci = offsets["commit"][i] + ei[0, k].item()
                gdi = _remap(ei[1, k].item())
                if gdi < 0:
                    continue
                c2d_src.append(gci)
                c2d_dst.append(gdi)
                if ea is not None:
                    c2d_attr.append(ea[k])

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

    if c2d_src:
        data["commit", "authored_by", "developer"].edge_index = torch.tensor(
            [c2d_src, c2d_dst], dtype=torch.long)
        if c2d_attr:
            data["commit", "authored_by", "developer"].edge_attr = torch.stack(c2d_attr)
    else:
        data["commit", "authored_by", "developer"].edge_index = torch.zeros((2, 0), dtype=torch.long)

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

    _add_same_entity_file_edges(data, commit_hashes, tables, offsets, commit_dates)
    _add_same_entity_function_edges(data, commit_hashes, tables, offsets, commit_dates)

    if include_parent_edges:
        _add_parent_of_edges(data, commit_hashes, tables, commit_idx_map, commit_dates)

    if include_vcc_fc_edges:
        _add_vcc_fc_edges(data, commit_hashes, tables, commit_idx_map, commit_dates)

    # ── Metadata ──────────────────────────────────────────────────────────────
    data._commit_hashes  = commit_hashes
    data._commit_idx_map = commit_idx_map

    return data


def _add_same_entity_file_edges(
    data: HeteroData,
    commit_hashes: list[str],
    tables: dict[str, pd.DataFrame],
    offsets: dict[str, list[int]],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (file, same_entity, file) — bidirectional edges between snapshot file nodes
    that correspond to the same filename in different commits.

    Edge attr (4-dim): [delta_complexity, delta_loc, delta_token_count, log1p(time_gap_days)]
    Delta is signed: B - A for the forward direction, A - B for the reverse.
    """
    file_info = tables.get("file_info")
    if file_info is None:
        _empty_edge(data, "file", "same_entity", "file")
        data["file", "same_entity", "file"].edge_attr = torch.zeros((0, 4), dtype=torch.float32)
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

    src, dst, feats = [], [], []
    for fname, entries in by_filename.items():
        if len(entries) < 2:
            continue
        for a, b in combinations(entries, 2):
            if a["commit_i"] == b["commit_i"]:
                continue
            d_cplx = b["complexity"]  - a["complexity"]
            d_loc  = b["loc"]         - a["loc"]
            d_tok  = b["token_count"] - a["token_count"]
            da = commit_dates.get(a["commit_hash"])
            db = commit_dates.get(b["commit_hash"])
            tgap = float(np.log1p(abs((db - da).days))) if da and db else 0.0
            src.append(a["global_idx"]); dst.append(b["global_idx"])
            feats.append([d_cplx, d_loc, d_tok, tgap])
            src.append(b["global_idx"]); dst.append(a["global_idx"])
            feats.append([-d_cplx, -d_loc, -d_tok, tgap])

    if src:
        data["file", "same_entity", "file"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["file", "same_entity", "file"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["file", "same_entity", "file"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["file", "same_entity", "file"].edge_attr  = torch.zeros((0, 4), dtype=torch.float32)


def _add_same_entity_function_edges(
    data: HeteroData,
    commit_hashes: list[str],
    tables: dict[str, pd.DataFrame],
    offsets: dict[str, list[int]],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (function, same_entity, function) — bidirectional edges between snapshot
    function nodes with the same (filename, name) key in different commits.

    Edge attr (3-dim): [delta_complexity, delta_loc, log1p(time_gap_days)]
    """
    function_info = tables.get("function_info")
    if function_info is None:
        _empty_edge(data, "function", "same_entity", "function")
        data["function", "same_entity", "function"].edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        return

    registry: list[dict] = []
    for i, h in enumerate(commit_hashes):
        funcs = function_info[function_info["hash"] == h].reset_index(drop=True)
        for fi, frow in funcs.iterrows():
            registry.append({
                "global_idx": offsets["function"][i] + fi,
                "key":        (str(frow.get("filename", "")), str(frow.get("name", ""))),
                "complexity": float(frow.get("complexity", 0) or 0),
                "loc":        float(frow.get("num_lines_of_code", 0) or 0),
                "commit_hash": h,
                "commit_i":   i,
            })

    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in registry:
        by_key[r["key"]].append(r)

    src, dst, feats = [], [], []
    for key, entries in by_key.items():
        if len(entries) < 2:
            continue
        for a, b in combinations(entries, 2):
            if a["commit_i"] == b["commit_i"]:
                continue
            d_cplx = b["complexity"] - a["complexity"]
            d_loc  = b["loc"]        - a["loc"]
            da = commit_dates.get(a["commit_hash"])
            db = commit_dates.get(b["commit_hash"])
            tgap = float(np.log1p(abs((db - da).days))) if da and db else 0.0
            src.append(a["global_idx"]); dst.append(b["global_idx"])
            feats.append([d_cplx, d_loc, tgap])
            src.append(b["global_idx"]); dst.append(a["global_idx"])
            feats.append([-d_cplx, -d_loc, tgap])

    if src:
        data["function", "same_entity", "function"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["function", "same_entity", "function"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["function", "same_entity", "function"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["function", "same_entity", "function"].edge_attr  = torch.zeros((0, 3), dtype=torch.float32)


def _add_parent_of_edges(
    data: HeteroData,
    commit_hashes: list[str],
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


def _add_vcc_fc_edges(
    data: HeteroData,
    commit_hashes: list[str],
    tables: dict[str, pd.DataFrame],
    commit_idx_map: dict[str, int],
    commit_dates: dict[str, pd.Timestamp],
) -> None:
    """
    (commit, fixes, commit) — VCC → FC edges from cve_fc_vcc_mapping.
    Only materialised when BOTH vcc and fc hashes are in commit_hashes.

    Direction: VCC → FC (vulnerability causes the fix).
    Edge attr (1-dim): [log1p(days_vcc_to_fc)]

    NOTE: mask these edges at training time to avoid label leakage for the
    target commit. They are useful for providing relational context to
    *other* commits in the multi-commit graph.
    """
    vcc_fc = tables.get("vcc_fc_mapping")
    if vcc_fc is None:
        _empty_edge(data, "commit", "fixes", "commit")
        data["commit", "fixes", "commit"].edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        return

    commit_set = set(commit_hashes)
    seen_pairs: set[tuple[int, int]] = set()
    src, dst, feats = [], [], []

    for _, row in vcc_fc.iterrows():
        fc_hash = str(row.get("fc_hash", "")).strip()
        vcc_raw = row.get("vcc_hash", "[]")
        try:
            vcc_list = eval(str(vcc_raw)) if isinstance(vcc_raw, str) else []
        except Exception:
            vcc_list = []
        if not isinstance(vcc_list, list):
            vcc_list = [vcc_list]

        if fc_hash not in commit_set:
            continue
        fc_idx = commit_idx_map[fc_hash]

        for vcc_hash in vcc_list:
            vcc_hash = str(vcc_hash).strip()
            if vcc_hash not in commit_set:
                continue
            vcc_idx = commit_idx_map[vcc_hash]
            pair = (vcc_idx, fc_idx)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            dv = commit_dates.get(vcc_hash)
            df_ts = commit_dates.get(fc_hash)
            days = abs((df_ts - dv).days) if dv and df_ts else 0.0
            src.append(vcc_idx)
            dst.append(fc_idx)
            feats.append([float(np.log1p(days))])

    if src:
        data["commit", "fixes", "commit"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data["commit", "fixes", "commit"].edge_attr  = torch.tensor(feats, dtype=torch.float32)
    else:
        data["commit", "fixes", "commit"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["commit", "fixes", "commit"].edge_attr  = torch.zeros((0, 1), dtype=torch.float32)


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
