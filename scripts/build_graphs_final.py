"""
Build per-commit graphs from the finalized package in
data_new/analysis_outputs/final_graph_inputs_v1/.

This builder intentionally uses a simpler topology than the legacy graph_ready
pipeline because the final package no longer preserves filename/function-name
identity columns. Anonymous file/function/hunk rows are attached directly to the
commit node.

Default policy:
- one graph per commit
- skip commits with no file/function/hunk rows
- never place labels into node tensors
- keep function delta/change-type columns out of the default graph
- use the legacy 5% ownership threshold for developer->file ownership edges

Usage:
  conda run -n thesis python scripts/build_graphs_final.py
  conda run -n thesis python scripts/build_graphs_final.py --limit 100
  conda run -n thesis python scripts/build_graphs_final.py --allow_commit_only
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


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data_new" / "analysis_outputs" / "final_graph_inputs_v1"
OUT_ROOT = ROOT / "outputs" / "final_graph_ready"
OUT_DIR = OUT_ROOT / "graphs"
MANIFEST_OUT = OUT_ROOT / "build_manifest.csv"
FAILED_OUT = OUT_ROOT / "failed_commits.jsonl"
OWNERSHIP_THRESHOLD = 0.05


COMMIT_FEAT_COLS = [
    "in_main_branch",
    "merge",
    "dmm_unit_size",
    "dmm_unit_complexity",
    "dmm_unit_interfacing",
    "author_timezone",
    "committer_timezone",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

ISSUE_FEAT_COLS = [
    "issue_open_90d",
    "issue_age_median",
    "issues_closed_last_90d",
    "issue_open_velocity_90d",
]

PR_FEAT_COLS = [
    "pr_count_90d",
    "pr_age_median",
    "pr_closed_last_90d",
    "pr_open_velocity_90d",
]

TAG_FEAT_COLS = [
    "days_since_prev_tag",
    "tags_last_365d",
    "avg_release_cadence_days",
    "days_since_prev_tag_norm",
]

AUTHOR_EDGE_FEAT_COLS = [
    "dev_experience_days",
    "dev_commits_before",
    "dev_is_new_contributor",
]

ISSUE_EDGE_FEAT_COLS = [
    "pr_to_issue_open_ratio_90d",
    "issue_to_pr_closed_ratio_90d",
    "has_issue_pr_gap",
]

PR_EDGE_FEAT_COLS = [
    "pr_to_issue_open_ratio_90d",
    "issue_to_pr_closed_ratio_90d",
    "has_issue_pr_gap",
]

TAG_EDGE_FEAT_COLS = [
    "activity_since_last_tag",
]

FILE_FEAT_COLS = [
    "num_lines_of_code",
    "complexity",
    "token_count",
]

COMMIT_FILE_EDGE_FEAT_COLS = [
    "n_owners",
    "max_own_ratio",
    "ownership_hhi",
    "total_lines_norm",
]

FUNCTION_FEAT_COLS = [
    "num_lines_of_code",
    "complexity",
    "token_count",
    "length",
    "top_nesting_level",
]

COMMIT_FUNCTION_EDGE_FEAT_COLS = [
    "loc_before",
    "complexity_before",
    "tokens_before",
    "delta_loc",
    "delta_complexity",
    "delta_tokens",
    "fct_modify",
    "fct_add",
    "fct_delete",
    "fct_rename",
    "fct_refactor",
]

HUNK_FEAT_TO_FUNCTION_SLOTS = {
    "complexity": 1,
    "token_count": 2,
}

DEV_FEAT_COLS = [
    "total_commits",
    "active_weeks",
    "commits_as_committer",
    "total_issues",
    "total_pull_requests",
]


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
    edge_attr: list[list[float]] | np.ndarray | None = None,
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


def load_tables() -> dict[str, object]:
    manifest = pd.read_csv(FINAL / "build_manifest.csv")
    commit_features = pd.read_csv(FINAL / "final_commit_level_features_v2_normalized_model_features.csv").drop_duplicates("hash").set_index("hash")
    commit_core_features = pd.read_csv(FINAL / "commit_features.csv").drop_duplicates("hash").set_index("hash")

    file_df = pd.read_csv(FINAL / "file_features.csv")
    file_df["_row_idx"] = np.arange(len(file_df))
    file_index = pd.read_csv(FINAL / "file_index.csv")
    if len(file_index) != len(file_df) or not file_index["hash"].equals(file_df["hash"]):
        raise ValueError("file_index.csv is not aligned with file_features.csv")
    file_df = pd.concat([file_df.reset_index(drop=True), file_index[["filename", "old_path", "new_path", "file_path"]].reset_index(drop=True)], axis=1)

    fn_df = pd.read_csv(FINAL / "function_features.csv")
    fn_df["_row_idx"] = np.arange(len(fn_df))
    function_index = pd.read_csv(FINAL / "function_index.csv")
    if len(function_index) != len(fn_df) or not function_index["hash"].equals(fn_df["hash"]):
        raise ValueError("function_index.csv is not aligned with function_features.csv")
    fn_df = pd.concat(
        [
            fn_df.reset_index(drop=True),
            function_index[
                [
                    "function_row_idx",
                    "name",
                    "filename",
                    "embedding_layer",
                    "source_match_count",
                    "is_zero_embedding",
                    "file_match_count",
                    "file_row_idx",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    file_function_edges = pd.read_csv(FINAL / "file_function_edges.csv")

    hunk_df = pd.read_csv(FINAL / "hunk_features.csv")
    hunk_df["_row_idx"] = np.arange(len(hunk_df))

    function_embeddings = np.load(FINAL / "function_embeddings.npy", mmap_mode="r")
    hunk_embeddings = np.load(FINAL / "hunk_embeddings.npy", mmap_mode="r")
    commit_msg_embeddings = np.load(FINAL / "commit_msg_embeddings.npy", mmap_mode="r")

    commit_msg_index = pd.read_csv(FINAL / "commit_msg_index.csv")
    msg_row_by_hash = dict(zip(commit_msg_index["hash"], range(len(commit_msg_index))))

    developer = pd.read_csv(FINAL / "developer_info_full_aligned_manifest.csv")
    developer = developer.drop_duplicates("canonical_dev_key").set_index("canonical_dev_key")
    email_rows = developer[developer["dev_id_type"].astype(str).str.lower() == "email"].copy()
    email_rows["_email_norm"] = email_rows["dev_id"].astype(str).str.strip().str.lower()
    canonical_by_email = dict(zip(email_rows["_email_norm"], email_rows.index))

    ownership = pd.read_csv(FINAL / "ownership_window_full_aligned_manifest.csv")
    ownership = ownership[ownership["window_days"] == 90].copy()
    ownership = ownership.dropna(subset=["canonical_dev_key"])

    commit_meta = pd.read_csv(
        ROOT / "data" / "graph_data" / "commit_info_full.csv",
        usecols=["hash", "repo_url", "author_date"],
    ).drop_duplicates("hash").set_index("hash")
    commit_author = pd.read_csv(ROOT / "data" / "graph_data" / "commit_author_full.csv")
    commit_author["role"] = commit_author["role"].astype(str).str.strip().str.lower()
    commit_author["_email_norm"] = commit_author["dev_id"].astype(str).str.strip().str.lower()

    return {
        "manifest": manifest,
        "commit_features": commit_features,
        "commit_core_features": commit_core_features,
        "file_by_hash": dict(tuple(file_df.groupby("hash", sort=False))),
        "function_by_hash": dict(tuple(fn_df.groupby("hash", sort=False))),
        "file_function_edges_by_hash": dict(tuple(file_function_edges.groupby("hash", sort=False))),
        "hunk_by_hash": dict(tuple(hunk_df.groupby("hash", sort=False))),
        "function_embeddings": function_embeddings,
        "hunk_embeddings": hunk_embeddings,
        "commit_msg_embeddings": commit_msg_embeddings,
        "msg_row_by_hash": msg_row_by_hash,
        "developer": developer,
        "canonical_by_email": canonical_by_email,
        "ownership_by_hash": dict(tuple(ownership.groupby("commit_hash", sort=False))),
        "commit_author_by_hash": dict(tuple(commit_author.groupby("commit_hash", sort=False))),
        "commit_meta": commit_meta,
    }


def build_graph(commit_hash: str, label: int, tables: dict[str, object], allow_commit_only: bool) -> HeteroData:
    data = HeteroData()

    commit_features = tables["commit_features"]
    commit_core_features = tables["commit_core_features"]
    if commit_hash not in commit_features.index:
        raise ValueError("missing_commit_features")
    if commit_hash not in commit_core_features.index:
        raise ValueError("missing_commit_core_features")

    cf_row = commit_features.loc[[commit_hash]]
    core_row = commit_core_features.loc[[commit_hash]]

    msg_row = tables["msg_row_by_hash"].get(commit_hash)
    if msg_row is not None:
        msg_vec = tables["commit_msg_embeddings"][msg_row].reshape(1, -1).astype(np.float32)
    else:
        msg_vec = np.zeros((1, 768), dtype=np.float32)
    commit_num = safe_values(core_row.reset_index(drop=True), COMMIT_FEAT_COLS)
    commit_x = np.concatenate([commit_num, msg_vec], axis=1) if commit_num.size else msg_vec
    data["commit"].x = torch.tensor(commit_x, dtype=torch.float32)

    file_rows = tables["file_by_hash"].get(commit_hash, pd.DataFrame())
    fn_rows = tables["function_by_hash"].get(commit_hash, pd.DataFrame())
    ff_edges = tables["file_function_edges_by_hash"].get(commit_hash, pd.DataFrame())
    hunk_rows = tables["hunk_by_hash"].get(commit_hash, pd.DataFrame())

    has_code = len(file_rows) > 0 or len(fn_rows) > 0 or len(hunk_rows) > 0
    if not has_code and not allow_commit_only:
        raise ValueError("no_code_rows")

    if len(file_rows) > 0:
        own_rows = tables["ownership_by_hash"].get(commit_hash, pd.DataFrame())
        own_stats = np.zeros((len(file_rows), len(COMMIT_FILE_EDGE_FEAT_COLS)), dtype=np.float32)
        file_path_to_idx = {
            str(path): idx
            for idx, path in enumerate(file_rows["file_path"].astype(str).tolist())
        }
        if len(own_rows) > 0:
            for fpath, grp in own_rows.groupby("file_path", sort=False):
                fi = file_path_to_idx.get(str(fpath))
                if fi is None:
                    continue
                ratios = pd.to_numeric(grp["ownership_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)
                total_lines = pd.to_numeric(grp["total_lines"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                if ratios.size == 0:
                    continue
                own_stats[fi, 0] = float(np.log1p((ratios >= OWNERSHIP_THRESHOLD).sum()))
                own_stats[fi, 1] = float(ratios.max(initial=0.0))
                own_stats[fi, 2] = float(np.square(ratios).sum())
                own_stats[fi, 3] = float(np.log1p(total_lines.max(initial=0.0)))
        data["file"].x = torch.tensor(
            safe_values(file_rows, FILE_FEAT_COLS),
            dtype=torch.float32,
        )
    else:
        own_stats = np.zeros((0, len(COMMIT_FILE_EDGE_FEAT_COLS)), dtype=np.float32)
        data["file"].x = torch.zeros((0, len(FILE_FEAT_COLS)), dtype=torch.float32)

    if len(fn_rows) > 0:
        fn_num = safe_values(fn_rows, FUNCTION_FEAT_COLS)
        fn_emb = tables["function_embeddings"][fn_rows["_row_idx"].to_numpy()]
        fn_edge = safe_values(fn_rows, COMMIT_FUNCTION_EDGE_FEAT_COLS)
        data["function"].x = torch.tensor(np.concatenate([fn_num, fn_emb], axis=1), dtype=torch.float32)
    else:
        fn_edge = np.zeros((0, len(COMMIT_FUNCTION_EDGE_FEAT_COLS)), dtype=np.float32)
        data["function"].x = torch.zeros((0, len(FUNCTION_FEAT_COLS) + 768), dtype=torch.float32)

    if len(hunk_rows) > 0:
        hunk_num = np.zeros((len(hunk_rows), len(FUNCTION_FEAT_COLS)), dtype=np.float32)
        for col, slot in HUNK_FEAT_TO_FUNCTION_SLOTS.items():
            hunk_num[:, slot] = pd.to_numeric(hunk_rows[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        hunk_emb = tables["hunk_embeddings"][hunk_rows["_row_idx"].to_numpy()]
        data["hunk"].x = torch.tensor(np.concatenate([hunk_num, hunk_emb], axis=1), dtype=torch.float32)
    else:
        data["hunk"].x = torch.zeros((0, len(FUNCTION_FEAT_COLS) + 768), dtype=torch.float32)

    developer = tables["developer"]
    own_rows = tables["ownership_by_hash"].get(commit_hash, pd.DataFrame())
    ca_rows = tables["commit_author_by_hash"].get(commit_hash, pd.DataFrame())
    canonical_by_email = tables["canonical_by_email"]

    author_keys: list[str] = []
    committer_keys: list[str] = []
    if len(ca_rows) > 0:
        author_keys = [
            canonical_by_email[email]
            for email in ca_rows.loc[ca_rows["role"] == "author", "_email_norm"].tolist()
            if email in canonical_by_email
        ]
        committer_keys = [
            canonical_by_email[email]
            for email in ca_rows.loc[ca_rows["role"] == "committer", "_email_norm"].tolist()
            if email in canonical_by_email
        ]

    owner_keys: list[str] = []
    if len(own_rows) > 0:
        own_rows = own_rows.copy()
        own_rows["ownership_ratio"] = pd.to_numeric(own_rows["ownership_ratio"], errors="coerce").fillna(0.0)
        own_rows = own_rows[own_rows["ownership_ratio"] >= OWNERSHIP_THRESHOLD]
        owner_keys = [k for k in own_rows["canonical_dev_key"].dropna().unique().tolist() if k in developer.index]

    dev_keys = list(dict.fromkeys(author_keys + committer_keys + owner_keys))

    if dev_keys:
        dev_rows = developer.loc[dev_keys]
        data["developer"].x = torch.tensor(safe_values(dev_rows.reset_index(drop=True), DEV_FEAT_COLS), dtype=torch.float32)
    else:
        data["developer"].x = torch.zeros((0, len(DEV_FEAT_COLS)), dtype=torch.float32)
    dev_idx_by_key = {key: idx for idx, key in enumerate(dev_keys)}

    data["issue"].x = torch.tensor(safe_values(cf_row.reset_index(drop=True), ISSUE_FEAT_COLS), dtype=torch.float32)
    data["pull_request"].x = torch.tensor(safe_values(cf_row.reset_index(drop=True), PR_FEAT_COLS), dtype=torch.float32)
    data["release_tag"].x = torch.tensor(safe_values(cf_row.reset_index(drop=True), TAG_FEAT_COLS), dtype=torch.float32)

    connect_commit(data, "file", "modifies_file", "in_commit", edge_attr=own_stats)
    connect_commit(data, "function", "modifies_func", "in_commit_fn", edge_attr=fn_edge)
    connect_commit(data, "hunk", "modifies_hunk", "in_commit_hunk")
    connect_commit(
        data,
        "issue",
        "has_issue",
        "linked_to_commit",
        edge_attr=np.tile(safe_values(cf_row.reset_index(drop=True), ISSUE_EDGE_FEAT_COLS), (max(data["issue"].x.size(0), 1), 1)),
    )
    connect_commit(
        data,
        "pull_request",
        "has_pr",
        "linked_to_commit",
        edge_attr=np.tile(safe_values(cf_row.reset_index(drop=True), PR_EDGE_FEAT_COLS), (max(data["pull_request"].x.size(0), 1), 1)),
    )
    connect_commit(
        data,
        "release_tag",
        "has_release",
        "release_of",
        edge_attr=np.tile(safe_values(cf_row.reset_index(drop=True), TAG_EDGE_FEAT_COLS), (max(data["release_tag"].x.size(0), 1), 1)),
    )

    if len(file_rows) > 0 and len(fn_rows) > 0 and len(ff_edges) > 0:
        local_file_idx = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(file_rows["_row_idx"].astype(int).tolist())
        }
        local_fn_idx = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(fn_rows["_row_idx"].astype(int).tolist())
        }
        ff_src: list[int] = []
        ff_dst: list[int] = []
        for row in ff_edges.itertuples(index=False):
            src = local_file_idx.get(int(row.file_row_idx))
            dst = local_fn_idx.get(int(row.function_row_idx))
            if src is None or dst is None:
                continue
            ff_src.append(src)
            ff_dst.append(dst)
        build_bipartite_edges(data, "file", "contains", "function", ff_src, ff_dst, "in_file")
    else:
        empty_edge(data, "file", "contains", "function")
        empty_edge(data, "function", "in_file", "file")

    build_bipartite_edges(
        data,
        "commit",
        "authored_by",
        "developer",
        [0 for _ in author_keys if _ in dev_idx_by_key],
        [dev_idx_by_key[k] for k in author_keys if k in dev_idx_by_key],
        "authored",
        edge_attr=np.tile(
            safe_values(cf_row.reset_index(drop=True), AUTHOR_EDGE_FEAT_COLS),
            (sum(1 for k in author_keys if k in dev_idx_by_key), 1),
        ),
    )
    build_bipartite_edges(
        data,
        "commit",
        "committed_by",
        "developer",
        [0 for _ in committer_keys if _ in dev_idx_by_key],
        [dev_idx_by_key[k] for k in committer_keys if k in dev_idx_by_key],
        "committed",
        edge_attr=np.tile(
            safe_values(cf_row.reset_index(drop=True), AUTHOR_EDGE_FEAT_COLS),
            (sum(1 for k in committer_keys if k in dev_idx_by_key), 1),
        ),
    )

    if len(file_rows) > 0 and len(own_rows) > 0 and dev_idx_by_key:
        file_path_to_idx = {
            str(path): idx
            for idx, path in enumerate(file_rows["file_path"].astype(str).tolist())
        }
        own_src: list[int] = []
        own_dst: list[int] = []
        own_attr: list[list[float]] = []
        for row in own_rows.itertuples(index=False):
            dev_idx = dev_idx_by_key.get(row.canonical_dev_key)
            file_idx = file_path_to_idx.get(str(row.file_path))
            if dev_idx is None or file_idx is None:
                continue
            own_src.append(dev_idx)
            own_dst.append(file_idx)
            own_ratio = float(getattr(row, "ownership_ratio", 0.0) or 0.0)
            total_lines = float(getattr(row, "total_lines", 0.0) or 0.0)
            lines_owned = float(getattr(row, "lines_owned", 0.0) or 0.0)
            edits = float(getattr(row, "edits_in_window", 0.0) or 0.0)
            own_attr.append([
                own_ratio,
                lines_owned / max(total_lines, 1.0),
                float(np.log1p(edits)),
            ])
        build_bipartite_edges(data, "developer", "owns", "file", own_src, own_dst, "owned_by", edge_attr=own_attr)
    else:
        empty_edge(data, "developer", "owns", "file")
        empty_edge(data, "file", "owned_by", "developer")

    data.y = torch.tensor([int(label)], dtype=torch.long)
    data["commit"].hash = commit_hash
    if commit_hash in tables["commit_meta"].index:
        meta = tables["commit_meta"].loc[commit_hash]
        data["commit"].repo_url = meta.get("repo_url", "")
        data["commit"].author_date = str(meta.get("author_date", ""))
    else:
        data["commit"].repo_url = ""
        data["commit"].author_date = ""
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graphs from the finalized graph-input package")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hashes", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_commit_only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    manifest = tables["manifest"].copy()

    if args.hashes:
        manifest = manifest[manifest["hash"].isin(args.hashes)].copy()
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    manifest_rows: list[dict[str, object]] = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    with open(FAILED_OUT, "w", encoding="utf-8") as failed_file:
        for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Building final-package graphs"):
            commit_hash = row.hash
            out_path = OUT_DIR / f"{commit_hash}.pt"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            started = time.time()
            try:
                data = build_graph(commit_hash, row.label, tables, allow_commit_only=args.allow_commit_only)
                torch.save(data, out_path)

                n_nodes = sum(data[nt].x.size(0) for nt in data.node_types)
                n_edges = sum(
                    data[et].edge_index.size(1)
                    for et in data.edge_types
                    if hasattr(data[et], "edge_index")
                )
                manifest_rows.append({
                    "hash": commit_hash,
                    "commit_type": row.commit_type,
                    "label": int(row.label),
                    "repo_url": getattr(data["commit"], "repo_url", ""),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "build_time_ms": int((time.time() - started) * 1000),
                })
                n_ok += 1
            except ValueError as exc:
                failed_file.write(json.dumps({"hash": commit_hash, "reason": str(exc)}) + "\n")
                n_fail += 1
            except Exception as exc:
                failed_file.write(json.dumps({"hash": commit_hash, "reason": f"exception:{type(exc).__name__}:{exc}"}) + "\n")
                n_fail += 1

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(MANIFEST_OUT, index=False)

    print(f"built={n_ok} skipped={n_skip} failed={n_fail}")
    if manifest_rows:
        built_df = pd.DataFrame(manifest_rows)
        print(f"avg_nodes={built_df['n_nodes'].mean():.1f}")
        print(f"avg_edges={built_df['n_edges'].mean():.1f}")


if __name__ == "__main__":
    main()
