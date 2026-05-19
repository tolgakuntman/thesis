from __future__ import annotations

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "ICVul_pp" / "graph_ready_sampling_v2"
OUT_ROOT = ROOT / "outputs" / "graph_ready_v2"
GRAPHS_DIR = OUT_ROOT / "graphs"
SPLIT_INDEX = OUT_ROOT / "split_index.csv"
SCALER_PATH = OUT_ROOT / "perrepo_scaler_v2.json"
AUDIT_DIR = OUT_ROOT / "audit"

SPLIT_COL = "repo_split"
SAMPLE_PER_SPLIT = 200
INTEGRATION_SAMPLE_SIZE = 50

EXPECTED_VAL_REPOS = {
    "https://github.com/ImageMagick/ImageMagick",
    "https://github.com/radareorg/radare2",
    "https://github.com/the-tcpdump-group/tcpdump",
    "https://github.com/php/php-src",
    "https://github.com/FreeRDP/FreeRDP",
}
EXPECTED_TEST_REPOS = {
    "https://github.com/FFmpeg/FFmpeg",
    "https://github.com/gpac/gpac",
    "https://github.com/OISF/suricata",
    "https://github.com/openssl/openssl",
    "https://github.com/redis/redis",
    "https://github.com/envoyproxy/envoy",
}

GROUPS: list[dict] = [
    {
        "group": "commit_node",
        "kind": "node",
        "target": "commit",
        "names": [
            "in_main_branch",
            "is_merge",
            "dmm_unit_complexity",
            "dmm_unit_interfacing",
            "dmm_unit_size",
            "tz_author_norm",
            "tz_committer_norm",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "has_sdlc_data",
            "repo_commits_90d",
            "repo_active_authors_90d",
        ],
    },
    {
        "group": "fn_node",
        "kind": "node",
        "target": "function",
        "names": [
            "num_lines_of_code",
            "complexity",
            "token_count",
            "length",
            "top_nesting_level",
            "loc_before",
            "complexity_before",
            "tokens_before",
        ] + [f"embedding_{i:03d}" for i in range(768)],
    },
    {
        "group": "file_node",
        "kind": "node",
        "target": "file",
        "names": ["num_lines_added", "num_lines_deleted", "complexity"],
    },
    {
        "group": "hunk_node",
        "kind": "node",
        "target": "hunk",
        "names": ["complexity", "token_count"] + [f"embedding_{i:03d}" for i in range(768)],
    },
    {
        "group": "dev_node",
        "kind": "node",
        "target": "developer",
        "names": [
            "repo_total_commits_before",
            "repo_active_weeks_before",
            "repo_tenure_days",
            "repo_commits_as_committer_before",
            "recent_commits_90d",
            "time_since_last_commit_days",
            "experience_percentile_in_repo",
            "cross_repo_commits_before",
            "num_repos_contributed_before",
        ],
    },
    {
        "group": "issue_node",
        "kind": "node",
        "target": "issue",
        "names": [f"dim_{i}" for i in range(4)],
    },
    {
        "group": "pr_node",
        "kind": "node",
        "target": "pull_request",
        "names": [f"dim_{i}" for i in range(4)],
    },
    {
        "group": "tag_node",
        "kind": "node",
        "target": "release_tag",
        "names": [f"dim_{i}" for i in range(4)],
    },
    {
        "group": "commit_file_edge",
        "kind": "edge",
        "target": ("commit", "modifies_file", "file"),
        "names": ["lines_added", "lines_deleted", "complexity", "ownership_ratio"],
    },
    {
        "group": "commit_fn_edge",
        "kind": "edge",
        "target": ("commit", "modifies_func", "function"),
        "names": [
            "loc_before",
            "complexity_before",
            "tokens_before",
            "delta_loc",
            "delta_complexity",
            "delta_tokens",
            "fct_MODIFY",
            "fct_ADD",
            "fct_DELETE",
            "fct_RENAME",
            "fct_REFACTOR",
        ],
    },
    {
        "group": "author_edge",
        "kind": "edge",
        "target": ("commit", "authored_by", "developer"),
        "names": ["recent_commits_30d", "recent_commits_180d", "dev_is_new_contributor"],
    },
    {
        "group": "committer_edge",
        "kind": "edge",
        "target": ("commit", "committed_by", "developer"),
        "names": ["recent_commits_30d", "recent_commits_180d", "dev_is_new_contributor"],
    },
    {
        "group": "owns_edge",
        "kind": "edge",
        "target": ("developer", "owns", "file"),
        "names": ["commits_touching_file_before", "days_since_last_touch", "ownership_ratio"],
    },
    {
        "group": "issue_edge",
        "kind": "edge",
        "target": ("commit", "has_issue", "issue"),
        "names": ["days_open", "issue_comment_count", "has_issue_pr_gap"],
    },
    {
        "group": "pr_edge",
        "kind": "edge",
        "target": ("commit", "has_pr", "pull_request"),
        "names": ["days_open", "pr_comment_count", "has_issue_pr_gap"],
    },
    {
        "group": "tag_edge",
        "kind": "edge",
        "target": ("commit", "has_release", "release_tag"),
        "names": ["activity_since_last_tag"],
    },
]


def load_graph(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(path, weights_only=False)


def extract_mean_feature_vector(g, spec: dict) -> np.ndarray:
    if spec["kind"] == "node":
        if spec["target"] not in g.node_types:
            return np.zeros(len(spec["names"]), dtype=np.float32)
        tensor = g[spec["target"]].x
    else:
        if spec["target"] not in g.edge_types:
            return np.zeros(len(spec["names"]), dtype=np.float32)
        tensor = getattr(g[spec["target"]], "edge_attr", None)
    if tensor is None or tensor.numel() == 0:
        return np.zeros(len(spec["names"]), dtype=np.float32)
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim != 2 or arr.shape[1] != len(spec["names"]):
        raise ValueError(f"Unexpected shape for {spec['group']}: {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros(arr.shape[1], dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)
    return arr.mean(axis=0).astype(np.float32)


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    if np.allclose(y_score, y_score[0]):
        return 0.5
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5
    if auc < 0.5:
        auc = 1.0 - auc
    return auc


def read_filtered_counts(csv_path: Path, key_col: str, keys: set[str], chunksize: int = 200_000) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(csv_path, usecols=[key_col], dtype={key_col: str}, chunksize=chunksize):
        sub = chunk[chunk[key_col].isin(keys)]
        if sub.empty:
            continue
        vc = sub[key_col].value_counts()
        for key, count in vc.items():
            counts[str(key)] += int(count)
    return dict(counts)


def audit_dedup_and_split_leakage(split: pd.DataFrame) -> dict:
    commit_info = pd.read_csv(
        DATA_ROOT / "commit_info.csv",
        usecols=["hash", "commit_label", "repo_url", "author_date"],
        dtype={"hash": str, "commit_label": str, "repo_url": str},
        low_memory=False,
    )

    split_dup_rows = int(split.duplicated("hash", keep=False).sum())
    split_dup_unique = int(split.loc[split.duplicated("hash", keep=False), "hash"].nunique())
    split_repo_conflicts = (
        split.groupby("hash")["repo_split"].nunique().loc[lambda s: s > 1].sort_values(ascending=False)
    )
    split_temporal_conflicts = (
        split.groupby("hash")["temporal_split"].nunique().loc[lambda s: s > 1].sort_values(ascending=False)
    )
    split_label_conflicts = (
        split.groupby("hash")["label"].nunique().loc[lambda s: s > 1].sort_values(ascending=False)
    )

    ci_dup_mask = commit_info.duplicated("hash", keep=False)
    ci_dup_rows = int(ci_dup_mask.sum())
    ci_dup_hashes = commit_info.loc[ci_dup_mask, "hash"].nunique()
    ci_label_conflicts = (
        commit_info.groupby("hash")["commit_label"].nunique().loc[lambda s: s > 1].sort_values(ascending=False)
    )

    built_hashes = set(split["hash"].astype(str))
    ci_for_built = commit_info[commit_info["hash"].astype(str).isin(built_hashes)].copy()
    ci_for_built_dup_hashes = int(ci_for_built.loc[ci_for_built.duplicated("hash", keep=False), "hash"].nunique())

    repo_leakage = split.groupby("repo_url")["repo_split"].nunique()
    repo_leakage = repo_leakage[repo_leakage > 1].sort_values(ascending=False)

    expected_val_actual = set(split.loc[split["repo_split"] == "val", "repo_url"].dropna().unique())
    expected_test_actual = set(split.loc[split["repo_split"] == "test", "repo_url"].dropna().unique())

    repo_hash_sets = {
        part: set(split.loc[split["repo_split"] == part, "hash"].astype(str))
        for part in ["train", "val", "test"]
    }
    temporal_hash_sets = {
        part: set(split.loc[split["temporal_split"] == part, "hash"].astype(str))
        for part in ["train", "val", "test"]
    }

    split["author_date"] = pd.to_datetime(split["author_date"], utc=True, errors="coerce")
    temporal_nonnull = split.dropna(subset=["author_date"]).copy()

    def boundary_stats(df: pd.DataFrame, col: str) -> dict:
        train_dates = df.loc[df[col] == "train", "author_date"]
        val_dates = df.loc[df[col] == "val", "author_date"]
        test_dates = df.loc[df[col] == "test", "author_date"]
        result = {
            "train_max": None if train_dates.empty else str(train_dates.max()),
            "val_min": None if val_dates.empty else str(val_dates.min()),
            "val_max": None if val_dates.empty else str(val_dates.max()),
            "test_min": None if test_dates.empty else str(test_dates.min()),
            "missing_dates": int(split["author_date"].isna().sum()),
        }
        if not train_dates.empty and not val_dates.empty:
            result["train_ge_val_min"] = int((train_dates >= val_dates.min()).sum())
        else:
            result["train_ge_val_min"] = 0
        if not val_dates.empty and not test_dates.empty:
            result["val_ge_test_min"] = int((val_dates >= test_dates.min()).sum())
        else:
            result["val_ge_test_min"] = 0
        return result

    summary = {
        "split_index_rows": int(len(split)),
        "split_duplicate_rows": split_dup_rows,
        "split_duplicate_unique_hashes": split_dup_unique,
        "split_repo_conflict_hashes": split_repo_conflicts.index.tolist()[:20],
        "split_temporal_conflict_hashes": split_temporal_conflicts.index.tolist()[:20],
        "split_label_conflict_hashes": split_label_conflicts.index.tolist()[:20],
        "commit_info_duplicate_rows": ci_dup_rows,
        "commit_info_duplicate_unique_hashes": int(ci_dup_hashes),
        "commit_info_label_conflict_hashes": ci_label_conflicts.index.tolist()[:20],
        "built_hashes_with_commit_info_duplicates": ci_for_built_dup_hashes,
        "repo_leakage_repo_urls": repo_leakage.index.tolist()[:50],
        "repo_split_hash_intersections": {
            "train_val": len(repo_hash_sets["train"] & repo_hash_sets["val"]),
            "train_test": len(repo_hash_sets["train"] & repo_hash_sets["test"]),
            "val_test": len(repo_hash_sets["val"] & repo_hash_sets["test"]),
        },
        "temporal_split_hash_intersections": {
            "train_val": len(temporal_hash_sets["train"] & temporal_hash_sets["val"]),
            "train_test": len(temporal_hash_sets["train"] & temporal_hash_sets["test"]),
            "val_test": len(temporal_hash_sets["val"] & temporal_hash_sets["test"]),
        },
        "repo_split_val_repos_missing": sorted(EXPECTED_VAL_REPOS - expected_val_actual),
        "repo_split_val_repos_unexpected": sorted(expected_val_actual - EXPECTED_VAL_REPOS),
        "repo_split_test_repos_missing": sorted(EXPECTED_TEST_REPOS - expected_test_actual),
        "repo_split_test_repos_unexpected": sorted(expected_test_actual - EXPECTED_TEST_REPOS),
        "temporal_boundaries": boundary_stats(temporal_nonnull, "temporal_split"),
    }
    return summary


def audit_scaler_leakage(split: pd.DataFrame) -> dict:
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    repo_to_split = split.groupby("repo_url")["repo_split"].agg(lambda s: sorted(set(s))).to_dict()
    leaked: dict[str, list[str]] = {}
    unknown: dict[str, list[str]] = {}
    per_group_counts: dict[str, int] = {}
    for group, entries in scaler.items():
        repos = list(entries.keys())
        per_group_counts[group] = len(repos)
        bad = [repo for repo in repos if repo in repo_to_split and repo_to_split[repo] != ["train"]]
        if bad:
            leaked[group] = sorted(bad)
        missing = [repo for repo in repos if repo not in repo_to_split]
        if missing:
            unknown[group] = sorted(missing)
    return {
        "groups": per_group_counts,
        "leaked_groups": leaked,
        "unknown_repo_groups": unknown,
    }


def audit_feature_leakage(split: pd.DataFrame) -> dict:
    rows = []
    for part in ["train", "val", "test"]:
        part_df = split[split[SPLIT_COL] == part].copy()
        part_df = part_df[part_df["hash"].astype(str).map(lambda h: (GRAPHS_DIR / f"{h}.pt").exists())]
        sample_n = min(SAMPLE_PER_SPLIT, len(part_df))
        sampled = part_df.sample(n=sample_n, random_state=SEED) if sample_n else part_df.iloc[0:0]
        for row in sampled.itertuples(index=False):
            rows.append({
                "hash": str(row.hash),
                "split": part,
                "label": int(row.label),
            })

    feature_rows = []
    pooled_features: dict[str, list[np.ndarray]] = defaultdict(list)
    pooled_labels: dict[str, list[int]] = defaultdict(list)
    per_split_features: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    per_split_labels: dict[tuple[str, str], list[int]] = defaultdict(list)

    for item in rows:
        g = load_graph(GRAPHS_DIR / f"{item['hash']}.pt")
        for spec in GROUPS:
            vec = extract_mean_feature_vector(g, spec)
            pooled_features[spec["group"]].append(vec)
            pooled_labels[spec["group"]].append(item["label"])
            per_split_features[(item["split"], spec["group"])].append(vec)
            per_split_labels[(item["split"], spec["group"])].append(item["label"])

    def build_auc_rows(scope: str, split_name: str, group: str, feats: list[np.ndarray], labels: list[int]) -> None:
        arr = np.vstack(feats)
        y = np.asarray(labels, dtype=np.int32)
        names = next(spec["names"] for spec in GROUPS if spec["group"] == group)
        for idx, name in enumerate(names):
            auc = auc_safe(y, arr[:, idx])
            feature_rows.append({
                "scope": scope,
                "split": split_name,
                "feature_group": group,
                "feature_idx": idx,
                "feature_name": name,
                "auc": auc,
                "n_samples": int(len(y)),
                "n_pos": int(y.sum()),
            })

    for group, feats in pooled_features.items():
        build_auc_rows("pooled", "all", group, feats, pooled_labels[group])
    for (part, group), feats in per_split_features.items():
        build_auc_rows("per_split", part, group, feats, per_split_labels[(part, group)])

    auc_df = pd.DataFrame(feature_rows).sort_values(["scope", "split", "auc"], ascending=[True, True, False])
    auc_df.to_csv(AUDIT_DIR / "feature_auc_report.csv", index=False)
    flagged = auc_df[auc_df["auc"] > 0.70].copy()
    flagged.to_csv(AUDIT_DIR / "feature_auc_flagged_gt_0_70.csv", index=False)
    return {
        "sample_counts": pd.DataFrame(rows).groupby("split").size().to_dict(),
        "flagged_count": int(len(flagged)),
        "top_flagged": flagged.head(50).to_dict(orient="records"),
        "top_overall": auc_df.sort_values("auc", ascending=False).head(50).to_dict(orient="records"),
    }


def audit_integration(split: pd.DataFrame) -> dict:
    sample_df = split.copy()
    sample_df = sample_df[sample_df["hash"].astype(str).map(lambda h: (GRAPHS_DIR / f"{h}.pt").exists())]
    groups = {
        "VCC": 17,
        "FC": 17,
        "normal": 16,
    }
    sampled_parts = []
    for label_name, n in groups.items():
        subset = sample_df[sample_df["commit_label"] == label_name]
        take = min(n, len(subset))
        if take:
            sampled_parts.append(subset.sample(n=take, random_state=SEED))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(n=min(INTEGRATION_SAMPLE_SIZE, len(sampled)), random_state=SEED).reset_index(drop=True)

    sample_hashes = set(sampled["hash"].astype(str))
    commit_labels = pd.read_csv(
        DATA_ROOT / "commit_info.csv",
        usecols=["hash", "commit_label"],
        dtype={"hash": str, "commit_label": str},
        low_memory=False,
    ).drop_duplicates("hash").set_index("hash")["commit_label"].to_dict()
    fn_counts = read_filtered_counts(DATA_ROOT / "function_info.csv", "hash", sample_hashes)
    file_counts = read_filtered_counts(DATA_ROOT / "file_info.csv", "hash", sample_hashes)

    mismatches = []
    passed = 0
    for row in sampled.itertuples(index=False):
        h = str(row.hash)
        g = load_graph(GRAPHS_DIR / f"{h}.pt")
        expected_label = 1 if commit_labels.get(h) == "VCC" else 0
        actual_label = int(g.y.item())
        func_count_expected = int(fn_counts.get(h, 0))
        file_count_expected = int(file_counts.get(h, 0))
        func_count_actual = int(g["function"].x.shape[0]) if "function" in g.node_types else 0
        file_count_actual = int(g["file"].x.shape[0]) if "file" in g.node_types else 0
        dev_count_actual = int(g["developer"].x.shape[0]) if "developer" in g.node_types else 0

        errors = []
        if actual_label != expected_label:
            errors.append(f"label mismatch graph={actual_label} raw={expected_label}")
        if func_count_actual != func_count_expected:
            errors.append(f"function count graph={func_count_actual} raw={func_count_expected}")
        if file_count_actual != file_count_expected:
            errors.append(f"file count graph={file_count_actual} raw={file_count_expected}")
        if dev_count_actual < 1:
            errors.append("developer count < 1")

        fn_edge = ("function", "in_commit_fn", "commit")
        if func_count_actual > 0:
            if fn_edge not in g.edge_types:
                errors.append("missing function->commit back-edge")
            else:
                src_nodes = set(g[fn_edge].edge_index[0].detach().cpu().numpy().tolist())
                if len(src_nodes) != func_count_actual:
                    errors.append(f"function back-edge coverage {len(src_nodes)}/{func_count_actual}")

        file_edge = ("file", "in_commit", "commit")
        if file_count_actual > 0:
            if file_edge not in g.edge_types:
                errors.append("missing file->commit back-edge")
            else:
                src_nodes = set(g[file_edge].edge_index[0].detach().cpu().numpy().tolist())
                if len(src_nodes) != file_count_actual:
                    errors.append(f"file back-edge coverage {len(src_nodes)}/{file_count_actual}")

        graph_has_code = func_count_expected > 0 or file_count_expected > 0
        if graph_has_code:
            fn_nonzero = False
            hunk_nonzero = False
            if "function" in g.node_types and g["function"].x.shape[0] > 0:
                fn_emb = g["function"].x[:, 8:]
                fn_nonzero = bool((torch.linalg.norm(fn_emb, dim=1) > 1e-6).any().item())
            if "hunk" in g.node_types and g["hunk"].x.shape[0] > 0:
                h_emb = g["hunk"].x[:, 2:]
                hunk_nonzero = bool((torch.linalg.norm(h_emb, dim=1) > 1e-6).any().item())
            if not (fn_nonzero or hunk_nonzero):
                errors.append("no non-zero function/hunk embedding for code commit")

        authored_edge = ("commit", "authored_by", "developer")
        if dev_count_actual > 0:
            if authored_edge not in g.edge_types or g[authored_edge].edge_index.shape[1] < 1:
                errors.append("missing commit->authored_by edge")

        if errors:
            mismatches.append({
                "hash": h,
                "commit_label": row.commit_label,
                "errors": " | ".join(errors),
            })
        else:
            passed += 1

    mismatch_df = pd.DataFrame(mismatches)
    mismatch_df.to_csv(AUDIT_DIR / "integration_mismatches.csv", index=False)
    return {
        "sampled_commits": int(len(sampled)),
        "passed": int(passed),
        "failed": int(len(mismatches)),
        "mismatches": mismatches[:50],
    }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(
        SPLIT_INDEX,
        dtype={
            "hash": str,
            "commit_label": str,
            "label": int,
            "repo_url": str,
            "repo_split": str,
            "temporal_split": str,
        },
        low_memory=False,
    )

    dedup_split = audit_dedup_and_split_leakage(split.copy())
    scaler = audit_scaler_leakage(split.copy())
    feature = audit_feature_leakage(split.copy())
    integration = audit_integration(split.copy())

    summary = {
        "dedup_split_leakage": dedup_split,
        "scaler_leakage": scaler,
        "feature_leakage": feature,
        "integration": integration,
    }

    with open(AUDIT_DIR / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
