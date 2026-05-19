"""
Audit the finalized graph-input package for graph-builder compatibility and
leakage-sensitive feature usage.

This script does not build graphs. It answers three migration questions:

1. Can the current legacy graph builder consume the final package as-is?
2. Which commits have enough code coverage to support graph-based training?
3. Which columns should be treated as metadata or ablation-only features?

Usage:
  conda run -n thesis python scripts/audit_final_package.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "data_new" / "analysis_outputs" / "final_graph_inputs_v1"


SAFE_FILE_FEATURES = [
    "num_lines_of_code",
    "complexity",
    "token_count",
]

SAFE_FUNCTION_FEATURES = [
    "num_lines_of_code",
    "complexity",
    "token_count",
    "length",
    "top_nesting_level",
]

SAFE_HUNK_FEATURES = [
    "complexity",
    "token_count",
]

SAFE_COMMIT_FEATURES = [
    "pr_count_90d",
    "pr_age_median",
    "pr_closed_last_90d",
    "pr_open_velocity_90d",
    "issue_open_90d",
    "issue_age_median",
    "issues_closed_last_90d",
    "issue_open_velocity_90d",
    "days_since_prev_tag",
    "tags_last_365d",
    "avg_release_cadence_days",
    "days_since_prev_tag_norm",
    "dev_experience_days",
    "dev_commits_before",
    "dev_is_new_contributor",
    "repo_commits_last_90d",
    "repo_active_authors_90d",
    "activity_since_last_tag",
    "pr_to_issue_open_ratio_90d",
    "issue_to_pr_closed_ratio_90d",
    "has_issue_pr_gap",
]

METADATA_COMMIT_FEATURES = [
    "repo_url",
    "commit_datetime",
    "has_sdlc_data",
]

ABLATON_ONLY_FUNCTION_FEATURES = [
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

LEGACY_REQUIRED_COLUMNS = {
    "file_features.csv": {"hash", "filename"},
    "function_features.csv": {"hash", "name", "filename"},
    "hunk_features.csv": {"hash", "filename"},
}


def read_csv(name: str, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(FINAL_DIR / name, usecols=usecols)


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def report_schema_compatibility() -> None:
    print_section("Schema Compatibility")
    incompatible = False
    for file_name, required_cols in LEGACY_REQUIRED_COLUMNS.items():
        cols = set(pd.read_csv(FINAL_DIR / file_name, nrows=0).columns)
        missing = sorted(required_cols - cols)
        if missing:
            incompatible = True
            print(f"{file_name}: incompatible with legacy topology, missing {missing}")
        else:
            print(f"{file_name}: compatible with legacy topology")

    if incompatible:
        print("Result: current legacy file->function graph structure cannot be rebuilt from the final package.")
        print("Recommended migration: connect anonymous file/function/hunk rows directly to the commit node.")


def report_coverage() -> None:
    print_section("Coverage")
    manifest = read_csv("build_manifest.csv", usecols=["hash", "commit_type", "label"])
    file_hashes = set(read_csv("file_features.csv", usecols=["hash"])["hash"])
    function_hashes = set(read_csv("function_features.csv", usecols=["hash"])["hash"])
    hunk_hashes = set(read_csv("hunk_features.csv", usecols=["hash"])["hash"])
    ownership_hashes = set(read_csv("ownership_window_full_aligned_manifest.csv", usecols=["commit_hash"])["commit_hash"])
    msg_hashes = set(read_csv("commit_msg_index.csv", usecols=["hash"])["hash"])

    manifest_hashes = set(manifest["hash"])
    code_hashes = file_hashes | function_hashes | hunk_hashes

    print(f"manifest commits: {len(manifest_hashes)}")
    print(f"commits with file rows: {len(file_hashes)}")
    print(f"commits with function rows: {len(function_hashes)}")
    print(f"commits with hunk rows: {len(hunk_hashes)}")
    print(f"commits with any code rows: {len(code_hashes)}")
    print(f"commits missing any code rows: {len(manifest_hashes - code_hashes)}")
    print(f"commits missing commit-message embedding: {len(manifest_hashes - msg_hashes)}")
    print(f"commits missing ownership rows: {len(manifest_hashes - ownership_hashes)}")
    print(f"commits with function rows but no file rows: {len((manifest_hashes - file_hashes) & function_hashes)}")

    grouped = manifest.assign(has_code=manifest["hash"].isin(code_hashes))
    print()
    print(grouped.groupby(["commit_type", "label", "has_code"]).size().to_string())

    print()
    print("Policy recommendation:")
    print("- Keep file-only and function-only commits as valid code graphs.")
    print("- Do not silently train on commit-only graphs for commits with no code rows.")
    print("- Split graph training and SDLC-only training explicitly if full manifest coverage is required.")


def report_embeddings() -> None:
    print_section("Embedding Alignment")
    function_features = read_csv("function_features.csv", usecols=["hash"])
    function_index = read_csv("function_embeddings_index.csv", usecols=["hash"])
    hunk_features = read_csv("hunk_features.csv", usecols=["hash"])
    hunk_index = read_csv("hunk_embeddings_index.csv", usecols=["hash"])

    print(
        "function embeddings aligned:",
        len(function_features) == len(function_index)
        and function_features["hash"].equals(function_index["hash"]),
    )
    print(
        "hunk embeddings aligned:",
        len(hunk_features) == len(hunk_index)
        and hunk_features["hash"].equals(hunk_index["hash"]),
    )


def report_leakage_surface() -> None:
    print_section("Leakage Surface")
    manifest_cols = list(pd.read_csv(FINAL_DIR / "build_manifest.csv", nrows=0).columns)
    commit_cols = list(pd.read_csv(
        FINAL_DIR / "final_commit_level_features_v2_normalized_model_features.csv",
        nrows=0,
    ).columns)
    fn_cols = list(pd.read_csv(FINAL_DIR / "function_features.csv", nrows=0).columns)

    print(f"label columns kept only in build_manifest: {manifest_cols}")
    print(f"commit metadata to exclude from tensors: {METADATA_COMMIT_FEATURES}")
    print(f"safer commit feature set: {SAFE_COMMIT_FEATURES}")
    print(f"safer file feature set: {SAFE_FILE_FEATURES}")
    print(f"safer function feature set: {SAFE_FUNCTION_FEATURES}")
    print(f"safer hunk feature set: {SAFE_HUNK_FEATURES}")

    risky_fn = [col for col in ABLATON_ONLY_FUNCTION_FEATURES if col in fn_cols]
    print(f"ablation-only function columns present: {risky_fn}")

    forbidden_commit = {"commit_type", "label", "y_binary"} & set(commit_cols)
    print(f"forbidden label columns in commit feature table: {sorted(forbidden_commit)}")

    print()
    print("Default rule:")
    print("- Build labels only from build_manifest.csv.")
    print("- Never place commit_type, label, y_binary, repo_url, commit_datetime, or has_sdlc_data into node tensors.")
    print("- Keep function delta/change-type columns out of the default training graph.")


def report_developer_join() -> None:
    print_section("Developer Join")
    dev = read_csv("developer_info_full_aligned_manifest.csv", usecols=["canonical_dev_key"])
    own = read_csv("ownership_window_full_aligned_manifest.csv", usecols=["canonical_dev_key"])

    dev_keys = set(dev["canonical_dev_key"])
    own_keys = set(own["canonical_dev_key"])

    print(f"developer keys in developer table: {len(dev_keys)}")
    print(f"developer keys in ownership table: {len(own_keys)}")
    print(f"ownership keys missing from developer table: {len(own_keys - dev_keys)}")
    print("Rule: join developer and ownership only on canonical_dev_key.")
    print("Rule: filter ownership rows to known developer keys or attach zero-feature developer nodes.")


def main() -> None:
    if not FINAL_DIR.exists():
        raise SystemExit(f"Missing final package directory: {FINAL_DIR}")

    print(f"Auditing final graph package: {FINAL_DIR}")
    report_schema_compatibility()
    report_coverage()
    report_embeddings()
    report_leakage_surface()
    report_developer_join()


if __name__ == "__main__":
    main()
