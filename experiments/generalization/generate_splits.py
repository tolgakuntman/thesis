"""
scripts/generate_splits.py

Generate and save all split-test split files.

For each strategy a CSV is written to:
    outputs/splits/<strategy_name>/split_index.csv

The CSV has columns: hash, label, repo_url, repo_split [, extra cols]
The 'repo_split' column contains train/val/test — compatible with train.py:
    python scripts/train.py --split_type repo_split \
        --split_index outputs/splits/<name>/split_index.csv

A metadata JSON and a leakage-check report are also saved per strategy.

Usage:
    python scripts/generate_splits.py [--out_dir outputs/splits] [--seed 42]
    python scripts/generate_splits.py --dry_run
    python scripts/generate_splits.py --splits repo temporal cwe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for split_strategies

from split_strategies import (
    repo_split,
    temporal_split,
    repo_temporal_split,
    developer_split,
    file_split,
    function_split,
    cwe_split,
    severity_split,
    hard_negative_split,
    graph_structure_split,
    cold_start_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Canonical data paths (mirror of graph_dataset.py) ────────────────────────

def _resolve_data_root() -> Path:
    parent = ROOT.parent
    for candidate in [
        parent / "ICVul_pp" / "graph_ready_sampling_v2",
        parent / "graph_ready_sampling_v2",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "ICVul_pp/graph_ready_sampling_v2 not found. "
        "Expected adjacent to thesis/ directory."
    )


def _resolve_split_index() -> Path:
    for candidate in [
        ROOT / "outputs" / "graph_ready_v2" / "split_index.csv",
        ROOT / "outputs" / "final_graph_ready" / "split_index.csv",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("split_index.csv not found. Run create_split_index_v2.py first.")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_base(split_index_path: Path) -> pd.DataFrame:
    """Load split_index.csv and parse author_date."""
    df = pd.read_csv(split_index_path, low_memory=False)
    df["author_date"] = pd.to_datetime(df["author_date"], utc=True, errors="coerce")
    df["hash"]  = df["hash"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_commit_info(data_root: Path) -> pd.DataFrame:
    usecols = [
        "hash", "cwe_id",
        "num_lines_changed", "num_files_changed",
        "num_lines_added", "num_lines_deleted",
        "dmm_unit_size", "dmm_unit_complexity", "dmm_unit_interfacing",
    ]
    path = data_root / "commit_info.csv"
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols      = [c for c in usecols if c in available]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["hash"] = df["hash"].astype(str)
    return df


def load_commit_author(data_root: Path) -> pd.DataFrame:
    path = data_root / "commit_author.csv"
    df   = pd.read_csv(path, low_memory=False)
    df["commit_hash"] = df["commit_hash"].astype(str)
    df["dev_id"]      = df["dev_id"].astype(str)
    return df


def load_file_info(data_root: Path) -> pd.DataFrame:
    for name in ("file_info_new_dedup.csv", "file_info_new.csv", "file_info.csv"):
        p = data_root / name
        if p.exists():
            available = pd.read_csv(p, nrows=0).columns.tolist()
            file_col  = "filename" if "filename" in available else "new_path"
            cols      = [c for c in ["hash", file_col] if c in available]
            df = pd.read_csv(p, usecols=cols, low_memory=False)
            df["hash"] = df["hash"].astype(str)
            return df
    logger.warning("file_info CSV not found — file_split will fall back to train-only")
    return pd.DataFrame(columns=["hash", "filename"])


def load_function_info(data_root: Path) -> pd.DataFrame:
    path = data_root / "function_info.csv"
    cols = ["hash", "name", "filename"]
    available = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in cols if c in available]
    df   = pd.read_csv(path, usecols=cols, low_memory=False)
    df["hash"] = df["hash"].astype(str)
    return df


# ── Metadata and leakage checks ───────────────────────────────────────────────

def build_metadata(name: str, df: pd.DataFrame) -> dict:
    meta: dict = {"strategy": name, "total": int(len(df))}
    for split in ["train", "val", "test"]:
        sub = df[df["repo_split"] == split]
        n   = len(sub)
        pos = int(sub["label"].sum())
        neg = n - pos
        repos = sub["repo_url"].dropna().unique().tolist()
        meta[split] = {
            "n_samples": n,
            "n_pos": pos,
            "n_neg": neg,
            "pos_ratio": round(pos / max(n, 1), 4),
            "n_repos": len(repos),
            "repos": sorted(repos),
        }
        if "author_date" in df.columns:
            dated = sub.dropna(subset=["author_date"])
            if len(dated):
                meta[split]["date_min"] = str(dated["author_date"].min())
                meta[split]["date_max"] = str(dated["author_date"].max())
    return meta


# Checks that are EXPECTED to fail by design for each strategy.
# These are marked as notes in the report, not hard failures.
_EXPECTED_BY_DESIGN: dict[str, frozenset[str]] = {
    # Repo split does not impose temporal ordering
    "repo":           frozenset(["temporal_train_after_test_start"]),
    # Temporal split intentionally puts same repos in train and test
    "temporal":       frozenset(["repo_train_test_overlap"]),
    # Splits by non-repo/non-temporal dimension → both checks fire by design
    "developer":      frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    "file":           frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    "function":       frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    "cwe":            frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    "severity":       frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    "hard_negative":  frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    "graph_structure":frozenset(["repo_train_test_overlap", "temporal_train_after_test_start"]),
    # Cold-start intentionally includes existing val repos in train → no repo-overlap concern there
    "cold_start":     frozenset(["temporal_train_after_test_start"]),
    # Repo-temporal: temporal check fires because test repos (FFmpeg, OpenSSL, …)
    # have commits from the 1990s that predate pool commits; by design the pool's
    # temporal cutoff governs training, not the absolute test-repo commit history.
    "repo_temporal":  frozenset(["temporal_train_after_test_start"]),
}


def leakage_check(name: str, df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Strict leakage checks.  Returns dict of {check_name: [violations]}.
    Each violation is a string describing the leaked sample.
    """
    errors: dict[str, list[str]] = {}

    train = df[df["repo_split"] == "train"]
    val   = df[df["repo_split"] == "val"]
    test  = df[df["repo_split"] == "test"]

    # 1. Duplicate hash across splits
    hash_splits = df.groupby("hash")["repo_split"].nunique()
    dup = hash_splits[hash_splits > 1].index.tolist()
    if dup:
        errors["duplicate_hash"] = [f"hash={h}" for h in dup[:20]]

    # 2. Repo overlap (train vs test)
    train_repos = set(train["repo_url"].dropna())
    test_repos  = set(test["repo_url"].dropna())
    val_repos   = set(val["repo_url"].dropna())
    tr_te = train_repos & test_repos
    if tr_te:
        errors["repo_train_test_overlap"] = sorted(tr_te)

    # 3. Same commit appears in multiple splits (should be caught by check 1, kept explicit)
    for split_a, set_a in [("train", set(train["hash"])), ("val", set(val["hash"]))]:
        te_overlap = set_a & set(test["hash"])
        if te_overlap:
            errors[f"commit_{split_a}_test_overlap"] = [str(h) for h in list(te_overlap)[:20]]

    # 4. Temporal leakage: any training commit dated after the earliest test commit
    if "author_date" in df.columns:
        test_dated  = test.dropna(subset=["author_date"])
        train_dated = train.dropna(subset=["author_date"])
        if len(test_dated) and len(train_dated):
            earliest_test = test_dated["author_date"].min()
            late_train    = train_dated[train_dated["author_date"] > earliest_test]
            if len(late_train):
                errors["temporal_train_after_test_start"] = [
                    f"hash={r.hash}  date={r.author_date}"
                    for r in late_train.head(5).itertuples()
                ]

    return errors


# ── Save utilities ────────────────────────────────────────────────────────────

def save_split(
    name: str,
    df: pd.DataFrame,
    out_dir: Path,
    dry_run: bool = False,
) -> dict:
    split_dir = out_dir / name
    if not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    meta = build_metadata(name, df)
    errors = leakage_check(name, df)
    expected = _EXPECTED_BY_DESIGN.get(name, frozenset())
    real_errors   = {k: v for k, v in errors.items() if k not in expected}
    design_notes  = {k: v for k, v in errors.items() if k in expected}
    meta["leakage_checks"]        = real_errors
    meta["leakage_by_design"]     = design_notes
    meta["leakage_ok"]            = len(real_errors) == 0

    if not dry_run:
        # Save split CSV (always include author_date if available)
        save_cols = ["hash", "label", "repo_url", "repo_split"]
        if "author_date" in df.columns:
            save_cols.append("author_date")
        for extra in ["hn_score", "graph_score"]:
            if extra in df.columns:
                save_cols.append(extra)
        df[save_cols].to_csv(split_dir / "split_index.csv", index=False)

        with open(split_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        report_lines = [f"=== Leakage report: {name} ==="]
        if real_errors:
            for check, msgs in real_errors.items():
                report_lines.append(f"FAIL  {check}:")
                for m in msgs[:10]:
                    report_lines.append(f"      {m}")
        else:
            report_lines.append("PASS  (no unexpected leakage)")
        if design_notes:
            report_lines.append("\nNOTE (expected by split design — not a bug):")
            for check, msgs in design_notes.items():
                report_lines.append(f"  BY-DESIGN  {check}: {len(msgs)} instance(s)")
        (split_dir / "leakage_report.txt").write_text("\n".join(report_lines))

    # Console summary
    n_real = sum(len(v) for v in real_errors.values())
    n_note = sum(len(v) for v in design_notes.values())
    status = "DRY  " if dry_run else "SAVED"
    logger.info(
        "%s  %-25s  train=%d  val=%d  test=%d  leakage_errors=%d  design_notes=%d",
        status, name,
        meta["train"]["n_samples"], meta["val"]["n_samples"], meta["test"]["n_samples"],
        n_real, n_note,
    )
    for check, msgs in real_errors.items():
        logger.warning("  LEAKAGE  %s: %s", check, msgs[0])

    return meta


# ── Strategy registry ─────────────────────────────────────────────────────────

ALL_SPLITS = [
    "repo", "temporal", "repo_temporal",
    "developer", "file", "function",
    "cwe", "severity",
    "hard_negative", "graph_structure", "cold_start",
]


def run_strategy(
    name: str,
    base_df: pd.DataFrame,
    commit_info: pd.DataFrame,
    commit_author: pd.DataFrame,
    file_info: pd.DataFrame,
    fn_info: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Dispatch to the appropriate strategy function."""

    # Merge commit_info features into base_df where needed
    def _enriched() -> pd.DataFrame:
        extra_cols = [
            "cwe_id", "num_lines_changed", "num_files_changed",
            "num_lines_added", "num_lines_deleted",
            "dmm_unit_size", "dmm_unit_complexity", "dmm_unit_interfacing",
        ]
        merge_cols = ["hash"] + [c for c in extra_cols if c in commit_info.columns]
        return base_df.merge(commit_info[merge_cols], on="hash", how="left")

    if name == "repo":
        return repo_split(base_df, seed=seed)

    elif name == "temporal":
        return temporal_split(base_df, seed=seed)

    elif name == "repo_temporal":
        return repo_temporal_split(base_df, seed=seed)

    elif name == "developer":
        return developer_split(base_df, commit_author, seed=seed)

    elif name == "file":
        return file_split(base_df, file_info, seed=seed)

    elif name == "function":
        return function_split(base_df, fn_info, seed=seed)

    elif name == "cwe":
        return cwe_split(_enriched(), seed=seed)

    elif name == "severity":
        return severity_split(_enriched(), seed=seed)

    elif name == "hard_negative":
        return hard_negative_split(_enriched(), seed=seed)

    elif name == "graph_structure":
        return graph_structure_split(_enriched(), seed=seed)

    elif name == "cold_start":
        return cold_start_split(base_df, seed=seed, n_cold_repos=5)

    else:
        raise ValueError(f"Unknown split strategy: {name!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate split-test split files")
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS,
                   choices=ALL_SPLITS,
                   help="Which splits to generate (default: all)")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: outputs/splits)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_index", default=None,
                   help="Override path to split_index.csv")
    p.add_argument("--dry_run", action="store_true",
                   help="Validate splits without writing files")
    p.add_argument("--continue_on_error", action="store_true",
                   help="Continue if a strategy fails")
    return p.parse_args()


def main():
    args   = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "splits"

    try:
        data_root = _resolve_data_root()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    try:
        split_index_path = Path(args.split_index) if args.split_index else _resolve_split_index()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    logger.info("Loading base data from %s", split_index_path)
    base_df = load_base(split_index_path)
    logger.info("Base dataset: %d commits  %d VCC", len(base_df), int(base_df["label"].sum()))

    logger.info("Loading auxiliary data from %s", data_root)
    commit_info   = load_commit_info(data_root)
    commit_author = load_commit_author(data_root)
    file_info     = load_file_info(data_root)
    fn_info       = load_function_info(data_root)

    all_meta: list[dict] = []
    n_ok, n_fail = 0, 0

    for name in args.splits:
        logger.info("\n%s  Generating: %s  %s", "="*30, name, "="*30)
        try:
            result_df = run_strategy(
                name, base_df, commit_info, commit_author, file_info, fn_info, args.seed
            )
            # Re-attach author_date for metadata / downstream temporal checks
            if "author_date" not in result_df.columns and "author_date" in base_df.columns:
                result_df = result_df.merge(
                    base_df[["hash", "author_date"]], on="hash", how="left"
                )
            meta = save_split(name, result_df, out_dir, dry_run=args.dry_run)
            all_meta.append(meta)
            n_ok += 1
        except Exception as exc:
            logger.error("FAILED: %s — %s", name, exc, exc_info=True)
            n_fail += 1
            if not args.continue_on_error:
                sys.exit(1)

    # Summary table
    if not args.dry_run and all_meta:
        summary_path = out_dir / "splits_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_meta, f, indent=2, default=str)
        logger.info("\nSummary saved → %s", summary_path)

    logger.info("\nDone: %d ok  %d failed", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
