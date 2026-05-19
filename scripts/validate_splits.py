"""
scripts/validate_splits.py

Independent validation of generated split files.

Loads each split from outputs/splits/<name>/split_index.csv and runs
strict leakage assertions.  Exits non-zero if any check fails.

Usage:
    python scripts/validate_splits.py
    python scripts/validate_splits.py --splits_dir outputs/splits
    python scripts/validate_splits.py --splits repo temporal cwe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Checks that are expected to fail by design for each split strategy.
# These are reported as INFO (by-design) rather than WARNING (fail).
_EXPECTED_BY_DESIGN: dict[str, frozenset[str]] = {
    "repo":            frozenset(["temporal_no_future_leak"]),
    "temporal":        frozenset(["no_repo_train_test_overlap"]),
    "repo_temporal":   frozenset(["temporal_no_future_leak"]),
    "developer":       frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "file":            frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "function":        frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "cwe":             frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "severity":        frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "hard_negative":   frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "graph_structure": frozenset(["no_repo_train_test_overlap", "temporal_no_future_leak"]),
    "cold_start":      frozenset(["temporal_no_future_leak"]),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Individual checks ─────────────────────────────────────────────────────────

def check_no_dup_hashes(df: pd.DataFrame) -> list[str]:
    dup = df.groupby("hash")["repo_split"].nunique()
    bad = dup[dup > 1].index.tolist()
    return [f"hash {h} appears in multiple splits" for h in bad[:20]]


def check_no_repo_train_test_overlap(df: pd.DataFrame) -> list[str]:
    train_repos = set(df[df["repo_split"] == "train"]["repo_url"].dropna())
    test_repos  = set(df[df["repo_split"] == "test"]["repo_url"].dropna())
    bad = train_repos & test_repos
    return [f"repo {r} in both train and test" for r in sorted(bad)]


def check_no_commit_train_test_overlap(df: pd.DataFrame) -> list[str]:
    train_hashes = set(df[df["repo_split"] == "train"]["hash"])
    test_hashes  = set(df[df["repo_split"] == "test"]["hash"])
    bad = train_hashes & test_hashes
    return [f"hash {h}" for h in list(bad)[:20]]


def check_no_commit_val_test_overlap(df: pd.DataFrame) -> list[str]:
    val_hashes  = set(df[df["repo_split"] == "val"]["hash"])
    test_hashes = set(df[df["repo_split"] == "test"]["hash"])
    bad = val_hashes & test_hashes
    return [f"hash {h}" for h in list(bad)[:20]]


def check_splits_nonempty(df: pd.DataFrame) -> list[str]:
    errors = []
    for split in ["train", "val", "test"]:
        n = (df["repo_split"] == split).sum()
        if n == 0:
            errors.append(f"split '{split}' is empty")
    return errors


def check_has_positives(df: pd.DataFrame) -> list[str]:
    errors = []
    for split in ["train", "val", "test"]:
        sub = df[df["repo_split"] == split]
        if len(sub) > 0 and int(sub["label"].sum()) == 0:
            errors.append(f"split '{split}' has no positive (VCC) samples")
    return errors


def check_no_unknown_splits(df: pd.DataFrame) -> list[str]:
    valid = {"train", "val", "test"}
    bad   = set(df["repo_split"].unique()) - valid
    return [f"unknown split value: {v!r}" for v in sorted(bad)]


def check_temporal_no_future_leak(df: pd.DataFrame) -> list[str]:
    if "author_date" not in df.columns:
        return []
    df2 = df.copy()
    df2["author_date"] = pd.to_datetime(df2["author_date"], utc=True, errors="coerce")
    test_dated  = df2[df2["repo_split"] == "test"].dropna(subset=["author_date"])
    train_dated = df2[df2["repo_split"] == "train"].dropna(subset=["author_date"])
    if not len(test_dated) or not len(train_dated):
        return []
    earliest_test = test_dated["author_date"].min()
    late = train_dated[train_dated["author_date"] > earliest_test]
    if len(late) > 0:
        frac = len(late) / len(train_dated)
        return [
            f"{len(late)} train commits ({100*frac:.1f}%) are dated after "
            f"earliest test commit ({earliest_test})"
        ]
    return []


# All check functions in order
_CHECKS = [
    ("no_unknown_splits",          check_no_unknown_splits),
    ("splits_nonempty",            check_splits_nonempty),
    ("no_dup_hashes",              check_no_dup_hashes),
    ("no_commit_train_test_overlap", check_no_commit_train_test_overlap),
    ("no_commit_val_test_overlap", check_no_commit_val_test_overlap),
    ("no_repo_train_test_overlap", check_no_repo_train_test_overlap),
    ("has_positives",              check_has_positives),
    ("temporal_no_future_leak",    check_temporal_no_future_leak),
]


# ── Validate one split ────────────────────────────────────────────────────────

def validate_one(name: str, split_dir: Path, strict: bool = True) -> bool:
    csv_path = split_dir / "split_index.csv"
    if not csv_path.exists():
        logger.error("MISSING  %-25s  %s", name, csv_path)
        return False

    df = pd.read_csv(csv_path, low_memory=False)
    n_train = (df["repo_split"] == "train").sum()
    n_val   = (df["repo_split"] == "val").sum()
    n_test  = (df["repo_split"] == "test").sum()
    n_pos   = int(df["label"].sum())
    logger.info(
        "VALIDATING  %-25s  total=%d  train=%d  val=%d  test=%d  pos=%d",
        name, len(df), n_train, n_val, n_test, n_pos,
    )

    expected  = _EXPECTED_BY_DESIGN.get(name, frozenset())
    all_ok    = True
    results: list[dict] = []
    for check_name, check_fn in _CHECKS:
        errors     = check_fn(df)
        by_design  = check_name in expected
        ok         = len(errors) == 0
        if ok:
            status = "PASS"
        elif by_design:
            status = "BY-DESIGN"
        else:
            status = "FAIL"
        results.append({"check": check_name, "status": status, "errors": errors[:5]})
        if ok or by_design:
            logger.info("  %s  %s", status, check_name)
            if by_design and errors:
                logger.info("       (%d instance(s) — expected for this split type)", len(errors))
        else:
            for msg in errors[:5]:
                logger.warning("  %s  %-35s  %s", status, check_name, msg)
            if len(errors) > 5:
                logger.warning("       ... and %d more", len(errors) - 5)
            if strict:
                all_ok = False

    # Save validation results
    report_path = split_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {"strategy": name, "n_rows": len(df), "checks": results},
            f, indent=2,
        )

    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Validate generated split files for leakage")
    p.add_argument("--splits_dir", default=None,
                   help="Directory containing split subdirs (default: outputs/splits)")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Which splits to validate (default: all found)")
    p.add_argument("--no_strict", action="store_true",
                   help="Report failures without exiting non-zero")
    return p.parse_args()


def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir) if args.splits_dir else ROOT / "outputs" / "splits"

    if not splits_dir.exists():
        logger.error("splits_dir not found: %s", splits_dir)
        logger.error("Run generate_splits.py first.")
        sys.exit(1)

    if args.splits:
        names = args.splits
    else:
        names = sorted([d.name for d in splits_dir.iterdir() if d.is_dir()])

    if not names:
        logger.warning("No split directories found in %s", splits_dir)
        sys.exit(0)

    logger.info("Validating %d splits in %s", len(names), splits_dir)

    all_ok   = True
    n_pass   = 0
    n_fail   = 0
    summary  = []

    for name in names:
        split_dir = splits_dir / name
        ok = validate_one(name, split_dir, strict=not args.no_strict)
        summary.append({"name": name, "ok": ok})
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            all_ok = False

    # Print summary table
    logger.info("\n%s  VALIDATION SUMMARY  %s", "="*20, "="*20)
    for s in summary:
        status = "PASS" if s["ok"] else "FAIL"
        logger.info("  %s  %s", status, s["name"])

    logger.info("\n  %d / %d splits passed", n_pass, len(names))

    if not all_ok:
        logger.error("Some splits have leakage issues.")
        if not args.no_strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
