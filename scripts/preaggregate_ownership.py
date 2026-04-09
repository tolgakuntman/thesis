"""
scripts/preaggregate_ownership.py

Pre-aggregates ownership_features_normalized.csv (2.19M rows, 3 window sizes)
into a per-(commit_hash, file_path) stats table using the 90-day window.

Output: data_new/graph_ready/ownership_stats_90d.csv

Columns in output:
  commit_hash       : str
  file_path         : str
  n_owners          : int   — number of distinct developers with ownership in window
  max_own_ratio     : float — max ownership_ratio across all owners (0–1)
  hhi               : float — Herfindahl-Hirschman Index = sum(ratio^2), measures concentration
  total_lines_norm  : float — normalized total lines in file (RobustScaler, from first row)
  total_devs_norm   : float — normalized total devs (RobustScaler, from first row)

Usage:
  conda activate thesis
  python scripts/preaggregate_ownership.py
"""

import pandas as pd
import numpy as np
import os
import time

BASE  = os.path.join(os.path.dirname(__file__), "..", "data_new", "graph_ready")
OUT   = os.path.join(BASE, "ownership_stats_90d.csv")
SRC   = os.path.join(BASE, "ownership_features_normalized.csv")

USECOLS = [
    "commit_hash", "file_path", "dev_id", "window_days",
    "ownership_ratio", "total_lines", "total_devs",
]


def main():
    t0 = time.time()
    print(f"Loading {SRC} ...")
    own = pd.read_csv(SRC, usecols=USECOLS, low_memory=False)
    print(f"  {len(own):,} rows loaded in {time.time()-t0:.1f}s")

    # Filter to 90-day window only
    own90 = own[own["window_days"] == 90].copy()
    print(f"  {len(own90):,} rows after window_days==90 filter")

    # Deduplicate: same (commit_hash, file_path, dev_id) can appear multiple times
    # (e.g. renamed files tracked under both old and new path). Keep max ownership_ratio.
    before = len(own90)
    own90 = own90.sort_values("ownership_ratio", ascending=False).drop_duplicates(
        subset=["commit_hash", "file_path", "dev_id"]
    )
    print(f"  {before - len(own90):,} duplicate (commit,file,dev) rows removed")

    t1 = time.time()
    print("Aggregating per (commit_hash, file_path) ...")

    # Per-group stats
    grp = own90.groupby(["commit_hash", "file_path"], sort=False)

    # n_owners: count of distinct dev_ids
    n_owners = grp["dev_id"].nunique().rename("n_owners")

    # max_ownership_ratio
    max_own = grp["ownership_ratio"].max().rename("max_own_ratio")

    # HHI = sum of squared ownership ratios (concentration measure, 0–1)
    hhi = grp["ownership_ratio"].apply(lambda x: (x**2).sum()).rename("hhi")

    # total_lines and total_devs are file-level (same for all dev rows in a file)
    # Take first row's value
    file_level = grp[["total_lines", "total_devs"]].first()
    file_level.columns = ["total_lines_norm", "total_devs_norm"]

    stats = pd.concat([n_owners, max_own, hhi, file_level], axis=1).reset_index()
    print(f"  {len(stats):,} (commit, file) pairs in {time.time()-t1:.1f}s")

    print(f"\nStats summary:")
    print(stats[["n_owners","max_own_ratio","hhi","total_lines_norm","total_devs_norm"]].describe().round(3).to_string())

    stats.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
