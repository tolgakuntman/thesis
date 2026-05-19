"""
scripts/create_split_index.py

Creates data_new/graph_ready/split_index.csv — a lookup table for train/val/test
assignment for every built graph, supporting two independent split strategies.

Columns:
  hash          : commit hash
  commit_type   : VCC / FC / normal
  label         : 1 (VCC) or 0 (FC/normal)
  repo_url      : GitHub repo URL
  author_date   : commit author timestamp (UTC)
  repo_split    : train / val / test  (based on repo)
  temporal_split: train / val / test  (based on author_date percentiles)

Repo split assignment:
  val  → ImageMagick/ImageMagick, gpac/gpac, FreeRDP/FreeRDP
  test → openssl/openssl, FFmpeg/FFmpeg, krb5/krb5
  train→ everything else (TF + remaining repos)

Temporal split assignment:
  Sorted by author_date across ALL buildable commits.
  train → earliest 70%
  val   → next 15%
  test  → latest 15%

Usage:
  conda activate thesis
  python scripts/create_split_index.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[1]
GR   = ROOT / "data_new" / "graph_ready"
GD   = ROOT / "data" / "graph_data"

OUT  = GR / "split_index.csv"


def repo_split(repo_url: str) -> str:
    # val: ImageMagick, GPAC, FreeRDP  (image/media/remote-desktop — domain-diverse from TF)
    if "ImageMagick/ImageMagick" in repo_url:
        return "val"
    if "gpac/gpac" in repo_url:
        return "val"
    if "FreeRDP/FreeRDP" in repo_url:
        return "val"
    # test: OpenSSL, FFmpeg, krb5  (crypto/multimedia/auth — security-focused cross-project)
    if "openssl/openssl" in repo_url:
        return "test"
    if "FFmpeg/FFmpeg" in repo_url:
        return "test"
    if "krb5/krb5" in repo_url:
        return "test"
    return "train"


def main():
    # ── load commit metadata ───────────────────────────────────────────────
    ci = pd.read_csv(
        GD / "commit_info_full.csv",
        usecols=["hash", "commit_type", "repo_url", "author_date"],
    )
    ci = ci.drop_duplicates(subset=["hash"])
    ci["author_date"] = pd.to_datetime(ci["author_date"], utc=True, errors="coerce")
    ci["label"] = (ci["commit_type"] == "VCC").astype(int)

    # ── restrict to built graphs only ─────────────────────────────────────
    built_hashes = {p.stem for p in (GR / "graphs").glob("*.pt")}
    print(f"Built graphs on disk: {len(built_hashes):,}")
    ci = ci[ci["hash"].isin(built_hashes)].copy()
    print(f"Matched in commit_info_full: {len(ci):,}")

    # ── repo split ─────────────────────────────────────────────────────────
    ci["repo_split"] = ci["repo_url"].apply(repo_split)
    print("\nRepo split:")
    print(ci.groupby("repo_split")[["label"]].agg(
        n=("label", "count"),
        n_vcc=("label", "sum"),
    ).assign(pct_vcc=lambda d: (d["n_vcc"] / d["n"]).round(3)).to_string())

    # ── temporal split (global, by author_date percentile) ─────────────────
    ci_dated = ci.dropna(subset=["author_date"]).sort_values("author_date")
    n = len(ci_dated)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    temporal = pd.Series("train", index=ci_dated.index)
    temporal.iloc[train_end:val_end] = "val"
    temporal.iloc[val_end:]          = "test"
    ci["temporal_split"] = temporal
    ci["temporal_split"] = ci["temporal_split"].fillna("train")  # undated → train

    print("\nTemporal split:")
    print(ci.groupby("temporal_split")[["label"]].agg(
        n=("label", "count"),
        n_vcc=("label", "sum"),
    ).assign(pct_vcc=lambda d: (d["n_vcc"] / d["n"]).round(3)).to_string())

    cutoff_val  = ci_dated.iloc[train_end]["author_date"].date()
    cutoff_test = ci_dated.iloc[val_end]["author_date"].date()
    print(f"\n  Train: up to {cutoff_val}")
    print(f"  Val:   {cutoff_val} → {cutoff_test}")
    print(f"  Test:  {cutoff_test} onwards")

    # ── save ───────────────────────────────────────────────────────────────
    out_cols = ["hash", "commit_type", "label", "repo_url", "author_date", "repo_split", "temporal_split"]
    ci[out_cols].to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}  ({len(ci):,} rows)")


if __name__ == "__main__":
    main()
