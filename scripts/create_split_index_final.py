"""
Create split_index.csv for graphs built from scripts/build_graphs_final.py.

This uses the built graph set from outputs/final_graph_ready/graphs and joins
repo/date metadata from commit_info_full.csv so the existing training loop can
reuse repo and temporal splits.

Usage:
  conda run -n thesis python scripts/create_split_index_final.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_GRAPHS = ROOT / "outputs" / "final_graph_ready"
COMMIT_INFO = ROOT / "data" / "graph_data" / "commit_info_full.csv"
OUT = FINAL_GRAPHS / "split_index.csv"


def repo_split(repo_url: str) -> str:
    # val: ~5,967 commits (14.9%), 13.7% VCC
    if "ImageMagick/ImageMagick" in repo_url:
        return "val"
    if "radareorg/radare2" in repo_url:
        return "val"
    if "the-tcpdump-group/tcpdump" in repo_url:
        return "val"
    if "php/php-src" in repo_url:
        return "val"
    if "FreeRDP/FreeRDP" in repo_url:
        return "val"
    # test: ~5,188 commits (12.9%), 14.4% VCC
    if "FFmpeg/FFmpeg" in repo_url:
        return "test"
    if "gpac/gpac" in repo_url:
        return "test"
    if "OISF/suricata" in repo_url:
        return "test"
    if "openssl/openssl" in repo_url:
        return "test"
    if "redis/redis" in repo_url:
        return "test"
    if "envoyproxy/envoy" in repo_url:
        return "test"
    # train: everything else including tensorflow/tensorflow
    return "train"


def main() -> None:
    built_hashes = {p.stem for p in (FINAL_GRAPHS / "graphs").glob("*.pt")}
    if not built_hashes:
        raise SystemExit(f"No built graphs found in {(FINAL_GRAPHS / 'graphs')}")

    ci = pd.read_csv(
        COMMIT_INFO,
        usecols=["hash", "commit_type", "repo_url", "author_date"],
    ).drop_duplicates("hash")
    ci = ci[ci["hash"].isin(built_hashes)].copy()
    ci["author_date"] = pd.to_datetime(ci["author_date"], utc=True, errors="coerce")
    ci["label"] = (ci["commit_type"] == "VCC").astype(int)

    ci["repo_split"] = ci["repo_url"].fillna("").apply(repo_split)

    ci_dated = ci.dropna(subset=["author_date"]).sort_values("author_date")
    n = len(ci_dated)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    temporal = pd.Series("train", index=ci_dated.index)
    temporal.iloc[train_end:val_end] = "val"
    temporal.iloc[val_end:] = "test"
    ci["temporal_split"] = temporal
    ci["temporal_split"] = ci["temporal_split"].fillna("train")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ci[["hash", "commit_type", "label", "repo_url", "author_date", "repo_split", "temporal_split"]].to_csv(OUT, index=False)
    print(f"Saved {len(ci)} rows to {OUT}")


if __name__ == "__main__":
    main()
