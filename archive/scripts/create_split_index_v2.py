"""
scripts/create_split_index_v2.py

Create split_index.csv for graphs built by scripts/build_graphs_v2.py.

Reads: outputs/graph_ready_v2/graphs/*.pt  (or just build_manifest.csv)
       ICVul_pp/graph_ready_sampling_v2/commit_info.csv  (for metadata)
Writes: outputs/graph_ready_v2/split_index.csv

Repo split (same as final_graph_ready for consistency):
  val  repos: ImageMagick, radare2, tcpdump, php-src, FreeRDP
  test repos: FFmpeg, gpac, suricata, openssl, redis, envoy
  train: everything else

Temporal split: chronological 70/15/15 by author_date.

Usage:
  python scripts/create_split_index_v2.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT      = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT.parent / "ICVul_pp" / "graph_ready_sampling_v2"
OUT_ROOT  = ROOT / "outputs" / "graph_ready_v2"
OUT       = OUT_ROOT / "split_index.csv"


def repo_split(repo_url: str) -> str:
    # val repos (~5,108 commits, ~16.1% VCC)
    if "ImageMagick/ImageMagick" in repo_url: return "val"
    if "radareorg/radare2"       in repo_url: return "val"
    if "the-tcpdump-group/tcpdump" in repo_url: return "val"
    if "php/php-src"             in repo_url: return "val"
    if "FreeRDP/FreeRDP"         in repo_url: return "val"
    # test repos (~3,431 commits, ~22.1% VCC)
    if "FFmpeg/FFmpeg"           in repo_url: return "test"
    if "gpac/gpac"               in repo_url: return "test"
    if "OISF/suricata"           in repo_url: return "test"
    if "openssl/openssl"         in repo_url: return "test"
    if "redis/redis"             in repo_url: return "test"
    if "envoyproxy/envoy"        in repo_url: return "test"
    # train: everything else (~25,858 commits, ~16.7% VCC)
    return "train"


def main() -> None:
    manifest_path = OUT_ROOT / "build_manifest.csv"
    graphs_dir    = OUT_ROOT / "graphs"

    # Use build_manifest if available, otherwise scan graphs dir
    if manifest_path.exists():
        built = pd.read_csv(manifest_path, usecols=["hash"])
        built_hashes = set(built["hash"].astype(str))
        print(f"Found {len(built_hashes):,} hashes in build_manifest.csv")
    elif graphs_dir.exists():
        built_hashes = {p.stem for p in graphs_dir.glob("*.pt")}
        print(f"Scanned {len(built_hashes):,} .pt files in {graphs_dir}")
    else:
        raise SystemExit(
            f"No graphs found. Run scripts/build_graphs_v2.py first.\n"
            f"Expected: {graphs_dir}"
        )

    ci = pd.read_csv(
        DATA_ROOT / "commit_info.csv",
        usecols=["hash", "commit_label", "repo_url", "author_date"],
        low_memory=False,
    ).drop_duplicates("hash")

    ci = ci[ci["hash"].astype(str).isin(built_hashes)].copy()
    ci["label"] = (ci["commit_label"] == "VCC").astype(int)
    ci["author_date"] = pd.to_datetime(ci["author_date"], utc=True, errors="coerce")
    ci["repo_split"] = ci["repo_url"].fillna("").apply(repo_split)

    # Temporal split: chronological 70/15/15
    ci_dated = ci.dropna(subset=["author_date"]).sort_values("author_date")
    n = len(ci_dated)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)
    temporal  = pd.Series("train", index=ci_dated.index)
    temporal.iloc[train_end:val_end] = "val"
    temporal.iloc[val_end:] = "test"
    ci["temporal_split"] = temporal
    ci["temporal_split"] = ci["temporal_split"].fillna("train")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ci[["hash", "commit_label", "label", "repo_url", "author_date",
        "repo_split", "temporal_split"]].to_csv(OUT, index=False)

    # Report
    print(f"\nSaved {len(ci):,} rows -> {OUT}")
    for split_col in ["repo_split", "temporal_split"]:
        print(f"\n{split_col}:")
        grp = ci.groupby([split_col, "commit_label"]).size().unstack(fill_value=0)
        grp["total"] = grp.sum(axis=1)
        grp["vcc_pct"] = (grp.get("VCC", 0) / grp["total"] * 100).round(1)
        print(grp.to_string())


if __name__ == "__main__":
    main()
