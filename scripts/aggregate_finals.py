"""
scripts/aggregate_finals.py

Aggregate Phase 6 multi-seed final results into a thesis-ready summary table.

Reads all checkpoints/final_*/seed_*/test_results.json (or test_results.json)
and groups by base run name, reporting mean ± std across seeds.

Usage:
    python scripts/aggregate_finals.py
    python scripts/aggregate_finals.py --checkpoints_dir /path/to/checkpoints
    python scripts/aggregate_finals.py --prefix final_   # only 'final_*' runs
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]

METRIC_KEYS = ["auc_pr", "auc_roc", "f1_best", "f1", "mcc", "precision", "recall",
               "thresh_best", "best_val_auc_pr"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_dir", default=str(ROOT / "checkpoints"),
                   help="Root checkpoints directory")
    p.add_argument("--prefix", default="final_",
                   help="Only include run dirs starting with this prefix")
    p.add_argument("--output", default=None,
                   help="Write JSON summary to this path (default: print only)")
    return p.parse_args()


def strip_seed_suffix(name: str) -> str:
    """Remove trailing _s42 / _s123 / _s7 etc. to get the base run name."""
    return re.sub(r"_s\d+$", "", name)


def load_all(ckpt_dir: Path, prefix: str) -> dict[str, list[dict]]:
    """Load test_results.json for all matching runs, grouped by base name."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.name.startswith(prefix):
            continue
        f = run_dir / "seed_42" / "test_results.json"  # multi-seed layout
        if not f.exists():
            # Try seeded subdirs  e.g. seed_123/
            for sd in sorted(run_dir.iterdir()):
                tf = sd / "test_results.json"
                if tf.exists():
                    r = json.loads(tf.read_text())
                    base = strip_seed_suffix(run_dir.name)
                    groups[base].append(r)
            # Also try flat layout
            f = run_dir / "test_results.json"
            if f.exists():
                r = json.loads(f.read_text())
                base = strip_seed_suffix(run_dir.name)
                groups[base].append(r)
        else:
            r = json.loads(f.read_text())
            base = strip_seed_suffix(run_dir.name)
            groups[base].append(r)
    return dict(groups)


def aggregate(results: list[dict]) -> dict:
    agg = {
        "n_seeds": len(results),
        "seeds": [r.get("seed", "?") for r in results],
        "split_type": results[0].get("split_type", "?"),
    }
    for k in METRIC_KEYS:
        vals = [float(r[k]) for r in results if k in r]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
    return agg


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoints_dir)
    groups = load_all(ckpt_dir, args.prefix)

    if not groups:
        print(f"No runs found matching prefix '{args.prefix}' in {ckpt_dir}")
        sys.exit(1)

    summary = {}
    print(f"\n{'='*100}")
    print(f"PHASE 6 MULTI-SEED FINAL RESULTS  (prefix='{args.prefix}')")
    print(f"{'='*100}")
    print(f"{'Run':<30} {'N':>3} {'Split':>8}  "
          f"{'AUC-PR':>13}  {'F1_best':>13}  {'MCC':>13}  {'AUC-ROC':>13}")
    print("-"*100)

    for base in sorted(groups):
        results = groups[base]
        agg = aggregate(results)
        summary[base] = agg

        def fmt(k):
            m = agg.get(f"{k}_mean")
            s = agg.get(f"{k}_std")
            if m is None: return "     N/A    "
            return f"{m:.4f}±{s:.4f}"

        print(f"{base:<30} {agg['n_seeds']:>3} {agg['split_type']:>8}  "
              f"{fmt('auc_pr'):>13}  {fmt('f1_best'):>13}  {fmt('mcc'):>13}  {fmt('auc_roc'):>13}")

    print()
    print("Key metrics (AUC-PR mean ± std):")
    for base, agg in sorted(summary.items(), key=lambda x: -(x[1].get("auc_pr_mean") or 0)):
        m = agg.get("auc_pr_mean", 0)
        s = agg.get("auc_pr_std", 0)
        split = agg.get("split_type", "?")
        n = agg["n_seeds"]
        print(f"  {base:<35} {m:.4f} ± {s:.4f}  [{split}, n={n}]")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary -> {out_path}")


if __name__ == "__main__":
    main()
