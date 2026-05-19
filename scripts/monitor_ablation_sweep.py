"""
scripts/monitor_ablation_sweep.py

Cron-job monitor for the ablation sweep.
Appends a timestamped progress update to outputs/ablation_sweep_v1/progress.log.

Usage: python scripts/monitor_ablation_sweep.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTDIR = ROOT / "outputs" / "ablation_sweep_v1"
LOG = OUTDIR / "progress.log"

RUNS = [
    ("abl_full",         "Full model"),
    ("abl_no_code_emb",  "No function BERT"),
    ("abl_no_text_emb",  "No hunk BERT"),
    ("abl_no_all_bert",  "No fn+hunk BERT"),
    ("abl_no_sdlc",      "No SDLC"),
    ("abl_no_developer", "No developer"),
    ("abl_code_only",    "Code-only"),
    ("abl_context_only", "Context-only"),
]


def load_metrics(run_name: str) -> list[dict]:
    p = OUTDIR / "checkpoints" / run_name / "metrics.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p)))
    return rows


def summarize_run(run_name: str) -> str:
    rows = load_metrics(run_name)
    if not rows:
        return "not started"
    val_rows = [r for r in rows if r["split"] == "val"]
    last_ep = max(int(r["epoch"]) for r in rows)
    test_p = OUTDIR / "checkpoints" / run_name / "test_results.json"
    if test_p.exists():
        tr = json.loads(test_p.read_text())
        return f"DONE  ep={last_ep}  test_F1*={tr.get('f1_best','?')}  AUC-PR={tr.get('auc_pr','?')}"
    if val_rows:
        best = max(val_rows, key=lambda r: float(r["f1_best"]))
        return f"ep={last_ep}/9  best_val_F1*={float(best['f1_best']):.4f} @ep{best['epoch']}"
    return f"ep={last_ep}/9  (train only)"


def main():
    lines = [
        "",
        f"=== Monitor update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===",
    ]

    done = 0
    best_overall = ("", 0.0)
    for run_name, desc in RUNS:
        summary = summarize_run(run_name)
        if summary.startswith("DONE"):
            done += 1
            # Extract test F1_best
            test_p = OUTDIR / "checkpoints" / run_name / "test_results.json"
            if test_p.exists():
                tr = json.loads(test_p.read_text())
                f1 = float(tr.get("f1_best", 0.0))
                if f1 > best_overall[1]:
                    best_overall = (run_name, f1)
        lines.append(f"  [{done if summary.startswith('DONE') else '?'}/8] {desc:30s}: {summary}")

    lines.append(f"  Completed: {done}/8 runs")
    if best_overall[0]:
        lines.append(f"  Best so far: {best_overall[0]}  test_F1*={best_overall[1]:.4f}")

    # Check if in-progress run exists
    in_progress = []
    for run_name, desc in RUNS:
        rows = load_metrics(run_name)
        test_p = OUTDIR / "checkpoints" / run_name / "test_results.json"
        if rows and not test_p.exists():
            last_ep = max(int(r["epoch"]) for r in rows)
            in_progress.append(f"{desc} (ep {last_ep}/9)")
    if in_progress:
        lines.append(f"  In progress: {', '.join(in_progress)}")

    text = "\n".join(lines)
    print(text)
    with open(LOG, "a") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
