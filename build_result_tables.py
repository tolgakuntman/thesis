"""
Build 6 result tables from GNN experiment checkpoints.
"""

import os
import json
import pandas as pd
import numpy as np

BASE = "C:/Users/User/OneDrive/thesis/thesis/checkpoints"


def load_run_metrics(run_dir):
    """
    Load metrics from a run directory.
    Returns (train_metrics, val_metrics) at the epoch with best val auc_pr.
    The run_dir may contain a seed_*/metrics.csv or just metrics.csv directly.
    Returns None if not found.
    """
    # Look for metrics.csv: first check seed_* subdirs, then direct
    metrics_path = None
    seed_dirs = [d for d in os.listdir(run_dir) if d.startswith("seed_") and os.path.isdir(os.path.join(run_dir, d))]
    if seed_dirs:
        # Use first seed dir (for single-seed runs)
        seed_dir = sorted(seed_dirs)[0]
        candidate = os.path.join(run_dir, seed_dir, "metrics.csv")
        if os.path.exists(candidate):
            metrics_path = candidate
    else:
        candidate = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(candidate):
            metrics_path = candidate

    if metrics_path is None:
        return None, None

    df = pd.read_csv(metrics_path)
    train_rows = df[df["split"] == "train"].reset_index(drop=True)
    val_rows = df[df["split"] == "val"].reset_index(drop=True)

    if val_rows.empty:
        return None, None

    best_idx = val_rows["auc_pr"].idxmax()
    val_best = val_rows.iloc[best_idx]
    # Train row at same epoch
    best_epoch = int(val_best["epoch"])
    train_at_epoch = train_rows[train_rows["epoch"] == best_epoch]
    if train_at_epoch.empty:
        train_best = train_rows.iloc[best_idx] if best_idx < len(train_rows) else train_rows.iloc[-1]
    else:
        train_best = train_at_epoch.iloc[0]

    return train_best, val_best


def load_test_results(run_dir):
    """Load test_results.json from run_dir (top-level) or seed_*/test_results.json."""
    # Try top-level first
    top = os.path.join(run_dir, "test_results.json")
    if os.path.exists(top):
        with open(top) as f:
            return json.load(f)
    # Try seed dirs
    seed_dirs = [d for d in os.listdir(run_dir) if d.startswith("seed_") and os.path.isdir(os.path.join(run_dir, d))]
    for sd in sorted(seed_dirs):
        candidate = os.path.join(run_dir, sd, "test_results.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                return json.load(f)
    return None


def extract_test_metrics(test_data):
    """Extract f1_best, auc_pr, mcc from test_results dict."""
    if test_data is None:
        return None, None, None
    f1 = test_data.get("f1_best", test_data.get("test_f1_best", None))
    auc = test_data.get("auc_pr", test_data.get("test_auc_pr", None))
    mcc = test_data.get("mcc", test_data.get("test_mcc", None))
    return f1, auc, mcc


def get_run_row(run_name, run_dir):
    """Build a single result row dict for a run."""
    train, val = load_run_metrics(run_dir)
    test_data = load_test_results(run_dir)
    test_f1, test_auc, test_mcc = extract_test_metrics(test_data)

    row = {"Run": run_name}
    if train is not None:
        row["Train F1"] = round(float(train["f1_best"]), 4)
        row["Train AUC"] = round(float(train["auc_pr"]), 4)
        row["Train MCC"] = round(float(train["mcc"]), 4)
    else:
        row["Train F1"] = row["Train AUC"] = row["Train MCC"] = None

    if val is not None:
        row["Val F1"] = round(float(val["f1_best"]), 4)
        row["Val AUC"] = round(float(val["auc_pr"]), 4)
        row["Val MCC"] = round(float(val["mcc"]), 4)
    else:
        row["Val F1"] = row["Val AUC"] = row["Val MCC"] = None

    row["Test F1"] = round(test_f1, 4) if test_f1 is not None else None
    row["Test AUC"] = round(test_auc, 4) if test_auc is not None else None
    row["Test MCC"] = round(test_mcc, 4) if test_mcc is not None else None

    return row


def get_multiseed_row(run_label, run_dirs):
    """
    Average metrics across multiple seed runs.
    run_dirs: list of (run_dir,) paths.
    Reports mean (and std if >1 seed).
    """
    rows = [get_run_row(run_label, rd) for rd in run_dirs]
    # Average numeric columns
    numeric_cols = ["Train F1", "Train AUC", "Train MCC", "Val F1", "Val AUC", "Val MCC",
                    "Test F1", "Test AUC", "Test MCC"]

    result = {"Run": run_label}
    for col in numeric_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            if len(vals) > 1:
                result[col] = f"{mean:.4f} ±{std:.4f}"
            else:
                result[col] = f"{mean:.4f}"
        else:
            result[col] = None
    return result


def df_to_markdown(df):
    """Convert dataframe to markdown table."""
    return df.to_markdown(index=False)


# ============================================================
# TABLE 1 — Phase 1 Big Ablations
# ============================================================
print("=" * 70)
print("TABLE 1 — Big Ablations (Phase 1)")
print("=" * 70)

phase1_dir = os.path.join(BASE, "phase1_ablations")
phase1_runs = sorted([d for d in os.listdir(phase1_dir)
                       if os.path.isdir(os.path.join(phase1_dir, d)) and not d.startswith("z")])

rows1 = []
for run in phase1_runs:
    run_dir = os.path.join(phase1_dir, run)
    rows1.append(get_run_row(run, run_dir))

df1 = pd.DataFrame(rows1)
# Sort by val AUC descending — extract numeric for sorting
def sort_key(x):
    if x is None:
        return -1
    if isinstance(x, str):
        return float(x.split(" ")[0])
    return float(x)

df1["_sort"] = df1["Val AUC"].apply(sort_key)
df1 = df1.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print(df_to_markdown(df1))
print()


# ============================================================
# TABLE 2 — Phase 2 Small Ablations
# ============================================================
print("=" * 70)
print("TABLE 2 — Small Ablations (Phase 2)")
print("=" * 70)

phase2_dir = os.path.join(BASE, "phase2")
phase2_runs = sorted([d for d in os.listdir(phase2_dir)
                       if os.path.isdir(os.path.join(phase2_dir, d))])

rows2 = []
for run in phase2_runs:
    run_dir = os.path.join(phase2_dir, run)
    rows2.append(get_run_row(run, run_dir))

df2 = pd.DataFrame(rows2)
df2["_sort"] = df2["Val AUC"].apply(sort_key)
df2 = df2.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print(df_to_markdown(df2))
print()


# ============================================================
# TABLE 3 — Temporal vs Repo Split
# ============================================================
print("=" * 70)
print("TABLE 3 — Temporal vs Repo Split")
print("=" * 70)

phase6_dir = os.path.join(BASE, "phase6")
phase34_dir = os.path.join(BASE, "phase3-4")

rows3 = []

# Phase 6 final_full_repo (3 seeds)
repo_seed_dirs = [os.path.join(phase6_dir, f"final_full_repo_s{s}") for s in [42, 123, 7]
                  if os.path.isdir(os.path.join(phase6_dir, f"final_full_repo_s{s}"))]
if repo_seed_dirs:
    rows3.append(get_multiseed_row("final_full_repo (Phase 6, mean 3 seeds)", repo_seed_dirs))

# Phase 6 final_full_temp (3 seeds)
temp_seed_dirs = [os.path.join(phase6_dir, f"final_full_temp_s{s}") for s in [42, 123, 7]
                  if os.path.isdir(os.path.join(phase6_dir, f"final_full_temp_s{s}"))]
if temp_seed_dirs:
    rows3.append(get_multiseed_row("final_full_temp (Phase 6, mean 3 seeds)", temp_seed_dirs))

# Phase 3-4 split comparison runs
p34_relevant = ["repo_combined_best", "temp_combined_best", "temp_full"]
for run in p34_relevant:
    run_dir = os.path.join(phase34_dir, run)
    if os.path.isdir(run_dir):
        rows3.append(get_run_row(run + " (Phase 3-4)", run_dir))

# Also add any other phase3-4 runs that look like split comparisons
for run in sorted(os.listdir(phase34_dir)):
    if run in p34_relevant:
        continue
    run_dir = os.path.join(phase34_dir, run)
    if os.path.isdir(run_dir):
        rows3.append(get_run_row(run + " (Phase 3-4)", run_dir))

df3 = pd.DataFrame(rows3)
df3["_sort"] = df3["Val AUC"].apply(sort_key)
df3 = df3.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print(df_to_markdown(df3))
print()


# ============================================================
# TABLE 4 — Architecture Comparison
# ============================================================
print("=" * 70)
print("TABLE 4 — Architecture Comparison (MLP vs RGCN vs HeteroSAGE)")
print("=" * 70)

rows4 = []

# MLP baseline from Phase 1
mlp_dir = os.path.join(phase1_dir, "struct_mlp_baseline")
if os.path.isdir(mlp_dir):
    rows4.append(get_run_row("MLP Baseline (Phase 1)", mlp_dir))

# HeteroSAGE: final_full_repo Phase 6 (3 seeds)
if repo_seed_dirs:
    rows4.append(get_multiseed_row("HeteroSAGE full (Phase 6, mean 3 seeds)", repo_seed_dirs))

# RGCN: final_rgcn_repo Phase 6 (3 seeds)
rgcn_seed_dirs = [os.path.join(phase6_dir, f"final_rgcn_repo_s{s}") for s in [42, 123, 7]
                  if os.path.isdir(os.path.join(phase6_dir, f"final_rgcn_repo_s{s}"))]
if rgcn_seed_dirs:
    rows4.append(get_multiseed_row("RGCN (Phase 6, mean 3 seeds)", rgcn_seed_dirs))

# Phase 3-4 RGCN runs
for run in sorted(os.listdir(phase34_dir)):
    if "rgcn" in run.lower() or "mlp" in run.lower():
        run_dir = os.path.join(phase34_dir, run)
        if os.path.isdir(run_dir):
            rows4.append(get_run_row(run + " (Phase 3-4)", run_dir))

df4 = pd.DataFrame(rows4)
df4["_sort"] = df4["Val AUC"].apply(sort_key)
df4 = df4.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print(df_to_markdown(df4))
print()


# ============================================================
# TABLE 5 — Change Representation Ablations
# ============================================================
print("=" * 70)
print("TABLE 5 — Change Representation Ablations (Phase 5.1)")
print("=" * 70)

phase51_dir = os.path.join(BASE, "phase_5_1")

rows5 = []

# Baseline: struct_no_hunk from Phase 1
nohunk_dir = os.path.join(phase1_dir, "struct_no_hunk")
if os.path.isdir(nohunk_dir):
    rows5.append(get_run_row("struct_no_hunk / baseline (Phase 1)", nohunk_dir))

# abl_chg_* runs from phase_5_1
chg_runs = ["abl_chg_fn_cat", "abl_chg_file_met", "abl_chg_commit_dmm", "abl_chg_all", "abl_chg_no_file"]
for run in chg_runs:
    run_dir = os.path.join(phase51_dir, run)
    if os.path.isdir(run_dir):
        rows5.append(get_run_row(run + " (Phase 5.1)", run_dir))
    else:
        print(f"  [WARNING] Not found: {run_dir}")

df5 = pd.DataFrame(rows5)
df5["_sort"] = df5["Val AUC"].apply(sort_key)
df5 = df5.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print(df_to_markdown(df5))
print()


# ============================================================
# TABLE 6 — Code Before/After/Delta Ablations
# ============================================================
print("=" * 70)
print("TABLE 6 — Code Before/After/Delta Ablations")
print("=" * 70)

rows6 = []

# Baseline: phase1_full_baseline
full_baseline_dir = os.path.join(phase1_dir, "phase1_full_baseline")
if os.path.isdir(full_baseline_dir):
    rows6.append(get_run_row("phase1_full_baseline / code_after_only (Phase 1)", full_baseline_dir))

# phase_5_1 code ablations
code_runs_51 = ["abl_code_with_before", "abl_code_before_only", "abl_delta_only",
                "abl_code_no_delta", "abl_pure_structural"]
for run in code_runs_51:
    run_dir = os.path.join(phase51_dir, run)
    if os.path.isdir(run_dir):
        rows6.append(get_run_row(run + " (Phase 5.1)", run_dir))
    else:
        print(f"  [WARNING] Not found: {run_dir}")

# Phase 6 reruns of failed Phase 5.1 jobs
for run in ["abl_code_with_before", "abl_code_before_only"]:
    run_dir = os.path.join(phase6_dir, run)
    if os.path.isdir(run_dir):
        rows6.append(get_run_row(run + " (Phase 6 rerun)", run_dir))

df6 = pd.DataFrame(rows6)
# Drop rows where all metrics are None/NaN (failed runs with no metrics)
metric_cols = ["Train F1", "Train AUC", "Train MCC", "Val F1", "Val AUC", "Val MCC",
               "Test F1", "Test AUC", "Test MCC"]
df6 = df6[df6["Val AUC"].notna() & (df6["Val AUC"] != "None")].copy()
df6["_sort"] = df6["Val AUC"].apply(sort_key)
df6 = df6.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
print(df_to_markdown(df6))
print()

print("Done.")
