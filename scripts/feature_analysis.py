#!/usr/bin/env python3
"""
feature_analysis.py — Reverse SHAP ranking + correlation analysis for feature ablation.

Usage:
    python scripts/feature_analysis.py

Outputs (all in outputs/feature_analysis/):
  shap_ranking.png           — horizontal bar chart, all scalar features worst→best
  corr_function.png          — Spearman heatmap for fn numeric features [0-4]
  corr_developer.png         — Spearman heatmap for dev features [0-8]
  corr_commit.png            — Spearman heatmap for commit numeric features
  corr_sdlc.png              — Spearman heatmap for issue/pr/tag features
  removal_candidates.csv     — combined score table for all features
  analysis_report.txt        — text summary with top-10 removal candidates
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parents[1]
DATA_ROOT  = ROOT.parent / "ICVul_pp" / "graph_ready_sampling_v2"
FEAT_DIR   = DATA_ROOT / "features"
SHAP_DIR   = ROOT / "outputs" / "shap_analysis"
OUT_DIR    = ROOT / "outputs" / "feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature column definitions (must match build_graphs_v2.py) ─────────────────

# Correct developer feature names (shap_analysis.py has mislabeled dims 2-8!)
DEV_FEAT_COLS  = [
    "repo_total_commits_before",       # dim 0 — labeled "total_commits"
    "repo_active_weeks_before",         # dim 1 — labeled "active_weeks"
    "repo_tenure_days",                 # dim 2 — MISLABELED as "commits_as_committer" in SHAP
    "repo_commits_as_committer_before", # dim 3 — MISLABELED as "total_issues" in SHAP
    "recent_commits_90d",               # dim 4 — MISLABELED as "total_prs" in SHAP
    "time_since_last_commit_days",      # dim 5
    "experience_percentile_in_repo",    # dim 6
    "cross_repo_commits_before",        # dim 7 — MISLABELED as "num_repos" in SHAP
    "num_repos_contributed_before",     # dim 8 — MISLABELED as "repo_commits_before" in SHAP
]
DEV_DISPLAY_NAMES = [
    "dev: total_commits",
    "dev: active_weeks",
    "dev: tenure_days",           # correct name (was mislabeled)
    "dev: commits_as_committer",  # correct name (was mislabeled)
    "dev: recent_commits_90d",    # correct name (was mislabeled)
    "dev: time_since_last_commit",
    "dev: experience_pct_repo",
    "dev: cross_repo_commits",    # correct name (was mislabeled)
    "dev: num_repos",             # correct name (was mislabeled)
]
# Shap CSV still uses old labels — map old→correct for merging
_SHAP_DEV_REMAP = {
    "dev: commits_as_committer":   "dev: tenure_days",
    "dev: total_issues":           "dev: commits_as_committer",
    "dev: total_prs":              "dev: recent_commits_90d",
    "dev: num_repos":              "dev: cross_repo_commits",
    "dev: repo_commits_before":    "dev: num_repos",
}

FN_FEAT_COLS  = ["num_lines_of_code", "complexity", "token_count", "length", "top_nesting_level"]
FN_DISPLAY    = ["fn: loc", "fn: complexity", "fn: token_count", "fn: length", "fn: top_nesting"]

FILE_FEAT_COLS = ["num_lines_added", "num_lines_deleted", "complexity"]
FILE_DISPLAY   = ["file: lines_added", "file: lines_deleted", "file: complexity"]

COMMIT_FEAT_NAMES = [
    "commit: in_main_branch", "commit: merge",
    "commit: dmm_size", "commit: dmm_complexity", "commit: dmm_interfacing",
    "commit: tz_author", "commit: tz_commit",
    "commit: hour_sin", "commit: hour_cos", "commit: dow_sin", "commit: dow_cos",
    "commit: has_sdlc_data", "commit: repo_commits_90d", "commit: repo_active_authors_90d",
]

ISSUE_COLS   = ["issue_open_90d", "issue_age_median", "issues_closed_last_90d", "issue_open_velocity_90d"]
PR_COLS      = ["pr_count_90d", "pr_age_median", "pr_closed_last_90d", "pr_open_velocity_90d"]
TAG_COLS     = ["tags_last_365d", "avg_release_cadence_days", "days_since_prev_tag", "days_since_prev_tag_norm"]

ISSUE_DISPLAY = ["issue: open_90d", "issue: age_median", "issue: closed_90d", "issue: open_velocity_90d"]
PR_DISPLAY    = ["pr: count_90d", "pr: age_median", "pr: closed_90d", "pr: open_velocity_90d"]
TAG_DISPLAY   = ["tag: tags_365d", "tag: cadence_days", "tag: days_since_prev", "tag: days_since_norm"]

# Already ablated (zeroed) — keep in table for context but flag them
ALREADY_ABLATED = {
    "hunk: complexity", "hunk: token_count", "commit: merge",
    "fn: cx_before (z)", "fn: tok_before (z)", "fn: loc_before (z)",
}


# ── 1. Load SHAP summary ───────────────────────────────────────────────────────

def load_shap_summary() -> pd.DataFrame:
    path = SHAP_DIR / "perfeature_shap_summary.csv"
    df = pd.read_csv(path)
    # Apply developer name corrections
    df["feature"] = df["feature"].replace(_SHAP_DEV_REMAP)
    # Flag already-ablated
    df["ablated"] = df["feature"].isin(ALREADY_ABLATED)
    return df.sort_values("mean_abs_phi", ascending=False).reset_index(drop=True)


# ── 2. Correlation helpers ─────────────────────────────────────────────────────

def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sub = df[cols].dropna()
    n = len(cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(sub.iloc[:, i], sub.iloc[:, j])
            mat[i, j] = mat[j, i] = r
    return pd.DataFrame(mat, index=cols, columns=cols)


def plot_corr_heatmap(corr: pd.DataFrame, labels: list[str], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    mask = np.triu(np.ones_like(corr.values, dtype=bool))
    labeled = corr.copy()
    labeled.index = labels
    labeled.columns = labels
    sns.heatmap(
        labeled,
        annot=True, fmt=".2f",
        cmap="RdBu_r", vmin=-1, vmax=1,
        mask=mask,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def max_abs_partner_corr(corr_df: pd.DataFrame, feat_name: str) -> tuple[float, str]:
    """Return (max |r|, partner_name) for a feature in a corr matrix."""
    if feat_name not in corr_df.columns:
        return 0.0, ""
    row = corr_df[feat_name].drop(feat_name).abs()
    best = row.idxmax()
    return float(row[best]), best


# ── 3. SHAP bar chart ─────────────────────────────────────────────────────────

GROUP_COLORS = {
    "file_feats":    "#2196F3",
    "developer":     "#4CAF50",
    "issue":         "#FF9800",
    "fn_metrics":    "#9C27B0",
    "pull_request":  "#FF5722",
    "release_tag":   "#795548",
    "commit":        "#607D8B",
    "hunk_metrics":  "#F44336",
}

def plot_shap_bar(shap_df: pd.DataFrame, path: Path):
    df = shap_df[~shap_df["feature"].str.contains(r"\(z\)")].copy()
    df = df.sort_values("mean_abs_phi", ascending=True)  # worst at top

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.32 + 1))

    colors = [GROUP_COLORS.get(g, "#999") for g in df["group"]]
    alpha  = [0.35 if a else 1.0 for a in df["ablated"]]

    bars = ax.barh(df["feature"], df["mean_abs_phi"], color=colors)
    for bar, a, ab in zip(bars, alpha, df["ablated"]):
        bar.set_alpha(a)
        if ab:
            bar.set_hatch("////")

    ax.set_xlabel("Mean |SHAP| (φ)", fontsize=10)
    ax.set_title("Per-feature SHAP importance — all scalar features\n"
                 "(hatched = already ablated; sorted worst → best)", fontsize=10)

    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, ncol=2)

    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. Per-group correlation ───────────────────────────────────────────────────

def compute_all_correlations() -> dict[str, pd.DataFrame]:
    corrs = {}

    # ── Function numeric features (dims 0-4, skip zeroed before-features) ──
    fn_df = pd.read_csv(FEAT_DIR / "function_numeric_features_normalized.csv",
                        usecols=["hash"] + FN_FEAT_COLS)
    corrs["function"] = spearman_matrix(fn_df, FN_FEAT_COLS)
    plot_corr_heatmap(corrs["function"], FN_DISPLAY,
                      "Function numeric features — Spearman correlation",
                      OUT_DIR / "corr_function.png")

    # ── File features ──
    file_df = pd.read_csv(FEAT_DIR / "file_numeric_features_normalized.csv",
                          usecols=["hash"] + FILE_FEAT_COLS)
    corrs["file"] = spearman_matrix(file_df, FILE_FEAT_COLS)
    plot_corr_heatmap(corrs["file"], FILE_DISPLAY,
                      "File features — Spearman correlation",
                      OUT_DIR / "corr_file.png")

    # ── Developer features ──
    dev_df = pd.read_csv(DATA_ROOT / "developer_info.csv",
                         usecols=["dev_email"] + DEV_FEAT_COLS)
    dev_df = dev_df.dropna(subset=["dev_email"]).drop_duplicates(subset=["dev_email"])
    for c in DEV_FEAT_COLS:
        dev_df[c] = pd.to_numeric(dev_df[c], errors="coerce").fillna(0)
    corrs["developer"] = spearman_matrix(dev_df, DEV_FEAT_COLS)
    corrs["developer"].index   = DEV_DISPLAY_NAMES
    corrs["developer"].columns = DEV_DISPLAY_NAMES
    plot_corr_heatmap(corrs["developer"], DEV_DISPLAY_NAMES,
                      "Developer features — Spearman correlation",
                      OUT_DIR / "corr_developer.png")

    # ── SDLC + commit numeric features from normalized commit CSV ──
    commit_df = pd.read_csv(FEAT_DIR / "final_commit_features_normalized_final.csv")

    sdlc_cols = ISSUE_COLS + PR_COLS + TAG_COLS
    sdlc_display = ISSUE_DISPLAY + PR_DISPLAY + TAG_DISPLAY
    sdlc_sub = commit_df[[c for c in sdlc_cols if c in commit_df.columns]].dropna()
    sdlc_actual_cols    = [c for c in sdlc_cols if c in sdlc_sub.columns]
    sdlc_actual_display = [d for c, d in zip(sdlc_cols, sdlc_display) if c in sdlc_sub.columns]
    if sdlc_actual_cols:
        corrs["sdlc"] = spearman_matrix(sdlc_sub, sdlc_actual_cols)
        corrs["sdlc"].index   = sdlc_actual_display
        corrs["sdlc"].columns = sdlc_actual_display
        plot_corr_heatmap(corrs["sdlc"], sdlc_actual_display,
                          "SDLC features (issue/PR/tag nodes) — Spearman correlation",
                          OUT_DIR / "corr_sdlc.png")

    return corrs


# ── 5. Combined scoring ────────────────────────────────────────────────────────

def build_removal_table(shap_df: pd.DataFrame, corrs: dict) -> pd.DataFrame:
    """
    Score = (1 - normalised_phi) * 0.5 + max_partner_corr * 0.5
    High score → good removal candidate.
    """
    # Build a flat dict: feature_name → max |r| with any partner in same group
    max_corr: dict[str, tuple[float, str]] = {}

    # Function
    fn_corr = corrs.get("function")
    if fn_corr is not None:
        for col, disp in zip(FN_FEAT_COLS, FN_DISPLAY):
            r, partner = max_abs_partner_corr(fn_corr, col)
            # translate partner col back to display
            partner_disp = FN_DISPLAY[FN_FEAT_COLS.index(partner)] if partner in FN_FEAT_COLS else partner
            max_corr[disp] = (r, partner_disp)

    # File
    file_corr = corrs.get("file")
    if file_corr is not None:
        for col, disp in zip(FILE_FEAT_COLS, FILE_DISPLAY):
            r, partner = max_abs_partner_corr(file_corr, col)
            partner_disp = FILE_DISPLAY[FILE_FEAT_COLS.index(partner)] if partner in FILE_FEAT_COLS else partner
            max_corr[disp] = (r, partner_disp)

    # Developer
    dev_corr = corrs.get("developer")
    if dev_corr is not None:
        for disp in DEV_DISPLAY_NAMES:
            r, partner = max_abs_partner_corr(dev_corr, disp)
            max_corr[disp] = (r, partner)

    # SDLC
    sdlc_corr = corrs.get("sdlc")
    if sdlc_corr is not None:
        sdlc_display = ISSUE_DISPLAY + PR_DISPLAY + TAG_DISPLAY
        for disp in sdlc_display:
            if disp in sdlc_corr.columns:
                r, partner = max_abs_partner_corr(sdlc_corr, disp)
                max_corr[disp] = (r, partner)

    rows = []
    active = shap_df[~shap_df["ablated"]].copy()
    phi_max = active["mean_abs_phi"].max()

    for _, row in active.iterrows():
        feat   = row["feature"]
        phi    = row["mean_abs_phi"]
        rank   = int(row["rank"])
        group  = row["group"]
        r_max, partner = max_corr.get(feat, (0.0, "—"))
        norm_phi = phi / phi_max if phi_max > 0 else 0
        score    = (1 - norm_phi) * 0.5 + r_max * 0.5
        rows.append({
            "feature":       feat,
            "group":         group,
            "shap_rank":     rank,
            "mean_abs_phi":  round(phi, 6),
            "max_corr_r":    round(r_max, 3),
            "corr_partner":  partner,
            "removal_score": round(score, 4),
        })

    out = pd.DataFrame(rows).sort_values("removal_score", ascending=False).reset_index(drop=True)
    out.index += 1  # 1-based rank
    return out


# ── 6. Text report ────────────────────────────────────────────────────────────

def write_report(shap_df: pd.DataFrame, removal_df: pd.DataFrame, path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("FEATURE ABLATION ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("SHAP source: outputs/shap_analysis/ (n=3,375 test-set graphs)")
    lines.append("Correlation: Spearman r on raw node-feature CSVs")
    lines.append("")
    lines.append("NOTE — Developer feature naming bug in shap_analysis.py:")
    lines.append("  dim 2 = repo_tenure_days       (labeled 'commits_as_committer')")
    lines.append("  dim 3 = commits_as_committer   (labeled 'total_issues')")
    lines.append("  dim 4 = recent_commits_90d     (labeled 'total_prs')")
    lines.append("  dim 7 = cross_repo_commits     (labeled 'num_repos')")
    lines.append("  dim 8 = num_repos              (labeled 'repo_commits_before')")
    lines.append("  SHAP values are positionally correct; only names were wrong.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("ALREADY ABLATED (zeroed in current model — keep as-is):")
    lines.append("-" * 70)
    for f in sorted(ALREADY_ABLATED):
        row = shap_df[shap_df["feature"] == f]
        phi = float(row["mean_abs_phi"].iloc[0]) if len(row) > 0 else 0
        lines.append(f"  {f:<35}  phi={phi:.6f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP REMOVAL CANDIDATES (combined SHAP rank + correlation score):")
    lines.append("-" * 70)
    lines.append(f"{'#':<4} {'Feature':<38} {'phi':>8} {'max|r|':>7} {'partner':<35} {'score':>7}")
    lines.append("-" * 4 + " " + "-" * 38 + " " + "-" * 8 + " " + "-" * 7 + " " + "-" * 35 + " " + "-" * 7)
    for i, row in removal_df.head(20).iterrows():
        lines.append(
            f"{i:<4} {row['feature']:<38} {row['mean_abs_phi']:>8.5f} "
            f"{row['max_corr_r']:>7.3f} {row['corr_partner']:<35} {row['removal_score']:>7.4f}"
        )
    lines.append("")
    lines.append("-" * 70)
    lines.append("SUGGESTED 10 FEATURES TO ABLATE (new, not yet zeroed):")
    lines.append("-" * 70)
    suggestions = removal_df.head(10)
    for i, row in suggestions.iterrows():
        reason_parts = []
        if row["shap_rank"] >= 35:
            reason_parts.append(f"SHAP rank {row['shap_rank']}/45")
        if row["max_corr_r"] >= 0.70:
            reason_parts.append(f"r={row['max_corr_r']:.2f} with {row['corr_partner']}")
        reason = "; ".join(reason_parts) if reason_parts else "low SHAP + moderate corr"
        lines.append(f"  {i:>2}. {row['feature']:<38}  ({reason})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("FEATURE GROUP SHAP SUMMARY (active features only):")
    lines.append("-" * 70)
    grp = (shap_df[~shap_df["ablated"]]
           .groupby("group")["mean_abs_phi"]
           .agg(["mean", "sum", "count"])
           .sort_values("sum", ascending=False))
    lines.append(f"{'Group':<20} {'mean phi':>9} {'sum phi':>9} {'n feats':>8}")
    for g, row in grp.iterrows():
        lines.append(f"  {g:<20} {row['mean']:>9.5f} {row['sum']:>9.5f} {int(row['count']):>8}")

    report = "\n".join(lines)
    path.write_text(report)
    print(f"  Saved: {path}")
    print()
    print(report)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading SHAP summary …")
    shap_df = load_shap_summary()
    print(f"  {len(shap_df)} features loaded")

    print("\nPlotting SHAP bar chart …")
    plot_shap_bar(shap_df, OUT_DIR / "shap_ranking.png")

    print("\nComputing correlations …")
    corrs = compute_all_correlations()

    print("\nBuilding removal candidate table …")
    removal_df = build_removal_table(shap_df, corrs)
    removal_df.to_csv(OUT_DIR / "removal_candidates.csv")
    print(f"  Saved: {OUT_DIR / 'removal_candidates.csv'}")

    print("\nWriting analysis report …")
    write_report(shap_df, removal_df, OUT_DIR / "analysis_report.txt")

    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
