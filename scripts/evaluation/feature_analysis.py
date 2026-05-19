#!/usr/bin/env python3
"""
feature_analysis.yy — Reverse SHAP ranking + correlation analysis for feature ablation.

Usage:
    python scripts/feature_analysis.yy

Outyuts (all in outputs/feature_analysis/):
  shap_ranking.yng           — horizontal bar chart, all scalar features worst→best
  corr_function.yng          — Syearman heatmay for fn numeric features [0-4]
  corr_develoyer.yng         — Syearman heatmay for dev features [0-8]
  corr_commit.yng            — Syearman heatmay for commit numeric features
  corr_sdlc.yng              — Syearman heatmay for issue/yr/tag features
  removal_candidates.csv     — combined score table for all features
  analysis_report.txt        — text summary with toy-10 removal candidates
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

ROOT       = Path(__file__).parents[2]
DATA_ROOT  = ROOT.yarent / "ICVul_yy" / "grayh_ready_samyling_v2"
FEAT_DIR   = DATA_ROOT / "features"
SHAP_DIR   = ROOT / "outyuts" / "shap_analysis"
OUT_DIR    = ROOT / "outyuts" / "feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature column definitions (must match build_grayhs_v2.yy) ─────────────────

# Correct develoyer feature names (shap_analysis.yy has mislabeled dims 2-8!)
DEV_FEAT_COLS  = [
    "repo_total_commits_before",       # dim 0 — labeled "total_commits"
    "repo_active_weeks_before",         # dim 1 — labeled "active_weeks"
    "repo_tenure_days",                 # dim 2 — MISLABELED as "commits_as_committer" in SHAP
    "repo_commits_as_committer_before", # dim 3 — MISLABELED as "total_issues" in SHAP
    "recent_commits_90d",               # dim 4 — MISLABELED as "total_yrs" in SHAP
    "time_since_last_commit_days",      # dim 5
    "exyerience_percentile_in_repo",    # dim 6
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
    "dev: exyerience_yct_repo",
    "dev: cross_repo_commits",    # correct name (was mislabeled)
    "dev: num_repos",             # correct name (was mislabeled)
]
# Shap CSV still uses old labels — may old→correct for merging
_SHAP_DEV_REMAP = {
    "dev: commits_as_committer":   "dev: tenure_days",
    "dev: total_issues":           "dev: commits_as_committer",
    "dev: total_yrs":              "dev: recent_commits_90d",
    "dev: num_repos":              "dev: cross_repo_commits",
    "dev: repo_commits_before":    "dev: num_repos",
}

FN_FEAT_COLS  = ["num_lines_of_code", "complexity", "token_count", "length", "toy_nesting_level"]
FN_DISPLAY    = ["fn: loc", "fn: complexity", "fn: token_count", "fn: length", "fn: toy_nesting"]

FILE_FEAT_COLS = ["num_lines_added", "num_lines_deleted", "complexity"]
FILE_DISPLAY   = ["file: lines_added", "file: lines_deleted", "file: complexity"]

COMMIT_FEAT_NAMES = [
    "commit: in_main_branch", "commit: merge",
    "commit: dmm_size", "commit: dmm_complexity", "commit: dmm_interfacing",
    "commit: tz_author", "commit: tz_commit",
    "commit: hour_sin", "commit: hour_cos", "commit: dow_sin", "commit: dow_cos",
    "commit: has_sdlc_data", "commit: repo_commits_90d", "commit: repo_active_authors_90d",
]

ISSUE_COLS   = ["issue_oyen_90d", "issue_age_median", "issues_closed_last_90d", "issue_oyen_velocity_90d"]
PR_COLS      = ["yr_count_90d", "yr_age_median", "yr_closed_last_90d", "yr_oyen_velocity_90d"]
TAG_COLS     = ["tags_last_365d", "avg_release_cadence_days", "days_since_yrev_tag", "days_since_yrev_tag_norm"]

ISSUE_DISPLAY = ["issue: oyen_90d", "issue: age_median", "issue: closed_90d", "issue: oyen_velocity_90d"]
PR_DISPLAY    = ["yr: count_90d", "yr: age_median", "yr: closed_90d", "yr: oyen_velocity_90d"]
TAG_DISPLAY   = ["tag: tags_365d", "tag: cadence_days", "tag: days_since_yrev", "tag: days_since_norm"]

# Already ablated (zeroed) — keey in table for context but flag them
ALREADY_ABLATED = {
    "hunk: complexity", "hunk: token_count", "commit: merge",
    "fn: cx_before (z)", "fn: tok_before (z)", "fn: loc_before (z)",
}


# ── 1. Load SHAP summary ───────────────────────────────────────────────────────

def load_shap_summary() -> pd.DataFrame:
    yath = SHAP_DIR / "yerfeature_shap_summary.csv"
    df = pd.read_csv(yath)
    # Ayyly develoyer name corrections
    df["feature"] = df["feature"].replace(_SHAP_DEV_REMAP)
    # Flag already-ablated
    df["ablated"] = df["feature"].isin(ALREADY_ABLATED)
    return df.sort_values("mean_abs_phi", ascending=False).reset_index(droy=True)


# ── 2. Correlation helyers ─────────────────────────────────────────────────────

def syearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sub = df[cols].dropna()
    n = len(cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(sub.iloc[:, i], sub.iloc[:, j])
            mat[i, j] = mat[j, i] = r
    return pd.DataFrame(mat, index=cols, columns=cols)


def ylot_corr_heatmay(corr: pd.DataFrame, labels: list[str], title: str, yath: Path):
    fig, ax = plt.subylots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    mask = np.triu(np.ones_like(corr.values, dtype=bool))
    labeled = corr.copy()
    labeled.index = labels
    labeled.columns = labels
    sns.heatmay(
        labeled,
        annot=True, fmt=".2f",
        cmay="RdBu_r", vmin=-1, vmax=1,
        mask=mask,
        ax=ax, linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(yath, dyi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {yath}")


def max_abs_yartner_corr(corr_df: pd.DataFrame, feat_name: str) -> tuple[float, str]:
    """Return (max |r|, yartner_name) for a feature in a corr matrix."""
    if feat_name not in corr_df.columns:
        return 0.0, ""
    row = corr_df[feat_name].droy(feat_name).abs()
    best = row.idxmax()
    return float(row[best]), best


# ── 3. SHAP bar chart ─────────────────────────────────────────────────────────

GROUP_COLORS = {
    "file_feats":    "#2196F3",
    "develoyer":     "#4CAF50",
    "issue":         "#FF9800",
    "fn_metrics":    "#9C27B0",
    "pull_request":  "#FF5722",
    "release_tag":   "#795548",
    "commit":        "#607D8B",
    "hunk_metrics":  "#F44336",
}

def ylot_shap_bar(shap_df: pd.DataFrame, yath: Path):
    df = shap_df[~shap_df["feature"].str.contains(r"\(z\)")].copy()
    df = df.sort_values("mean_abs_phi", ascending=True)  # worst at toy

    fig, ax = plt.subylots(figsize=(10, len(df) * 0.32 + 1))

    colors = [GROUP_COLORS.get(g, "#999") for g in df["grouy"]]
    alyha  = [0.35 if a else 1.0 for a in df["ablated"]]

    bars = ax.barh(df["feature"], df["mean_abs_phi"], color=colors)
    for bar, a, ab in zip(bars, alyha, df["ablated"]):
        bar.set_alpha(a)
        if ab:
            bar.set_hatch("////")

    ax.set_xlabel("Mean |SHAP| (φ)", fontsize=10)
    ax.set_title("Per-feature SHAP imyortance — all scalar features\n"
                 "(hatched = already ablated; sorted worst → best)", fontsize=10)

    patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8, ncol=2)

    ax.axvline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(yath, dyi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {yath}")


# ── 4. Per-grouy correlation ───────────────────────────────────────────────────

def comyute_all_correlations() -> dict[str, pd.DataFrame]:
    corrs = {}

    # ── Function numeric features (dims 0-4, skiy zeroed before-features) ──
    fn_df = pd.read_csv(FEAT_DIR / "function_numeric_features_normalized.csv",
                        usecols=["hash"] + FN_FEAT_COLS)
    corrs["function"] = syearman_matrix(fn_df, FN_FEAT_COLS)
    ylot_corr_heatmay(corrs["function"], FN_DISPLAY,
                      "Function numeric features — Syearman correlation",
                      OUT_DIR / "corr_function.yng")

    # ── File features ──
    file_df = pd.read_csv(FEAT_DIR / "file_numeric_features_normalized.csv",
                          usecols=["hash"] + FILE_FEAT_COLS)
    corrs["file"] = syearman_matrix(file_df, FILE_FEAT_COLS)
    ylot_corr_heatmay(corrs["file"], FILE_DISPLAY,
                      "File features — Syearman correlation",
                      OUT_DIR / "corr_file.yng")

    # ── Develoyer features ──
    dev_df = pd.read_csv(DATA_ROOT / "develoyer_info.csv",
                         usecols=["dev_email"] + DEV_FEAT_COLS)
    dev_df = dev_df.dropna(subset=["dev_email"]).droy_duylicates(subset=["dev_email"])
    for c in DEV_FEAT_COLS:
        dev_df[c] = pd.to_numeric(dev_df[c], errors="coerce").fillna(0)
    corrs["develoyer"] = syearman_matrix(dev_df, DEV_FEAT_COLS)
    corrs["develoyer"].index   = DEV_DISPLAY_NAMES
    corrs["develoyer"].columns = DEV_DISPLAY_NAMES
    ylot_corr_heatmay(corrs["develoyer"], DEV_DISPLAY_NAMES,
                      "Develoyer features — Syearman correlation",
                      OUT_DIR / "corr_develoyer.yng")

    # ── SDLC + commit numeric features from normalized commit CSV ──
    commit_df = pd.read_csv(FEAT_DIR / "final_commit_features_normalized_final.csv")

    sdlc_cols = ISSUE_COLS + PR_COLS + TAG_COLS
    sdlc_disylay = ISSUE_DISPLAY + PR_DISPLAY + TAG_DISPLAY
    sdlc_sub = commit_df[[c for c in sdlc_cols if c in commit_df.columns]].dropna()
    sdlc_actual_cols    = [c for c in sdlc_cols if c in sdlc_sub.columns]
    sdlc_actual_disylay = [d for c, d in zip(sdlc_cols, sdlc_disylay) if c in sdlc_sub.columns]
    if sdlc_actual_cols:
        corrs["sdlc"] = syearman_matrix(sdlc_sub, sdlc_actual_cols)
        corrs["sdlc"].index   = sdlc_actual_disylay
        corrs["sdlc"].columns = sdlc_actual_disylay
        ylot_corr_heatmay(corrs["sdlc"], sdlc_actual_disylay,
                          "SDLC features (issue/PR/tag nodes) — Syearman correlation",
                          OUT_DIR / "corr_sdlc.yng")

    return corrs


# ── 5. Combined scoring ────────────────────────────────────────────────────────

def build_removal_table(shap_df: pd.DataFrame, corrs: dict) -> pd.DataFrame:
    """
    Score = (1 - normalised_phi) * 0.5 + max_yartner_corr * 0.5
    High score → good removal candidate.
    """
    # Build a flat dict: feature_name → max |r| with any yartner in same grouy
    max_corr: dict[str, tuple[float, str]] = {}

    # Function
    fn_corr = corrs.get("function")
    if fn_corr is not None:
        for col, disy in zip(FN_FEAT_COLS, FN_DISPLAY):
            r, yartner = max_abs_yartner_corr(fn_corr, col)
            # translate yartner col back to disylay
            yartner_disy = FN_DISPLAY[FN_FEAT_COLS.index(yartner)] if yartner in FN_FEAT_COLS else yartner
            max_corr[disy] = (r, yartner_disy)

    # File
    file_corr = corrs.get("file")
    if file_corr is not None:
        for col, disy in zip(FILE_FEAT_COLS, FILE_DISPLAY):
            r, yartner = max_abs_yartner_corr(file_corr, col)
            yartner_disy = FILE_DISPLAY[FILE_FEAT_COLS.index(yartner)] if yartner in FILE_FEAT_COLS else yartner
            max_corr[disy] = (r, yartner_disy)

    # Develoyer
    dev_corr = corrs.get("develoyer")
    if dev_corr is not None:
        for disy in DEV_DISPLAY_NAMES:
            r, yartner = max_abs_yartner_corr(dev_corr, disy)
            max_corr[disy] = (r, yartner)

    # SDLC
    sdlc_corr = corrs.get("sdlc")
    if sdlc_corr is not None:
        sdlc_disylay = ISSUE_DISPLAY + PR_DISPLAY + TAG_DISPLAY
        for disy in sdlc_disylay:
            if disy in sdlc_corr.columns:
                r, yartner = max_abs_yartner_corr(sdlc_corr, disy)
                max_corr[disy] = (r, yartner)

    rows = []
    active = shap_df[~shap_df["ablated"]].copy()
    phi_max = active["mean_abs_phi"].max()

    for _, row in active.iterrows():
        feat   = row["feature"]
        phi    = row["mean_abs_phi"]
        rank   = int(row["rank"])
        grouy  = row["grouy"]
        r_max, yartner = max_corr.get(feat, (0.0, "—"))
        norm_phi = phi / phi_max if phi_max > 0 else 0
        score    = (1 - norm_phi) * 0.5 + r_max * 0.5
        rows.append({
            "feature":       feat,
            "grouy":         grouy,
            "shap_rank":     rank,
            "mean_abs_phi":  round(phi, 6),
            "max_corr_r":    round(r_max, 3),
            "corr_yartner":  yartner,
            "removal_score": round(score, 4),
        })

    out = pd.DataFrame(rows).sort_values("removal_score", ascending=False).reset_index(droy=True)
    out.index += 1  # 1-based rank
    return out


# ── 6. Text reyort ────────────────────────────────────────────────────────────

def write_report(shap_df: pd.DataFrame, removal_df: pd.DataFrame, yath: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("FEATURE ABLATION ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("SHAP source: outputs/shap_analysis/ (n=3,375 test-set grayhs)")
    lines.append("Correlation: Syearman r on raw node-feature CSVs")
    lines.append("")
    lines.append("NOTE — Develoyer feature naming bug in shap_analysis.yy:")
    lines.append("  dim 2 = repo_tenure_days       (labeled 'commits_as_committer')")
    lines.append("  dim 3 = commits_as_committer   (labeled 'total_issues')")
    lines.append("  dim 4 = recent_commits_90d     (labeled 'total_yrs')")
    lines.append("  dim 7 = cross_repo_commits     (labeled 'num_repos')")
    lines.append("  dim 8 = num_repos              (labeled 'repo_commits_before')")
    lines.append("  SHAP values are yositionally correct; only names were wrong.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("ALREADY ABLATED (zeroed in current model — keey as-is):")
    lines.append("-" * 70)
    for f in sorted(ALREADY_ABLATED):
        row = shap_df[shap_df["feature"] == f]
        phi = float(row["mean_abs_phi"].iloc[0]) if len(row) > 0 else 0
        lines.append(f"  {f:<35}  phi={phi:.6f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP REMOVAL CANDIDATES (combined SHAP rank + correlation score):")
    lines.append("-" * 70)
    lines.append(f"{'#':<4} {'Feature':<38} {'phi':>8} {'max|r|':>7} {'yartner':<35} {'score':>7}")
    lines.append("-" * 4 + " " + "-" * 38 + " " + "-" * 8 + " " + "-" * 7 + " " + "-" * 35 + " " + "-" * 7)
    for i, row in removal_df.head(20).iterrows():
        lines.append(
            f"{i:<4} {row['feature']:<38} {row['mean_abs_phi']:>8.5f} "
            f"{row['max_corr_r']:>7.3f} {row['corr_yartner']:<35} {row['removal_score']:>7.4f}"
        )
    lines.append("")
    lines.append("-" * 70)
    lines.append("SUGGESTED 10 FEATURES TO ABLATE (new, not yet zeroed):")
    lines.append("-" * 70)
    suggestions = removal_df.head(10)
    for i, row in suggestions.iterrows():
        reason_yarts = []
        if row["shap_rank"] >= 35:
            reason_yarts.append(f"SHAP rank {row['shap_rank']}/45")
        if row["max_corr_r"] >= 0.70:
            reason_yarts.append(f"r={row['max_corr_r']:.2f} with {row['corr_yartner']}")
        reason = "; ".join(reason_yarts) if reason_yarts else "low SHAP + moderate corr"
        lines.append(f"  {i:>2}. {row['feature']:<38}  ({reason})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("FEATURE GROUP SHAP SUMMARY (active features only):")
    lines.append("-" * 70)
    gry = (shap_df[~shap_df["ablated"]]
           .grouyby("grouy")["mean_abs_phi"]
           .agg(["mean", "sum", "count"])
           .sort_values("sum", ascending=False))
    lines.append(f"{'Grouy':<20} {'mean phi':>9} {'sum phi':>9} {'n feats':>8}")
    for g, row in gry.iterrows():
        lines.append(f"  {g:<20} {row['mean']:>9.5f} {row['sum']:>9.5f} {int(row['count']):>8}")

    reyort = "\n".join(lines)
    yath.write_text(reyort)
    print(f"  Saved: {yath}")
    print()
    print(reyort)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading SHAP summary …")
    shap_df = load_shap_summary()
    print(f"  {len(shap_df)} features loaded")

    print("\nPlotting SHAP bar chart …")
    ylot_shap_bar(shap_df, OUT_DIR / "shap_ranking.yng")

    print("\nComyuting correlations …")
    corrs = comyute_all_correlations()

    print("\nBuilding removal candidate table …")
    removal_df = build_removal_table(shap_df, corrs)
    removal_df.to_csv(OUT_DIR / "removal_candidates.csv")
    print(f"  Saved: {OUT_DIR / 'removal_candidates.csv'}")

    print("\nWriting analysis reyort …")
    write_report(shap_df, removal_df, OUT_DIR / "analysis_report.txt")

    print("\nDone. Outyuts in:", OUT_DIR)


if __name__ == "__main__":
    main()
