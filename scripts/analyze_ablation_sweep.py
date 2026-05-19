"""
scripts/analyze_ablation_sweep.py

Generate structured analysis report from completed ablation sweep.
Reads test_results.json and metrics.csv from each run.

Output files (written to outputs/ablation_sweep_v1/):
    results.csv          - consolidated metrics table
    results.json         - same as JSON
    analysis_report.md   - full markdown analysis
    summary.txt          - plain-text key findings

Usage: python scripts/analyze_ablation_sweep.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTDIR = ROOT / "outputs" / "ablation_sweep_v1"

RUNS = [
    ("abl_full",         "Full model (reference)"),
    ("abl_no_code_emb",  "No function BERT"),
    ("abl_no_text_emb",  "No hunk BERT"),
    ("abl_no_all_bert",  "No fn+hunk BERT"),
    ("abl_no_sdlc",      "No SDLC (issue/PR/tag)"),
    ("abl_no_developer", "No developer"),
    ("abl_code_only",    "Code-only (no process context)"),
    ("abl_context_only", "Context-only (no code content)"),
]

# Feature group descriptions for the report
FEATURE_GROUPS = {
    "abl_no_code_emb":  "function-level GraphCodeBERT embeddings (768 dims of 776)",
    "abl_no_text_emb":  "hunk-level GraphCodeBERT diff embeddings (768 dims of 770)",
    "abl_no_all_bert":  "ALL BERT embeddings (function + hunk, ~1536 dims)",
    "abl_no_sdlc":      "issue/PR/release_tag node features + relation edge attrs",
    "abl_no_developer": "developer node features + authored_by/committed_by/owns edge attrs (9+6 dims)",
    "abl_code_only":    "all process context (developer + SDLC) — code signal only",
    "abl_context_only": "all code content (fn BERT + hunk BERT + code metrics + fn categorical)",
}


def load_test_results(run_name: str) -> dict | None:
    p = OUTDIR / "checkpoints" / run_name / "test_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_val_curve(run_name: str) -> list[dict]:
    p = OUTDIR / "checkpoints" / run_name / "metrics.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p)))
    return [r for r in rows if r["split"] == "val"]


def load_train_curve(run_name: str) -> list[dict]:
    p = OUTDIR / "checkpoints" / run_name / "metrics.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p)))
    return [r for r in rows if r["split"] == "train"]


def fmt(v, decimals=4):
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def delta(run_v, ref_v, decimals=4) -> str:
    try:
        d = float(run_v) - float(ref_v)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def main():
    # ── Load all results ───────────────────────────────────────────────────────
    results = {}
    for run_name, desc in RUNS:
        tr = load_test_results(run_name)
        if tr:
            results[run_name] = {**tr, "description": desc}
        else:
            print(f"  WARNING: {run_name} has no test_results.json — skipping")

    if "abl_full" not in results:
        print("ERROR: Reference run abl_full not complete. Cannot generate report.")
        return

    ref = results["abl_full"]

    # Save results CSV and JSON
    fields = ["run", "description", "f1_best", "f1", "precision", "recall",
              "auc_pr", "auc_roc", "mcc", "thresh_best", "loss"]
    rows = []
    for run_name, desc in RUNS:
        if run_name in results:
            r = results[run_name]
            rows.append({
                "run": run_name,
                "description": desc,
                **{k: r.get(k, "") for k in fields[2:]},
            })

    with open(OUTDIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ── Sort by F1_best ────────────────────────────────────────────────────────
    ranked = sorted(
        [(rn, d) for rn, d in results.items()],
        key=lambda x: float(x[1].get("f1_best", 0.0)),
        reverse=True,
    )

    # ── Analysis report ────────────────────────────────────────────────────────
    lines = []

    lines += [
        "# Feature Ablation Study — HeteroSAGE VCC Detection",
        "",
        "**Date:** 2026-04-10  |  **Dataset:** graph_ready_v2  |  **Split:** repo_split",
        "**Base config:** gen_C (dropout=0.4, wd=5e-4, focal α=0.65 γ=1.5, lr=1e-3, seed=42, 10 epochs)",
        "**Masking strategy:** zero feature slices at load time — tensor shapes unchanged",
        "",
        "---",
        "",
        "## 1. Ranked Performance Table",
        "",
        "Sorted by test F1_best (primary metric).",
        "",
        "| Rank | Run | Description | F1_best | Δ F1_best | F1@0.5 | Prec | Recall | AUC-PR | MCC | Thresh |",
        "|------|-----|-------------|---------|-----------|--------|------|--------|--------|-----|--------|",
    ]
    for i, (run_name, r) in enumerate(ranked):
        d = delta(r.get("f1_best"), ref.get("f1_best"))
        ref_marker = " ★" if run_name == "abl_full" else ""
        lines.append(
            f"| {i+1} | `{run_name}`{ref_marker} | {r['description']} "
            f"| {fmt(r.get('f1_best'))} | {d} "
            f"| {fmt(r.get('f1'))} | {fmt(r.get('precision'))} | {fmt(r.get('recall'))} "
            f"| {fmt(r.get('auc_pr'))} | {fmt(r.get('mcc'))} | {fmt(r.get('thresh_best'), 2)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Performance Drop vs Full Model",
        "",
        "Negative Δ = degradation. Sorted worst-to-best degradation.",
        "",
        "| Run | Removed features | Δ F1_best | Δ AUC-PR | Δ MCC |",
        "|-----|-----------------|-----------|----------|-------|",
    ]
    ablation_rows = [(rn, r) for rn, r in results.items() if rn != "abl_full"]
    ablation_rows_sorted = sorted(
        ablation_rows,
        key=lambda x: float(x[1].get("f1_best", 0.0)) - float(ref.get("f1_best", 0.0)),
    )
    for run_name, r in ablation_rows_sorted:
        desc = FEATURE_GROUPS.get(run_name, r["description"])
        d_f1 = delta(r.get("f1_best"), ref.get("f1_best"))
        d_pr = delta(r.get("auc_pr"), ref.get("auc_pr"))
        d_mcc = delta(r.get("mcc"), ref.get("mcc"))
        lines.append(f"| `{run_name}` | {desc} | {d_f1} | {d_pr} | {d_mcc} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Feature Group Contributions",
        "",
    ]

    # Individual feature group analysis
    def interpret(run_name: str) -> list[str]:
        if run_name not in results:
            return [f"*Run {run_name} not available.*"]
        r = results[run_name]
        d_f1 = float(r.get("f1_best", 0.0)) - float(ref.get("f1_best", 0.0))
        d_pr = float(r.get("auc_pr", 0.0)) - float(ref.get("auc_pr", 0.0))
        prec = float(r.get("precision", 0.0))
        rec = float(r.get("recall", 0.0))
        thresh = float(r.get("thresh_best", 0.5))
        ref_thresh = float(ref.get("thresh_best", 0.5))
        thresh_note = ""
        if abs(thresh - ref_thresh) >= 0.1:
            direction = "higher" if thresh > ref_thresh else "lower"
            thresh_note = (
                f" Optimal threshold shifted {direction} to {thresh:.2f} "
                f"(vs {ref_thresh:.2f} in full model), suggesting calibration change."
            )

        importance = "**critical**" if d_f1 < -0.03 else ("**notable**" if d_f1 < -0.01 else "**marginal**")
        verdict = f"Removing these features causes Δ F1_best={d_f1:+.4f}, Δ AUC-PR={d_pr:+.4f}."
        impact = f"Impact is {importance}.{thresh_note}"
        prec_rec = f"Precision={prec:.4f}, Recall={rec:.4f} (ratio {prec/max(rec,0.001):.2f}:1)."
        return [verdict, impact, prec_rec]

    feature_order = [
        ("abl_no_code_emb",  "### 3.1 Function-level Code Embeddings (GraphCodeBERT)"),
        ("abl_no_text_emb",  "### 3.2 Hunk-level Text Embeddings (diff-level BERT)"),
        ("abl_no_all_bert",  "### 3.3 All BERT Embeddings (function + hunk combined)"),
        ("abl_no_sdlc",      "### 3.4 SDLC Context (issue / PR / release tag)"),
        ("abl_no_developer", "### 3.5 Developer & Ownership Features"),
        ("abl_code_only",    "### 3.6 Code-only (all process context removed)"),
        ("abl_context_only", "### 3.7 Context-only (all code content removed)"),
    ]

    for run_name, section_title in feature_order:
        lines.append(section_title)
        lines.append("")
        if run_name in results:
            for sentence in interpret(run_name):
                lines.append(sentence)
                lines.append("")
        else:
            lines.append("*Run not available.*")
            lines.append("")

    lines += [
        "---",
        "",
        "## 4. Precision vs Recall Behavior",
        "",
        "| Run | Precision | Recall | P/R ratio | Thresh |",
        "|-----|-----------|--------|-----------|--------|",
    ]
    for run_name, desc in RUNS:
        if run_name in results:
            r = results[run_name]
            prec = float(r.get("precision", 0.0))
            rec = float(r.get("recall", 0.0))
            ratio = prec / max(rec, 0.001)
            thresh = float(r.get("thresh_best", 0.5))
            lines.append(
                f"| `{run_name}` | {prec:.4f} | {rec:.4f} | {ratio:.2f} | {thresh:.2f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 5. Calibration & Threshold Observations",
        "",
    ]
    ref_thresh = float(ref.get("thresh_best", 0.5))
    lines.append(f"Full model optimal threshold: **{ref_thresh:.2f}**")
    lines.append("")
    calibration_notes = []
    for run_name, desc in RUNS[1:]:
        if run_name in results:
            thresh = float(results[run_name].get("thresh_best", 0.5))
            shift = thresh - ref_thresh
            if abs(shift) >= 0.10:
                direction = "↑" if shift > 0 else "↓"
                calibration_notes.append(
                    f"- `{run_name}`: threshold drifted to {thresh:.2f} ({direction}{abs(shift):.2f})"
                    f" — model {'under-confident (scores compressed toward 0.5)' if shift > 0 else 'over-confident'}"
                )
    if calibration_notes:
        lines.append("**Notable threshold shifts:**")
        lines += calibration_notes
    else:
        lines.append("No major threshold shifts detected — calibration stable across ablations.")
    lines.append("")

    # ── Val curve peak summary ─────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 6. Validation Curve Peaks",
        "",
        "| Run | Best val F1* | Best epoch | Best val AUC-PR |",
        "|-----|-------------|-----------|----------------|",
    ]
    for run_name, desc in RUNS:
        curve = load_val_curve(run_name)
        if curve:
            best_f1_row = max(curve, key=lambda r: float(r["f1_best"]))
            best_pr_row = max(curve, key=lambda r: float(r["auc_pr"]))
            lines.append(
                f"| `{run_name}` | {float(best_f1_row['f1_best']):.4f} @ep{best_f1_row['epoch']}"
                f" | {best_f1_row['epoch']}"
                f" | {float(best_pr_row['auc_pr']):.4f} @ep{best_pr_row['epoch']} |"
            )
        else:
            lines.append(f"| `{run_name}` | N/A | N/A | N/A |")

    # ── Key conclusions ────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## 7. Key Conclusions",
        "",
    ]

    # Generate conclusions dynamically
    conclusions = []

    # Which feature group matters most?
    if ablation_rows_sorted:
        worst_run, worst_r = ablation_rows_sorted[0]
        worst_d = float(worst_r.get("f1_best", 0.0)) - float(ref.get("f1_best", 0.0))
        if worst_d < -0.02:
            conclusions.append(
                f"1. **Most impactful feature group**: `{worst_run}` — removing it causes "
                f"the largest F1_best drop (Δ={worst_d:+.4f})."
            )

    # Code vs context
    code_only_f1 = float(results.get("abl_code_only", {}).get("f1_best", 0.0))
    ctx_only_f1 = float(results.get("abl_context_only", {}).get("f1_best", 0.0))
    ref_f1 = float(ref.get("f1_best", 0.0))
    if "abl_code_only" in results and "abl_context_only" in results:
        if code_only_f1 > ctx_only_f1:
            conclusions.append(
                f"2. **Code vs context**: Code-only retains more signal "
                f"(F1_best={code_only_f1:.4f}) than context-only ({ctx_only_f1:.4f}). "
                "The model relies more on code content than process metadata."
            )
        else:
            conclusions.append(
                f"2. **Code vs context**: Context-only retains more signal "
                f"(F1_best={ctx_only_f1:.4f}) than code-only ({code_only_f1:.4f}). "
                "The model relies more on process metadata than code content — unexpected."
            )

    # BERT contribution
    fn_emb_d = float(results.get("abl_no_code_emb", {}).get("f1_best", 0.0)) - ref_f1
    hunk_emb_d = float(results.get("abl_no_text_emb", {}).get("f1_best", 0.0)) - ref_f1
    all_bert_d = float(results.get("abl_no_all_bert", {}).get("f1_best", 0.0)) - ref_f1
    if "abl_no_code_emb" in results:
        conclusions.append(
            f"3. **Function BERT embeddings**: Δ F1_best={fn_emb_d:+.4f}. "
            + ("Critical — confirms GraphCodeBERT carries essential cross-repo signal." if fn_emb_d < -0.02
               else "Marginal — model compensates via other features.")
        )
    if "abl_no_text_emb" in results:
        conclusions.append(
            f"4. **Hunk BERT embeddings**: Δ F1_best={hunk_emb_d:+.4f}. "
            + ("Important — diff-level context adds complementary signal to function-level." if hunk_emb_d < -0.02
               else "Marginal — hunk embeddings are redundant with function-level BERT.")
        )

    # Developer/SDLC
    dev_d = float(results.get("abl_no_developer", {}).get("f1_best", 0.0)) - ref_f1
    sdlc_d = float(results.get("abl_no_sdlc", {}).get("f1_best", 0.0)) - ref_f1
    if "abl_no_developer" in results:
        conclusions.append(
            f"5. **Developer & ownership features**: Δ F1_best={dev_d:+.4f}. "
            + ("Notable contribution — developer history and ownership encode real risk patterns." if abs(dev_d) > 0.01
               else "Negligible — developer features are not a meaningful signal for VCC detection.")
        )
    if "abl_no_sdlc" in results:
        conclusions.append(
            f"6. **SDLC context (issue/PR/tag)**: Δ F1_best={sdlc_d:+.4f}. "
            + ("Meaningful — issue/PR linkage provides useful commit-type context." if abs(sdlc_d) > 0.01
               else "Negligible — SDLC linkage is not predictive of vulnerability contribution.")
        )

    # Generalization caveat
    conclusions.append(
        "7. **Generalization caveat**: All runs use repo-split evaluation. "
        "The ~0.25 F1 train-val gap is structural (domain shift). "
        "Ablation differences reflect relative information content in test repos, "
        "not absolute model quality. Interpret deltas critically."
    )

    for c in conclusions:
        lines.append(c)
        lines.append("")

    lines += [
        "---",
        "",
        "## 8. Potential Issues & Caveats",
        "",
        "- **10-epoch training**: All runs stop at ep10 to be consistent. "
        "Some ablations may not have fully converged — peaks may differ by config. "
        "Results are directionally reliable but absolute numbers may shift with full training.",
        "",
        "- **Masking vs removal**: Zeroing features keeps tensor shapes identical. "
        "The model still receives the zero vectors through its projection layers, "
        "which may learn to ignore them or assign them arbitrary meaning. "
        "True removal (changing model architecture) would be cleaner but breaks comparability.",
        "",
        "- **BERT dimension dominance**: Function nodes are 776-dim where 768 are BERT. "
        "Zeroing BERT removes ~99% of the information in function nodes. "
        "The 'no code emb' result measures almost the entire function node contribution.",
        "",
        "- **Edge attr interaction**: In `abl_no_developer`, developer nodes are zeroed "
        "but they still exist as graph nodes — messages still flow through them (as zeros). "
        "This is equivalent to removing the signal but not the topology.",
        "",
        "- **Seed=42 only**: Single-seed runs. F1 variance on this dataset is ~±0.005-0.010 "
        "based on prior experiments. Differences >0.015 are reliable; smaller deltas may be noise.",
    ]

    report = "\n".join(lines)
    (OUTDIR / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"Analysis report written: {OUTDIR / 'analysis_report.md'}")

    # ── Summary text ───────────────────────────────────────────────────────────
    summary_lines = [
        "FEATURE ABLATION SWEEP — KEY FINDINGS",
        "=" * 50,
        "",
        f"Reference (abl_full): F1_best={fmt(ref.get('f1_best'))}  AUC-PR={fmt(ref.get('auc_pr'))}",
        "",
        "Performance drops (Δ F1_best vs full model):",
    ]
    for run_name, r in ablation_rows_sorted:
        d_f1 = float(r.get("f1_best", 0.0)) - float(ref.get("f1_best", 0.0))
        summary_lines.append(
            f"  {run_name:25s}: Δ={d_f1:+.4f}  "
            f"(F1*={fmt(r.get('f1_best'))}, AUC-PR={fmt(r.get('auc_pr'))})"
        )

    summary_lines += [
        "",
        "Quick interpretation:",
        f"  Most important removed group: {ablation_rows_sorted[0][0] if ablation_rows_sorted else 'N/A'}",
        f"  Code-only F1*: {fmt(results.get('abl_code_only', {}).get('f1_best', 'N/A'))}",
        f"  Context-only F1*: {fmt(results.get('abl_context_only', {}).get('f1_best', 'N/A'))}",
        "",
        "See analysis_report.md for full interpretation.",
    ]

    summary = "\n".join(summary_lines)
    (OUTDIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Summary written: {OUTDIR / 'summary.txt'}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
