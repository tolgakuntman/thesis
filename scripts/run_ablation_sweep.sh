#!/usr/bin/env bash
# run_ablation_sweep.sh
#
# Feature ablation sweep for the HeteroSAGE VCC detection model.
# Systematically zeros feature groups to measure each group's contribution.
#
# All runs use gen_C "winner" settings:
#   dropout=0.4, wd=5e-4, focal α=0.65 γ=1.5, lr=1e-3, warmup=3ep
#   10 epochs (consistent with gen sweep), patience=10 (no early stopping),
#   batch_size=128, seed=42, perrepo_norm
#
# Output: outputs/ablation_sweep_v1/checkpoints/<run_name>/
# Logs:   outputs/ablation_sweep_v1/logs/<run_name>.log

set -e
cd "$(dirname "$0")/.."

OUTDIR="outputs/ablation_sweep_v1"
mkdir -p "$OUTDIR/logs"

BASE="--graphs_dir outputs/graph_ready_v2/graphs \
      --split_index outputs/graph_ready_v2/split_index.csv \
      --perrepo_scaler outputs/graph_ready_v2/perrepo_scaler_v2.json \
      --perrepo_norm \
      --output_dir $OUTDIR/checkpoints \
      --epochs 10 --patience 10 --batch_size 128 --num_workers 0 \
      --hidden 128 --lr 1e-3 --warmup_epochs 3 \
      --dropout 0.4 --weight_decay 5e-4 \
      --focal_alpha 0.65 --focal_gamma 1.5 \
      --seed 42"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTDIR/progress.log"; }

log "=== Ablation sweep started: $(date) ==="
log "8 configurations × 10 epochs each"
log ""

# ── 1: Full model (reference) ─────────────────────────────────────────────────
log "[1/8] abl_full — full model, no masking"
python scripts/train.py $BASE --run_name abl_full \
    > "$OUTDIR/logs/abl_full.log" 2>&1
log "[1/8] DONE: abl_full"

# ── 2: No function BERT embeddings ────────────────────────────────────────────
log "[2/8] abl_no_code_emb — zero function GraphCodeBERT (768 dims)"
python scripts/train.py $BASE --run_name abl_no_code_emb \
    --ablate_code_emb \
    > "$OUTDIR/logs/abl_no_code_emb.log" 2>&1
log "[2/8] DONE: abl_no_code_emb"

# ── 3: No hunk BERT embeddings ────────────────────────────────────────────────
log "[3/8] abl_no_text_emb — zero hunk GraphCodeBERT diff embeddings (768 dims)"
python scripts/train.py $BASE --run_name abl_no_text_emb \
    --ablate_hunk_emb \
    > "$OUTDIR/logs/abl_no_text_emb.log" 2>&1
log "[3/8] DONE: abl_no_text_emb"

# ── 4: No BERT embeddings at all ──────────────────────────────────────────────
log "[4/8] abl_no_all_bert — zero both function AND hunk BERT embeddings"
python scripts/train.py $BASE --run_name abl_no_all_bert \
    --ablate_code_emb --ablate_hunk_emb \
    > "$OUTDIR/logs/abl_no_all_bert.log" 2>&1
log "[4/8] DONE: abl_no_all_bert"

# ── 5: No SDLC (issue/PR/tag) — keep developer ───────────────────────────────
log "[5/8] abl_no_sdlc — zero issue/PR/tag node features + edges (keep developer)"
python scripts/train.py $BASE --run_name abl_no_sdlc \
    --keep_sdlc_developer_only \
    > "$OUTDIR/logs/abl_no_sdlc.log" 2>&1
log "[5/8] DONE: abl_no_sdlc"

# ── 6: No developer — keep SDLC ──────────────────────────────────────────────
log "[6/8] abl_no_developer — zero developer node features + dev edge attrs (keep SDLC)"
python scripts/train.py $BASE --run_name abl_no_developer \
    --ablate_developer \
    > "$OUTDIR/logs/abl_no_developer.log" 2>&1
log "[6/8] DONE: abl_no_developer"

# ── 7: Code-only (no developer, no SDLC) ─────────────────────────────────────
log "[7/8] abl_code_only — zero ALL process context (developer + SDLC)"
python scripts/train.py $BASE --run_name abl_code_only \
    --keep_sdlc_developer_only --ablate_developer \
    > "$OUTDIR/logs/abl_code_only.log" 2>&1
log "[7/8] DONE: abl_code_only"

# ── 8: Context-only (no code content) ────────────────────────────────────────
log "[8/8] abl_context_only — zero ALL code content (fn BERT + hunk BERT + code metrics)"
python scripts/train.py $BASE --run_name abl_context_only \
    --ablate_code_emb --ablate_hunk_emb \
    --ablate_fn_code_metrics --ablate_file_code_metrics --ablate_fn_categorical \
    > "$OUTDIR/logs/abl_context_only.log" 2>&1
log "[8/8] DONE: abl_context_only"

log ""
log "=== All 8 ablation runs complete: $(date) ==="

# Collect test results into a summary CSV
python - <<'PY'
import json, csv
from pathlib import Path

outdir = Path("outputs/ablation_sweep_v1")
runs = [
    ("abl_full",         "Full model (no masking)"),
    ("abl_no_code_emb",  "No function BERT"),
    ("abl_no_text_emb",  "No hunk BERT"),
    ("abl_no_all_bert",  "No function + hunk BERT"),
    ("abl_no_sdlc",      "No SDLC (issue/PR/tag)"),
    ("abl_no_developer", "No developer"),
    ("abl_code_only",    "Code-only (no process context)"),
    ("abl_context_only", "Context-only (no code content)"),
]

fields = ["run", "description", "f1_best", "f1", "precision", "recall",
          "auc_pr", "auc_roc", "mcc", "thresh_best", "loss"]
rows = []
for run, desc in runs:
    p = outdir / "checkpoints" / run / "test_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        row = {"run": run, "description": desc}
        for k in fields[2:]:
            row[k] = d.get(k, "")
        rows.append(row)
    else:
        print(f"  MISSING: {run}")

if rows:
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(outdir / "results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results saved: {len(rows)}/{len(runs)} runs complete")
PY
