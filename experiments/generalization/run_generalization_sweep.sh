#!/usr/bin/env bash
# run_generalization_sweep.sh
#
# 10-epoch local sweep to find better-generalizing configs.
# Each run uses REG04 as the base (dropout=0.4, wd=5e-4, bs=128)
# and varies one axis at a time.
#
# Usage (from repo root, with conda env active):
#   bash scripts/run_generalization_sweep.sh
#
# Results land in checkpoints/<run_name>/metrics.csv

set -e
cd "$(dirname "$0")/.."

BASE="--graphs_dir outputs/graph_ready_v2/graphs \
      --split_index outputs/graph_ready_v2/split_index.csv \
      --perrepo_scaler outputs/graph_ready_v2/perrepo_scaler_v2.json \
      --perrepo_norm --epochs 10 --patience 10 --batch_size 128 --num_workers 0 \
      --hidden 128 --lr 1e-3 --warmup_epochs 3"

# ── A: REG04 reference (dropout=0.4, wd=5e-4) ─────────────────────────────────
echo "=== [1/5] REG04 reference ==="
python scripts/train.py $BASE \
  --run_name gen_A_reg04 \
  --dropout 0.4 --weight_decay 5e-4 \
  --focal_alpha 0.75 --focal_gamma 2.0

# ── B: Stronger regularization (dropout=0.5, wd=1e-3) ─────────────────────────
echo "=== [2/5] Stronger reg ==="
python scripts/train.py $BASE \
  --run_name gen_B_stronger_reg \
  --dropout 0.5 --weight_decay 1e-3 \
  --focal_alpha 0.75 --focal_gamma 2.0

# ── C: Softer focal loss (alpha=0.65, gamma=1.5) ──────────────────────────────
# Lower alpha → less class-weighting pressure on positives;
# lower gamma → less focus on hard examples → smoother gradient signal.
echo "=== [3/5] Softer focal ==="
python scripts/train.py $BASE \
  --run_name gen_C_focal_soft \
  --dropout 0.4 --weight_decay 5e-4 \
  --focal_alpha 0.65 --focal_gamma 1.5

# ── D: Ablate code embeddings ──────────────────────────────────────────────────
# BERT dims are 768/776 of function node features — likely encodes
# repo-specific code style, which doesn't generalize across repos.
echo "=== [4/5] No code emb ==="
python scripts/train.py $BASE \
  --run_name gen_D_no_code_emb \
  --dropout 0.4 --weight_decay 5e-4 \
  --focal_alpha 0.75 --focal_gamma 2.0 \
  --ablate_code_emb

# ── E: Input feature dropout (feat_dropout=0.1) ───────────────────────────────
# Randomly zeros 10% of raw feature dims before projection each forward pass.
# Encourages the model not to rely on specific feature correlations.
echo "=== [5/5] Feature dropout 0.1 ==="
python scripts/train.py $BASE \
  --run_name gen_E_feat_dropout \
  --dropout 0.4 --weight_decay 5e-4 \
  --focal_alpha 0.75 --focal_gamma 2.0 \
  --feat_dropout 0.1

echo ""
echo "=== Sweep complete. Summary of val AUC-PR across runs: ==="
for run in gen_A_reg04 gen_B_stronger_reg gen_C_focal_soft gen_D_no_code_emb gen_E_feat_dropout; do
  csv="checkpoints/$run/metrics.csv"
  if [ -f "$csv" ]; then
    best=$(python - <<'PY' "$csv"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['split']=='val']
best = max(rows, key=lambda r: float(r['auc_pr']))
print(f"  val AUC-PR={best['auc_pr']}  ep={best['epoch']}  gap={round(float(best['auc_pr'])-float([r for r in csv.DictReader(open(sys.argv[1])) if r['split']=='train' and r['epoch']==best['epoch']][0]['auc_pr']),3)}")
PY
)
    echo "$run: $best"
  fi
done
