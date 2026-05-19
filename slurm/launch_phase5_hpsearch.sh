#!/bin/bash
# Phase 5 — Bounded hyperparameter search on struct_no_hunk (best repo-split config).
#
# Axes searched (plan §9):
#   dropout: 0.2, 0.4, 0.5   (current best: 0.3)
#   weight_decay: 1e-5, 1e-4, 1e-3  (current best: 5e-4)
#   lr: 5e-4, 3e-3            (current best: 1e-3)
#   focal_alpha: 0.5, 0.75    (current best: 0.65)
#   focal_gamma: 1.0, 2.0     (current best: 1.5)
#   hidden: 64, 96            (current best: 128)
#
# Each run: repo_split, seed=42, --exclude_node_types hunk, all other defaults.
# Accept a tuned config only if val AUC-PR improves >=0.01 over struct_no_hunk (0.7435).
#
# Usage:
#   cd /data/leuven/380/vsc38046/Thesis/thesis/slurm
#   bash launch_phase5_hpsearch.sh

set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/train_ablation.slurm"
BASE_NODES="hunk"

submit() {
  local RUN_NAME="$1"
  local ABLATE_FLAGS="$2"
  echo "Submitting: ${RUN_NAME}"
  sbatch \
    --job-name="${RUN_NAME}" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="${BASE_NODES}",EXCLUDE_RELS="",SPLIT_TYPE="repo_split",SEED="42",ABLATE_FLAGS="${ABLATE_FLAGS}" \
    "${SLURM_SCRIPT}"
}

echo "Phase 5 — HP search on struct_no_hunk base"
echo "Reference val AUC-PR to beat: 0.7435  (struct_no_hunk, seed=42)"
echo ""

# --- Dropout ---
submit "hp_drop02" "--dropout 0.2"
submit "hp_drop04" "--dropout 0.4"
submit "hp_drop05" "--dropout 0.5"

# --- Weight decay ---
submit "hp_wd1e5"  "--weight_decay 1e-5"
submit "hp_wd1e4"  "--weight_decay 1e-4"
submit "hp_wd1e3"  "--weight_decay 1e-3"

# --- Learning rate ---
submit "hp_lr5e4"  "--lr 5e-4"
submit "hp_lr3e3"  "--lr 3e-3"

# --- Focal loss ---
submit "hp_fa05"   "--focal_alpha 0.5"
submit "hp_fa075"  "--focal_alpha 0.75"
submit "hp_fg10"   "--focal_gamma 1.0"
submit "hp_fg20"   "--focal_gamma 2.0"

# --- Hidden dimension ---
submit "hp_h64"    "--hidden 64"
submit "hp_h96"    "--hidden 96"

echo ""
echo "All Phase 5 HP search jobs submitted (14 jobs)."
echo "After reviewing val AUC-PR in checkpoints/, run launch_phase5_top3.sh"
echo "with the 3 best candidates for full training confirmation."
