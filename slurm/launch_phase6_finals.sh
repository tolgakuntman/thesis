#!/bin/bash
# Phase 6 — Multi-seed final confirmation runs.
#
# 4 thesis-final configurations × 2 splits × 3 seeds = up to 24 GPU jobs.
# In practice: 4 configs × repo_split × 3 seeds = 12 jobs (mandatory)
#              + full model × temporal_split × 3 seeds = 3 jobs (for temporal claim)
# Total: 15 jobs.
#
# Configs:
#   A) full_model     — best temporal; interpretable baseline
#   B) struct_no_hunk — best repo-split structural finding
#   C) dev_topo_only  — best Phase 2 granular finding
#   D) rgcn_b4        — architecture comparison (quantifies HeteroSAGE advantage)
#
# Usage:
#   cd /data/leuven/380/vsc38046/Thesis/thesis/slurm
#   bash launch_phase6_finals.sh
#
# IMPORTANT: If Phase 5 found a better config (val AUC-PR gain >=0.01),
# replace the relevant ABLATE_FLAGS below before running.

set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/train_ablation.slurm"

submit_multiseed() {
  local BASE_RUN_NAME="$1"
  local EXCLUDE_NODES="$2"
  local EXCLUDE_RELS="$3"
  local ABLATE_FLAGS="$4"
  local SPLIT_TYPE="$5"

  for SEED in 42 123 7; do
    local RUN_NAME="${BASE_RUN_NAME}_s${SEED}"
    echo "Submitting: ${RUN_NAME} [${SPLIT_TYPE}, seed=${SEED}]"
    sbatch \
      --job-name="${RUN_NAME}" \
      --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="${EXCLUDE_NODES}",EXCLUDE_RELS="${EXCLUDE_RELS}",SPLIT_TYPE="${SPLIT_TYPE}",SEED="${SEED}",ABLATE_FLAGS="${ABLATE_FLAGS}" \
      "${SLURM_SCRIPT}"
  done
}

echo "========================================"
echo "Phase 6 — Multi-seed final confirmation"
echo "Seeds: 42, 123, 7"
echo "========================================"

echo ""
echo "--- A) Full model (best temporal, interpretable baseline) ---"
# Repo split
submit_multiseed "final_full_repo"     ""      ""  ""  "repo_split"
# Temporal split (3 extra seeds for temporal claim)
submit_multiseed "final_full_temp"     ""      ""  ""  "temporal_split"

echo ""
echo "--- B) struct_no_hunk (best repo-split structural finding) ---"
submit_multiseed "final_nohunk_repo"   "hunk"  ""  ""  "repo_split"

echo ""
echo "--- C) dev_topo_only (best Phase 2 granular finding) ---"
submit_multiseed "final_devtopo_repo"  "hunk"  ""  "--ablate_developer_feats"  "repo_split"

echo ""
echo "--- D) RGCN b4 no_hunk (architecture comparison) ---"
submit_multiseed "final_rgcn_repo"     "hunk"  ""  "--model rgcn --num_bases 4 --dropout 0.3 --weight_decay 1e-4"  "repo_split"

echo ""
echo "--- E) Batch size 512 (Phase 5.1 bs=256 cleared threshold; test if 512 continues trend) ---"
RUN_NAME="hp_bs512"
echo "Submitting: ${RUN_NAME} [repo_split, seed=42]"
sbatch \
  --job-name="${RUN_NAME}" \
  --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="hunk",EXCLUDE_RELS="",SPLIT_TYPE="repo_split",SEED="42",ABLATE_FLAGS="--batch_size 512" \
  "${SLURM_SCRIPT}"

echo ""
echo "--- F) Code-before ablations (Phase 5.1 reruns, single seed=42) ---"
# These jobs failed in Phase 5.1 due to a path mismatch for the code-before embeddings.
# Path fix applied in src/graph_dataset.py (_resolve_data_root now tries both layouts).

# code_before + code_after (function.x: 776 -> 1544 dims): does before-state help?
RUN_NAME="abl_code_with_before"
echo "Submitting: ${RUN_NAME} [repo_split, seed=42]"
sbatch \
  --job-name="${RUN_NAME}" \
  --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="hunk",EXCLUDE_RELS="",SPLIT_TYPE="repo_split",SEED="42",ABLATE_FLAGS="--include_code_before" \
  "${SLURM_SCRIPT}"

# code_before only (zero code_after): is before-state more predictive than after-state?
RUN_NAME="abl_code_before_only"
echo "Submitting: ${RUN_NAME} [repo_split, seed=42]"
sbatch \
  --job-name="${RUN_NAME}" \
  --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="hunk",EXCLUDE_RELS="",SPLIT_TYPE="repo_split",SEED="42",ABLATE_FLAGS="--include_code_before --ablate_code_emb" \
  "${SLURM_SCRIPT}"

echo ""
echo "All Phase 6 final runs submitted."
echo "Results will be in: checkpoints/final_*_s{42,123,7}/, checkpoints/hp_bs512/, and checkpoints/abl_code_*/"
echo ""
echo "After all jobs complete, aggregate with:"
echo "  python scripts/aggregate_finals.py"
