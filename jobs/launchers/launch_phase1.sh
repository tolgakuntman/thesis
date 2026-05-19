#!/bin/bash
# Phase 1 structural ablation sweep launcher.
#
# Submits all Phase 1 runs to the SLURM queue using train_ablation.slurm.
# Run from the thesis/jobs/launchers/ directory on VSC:
#   cd /data/leuven/380/vsc38046/Thesis/thesis/jobs/launchers
#   bash launch_phase1.sh
#
# Each job name = run name, so logs are identifiable.
# All runs use: repo_split, seed=42, frozen protocol.

set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/../training/train_ablation.slurm"

submit() {
  local RUN_NAME="$1"
  local EXCLUDE_NODES="$2"
  local EXCLUDE_RELS="$3"
  local ABLATE_FLAGS="${4:-}"
  local SPLIT_TYPE="${5:-repo_split}"
  local SEED="${6:-42}"

  echo "Submitting: ${RUN_NAME}"
  sbatch \
    --job-name="${RUN_NAME}" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="${EXCLUDE_NODES}",EXCLUDE_RELS="${EXCLUDE_RELS}",SPLIT_TYPE="${SPLIT_TYPE}",SEED="${SEED}",ABLATE_FLAGS="${ABLATE_FLAGS}" \
    "${SLURM_SCRIPT}"
}

echo "========================================"
echo "Phase 1A â€” Node-type structural ablations"
echo "========================================"

# 1A-1: Remove hunk nodes (hypothesis: dead weight / redundant with function embedding)
submit "struct_no_hunk"    "hunk"                                     ""  ""

# 1A-2: Remove SDLC nodes (hypothesis: noisy under repo split)
submit "struct_no_sdlc"    "issue pull_request release_tag"           ""  ""

# 1A-3: Remove developer nodes (hypothesis: calibration only, not core signal)
submit "struct_no_dev"     "developer"                                ""  ""

# 1A-4: Remove all process nodes (hypothesis: code side generalizes better)
submit "struct_no_process" "developer issue pull_request release_tag" ""  ""

# 1A-5: Minimal graph â€” only commit + file + function
submit "struct_minimal"    "hunk developer issue pull_request release_tag" ""  ""

echo ""
echo "========================================"
echo "Phase 1B â€” Edge-type structural ablations"
echo "========================================"

# 1B-1: Remove commit<->function edges (tests if commit-fn is main code signal)
submit "struct_no_commit_fn"   ""  "modifies_func in_commit_fn"  ""

# 1B-2: Remove commit<->file edges (tests if file path is redundant)
submit "struct_no_commit_file" ""  "modifies_file in_commit"     ""

# 1B-3: Remove file<->function hierarchy
submit "struct_no_file_fn"     ""  "contains in_file"            ""

echo ""
echo "========================================"
echo "Phase 1C â€” Baselines (zeroing, for comparison with structural)"
echo "========================================"

# Full model (baseline for this phase â€” frozen protocol)
submit "phase1_full_baseline"  ""  ""  ""

echo ""
echo "========================================"
echo "Phase 1C â€” No-graph MLP null baseline"
echo "========================================"

# MLP baseline: no message passing at all â€” pure commit node features
echo "Submitting: struct_mlp_baseline"
sbatch \
  --job-name="struct_mlp_baseline" \
  --export=ALL,RUN_NAME="struct_mlp_baseline",EXCLUDE_NODES="",EXCLUDE_RELS="",SPLIT_TYPE="repo_split",SEED="42",ABLATE_FLAGS="--model mlp" \
  "${SLURM_SCRIPT}"

echo ""
echo "All Phase 1 jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs in: /data/leuven/380/vsc38046/Thesis/logs/"
