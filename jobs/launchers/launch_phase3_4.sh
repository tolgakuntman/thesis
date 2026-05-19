#!/bin/bash
# Phase 3 (temporal split) + Phase 4 (RGCN benchmark) launcher.
#
# Filled in after Phase 1+2 decision gates (2026-04-11).
#
# Best config from Phase 1+2:
#   - Structurally exclude: hunk, pull_request, release_tag
#   - Ablate: developer node features (keep dev edges)
#   = "combined_best": issues + dev-topology only
#
# Usage:
#   cd /data/leuven/380/vsc38046/Thesis/thesis/jobs/launchers
#   bash launch_phase3_4.sh

set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/../training/train_ablation.slurm"

# â”€â”€ Best config (from Phase 1+2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Exclude hunk (structural, Phase 1) + PR + release_tag (SDLC, Phase 2)
# + zero developer node features (keep dev edges = topology signal, Phase 2)
BEST_NODES="hunk pull_request release_tag"
BEST_RELS=""
BEST_ABLATE="--ablate_developer_feats"

# Phase 1 base (simpler, for comparison)
NOHUNK_NODES="hunk"
NOHUNK_ABLATE=""
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

submit() {
  local RUN_NAME="$1"
  local EXCLUDE_NODES="$2"
  local EXCLUDE_RELS="$3"
  local ABLATE_FLAGS="$4"
  local SPLIT_TYPE="$5"
  local SEED="${6:-42}"

  echo "Submitting: ${RUN_NAME} [${SPLIT_TYPE}]"
  sbatch \
    --job-name="${RUN_NAME}" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="${EXCLUDE_NODES}",EXCLUDE_RELS="${EXCLUDE_RELS}",SPLIT_TYPE="${SPLIT_TYPE}",SEED="${SEED}",ABLATE_FLAGS="${ABLATE_FLAGS}" \
    "${SLURM_SCRIPT}"
}

echo "========================================"
echo "Phase 3 â€” Temporal split evaluation"
echo "========================================"

# 3-1: Full model on temporal split (upper-bound reference)
submit "temp_full"          ""                           ""  ""                  "temporal_split"

# 3-2: Phase 1 best (no_hunk) on temporal split
submit "temp_no_hunk"       "${NOHUNK_NODES}"            ""  "${NOHUNK_ABLATE}"  "temporal_split"

# 3-3: Combined best config on temporal split
submit "temp_combined_best" "${BEST_NODES}"              ""  "${BEST_ABLATE}"    "temporal_split"

# 3-4: No SDLC (all 3 removed) on temporal split
submit "temp_no_sdlc"       "hunk issue pull_request release_tag"  ""  ""        "temporal_split"

# 3-5: No developer (structural) on temporal split
submit "temp_no_dev"        "hunk developer"             ""  ""                  "temporal_split"

# 3-6: No function BERT on temporal split (tests if BERT is still critical)
submit "temp_no_fn_bert"    "${NOHUNK_NODES}"            ""  "--ablate_code_emb" "temporal_split"

# 3-7: Issues only + dev-topo (Phase 2 winners combined) on temporal split
#      = BEST_NODES + BEST_ABLATE = temp_combined_best (already above, 3-3)

echo ""
echo "========================================"
echo "Phase 3 extra â€” Combined best on repo split"
echo "(validates the combination before temporal generalization claims)"
echo "========================================"

# Combined best on repo split â€” must beat both individual components
submit "repo_combined_best" "${BEST_NODES}"              ""  "${BEST_ABLATE}"    "repo_split"

echo ""
echo "========================================"
echo "Phase 4 â€” RGCN benchmark"
echo "========================================"

# 4-1: RGCN b4 on repo split with no_hunk base
submit "rgcn_b4_repo"       "${NOHUNK_NODES}"  ""  "--model rgcn --num_bases 4 --dropout 0.3 --weight_decay 1e-4"  "repo_split"

# 4-2: RGCN b8 on repo split with no_hunk base
submit "rgcn_b8_repo"       "${NOHUNK_NODES}"  ""  "--model rgcn --num_bases 8 --dropout 0.3 --weight_decay 1e-4"  "repo_split"

# 4-3: RGCN with best combined config â€” repo split
submit "rgcn_best_repo"     "${BEST_NODES}"    ""  "--model rgcn --num_bases 4 --dropout 0.3 --weight_decay 1e-4 ${BEST_ABLATE}"  "repo_split"

# 4-4: RGCN b4 with best config â€” temporal split (cross-architecture generalization test)
submit "rgcn_best_temporal" "${BEST_NODES}"    ""  "--model rgcn --num_bases 4 --dropout 0.3 --weight_decay 1e-4 ${BEST_ABLATE}"  "temporal_split"

echo ""
echo "All Phase 3+4 jobs submitted ($(( 6 + 1 + 4 )) jobs)."
echo "Monitor with: squeue -u \$USER"
echo "Logs in: /data/leuven/380/vsc38046/Thesis/logs/"
