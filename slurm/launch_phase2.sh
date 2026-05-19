#!/bin/bash
# Phase 2 fine-grained ablation sweep launcher.
#
# All runs use struct_no_hunk as the base (exclude hunk structurally).
# Run from the thesis/slurm/ directory on VSC:
#   cd /data/leuven/380/vsc38046/Thesis/thesis/slurm
#   bash launch_phase2.sh
#
# Sections:
#   2A - Code-side decomposition (fn BERT vs code metrics)
#   2C - SDLC component decomposition (issues / PRs / tags individually)
#   2D - Developer decomposition (node features vs edge topology)

set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/train_ablation.slurm"

# Base exclusion: always exclude hunk (our Phase 1 best)
BASE_NODES="hunk"

submit() {
  local RUN_NAME="$1"
  local EXTRA_NODES="$2"     # additional node types to exclude beyond hunk
  local EXTRA_RELS="$3"      # edge rels to exclude
  local ABLATE_FLAGS="$4"
  local SPLIT_TYPE="${5:-repo_split}"
  local SEED="${6:-42}"

  # Combine base exclusions with run-specific ones
  local EXCLUDE_NODES="${BASE_NODES}"
  [ -n "${EXTRA_NODES}" ] && EXCLUDE_NODES="${EXCLUDE_NODES} ${EXTRA_NODES}"

  echo "Submitting: ${RUN_NAME}"
  sbatch \
    --job-name="${RUN_NAME}" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="${EXCLUDE_NODES}",EXCLUDE_RELS="${EXTRA_RELS}",SPLIT_TYPE="${SPLIT_TYPE}",SEED="${SEED}",ABLATE_FLAGS="${ABLATE_FLAGS}" \
    "${SLURM_SCRIPT}"
}

echo "========================================"
echo "Phase 2A — Code-side feature decomposition"
echo "(base: struct_no_hunk)"
echo "========================================"

# No function BERT embedding — tests if BERT is truly the critical code signal
submit "p2_no_fn_bert"        ""  ""  "--ablate_code_emb"

# No function code metrics — tests if numeric code features add anything beyond BERT
submit "p2_no_fn_metrics"     ""  ""  "--ablate_fn_code_metrics"

# No commit message embedding
submit "p2_no_msg_emb"        ""  ""  "--ablate_msg_emb"

echo ""
echo "========================================"
echo "Phase 2C — SDLC component decomposition"
echo "(base: struct_no_hunk; full SDLC removal = struct_no_sdlc, already done)"
echo "========================================"

# Individual SDLC component removals
submit "p2_no_issues"         "issue"                    ""  ""
submit "p2_no_prs"            "pull_request"             ""  ""
submit "p2_no_tags"           "release_tag"              ""  ""

# Pairwise SDLC removals (to isolate which component drives the SDLC effect)
submit "p2_no_issues_prs"     "issue pull_request"       ""  ""
submit "p2_no_issues_tags"    "issue release_tag"        ""  ""
submit "p2_no_prs_tags"       "pull_request release_tag" ""  ""

echo ""
echo "========================================"
echo "Phase 2D — Developer decomposition"
echo "(base: struct_no_hunk; full dev removal = struct_no_dev, already done)"
echo "========================================"

# Zero only developer node features — keep edge topology
# (tests whether graph connectivity, not node attributes, carries dev signal)
submit "p2_dev_topo_only"     ""  ""  "--ablate_developer_feats"

# Remove developer edges, keep developer node features
# (tests whether node features, not connectivity, carry dev signal)
# Achieved by excluding all dev-touching edge relations
submit "p2_dev_feats_only"    ""  "authored_by authored committed_by committed owns owned_by"  ""

echo ""
echo "All Phase 2 jobs submitted."
echo "Monitor with: squeue -u \$USER"
echo "Logs in: /data/leuven/380/vsc38046/Thesis/logs/"
