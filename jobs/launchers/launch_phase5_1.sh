#!/bin/bash
# Phase 5.1 â€” Extended HP combinations + batch size + change representation ablations.
#
# Three axes:
#   A) HP combinations (6 jobs): pairs/triples of close-to-threshold Phase 5 winners
#      - lr=3e-3 (was +0.007 val) and hidden=64 (was +0.006 val) showed promise alone.
#        Test their interactions with focal calibration and regularization changes.
#
#   B) Batch size tuning (2 jobs): bs=64, bs=256 vs current bs=128
#      - Larger batches: faster, smoother gradients; smaller: noisier but can help minority class
#
#   C) Change representation ablations (5 jobs): how much does the change encoding matter?
#      - ablate_fn_categorical : remove fct_* one-hot (what kind of change: MODIFY/ADD/etc.)
#      - ablate_file_code_metrics : remove file-level change volume (lines added/deleted)
#      - ablate_commit_stats   : remove DMM change quality metrics (dmm_size, dmm_cmplx, dmm_iface)
#      - all three combined    : full change representation stripped
#      - fn_categorical + commit_stats (no file metrics, to isolate file vs semantic change info)
#
# All Phase 5.1 runs: repo_split, seed=42, struct_no_hunk base (--exclude_node_types hunk)
# Accept only if val AUC-PR improves >=0.01 over struct_no_hunk reference (val=0.7435).
#
# Usage:
#   cd /data/leuven/380/vsc38046/Thesis/thesis/jobs/launchers
#   bash launch_phase5_1.sh

set -euo pipefail

SLURM_SCRIPT="$(dirname "$0")/../training/train_ablation.slurm"
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

submit_bs() {
  local RUN_NAME="$1"
  local ABLATE_FLAGS="$2"
  echo "Submitting: ${RUN_NAME}"
  sbatch \
    --job-name="${RUN_NAME}" \
    --export=ALL,RUN_NAME="${RUN_NAME}",EXCLUDE_NODES="${BASE_NODES}",EXCLUDE_RELS="",SPLIT_TYPE="repo_split",SEED="42",ABLATE_FLAGS="${ABLATE_FLAGS}" \
    "${SLURM_SCRIPT}"
}

echo "Phase 5.1 â€” HP combinations + batch size + change representation ablations"
echo "Reference val AUC-PR to beat: 0.7435  (struct_no_hunk, seed=42)"
echo ""

# â”€â”€ A) HP combinations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "--- A) HP combinations ---"

# lr=3e-3 was closest (+0.007); pair with focal_alpha=0.5 (less VCC emphasis, better precision)
submit "hp_c_lr3e3_fa05"    "--lr 3e-3 --focal_alpha 0.5"

# lr=3e-3 with more regularization (dropout=0.4 was neutral in Phase 5; together might generalize)
submit "hp_c_lr3e3_drop04"  "--lr 3e-3 --dropout 0.4"

# focal_alpha=0.5 with dropout=0.4 (calibration + regularization without touching lr)
submit "hp_c_fa05_drop04"   "--focal_alpha 0.5 --dropout 0.4"

# hidden=64 (+0.006) with focal_alpha=0.5 (smaller model tends to benefit from less focal pressure)
submit "hp_c_h64_fa05"      "--hidden 64 --focal_alpha 0.5"

# lr=3e-3 + hidden=64 (both moved val positively; test if they reinforce or cancel)
submit "hp_c_lr3e3_h64"     "--lr 3e-3 --hidden 64"

# Triple: lr=3e-3 + dropout=0.4 + focal_alpha=0.5 (maximum calibration shift)
submit "hp_c_lr3e3_drop04_fa05"  "--lr 3e-3 --dropout 0.4 --focal_alpha 0.5"

echo ""

# â”€â”€ B) Batch size tuning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "--- B) Batch size ---"

# bs=64: smaller batches â†’ noisier gradients, may escape local minima; helps minority-class recall
submit_bs "hp_bs64"   "--batch_size 64"

# bs=256: larger batches â†’ smoother loss landscape, lower wall time; may hurt minority recall
submit_bs "hp_bs256"  "--batch_size 256"

echo ""

# â”€â”€ C) Change representation ablations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "--- C) Change representation ablations ---"

# Remove fct_* one-hot on commitâ†’fn edges (change type: MODIFY/ADD/DELETE/RENAME/REFACTOR)
# Tests: does knowing *what kind* of change happened help predict VCC?
submit "abl_chg_fn_cat"     "--ablate_fn_categorical"

# Remove file-level change volume from file.x dims 0-2 (lines_added, lines_deleted, complexity_delta)
# Tests: does the *size* of file-level changes matter beyond function-level signals?
submit "abl_chg_file_met"   "--ablate_file_code_metrics"

# Remove DMM change quality metrics from commit.x dims 2-4 (dmm_size, dmm_cmplx, dmm_iface)
# Tests: do commit-level diffusion metrics (change quality proxies) help VCC detection?
submit "abl_chg_commit_dmm" "--ablate_commit_stats"

# All three change signals stripped simultaneously
# Tests: pure structural + embedding model â€” no explicit change encoding at all
submit "abl_chg_all"        "--ablate_fn_categorical --ablate_file_code_metrics --ablate_commit_stats"

# fn_categorical + commit_stats only (keeps file volume metrics)
# Separates semantic change encoding from volumetric file change signal
submit "abl_chg_no_file"    "--ablate_fn_categorical --ablate_commit_stats"

# â”€â”€ D) Code before / after / delta ablations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "--- D) Code before/after/delta ablations ---"

# Baseline: struct_no_hunk uses only code_after embedding (dims 8-775 of function.x).
# These jobs test what each component of the code representation contributes.

# Add code_before embedding (function.x: 776 â†’ 1544 dims). Does BEFORE state help?
# Compare val AUC-PR vs baseline 0.7435 to assess whether before-state knowledge helps.
submit "abl_code_with_before"    "--include_code_before"

# code_before only (no code_after): zero the after embedding, keep only before state.
# Tests: is the function state BEFORE the change more predictive than AFTER?
submit "abl_code_before_only"    "--include_code_before --ablate_code_emb"

# No code embeddings at all (zero both after emb AND current-state metrics).
# Tests: can the model predict VCC from structural + change signals alone (no code content)?
submit "abl_delta_only"          "--ablate_code_emb --ablate_fn_code_metrics"

# Code state but no change signals: keep code_after emb + state metrics, zero all change repr.
# Tests: does knowing HOW something changed help beyond knowing WHAT the code looks like?
submit "abl_code_no_delta"       "--ablate_fn_categorical --ablate_file_code_metrics --ablate_commit_stats"

# Pure structural: no code, no change signals. Only graph topology + developer/SDLC.
# Tests: absolute lower bound of non-code information for VCC detection.
submit "abl_pure_structural"     "--ablate_code_emb --ablate_fn_code_metrics --ablate_fn_categorical --ablate_file_code_metrics --ablate_commit_stats"

echo ""
echo "All Phase 5.1 jobs submitted (20 jobs total: 6 HP combos + 2 batch + 5 change ablations + 5 code before/after/delta)."
echo "After reviewing val AUC-PR in checkpoints/hp_c_* and hp_bs* and abl_chg_* and abl_code_*:"
echo "  - If any hp_c_* beats 0.7435 by >=0.01: update Phase 6 launch_phase6_finals.sh with winning config"
echo "  - abl_chg_* + abl_code_* feed Phase 6 thesis narrative (change representation contribution)"
echo ""
echo "Aggregate all Phase 5.1 checkpoints with:"
echo "  python scripts/evaluation/aggregate_finals.py --prefix hp_c_  --output checkpoints/phase5_1_hp_summary.json"
echo "  python scripts/evaluation/aggregate_finals.py --prefix abl_chg_ --output checkpoints/phase5_1_abl_summary.json"
echo "  python scripts/evaluation/aggregate_finals.py --prefix abl_code_ --output checkpoints/phase5_1_code_abl_summary.json"
echo "  python scripts/evaluation/aggregate_finals.py --prefix abl_      --output checkpoints/phase5_1_all_abl_summary.json"
