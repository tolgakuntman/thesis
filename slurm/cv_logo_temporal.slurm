#!/bin/bash
#SBATCH --job-name=cv_logo_temp
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/cv_logo_temp_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/cv_logo_temp_%j.err

# Runs two cross-validation experiments sequentially in a single job:
#
#   1. LOGO-10  (Leave-One-Repo-Out, top-10 repos from the repo-split pool)
#      Pool  = repo_split != 'test'       Test = repo_split == 'test'
#      Fold i: val = top-repo-i, train = all 30k other pool commits
#      Motivation: trains on maximum data; measures per-repo generalisation
#                  free of the greedy-balanced fold-assignment artefact.
#
#   2. kfold-5-temporal  (walk-forward temporal cross-validation)
#      Pool  = temporal_split in ['train','val']  Test = temporal_split == 'test'
#      10 date-sorted chunks; fold i: train=chunks[0..4+i], val=chunk[5+i]
#      Minimum training fraction: 50 % of pool (14 k commits for fold 0).
#      Motivation: validation is STRICTLY LATER than training → no temporal
#                  leakage present in random repo-grouped k-fold.
#
# Estimated wall-clock: ~5 h (LOGO) + ~2 h (temporal) = ~7 h total.
# Increase --time to 48:00:00 if you observe a timeout.

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

# Conda
source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate "icvul++"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

python -c "import torch_geometric" 2>/dev/null || {
  echo "Installing torch_geometric..."
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+''.join(v.split('.'))[:3] if v else 'cpu')")
  pip install torch_geometric --quiet
  pip install torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" --quiet
  echo "torch_geometric installed."
}

# Copy graphs to node-local scratch (shared by both experiments)
SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs in scratch."

mkdir -p "${BASE_DIR}/logs"
cd "${THESIS_DIR}"

# ── Experiment 1: LOGO-10 ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "EXPERIMENT 1: LOGO-10 (leave-one-repo-out, top 10 repos)"
echo "============================================================"

python scripts/cv_train.py \
  --mode           logo \
  --k              10 \
  --graphs_dir     "${SCRATCH}" \
  --split_index    "${SPLIT_INDEX}" \
  --perrepo_scaler "${SCALER}" \
  --output_dir     "${BASE_DIR}/checkpoints" \
  --run_name       logo10_sage_repo \
  --perrepo_norm \
  --seed           42 \
  --model          sage \
  --hidden         128 \
  --dropout        0.4 \
  --epochs         60 \
  --patience       10 \
  --batch_size     128 \
  --lr             1e-3 \
  --weight_decay   5e-4 \
  --warmup_epochs  3 \
  --focal_gamma    1.5 \
  --focal_alpha    0.65 \
  --ckpt_every     0 \
  --num_workers    4

echo "LOGO-10 complete."

# ── Experiment 2: temporal k-fold-5 ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "EXPERIMENT 2: temporal k-fold-5 (walk-forward)"
echo "============================================================"

python scripts/cv_train.py \
  --mode           temporal \
  --k              5 \
  --graphs_dir     "${SCRATCH}" \
  --split_index    "${SPLIT_INDEX}" \
  --perrepo_scaler "${SCALER}" \
  --output_dir     "${BASE_DIR}/checkpoints" \
  --run_name       kfold5_sage_temporal \
  --perrepo_norm \
  --seed           42 \
  --model          sage \
  --hidden         128 \
  --dropout        0.4 \
  --epochs         60 \
  --patience       10 \
  --batch_size     128 \
  --lr             1e-3 \
  --weight_decay   5e-4 \
  --warmup_epochs  3 \
  --focal_gamma    1.5 \
  --focal_alpha    0.65 \
  --ckpt_every     0 \
  --num_workers    4

echo "Temporal k-fold-5 complete."
echo ""
echo "All experiments done. Results in ${BASE_DIR}/checkpoints/"
