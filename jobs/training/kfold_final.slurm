#!/bin/bash
#SBATCH --job-name=kfold_final
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/kfold_final_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/kfold_final_%j.err

# Final regularized + SHAP-ablated k-fold run.
#
# Combines three orthogonal improvements over the kfold5_sage_repo baseline:
#
#   1. Stronger regularization
#      dropout      0.4  → 0.5
#      weight_decay 5e-4 → 2e-3
#
#   2. SHAP-justified feature ablations (exact Shapley + gradient×input,
#      both methods agree the ablated features are near-zero):
#      --ablate_hunk_metrics  zero hunk dims 0-1 (complexity, token_count)
#                             phi=0.0011/0.0004, grad=0.00098/0.00038
#      --ablate_commit_merge  zero commit dim 1 (merge flag)
#                             phi=0.0015, grad=0.000028
#
# All other hyperparameters unchanged from the baseline.
# Goal: reduce train-val AUC-PR gap (baseline ~0.131) without hurting test AUC-PR.

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

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
}

SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs in scratch."

mkdir -p "${BASE_DIR}/logs"
cd "${THESIS_DIR}"

python scripts/training/kfold_train.py \
  --graphs_dir        "${SCRATCH}" \
  --split_index       "${SPLIT_INDEX}" \
  --perrepo_scaler    "${SCALER}" \
  --output_dir        "${BASE_DIR}/checkpoints" \
  --run_name          kfold5_sage_final \
  --perrepo_norm \
  --ablate_hunk_metrics \
  --ablate_commit_merge \
  --k                 5 \
  --fold_seed         0 \
  --seed              42 \
  --model             sage \
  --hidden            128 \
  --dropout           0.5 \
  --epochs            60 \
  --patience          10 \
  --batch_size        128 \
  --lr                1e-3 \
  --weight_decay      2e-3 \
  --warmup_epochs     3 \
  --focal_gamma       1.5 \
  --focal_alpha       0.65 \
  --ckpt_every        0 \
  --num_workers       4

echo "kfold5_sage_final complete."
