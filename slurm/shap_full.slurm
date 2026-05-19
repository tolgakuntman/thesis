#!/bin/bash
#SBATCH --job-name=shap_full
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/shap_full_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/shap_full_%j.err

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
CKPT="${BASE_DIR}/checkpoints/kfold5_sage_repo/fold_0/best.pt"
OUT_DIR="${BASE_DIR}/outputs/shap_kfold_fold0_full"

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
}

mkdir -p "${BASE_DIR}/logs"

cd "${THESIS_DIR}"

# Full SHAP analysis on fold_0 checkpoint (best val AUC-PR=0.8977)
# Uses entire test split (3,375 graphs): FFmpeg, gpac, suricata, openssl, redis, envoy
# --full_features: gradient x input for all 50 named dims + 2 BERT blocks
# --per_feature_shap: exact Shapley per scalar dim (256-512 coalitions each group)

python scripts/shap_analysis.py \
  --checkpoint   "${CKPT}" \
  --output_dir   "${OUT_DIR}" \
  --split        test \
  --split_type   repo_split \
  --full_features \
  --per_feature_shap \
  --n_kernel_samples 500 \
  --batch_size   128

echo "Full SHAP analysis complete. Outputs in ${OUT_DIR}"
