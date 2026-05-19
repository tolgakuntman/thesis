#!/bin/bash
#SBATCH --job-name=kfold_reg
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/kfold_reg_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/kfold_reg_%j.err

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

# Conda
source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate "icvul++"

# HuggingFace cache (avoid home-dir quota)
export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

# Ensure torch_geometric is installed (no-op if already present)
python -c "import torch_geometric" 2>/dev/null || {
  echo "Installing torch_geometric..."
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+''.join(v.split('.'))[:3] if v else 'cpu')")
  pip install torch_geometric --quiet
  pip install torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" --quiet
  echo "torch_geometric installed."
}

# Copy graphs to node-local scratch for fast I/O during k folds
SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs in scratch."

mkdir -p "${BASE_DIR}/logs"

cd "${THESIS_DIR}"

# ── Regularized k-fold cross-validation ─────────────────────────────────────
# Changes from kfold5_sage_repo baseline:
#   dropout         0.4 → 0.5    (stronger dropout)
#   weight_decay    5e-4 → 2e-3  (stronger weight decay)
#   ablate_hunk_metrics           (zero hunk dims 0-1: complexity + token_count)
#     SHAP justification: per-feature phi=0.0011 and 0.0004 — effectively zero.
#     Dims 2-769 (hunk BERT, phi~0.086 group) are kept intact.
# All other node types/features retained: SHAP shows meaningful signal in every
# remaining group (file_feats>fn_bert>hunk_bert>fn_metrics>developer>sdlc).

python scripts/kfold_train.py \
  --graphs_dir        "${SCRATCH}" \
  --split_index       "${SPLIT_INDEX}" \
  --perrepo_scaler    "${SCALER}" \
  --output_dir        "${BASE_DIR}/checkpoints" \
  --run_name          kfold5_sage_reg \
  --perrepo_norm \
  --ablate_hunk_metrics \
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

echo "Regularized k-fold training complete."
