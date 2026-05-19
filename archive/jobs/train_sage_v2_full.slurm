#!/bin/bash
#SBATCH --job-name=sage_v2_full
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/sage_v2_full_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/sage_v2_full_%j.err

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

# Scratch for fast graph I/O (optional but recommended on A100 node)
SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs."

mkdir -p "${BASE_DIR}/logs"

cd "${THESIS_DIR}"

python scripts/train.py \
  --graphs_dir "${SCRATCH}" \
  --split_index "${SPLIT_INDEX}" \
  --perrepo_scaler "${SCALER}" \
  --run_name sage_v2_baseline_full \
  --output_dir "${BASE_DIR}/checkpoints" \
  --perrepo_norm \
  --epochs 100 \
  --patience 15 \
  --batch_size 256 \
  --hidden 128 \
  --dropout 0.3 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --warmup_epochs 5 \
  --focal_gamma 2.0 \
  --focal_alpha 0.75 \
  --ckpt_every 10 \
  --num_workers 4 \
  --seed 42

echo "Training complete."
