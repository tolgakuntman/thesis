#!/bin/bash
# Quick smoke test — 2 epochs each for structural exclusion, temporal split,
# RGCN, and MLP baseline. Verifies the model builds and produces metrics
# before submitting the full Phase 1 sweep.
#
# Run interactively or as a short batch job (1-2 min on CPU):
#   sbatch smoke_test.slurm
#   # or on a login node (small test only):
#   bash smoke_test.slurm

#SBATCH --job-name=smoke_test
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/smoke_test_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/smoke_test_%j.err

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate "icvul++"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}" "${BASE_DIR}/logs"

python -c "import torch_geometric" 2>/dev/null || {
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+''.join(v.split('.'))[:3] if v else 'cpu')")
  pip install torch_geometric --quiet
  pip install torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" --quiet
}

cd "${THESIS_DIR}"

COMMON="--graphs_dir ${GRAPHS_DIR} --split_index ${SPLIT_INDEX} --perrepo_scaler ${SCALER} \
  --output_dir ${BASE_DIR}/checkpoints --perrepo_norm --epochs 2 --patience 999 \
  --batch_size 128 --num_workers 4 --ckpt_every 0"

echo "========================================"
echo "SMOKE TEST 1: structural exclusion (no hunk)"
echo "========================================"
python scripts/train.py ${COMMON} \
  --run_name smoke_no_hunk \
  --exclude_node_types hunk \
  --split_type repo_split --seed 42

echo "========================================"
echo "SMOKE TEST 2: temporal split"
echo "========================================"
python scripts/train.py ${COMMON} \
  --run_name smoke_temporal \
  --split_type temporal_split --seed 42

echo "========================================"
echo "SMOKE TEST 3: RGCN"
echo "========================================"
python scripts/train.py ${COMMON} \
  --run_name smoke_rgcn \
  --model rgcn --num_bases 4 \
  --split_type repo_split --seed 42

echo "========================================"
echo "SMOKE TEST 4: MLP no-graph baseline"
echo "========================================"
python scripts/train.py ${COMMON} \
  --run_name smoke_mlp \
  --model mlp \
  --split_type repo_split --seed 42

echo "========================================"
echo "SMOKE TEST 5: multi-seed (seeds 42 123)"
echo "========================================"
python scripts/train.py ${COMMON} \
  --run_name smoke_multiseed \
  --seeds 42 123 \
  --split_type repo_split

echo ""
echo "All smoke tests passed."
