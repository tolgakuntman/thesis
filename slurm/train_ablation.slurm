#!/bin/bash
# Parameterized SLURM template for structural ablation runs.
#
# Required env vars (set via sbatch --export or a launcher script):
#   RUN_NAME        - unique experiment name (e.g. struct_no_hunk)
#   EXCLUDE_NODES   - space-separated node types to exclude (or "" for none)
#   EXCLUDE_RELS    - space-separated edge rels to exclude (or "" for none)
#   SPLIT_TYPE      - repo_split or temporal_split
#   SEED            - random seed (e.g. 42)
#
# Optional env vars:
#   ABLATE_FLAGS    - any extra --ablate_* flags (e.g. "--ablate_sdlc")
#   EPOCHS          - override default 60
#   PATIENCE        - override default 10
#
# Example sbatch invocation:
#   RUN_NAME=struct_no_hunk EXCLUDE_NODES="hunk" EXCLUDE_RELS="" \
#   SPLIT_TYPE=repo_split SEED=42 \
#   sbatch train_ablation.slurm

#SBATCH --job-name=ablation
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/%x_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/%x_%j.err

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-ablation_run}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
EXCLUDE_RELS="${EXCLUDE_RELS:-}"
SPLIT_TYPE="${SPLIT_TYPE:-repo_split}"
SEED="${SEED:-42}"
ABLATE_FLAGS="${ABLATE_FLAGS:-}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-10}"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

# ── Conda ─────────────────────────────────────────────────────────────────────
source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate "icvul++"

# ── HuggingFace cache ─────────────────────────────────────────────────────────
export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

# ── torch_geometric ───────────────────────────────────────────────────────────
python -c "import torch_geometric" 2>/dev/null || {
  echo "Installing torch_geometric..."
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+''.join(v.split('.'))[:3] if v else 'cpu')")
  pip install torch_geometric --quiet
  pip install torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" --quiet
  echo "torch_geometric installed."
}

# ── Scratch copy ──────────────────────────────────────────────────────────────
SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs."

mkdir -p "${BASE_DIR}/logs"

cd "${THESIS_DIR}"

# ── Build exclusion flags ─────────────────────────────────────────────────────
EXCL_NODE_FLAG=""
if [ -n "${EXCLUDE_NODES}" ]; then
  EXCL_NODE_FLAG="--exclude_node_types ${EXCLUDE_NODES}"
fi

EXCL_REL_FLAG=""
if [ -n "${EXCLUDE_RELS}" ]; then
  EXCL_REL_FLAG="--exclude_edge_rels ${EXCLUDE_RELS}"
fi

# ── Training ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "RUN_NAME    : ${RUN_NAME}"
echo "EXCLUDE_NODES: ${EXCLUDE_NODES:-<none>}"
echo "EXCLUDE_RELS : ${EXCLUDE_RELS:-<none>}"
echo "SPLIT_TYPE  : ${SPLIT_TYPE}"
echo "SEED        : ${SEED}"
echo "EPOCHS      : ${EPOCHS}"
echo "PATIENCE    : ${PATIENCE}"
echo "========================================"

python scripts/train.py \
  --graphs_dir   "${SCRATCH}" \
  --split_index  "${SPLIT_INDEX}" \
  --perrepo_scaler "${SCALER}" \
  --run_name     "${RUN_NAME}" \
  --output_dir   "${BASE_DIR}/checkpoints" \
  --split_type   "${SPLIT_TYPE}" \
  --seed         "${SEED}" \
  --perrepo_norm \
  --epochs       "${EPOCHS}" \
  --patience     "${PATIENCE}" \
  --warmup_epochs 3 \
  --batch_size   128 \
  --hidden       128 \
  --dropout      0.3 \
  --lr           1e-3 \
  --weight_decay 5e-4 \
  --focal_gamma  1.5 \
  --focal_alpha  0.65 \
  --ckpt_every   10 \
  --num_workers  4 \
  ${EXCL_NODE_FLAG} \
  ${EXCL_REL_FLAG} \
  ${ABLATE_FLAGS}

echo "Training complete: ${RUN_NAME}"
