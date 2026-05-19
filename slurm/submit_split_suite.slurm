#!/bin/bash
#SBATCH --job-name=split_suite
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/split_suite_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/split_suite_%j.err

# Split-test experiment suite.
# Runs all 11 split strategies sequentially (no random split).
# Each split trains a HeteroSAGE model using the final protocol.
#
# Total estimated time: ~3-6h depending on split sizes.
# Generates: checkpoints/split_suite_*/  +  combined results table.

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"
SPLITS_OUT="${BASE_DIR}/outputs/splits"
CKPT_DIR="${BASE_DIR}/checkpoints"
LOG_DIR="${BASE_DIR}/logs"

source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate "icvul++"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}" "${LOG_DIR}" "${SPLITS_OUT}" "${CKPT_DIR}"

# Install torch_geometric if missing (first-run on a new node)
python -c "import torch_geometric" 2>/dev/null || {
  echo "Installing torch_geometric..."
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=$(python -c "import torch; v=torch.version.cuda or ''; print('cu'+''.join(v.split('.'))[:3] if v else 'cpu')")
  pip install torch_geometric --quiet
  pip install torch_scatter torch_sparse torch_cluster \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html" --quiet
}

# Copy graphs to local scratch for I/O speed
SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls "${SCRATCH}" | wc -l) graphs in scratch."

cd "${THESIS_DIR}"

# ── Step 1: Generate all splits ───────────────────────────────────────────────
echo "=== Step 1: Generating splits ==="
python scripts/generate_splits.py \
  --split_index "${SPLIT_INDEX}" \
  --out_dir     "${SPLITS_OUT}" \
  --seed        42 \
  --continue_on_error

# ── Step 2: Validate splits ───────────────────────────────────────────────────
echo "=== Step 2: Validating splits ==="
python scripts/validate_splits.py \
  --splits_dir "${SPLITS_OUT}" \
  --no_strict  # report but don't abort on non-critical checks

# ── Step 3: Run training for all splits ──────────────────────────────────────
echo "=== Step 3: Running split suite ==="
python scripts/run_split_suite.py \
  --splits_dir    "${SPLITS_OUT}" \
  --output_dir    "${CKPT_DIR}" \
  --graphs_dir    "${SCRATCH}" \
  --perrepo_scaler "${SCALER}" \
  --run_name      split_suite \
  --model         sage \
  --hidden        128 \
  --dropout       0.5 \
  --epochs        60 \
  --batch_size    128 \
  --lr            1e-3 \
  --weight_decay  2e-3 \
  --patience      10 \
  --warmup_epochs 3 \
  --focal_gamma   1.5 \
  --focal_alpha   0.65 \
  --seed          42 \
  --num_workers   4 \
  --perrepo_norm \
  --ablate_hunk_metrics \
  --ablate_commit_merge \
  --skip_existing \
  --continue_on_error

echo "Split suite complete."
echo "Results: ${CKPT_DIR}/split_suite_results.csv"
