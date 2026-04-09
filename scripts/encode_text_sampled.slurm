#!/bin/bash

#SBATCH --job-name=encode_text_sampled

#SBATCH --account=lp_ideasgrouptproject

#SBATCH --clusters=wice

#SBATCH --partition=gpu_a100

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=16G

#SBATCH --gpus-per-node=1

#SBATCH --time=08:00:00

#SBATCH --output=logs/encode_text_sampled_%j.out

#SBATCH --error=logs/encode_text_sampled_%j.err


echo "=========================================="
echo "Text Encoding — Sampled (all-mpnet-base-v2)"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

BASE_DIR="/data/leuven/380/vsc38046/Thesis"

set -euo pipefail
mkdir -p "${BASE_DIR}/logs"

module purge
module load cluster/wice/gpu
module load Miniconda3
source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh


eval "$(conda shell.bash hook)"
conda activate "icvul++"

export PYTHONIOENCODING=utf-8
export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "Python:  $(which python)"
echo "Device check:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

# =============================================================================
# STEP 1 — Preprocess (creates *_clean.csv in data_new/processed/sampled/)
# =============================================================================
echo "STEP 1: Preprocessing"
echo "Start: $(date)"

python "${BASE_DIR}/src/text_encoding/preprocess_text_sampled.py"
echo "Preprocessing done: $(date)"
echo ""

# =============================================================================
# STEP 2 — Encode with SentenceTransformer
# =============================================================================
echo "STEP 2: Encoding"
echo "Start: $(date)"

python "${BASE_DIR}/src/text_encoding/encode_text_sampled.py"
TEXT_EXIT=$?

echo "Exit code: $TEXT_EXIT"
echo "End: $(date)"
echo ""

echo "Output files:"
ls -lh "${BASE_DIR}/data_new/processed/sampled/"*embeddings* 2>/dev/null || echo "  (none found)"

exit $TEXT_EXIT
