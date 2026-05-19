#!/bin/bash

#SBATCH --job-name=encode_code_sampled

#SBATCH --account=lp_ideasgrouptproject

#SBATCH --clusters=genius

#SBATCH --partition=gpu_v100

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --gpus-per-node=1

#SBATCH --time=08:00:00

#SBATCH --output=logs/encode_code_sampled_%j.out

#SBATCH --error=logs/encode_code_sampled_%j.err


echo "=========================================="
echo "Code Encoding — Sampled (GraphCodeBERT)"
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

conda activate "icvul++"
conda info --envs
eval "$(conda shell.bash hook)"
conda activate icvul++

export PYTHONIOENCODING=utf-8
export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"

echo "Python:  $(which python)"
echo "Device check:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

echo "Encoding function code with GraphCodeBERT"
echo "Start: $(date)"

python "${BASE_DIR}/src/text_encoding/encode_code_gcb_sampled.py"
CODE_EXIT=$?

echo "Exit code: $CODE_EXIT"
echo "End: $(date)"
echo ""

echo "Output files:"
ls -lh "${BASE_DIR}/data_new/processed/sampled/function_code_embeddings*" 2>/dev/null || echo "  (none found)"

exit $CODE_EXIT
