#!/bin/bash

#SBATCH --job-name=vcc_rgcn_temp
#SBATCH --account=lp_ideasgrouptproject

#SBATCH --clusters=wice

#SBATCH --partition=gpu_a100

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=40G

#SBATCH --gpus-per-node=1

#SBATCH --time=08:00:00
#SBATCH --output=logs/rgcn_temporal_%j.out
#SBATCH --error=logs/rgcn_temporal_%j.err

echo "============================================================"
echo "VCC Detection — HeteroRGCN (ablation)  |  temporal_split"
echo "============================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

BASE_DIR="/data/leuven/383/vsc38347/thesis"
set -euo pipefail
mkdir -p "${BASE_DIR}/logs"
mkdir -p "${BASE_DIR}/checkpoints"

# ── environment ──────────────────────────────────────────────────────────────
module purge
module load cluster/wice/gpu
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate icvulpp

export PYTHONIOENCODING=utf-8

echo "Python:  $(which python)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo ""

# ── copy graphs to local node scratch ─────────────────────────────────────────
echo "Copying 29k graphs to local scratch ($TMPDIR)..."
GRAPHS_SRC="${BASE_DIR}/data_new/graph_ready/graphs"
GRAPHS_DST="${TMPDIR}/graphs"
mkdir -p "${GRAPHS_DST}"
cp "${GRAPHS_SRC}"/*.pt "${GRAPHS_DST}/"
echo "Copy complete: $(date)  ($(ls ${GRAPHS_DST} | wc -l) files)"
echo ""

# ── train ─────────────────────────────────────────────────────────────────────
echo "Training HeteroRGCN — temporal_split"
echo "Start: $(date)"

python "${BASE_DIR}/scripts/train.py" \
    --model         rgcn \
    --split_type    temporal_split \
    --run_name      rgcn_temporal_v1 \
    --hidden        128 \
    --dropout       0.3 \
    --epochs        100 \
    --batch_size    512 \
    --lr            1e-3 \
    --weight_decay  1e-4 \
    --patience      15 \
    --warmup_epochs 5 \
    --focal_gamma   2.0 \
    --focal_alpha   0.75 \
    --num_workers   4 \
    --output_dir    "${BASE_DIR}/checkpoints" \
    --graphs_dir    "${GRAPHS_DST}" \
    --split_index   "${BASE_DIR}/data_new/graph_ready/split_index.csv"

EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"

echo ""
echo "Checkpoints:"
ls -lh "${BASE_DIR}/checkpoints/rgcn_temporal_v1/" 2>/dev/null || echo "  (not found)"

exit $EXIT_CODE
