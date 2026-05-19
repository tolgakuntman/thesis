#!/bin/bash
#SBATCH --job-name=kfold5_pruned_v2
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/kfold5_pruned_v2_%j.out
#SBATCH --error=logs/kfold5_pruned_v2_%j.err

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"

export HF_HOME="${BASE_DIR}/.cache/huggingface"

source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate icvul++

# Auto-install torch_geometric if missing
python -c "import torch_geometric" 2>/dev/null || {
    echo "Installing torch_geometric …"
    pip install torch_geometric torch_scatter torch_sparse torch_cluster \
        -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)" | cut -d'+' -f1)+cu124.html
}

cd "${THESIS_DIR}"
mkdir -p logs

GRAPHS_SRC="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

# Copy graphs to node-local TMPDIR for fast I/O
echo "Copying graphs to \$TMPDIR …"
mkdir -p "$TMPDIR/graphs"
cp "${GRAPHS_SRC}"/*.pt "$TMPDIR/graphs/"
echo "  Done: $(ls "$TMPDIR/graphs" | wc -l) graphs"

echo "=== kfold5_sage_pruned_v2: round-2 feature pruning ==="
echo "Base HPs from kfold5_sage_repo (hidden=128, dropout=0.4, wd=5e-4, perrepo_norm)"
echo "Round-1: hunk_metrics + commit_merge + 11 SHAP+corr dims (ablate_pruned_features)"
echo "Round-2: 10 bottom-phi dims: fn[3], tag[1], dev[5,7,8], commit[3,4,7,10], pr[3]"
echo "Total scalar dims zeroed after both rounds: 24 of 45 active (53%)"
echo "Job started: $(date)"

python scripts/kfold_train.py \
    --run_name       kfold5_sage_pruned_v2 \
    --k              5 \
    --model          sage \
    --hidden         128 \
    --dropout        0.4 \
    --weight_decay   5e-4 \
    --lr             1e-3 \
    --epochs         60 \
    --patience       10 \
    --warmup_epochs  3 \
    --batch_size     128 \
    --focal_alpha    0.65 \
    --focal_gamma    1.5 \
    --seed           42 \
    --graphs_dir     "$TMPDIR/graphs" \
    --split_index    "${SPLIT_INDEX}" \
    --perrepo_norm \
    --perrepo_scaler "${SCALER}" \
    --ablate_hunk_metrics \
    --ablate_commit_merge \
    --ablate_pruned_features \
    --ablate_pruned_v2

echo "Job finished: $(date)"
