#!/bin/bash
#SBATCH --job-name=kfold_aug
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/kfold_aug_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/kfold_aug_%j.err

# Graph-augmentation ablation run.
#
# Baseline regularization (dropout=0.4, wd=5e-4) — reverts the stronger-reg
# changes from kfold_final that narrowed the gap by only 0.004 and hurt test MCC.
# No SHAP-ablations (ablate_hunk_metrics / ablate_commit_merge also hurt test MCC).
#
# Augmentation:
#   --aug_node_mask_p 0.3   per-node-type masking probability (excl. commit)
#   --aug_edge_drop_p 0.2   per-edge drop fraction (both directions dropped together)
#
# Baseline: checkpoints/kfold5_sage_repo — test_auc_pr=0.8539, test_mcc=0.6275,
#           avg train-val gap=0.1228.

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

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

SCRATCH="${TMPDIR:-/tmp}/graphs_$$"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs in scratch."

mkdir -p "${BASE_DIR}/logs"
cd "${THESIS_DIR}"

python scripts/kfold_train.py \
  --graphs_dir        "${SCRATCH}" \
  --split_index       "${SPLIT_INDEX}" \
  --perrepo_scaler    "${SCALER}" \
  --output_dir        "${BASE_DIR}/checkpoints" \
  --run_name          kfold5_sage_aug \
  --perrepo_norm \
  --k                 5 \
  --fold_seed         0 \
  --seed              42 \
  --model             sage \
  --hidden            128 \
  --dropout           0.4 \
  --epochs            60 \
  --patience          10 \
  --batch_size        128 \
  --lr                1e-3 \
  --weight_decay      5e-4 \
  --warmup_epochs     3 \
  --focal_gamma       1.5 \
  --focal_alpha       0.65 \
  --aug_node_mask_p   0.3 \
  --aug_edge_drop_p   0.2 \
  --ckpt_every        0 \
  --num_workers       4

echo "kfold5_sage_aug complete."
