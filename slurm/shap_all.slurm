#!/bin/bash
#SBATCH --job-name=shap_all
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/shap_all_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/shap_all_%j.err

# Full per-feature SHAP analysis on the best fold of all three CV experiments.
#
# Checkpoints (best val AUC-PR):
#   kfold5_sage_repo  fold_0  val=0.8977  → repo_split  test
#   logo10_sage_repo  fold_0  val=0.9242  → repo_split  test
#   kfold5_sage_temporal fold_2 val=0.8974 → temporal_split test
#
# Per experiment:
#   (A) Exact group Shapley (128 coalitions, K=7)
#   (B) Commit gradient × input (14 dims)
#   (C) Full feature gradient × input (48 scalar + 2 BERT blocks)
#   (D) Per-feature exact Shapley for all scalar dims:
#       fn_metrics(256), file(8), hunk_metrics(4), developer(512),
#       issue(16), pr(16), tag(16), commit-KernelSHAP(502)
#
# Estimated wall-clock: ~2 h per experiment × 3 = ~6 h total.

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"
CKPTS="${BASE_DIR}/checkpoints"
OUT="${BASE_DIR}/outputs"

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

mkdir -p "${BASE_DIR}/logs"
cd "${THESIS_DIR}"

SHAP_COMMON_ARGS="
  --graphs_dir     ${GRAPHS_DIR}
  --split_index    ${SPLIT_INDEX}
  --perrepo_scaler ${SCALER}
  --per_feature_shap
  --full_features
  --n_kernel_samples 500
  --batch_size     128
"

# ── Experiment 1: kfold5_sage_repo fold_0 ────────────────────────────────────
echo ""
echo "============================================================"
echo "SHAP 1: kfold5_sage_repo / fold_0  (repo_split test)"
echo "============================================================"

python scripts/shap_analysis.py \
  --checkpoint   "${CKPTS}/kfold5_sage_repo/fold_0/best.pt" \
  --split_type   repo_split \
  --split        test \
  --output_dir   "${OUT}/shap_repo_fold0" \
  ${SHAP_COMMON_ARGS}

echo "SHAP 1 done."

# ── Experiment 2: logo10_sage_repo fold_0 ────────────────────────────────────
echo ""
echo "============================================================"
echo "SHAP 2: logo10_sage_repo / fold_0  (tensorflow val, repo_split test)"
echo "============================================================"

python scripts/shap_analysis.py \
  --checkpoint   "${CKPTS}/logo10_sage_repo/fold_0/best.pt" \
  --split_type   repo_split \
  --split        test \
  --output_dir   "${OUT}/shap_logo_fold0" \
  ${SHAP_COMMON_ARGS}

echo "SHAP 2 done."

# ── Experiment 3: kfold5_sage_temporal fold_2 ────────────────────────────────
echo ""
echo "============================================================"
echo "SHAP 3: kfold5_sage_temporal / fold_2  (temporal_split test)"
echo "============================================================"

python scripts/shap_analysis.py \
  --checkpoint   "${CKPTS}/kfold5_sage_temporal/fold_2/best.pt" \
  --split_type   temporal_split \
  --split        test \
  --output_dir   "${OUT}/shap_temporal_fold2" \
  ${SHAP_COMMON_ARGS}

echo "SHAP 3 done."
echo ""
echo "All SHAP analyses complete. Results in ${OUT}/shap_*"
