#!/bin/bash
#SBATCH --job-name=shap_full_v2
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/shap_full_v2_%j.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/shap_full_v2_%j.err

# Full per-feature SHAP analysis — v2 (includes fn_bert + hunk_bert as players).
#
# Added to PERFEATURE_GROUPS vs v1:
#   fn_bert  : function.x[:,8:776]  treated as 1 player  (2 coalitions, exact)
#   hunk_bert: hunk.x[:,2:770]      treated as 1 player  (2 coalitions, exact)
#
# This gives a unified ranking of ALL 50 features (48 scalar + 2 BERT blocks)
# on the same Shapley scale, enabling a correct top-10 ablation.
#
# Checkpoint: kfold5_sage_repo fold_0/best.pt (the model the original SHAP used)
# Output:     outputs/shap_repo_fold0_v2/
# After this job: read perfeature_shap_summary.csv, identify top-10, then
# submit shap_top10_ablation.slurm (update the feature list if needed).

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
CKPT="${BASE_DIR}/checkpoints/kfold5_sage_repo/fold_0/best.pt"
OUT_DIR="${BASE_DIR}/outputs/shap_repo_fold0_v2"

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

python scripts/shap_analysis.py \
  --checkpoint        "${CKPT}" \
  --output_dir        "${OUT_DIR}" \
  --split             test \
  --split_type        repo_split \
  --per_feature_shap \
  --n_kernel_samples  500 \
  --batch_size        128

echo "Per-feature SHAP v2 complete. Outputs in ${OUT_DIR}"
echo ""
echo "Top-10 features (check perfeature_shap_summary.csv, rank 1-10):"
python -c "
import pandas as pd, sys
df = pd.read_csv('${OUT_DIR}/perfeature_shap_summary.csv')
print(df[['rank','group','feature','mean_abs_phi']].head(10).to_string(index=False))
"
