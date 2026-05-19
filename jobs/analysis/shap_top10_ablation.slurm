#!/bin/bash
#SBATCH --job-name=shap_top10_abl
#SBATCH --account=lp_ideasgrouptproject
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=1,4
#SBATCH --output=/data/leuven/380/vsc38046/Thesis/logs/shap_top10_abl_%A_%a.out
#SBATCH --error=/data/leuven/380/vsc38046/Thesis/logs/shap_top10_abl_%A_%a.err

# Per-feature ablation: remove each of the top-10 SHAP features one at a time.
# Baseline: kfold5_sage_repo (repo_split, HeteroSAGE, perrepo_norm, no ablations).
# Each array task (1-10) removes exactly one feature and re-trains k=5 folds
# with identical hyperparameters to the baseline.
#
# Rankings from outputs/shap_repo_fold0_v2/perfeature_shap_summary.csv
# (n=500, kfold5_sage_repo fold_0, per-feature Shapley including BERT blocks):
#
#  Task  Feature                       phi     Node            Dims/Flag
#   1    file: lines_added             0.1518  file.x          col 0
#   2    fn: bert_emb                  0.1260  function.x      8:776   (--ablate_code_emb)
#   3    file: lines_deleted           0.0485  file.x          col 1
#   4    hunk: bert_emb                0.0430  hunk.x          2:770   (--ablate_hunk_emb)
#   5    file: complexity              0.0367  file.x          col 2
#   6    commit: in_main_branch        0.0309  commit.x        col 0
#   7    commit: has_sdlc_data         0.0300  commit.x        col 11
#   8    dev: commits_as_committer     0.0295  developer.x     col 2
#   9    dev: active_weeks             0.0293  developer.x     col 1
#  10    issue: age_median             0.0270  issue.x         col 1

set -euo pipefail

BASE_DIR="/data/leuven/380/vsc38046/Thesis"
THESIS_DIR="${BASE_DIR}/thesis"
GRAPHS_DIR="${BASE_DIR}/outputs/graph_ready_v2/graphs"
SPLIT_INDEX="${BASE_DIR}/outputs/graph_ready_v2/split_index.csv"
SCALER="${BASE_DIR}/outputs/graph_ready_v2/perrepo_scaler_v2.json"

# ── Per-task ablation flags ───────────────────────────────────────────────────
TASK=${SLURM_ARRAY_TASK_ID}
EXTRA_FLAGS=()

case $TASK in
  1)  RUN_NAME="kfold5_sage_no_file_lines_added"
      EXTRA_FLAGS+=(--ablate_single_dim "file:0") ;;
  2)  RUN_NAME="kfold5_sage_no_fn_bert"
      EXTRA_FLAGS+=(--ablate_code_emb) ;;
  3)  RUN_NAME="kfold5_sage_no_file_lines_deleted"
      EXTRA_FLAGS+=(--ablate_single_dim "file:1") ;;
  4)  RUN_NAME="kfold5_sage_no_hunk_bert"
      EXTRA_FLAGS+=(--ablate_hunk_emb) ;;
  5)  RUN_NAME="kfold5_sage_no_file_complexity"
      EXTRA_FLAGS+=(--ablate_single_dim "file:2") ;;
  6)  RUN_NAME="kfold5_sage_no_commit_in_main_branch"
      EXTRA_FLAGS+=(--ablate_single_dim "commit:0") ;;
  7)  RUN_NAME="kfold5_sage_no_commit_has_sdlc_data"
      EXTRA_FLAGS+=(--ablate_single_dim "commit:11") ;;
  8)  RUN_NAME="kfold5_sage_no_dev_commits_as_committer"
      EXTRA_FLAGS+=(--ablate_single_dim "developer:2") ;;
  9)  RUN_NAME="kfold5_sage_no_dev_active_weeks"
      EXTRA_FLAGS+=(--ablate_single_dim "developer:1") ;;
  10) RUN_NAME="kfold5_sage_no_issue_age_median"
      EXTRA_FLAGS+=(--ablate_single_dim "issue:1") ;;
  *)  echo "Unknown task id: $TASK"; exit 1 ;;
esac

echo "Array task ${TASK}/10: ${RUN_NAME}"
echo "Extra flags: ${EXTRA_FLAGS[*]}"

# ── Environment setup ─────────────────────────────────────────────────────────
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

# ── Copy graphs to node-local scratch ─────────────────────────────────────────
SCRATCH="${TMPDIR:-/tmp}/graphs_${SLURM_JOB_ID}_${TASK}"
echo "Copying graphs to scratch: ${SCRATCH}"
mkdir -p "${SCRATCH}"
cp -r "${GRAPHS_DIR}/." "${SCRATCH}/"
echo "Copy done. $(ls ${SCRATCH} | wc -l) graphs in scratch."

mkdir -p "${BASE_DIR}/logs"
cd "${THESIS_DIR}"

# ── Train ─────────────────────────────────────────────────────────────────────
python scripts/training/kfold_train.py \
  --graphs_dir         "${SCRATCH}" \
  --split_index        "${SPLIT_INDEX}" \
  --perrepo_scaler     "${SCALER}" \
  --output_dir         "${BASE_DIR}/checkpoints/shap_top10_ablation" \
  --run_name           "${RUN_NAME}" \
  "${EXTRA_FLAGS[@]}" \
  --perrepo_norm \
  --k                  5 \
  --fold_seed          0 \
  --seed               42 \
  --model              sage \
  --hidden             128 \
  --dropout            0.4 \
  --epochs             60 \
  --patience           10 \
  --batch_size         128 \
  --lr                 1e-3 \
  --weight_decay       5e-4 \
  --warmup_epochs      3 \
  --focal_gamma        1.5 \
  --focal_alpha        0.65 \
  --ckpt_every         0 \
  --num_workers        4

echo "Task ${TASK} (${RUN_NAME}) complete."
