# GNN-Based Commit-Level VCC Detector

Binary classifier that predicts whether a commit is a Vulnerability Contributing Commit (VCC) using a heterogeneous GNN over per-commit ego-graphs.

**Label definition:** VCC = 1, FC and normal commits = 0. This is vulnerability detection, not fix discrimination.

---

## Environment

```bash
conda activate thesis
```

---

## Pipeline Overview

```
ICVul++ extraction
      ↓
final_graph_inputs_v1/   ← normalized CSVs + embeddings
      ↓
build_graphs_final.py    ← builds one .pt HeteroData per commit
      ↓
create_split_index_final.py  ← repo-level train/val/test split
      ↓
compute_perrepo_scaler.py    ← per-repo normalization stats (train only)
      ↓
train.py                 ← trains HeteroSAGE or HeteroRGCN
      ↓
test eval (auto at end of train.py using best.pt)
```

---

## Step 1 — Build Graphs

Reads from `data_new/analysis_outputs/final_graph_inputs_v1/` and writes one `.pt` file per commit to `outputs/final_graph_ready/graphs/`.

```bash
python scripts/build_graphs_final.py
python scripts/build_graphs_final.py --limit 100   # smoke test
```

Each graph is a `HeteroData` object with node types: `commit`, `file`, `function`, `hunk`, `developer`, `issue`, `pull_request`, `release_tag`.

**Important:** never rebuild graphs over the matched-normals benchmark without updating the input path. The builder reads from a hardcoded `FINAL` path — override it explicitly if using a new benchmark package.

---

## Step 2 — Create Split Index

Assigns each built commit to train/val/test at the **repo level** (no commit from a held-out repo appears in train).

```bash
python scripts/create_split_index_final.py
```

Output: `outputs/final_graph_ready/split_index.csv` with columns `hash, label, repo_url, repo_split`.

**Val repos (5):** ImageMagick, radare2, tcpdump, php-src, FreeRDP  
**Test repos (6):** FFmpeg, gpac, suricata, openssl, redis, envoy  
**Train:** everything else (including tensorflow)

Do not use `temporal_split` for thesis results — VCC/FC pairs cross the train/test boundary.

---

## Step 3 — Per-Repo Scaler

Computes per-repo mean/std for continuous numeric features from **training graphs only**. Val/test repos are excluded to prevent data leakage into normalization statistics.

```bash
python scripts/compute_perrepo_scaler.py
```

Output: `outputs/final_graph_ready/perrepo_function_scaler.json`

Must be rerun whenever the training set changes (e.g. new benchmark family).

---

## Step 4 — Train

```bash
# Primary model (HeteroSAGE, repo split)
python scripts/train.py --run_name sage_matched_v1 --perrepo_norm

# Ablation: no code/file size metrics
python scripts/train.py --run_name sage_nosize_v1 --perrepo_norm \
    --ablate_fn_code_metrics --ablate_file_code_metrics

# Override paths for a new benchmark family
python scripts/train.py \
    --graphs_dir data_new/analysis_outputs/matched_normals_v1/graphs \
    --split_index data_new/analysis_outputs/matched_normals_v1/split_index.csv \
    --run_name sage_matched_v1 --perrepo_norm
```

Key defaults: `--hidden 128 --dropout 0.3 --lr 1e-3 --epochs 100 --batch_size 128 --patience 15`

**Checkpoints** are saved to `outputs/runs/<run_name>/`:
- `best.pt` — highest val AUC-PR (used for test eval)
- `latest.pt` — resumed after preemption
- `metrics.csv` — per-epoch train/val metrics

Test evaluation runs automatically at the end using `best.pt`.

**Primary metric:** AUC-PR (threshold-free, imbalance-aware). Secondary: MCC, F1.

---

## Step 5 — Ablation Matrix

All ablations use `--ablate_fn_code_metrics --ablate_file_code_metrics` (no size features) as the baseline regime.

| Run | Extra flags |
|---|---|
| Baseline (no size) | *(none beyond base)* |
| No code embedding | `--ablate_code_emb` |
| No message embedding | `--ablate_msg_emb` |
| No SDLC | `--ablate_sdlc` |
| No fct flags | `--ablate_fn_categorical` |
| No embeddings | `--ablate_code_emb --ablate_msg_emb` |

---

## Benchmark Families

| Family | Location | Notes |
|---|---|---|
| `final_graph_ready` | `outputs/final_graph_ready/` | Original stratified normal sample (31,513 normals). Do not use for main thesis results. |
| `matched_normals_v1` | `outputs/final_graph_ready_matched_normals_v1/` | Complexity-matched normals (cap×20). **Primary thesis benchmark.** |

Do not compare numbers across benchmark families without explicitly labeling them as cross-benchmark.

---

## Repository Structure

```
src/
  graph_builder.py       # Core graph construction (legacy — see build_graphs_final.py)
  graph_dataset.py       # PyG Dataset, make_loader(), perrepo_norm, ablation flags
  model.py               # HeteroSAGE (primary), HeteroRGCN (ablation), FocalLoss
  data_structure.py      # Column definitions for CSV tables
scripts/
  build_graphs_final.py        # Graph builder for final_graph_inputs_v1 package
  create_split_index_final.py  # Repo-level split index
  compute_perrepo_scaler.py    # Per-repo normalization stats
  sample_matched_normals.py    # Matched-normal sampling (produces benchmark_manifest.csv)
  train.py                     # Training loop
  validate_graphs.py           # Sanity checks on built graphs
  *.slurm                      # VSC/HPC job scripts
docs/
  GRAPH_CONSTRUCTION.md        # Graph schema, node/edge types, feature dims
  DATA_README.md               # Full CSV schema reference
env/
  thesis.yml                   # Conda environment spec
notebooks/                     # Exploratory notebooks (not part of pipeline)
```

---

## What Goes in Git

**Include:**
- `src/` — all model and dataset code
- `scripts/` — all pipeline scripts and SLURM files
- `docs/` — schema documentation
- `env/thesis.yml` — environment spec
- `CLAUDE.md`, `AGENTS.md`, `README.md`
- `notebooks/` — if they contain reproducible analysis

**Exclude (add to `.gitignore`):**
- `data/`, `data_new/` — all CSV tables and raw data (too large, not reproducible from this repo)
- `outputs/` — built graphs (`.pt`), checkpoints, metrics (reproducible from scripts)
- `*.npy` — embedding arrays
- `*.json` scaler files (reproducible from `compute_perrepo_scaler.py`)
- `__pycache__/`, `.ipynb_checkpoints/`
- `.claude/` — agent memory and worktrees (local tooling, not thesis code)

The repo should be runnable given: (1) the conda env, (2) the `final_graph_inputs_v1/` package provided separately (e.g. as a data release or shared drive link).
