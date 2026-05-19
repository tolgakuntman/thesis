# Heterogeneous GNN for Just-in-Time Vulnerability Prediction

Binary classifier that detects Vulnerability Contributing Commits (VCCs) at commit time using a heterogeneous graph neural network over per-commit ego-graphs extracted from the ICVul++ dataset.

**Primary metric:** AUC-PR (threshold-free, class-imbalance-aware)  
**Secondary metrics:** MCC, F1 (best-sweep threshold)  
**Label:** VCC = 1, Fix Commits (FC) and normal commits = 0

---

## Research Objective

Just-in-time (JIT) vulnerability prediction aims to flag a commit as vulnerability-introducing *at the time it is made*, enabling early intervention before a CVE is filed. Existing approaches rely on flat commit-level features or treat code changes as text. This thesis investigates whether a heterogeneous graph representation — encoding commit, file, function, hunk, developer, issue, pull request, and release tag as typed graph nodes — can improve AUC-PR over flat-feature baselines and reveal which structural components contribute most through ablation.

---

## Dataset: ICVul++

ICVul++ is an extended version of the ICVul dataset of C/C++ CVE-linked commits extracted from open-source projects on GitHub. It enriches each commit with:
- **Code structure:** file-level and function-level change statistics, hunk-level diff embeddings (GraphCodeBERT)
- **Developer context:** per-developer experience metrics, file ownership history
- **SDLC metadata:** linked issue reports, pull requests, and release tags
- **Commit metadata:** DMM change-quality metrics, author/timezone features, merge flag

The **matched-normals-v1 benchmark** (primary thesis benchmark) uses complexity-matched normal (non-VCC, non-FC) commits sampled to match VCCs in a 6-dimensional feature space, capped at ×20 ratio. This reduces selection bias in the negative class compared to stratified random sampling.

> **Data availability:** The dataset is not included in this repository. Contact the authors or see the thesis for the data release process.

---

## Graph Construction

Each commit is represented as a **per-commit ego-graph** (one `HeteroData` object per commit):

| Node type | Features | Description |
|-----------|----------|-------------|
| `commit` | 14 dims | Change stats, DMM metrics, time features, message embedding |
| `file` | 3 dims | Lines added/deleted, complexity |
| `function` | 776 dims | Code metrics (8) + GraphCodeBERT embedding (768) |
| `hunk` | 770 dims | Complexity, token count (2) + GraphCodeBERT embedding (768) |
| `developer` | 9 dims | Experience, ownership, cross-repo activity |
| `issue` | 4 dims | Linked issue open/close velocity |
| `pull_request` | 4 dims | PR velocity metrics |
| `release_tag` | 4 dims | Release proximity features |

Edge types (all bidirectional): `modifies_file`, `contains`, `modifies_func`, `modifies_hunk`, `authored_by`, `committed_by`, `owns`, `has_issue`, `has_pr`, `has_release`.

---

## Model Architecture: HeteroSAGE

```
Input: per-commit HeteroData (8 node types, 10 edge types)
  ↓
Per-type LazyLinear projection → hidden (128)
  ↓
HeteroConv Layer 1:
  GATv2Conv (edge-attr aware) for edge types with edge attributes
  SAGEConv (mean aggregation) for topology-only edge types
  + LayerNorm + ReLU + Dropout
  ↓
HeteroConv Layer 2 (same)
  ↓
Commit node readout h['commit'] ∈ R^{B×128}
  ↓
Linear(128 → 1) → scalar logit (pre-sigmoid)
```

**Loss:** Focal loss (α=0.65, γ=1.5) — down-weights easy negatives in the imbalanced VCC/normal setting.

**Baseline models:**
- `HeteroRGCN` — same interface, replaces GATv2Conv/SAGEConv with R-GCN basis decomposition (no edge features)
- `CommitMLP` — commit-node features only, no message passing

---

## Experiment Overview

| Experiment | Split | Purpose |
|-----------|-------|---------|
| Repo-split k-fold | Repo-level | Primary generalization result (cross-project) |
| Temporal split | Date-ordered | Temporal generalization claim |
| Ablation matrix | Repo-split | Feature/component attribution |
| Generalization suite | 9 split strategies | Robustness across partition schemes |
| SHAP analysis | Repo-split test | Feature attribution (exact Shapley + gradient×input) |

**Primary result:** `jobs/training/kfold_final.slurm` (k=5, repo-grouped, SHAP-justified ablations)

---

## Installation

### Prerequisites

- conda (Miniconda or Anaconda)
- CUDA 12.x (for GPU training; CPU works but is slow)
- ~64GB RAM for full dataset loading

### Create environment

```bash
conda env create -f environment.yml
conda activate thesis
```

### Or install via pip

```bash
pip install -r requirements.txt
pip install -e .          # install src/ as editable package
```

---

## Dataset Preparation

> Requires the `final_graph_inputs_v1/` data package (provided separately).

### Step 1 — Build per-commit graphs

```bash
python scripts/data/build_graphs_final.py
python scripts/data/build_graphs_final.py --limit 100   # smoke test
```

Output: `outputs/final_graph_ready/graphs/<hash>.pt` (one HeteroData per commit)

### Step 2 — Create split index

```bash
python scripts/data/create_split_index_final.py
```

Output: `outputs/final_graph_ready/split_index.csv`

- **Val repos (5):** ImageMagick, radare2, tcpdump, php-src, FreeRDP
- **Test repos (6):** FFmpeg, gpac, suricata, openssl, redis, envoy
- **Train:** all remaining repos (including tensorflow)

### Step 3 — Compute per-repo scalers

```bash
python scripts/data/compute_perrepo_scaler.py
```

Output: `outputs/final_graph_ready/perrepo_function_scaler.json`

> Computed from training graphs only to prevent data leakage into normalization statistics.

---

## Training

### Primary model (HeteroSAGE, repo split)

```bash
python scripts/training/train.py \
    --run_name sage_thesis \
    --perrepo_norm \
    --seeds 42 123 7
```

Checkpoints saved to `checkpoints/sage_thesis/seed_{42,123,7}/`:
- `best.pt` — highest val AUC-PR
- `latest.pt` — most recent epoch (resume after preemption)
- `metrics.csv` — per-epoch train/val metrics
- `test_results.json` — final test evaluation

### K-fold cross-validation (primary thesis result)

```bash
python scripts/training/kfold_train.py \
    --k 5 \
    --seed 42 \
    --perrepo_norm \
    --ablate_hunk_metrics \
    --ablate_commit_merge \
    --dropout 0.5 \
    --weight_decay 2e-3
```

### Override data paths

```bash
python scripts/training/train.py \
    --graphs_dir /path/to/graphs \
    --split_index /path/to/split_index.csv \
    --perrepo_scaler /path/to/scaler.json \
    --run_name my_run
```

---

## Evaluation

### SHAP attribution analysis

```bash
python scripts/evaluation/shap_analysis.py \
    --checkpoint checkpoints/sage_thesis/seed_42/best.pt \
    --output_dir outputs/shap_analysis
```

Outputs: `shap_values.csv`, `shap_summary.csv`, `commit_gradients.csv`, bar/violin plots.

### Build result tables

```bash
python scripts/evaluation/build_result_tables.py
```

Reads from `checkpoints/` and prints LaTeX-ready result tables.

---

## Reproducing Thesis Experiments

### Full ablation sweep

```bash
bash experiments/ablations/run_ablation_sweep.sh
```

### Generalization split suite

```bash
python experiments/generalization/generate_splits.py
python experiments/generalization/run_split_suite.py
```

### Monitor running sweep

```bash
python experiments/ablations/monitor_ablation_sweep.py
python experiments/ablations/analyze_ablation_sweep.py
```

---

## HPC / SLURM Execution

All job scripts are in `jobs/`. The primary thesis result jobs:

```bash
# Primary k-fold training (run from repo root on HPC)
sbatch jobs/training/kfold_final.slurm

# SHAP analysis after training
sbatch jobs/analysis/shap_top10_ablation.slurm

# Phase 6 multi-seed final confirmation
cd jobs/launchers && bash launch_phase6_finals.sh
```

### Scratch-disk optimization

The `kfold_final.slurm` script copies graphs to `$TMPDIR` (local NVMe) before training. This reduces I/O bottleneck on shared filesystems. Adjust `GRAPHS_DIR`, `SPLIT_INDEX`, and `SCALER` paths at the top of the script for your cluster layout.

### Environment setup on HPC

```bash
module load miniconda3
conda activate icvul++             # or: conda activate thesis
export HF_HOME=/path/to/.cache/huggingface
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── setup.py
├── .gitignore
│
├── src/                          # Core library (model, dataset, graph construction)
│   ├── model.py                  # HeteroSAGE, HeteroRGCN, CommitMLP, FocalLoss
│   ├── graph_dataset.py          # VulnCommitDataset, make_loader, ablation flags
│   ├── graph_builder.py          # Graph construction utilities
│   └── data_structure.py         # CSV column schema definitions
│
├── configs/
│   ├── default_training.yaml     # Frozen training protocol (2026-04-11)
│   └── ablations.yaml            # Ablation study configurations
│
├── scripts/
│   ├── data/                     # Data preparation pipeline
│   │   ├── build_graphs_final.py
│   │   ├── create_split_index_final.py
│   │   ├── compute_perrepo_scaler.py
│   │   ├── sample_matched_normals.py
│   │   └── preaggregate_ownership.py
│   ├── training/                 # Training loops
│   │   ├── train.py              # Single/multi-seed training
│   │   └── kfold_train.py        # Repo-grouped k-fold CV
│   ├── evaluation/               # Analysis and reporting
│   │   ├── shap_analysis.py
│   │   ├── feature_analysis.py
│   │   ├── aggregate_finals.py
│   │   └── build_result_tables.py
│   └── validation/               # Data integrity checks
│       ├── validate_graphs.py
│       └── validate_splits.py
│
├── experiments/
│   ├── ablations/                # Ablation sweep driver scripts
│   │   ├── run_ablation_sweep.sh
│   │   ├── analyze_ablation_sweep.py
│   │   └── monitor_ablation_sweep.py
│   └── generalization/           # Split-generalization experiments
│       ├── split_strategies.py
│       ├── generate_splits.py
│       ├── run_split_suite.py
│       └── run_generalization_sweep.sh
│
├── jobs/                         # HPC / SLURM job scripts
│   ├── training/                 # Training jobs
│   │   ├── kfold_final.slurm     # PRIMARY thesis result
│   │   ├── train_ablation.slurm  # Ablation / phase launcher template
│   │   ├── cv_logo_temporal.slurm
│   │   └── smoke_test.slurm
│   ├── analysis/                 # Analysis jobs
│   │   └── shap_top10_ablation.slurm
│   ├── pipelines/                # Multi-step pipeline jobs
│   │   └── submit_split_suite.slurm
│   └── launchers/                # Phase orchestration scripts
│       ├── launch_phase6_finals.sh  # PRIMARY phase launcher
│       └── launch_phase{1,2,3_4,5_1,5_hpsearch}.sh
│
├── notebooks/
│   ├── exploration/              # Data exploration
│   ├── graph_construction/       # Graph construction walkthroughs
│   ├── demos/                    # Single/multi-commit demos
│   └── training/                 # Training experiments in notebooks
│
├── outputs/                      # Generated (gitignored — reproducible from scripts)
│   ├── final_graph_ready/        # Built graphs + split index + scaler
│   ├── runs/                     # Training outputs
│   └── shap_*/                   # SHAP analysis outputs
│
├── checkpoints/                  # Model checkpoints (gitignored)
│
└── archive/                      # Superseded files (reference only)
    ├── README.md
    ├── scripts/                  # Legacy pipeline scripts (v0, v2)
    └── jobs/                     # Legacy SLURM jobs
```

---

## Output Locations

| Artifact | Location |
|----------|----------|
| Built graphs | `outputs/final_graph_ready/graphs/<hash>.pt` |
| Split index | `outputs/final_graph_ready/split_index.csv` |
| Per-repo scaler | `outputs/final_graph_ready/perrepo_function_scaler.json` |
| Training checkpoints | `checkpoints/<run_name>/seed_<N>/best.pt` |
| Per-epoch metrics | `checkpoints/<run_name>/seed_<N>/metrics.csv` |
| Test results | `checkpoints/<run_name>/seed_<N>/test_results.json` |
| Multi-seed aggregate | `checkpoints/<run_name>/seed_results.json` |
| K-fold summary | `checkpoints/<run_name>/kfold_summary.json` |
| SHAP values | `outputs/shap_<run>/shap_values.csv` |

---

## Benchmark Families

| Family | Location | Notes |
|--------|----------|-------|
| `final_graph_ready` | `outputs/final_graph_ready/` | Original stratified normal sample (31,513 normals). **Not used for main thesis results.** |
| `matched_normals_v1` | `outputs/final_graph_ready_matched_normals_v1/` | Complexity-matched normals (cap×20). **Primary thesis benchmark.** |

Do not compare numbers across benchmark families without explicit labeling.

---

## Random Seed Handling

- **Primary seeds:** 42, 123, 7 (used for all multi-seed thesis results)
- Each seed initializes both PyTorch (`torch.manual_seed`) and NumPy (`np.random.seed`)
- K-fold partition uses `--fold_seed` (default 0) for deterministic repo-to-fold assignment
- The split index (`split_index.csv`) is deterministic given the repo list; no random seed needed

---

## What Is and Is Not Versioned

**Tracked in git:**
- `src/` — model, dataset, graph builder
- `scripts/`, `experiments/`, `jobs/` — all pipeline scripts
- `configs/` — training configuration files
- `notebooks/` — exploratory and demo notebooks
- `README.md`, `requirements.txt`, `environment.yml`, `setup.py`

**Not tracked (gitignored — reproducible from scripts):**
- `data/`, `data_new/` — raw CSV tables and embeddings (symlinks to ICVul/ICVul++)
- `outputs/` — built graphs, SHAP outputs, split files
- `checkpoints/` — model checkpoints
- `*.npy`, `*.pkl`, `*.pt` — large binary files

---

## License

This repository is released under the MIT License. See `LICENSE` for details.

The ICVul++ dataset is subject to its own license terms. Consult the dataset documentation before use in commercial or derivative works.
