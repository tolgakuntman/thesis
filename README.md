# Heterogeneous GNN for Just-in-Time Vulnerability Prediction

Binary classifier for Vulnerability Contributing Commits (VCCs) at commit time, using a heterogeneous graph neural network over per-commit ego-graphs from the ICVul++ dataset.

**Primary metric:** F1 (best-sweep threshold)  
**Secondary metrics:** MCC, AUC-PR  
**Checkpointing criterion:** val AUC-PR (threshold-free, imbalance-robust)  
**Label:** VCC = 1, Fix Commits (FC) and normal commits = 0

---

## Repository Scope

This repository covers:
- Per-commit heterogeneous graph construction from ICVul++ tabular features
- HeteroSAGE (GATv2Conv + SAGEConv) and HeteroRGCN model implementations
- Training and evaluation pipelines: single-split, k-fold, ablation suite, generalization suite
- SHAP and gradient×input attribution analysis
- SLURM job scripts for HPC execution

Dataset collection, metadata extraction, and raw feature engineering are maintained separately — see [Dataset (ICVul++)](#dataset-icvul) below.

---

## Dataset (ICVul++)

The dataset extension and metadata pipeline are in a separate repository:

**[https://github.com/tolgakuntman/ICVulPP](https://github.com/tolgakuntman/ICVulPP)**

ICVulPP handles CVE commit extraction and enrichment with file/function/hunk-level metrics, SDLC metadata (issues, PRs, releases), and developer experience features.

This repository consumes the tabular CSVs and embeddings produced by ICVulPP and builds `HeteroData` graph objects from them.

> The ICVul++ dataset is not included here. See the ICVulPP repository for data access.

---

## Research Objective

Just-in-time (JIT) vulnerability prediction identifies vulnerability-introducing commits at commit time, before a CVE is filed. Prior work uses flat commit-level features or treats code changes as text. This thesis evaluates whether a heterogeneous graph representation — encoding commit, file, function, hunk, developer, issue, pull request, and release tag as typed nodes — improves F1 over flat-feature baselines, and uses ablations and SHAP attribution to identify which components contribute.

---

## Graph Construction

Each commit maps to a per-commit ego-graph (one `HeteroData` object):

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
| Temporal split | Date-ordered | Temporal generalization |
| Ablation matrix | Repo-split | Feature/component attribution |
| Generalization suite | 9 split strategies | Robustness across partition schemes |
| SHAP analysis | Repo-split test | Feature attribution (exact Shapley + gradient×input) |

**Primary result:** `jobs/training/kfold_final.slurm` (k=5, repo-grouped, SHAP-justified ablations)

---

## Installation

### Prerequisites

- conda (Miniconda or Anaconda)
- CUDA 12.x (for GPU training; CPU works but is slow)
- ~64 GB RAM for full dataset loading

### Create environment

```bash
conda env create -f environment.yml
conda activate thesis
```

### Or install via pip

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Dataset Preparation

> Requires the `final_graph_inputs_v1/` data package (provided separately via ICVulPP).

### Step 1 — Build per-commit graphs

```bash
python scripts/data/build_graphs_final.py
python scripts/data/build_graphs_final.py --limit 100   # smoke test
```

Output: `outputs/final_graph_ready/graphs/<hash>.pt` (one `HeteroData` per commit)

### Step 2 — Create split index

```bash
python scripts/data/create_split_index_final.py
```

Output: `outputs/final_graph_ready/split_index.csv`

- **Val repos (5):** ImageMagick, radare2, tcpdump, php-src, FreeRDP
- **Test repos (6):** FFmpeg, gpac, suricata, openssl, redis, envoy
- **Train:** all remaining repos

### Step 3 — Compute per-repo scalers

```bash
python scripts/data/compute_perrepo_scaler.py
```

Output: `outputs/final_graph_ready/perrepo_function_scaler.json`

> Fit on training graphs only to prevent leakage into normalization statistics.

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
- `best.pt` — highest val AUC-PR (checkpointing criterion, not primary reported metric)
- `latest.pt` — most recent epoch (for preemption recovery)
- `metrics.csv` — per-epoch train/val metrics
- `test_results.json` — final test evaluation (F1, MCC, AUC-PR)

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

All job scripts are in `jobs/`. Primary thesis result jobs:

```bash
# Primary k-fold training
sbatch jobs/training/kfold_final.slurm

# SHAP analysis after training
sbatch jobs/analysis/shap_top10_ablation.slurm

# Phase 6 multi-seed final confirmation
cd jobs/launchers && bash launch_phase6_finals.sh
```

### Scratch-disk optimization

`kfold_final.slurm` copies graphs to `$TMPDIR` (local NVMe) before training to reduce I/O on shared filesystems. Adjust `GRAPHS_DIR`, `SPLIT_INDEX`, and `SCALER` at the top of the script for your cluster layout.

### Environment setup on HPC

```bash
source /data/leuven/380/vsc38046/miniconda3/etc/profile.d/conda.sh
conda activate "icvul++"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
mkdir -p "${HF_HOME}"
```

> `module load miniconda3` is not available in non-interactive SLURM shells — use `source` directly. Set `HF_HOME` to a data-disk path; the home directory quota will fill immediately otherwise.

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
├── src/                          # Core library
│   ├── model.py                  # HeteroSAGE, HeteroRGCN, CommitMLP, FocalLoss
│   ├── graph_dataset.py          # VulnCommitDataset, make_loader, ablation flags
│   ├── graph_builder.py          # Graph construction
│   └── data_structure.py         # CSV column schema definitions
│
├── configs/
│   ├── default_training.yaml     # Frozen training protocol (2026-04-11)
│   └── ablations.yaml            # Ablation configurations
│
├── scripts/
│   ├── data/                     # Data preparation
│   │   ├── build_graphs_final.py
│   │   ├── create_split_index_final.py
│   │   ├── compute_perrepo_scaler.py
│   │   ├── sample_matched_normals.py
│   │   └── preaggregate_ownership.py
│   ├── training/
│   │   ├── train.py              # Single/multi-seed training
│   │   └── kfold_train.py        # Repo-grouped k-fold CV
│   ├── evaluation/
│   │   ├── shap_analysis.py
│   │   ├── feature_analysis.py
│   │   ├── aggregate_finals.py
│   │   └── build_result_tables.py
│   └── validation/
│       ├── validate_graphs.py
│       └── validate_splits.py
│
├── experiments/
│   ├── ablations/
│   │   ├── run_ablation_sweep.sh
│   │   ├── analyze_ablation_sweep.py
│   │   └── monitor_ablation_sweep.py
│   └── generalization/
│       ├── split_strategies.py
│       ├── generate_splits.py
│       ├── run_split_suite.py
│       └── run_generalization_sweep.sh
│
├── jobs/
│   ├── training/
│   │   ├── kfold_final.slurm     # PRIMARY thesis result
│   │   ├── train_ablation.slurm
│   │   ├── cv_logo_temporal.slurm
│   │   └── smoke_test.slurm
│   ├── analysis/
│   │   └── shap_top10_ablation.slurm
│   ├── pipelines/
│   │   └── submit_split_suite.slurm
│   └── launchers/
│       ├── launch_phase6_finals.sh
│       └── launch_phase{1,2,3_4,5_1,5_hpsearch}.sh
│
├── notebooks/
│   ├── exploration/
│   ├── graph_construction/
│   ├── demos/
│   └── training/
│
├── outputs/                      # Generated (gitignored — reproducible from scripts)
│   ├── final_graph_ready/
│   ├── runs/
│   └── shap_*/
│
├── checkpoints/                  # Model checkpoints (gitignored)
│
└── archive/                      # Superseded files (reference only)
    ├── README.md
    ├── scripts/
    └── jobs/
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
| `final_graph_ready` | `outputs/final_graph_ready/` | Stratified normal sample (31,513 normals). Not used for main thesis results. |
| `matched_normals_v1` | `outputs/final_graph_ready_matched_normals_v1/` | Complexity-matched normals (cap ×20). **Primary thesis benchmark.** |

Do not compare numbers across benchmark families without explicit labeling.

---

## Reproducibility Notes

- **Primary seeds:** 42, 123, 7 (all multi-seed thesis results)
- Each seed initializes PyTorch (`torch.manual_seed`) and NumPy (`np.random.seed`)
- K-fold partition uses `--fold_seed` (default 0) for deterministic repo-to-fold assignment
- `split_index.csv` is deterministic given the repo list; no random seed is needed for splits
- The frozen training protocol is in `configs/default_training.yaml`

---

## What Is and Is Not Versioned

**Tracked in git:**
- `src/` — model, dataset, graph builder
- `scripts/`, `experiments/`, `jobs/` — all pipeline scripts
- `configs/` — training configuration
- `notebooks/`
- `README.md`, `requirements.txt`, `environment.yml`, `setup.py`

**Not tracked (gitignored):**
- `data/`, `data_new/` — raw CSVs and embeddings (symlinks to ICVul/ICVul++)
- `outputs/` — built graphs, SHAP outputs, split files
- `checkpoints/` — model checkpoints
- `*.npy`, `*.pkl`, `*.pt`

---

## License

MIT License. See `LICENSE` for details.

The ICVul++ dataset is subject to separate license terms. See the ICVulPP repository before redistribution or commercial use.
