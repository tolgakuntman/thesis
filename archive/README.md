# Archive — Superseded and Legacy Files

This directory contains scripts and job files that have been superseded by newer
versions or are no longer part of the main pipeline. They are preserved for
reference and historical reproducibility.

**Do not use these files for current experiments.**

---

## archive/scripts/

| File | Superseded By | Notes |
|------|--------------|-------|
| `build_graphs_v0.py` | `scripts/data/build_graphs_final.py` | Original graph builder (legacy topology) |
| `build_graphs_v2.py` | `scripts/data/build_graphs_final.py` | v2 graph builder (anonymized nodes) |
| `create_split_index_v0.py` | `scripts/data/create_splits.py` | Original split creator |
| `create_split_index_v2.py` | `scripts/data/create_splits.py` | v2 split creator |
| `compute_perrepo_scaler_v2.py` | `scripts/data/compute_scalers.py` | v2 scaler (different schema) |
| `cv_train.py` | `scripts/training/kfold_train.py` | Simple cross-validation (superseded by repo-grouped k-fold) |
| `audit_graph_ready_v2.py` | — | v2-specific graph audit tool |
| `validate_features_v2.py` | — | v2-specific feature validator |
| `visualize_graphs_v2.py` | — | v2-specific graph visualizer |
| `audit_final_package.py` | — | One-off audit of the final graph package |
| `find_candidates.py` | — | Early-stage candidate identification script |

---

## archive/jobs/

| File | Superseded By | Notes |
|------|--------------|-------|
| `kfold_train.slurm` | `jobs/training/kfold_final.slurm` | Baseline k-fold (no SHAP ablations) |
| `kfold5_pruned.slurm` | `jobs/training/kfold_final.slurm` | Pruned features experiment |
| `kfold5_pruned_v2.slurm` | `jobs/training/kfold_final.slurm` | Pruned v2 features experiment |
| `kfold_reg.slurm` | `jobs/training/kfold_final.slurm` | Regularization sweep |
| `kfold_aug.slurm` | `jobs/training/kfold_final.slurm` | Data augmentation experiment |
| `shap_all.slurm` | `jobs/analysis/shap_top10_ablation.slurm` | Early SHAP runs |
| `shap_full.slurm` | `jobs/analysis/shap_top10_ablation.slurm` | Full SHAP run (superseded) |
| `shap_full_v2.slurm` | `jobs/analysis/shap_top10_ablation.slurm` | v2 SHAP run |
| `train_sage_v2_full.slurm` | `jobs/training/kfold_final.slurm` | v2-graph-specific training |
