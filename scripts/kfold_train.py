"""
scripts/kfold_train.py

Repo-grouped k-fold cross-validation for the VCC GNN.

Strategy
--------
  Pool  = all commits NOT in the fixed test repos (repo_split != 'test')
  Repos are partitioned into k groups using greedy balanced assignment:
    sort repos by commit count descending, assign each to the fold with
    the fewest commits so far.  This keeps fold sizes balanced without
    leaking repo identity between train and val within any fold.

  Fixed test repos (repo_split == 'test') are held out entirely.
  Each fold's best model is evaluated on this fixed test set so we can
  check whether val metrics track test metrics across folds.

Why repo-grouped?
  The original repo_split protocol tests cross-project generalisation.
  Doing commit-level k-fold would put commits from the same project in
  both train and val, destroying that guarantee.  Repo-grouped k-fold
  preserves it: within every fold, the model has never seen any commit
  from the val repos.

Output
------
  checkpoints/<run_name>/
    kfold_config.json          — args + fold→repo mapping
    fold_0/
      metrics.csv              — per-epoch train/val metrics
      best.pt                  — highest val AUC-PR checkpoint
      fold_val_results.json    — final val metrics (best model)
      fold_test_results.json   — final test metrics (best model)
    fold_1/ ...
    kfold_summary.json         — per-fold table + mean ± std

Usage (same args as train.py, plus --k and --fold_seed)
-------
  python scripts/kfold_train.py --k 5 --seed 42
  python scripts/kfold_train.py --k 5 --perrepo_norm \\
      --graphs_dir $TMPDIR/graphs --split_index /path/to/split_index.csv

Protocol defaults match the frozen Phase-6 protocol:
  epochs=60, patience=10, warmup=3, batch=128,
  focal_alpha=0.65, focal_gamma=1.5, dropout=0.4, wd=5e-4
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_dataset import VulnCommitDataset, V2_SPLIT_INDEX, make_loader
from src.model import CommitMLP, FocalLoss, HeteroRGCN, HeteroSAGE


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Repo-grouped k-fold cross-validation for the VCC GNN"
    )

    # K-fold
    p.add_argument("--k", type=int, default=5,
                   help="Number of cross-validation folds (default: 5)")
    p.add_argument("--fold_seed", type=int, default=0,
                   help="RNG seed for fold assignment (reproducible repo grouping)")

    # Data
    p.add_argument("--graphs_dir",  default=None,
                   help="Override path to .pt graph files (set to $TMPDIR/graphs on HPC)")
    p.add_argument("--split_index", default=None,
                   help="Override path to split_index.csv (must have repo_split column)")

    # Model
    p.add_argument("--model",        default="sage", choices=["sage", "rgcn", "mlp"])
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--feat_dropout", type=float, default=0.0)
    p.add_argument("--num_bases",    type=int,   default=4,
                   help="RGCN basis matrices (only used with --model rgcn)")

    # Structural exclusions (true graph removal, same as train.py)
    p.add_argument("--exclude_node_types", nargs="*", default=[])
    p.add_argument("--exclude_edge_rels",  nargs="*", default=[])

    # Training protocol (defaults = frozen Phase-6 protocol)
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--focal_gamma",   type=float, default=1.5)
    p.add_argument("--focal_alpha",   type=float, default=0.65)
    p.add_argument("--seed",          type=int,   default=42,
                   help="Seed for model weight init and data sampling (same across all folds)")

    # Ablations (zeroing, not structural removal)
    p.add_argument("--ablate_code_emb",       action="store_true")
    p.add_argument("--ablate_fn_categorical",  action="store_true")
    p.add_argument("--ablate_fn_code_metrics", action="store_true")
    p.add_argument("--ablate_sdlc",           action="store_true")
    p.add_argument("--ablate_hunk_emb",       action="store_true")
    p.add_argument("--ablate_hunk_metrics",   action="store_true",
                   help="Zero hunk dims 0-1 (complexity, token_count); SHAP phi~0.001")
    p.add_argument("--ablate_commit_merge",   action="store_true",
                   help="Zero commit dim 1 (merge flag); SHAP phi=0.0015, grad=0.000028")
    p.add_argument("--ablate_developer",      action="store_true")
    p.add_argument("--ablate_commit_stats",   action="store_true")
    p.add_argument("--ablate_pruned_features", action="store_true",
                   help="Zero 11 dims selected by SHAP+correlation: fn[0], iss[0,2], pr[0,2], "
                        "tag[0,3], dev[6], commit[2,5,6]")
    p.add_argument("--ablate_pruned_v2", action="store_true",
                   help="Round-2: zero 10 bottom-phi dims (phi<=0.007) — fn[3], tag[1], "
                        "dev[5,7,8], commit[3,4,7,10], pr[3]. Use with --ablate_pruned_features.")
    p.add_argument("--ablate_single_dim", default=None,
                   help="Zero one feature dim: 'node_type:dim_idx', e.g. 'file:0'. "
                        "Used for per-feature SHAP-guided ablations.")
    p.add_argument("--perrepo_norm",          action="store_true")
    p.add_argument("--perrepo_scaler",        default=None)

    # Graph augmentation (training only)
    p.add_argument("--aug_node_mask_p", type=float, default=0.0,
                   help="Per-node-type masking probability at training time (0=off)")
    p.add_argument("--aug_edge_drop_p", type=float, default=0.0,
                   help="Per-edge drop fraction at training time (0=off)")

    # Checkpointing
    p.add_argument("--run_name",   default=None,
                   help="Experiment name (default: auto-generated)")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="Save epoch_NNN.pt every N epochs (0=disable, recommended for k-fold "
                        "to avoid filling disk with k×N checkpoints)")

    # Hardware
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None)

    return p.parse_args()


# ── fold assignment ───────────────────────────────────────────────────────────

def assign_repo_folds(pool: pd.DataFrame, k: int) -> dict:
    """
    Greedy balanced repo → fold assignment.

    Sort repos by commit count descending; assign each to the fold with the
    fewest cumulative commits.  This approximates equal fold sizes without
    requiring an ILP solver.

    Returns {repo_url: fold_id (0-indexed)}.
    """
    repo_counts = pool.groupby("repo_url").size().sort_values(ascending=False)
    fold_sizes = [0] * k
    assignments: dict[str, int] = {}
    for repo, count in repo_counts.items():
        i = int(np.argmin(fold_sizes))
        assignments[repo] = i
        fold_sizes[i] += int(count)
    return assignments


# ── metrics + training utilities (mirrors train.py) ──────────────────────────

def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    probs = 1.0 / (1.0 + np.exp(-logits))
    auc_pr  = average_precision_score(labels, probs)
    auc_roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    preds_05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(labels, preds_05, zero_division=0)
    prec  = precision_score(labels, preds_05, zero_division=0)
    rec   = recall_score(labels, preds_05, zero_division=0)
    mcc   = matthews_corrcoef(labels, preds_05)
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds_t = (probs >= t).astype(int)
        f1_t = f1_score(labels, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, float(t)
    return {
        "f1":          round(f1_05, 4),
        "f1_best":     round(best_f1, 4),
        "precision":   round(prec, 4),
        "recall":      round(rec, 4),
        "auc_pr":      round(auc_pr, 4),
        "auc_roc":     round(auc_roc, 4),
        "mcc":         round(mcc, 4),
        "thresh_best": round(best_thresh, 2),
    }


def run_epoch(model, loader, criterion, optimizer, device, is_train: bool) -> dict:
    model.train(is_train)
    total_loss = 0.0
    all_logits, all_labels = [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                batch.x_dict,
                batch.edge_index_dict,
                getattr(batch, "edge_attr_dict", None),
            )
            labels = batch.y.squeeze(-1)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.detach().cpu().float().numpy())
            all_labels.append(labels.cpu().numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = round(total_loss / max(len(all_labels), 1), 5)
    return metrics


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(args, device):
    if args.model == "sage":
        return HeteroSAGE(
            hidden=args.hidden,
            dropout=args.dropout,
            feat_dropout=args.feat_dropout,
            exclude_node_types=args.exclude_node_types or [],
            exclude_edge_rels=args.exclude_edge_rels or [],
        ).to(device)
    elif args.model == "rgcn":
        return HeteroRGCN(
            hidden=args.hidden,
            dropout=args.dropout,
            num_bases=args.num_bases,
            exclude_node_types=args.exclude_node_types or [],
            exclude_edge_rels=args.exclude_edge_rels or [],
        ).to(device)
    return CommitMLP(hidden=args.hidden, dropout=args.dropout).to(device)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_auc_pr, history):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_auc_pr":      best_val_auc_pr,
        "metrics_history":      history,
    }, path)


# ── single fold ───────────────────────────────────────────────────────────────

def train_one_fold(
    args,
    fold_id: int,
    fold_dir: Path,
    train_records: pd.DataFrame,
    val_records: pd.DataFrame,
    test_records: pd.DataFrame,
    ds_kwargs: dict,
    device,
    pin_memory: bool,
) -> tuple[dict, dict]:
    """
    Train one fold and evaluate on both fold-val and fixed test.
    Returns (val_metrics, test_metrics) from the best checkpoint.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_aug: dict = {"is_train": True}
    if args.aug_node_mask_p > 0.0:
        train_aug["aug_node_mask_p"] = args.aug_node_mask_p
    if args.aug_edge_drop_p > 0.0:
        train_aug["aug_edge_drop_p"] = args.aug_edge_drop_p
    train_ds = VulnCommitDataset(records_df=train_records, **ds_kwargs, **train_aug)
    val_ds   = VulnCommitDataset(records_df=val_records,   **ds_kwargs)

    num_workers = args.num_workers
    train_loader = make_loader(train_ds, args.batch_size, num_workers,
                               is_train=True, pin_memory=pin_memory)
    val_loader   = make_loader(val_ds,   args.batch_size, num_workers,
                               is_train=False, pin_memory=pin_memory)

    # Build model and materialise LazyLinear weights via a dummy forward pass
    model = build_model(args, device)
    _dummy = next(iter(train_loader))
    with torch.no_grad():
        model(
            {k: v.to(device) for k, v in _dummy.x_dict.items()},
            {k: v.to(device) for k, v in _dummy.edge_index_dict.items()},
            edge_attr_dict={
                et: _dummy[et].edge_attr.to(device)
                for et in _dummy.edge_types
                if hasattr(_dummy[et], "edge_attr")
            },
        )
    del _dummy
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Fold {fold_id} | {args.model.upper()} params: {n_params:,}", flush=True)

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    fold_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc_pr  = 0.0
    best_val_mcc     = -1.0
    metrics_history  = []
    no_improve_count = 0

    csv_path   = fold_dir / "metrics.csv"
    csv_fields = ["epoch", "split", "loss", "f1", "f1_best", "precision", "recall",
                  "auc_pr", "auc_roc", "mcc", "thresh_best", "lr", "elapsed_s"]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    print(f"\n  {'Ep':>4}  {'Split':6}  {'Loss':>8}  {'F1':>6}  {'F1*':>6}  "
          f"{'AUC-PR':>7}  {'AUC-ROC':>8}  {'MCC':>6}")
    print("  " + "-" * 68)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        scheduler.step()
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)
        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        def _row(split_name, m):
            return {**m, "epoch": epoch, "split": split_name,
                    "lr": round(cur_lr, 7), "elapsed_s": round(elapsed, 1)}
        csv_writer.writerow(_row("train", train_m))
        csv_writer.writerow(_row("val",   val_m))
        csv_file.flush()
        metrics_history.append({"epoch": epoch, "train": train_m, "val": val_m})

        def _fmt(m):
            return (f"{m['loss']:8.5f}  {m['f1']:6.4f}  {m['f1_best']:6.4f}  "
                    f"{m['auc_pr']:7.4f}  {m['auc_roc']:8.4f}  {m['mcc']:6.4f}")
        print(f"  {epoch:4d}  {'train':6}  {_fmt(train_m)}")
        print(f"  {'':4}  {'val':6}  {_fmt(val_m)}")

        val_auc_pr = val_m["auc_pr"]
        val_mcc    = val_m["mcc"]
        is_best    = (val_auc_pr > best_val_auc_pr or
                      (val_auc_pr == best_val_auc_pr and val_mcc > best_val_mcc))
        if is_best:
            best_val_auc_pr = val_auc_pr
            best_val_mcc    = val_mcc
            no_improve_count = 0
            save_checkpoint(fold_dir / "best.pt", model, optimizer, scheduler,
                            epoch, best_val_auc_pr, metrics_history)
            print(f"  ** New best val AUC-PR: {best_val_auc_pr:.4f}  MCC: {best_val_mcc:.4f}", flush=True)
        else:
            no_improve_count += 1

        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            save_checkpoint(fold_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                            scheduler, epoch, best_val_auc_pr, metrics_history)

        if no_improve_count >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    csv_file.close()
    print(f"\n  Fold {fold_id} training done.  Best val AUC-PR: {best_val_auc_pr:.4f}", flush=True)

    # Reload best checkpoint
    ckpt = torch.load(fold_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Final val evaluation with best model
    val_m_final = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
    val_m_final["fold"]             = fold_id
    val_m_final["best_val_auc_pr"]  = best_val_auc_pr
    with open(fold_dir / "fold_val_results.json", "w") as f:
        json.dump(val_m_final, f, indent=2)

    # Free train/val loaders before test evaluation to reduce peak memory on GPU
    del train_loader, val_loader, train_ds, val_ds

    # Test evaluation (fixed test set)
    print(f"  Evaluating fold {fold_id} on fixed test set...", flush=True)
    test_ds     = VulnCommitDataset(records_df=test_records, **ds_kwargs)
    test_loader = make_loader(test_ds, args.batch_size, num_workers,
                              is_train=False, pin_memory=pin_memory)
    test_m = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)
    test_m["fold"] = fold_id
    print(f"  Fold {fold_id} test AUC-PR: {test_m['auc_pr']:.4f}  "
          f"F1*: {test_m['f1_best']:.4f}  MCC: {test_m['mcc']:.4f}", flush=True)
    with open(fold_dir / "fold_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    return val_m_final, test_m


# ── aggregate across folds ────────────────────────────────────────────────────

def aggregate_fold_results(fold_results: list[dict], prefix: str) -> dict:
    """Compute mean ± std across folds, excluding non-numeric keys."""
    skip = {"fold", "best_val_auc_pr"}
    agg: dict = {"n_folds": len(fold_results)}
    for key in fold_results[0]:
        if key in skip:
            continue
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if vals:
            agg[f"{prefix}{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{prefix}{key}_std"]  = round(float(np.std(vals)),  4)
    return agg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if os.name == "nt" and args.num_workers > 0:
        print(f"Windows detected; forcing num_workers=0 (was {args.num_workers})")
        args.num_workers = 0
    pin_memory = (device.type == "cuda")

    # Auto run name
    if args.run_name is None:
        parts = [f"kfold{args.k}", args.model, f"h{args.hidden}", f"drop{args.dropout}"]
        if args.perrepo_norm:         parts.append("perrepo")
        if args.ablate_sdlc:          parts.append("noSDLC")
        if args.ablate_code_emb:      parts.append("noCode")
        if args.ablate_developer:     parts.append("noDev")
        if args.exclude_node_types:   parts.append("excl_" + "_".join(sorted(args.exclude_node_types)))
        args.run_name = "_".join(parts)

    ckpt_dir = ROOT / args.output_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun: {args.run_name}")
    print(f"k={args.k}, seed={args.seed}, fold_seed={args.fold_seed}")
    print(f"Checkpoints -> {ckpt_dir}")

    # Load split index
    split_index_path = Path(args.split_index) if args.split_index else V2_SPLIT_INDEX
    df = pd.read_csv(split_index_path)
    print(f"\nSplit index: {len(df):,} rows  ({split_index_path})")

    # Fixed test set — held out during all fold training
    test_records = df[df["repo_split"] == "test"][["hash", "label", "repo_url"]].copy()
    test_repo_names = sorted(
        test_records["repo_url"].str.split("/").str[-1].unique()
    )
    print(f"Fixed test: {len(test_records):,} commits, "
          f"{int(test_records['label'].sum()):,} VCC  "
          f"repos: {test_repo_names}")

    # K-fold pool: everything except the fixed test repos
    pool = df[df["repo_split"] != "test"].copy()
    n_pool_vcc = int(pool["label"].sum())
    print(f"K-fold pool: {len(pool):,} commits, {n_pool_vcc:,} VCC, "
          f"{pool['repo_url'].nunique()} repos")

    # Assign repos to k folds (deterministic given fold_seed via sort stability)
    np.random.seed(args.fold_seed)
    fold_assignments = assign_repo_folds(pool, args.k)
    pool = pool.copy()
    pool["fold_id"] = pool["repo_url"].map(fold_assignments)

    # Report fold composition
    print(f"\nFold assignment  (k={args.k}, fold_seed={args.fold_seed}):")
    for fid in range(args.k):
        fold_rows  = pool[pool["fold_id"] == fid]
        repo_names = sorted(fold_rows["repo_url"].str.split("/").str[-1].unique())
        n_commits  = len(fold_rows)
        n_vcc      = int(fold_rows["label"].sum())
        print(f"  Fold {fid}: {n_commits:5,} commits, {n_vcc:4,} VCC "
              f"({n_vcc/n_commits*100:.1f}%)  repos: {repo_names}")

    # Save full config (including fold → repo mapping)
    config = {
        **{k: v for k, v in vars(args).items()},
        "fold_assignments":  {str(r): int(f) for r, f in fold_assignments.items()},
        "test_repos":        test_repo_names,
        "pool_n":            int(len(pool)),
        "test_n":            int(len(test_records)),
    }
    with open(ckpt_dir / "kfold_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Dataset kwargs shared across all folds
    ds_kwargs: dict = {}
    if args.graphs_dir:
        ds_kwargs["graphs_dir"] = args.graphs_dir
    if args.ablate_code_emb:
        ds_kwargs["ablate_code_emb"] = True
    if args.ablate_fn_categorical:
        ds_kwargs["ablate_fn_categorical"] = True
    if args.ablate_fn_code_metrics:
        ds_kwargs["ablate_fn_code_metrics"] = True
    if args.ablate_sdlc:
        ds_kwargs["ablate_sdlc"] = True
    if args.ablate_hunk_emb:
        ds_kwargs["ablate_hunk_emb"] = True
    if args.ablate_hunk_metrics:
        ds_kwargs["ablate_hunk_metrics"] = True
    if args.ablate_commit_merge:
        ds_kwargs["ablate_commit_merge"] = True
    if args.ablate_developer:
        ds_kwargs["ablate_developer"] = True
    if args.ablate_commit_stats:
        ds_kwargs["ablate_commit_stats"] = True
    if args.ablate_pruned_features:
        ds_kwargs["ablate_pruned_features"] = True
    if args.ablate_pruned_v2:
        ds_kwargs["ablate_pruned_v2"] = True
    if args.ablate_single_dim:
        ds_kwargs["ablate_single_dim"] = args.ablate_single_dim
    if args.perrepo_norm:
        ds_kwargs["perrepo_norm"] = True
        if args.perrepo_scaler:
            ds_kwargs["perrepo_scaler_path"] = args.perrepo_scaler

    # ── run all k folds ──────────────────────────────────────────────────────
    all_val_results:  list[dict] = []
    all_test_results: list[dict] = []

    for fold_id in range(args.k):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_id + 1} / {args.k}")
        print(f"{'='*70}")

        val_records   = pool[pool["fold_id"] == fold_id][["hash", "label", "repo_url"]].copy()
        train_records = pool[pool["fold_id"] != fold_id][["hash", "label", "repo_url"]].copy()

        n_train_vcc = int(train_records["label"].sum())
        n_val_vcc   = int(val_records["label"].sum())
        print(f"  Train: {len(train_records):,} commits, {n_train_vcc:,} VCC  "
              f"| Val: {len(val_records):,} commits, {n_val_vcc:,} VCC  "
              f"| Test: {len(test_records):,} (fixed)")

        fold_dir = ckpt_dir / f"fold_{fold_id}"
        val_m, test_m = train_one_fold(
            args, fold_id, fold_dir,
            train_records, val_records, test_records,
            ds_kwargs, device, pin_memory,
        )
        all_val_results.append(val_m)
        all_test_results.append(test_m)

    # ── aggregate & report ───────────────────────────────────────────────────
    val_agg  = aggregate_fold_results(all_val_results,  "val_")
    test_agg = aggregate_fold_results(all_test_results, "test_")

    summary = {
        "run_name":       args.run_name,
        "k":              args.k,
        "seed":           args.seed,
        "fold_seed":      args.fold_seed,
        "per_fold_val":   all_val_results,
        "per_fold_test":  all_test_results,
        "aggregate":      {**val_agg, **test_agg},
    }
    with open(ckpt_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    agg = summary["aggregate"]
    print(f"\n{'='*70}")
    print(f"K-FOLD SUMMARY  (k={args.k}, seed={args.seed})")
    print(f"{'='*70}")
    hdr = (f"  {'Fold':>4}  {'Val AUC-PR':>10}  {'Val F1*':>8}  {'Val MCC':>8}  "
           f"{'Test AUC-PR':>11}  {'Test F1*':>8}  {'Test MCC':>8}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for i, (v, t) in enumerate(zip(all_val_results, all_test_results)):
        print(f"  {i:4d}  {v['auc_pr']:10.4f}  {v['f1_best']:8.4f}  {v['mcc']:8.4f}  "
              f"{t['auc_pr']:11.4f}  {t['f1_best']:8.4f}  {t['mcc']:8.4f}")
    print(sep)
    print(f"  {'mean':4}  "
          f"{agg.get('val_auc_pr_mean', 0):10.4f}  "
          f"{agg.get('val_f1_best_mean', 0):8.4f}  "
          f"{agg.get('val_mcc_mean', 0):8.4f}  "
          f"{agg.get('test_auc_pr_mean', 0):11.4f}  "
          f"{agg.get('test_f1_best_mean', 0):8.4f}  "
          f"{agg.get('test_mcc_mean', 0):8.4f}")
    print(f"  {'std':4}  "
          f"{agg.get('val_auc_pr_std', 0):10.4f}  "
          f"{agg.get('val_f1_best_std', 0):8.4f}  "
          f"{agg.get('val_mcc_std', 0):8.4f}  "
          f"{agg.get('test_auc_pr_std', 0):11.4f}  "
          f"{agg.get('test_f1_best_std', 0):8.4f}  "
          f"{agg.get('test_mcc_std', 0):8.4f}")
    print(f"\nFull results -> {ckpt_dir / 'kfold_summary.json'}")


if __name__ == "__main__":
    main()
