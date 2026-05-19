"""
scripts/cv_train.py

Two alternative cross-validation strategies:

  --mode logo
    Leave-One-Repo-Out on the k largest repos in the repo-split pool.
    Pool  = repo_split != 'test'       Test = repo_split == 'test' (fixed, same as kfold_train)
    Fold i: val = top-repo-i's commits, train = ALL other pool commits (not just the other top-k).
    Motivation: maximum training data per fold; tests generalisation to each specific large repo.

  --mode temporal
    Walk-forward temporal k-fold.
    Pool  = temporal_split in ['train','val']   Test = temporal_split == 'test'
    Sorted by author_date, divided into 2k equal chunks.
    Fold i (i=0..k-1): train = chunks[0..k-1+i],  val = chunk[k+i].
    Minimum training size = k/2k = 50% of the pool.
    Motivation: validation is always STRICTLY LATER than training — no temporal leakage.
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
        description="LOGO or temporal walk-forward cross-validation for the VCC GNN"
    )

    p.add_argument("--mode", required=True, choices=["logo", "temporal"],
                   help="'logo': leave-one-repo-out on top-k repos; "
                        "'temporal': walk-forward temporal k-fold")

    # CV parameters
    p.add_argument("--k", type=int, default=5,
                   help="LOGO: number of repos to leave out (default 10 recommended); "
                        "temporal: number of walk-forward folds (default 5)")

    # Data
    p.add_argument("--graphs_dir",  default=None)
    p.add_argument("--split_index", default=None,
                   help="Path to split_index.csv (must have repo_split + temporal_split + author_date)")

    # Model
    p.add_argument("--model",        default="sage", choices=["sage", "rgcn", "mlp"])
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--dropout",      type=float, default=0.4)
    p.add_argument("--feat_dropout", type=float, default=0.0)
    p.add_argument("--num_bases",    type=int,   default=4)

    p.add_argument("--exclude_node_types", nargs="*", default=[])
    p.add_argument("--exclude_edge_rels",  nargs="*", default=[])

    # Training protocol
    p.add_argument("--epochs",        type=int,   default=60)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=5e-4)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--warmup_epochs", type=int,   default=3)
    p.add_argument("--focal_gamma",   type=float, default=1.5)
    p.add_argument("--focal_alpha",   type=float, default=0.65)
    p.add_argument("--seed",          type=int,   default=42)

    # Ablations
    p.add_argument("--ablate_code_emb",       action="store_true")
    p.add_argument("--ablate_fn_categorical",  action="store_true")
    p.add_argument("--ablate_fn_code_metrics", action="store_true")
    p.add_argument("--ablate_sdlc",           action="store_true")
    p.add_argument("--ablate_hunk_emb",       action="store_true")
    p.add_argument("--ablate_hunk_metrics",   action="store_true")
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
    p.add_argument("--perrepo_norm",          action="store_true")
    p.add_argument("--perrepo_scaler",        default=None)

    # Checkpointing
    p.add_argument("--run_name",   default=None)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--ckpt_every", type=int, default=0)

    # Hardware
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None)

    return p.parse_args()


# ── fold assignment ───────────────────────────────────────────────────────────

def assign_logo_folds(pool: pd.DataFrame, k: int) -> list[str]:
    """
    Return the k repos with the most commits from pool (descending).
    Each becomes one LOGO fold: val = that repo, train = all other pool commits.
    """
    top = (
        pool.groupby("repo_url")
        .size()
        .sort_values(ascending=False)
        .head(k)
        .index.tolist()
    )
    return top


def assign_temporal_folds(pool: pd.DataFrame, k: int) -> pd.Series:
    """
    Sort pool by author_date, divide into 2k equal chunks (by commit count).
    Return a Series of chunk_id (0-indexed, same index as pool).

    Walk-forward schedule: fold i uses chunks[0..k-1+i] as train, chunk[k+i] as val.
    Minimum training fraction: k / 2k = 50%.
    """
    n_chunks = 2 * k
    pool_sorted = pool.sort_values("author_date").copy()
    n = len(pool_sorted)
    # Assign chunk_id uniformly by rank
    chunk_ids = np.minimum(
        np.arange(n) * n_chunks // n,
        n_chunks - 1,
    )
    pool_sorted["chunk_id"] = chunk_ids
    return pool_sorted["chunk_id"]


# ── training utilities ────────────────────────────────────────────────────────

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


def aggregate_fold_results(fold_results: list[dict], prefix: str) -> dict:
    skip = {"fold", "best_val_auc_pr", "val_repo"}
    agg: dict = {"n_folds": len(fold_results)}
    for key in fold_results[0]:
        if key in skip:
            continue
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if vals:
            agg[f"{prefix}{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{prefix}{key}_std"]  = round(float(np.std(vals)),  4)
    return agg


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
    fold_label: str = "",
) -> tuple[dict, dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = VulnCommitDataset(records_df=train_records, **ds_kwargs)
    val_ds   = VulnCommitDataset(records_df=val_records,   **ds_kwargs)

    num_workers = args.num_workers
    train_loader = make_loader(train_ds, args.batch_size, num_workers,
                               is_train=True,  pin_memory=pin_memory)
    val_loader   = make_loader(val_ds,   args.batch_size, num_workers,
                               is_train=False, pin_memory=pin_memory)

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
    lbl = f" [{fold_label}]" if fold_label else ""
    print(f"  Fold {fold_id}{lbl} | {args.model.upper()} params: {n_params:,}", flush=True)

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

    ckpt = torch.load(fold_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_m_final = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)
    val_m_final["fold"] = fold_id
    val_m_final["best_val_auc_pr"] = best_val_auc_pr
    if fold_label:
        val_m_final["val_repo"] = fold_label
    with open(fold_dir / "fold_val_results.json", "w") as f:
        json.dump(val_m_final, f, indent=2)

    del train_loader, val_loader, train_ds, val_ds

    print(f"  Evaluating fold {fold_id} on fixed test set...", flush=True)
    test_ds     = VulnCommitDataset(records_df=test_records, **ds_kwargs)
    test_loader = make_loader(test_ds, args.batch_size, num_workers,
                              is_train=False, pin_memory=pin_memory)
    test_m = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)
    test_m["fold"] = fold_id
    if fold_label:
        test_m["val_repo"] = fold_label
    print(f"  Fold {fold_id} test AUC-PR: {test_m['auc_pr']:.4f}  "
          f"F1*: {test_m['f1_best']:.4f}  MCC: {test_m['mcc']:.4f}", flush=True)
    with open(fold_dir / "fold_test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    return val_m_final, test_m


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if os.name == "nt" and args.num_workers > 0:
        print(f"Windows: forcing num_workers=0")
        args.num_workers = 0
    pin_memory = (device.type == "cuda")

    # Auto run name
    if args.run_name is None:
        prefix = "logo" if args.mode == "logo" else "kfold_temporal"
        args.run_name = f"{prefix}{args.k}_sage_repo"

    ckpt_dir = ROOT / args.output_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nMode: {args.mode.upper()}   Run: {args.run_name}")
    print(f"k={args.k}, seed={args.seed}")
    print(f"Checkpoints -> {ckpt_dir}")

    split_index_path = Path(args.split_index) if args.split_index else V2_SPLIT_INDEX
    df = pd.read_csv(split_index_path)
    print(f"\nSplit index: {len(df):,} rows  ({split_index_path})")

    # ── mode-specific pool and test set ──────────────────────────────────────

    if args.mode == "logo":
        # Pool = everything not in the fixed repo test set
        pool = df[df["repo_split"] != "test"].copy()
        test_records = df[df["repo_split"] == "test"][["hash", "label", "repo_url"]].copy()
        test_src = "repo_split == 'test'"

        top_repos = assign_logo_folds(pool, args.k)
        fold_specs = []  # list of (fold_id, val_repo, train_mask, val_mask)
        for i, repo in enumerate(top_repos):
            val_mask   = pool["repo_url"] == repo
            train_mask = ~val_mask
            n_val  = int(val_mask.sum())
            n_vcc  = int(pool[val_mask]["label"].sum())
            n_train = int(train_mask.sum())
            name = repo.split("/")[-1]
            print(f"  Fold {i}: val = {name} ({n_val:,} commits, {n_vcc} VCC) "
                  f"| train = {n_train:,} commits")
            fold_specs.append((i, repo, train_mask, val_mask))

    else:  # temporal
        # Pool = temporal train + val
        pool = df[df["temporal_split"].isin(["train", "val"])].copy()
        test_records = df[df["temporal_split"] == "test"][["hash", "label", "repo_url"]].copy()
        test_src = "temporal_split == 'test'"

        # Sort pool by date and assign chunk_ids
        pool["author_date"] = pd.to_datetime(pool["author_date"], errors="coerce")
        pool = pool.sort_values("author_date").reset_index(drop=True)
        n_chunks = 2 * args.k
        pool["chunk_id"] = np.minimum(
            np.arange(len(pool)) * n_chunks // len(pool),
            n_chunks - 1,
        )
        date_range = (pool["author_date"].min().date(), pool["author_date"].max().date())
        print(f"  Temporal pool: {len(pool):,} commits, {int(pool['label'].sum())} VCC")
        print(f"  Date range: {date_range[0]} to {date_range[1]}")
        print(f"  {n_chunks} chunks of ~{len(pool)//n_chunks} commits each")
        print()

        fold_specs = []
        for i in range(args.k):
            train_chunk_max = args.k - 1 + i      # inclusive upper chunk for train
            val_chunk       = args.k + i
            train_mask = pool["chunk_id"] <= train_chunk_max
            val_mask   = pool["chunk_id"] == val_chunk
            n_train = int(train_mask.sum())
            n_val   = int(val_mask.sum())
            n_vcc_train = int(pool[train_mask]["label"].sum())
            n_vcc_val   = int(pool[val_mask]["label"].sum())
            t_range_val = pool[val_mask]["author_date"]
            date_val = f"{t_range_val.min().date()} .. {t_range_val.max().date()}"
            print(f"  Fold {i}: train = {n_train:,} commits ({n_vcc_train} VCC, "
                  f"chunks 0..{train_chunk_max}) | val = {n_val} commits ({n_vcc_val} VCC, "
                  f"chunk {val_chunk}: {date_val})")
            fold_specs.append((i, f"chunk_{val_chunk}", train_mask, val_mask))

    test_repo_names = sorted(test_records["repo_url"].str.split("/").str[-1].unique())
    print(f"\nFixed test ({test_src}): {len(test_records):,} commits, "
          f"{int(test_records['label'].sum())} VCC  repos: {test_repo_names}")

    # ── dataset kwargs ────────────────────────────────────────────────────────

    ds_kwargs: dict = {}
    if args.graphs_dir:
        ds_kwargs["graphs_dir"] = args.graphs_dir
    if args.ablate_code_emb:       ds_kwargs["ablate_code_emb"] = True
    if args.ablate_fn_categorical:  ds_kwargs["ablate_fn_categorical"] = True
    if args.ablate_fn_code_metrics: ds_kwargs["ablate_fn_code_metrics"] = True
    if args.ablate_sdlc:            ds_kwargs["ablate_sdlc"] = True
    if args.ablate_hunk_emb:        ds_kwargs["ablate_hunk_emb"] = True
    if args.ablate_hunk_metrics:    ds_kwargs["ablate_hunk_metrics"] = True
    if args.ablate_commit_merge:    ds_kwargs["ablate_commit_merge"] = True
    if args.ablate_developer:       ds_kwargs["ablate_developer"] = True
    if args.ablate_commit_stats:    ds_kwargs["ablate_commit_stats"] = True
    if args.ablate_pruned_features: ds_kwargs["ablate_pruned_features"] = True
    if args.ablate_pruned_v2:       ds_kwargs["ablate_pruned_v2"] = True
    if args.perrepo_norm:
        ds_kwargs["perrepo_norm"] = True
        if args.perrepo_scaler:
            ds_kwargs["perrepo_scaler_path"] = args.perrepo_scaler

    # ── save config ───────────────────────────────────────────────────────────

    config = {
        **{k: v for k, v in vars(args).items()},
        "test_repos": test_repo_names,
        "pool_n":     int(len(pool)),
        "test_n":     int(len(test_records)),
    }
    if args.mode == "logo":
        config["logo_repos"] = [r.split("/")[-1] for r in top_repos]
    with open(ckpt_dir / "kfold_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── run all folds ─────────────────────────────────────────────────────────

    all_val_results:  list[dict] = []
    all_test_results: list[dict] = []

    for fold_id, fold_label, train_mask, val_mask in fold_specs:
        print(f"\n{'='*70}")
        print(f"FOLD {fold_id + 1} / {args.k}  [{fold_label}]")
        print(f"{'='*70}")

        train_records = pool[train_mask][["hash", "label", "repo_url"]].copy()
        val_records   = pool[val_mask][["hash", "label", "repo_url"]].copy()

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
            fold_label=fold_label,
        )
        all_val_results.append(val_m)
        all_test_results.append(test_m)

    # ── aggregate & report ────────────────────────────────────────────────────

    val_agg  = aggregate_fold_results(all_val_results,  "val_")
    test_agg = aggregate_fold_results(all_test_results, "test_")

    summary = {
        "run_name":      args.run_name,
        "mode":          args.mode,
        "k":             args.k,
        "seed":          args.seed,
        "per_fold_val":  all_val_results,
        "per_fold_test": all_test_results,
        "aggregate":     {**val_agg, **test_agg},
    }
    with open(ckpt_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    agg = summary["aggregate"]
    print(f"\n{'='*70}")
    print(f"SUMMARY  ({args.mode.upper()}, k={args.k}, seed={args.seed})")
    print(f"{'='*70}")
    hdr = (f"  {'Fold':>4}  {'Label':20}  {'Val AUC-PR':>10}  {'Val F1*':>8}  "
           f"{'Test AUC-PR':>11}  {'Test F1*':>8}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)
    for v, t in zip(all_val_results, all_test_results):
        lbl = v.get("val_repo", "")[:20]
        print(f"  {v['fold']:4d}  {lbl:20}  {v['auc_pr']:10.4f}  {v['f1_best']:8.4f}  "
              f"{t['auc_pr']:11.4f}  {t['f1_best']:8.4f}")
    print(sep)
    print(f"  {'mean':4}  {'':20}  "
          f"{agg.get('val_auc_pr_mean',0):10.4f}  {agg.get('val_f1_best_mean',0):8.4f}  "
          f"{agg.get('test_auc_pr_mean',0):11.4f}  {agg.get('test_f1_best_mean',0):8.4f}")
    print(f"  {'std':4}  {'':20}  "
          f"{agg.get('val_auc_pr_std',0):10.4f}  {agg.get('val_f1_best_std',0):8.4f}  "
          f"{agg.get('test_auc_pr_std',0):11.4f}  {agg.get('test_f1_best_std',0):8.4f}")
    print(f"\nFull results -> {ckpt_dir / 'kfold_summary.json'}")


if __name__ == "__main__":
    main()
