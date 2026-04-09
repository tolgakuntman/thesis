"""
scripts/train.py

Training loop for the commit-level VCC heterogeneous GNN.

Checkpointing (saved to --output_dir / --run_name):
    best.pt         — model with highest val AUC-PR (used for test evaluation)
    latest.pt       — overwritten every epoch (resume after preemption)
    epoch_NNN.pt    — saved every --ckpt_every epochs (milestone snapshots)
    metrics.csv     — per-epoch train/val metrics log

Usage:
    conda activate thesis
    python scripts/train.py                              # defaults: repo_split, HeteroSAGE
    python scripts/train.py --split_type temporal_split
    python scripts/train.py --model rgcn                # ablation
    python scripts/train.py --resume checkpoints/run1/latest.pt

Key defaults:
    --hidden 128  --dropout 0.3  --lr 1e-3  --epochs 100
    --batch_size 128  --patience 15  --split_type repo_split

Metrics reported each epoch:
    loss, F1 (threshold=0.5), F1_best (best-sweep threshold), Precision, Recall,
    AUC-PR, AUC-ROC, MCC
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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

from src.graph_dataset import VulnCommitDataset, make_loader
from src.model import FocalLoss, HeteroRGCN, HeteroSAGE


# ── argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train heterogeneous GNN for VCC detection")

    # Data
    p.add_argument("--split_type", default="repo_split",
                   choices=["repo_split", "temporal_split"],
                   help="Which split column to use from split_index.csv")
    p.add_argument("--graphs_dir", default=None,
                   help="Override path to .pt graph files (default: data_new/graph_ready/graphs/). "
                        "Set to $TMPDIR/graphs on HPC for local-scratch speedup.")
    p.add_argument("--split_index", default=None,
                   help="Override path to split_index.csv")

    # Model
    p.add_argument("--model",   default="sage", choices=["sage", "rgcn"],
                   help="sage = HeteroSAGE (primary), rgcn = HeteroRGCN (ablation)")
    p.add_argument("--hidden",  type=int,   default=128)
    p.add_argument("--dropout", type=float, default=0.3)

    # Training
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=15,
                   help="Early stopping patience (epochs without val AUC-PR improvement)")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Linear LR warmup before cosine decay")

    # Focal loss
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.75)

    # Checkpointing
    p.add_argument("--run_name",    default=None,
                   help="Experiment name (default: auto-generated from args)")
    p.add_argument("--output_dir",  default="checkpoints",
                   help="Root directory for checkpoints and logs")
    p.add_argument("--ckpt_every",  type=int, default=10,
                   help="Save epoch_NNN.pt every N epochs (0 = disable)")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from (latest.pt)")
    p.add_argument("--ablate_code_emb", action="store_true",
                   help="Zero out GraphCodeBERT dims in function nodes (ablation)")
    p.add_argument("--ablate_msg_emb", action="store_true",
                   help="Zero out commit message embedding dims 6: (ablation)")
    p.add_argument("--ablate_fn_categorical", action="store_true",
                   help="Zero out fct_* one-hot dims 5-9 in function nodes")
    p.add_argument("--ablate_fn_code_metrics", action="store_true",
                   help="Zero out function code metric dims 0-4 (LOC, complexity, token_count, length, nesting)")
    p.add_argument("--ablate_file_code_metrics", action="store_true",
                   help="Zero out file code metric dims 0-2 (num_lines_of_code, complexity, token_count)")
    p.add_argument("--ablate_sdlc", action="store_true",
                   help="Zero out all SDLC node features (issue, pull_request, release_tag)")
    p.add_argument("--perrepo_norm", action="store_true",
                   help="Re-normalize function size features (dims 0-4) per repo at load time")
    p.add_argument("--repo_balanced", action="store_true",
                   help="Weight sampler by 1/(repo_size * class_size_in_repo) to balance across repos")
    p.add_argument("--keep_sdlc_developer_only", action="store_true",
                   help="Keep only developer-side SDLC context; zero issue/PR/tag SDLC features")
    p.add_argument("--keep_sdlc_issue_pr_tag_only", action="store_true",
                   help="Keep only issue/PR/tag SDLC context; zero developer-side SDLC features")
    p.add_argument("--keep_sdlc_edge_only", action="store_true",
                   help="Keep only SDLC edge features; zero SDLC node features")
    p.add_argument("--keep_sdlc_node_only", action="store_true",
                   help="Keep only SDLC node features; zero SDLC edge features")

    # Hardware
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      default=None,
                   help="cuda / cpu (default: auto-detect)")
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute full evaluation metrics from raw logits and integer labels.

    Returns dict with: loss_*, f1, f1_best, precision, recall,
                       auc_pr, auc_roc, mcc, threshold_best
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # AUC metrics — threshold-free
    auc_pr  = average_precision_score(labels, probs)
    auc_roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0

    # Threshold = 0.5
    preds_05 = (probs >= 0.5).astype(int)
    f1_05  = f1_score(labels, preds_05, zero_division=0)
    prec   = precision_score(labels, preds_05, zero_division=0)
    rec    = recall_score(labels, preds_05, zero_division=0)
    mcc    = matthews_corrcoef(labels, preds_05)

    # Best F1 threshold sweep (0.05 to 0.95)
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        preds_t = (probs >= t).astype(int)
        f1_t = f1_score(labels, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1, best_thresh = f1_t, t

    return {
        "f1":           round(f1_05, 4),
        "f1_best":      round(best_f1, 4),
        "precision":    round(prec, 4),
        "recall":       round(rec, 4),
        "auc_pr":       round(auc_pr, 4),
        "auc_roc":      round(auc_roc, 4),
        "mcc":          round(mcc, 4),
        "thresh_best":  round(best_thresh, 2),
    }


# ── one epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train(is_train)
    total_loss = 0.0
    all_logits, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x_dict, batch.edge_index_dict, getattr(batch, "edge_attr_dict", None))
            labels = batch.y.squeeze(-1)  # [B]

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
    avg_loss   = total_loss / max(len(all_labels), 1)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = round(avg_loss, 5)
    return metrics


# ── checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    best_val_auc_pr: float, metrics_history: list):
    torch.save({
        "epoch":             epoch,
        "model_state_dict":  model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_auc_pr":   best_val_auc_pr,
        "metrics_history":   metrics_history,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_val_auc_pr"], ckpt.get("metrics_history", [])


# ── LR schedule ───────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Run name
    if args.run_name is None:
        args.run_name = (
            f"{args.model}_{args.split_type}_h{args.hidden}_"
            f"lr{args.lr}_bs{args.batch_size}"
        )
    ckpt_dir = ROOT / args.output_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # Save run config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── datasets & loaders ────────────────────────────────────────────────
    ds_kwargs = {}
    if args.graphs_dir:
        ds_kwargs["graphs_dir"] = args.graphs_dir
    if args.split_index:
        ds_kwargs["split_index_path"] = args.split_index
    if args.ablate_code_emb:
        ds_kwargs["ablate_code_emb"] = True
        print("ABLATION: GraphCodeBERT dims zeroed out in function nodes")
    if args.ablate_msg_emb:
        ds_kwargs["ablate_msg_emb"] = True
        print("ABLATION: Commit message embedding dims (6:) zeroed out")
    if args.ablate_fn_categorical:
        ds_kwargs["ablate_fn_categorical"] = True
        print("ABLATION: fct_* one-hot dims 5-9 zeroed in function nodes")
    if args.ablate_fn_code_metrics:
        ds_kwargs["ablate_fn_code_metrics"] = True
        print("ABLATION: Function code metric dims 0-4 zeroed (LOC/complexity/token_count/length/nesting)")
    if args.ablate_file_code_metrics:
        ds_kwargs["ablate_file_code_metrics"] = True
        print("ABLATION: File code metric dims 0-2 zeroed (num_lines_of_code/complexity/token_count)")
    if args.ablate_sdlc:
        ds_kwargs["ablate_sdlc"] = True
    if args.perrepo_norm:
        ds_kwargs["perrepo_norm"] = True
        print("ABLATION: continuous numeric graph features re-normalized per training repo")
    if args.keep_sdlc_developer_only:
        ds_kwargs["keep_sdlc_developer_only"] = True
        print("ABLATION: keep only developer-side SDLC context")
    if args.keep_sdlc_issue_pr_tag_only:
        ds_kwargs["keep_sdlc_issue_pr_tag_only"] = True
        print("ABLATION: keep only issue/PR/tag-side SDLC context")
    if args.keep_sdlc_edge_only:
        ds_kwargs["keep_sdlc_edge_only"] = True
        print("ABLATION: keep only SDLC edge features")
    if args.keep_sdlc_node_only:
        ds_kwargs["keep_sdlc_node_only"] = True
        print("ABLATION: keep only SDLC node features")

    train_ds = VulnCommitDataset(split_type=args.split_type, split="train", **ds_kwargs)
    val_ds   = VulnCommitDataset(split_type=args.split_type, split="val",   **ds_kwargs)

    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, is_train=True,
                               repo_balanced=args.repo_balanced)
    val_loader   = make_loader(val_ds,   args.batch_size, args.num_workers, is_train=False)

    # ── model ─────────────────────────────────────────────────────────────
    if args.model == "sage":
        model = HeteroSAGE(hidden=args.hidden, dropout=args.dropout)
    else:
        model = HeteroRGCN(hidden=args.hidden, dropout=args.dropout)
    model = model.to(device)
    # Materialize LazyLinear weights before counting parameters
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
    print(f"Model: {args.model.upper()}  params: {n_params:,}")

    # ── loss, optimizer, scheduler ────────────────────────────────────────
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_auc_pr = 0.0
    metrics_history = []
    no_improve_count = 0

    if args.resume:
        resume_path = Path(args.resume)
        print(f"Resuming from {resume_path}")
        start_epoch, best_val_auc_pr, metrics_history = load_checkpoint(
            resume_path, model, optimizer, scheduler
        )
        start_epoch += 1
        print(f"  → resumed at epoch {start_epoch}, best_val_auc_pr={best_val_auc_pr:.4f}")

    # ── metrics CSV ───────────────────────────────────────────────────────
    csv_path = ckpt_dir / "metrics.csv"
    csv_fields = [
        "epoch", "split", "loss", "f1", "f1_best", "precision", "recall",
        "auc_pr", "auc_roc", "mcc", "thresh_best", "lr", "elapsed_s",
    ]
    csv_exists = csv_path.exists() and args.resume
    csv_file   = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    if not csv_exists:
        csv_writer.writeheader()

    # ── training loop ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"{'Ep':>4}  {'Split':6}  {'Loss':>8}  {'F1':>6}  {'F1*':>6}  "
          f"{'AUC-PR':>7}  {'AUC-ROC':>8}  {'MCC':>6}")
    print("─" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        scheduler.step()

        # Validate
        val_m = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)

        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        # Log
        def _row(split, m):
            return {**m, "epoch": epoch, "split": split, "lr": round(cur_lr, 7),
                    "elapsed_s": round(elapsed, 1)}

        csv_writer.writerow(_row("train", train_m))
        csv_writer.writerow(_row("val",   val_m))
        csv_file.flush()
        metrics_history.append({"epoch": epoch, "train": train_m, "val": val_m})

        def _fmt(m):
            return (f"{m['loss']:8.5f}  {m['f1']:6.4f}  {m['f1_best']:6.4f}  "
                    f"{m['auc_pr']:7.4f}  {m['auc_roc']:8.4f}  {m['mcc']:6.4f}")
        print(f"{epoch:4d}  {'train':6}  {_fmt(train_m)}")
        print(f"{'':4}  {'val':6}  {_fmt(val_m)}")

        # Checkpointing
        is_best = val_m["auc_pr"] > best_val_auc_pr
        if is_best:
            best_val_auc_pr = val_m["auc_pr"]
            no_improve_count = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler,
                            epoch, best_val_auc_pr, metrics_history)
            print(f"  ✓ New best val AUC-PR: {best_val_auc_pr:.4f}")
        else:
            no_improve_count += 1

        # Always overwrite latest.pt (for resume after preemption)
        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler,
                        epoch, best_val_auc_pr, metrics_history)

        # Milestone snapshots every N epochs
        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                            scheduler, epoch, best_val_auc_pr, metrics_history)

        # Early stopping
        if no_improve_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no val AUC-PR improvement for {args.patience} epochs)")
            break

    csv_file.close()
    print("\n" + "─" * 70)
    print(f"Training complete.  Best val AUC-PR: {best_val_auc_pr:.4f}")
    print(f"Best model saved → {ckpt_dir / 'best.pt'}")

    # ── test evaluation (using best model) ────────────────────────────────
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(ckpt_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_ds     = VulnCommitDataset(split_type=args.split_type, split="test", **ds_kwargs)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, is_train=False)
    test_m = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)

    print(f"\nTest results ({args.split_type}):")
    for k, v in test_m.items():
        print(f"  {k:15s}: {v}")

    with open(ckpt_dir / "test_results.json", "w") as f:
        json.dump({"split_type": args.split_type, **test_m}, f, indent=2)
    print(f"Saved → {ckpt_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
