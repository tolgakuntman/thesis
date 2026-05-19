"""
scripts/train.py

Training loop for the commit-level VCC heterogeneous GNN.

Checkpointing (saved to --output_dir / --run_name):
    best.pt         - model with highest val AUC-PR (primary checkpoint criterion)
    latest.pt       - overwritten every epoch (resume after preemption)
    epoch_NNN.pt    - saved every --ckpt_every epochs (milestone snapshots)
    metrics.csv     - per-epoch train/val metrics log
    config.json     - full run configuration
    test_results.json - final test metrics
    seed_results.json - aggregated multi-seed results (when --seeds used)

Usage:
    conda activate thesis
    python scripts/train.py                              # defaults: repo_split, HeteroSAGE
    python scripts/train.py --split_type temporal_split
    python scripts/train.py --model rgcn                # ablation
    python scripts/train.py --exclude_node_types hunk developer  # structural ablation
    python scripts/train.py --seeds 42 123 7            # multi-seed run

Protocol (frozen 2026-04-11):
    epochs=60, patience=10, warmup=3, batch_size=128
    focal_alpha=0.65, focal_gamma=1.5
    checkpoint criterion: val AUC-PR (tie-break: val MCC)
    split: repo_split for structural comparisons
    seeds for final runs: 42, 123, 7

Metrics reported each epoch:
    loss, F1 (threshold=0.5), F1_best (best-sweep threshold), Precision, Recall,
    AUC-PR, AUC-ROC, MCC, threshold_best
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
from src.model import CommitMLP, FocalLoss, HeteroRGCN, HeteroSAGE


# -- argument parsing ---------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train heterogeneous GNN for VCC detection")

    # Data
    p.add_argument("--split_type", default="repo_split",
                   choices=["repo_split", "temporal_split"],
                   help="Which split column to use from split_index.csv")
    p.add_argument("--graphs_dir", default=None,
                   help="Override path to .pt graph files (default: auto-detected). "
                        "Set to $TMPDIR/graphs on HPC for local-scratch speedup.")
    p.add_argument("--split_index", default=None,
                   help="Override path to split_index.csv")

    # Model
    p.add_argument("--model",   default="sage", choices=["sage", "rgcn", "mlp"],
                   help="sage = HeteroSAGE (primary), rgcn = HeteroRGCN (ablation), "
                        "mlp = CommitMLP no-graph baseline")
    p.add_argument("--hidden",        type=int,   default=128)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--feat_dropout",  type=float, default=0.0,
                   help="Input feature dropout applied before node projection (default: 0 = off)")
    p.add_argument("--num_bases",     type=int,   default=4,
                   help="R-GCN basis matrices (only used with --model rgcn)")

    # Structural exclusion (true graph removal, not zeroing)
    p.add_argument("--exclude_node_types", nargs="*", default=[],
                   metavar="NODE_TYPE",
                   help="Node types to structurally remove from the graph. "
                        "Edges touching excluded types are also removed. "
                        "Example: --exclude_node_types hunk developer")
    p.add_argument("--exclude_edge_rels", nargs="*", default=[],
                   metavar="REL_NAME",
                   help="Edge relation names to structurally remove. "
                        "Example: --exclude_edge_rels modifies_func in_commit_fn")

    # Training
    p.add_argument("--epochs",     type=int,   default=60,
                   help="Max training epochs (protocol default: 60)")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early stopping patience (protocol default: 10)")
    p.add_argument("--warmup_epochs", type=int, default=3,
                   help="Linear LR warmup before cosine decay (protocol default: 3)")

    # Focal loss (protocol defaults: alpha=0.65, gamma=1.5)
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--focal_alpha", type=float, default=0.65)

    # Seeds — multi-seed support
    p.add_argument("--seed",  type=int, default=42,
                   help="Single seed (used when --seeds is not provided)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multiple seeds for repeated runs. When provided, --seed is ignored. "
                        "Results are aggregated and saved to seed_results.json. "
                        "Example: --seeds 42 123 7")

    # Checkpointing
    p.add_argument("--run_name",    default=None,
                   help="Experiment name (default: auto-generated from args)")
    p.add_argument("--output_dir",  default="checkpoints",
                   help="Root directory for checkpoints and logs")
    p.add_argument("--ckpt_every",  type=int, default=10,
                   help="Save epoch_NNN.pt every N epochs (0 = disable)")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from (latest.pt)")

    # Feature ablations (zeroing, not structural removal — legacy interface)
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
    p.add_argument("--ablate_hunk_emb", action="store_true",
                   help="Zero out GraphCodeBERT dims in hunk nodes (last 768 of 770)")
    p.add_argument("--ablate_hunk_metrics", action="store_true",
                   help="Zero hunk dims 0-1 (complexity, token_count); SHAP phi~0.001")
    p.add_argument("--ablate_commit_merge", action="store_true",
                   help="Zero commit dim 1 (merge flag); SHAP phi=0.0015, grad=0.000028")
    p.add_argument("--ablate_developer", action="store_true",
                   help="Zero out developer node features and all developer-related edge attributes")
    p.add_argument("--ablate_developer_feats", action="store_true",
                   help="Zero out only developer node features; keep developer edge attrs intact "
                        "(tests developer topology vs feature contribution)")
    p.add_argument("--ablate_commit_stats", action="store_true",
                   help="Zero out DMM change-quality metrics from commit.x dims 2-4 "
                        "(dmm_size, dmm_cmplx, dmm_iface — tests commit-level change representation)")
    p.add_argument("--include_code_before", action="store_true",
                   help="Append code-before GraphCodeBERT embedding (768 dims) to function.x. "
                        "function.x grows from 776 to 1544 dims. "
                        "Tests whether knowing function state BEFORE the change helps VCC detection.")
    p.add_argument("--ablate_code_before_emb", action="store_true",
                   help="Zero out the code-before embedding dims (776+) injected by --include_code_before. "
                        "Only meaningful when --include_code_before is also set.")
    p.add_argument("--perrepo_norm", action="store_true",
                   help="Re-normalize continuous features per repo at load time")
    p.add_argument("--perrepo_scaler", default=None,
                   help="Override path to perrepo scaler JSON (default: auto-detected from graphs_dir)")
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

    return p.parse_args()


# -- run name -----------------------------------------------------------------

def _build_run_name(args) -> str:
    """Auto-generate a descriptive run name from key args."""
    parts = [args.model, args.split_type, f"h{args.hidden}", f"lr{args.lr}"]
    if args.exclude_node_types:
        parts.append("excl_" + "_".join(sorted(args.exclude_node_types)))
    if args.exclude_edge_rels:
        parts.append("exclr_" + "_".join(sorted(args.exclude_edge_rels)))
    if args.ablate_code_emb:     parts.append("noCodeEmb")
    if args.ablate_sdlc:         parts.append("noSDLC")
    if args.ablate_developer:    parts.append("noDev")
    if args.ablate_hunk_emb:     parts.append("noHunkEmb")
    if args.perrepo_norm:        parts.append("perrepo")
    return "_".join(parts)


# -- metrics -----------------------------------------------------------------

def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute full evaluation metrics from raw logits and integer labels.

    Returns dict with: loss (set externally), f1, f1_best, precision, recall,
                       auc_pr, auc_roc, mcc, thresh_best
    """
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # AUC metrics - threshold-free
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


# -- one epoch ---------------------------------------------------------------

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


# -- checkpointing ------------------------------------------------------------

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


# -- LR schedule --------------------------------------------------------------

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -- build model --------------------------------------------------------------

def build_model(args, device):
    if args.model == "sage":
        model = HeteroSAGE(
            hidden=args.hidden,
            dropout=args.dropout,
            feat_dropout=args.feat_dropout,
            exclude_node_types=args.exclude_node_types or [],
            exclude_edge_rels=args.exclude_edge_rels or [],
        )
    elif args.model == "rgcn":
        model = HeteroRGCN(
            hidden=args.hidden,
            dropout=args.dropout,
            num_bases=args.num_bases,
            exclude_node_types=args.exclude_node_types or [],
            exclude_edge_rels=args.exclude_edge_rels or [],
        )
    else:  # mlp
        model = CommitMLP(hidden=args.hidden, dropout=args.dropout)
    return model.to(device)


# -- single seed training run -------------------------------------------------

def train_one_seed(args, seed: int, ckpt_dir: Path, ds_kwargs: dict, device, pin_memory: bool) -> dict:
    """Run a full training + test evaluation for one seed. Returns test metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = VulnCommitDataset(split_type=args.split_type, split="train", **ds_kwargs)
    val_ds   = VulnCommitDataset(split_type=args.split_type, split="val",   **ds_kwargs)

    num_workers = args.num_workers
    train_loader = make_loader(train_ds, args.batch_size, num_workers, is_train=True,
                               pin_memory=pin_memory, repo_balanced=args.repo_balanced)
    val_loader   = make_loader(val_ds,   args.batch_size, num_workers, is_train=False,
                               pin_memory=pin_memory)

    # Build and materialize model
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
    print(f"  Seed {seed} | Model: {args.model.upper()}  params: {n_params:,}")
    if args.exclude_node_types:
        print(f"  Structurally excluded nodes: {args.exclude_node_types}")
    if args.exclude_edge_rels:
        print(f"  Structurally excluded rels:  {args.exclude_edge_rels}")

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # Per-seed checkpoint subdirectory
    seed_dir = ckpt_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    start_epoch     = 0
    best_val_auc_pr = 0.0
    best_val_mcc    = -1.0   # tie-break for identical AUC-PR
    metrics_history = []
    no_improve_count = 0

    # Resume support (only meaningful for single-seed runs)
    if args.resume and len([args.resume]) == 1:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"  Resuming from {resume_path}")
            start_epoch, best_val_auc_pr, metrics_history = load_checkpoint(
                resume_path, model, optimizer, scheduler
            )
            start_epoch += 1
            print(f"  -> resumed at epoch {start_epoch}, best_val_auc_pr={best_val_auc_pr:.4f}")

    csv_path = seed_dir / "metrics.csv"
    csv_fields = [
        "epoch", "split", "loss", "f1", "f1_best", "precision", "recall",
        "auc_pr", "auc_roc", "mcc", "thresh_best", "lr", "elapsed_s",
    ]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    print(f"\n  {'Ep':>4}  {'Split':6}  {'Loss':>8}  {'F1':>6}  {'F1*':>6}  "
          f"{'AUC-PR':>7}  {'AUC-ROC':>8}  {'MCC':>6}")
    print("  " + "-" * 68)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        scheduler.step()
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, is_train=False)

        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

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
        print(f"  {epoch:4d}  {'train':6}  {_fmt(train_m)}")
        print(f"  {'':4}  {'val':6}  {_fmt(val_m)}")

        # Checkpoint criterion: primary = val AUC-PR, tie-break = val MCC
        val_auc_pr = val_m["auc_pr"]
        val_mcc    = val_m["mcc"]
        is_best = (val_auc_pr > best_val_auc_pr or
                   (val_auc_pr == best_val_auc_pr and val_mcc > best_val_mcc))
        if is_best:
            best_val_auc_pr = val_auc_pr
            best_val_mcc    = val_mcc
            no_improve_count = 0
            save_checkpoint(seed_dir / "best.pt", model, optimizer, scheduler,
                            epoch, best_val_auc_pr, metrics_history)
            print(f"  ** New best val AUC-PR: {best_val_auc_pr:.4f}  MCC: {best_val_mcc:.4f}")
        else:
            no_improve_count += 1

        save_checkpoint(seed_dir / "latest.pt", model, optimizer, scheduler,
                        epoch, best_val_auc_pr, metrics_history)

        if args.ckpt_every > 0 and (epoch + 1) % args.ckpt_every == 0:
            save_checkpoint(seed_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                            scheduler, epoch, best_val_auc_pr, metrics_history)

        if no_improve_count >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no val AUC-PR improvement for {args.patience} epochs)")
            break

    csv_file.close()
    print(f"\n  Seed {seed} training done. Best val AUC-PR: {best_val_auc_pr:.4f}")

    # Test evaluation using best model
    print(f"  Loading best model for test evaluation...")
    ckpt = torch.load(seed_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_ds     = VulnCommitDataset(split_type=args.split_type, split="test", **ds_kwargs)
    test_loader = make_loader(test_ds, args.batch_size, num_workers, is_train=False,
                              pin_memory=pin_memory)
    test_m = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)

    print(f"\n  Test results (seed={seed}, {args.split_type}):")
    for k, v in test_m.items():
        print(f"    {k:15s}: {v}")

    test_m["seed"] = seed
    test_m["best_val_auc_pr"] = best_val_auc_pr
    test_m["split_type"] = args.split_type
    with open(seed_dir / "test_results.json", "w") as f:
        json.dump(test_m, f, indent=2)

    return test_m


# -- aggregate multi-seed results ---------------------------------------------

def aggregate_seed_results(results: list[dict]) -> dict:
    """Compute mean ± std across seeds for all numeric metrics."""
    metric_keys = [k for k in results[0] if k not in ("seed", "split_type")]
    agg = {"seeds": [r["seed"] for r in results], "n_seeds": len(results)}
    for k in metric_keys:
        vals = [r[k] for r in results if isinstance(r.get(k), (int, float))]
        if vals:
            agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"]  = round(float(np.std(vals)),  4)
    return agg


# -- main --------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve seed list
    seeds = args.seeds if args.seeds is not None else [args.seed]

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

    # Run name
    if args.run_name is None:
        args.run_name = _build_run_name(args)
    ckpt_dir = ROOT / args.output_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {args.run_name}")
    print(f"Checkpoints -> {ckpt_dir}")
    print(f"Seeds: {seeds}")

    # If resuming, reload saved config
    if args.resume:
        saved_config = ckpt_dir / "config.json"
        if saved_config.exists():
            with open(saved_config) as f:
                saved = json.load(f)
            cli_epochs = args.epochs
            cli_resume = args.resume
            for k, v in saved.items():
                if k not in ("epochs", "resume", "seeds", "seed"):
                    setattr(args, k, v)
            args.epochs = cli_epochs
            args.resume = cli_resume
            print(f"Resume: restored config from {saved_config}")

    # Save run config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({**vars(args), "seeds_resolved": seeds}, f, indent=2)

    # Build dataset kwargs
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
    if args.ablate_fn_code_metrics:
        ds_kwargs["ablate_fn_code_metrics"] = True
    if args.ablate_file_code_metrics:
        ds_kwargs["ablate_file_code_metrics"] = True
    if args.ablate_sdlc:
        ds_kwargs["ablate_sdlc"] = True
        print("ABLATION: SDLC node features zeroed out (issue/PR/tag)")
    if args.ablate_hunk_emb:
        ds_kwargs["ablate_hunk_emb"] = True
        print("ABLATION: GraphCodeBERT dims zeroed out in hunk nodes")
    if args.ablate_hunk_metrics:
        ds_kwargs["ablate_hunk_metrics"] = True
        print("ABLATION: Hunk dims 0-1 (complexity, token_count) zeroed")
    if args.ablate_commit_merge:
        ds_kwargs["ablate_commit_merge"] = True
        print("ABLATION: Commit dim 1 (merge flag) zeroed")
    if args.ablate_developer:
        ds_kwargs["ablate_developer"] = True
        print("ABLATION: Developer node features and dev edge attrs zeroed out")
    if args.ablate_developer_feats:
        ds_kwargs["ablate_developer_feats"] = True
        print("ABLATION: Developer node features zeroed (edge topology kept)")
    if args.ablate_commit_stats:
        ds_kwargs["ablate_commit_stats"] = True
        print("ABLATION: Commit DMM stats zeroed (dims 2-4: dmm_size, dmm_cmplx, dmm_iface)")
    if args.include_code_before:
        ds_kwargs["include_code_before"] = True
        print("CODE-BEFORE: Injecting code-before embedding (768 dims) → function.x: 776→1544 dims")
    if args.ablate_code_before_emb:
        ds_kwargs["ablate_code_before_emb"] = True
        print("ABLATION: Code-before embedding dims (776+) zeroed")
    if args.perrepo_norm:
        ds_kwargs["perrepo_norm"] = True
        if args.perrepo_scaler:
            ds_kwargs["perrepo_scaler_path"] = args.perrepo_scaler
        print("Per-repo normalization: enabled")
    if args.keep_sdlc_developer_only:
        ds_kwargs["keep_sdlc_developer_only"] = True
    if args.keep_sdlc_issue_pr_tag_only:
        ds_kwargs["keep_sdlc_issue_pr_tag_only"] = True
    if args.keep_sdlc_edge_only:
        ds_kwargs["keep_sdlc_edge_only"] = True
    if args.keep_sdlc_node_only:
        ds_kwargs["keep_sdlc_node_only"] = True

    # -- multi-seed training loop ------------------------------------------
    all_results = []
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}  ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*70}")
        result = train_one_seed(args, seed, ckpt_dir, ds_kwargs, device, pin_memory)
        all_results.append(result)

    # -- aggregate and report ----------------------------------------------
    if len(all_results) == 1:
        # Single seed: copy test_results to top-level for convenience
        final = all_results[0]
        with open(ckpt_dir / "test_results.json", "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nTest results saved -> {ckpt_dir / 'test_results.json'}")
    else:
        # Multi-seed: save individual results + aggregate
        agg = aggregate_seed_results(all_results)
        agg["split_type"] = args.split_type
        agg["run_name"]   = args.run_name
        with open(ckpt_dir / "seed_results.json", "w") as f:
            json.dump({"per_seed": all_results, "aggregate": agg}, f, indent=2)

        print(f"\n{'='*70}")
        print(f"MULTI-SEED SUMMARY  ({len(seeds)} seeds: {seeds})")
        print(f"{'='*70}")
        key_metrics = ["f1_best", "auc_pr", "auc_roc", "mcc", "precision", "recall"]
        for k in key_metrics:
            m = agg.get(f"{k}_mean", "N/A")
            s = agg.get(f"{k}_std",  "N/A")
            print(f"  {k:15s}: {m:.4f} ± {s:.4f}")
        print(f"\nFull results saved -> {ckpt_dir / 'seed_results.json'}")


if __name__ == "__main__":
    main()
