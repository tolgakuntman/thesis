"""
scripts/shap_analysis.py

SHAP attribution analysis for the HeteroSAGE VCC detection model.

Computes two complementary attribution signals on the test split:

  (A) Feature-group Shapley values — exact over K=7 groups, 128 coalitions
      Groups: fn_bert, fn_metrics, file_feats, hunk_bert, hunk_metrics,
              developer, sdlc
      For each coalition S, zero out features NOT in S and record σ(logit).
      Then compute exact Shapley values using the multinomial weight formula.
      No external SHAP library needed — implemented in plain numpy.

  (B) Commit-node gradient × input attribution for the 14 named commit dims
      Uses saliency (∂σ(logit)/∂commit.x · commit.x) as a first-order
      Taylor attribution with a zero baseline.

Outputs (written to --output_dir):
  shap_values.csv        — per-graph Shapley values (N × K columns)
  shap_summary.csv       — group-level summary sorted by mean |φ|
  commit_gradients.csv   — per-graph commit-feature attributions (N × 14)
  commit_feat_summary.csv — dim-level summary sorted by mean |grad×x|
  shap_barplot.png       — horizontal bar chart: mean |φ| per group
  shap_violin.png        — violin plot of φ per group, split by label
  commit_feat_barplot.png — horizontal bar chart: 14 commit features

Usage:
  # Sanity check on 50 samples (CPU, seconds):
  python scripts/shap_analysis.py \\
      --checkpoint checkpoints/phase6/final_full_repo_s42/seed_42/best.pt \\
      --n_samples 50 --output_dir outputs/shap_test

  # Full test set:
  python scripts/shap_analysis.py \\
      --checkpoint checkpoints/phase6/final_full_repo_s42/seed_42/best.pt \\
      --output_dir outputs/shap_analysis
"""

import argparse
import json
import sys
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.graph_dataset import VulnCommitDataset, make_loader
from src.model import HeteroSAGE

# ── Feature group definitions ─────────────────────────────────────────────────

# Order here defines the bit-index in coalition bitmasks.
GROUPS = [
    "fn_bert",       # bit 0 — GraphCodeBERT in function.x[8:776]
    "fn_metrics",    # bit 1 — code metrics in function.x[0:8]
    "file_feats",    # bit 2 — all of file.x
    "hunk_bert",     # bit 3 — GraphCodeBERT in hunk.x[2:770]
    "hunk_metrics",  # bit 4 — hunk numeric in hunk.x[0:2]
    "developer",     # bit 5 — developer.x (topology kept)
    "sdlc",          # bit 6 — issue.x / pull_request.x / release_tag.x
]
K = len(GROUPS)  # 7 → 128 coalitions

GROUP_LABELS = {
    "fn_bert":     "Function BERT",
    "fn_metrics":  "Function metrics",
    "file_feats":  "File features",
    "hunk_bert":   "Hunk BERT",
    "hunk_metrics":"Hunk metrics",
    "developer":   "Developer features",
    "sdlc":        "SDLC features",
}

# commit.x layout (14 dims) from build_graphs_v2.py:
#   [0]  in_main_branch      — binary
#   [1]  merge               — binary
#   [2]  dmm_unit_size       — log1p
#   [3]  dmm_unit_complexity — log1p
#   [4]  dmm_unit_interfacing— log1p
#   [5]  tz_author           — normalized timezone
#   [6]  tz_commit           — normalized timezone
#   [7]  hour_sin            — from author_date
#   [8]  hour_cos
#   [9]  dow_sin
#   [10] dow_cos
#   [11] has_sdlc_data       — binary
#   [12] repo_commits_last_90d — normalized
#   [13] repo_active_authors_90d — normalized
COMMIT_FEAT_NAMES = [
    "in_main_branch",
    "merge",
    "dmm_size",
    "dmm_complexity",
    "dmm_interfacing",
    "tz_author",
    "tz_commit",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "has_sdlc_data",
    "repo_commits_90d",
    "repo_active_authors_90d",
]


# ── Masking ───────────────────────────────────────────────────────────────────

def apply_group_mask(batch, included_groups: frozenset):
    """
    Return a cloned batch with features zeroed for groups NOT in included_groups.

    Masking convention (matches ablation semantics from train.py):
      fn_bert    : function.x[:, 8:776]  — GraphCodeBERT embedding
      fn_metrics : function.x[:, 0:8]   — numeric code metrics
      file_feats : file.x[:, :]         — all file features
      hunk_bert  : hunk.x[:, 2:770]     — GraphCodeBERT embedding
      hunk_metrics: hunk.x[:, 0:2]      — numeric hunk metrics
      developer  : developer.x[:, :]    — features only; topology preserved
      sdlc       : issue.x / pull_request.x / release_tag.x (all dims)

    Edge attributes are left intact so topology signals remain constant
    across all coalitions (this matches ablate_*_feats, not ablate_* ablations).
    """
    b = batch.clone()

    fn_x = b["function"].x
    if fn_x is not None and fn_x.numel() > 0:
        if "fn_bert" not in included_groups and fn_x.size(1) > 8:
            fn_x[:, 8:min(776, fn_x.size(1))] = 0.0
        if "fn_metrics" not in included_groups:
            fn_x[:, 0:min(8, fn_x.size(1))] = 0.0

    if "file_feats" not in included_groups:
        file_x = b["file"].x
        if file_x is not None and file_x.numel() > 0:
            file_x[:] = 0.0

    hunk_x = b["hunk"].x
    if hunk_x is not None and hunk_x.numel() > 0:
        if "hunk_bert" not in included_groups and hunk_x.size(1) > 2:
            hunk_x[:, 2:min(770, hunk_x.size(1))] = 0.0
        if "hunk_metrics" not in included_groups:
            hunk_x[:, 0:min(2, hunk_x.size(1))] = 0.0

    if "developer" not in included_groups:
        dev_x = b["developer"].x
        if dev_x is not None and dev_x.numel() > 0:
            dev_x[:] = 0.0

    if "sdlc" not in included_groups:
        for nt in ("issue", "pull_request", "release_tag"):
            x = b[nt].x
            if x is not None and x.numel() > 0:
                x[:] = 0.0

    return b


# ── Forward pass helpers ──────────────────────────────────────────────────────

def _edge_attr_dict(batch):
    """Extract edge_attr_dict from a PyG batch (matches train.py pattern)."""
    return {
        et: batch[et].edge_attr
        for et in batch.edge_types
        if hasattr(batch[et], "edge_attr") and batch[et].edge_attr is not None
    }


@torch.no_grad()
def _infer(model, batch, device):
    """Move batch to device, run forward pass, return σ(logit) as numpy [B]."""
    batch = batch.to(device)
    logits = model(batch.x_dict, batch.edge_index_dict, _edge_attr_dict(batch))
    return torch.sigmoid(logits).cpu().float().numpy()


# ── (A) Shapley values ────────────────────────────────────────────────────────

def compute_shap_values(model, loader, device):
    """
    Compute exact Shapley values for all K feature groups.

    Algorithm:
      1. For each of 2^K coalition bitmasks, apply the corresponding group
         mask to every batch and record σ(logit) per sample.
      2. From the full coalition×sample probability matrix, compute exact
         Shapley values using the multinomial weight formula.

    Returns:
        probs_matrix : np.ndarray[2^K, N]  — σ(logit) per coalition × sample
        phi          : np.ndarray[N, K]    — exact Shapley values
        hashes       : list[str]           — commit hashes in loader order
        labels       : np.ndarray[N]       — binary labels
    """
    n_coalitions = 1 << K  # 128

    # Collect metadata in a single pass (before the 128-coalition loop)
    hashes, labels = [], []
    for batch in loader:
        h = batch["commit"].hash
        hashes.extend(h if isinstance(h, (list, tuple)) else [h])
        labels.extend(batch.y.squeeze(-1).numpy().tolist())
    N = len(labels)
    labels_arr = np.array(labels, dtype=np.int32)

    probs_matrix = np.zeros((n_coalitions, N), dtype=np.float32)

    # Precompute frozensets for each bitmask
    mask_to_groups = {
        m: frozenset(GROUPS[i] for i in range(K) if (m >> i) & 1)
        for m in range(n_coalitions)
    }

    for mask in tqdm(range(n_coalitions), desc="Coalitions", ncols=80):
        included = mask_to_groups[mask]
        sample_idx = 0
        for batch in loader:
            masked = apply_group_mask(batch, included)
            probs = _infer(model, masked, device)
            bs = len(probs)
            probs_matrix[mask, sample_idx:sample_idx + bs] = probs
            sample_idx += bs

    phi = _shapley_from_matrix(probs_matrix)
    return probs_matrix, phi, hashes, labels_arr


def _shapley_from_matrix(probs_matrix):
    """
    Compute exact Shapley values from the full coalition prob matrix (K=7 groups).

    phi_i(x) = sum_{S subset N\{i}} [|S|!(K-|S|-1)!/K!] * [v(S+{i}) - v(S)]

    Args:
        probs_matrix : [2^K, N]
    Returns:
        phi          : [N, K]
    """
    return _shapley_from_matrix_k(probs_matrix, K)


def _shapley_from_matrix_k(probs_matrix, K_local):
    """
    General exact Shapley computation for any number of players K_local.

    Args:
        probs_matrix : [2^K_local, N]
        K_local      : number of players
    Returns:
        phi          : [N, K_local]
    """
    N = probs_matrix.shape[1]
    phi = np.zeros((N, K_local), dtype=np.float64)

    weight_for_size = [
        factorial(s) * factorial(K_local - s - 1) / factorial(K_local)
        for s in range(K_local)
    ]

    for i in range(K_local):
        i_bit = 1 << i
        for S_mask in range(1 << K_local):
            if (S_mask >> i) & 1:
                continue
            S_size = bin(S_mask).count("1")
            w = weight_for_size[S_size]
            S_with_i = S_mask | i_bit
            phi[:, i] += w * (probs_matrix[S_with_i] - probs_matrix[S_mask])

    return phi.astype(np.float32)


def verify_efficiency_axiom(phi, probs_matrix):
    """
    Efficiency axiom: Σ_i φ_i = v(full) − v(empty).
    Returns (max_abs_error, mean_abs_error).
    """
    full_mask  = (1 << K) - 1
    empty_mask = 0
    expected = probs_matrix[full_mask] - probs_matrix[empty_mask]
    actual   = phi.sum(axis=1)
    err      = np.abs(actual - expected)
    return float(err.max()), float(err.mean())


# ── (D) Per-feature Shapley ───────────────────────────────────────────────────
#
# For each node type's scalar features, compute exact Shapley values treating
# each individual dimension as a player.  This decomposes the total group
# contribution (already measured in section A) into individual feature
# contributions.
#
# Methods:
#   exact  — enumerate all 2^M coalitions.  Tractable for M <= 12.
#   kernel — KernelSHAP (weighted random sampling + WLS).  Used for commit
#             (14 dims -> 16384 coalitions, ~8h CPU for exact).
#
# In all cases the masking zeroes only dims within the target node type;
# all other features remain at their full perrepo-normalised values, so
# phi_i measures the marginal contribution of that single dim given the
# rest of the model input is intact.

# Feature group specs for per-feature Shapley
PERFEATURE_GROUPS = [
    {
        "key":       "fn_metrics",
        "node_type": "function",
        "dims":      list(range(8)),
        "names":     [
            "fn: loc", "fn: complexity", "fn: token_count", "fn: length",
            "fn: top_nesting", "fn: loc_before (z)", "fn: cx_before (z)", "fn: tok_before (z)",
        ],
        "method": "exact",   # 2^8 = 256 coalitions
    },
    {
        "key":       "file_feats",
        "node_type": "file",
        "dims":      [0, 1, 2],
        "names":     ["file: lines_added", "file: lines_deleted", "file: complexity"],
        "method": "exact",   # 2^3 = 8
    },
    {
        "key":       "hunk_metrics",
        "node_type": "hunk",
        "dims":      [0, 1],
        "names":     ["hunk: complexity", "hunk: token_count"],
        "method": "exact",   # 2^2 = 4
    },
    {
        "key":       "developer",
        "node_type": "developer",
        "dims":      list(range(9)),
        "names":     [
            "dev: total_commits", "dev: active_weeks", "dev: commits_as_committer",
            "dev: total_issues", "dev: total_prs", "dev: time_since_last_commit",
            "dev: experience_pct_repo", "dev: num_repos", "dev: repo_commits_before",
        ],
        "method": "exact",   # 2^9 = 512
    },
    {
        "key":       "issue",
        "node_type": "issue",
        "dims":      list(range(4)),
        "names":     [
            "issue: open_90d", "issue: age_median",
            "issue: closed_90d", "issue: open_velocity_90d",
        ],
        "method": "exact",   # 2^4 = 16
    },
    {
        "key":       "pull_request",
        "node_type": "pull_request",
        "dims":      list(range(4)),
        "names":     [
            "pr: count_90d", "pr: age_median",
            "pr: closed_90d", "pr: open_velocity_90d",
        ],
        "method": "exact",   # 2^4 = 16
    },
    {
        "key":       "release_tag",
        "node_type": "release_tag",
        "dims":      list(range(4)),
        "names":     [
            "tag: tags_365d", "tag: cadence_days",
            "tag: days_since_prev", "tag: days_since_norm",
        ],
        "method": "exact",   # 2^4 = 16
    },
    {
        "key":       "commit",
        "node_type": "commit",
        "dims":      list(range(14)),
        "names":     [
            "commit: in_main_branch", "commit: merge",
            "commit: dmm_size", "commit: dmm_complexity", "commit: dmm_interfacing",
            "commit: tz_author", "commit: tz_commit",
            "commit: hour_sin", "commit: hour_cos",
            "commit: dow_sin", "commit: dow_cos",
            "commit: has_sdlc_data",
            "commit: repo_commits_90d", "commit: repo_active_authors_90d",
        ],
        "method": "kernel",  # 2^14 ~= 16k coalitions; use KernelSHAP
        "n_kernel_override": 300,  # 300 random coalitions sufficient for M=14
    },
    # BERT embedding blocks treated as single players (M=1 → 2 coalitions each).
    # Zeroing the whole slice = same masking as --ablate_code_emb / --ablate_hunk_emb.
    {
        "key":       "fn_bert",
        "node_type": "function",
        "dims":      [(8, 776)],   # one player = contiguous range; 2^1 = 2 coalitions
        "names":     ["fn: bert_emb"],
        "method":    "exact",
    },
    {
        "key":       "hunk_bert",
        "node_type": "hunk",
        "dims":      [(2, 770)],   # one player = contiguous range; 2^1 = 2 coalitions
        "names":     ["hunk: bert_emb"],
        "method":    "exact",
    },
]


def _preload_batches(loader, device):
    """
    Move all batches in loader to device once and return as a list.
    Avoids repeated host-to-device transfers across coalition loops.
    """
    return [batch.to(device) for batch in loader]


@torch.no_grad()
def _infer_batch(model, batch):
    """Run forward pass on an already-device-resident batch."""
    logits = model(batch.x_dict, batch.edge_index_dict, _edge_attr_dict(batch))
    return torch.sigmoid(logits).cpu().float().numpy()


def _run_dim_coalitions_fast(model, batches, node_type, dims, n_coalitions, N, desc):
    """
    Run inference for each of n_coalitions bitmasks using in-place save/restore.

    dims: list of players. Each player is either:
      - int          → single column index
      - (start, end) → contiguous slice treated as one player (e.g. BERT embedding block)

    Returns:
        probs_matrix : np.ndarray[n_coalitions, N]
    """
    probs_matrix = np.zeros((n_coalitions, N), dtype=np.float32)

    for mask_int in tqdm(range(n_coalitions), desc=desc, ncols=80):
        excluded = [dims[i] for i in range(len(dims)) if not ((mask_int >> i) & 1)]

        sample_idx = 0
        for batch in batches:
            x = batch[node_type].x
            has_x = x is not None and x.numel() > 0

            saved = {}
            if has_x and excluded:
                for player in excluded:
                    if isinstance(player, tuple):
                        s, e = player[0], min(player[1], x.size(1))
                        if s < x.size(1):
                            saved[player] = x[:, s:e].clone()
                            x[:, s:e] = 0.0
                    else:
                        if player < x.size(1):
                            saved[player] = x[:, player].clone()
                            x[:, player] = 0.0

            probs = _infer_batch(model, batch)

            if has_x and excluded:
                for player, val in saved.items():
                    if isinstance(player, tuple):
                        s, e = player[0], min(player[1], x.size(1))
                        x[:, s:e] = val
                    else:
                        x[:, player] = val

            bs = len(probs)
            probs_matrix[mask_int, sample_idx:sample_idx + bs] = probs
            sample_idx += bs

    return probs_matrix


def compute_perfeature_shap_exact(model, batches, group_spec, N):
    """
    Exact per-feature Shapley for one group spec (method='exact').
    batches: list of device-resident batch objects.

    Returns:
        phi          : np.ndarray[N, M]
        probs_matrix : np.ndarray[2^M, N]
    """
    node_type = group_spec["node_type"]
    dims      = group_spec["dims"]
    M         = len(dims)
    n_coal    = 1 << M

    desc = f"  [{group_spec['key']}] coalitions ({n_coal})"
    probs_matrix = _run_dim_coalitions_fast(model, batches, node_type, dims, n_coal, N, desc)
    phi = _shapley_from_matrix_k(probs_matrix, M)
    return phi, probs_matrix


def compute_perfeature_shap_kernel(model, batches, group_spec, N, n_kernel_samples):
    """
    KernelSHAP for groups where 2^M is too large (method='kernel').

    Algorithm:
      1. Always include the all-zeros and all-ones coalition (boundary conditions).
      2. Sample n_kernel_samples random binary masks, excluding boundaries.
      3. Compute v(mask) using in-place save/restore (fast).
      4. Assign Shapley kernel weights: w(S) = (M-1) / [C(M,|S|) * |S| * (M-|S|)]
         with large weight for boundaries.
      5. Weighted least-squares to estimate phi; enforce efficiency by scaling.

    Returns:
        phi : np.ndarray[N, M]
    """
    from math import comb as math_comb

    node_type = group_spec["node_type"]
    dims      = group_spec["dims"]
    M         = len(dims)

    # ── Step 1: Build coalition set ────────────────────────────────────────
    full_mask  = (1 << M) - 1
    empty_mask = 0

    rng = np.random.default_rng(seed=42)
    random_masks = set()
    while len(random_masks) < min(n_kernel_samples, (1 << M) - 2):
        m = int(rng.integers(1, full_mask))
        random_masks.add(m)

    all_masks = [empty_mask, full_mask] + list(random_masks)
    n_total   = len(all_masks)

    # ── Step 2: Indicator matrix Z and kernel weights ─────────────────────
    Z = np.zeros((n_total, M), dtype=np.float32)
    w = np.zeros(n_total, dtype=np.float64)

    BIG = 1e6
    for row_i, mask_int in enumerate(all_masks):
        bits = [(mask_int >> b) & 1 for b in range(M)]
        Z[row_i] = bits
        s = int(sum(bits))
        if s == 0 or s == M:
            w[row_i] = BIG
        else:
            w[row_i] = (M - 1) / (math_comb(M, s) * s * (M - s))

    # ── Step 3: Run model (in-place, fast) ────────────────────────────────
    probs_matrix = np.zeros((n_total, N), dtype=np.float32)
    desc = f"  [{group_spec['key']}] KernelSHAP ({n_total} samples)"
    for row_i, mask_int in enumerate(tqdm(all_masks, desc=desc, ncols=80)):
        excluded = [dims[b] for b in range(M) if not ((mask_int >> b) & 1)]
        sample_idx = 0
        for batch in batches:
            x = batch[node_type].x
            has_x = x is not None and x.numel() > 0
            saved = {}
            if has_x and excluded:
                for player in excluded:
                    if isinstance(player, tuple):
                        s, e = player[0], min(player[1], x.size(1))
                        if s < x.size(1):
                            saved[player] = x[:, s:e].clone()
                            x[:, s:e] = 0.0
                    else:
                        if player < x.size(1):
                            saved[player] = x[:, player].clone()
                            x[:, player] = 0.0
            probs = _infer_batch(model, batch)
            if has_x and excluded:
                for player, val in saved.items():
                    if isinstance(player, tuple):
                        s, e = player[0], min(player[1], x.size(1))
                        x[:, s:e] = val
                    else:
                        x[:, player] = val
            bs = len(probs)
            probs_matrix[row_i, sample_idx:sample_idx + bs] = probs
            sample_idx += bs

    # ── Step 4: Weighted least squares per sample ──────────────────────────
    # Solve:  phi* = argmin ||diag(w)^(1/2) (Z phi - v_centered)||^2
    #         s.t.  sum(phi) = v(full) - v(empty)  [efficiency]
    #
    # Standard WOLS: phi = (Z^T W Z)^{-1} Z^T W v
    # Then adjust for efficiency constraint.
    W_diag = np.diag(w)
    ZtW    = Z.T @ W_diag          # [M, n_total]
    ZtWZ   = ZtW @ Z               # [M, M]

    # Regularize to avoid singular matrix
    ZtWZ  += np.eye(M) * 1e-8

    ZtWZ_inv = np.linalg.inv(ZtWZ)  # [M, M]

    v_centered = (probs_matrix - probs_matrix[0])  # subtract v(empty), shape [n_total, N]

    # phi_raw[N, M]
    phi_raw = (ZtWZ_inv @ ZtW @ v_centered).T.astype(np.float32)  # [N, M]

    # Enforce efficiency: phi_adj = phi_raw * scale such that sum(phi_adj) = v(full)-v(empty)
    v_diff    = (probs_matrix[1] - probs_matrix[0]).astype(np.float64)  # [N]
    phi_sum   = phi_raw.sum(axis=1).astype(np.float64)  # [N]

    # Scale each sample's phi so that sum matches v_diff
    # If phi_sum is near 0 but v_diff is also near 0, no adjustment needed
    eps = 1e-12
    scale = np.where(np.abs(phi_sum) > eps, v_diff / phi_sum, 1.0)
    phi   = (phi_raw * scale[:, np.newaxis]).astype(np.float32)

    return phi


def compute_all_perfeature_shap(model, loader, device, n_kernel_samples=1000):
    """
    Run per-feature Shapley for all PERFEATURE_GROUPS.

    Preloads all batches to device once to avoid repeated host-device transfers
    across the many coalition loops.

    Returns:
        results   : list of dicts, one per group:
                      {key, names, phi[N, M], method}
        hashes    : list[str]
        labels    : np.ndarray[N]
        probs_full: np.ndarray[N]  (full-model probs with no masking)
    """
    # ── Preload all batches to device once ─────────────────────────────────
    print(f"  Preloading batches to {device}...")
    batches = _preload_batches(loader, device)

    # Collect metadata
    hashes, labels = [], []
    for batch in batches:
        h = batch["commit"].hash
        hashes.extend(h if isinstance(h, (list, tuple)) else [h])
        labels.extend(batch.y.squeeze(-1).cpu().numpy().tolist())
    N = len(labels)
    labels_arr = np.array(labels, dtype=np.int32)

    # Run full model (no masking) to get baseline probs
    probs_full = np.zeros(N, dtype=np.float32)
    idx = 0
    for batch in batches:
        p = _infer_batch(model, batch)
        probs_full[idx:idx + len(p)] = p
        idx += len(p)

    results = []
    for spec in PERFEATURE_GROUPS:
        print(f"\n  Group: {spec['key']}  ({spec['method']}, M={len(spec['dims'])})")
        if spec["method"] == "exact":
            phi, probs_mat = compute_perfeature_shap_exact(model, batches, spec, N)
            full_m  = (1 << len(spec["dims"])) - 1
            empty_m = 0
            exp = probs_mat[full_m] - probs_mat[empty_m]
            err = np.abs(phi.sum(axis=1) - exp)
            print(f"    Efficiency check: max|err|={err.max():.2e}, mean|err|={err.mean():.2e}")
        else:
            k = spec.get("n_kernel_override", n_kernel_samples)
            phi = compute_perfeature_shap_kernel(
                model, batches, spec, N, k
            )

        results.append({
            "key":    spec["key"],
            "names":  spec["names"],
            "phi":    phi,
            "method": spec["method"],
        })

    return results, hashes, labels_arr, probs_full


def build_perfeature_summary(results, labels, probs_full):
    """
    Build a flat summary DataFrame from per-feature Shapley results.

    Columns: group, feature, method, mean_abs_phi, mean_phi_vcc, mean_phi_normal,
             median_abs_phi, rank
    """
    rows = []
    for res in results:
        phi    = res["phi"]
        names  = res["names"]
        method = res["method"]
        key    = res["key"]
        M      = len(names)
        for i in range(M):
            v = phi[:, i]
            rows.append({
                "group":           key,
                "feature":         names[i],
                "method":          method,
                "mean_abs_phi":    float(np.abs(v).mean()),
                "mean_phi_vcc":    float(v[labels == 1].mean()) if (labels == 1).any() else float("nan"),
                "mean_phi_normal": float(v[labels == 0].mean()) if (labels == 0).any() else float("nan"),
                "median_abs_phi":  float(np.median(np.abs(v))),
            })
    df = pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# ── (B) Commit gradient × input ───────────────────────────────────────────────

def compute_commit_gradients(model, loader, device):
    """
    Gradient × input attribution for each of the 14 commit.x dimensions.

    For each sample: attr_i = (∂σ(logit)/∂commit.x_i) · commit.x_i
    This is the first-order Taylor attribution with a zero baseline.

    Returns:
        grad_x : np.ndarray[N, 14]
        hashes : list[str]
        labels : np.ndarray[N]
        probs  : np.ndarray[N]
    """
    model.eval()
    all_grad_x, all_hashes, all_labels, all_probs = [], [], [], []

    for batch in tqdm(loader, desc="Commit gradients", ncols=80):
        batch = batch.to(device)

        # Detach commit.x from graph and re-enable gradients
        commit_x = batch["commit"].x.detach().requires_grad_(True)

        x_dict = dict(batch.x_dict)  # shallow copy
        x_dict["commit"] = commit_x  # swap in the grad-enabled tensor

        logits = model(x_dict, batch.edge_index_dict, _edge_attr_dict(batch))
        probs  = torch.sigmoid(logits)

        # ∂σ(logit_sum)/∂commit_x  →  shape [B, 14]
        grad = torch.autograd.grad(probs.sum(), commit_x)[0]

        # Saliency: gradient × input (baseline = 0 → baseline term vanishes)
        gx = (grad * commit_x.detach()).cpu().numpy()

        all_grad_x.append(gx)
        h = batch["commit"].hash
        all_hashes.extend(h if isinstance(h, (list, tuple)) else [h])
        all_labels.extend(batch.y.squeeze(-1).cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    return (
        np.vstack(all_grad_x).astype(np.float32),
        all_hashes,
        np.array(all_labels, dtype=np.int32),
        np.array(all_probs,  dtype=np.float32),
    )


# ── (C) Full feature-level gradient × input ──────────────────────────────────

# Named features per node type. Each entry: (node_type, slice_or_indices, name)
# BERT embedding blocks are aggregated to a single score (mean |grad x input|).
NAMED_FEATURES: list[tuple[str, int, str]] = [
    # commit.x (14 dims)
    ("commit",     0,  "commit: in_main_branch"),
    ("commit",     1,  "commit: merge"),
    ("commit",     2,  "commit: dmm_size"),
    ("commit",     3,  "commit: dmm_complexity"),
    ("commit",     4,  "commit: dmm_interfacing"),
    ("commit",     5,  "commit: tz_author"),
    ("commit",     6,  "commit: tz_commit"),
    ("commit",     7,  "commit: hour_sin"),
    ("commit",     8,  "commit: hour_cos"),
    ("commit",     9,  "commit: dow_sin"),
    ("commit",    10,  "commit: dow_cos"),
    ("commit",    11,  "commit: has_sdlc_data"),
    ("commit",    12,  "commit: repo_commits_90d"),
    ("commit",    13,  "commit: repo_active_authors_90d"),
    # function.x numeric dims (0-7); dims 5-7 are leakage-zeroed so expect ~0
    ("function",   0,  "fn: loc"),
    ("function",   1,  "fn: complexity"),
    ("function",   2,  "fn: token_count"),
    ("function",   3,  "fn: length"),
    ("function",   4,  "fn: top_nesting_level"),
    ("function",   5,  "fn: loc_before (zeroed)"),
    ("function",   6,  "fn: complexity_before (zeroed)"),
    ("function",   7,  "fn: tokens_before (zeroed)"),
    # file.x (3 dims)
    ("file",       0,  "file: lines_added"),
    ("file",       1,  "file: lines_deleted"),
    ("file",       2,  "file: complexity"),
    # hunk.x numeric dims (0-1)
    ("hunk",       0,  "hunk: complexity"),
    ("hunk",       1,  "hunk: token_count"),
    # developer.x (9 dims)
    ("developer",  0,  "dev: total_commits"),
    ("developer",  1,  "dev: active_weeks"),
    ("developer",  2,  "dev: commits_as_committer"),
    ("developer",  3,  "dev: total_issues"),
    ("developer",  4,  "dev: total_pull_requests"),
    ("developer",  5,  "dev: time_since_last_commit"),
    ("developer",  6,  "dev: experience_pct_repo"),
    ("developer",  7,  "dev: num_repos"),
    ("developer",  8,  "dev: repo_commits_before"),
    # issue.x (4 dims)
    ("issue",      0,  "issue: issue_open_90d"),
    ("issue",      1,  "issue: issue_age_median"),
    ("issue",      2,  "issue: issues_closed_last_90d"),
    ("issue",      3,  "issue: issue_open_velocity_90d"),
    # pull_request.x (4 dims)
    ("pull_request", 0, "pr: pr_count_90d"),
    ("pull_request", 1, "pr: pr_age_median"),
    ("pull_request", 2, "pr: pr_closed_last_90d"),
    ("pull_request", 3, "pr: pr_open_velocity_90d"),
    # release_tag.x (4 dims)
    ("release_tag",  0, "tag: tags_last_365d"),
    ("release_tag",  1, "tag: avg_release_cadence_days"),
    ("release_tag",  2, "tag: days_since_prev_tag"),
    ("release_tag",  3, "tag: days_since_prev_tag_norm"),
]

# Embedding blocks — aggregated to one score per block (mean |grad x input|)
EMBEDDING_BLOCKS: list[tuple[str, slice, str]] = [
    ("function", slice(8, 776),  "fn: BERT embedding (768d)"),
    ("hunk",     slice(2, 770),  "hunk: BERT embedding (768d)"),
]


def compute_all_feature_gradients(model, loader, device):
    """
    Compute gradient x input attribution for every named feature dimension
    across all node types, plus aggregated scores for BERT embedding blocks.

    Returns:
        named_attr   : np.ndarray[N, len(NAMED_FEATURES)]  per-sample attributions
        embed_attr   : np.ndarray[N, len(EMBEDDING_BLOCKS)] mean|grad x x| per block
        hashes       : list[str]
        labels       : np.ndarray[N]
        probs        : np.ndarray[N]
    """
    model.eval()

    n_named = len(NAMED_FEATURES)
    n_embed = len(EMBEDDING_BLOCKS)

    all_named, all_embed = [], []
    all_hashes, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="All-feature gradients", ncols=80):
        batch = batch.to(device)

        # Enable gradients on all node feature tensors simultaneously
        x_dict_grad = {}
        grad_tensors = {}  # node_type -> grad-enabled tensor
        for nt in batch.node_types:
            x = batch[nt].x
            if x is not None and x.numel() > 0:
                xg = x.detach().requires_grad_(True)
                x_dict_grad[nt] = xg
                grad_tensors[nt] = xg
            else:
                x_dict_grad[nt] = x  # keep as-is (empty)

        logits = model(x_dict_grad, batch.edge_index_dict, _edge_attr_dict(batch))
        probs  = torch.sigmoid(logits)

        # Single backward pass to get all gradients at once
        probs.sum().backward()

        B = logits.size(0)

        # ── Named scalar features ──────────────────────────────────────────
        named_row = np.zeros((B, n_named), dtype=np.float32)
        for col, (nt, dim, _) in enumerate(NAMED_FEATURES):
            xg = grad_tensors.get(nt)
            if xg is None or xg.grad is None or xg.size(0) == 0:
                continue
            g = xg.grad  # [N_nodes, feat_dim]
            x = xg.detach()
            gx = (g * x)[:, dim]  # [N_nodes]

            if nt == "commit":
                # 1 commit node per graph in the batch → directly index by graph
                named_row[:, col] = gx.cpu().numpy()
            else:
                # Multiple nodes per graph — aggregate by graph using batch vector
                batch_vec = batch[nt].batch  # [N_nodes] graph index in [0, B)
                agg = torch.zeros(B, device=device)
                agg.scatter_add_(0, batch_vec, gx)
                counts = torch.zeros(B, device=device)
                counts.scatter_add_(0, batch_vec, torch.ones_like(gx))
                # Mean over nodes; graphs with 0 nodes get 0
                safe_counts = counts.clamp(min=1.0)
                named_row[:, col] = (agg / safe_counts).cpu().numpy()

        all_named.append(named_row)

        # ── Embedding blocks ───────────────────────────────────────────────
        # Use signed SUM over all embedding dims so the block attribution is
        # directly comparable in scale to individual scalar feature attributions
        # (and consistent with SHAP group values: removing 768 dims at once).
        embed_row = np.zeros((B, n_embed), dtype=np.float32)
        for col, (nt, slc, _) in enumerate(EMBEDDING_BLOCKS):
            xg = grad_tensors.get(nt)
            if xg is None or xg.grad is None or xg.size(0) == 0:
                continue
            g  = xg.grad[:, slc]
            x  = xg.detach()[:, slc]
            gx_block = (g * x).sum(dim=1)  # [N_nodes] signed sum over embedding dims

            batch_vec = batch[nt].batch
            agg    = torch.zeros(B, device=device)
            counts = torch.zeros(B, device=device)
            agg.scatter_add_(0, batch_vec, gx_block)
            counts.scatter_add_(0, batch_vec, torch.ones_like(gx_block))
            safe_counts = counts.clamp(min=1.0)
            embed_row[:, col] = (agg / safe_counts).cpu().numpy()

        all_embed.append(embed_row)

        h = batch["commit"].hash
        all_hashes.extend(h if isinstance(h, (list, tuple)) else [h])
        all_labels.extend(batch.y.squeeze(-1).cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        # Zero gradients for next batch
        for xg in grad_tensors.values():
            if xg.grad is not None:
                xg.grad = None

    return (
        np.vstack(all_named).astype(np.float32),
        np.vstack(all_embed).astype(np.float32),
        all_hashes,
        np.array(all_labels, dtype=np.int32),
        np.array(all_probs,  dtype=np.float32),
    )


def plot_full_feature_barplot(
    full_summary: pd.DataFrame,
    output_path: Path,
    top_n: int = 40,
):
    """
    Horizontal bar chart of mean |grad x input| for all named features
    plus embedding block aggregates, top_n shown.
    Color-coded by node type.
    """
    NODE_COLORS = {
        "commit":       "#1f77b4",
        "fn":           "#2ca02c",
        "file":         "#ff7f0e",
        "hunk":         "#9467bd",
        "dev":          "#d62728",
        "issue":        "#8c564b",
        "pr":           "#e377c2",
        "tag":          "#7f7f7f",
    }

    def _node_prefix(feat_name: str) -> str:
        return feat_name.split(":")[0].strip()

    df = full_summary.sort_values("mean_abs_attr", ascending=False).head(top_n)
    df = df.sort_values("mean_abs_attr", ascending=True)  # flip for horizontal bar

    colors = [NODE_COLORS.get(_node_prefix(f), "#aec7e8") for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(9, max(6, top_n * 0.28)))
    bars = ax.barh(df["feature"].values, df["mean_abs_attr"].values, color=colors)

    for bar, val in zip(bars, df["mean_abs_attr"].values):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left", fontsize=7)

    ax.set_xlabel("Mean |gradient x input| attribution", fontsize=11)
    ax.set_title(
        f"Full feature attribution (top {top_n} of {len(full_summary)})\n"
        "HeteroSAGE | test set (OpenSC) | repo split",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=df["mean_abs_attr"].max() * 1.35)

    # Legend for node types present
    seen = {}
    for feat, color in zip(df["feature"].values, colors):
        pfx = _node_prefix(feat)
        if pfx not in seen:
            seen[pfx] = color
    handles = [mpatches.Patch(facecolor=c, label=p) for p, c in seen.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, title="Node type")

    _despine(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _despine(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_perfeature_barplot(summary_df: pd.DataFrame, output_path: Path, top_n: int = 50):
    """
    Horizontal bar chart of mean |phi| per individual feature, sorted descending.
    Color-coded by node type prefix.  Kernel-SHAP features marked with (*).
    """
    NODE_COLORS = {
        "commit":       "#1f77b4",
        "fn":           "#2ca02c",
        "file":         "#ff7f0e",
        "hunk":         "#9467bd",
        "dev":          "#d62728",
        "issue":        "#8c564b",
        "pr":           "#e377c2",
        "tag":          "#7f7f7f",
    }

    def _pfx(feat):
        return feat.split(":")[0].strip()

    df = summary_df.sort_values("mean_abs_phi", ascending=False).head(top_n).copy()
    # Mark kernel-SHAP features
    df["display"] = df.apply(
        lambda r: r["feature"] + " (*)" if r["method"] == "kernel" else r["feature"],
        axis=1,
    )
    df = df.sort_values("mean_abs_phi", ascending=True)

    colors = [NODE_COLORS.get(_pfx(f), "#aec7e8") for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(9, max(6, min(top_n, len(df)) * 0.3)))
    bars = ax.barh(df["display"].values, df["mean_abs_phi"].values, color=colors)

    for bar, val in zip(bars, df["mean_abs_phi"].values):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left", fontsize=7)

    ax.set_xlabel("Mean |SHAP value|  (delta predicted probability)", fontsize=11)
    ax.set_title(
        f"Per-feature SHAP importance (top {min(top_n, len(df))} of {len(summary_df)})\n"
        "HeteroSAGE | test set (OpenSC) | repo split  [(*) = KernelSHAP]",
        fontsize=11,
    )
    ax.set_xlim(left=0, right=df["mean_abs_phi"].max() * 1.35)

    seen = {}
    for feat, color in zip(df["feature"].values, colors):
        pfx = _pfx(feat)
        if pfx not in seen:
            seen[pfx] = color
    handles = [mpatches.Patch(facecolor=c, label=p) for p, c in seen.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9, title="Node type")

    _despine(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_perfeature_violin(summary_df: pd.DataFrame, results: list, labels: np.ndarray,
                           output_path: Path, top_n: int = 20):
    """
    Violin plot of per-feature phi distributions split by label (VCC vs normal),
    for the top_n features by mean |phi|.
    """
    vcc_color  = "#d62728"
    norm_color = "#1f77b4"

    top_feats = summary_df.sort_values("mean_abs_phi", ascending=False).head(top_n)

    # Build lookup: feature name -> phi column
    feat_to_phi = {}
    for res in results:
        for i, name in enumerate(res["names"]):
            feat_to_phi[name] = res["phi"][:, i]

    fig, ax = plt.subplots(figsize=(max(12, top_n * 0.7), 5))
    positions = np.arange(top_n)

    for pos_i, (_, row) in enumerate(top_feats.iterrows()):
        v = feat_to_phi.get(row["feature"])
        if v is None:
            continue
        vcc_vals  = v[labels == 1]
        norm_vals = v[labels == 0]
        for vals, color, offset in [
            (norm_vals, norm_color, -0.2),
            (vcc_vals,  vcc_color,  +0.2),
        ]:
            if len(vals) < 4:
                continue
            parts = ax.violinplot(
                [vals], positions=[pos_i + offset],
                widths=0.35, showmedians=True, showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(positions)
    feat_labels = [
        (row["feature"] + " (*)") if row["method"] == "kernel" else row["feature"]
        for _, row in top_feats.iterrows()
    ]
    ax.set_xticklabels(feat_labels, rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("SHAP value  (delta predicted probability)", fontsize=11)
    ax.set_title(
        f"Per-feature SHAP distribution by label (top {top_n})\n"
        "HeteroSAGE | test set (OpenSC) | repo split",
        fontsize=12,
    )
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=norm_color, alpha=0.6, label="Normal"),
            mpatches.Patch(facecolor=vcc_color,  alpha=0.6, label="VCC"),
        ],
        loc="upper right", fontsize=10,
    )
    _despine(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_shap_barplot(shap_summary: pd.DataFrame, output_path: Path):
    """Horizontal bar chart of mean |φ| per group, sorted descending."""
    df = shap_summary.sort_values("mean_abs_phi", ascending=True)
    labels = [GROUP_LABELS.get(g, g) for g in df["group"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(df)
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, n))
    bars = ax.barh(labels, df["mean_abs_phi"].values, color=colors)

    # Annotate bars with values
    for bar, val in zip(bars, df["mean_abs_phi"].values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Mean |SHAP value|  (delta predicted probability)", fontsize=11)
    ax.set_title(
        "Feature-group SHAP importance\n"
        "HeteroSAGE | test set (OpenSC) | repo split",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=df["mean_abs_phi"].max() * 1.25)
    _despine(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_shap_violin(phi: np.ndarray, labels: np.ndarray, output_path: Path):
    """Violin plot of φ per group, split by label (blue=normal, red=VCC)."""
    vcc_color  = "#d62728"
    norm_color = "#1f77b4"

    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(K)

    for i in range(K):
        vcc_vals  = phi[labels == 1, i]
        norm_vals = phi[labels == 0, i]

        for vals, color, offset in [
            (norm_vals, norm_color, -0.2),
            (vcc_vals,  vcc_color,  +0.2),
        ]:
            if len(vals) < 4:
                continue
            parts = ax.violinplot(
                [vals],
                positions=[i + offset],
                widths=0.35,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            parts["cmedians"].set_color(color)
            parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [GROUP_LABELS.get(g, g) for g in GROUPS],
        rotation=20, ha="right", fontsize=10,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("SHAP value  (delta predicted probability)", fontsize=11)
    ax.set_title(
        "SHAP distribution by feature group and label\n"
        "HeteroSAGE | test set (OpenSC) | repo split",
        fontsize=12,
    )
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=norm_color, alpha=0.6, label="Normal commit"),
            mpatches.Patch(facecolor=vcc_color,  alpha=0.6, label="VCC"),
        ],
        loc="upper right",
        fontsize=10,
    )
    _despine(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_commit_barplot(commit_summary: pd.DataFrame, output_path: Path):
    """Horizontal bar chart of mean |grad×x| per commit feature, sorted descending."""
    df = commit_summary.sort_values("mean_abs_attr", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(df)
    colors = plt.cm.Oranges(np.linspace(0.4, 0.85, n))
    bars = ax.barh(df["feature"].values, df["mean_abs_attr"].values, color=colors)

    for bar, val in zip(bars, df["mean_abs_attr"].values):
        ax.text(bar.get_width() + bar.get_width() * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left", fontsize=8)

    ax.set_xlabel("Mean |gradient x input| attribution", fontsize=11)
    ax.set_title(
        "Commit-node feature attribution  (gradient x input)\n"
        "HeteroSAGE | test set (OpenSC) | repo split",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=df["mean_abs_attr"].max() * 1.3)
    _despine(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> tuple[HeteroSAGE, dict]:
    """
    Load HeteroSAGE from a checkpoint.

    Reads config.json from the checkpoint's parent-or-grandparent directory
    to reproduce the original model configuration (hidden, dropout, exclusions).
    """
    # config.json is written one level above the seed subdirectory
    for config_candidate in [
        ckpt_path.parent.parent / "config.json",
        ckpt_path.parent.parent / "kfold_config.json",
        ckpt_path.parent / "config.json",
        ckpt_path.parent / "kfold_config.json",
    ]:
        if config_candidate.exists():
            config_path = config_candidate
            break
    else:
        raise FileNotFoundError(
            f"config.json not found near {ckpt_path}. "
            "Tried parent and grandparent directories."
        )

    print(f"Config: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    model = HeteroSAGE(
        hidden=config.get("hidden", 128),
        dropout=config.get("dropout", 0.3),
        feat_dropout=0.0,
        exclude_node_types=config.get("exclude_node_types") or [],
        exclude_edge_rels=config.get("exclude_edge_rels") or [],
    )
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    epoch  = ckpt.get("epoch", "?")
    best_v = ckpt.get("best_val_auc_pr", float("nan"))
    print(f"Checkpoint: epoch={epoch}, best_val_AUC-PR={best_v:.4f}")

    return model, config


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SHAP attribution analysis for the HeteroSAGE VCC model"
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to best.pt  "
             "(e.g. checkpoints/phase6/final_full_repo_s42/seed_42/best.pt)",
    )
    p.add_argument(
        "--split_type", default="repo_split",
        choices=["repo_split", "temporal_split"],
        help="Split type (default: repo_split)",
    )
    p.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="Which split to analyse (default: test)",
    )
    p.add_argument(
        "--n_samples", type=int, default=0,
        help="Limit to first N graphs (0 = full split)",
    )
    p.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for inference (default: 128)",
    )
    p.add_argument(
        "--output_dir", default="outputs/shap_analysis",
        help="Directory for CSVs and plots (default: outputs/shap_analysis)",
    )
    p.add_argument(
        "--device", default=None,
        help="cuda / cpu (default: auto-detect)",
    )
    p.add_argument(
        "--skip_gradients", action="store_true",
        help="Skip commit-feature gradient computation (SHAP values only)",
    )
    p.add_argument(
        "--full_features", action="store_true",
        help="Compute gradient x input for ALL named feature dims across all node types "
             "(48 scalar dims + 2 BERT blocks). Produces full_feature_attribution.csv "
             "and full_feature_barplot.png in addition to the standard outputs.",
    )
    p.add_argument(
        "--full_features_only", action="store_true",
        help="Skip SHAP group computation; run only full feature gradient attribution. "
             "Useful for a quick per-feature run without the 128-coalition overhead.",
    )
    p.add_argument(
        "--per_feature_shap", action="store_true",
        help="Compute exact per-feature Shapley values for all scalar feature dims "
             "across all node types.  Exact for M<=9 dims (up to 512 coalitions). "
             "KernelSHAP for commit (14 dims).  Produces perfeature_shap_summary.csv, "
             "perfeature_shap_values.csv, perfeature_shap_barplot.png, and "
             "perfeature_shap_violin.png.",
    )
    p.add_argument(
        "--n_kernel_samples", type=int, default=1000,
        help="Number of random coalitions for KernelSHAP (commit node, 14 dims). "
             "Default: 1000.  Higher = more accurate but slower.",
    )
    p.add_argument(
        "--graphs_dir", default=None,
        help="Override path to pre-built graph .pt files.",
    )
    p.add_argument(
        "--split_index", default=None,
        help="Override path to split_index.csv.",
    )
    p.add_argument(
        "--perrepo_scaler", default=None,
        help="Override path to perrepo_scaler_v2.json.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model, config = load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Dataset
    print(f"\nLoading {args.split_type}/{args.split} dataset...")
    ds_kwargs: dict = dict(
        split_type=args.split_type,
        split=args.split,
        perrepo_norm=config.get("perrepo_norm", False),
    )
    if args.graphs_dir:
        ds_kwargs["graphs_dir"] = args.graphs_dir
    if args.split_index:
        ds_kwargs["split_index_path"] = args.split_index
    if args.perrepo_scaler:
        ds_kwargs["perrepo_scaler_path"] = args.perrepo_scaler
    dataset = VulnCommitDataset(**ds_kwargs)

    if args.n_samples > 0:
        from torch.utils.data import Subset as TorchSubset
        n = min(args.n_samples, len(dataset))
        dataset = TorchSubset(dataset, list(range(n)))
        print(f"Subset: first {n} graphs")

    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        is_train=False,
        pin_memory=(device.type == "cuda"),
    )
    N = len(dataset)
    print(f"Total graphs: {N}")
    print(f"Coalitions:   {1 << K}  (K={K} groups)")

    # ── (A) Shapley values ─────────────────────────────────────────────────
    if args.full_features_only:
        # Skip SHAP; jump straight to full feature gradients
        phi    = None
        labels = None
        hashes = None
        print("Skipping SHAP group computation (--full_features_only).")
    else:
        print(f"\n{'='*60}")
        print(f"(A) Exact Shapley values  [{K} groups x {1<<K} coalitions x {N} graphs]")
        print(f"{'='*60}")

        probs_matrix, phi, hashes, labels = compute_shap_values(model, loader, device)

        # Verify efficiency axiom
        max_err, mean_err = verify_efficiency_axiom(phi, probs_matrix)
        print(f"\nEfficiency axiom check - max |err|: {max_err:.2e}, mean |err|: {mean_err:.2e}")
        if max_err > 1e-3:
            print("WARNING: error > 1e-3 - inspect masking logic before using results.")

        # Full-coalition probabilities (= model predictions)
        full_mask  = (1 << K) - 1
        probs_full = probs_matrix[full_mask]

        # Save per-sample SHAP values
        shap_df = pd.DataFrame(
            {"hash": hashes, "label": labels, "prob": probs_full.round(6)},
        )
        for i, g in enumerate(GROUPS):
            shap_df[f"phi_{g}"] = phi[:, i].round(6)
        shap_df.to_csv(output_dir / "shap_values.csv", index=False)

        # Group-level summary
        rows = []
        for i, g in enumerate(GROUPS):
            v = phi[:, i]
            rows.append({
                "group":           g,
                "mean_abs_phi":    float(np.abs(v).mean()),
                "mean_phi_vcc":    float(v[labels == 1].mean()) if (labels == 1).any() else float("nan"),
                "mean_phi_normal": float(v[labels == 0].mean()) if (labels == 0).any() else float("nan"),
                "median_abs_phi":  float(np.median(np.abs(v))),
            })
        shap_summary = pd.DataFrame(rows).sort_values("mean_abs_phi", ascending=False).reset_index(drop=True)
        shap_summary["rank"] = np.arange(1, len(shap_summary) + 1)
        shap_summary.to_csv(output_dir / "shap_summary.csv", index=False)

        print("\nSHAP Group Summary (sorted by mean |phi|):")
        print(shap_summary.to_string(index=False, float_format="{:.5f}".format))

        # Plots
        print("\nGenerating plots...")
        plot_shap_barplot(shap_summary, output_dir / "shap_barplot.png")
        plot_shap_violin(phi, labels, output_dir / "shap_violin.png")

    # ── (B) Commit gradient × input ───────────────────────────────────────
    if not args.skip_gradients and not args.full_features_only:
        print(f"\n{'='*60}")
        print(f"(B) Commit-node gradient x input  [{N} graphs x 14 dims]")
        print(f"{'='*60}")

        grad_x, g_hashes, g_labels, g_probs = compute_commit_gradients(
            model, loader, device
        )

        # Per-sample output
        commit_grad_df = pd.DataFrame(
            {"hash": g_hashes, "label": g_labels, "prob": g_probs.round(6)},
        )
        for dim_i, feat in enumerate(COMMIT_FEAT_NAMES):
            commit_grad_df[feat] = grad_x[:, dim_i].round(8)
        commit_grad_df.to_csv(output_dir / "commit_gradients.csv", index=False)

        # Feature summary
        commit_rows = []
        for dim_i, feat in enumerate(COMMIT_FEAT_NAMES):
            v = grad_x[:, dim_i]
            commit_rows.append({
                "dim":            dim_i,
                "feature":        feat,
                "mean_abs_attr":  float(np.abs(v).mean()),
                "mean_attr_vcc":  float(v[g_labels == 1].mean()) if (g_labels == 1).any() else float("nan"),
                "mean_attr_normal": float(v[g_labels == 0].mean()) if (g_labels == 0).any() else float("nan"),
            })
        commit_summary = pd.DataFrame(commit_rows).sort_values(
            "mean_abs_attr", ascending=False
        ).reset_index(drop=True)
        commit_summary["rank"] = np.arange(1, len(commit_summary) + 1)
        commit_summary.to_csv(output_dir / "commit_feat_summary.csv", index=False)

        print("\nCommit Feature Attribution Summary (sorted by mean |attr|):")
        print(commit_summary.to_string(index=False, float_format="{:.6f}".format))

        print("\nGenerating commit attribution plot...")
        plot_commit_barplot(commit_summary, output_dir / "commit_feat_barplot.png")

    # ── (C) Full feature-level gradient attribution ────────────────────────
    if args.full_features or args.full_features_only:
        print(f"\n{'='*60}")
        print(f"(C) Full feature gradients  [{N} graphs x {len(NAMED_FEATURES)} dims + {len(EMBEDDING_BLOCKS)} BERT blocks]")
        print(f"{'='*60}")

        named_attr, embed_attr, f_hashes, f_labels, f_probs = compute_all_feature_gradients(
            model, loader, device
        )

        # Build unified summary: named scalar features + embedding blocks
        full_rows = []

        for col, (nt, dim, feat_name) in enumerate(NAMED_FEATURES):
            v = named_attr[:, col]
            full_rows.append({
                "node_type":      nt,
                "feature":        feat_name,
                "mean_abs_attr":  float(np.abs(v).mean()),
                "mean_attr_vcc":  float(v[f_labels == 1].mean()) if (f_labels == 1).any() else float("nan"),
                "mean_attr_normal": float(v[f_labels == 0].mean()) if (f_labels == 0).any() else float("nan"),
            })

        for col, (nt, slc, feat_name) in enumerate(EMBEDDING_BLOCKS):
            v = embed_attr[:, col]
            full_rows.append({
                "node_type":      nt,
                "feature":        feat_name,
                "mean_abs_attr":  float(v.mean()),   # already |attr| per node, averaged
                "mean_attr_vcc":  float(v[f_labels == 1].mean()) if (f_labels == 1).any() else float("nan"),
                "mean_attr_normal": float(v[f_labels == 0].mean()) if (f_labels == 0).any() else float("nan"),
            })

        full_summary = pd.DataFrame(full_rows).sort_values(
            "mean_abs_attr", ascending=False
        ).reset_index(drop=True)
        full_summary["rank"] = np.arange(1, len(full_summary) + 1)
        full_summary.to_csv(output_dir / "full_feature_attribution.csv", index=False)

        print("\nFull Feature Attribution Summary (sorted by mean |attr|):")
        print(full_summary.to_string(index=False, float_format="{:.6f}".format))

        print("\nGenerating full feature attribution plot...")
        plot_full_feature_barplot(
            full_summary,
            output_dir / "full_feature_barplot.png",
            top_n=40,
        )

        # Also save per-sample CSV
        full_per_sample = pd.DataFrame({
            "hash": f_hashes, "label": f_labels, "prob": f_probs.round(6),
        })
        for col, (_, _, feat_name) in enumerate(NAMED_FEATURES):
            safe_col = feat_name.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", "")
            full_per_sample[safe_col] = named_attr[:, col].round(8)
        for col, (_, _, feat_name) in enumerate(EMBEDDING_BLOCKS):
            safe_col = feat_name.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            full_per_sample[safe_col] = embed_attr[:, col].round(8)
        full_per_sample.to_csv(output_dir / "full_feature_gradients.csv", index=False)
        print(f"  Saved: full_feature_attribution.csv, full_feature_gradients.csv")

    # ── (D) Per-feature Shapley ────────────────────────────────────────────
    if args.per_feature_shap:
        print(f"\n{'='*60}")
        total_dims = sum(len(s["dims"]) for s in PERFEATURE_GROUPS)
        total_coal = sum(
            (1 << len(s["dims"])) if s["method"] == "exact"
            else (args.n_kernel_samples + 2)
            for s in PERFEATURE_GROUPS
        )
        print(f"(D) Per-feature Shapley  [{total_dims} dims, ~{total_coal} total coalitions]")
        group_info = ", ".join(
            f"{s['key']}({len(s['dims'])}d/{s['method']})" for s in PERFEATURE_GROUPS
        )
        print(f"    Groups: {group_info}")
        print(f"{'='*60}")

        pf_results, pf_hashes, pf_labels, pf_probs = compute_all_perfeature_shap(
            model, loader, device, n_kernel_samples=args.n_kernel_samples
        )

        # Build summary
        pf_summary = build_perfeature_summary(pf_results, pf_labels, pf_probs)
        pf_summary.to_csv(output_dir / "perfeature_shap_summary.csv", index=False)

        print("\nPer-feature SHAP Summary (top 30 by mean |phi|):")
        print(
            pf_summary.head(30).to_string(index=False, float_format="{:.6f}".format)
        )

        # Per-sample CSV
        pf_per_sample = pd.DataFrame({
            "hash": pf_hashes, "label": pf_labels, "prob": pf_probs.round(6),
        })
        for res in pf_results:
            for i, name in enumerate(res["names"]):
                safe = name.replace(":", "_").replace(" ", "_").replace("(", "").replace(")", "")
                pf_per_sample[safe] = res["phi"][:, i].round(8)
        pf_per_sample.to_csv(output_dir / "perfeature_shap_values.csv", index=False)

        print("\nGenerating per-feature SHAP plots...")
        plot_perfeature_barplot(pf_summary, output_dir / "perfeature_shap_barplot.png", top_n=50)
        plot_perfeature_violin(
            pf_summary, pf_results, pf_labels,
            output_dir / "perfeature_shap_violin.png", top_n=20
        )
        print("  Saved: perfeature_shap_summary.csv, perfeature_shap_values.csv")

    print(f"\nAll outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
