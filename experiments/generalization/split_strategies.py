"""
scripts/split_strategies.py

Split strategy implementations for split-test generalization experiments.

Each strategy function returns a DataFrame with columns:
    hash, label, repo_url, repo_split   [+ optional extra columns]
where repo_split ∈ {train, val, test}.

All strategies are deterministic given a fixed seed.

Usage:
    from scripts.split_strategies import (
        repo_split, temporal_split, repo_temporal_split,
        developer_split, file_split, function_split,
        cwe_split, severity_split, hard_negative_split,
        graph_structure_split, cold_start_split,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Repository constants (mirror of create_split_index_v2.py) ────────────────

_VAL_REPOS = (
    "ImageMagick/ImageMagick",
    "radareorg/radare2",
    "the-tcpdump-group/tcpdump",
    "php/php-src",
    "FreeRDP/FreeRDP",
)

_TEST_REPOS = (
    "FFmpeg/FFmpeg",
    "gpac/gpac",
    "OISF/suricata",
    "openssl/openssl",
    "redis/redis",
    "envoyproxy/envoy",
)


def _classify_repo(url: str) -> str:
    url = str(url or "")
    for r in _TEST_REPOS:
        if r in url:
            return "test"
    for r in _VAL_REPOS:
        if r in url:
            return "val"
    return "train"


def _log_split(name: str, df: pd.DataFrame) -> None:
    for split in ["train", "val", "test"]:
        sub = df[df["repo_split"] == split]
        n   = len(sub)
        pos = int(sub["label"].sum()) if n > 0 else 0
        logger.info(
            "  %-20s | %-5s : %6d total  %5d pos  (%.1f%%)",
            name, split, n, pos, 100 * pos / max(n, 1),
        )


def _base(df: pd.DataFrame) -> pd.DataFrame:
    return df[["hash", "label", "repo_url"]].copy()


# ── 1. Repo split ─────────────────────────────────────────────────────────────

def repo_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Hold-out by repository — same fixed assignment as create_split_index_v2.py.

    val  repos: ImageMagick, radare2, tcpdump, php-src, FreeRDP
    test repos: FFmpeg, gpac, suricata, openssl, redis, envoy
    train: everything else
    """
    out = _base(df)
    out["repo_split"] = out["repo_url"].fillna("").apply(_classify_repo)
    _log_split("repo_split", out)
    return out


# ── 2. Temporal split ─────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Chronological split by author_date.  No future information in train.
    Commits without a parseable date are assigned to train (conservative).
    """
    out = _base(df)
    out["repo_split"] = "train"

    has_date = df["author_date"].notna()
    dated = df[has_date].sort_values("author_date").copy()
    n = len(dated)
    i_val  = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))

    splits = pd.Series("train", index=dated.index, dtype=str)
    splits.iloc[i_val:i_test] = "val"
    splits.iloc[i_test:]      = "test"
    out.loc[dated.index, "repo_split"] = splits

    n_missing = (~has_date).sum()
    if n_missing:
        logger.info("temporal_split: %d commits have no author_date → train", n_missing)

    _log_split("temporal_split", out)
    return out


# ── 3. Repo-temporal split ────────────────────────────────────────────────────

def repo_temporal_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Test  = held-out repositories (same as repo_split).
    Val   = held-out val repositories.
    Train = remaining repos, restricted to the earliest train_frac fraction
            by commit date (prevents future-leak into the training window).
    Pool commits after the val cutoff date are excluded (too recent).

    Limitation: temporal boundary is set from pool commits only; test-repo
    commit dates are not used to set the cutoff.
    """
    repo_class = df["repo_url"].fillna("").apply(_classify_repo)

    test_df = _base(df[repo_class == "test"])
    val_df  = _base(df[repo_class == "val"])
    test_df["repo_split"] = "test"
    val_df["repo_split"]  = "val"

    pool = df[repo_class == "train"].copy()
    has_date_pool = pool["author_date"].notna()
    dated_pool    = pool[has_date_pool].sort_values("author_date")
    n = len(dated_pool)
    i_val  = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))

    pool_split = pd.Series("train", index=dated_pool.index, dtype=str)
    pool_split.iloc[i_val:i_test] = "val"
    pool_split.iloc[i_test:]      = "exclude"  # too recent — excluded
    dated_pool = dated_pool.copy()
    dated_pool["repo_split"] = pool_split.values

    undated_pool = _base(pool[~has_date_pool])
    undated_pool["repo_split"] = "train"

    kept_pool = dated_pool[dated_pool["repo_split"] != "exclude"]
    n_excl    = (dated_pool["repo_split"] == "exclude").sum()
    logger.info("repo_temporal_split: %d pool commits excluded (post val-boundary)", n_excl)

    out = pd.concat(
        [
            kept_pool[["hash", "label", "repo_url", "repo_split"]],
            undated_pool,
            val_df,
            test_df,
        ],
        ignore_index=True,
    )
    _log_split("repo_temporal_split", out)
    return out


# ── 4. Developer split ────────────────────────────────────────────────────────

def developer_split(
    df: pd.DataFrame,
    commit_author: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Hold out developer identities for testing.

    Developers are randomly partitioned into train / val / test groups.
    Commit assignment (cascading):
      - any dev in test group  → commit goes to test
      - any dev in val group   → commit goes to val
      - otherwise              → train
    Commits with no developer record → train (conservative).
    """
    rng = np.random.default_rng(seed)

    dev_map: dict[str, set[str]] = {}
    for row in commit_author.itertuples(index=False):
        dev_map.setdefault(str(row.commit_hash), set()).add(str(row.dev_id))

    all_devs = sorted({d for devs in dev_map.values() for d in devs})
    rng.shuffle(all_devs)
    n        = len(all_devs)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    test_devs = frozenset(all_devs[:n_test])
    val_devs  = frozenset(all_devs[n_test : n_test + n_val])

    logger.info(
        "developer_split: total_devs=%d  test=%d  val=%d  train=%d",
        n, len(test_devs), len(val_devs), n - len(test_devs) - len(val_devs),
    )

    def _assign(h: str) -> str:
        devs = dev_map.get(str(h), frozenset())
        if devs & test_devs:
            return "test"
        if devs & val_devs:
            return "val"
        return "train"

    out = _base(df)
    out["repo_split"] = out["hash"].astype(str).apply(_assign)

    n_no_dev = out["hash"].astype(str).apply(lambda h: h not in dev_map).sum()
    if n_no_dev:
        logger.info("developer_split: %d commits have no dev record → train", n_no_dev)

    _log_split("developer_split", out)
    return out


# ── 5. File split ─────────────────────────────────────────────────────────────

def file_split(
    df: pd.DataFrame,
    file_info: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Hold out files for testing.  File identity = filename (new_path).

    Files are randomly partitioned; commits are assigned cascading:
      any touched file in test → test; any in val → val; else → train.
    Commits with no file record → train.
    """
    rng = np.random.default_rng(seed)

    file_col = "filename" if "filename" in file_info.columns else "new_path"
    commit_files: dict[str, set[str]] = {}
    for row in file_info.itertuples(index=False):
        h = str(getattr(row, "hash", ""))
        f = str(getattr(row, file_col, ""))
        if h and f:
            commit_files.setdefault(h, set()).add(f)

    all_files = sorted({f for fs in commit_files.values() for f in fs})
    rng.shuffle(all_files)
    n        = len(all_files)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    test_files = frozenset(all_files[:n_test])
    val_files  = frozenset(all_files[n_test : n_test + n_val])

    logger.info("file_split: total_files=%d  test=%d  val=%d", n, len(test_files), len(val_files))

    def _assign(h: str) -> str:
        files = commit_files.get(str(h), frozenset())
        if not files:
            return "train"
        if files & test_files:
            return "test"
        if files & val_files:
            return "val"
        return "train"

    out = _base(df)
    out["repo_split"] = out["hash"].astype(str).apply(_assign)
    _log_split("file_split", out)
    return out


# ── 6. Function split ─────────────────────────────────────────────────────────

def function_split(
    df: pd.DataFrame,
    fn_info: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Hold out functions for testing.  Function identity = (name, filename).

    Functions are randomly partitioned; commits assigned cascading:
      any touched function in test → test; any in val → val; else → train.
    """
    rng = np.random.default_rng(seed)

    commit_fns: dict[str, set[tuple[str, str]]] = {}
    for row in fn_info.itertuples(index=False):
        h   = str(getattr(row, "hash", ""))
        key = (str(getattr(row, "name", "")), str(getattr(row, "filename", "")))
        if h:
            commit_fns.setdefault(h, set()).add(key)

    all_fns = sorted({fn for fns in commit_fns.values() for fn in fns})
    rng.shuffle(all_fns)
    n        = len(all_fns)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    test_fns = frozenset(all_fns[:n_test])
    val_fns  = frozenset(all_fns[n_test : n_test + n_val])

    logger.info("function_split: total_fns=%d  test=%d  val=%d", n, len(test_fns), len(val_fns))

    def _assign(h: str) -> str:
        fns = commit_fns.get(str(h), frozenset())
        if not fns:
            return "train"
        if fns & test_fns:
            return "test"
        if fns & val_fns:
            return "val"
        return "train"

    out = _base(df)
    out["repo_split"] = out["hash"].astype(str).apply(_assign)
    _log_split("function_split", out)
    return out


# ── 7. CWE split ──────────────────────────────────────────────────────────────

def cwe_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.20,
    val_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Hold out CWE vulnerability classes for testing.

    VCC commits: assigned by CWE group (cascading if multiple CWEs).
    Non-VCC commits: always train (no vulnerability class to leak).
    Commits with no CWE (negatives or CWE-noinfo): train.

    Requires df to have column 'cwe_id'.
    """
    if "cwe_id" not in df.columns:
        raise ValueError("cwe_split requires 'cwe_id' column in df")

    rng = np.random.default_rng(seed)

    def _parse(s) -> frozenset[str]:
        if pd.isna(s) or not str(s).strip():
            return frozenset()
        return frozenset(c.strip() for c in str(s).split(",") if c.strip())

    df = df.copy()
    df["_cwes"] = df["cwe_id"].apply(_parse)

    all_cwes = sorted({c for cwes in df["_cwes"] for c in cwes})
    rng.shuffle(all_cwes)
    n        = len(all_cwes)
    n_test   = int(n * test_frac)
    n_val    = int(n * val_frac)
    test_cwes = frozenset(all_cwes[:n_test])
    val_cwes  = frozenset(all_cwes[n_test : n_test + n_val])

    logger.info(
        "cwe_split: total_cwes=%d  test=%d  val=%d\n  test: %s\n  val:  %s",
        n, len(test_cwes), len(val_cwes),
        sorted(test_cwes), sorted(val_cwes),
    )

    def _assign(row) -> str:
        cwes = row["_cwes"]
        if not cwes or int(row["label"]) == 0:
            return "train"
        if cwes & test_cwes:
            return "test"
        if cwes & val_cwes:
            return "val"
        return "train"

    out = _base(df)
    out["repo_split"] = df.apply(_assign, axis=1)
    _log_split("cwe_split", out)
    return out


# ── 8. Severity split ─────────────────────────────────────────────────────────

# Static CWE → severity mapping derived from common CVSS v3 base scores.
# Unknown CWEs default to MEDIUM (conservative).
_CWE_SEVERITY: dict[str, str] = {
    # Memory safety — HIGH / CRITICAL
    "CWE-119": "HIGH",   "CWE-120": "HIGH",   "CWE-121": "CRITICAL",
    "CWE-122": "HIGH",   "CWE-125": "HIGH",   "CWE-416": "HIGH",
    "CWE-415": "HIGH",   "CWE-190": "HIGH",   "CWE-191": "HIGH",
    "CWE-787": "CRITICAL", "CWE-824": "HIGH", "CWE-843": "HIGH",
    "CWE-476": "MEDIUM",
    # Injection — HIGH / CRITICAL
    "CWE-78":  "CRITICAL", "CWE-89": "HIGH",  "CWE-94": "CRITICAL",
    "CWE-77":  "HIGH",     "CWE-134": "HIGH", "CWE-74": "HIGH",
    # Auth & access — MEDIUM / HIGH
    "CWE-22":  "HIGH",   "CWE-287": "HIGH",   "CWE-306": "HIGH",
    "CWE-284": "HIGH",   "CWE-285": "HIGH",
    "CWE-23":  "MEDIUM", "CWE-307": "MEDIUM", "CWE-862": "MEDIUM",
    "CWE-863": "MEDIUM",
    # Cryptography — MEDIUM
    "CWE-326": "MEDIUM", "CWE-327": "MEDIUM", "CWE-330": "MEDIUM",
    "CWE-338": "MEDIUM", "CWE-347": "HIGH",
    # Info disclosure — LOW / MEDIUM
    "CWE-200": "MEDIUM", "CWE-201": "LOW",    "CWE-203": "LOW",
    "CWE-209": "LOW",
    # Logic / DoS — LOW / MEDIUM
    "CWE-369": "MEDIUM", "CWE-400": "MEDIUM", "CWE-404": "LOW",
    "CWE-770": "MEDIUM", "CWE-835": "MEDIUM", "CWE-362": "MEDIUM",
    "CWE-20":  "MEDIUM",
    # NVD catch-alls
    "CWE-NVD-CWE-noinfo": "MEDIUM",
    "CWE-NVD-CWE-Other":  "MEDIUM",
}

_SEV_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def severity_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_severities: frozenset[str] | None = None,
    val_severities:   frozenset[str] | None = None,
    test_severities:  frozenset[str] | None = None,
) -> pd.DataFrame:
    """
    Generalization test: train on LOW/MEDIUM severity VCCs, test on HIGH/CRITICAL.

    Non-VCC commits (label=0) always go to train (no severity leakage risk).
    VCCs with unknown / missing CWE → train (conservative).

    Requires df to have column 'cwe_id'.
    """
    if "cwe_id" not in df.columns:
        raise ValueError("severity_split requires 'cwe_id' column in df")

    if train_severities is None: train_severities = frozenset(["LOW", "MEDIUM"])
    if val_severities   is None: val_severities   = frozenset(["MEDIUM"])
    if test_severities  is None: test_severities  = frozenset(["HIGH", "CRITICAL"])

    def _max_sev(cwe_str) -> str | None:
        if pd.isna(cwe_str) or not str(cwe_str).strip():
            return None
        cwes = [c.strip() for c in str(cwe_str).split(",") if c.strip()]
        sevs = [_CWE_SEVERITY.get(c, "MEDIUM") for c in cwes]
        return max(sevs, key=lambda s: _SEV_RANK.get(s, 1)) if sevs else None

    df = df.copy()
    df["_sev"] = df["cwe_id"].apply(_max_sev)

    sev_dist = df[df["label"] == 1]["_sev"].value_counts()
    logger.info("severity_split: VCC severity distribution:\n%s", sev_dist.to_string())

    def _assign(row) -> str:
        if int(row["label"]) == 0 or row["_sev"] is None:
            return "train"
        sev = row["_sev"]
        if sev in test_severities:
            return "test"
        if sev in val_severities:
            return "val"
        return "train"

    out = _base(df)
    out["repo_split"] = df.apply(_assign, axis=1)
    _log_split("severity_split", out)
    return out


# ── 9. Hard-negative split ────────────────────────────────────────────────────

_HN_DEFAULT_WEIGHTS: dict[str, float] = {
    "num_lines_changed":    0.30,
    "num_files_changed":    0.25,
    "dmm_unit_size":        0.15,
    "dmm_unit_complexity":  0.15,
    "dmm_unit_interfacing": 0.15,
}


def hard_negative_split(
    df: pd.DataFrame,
    seed: int = 42,
    score_weights: dict[str, float] | None = None,
    hard_frac: float = 0.20,
) -> pd.DataFrame:
    """
    Train on easy negatives + all VCCs.  Test on hard negatives + VCCs from test repos.

    Hard-negative score = weighted, min-max-normalized sum of commit complexity features.
    The top hard_frac fraction of negatives by score is designated as hard negatives.

    VCC assignment follows repo_split (test/val repos → test/val; others → train).
    Output includes column 'hn_score' (0.0 for VCCs, normalized score for negatives).

    Requires df to have the score feature columns (num_lines_changed, etc.).
    """
    if score_weights is None:
        score_weights = _HN_DEFAULT_WEIGHTS

    df = df.copy()
    score_cols = [c for c in score_weights if c in df.columns]

    if not score_cols:
        logger.warning(
            "hard_negative_split: none of the score columns found in df — "
            "available columns: %s", list(df.columns)
        )
        df["hn_score"] = 0.0
        hard_neg_idx: set[int] = set()
    else:
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        norm_cols = []
        for col in score_cols:
            col_max = df[col].max()
            nc = f"_n_{col}"
            df[nc] = df[col] / col_max if col_max > 0 else 0.0
            norm_cols.append((nc, score_weights[col]))

        total_w    = sum(w for _, w in norm_cols)
        df["hn_score"] = sum(df[nc] * w for nc, w in norm_cols) / total_w

        neg_df        = df[df["label"] == 0].sort_values("hn_score", ascending=False)
        n_hard        = int(len(neg_df) * hard_frac)
        hard_neg_idx  = set(neg_df.head(n_hard).index)

        logger.info(
            "hard_negative_split: %d hard negatives (top %.0f%% by score) → test",
            n_hard, 100 * hard_frac,
        )

    repo_class = df["repo_url"].fillna("").apply(_classify_repo)

    def _assign(row) -> str:
        if int(row["label"]) == 1:      # VCC
            return repo_class.loc[row.name]
        return "test" if row.name in hard_neg_idx else "train"

    df["repo_split"] = df.apply(_assign, axis=1)
    out = df[["hash", "label", "repo_url", "repo_split", "hn_score"]].copy()
    _log_split("hard_negative_split", out)
    return out


# ── 10. Graph-structure split ─────────────────────────────────────────────────

def graph_structure_split(
    df: pd.DataFrame,
    seed: int = 42,
    small_pct: float = 0.50,
    medium_pct: float = 0.25,
) -> pd.DataFrame:
    """
    Train on small/simple graphs, val on medium, test on large/complex graphs.

    Complexity proxy: num_lines_changed + num_files_changed (available from commit_info).
    Percentile thresholds: [0, small_pct) → train, [small_pct, small_pct+medium_pct) → val,
    [small_pct+medium_pct, 1] → test.

    Saves 'graph_score' column (composite size proxy, unnormalized).
    """
    df = df.copy()
    size_cols = [c for c in ["num_lines_changed", "num_files_changed"] if c in df.columns]
    if size_cols:
        for col in size_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df["graph_score"] = df[size_cols].sum(axis=1)
    else:
        logger.warning(
            "graph_structure_split: size columns not found — using row index as proxy"
        )
        df["graph_score"] = np.arange(len(df), dtype=float)

    df["_rank_pct"] = df["graph_score"].rank(method="first", pct=True)
    cut_val  = small_pct
    cut_test = small_pct + medium_pct

    def _assign(r: float) -> str:
        if r <= cut_val:
            return "train"
        if r <= cut_test:
            return "val"
        return "test"

    df["repo_split"] = df["_rank_pct"].apply(_assign)
    out = df[["hash", "label", "repo_url", "repo_split", "graph_score"]].copy()
    _log_split("graph_structure_split", out)
    return out


# ── 11. Cold-start split ──────────────────────────────────────────────────────

def cold_start_split(
    df: pd.DataFrame,
    seed: int = 42,
    n_cold_repos: int = 5,
    few_shot_commits: int = 0,
) -> pd.DataFrame:
    """
    Simulate deployment on new repositories with no training history.

    Val  = existing val repos (same as repo_split -- kept for comparability).
    Test = existing test repos + cold-start pool repos.
    Train = remaining pool repos (no overlap with cold/test/val).

    Cold-start repos are chosen from pool repos (not in existing val/test)
    by selecting those with the fewest commits, simulating 'new' projects.

    few_shot_commits=0  -> strict zero-shot cold-start
    few_shot_commits>0  -> few-shot: earliest N commits from cold repos promoted to train

    Limitation: selection by commit count may still include repos with long histories;
    for stricter recency filtering, override via --cold_start_repos in generate_splits.py.
    """
    rng = np.random.default_rng(seed)

    repo_class         = df["repo_url"].fillna("").apply(_classify_repo)
    existing_test_mask = repo_class == "test"
    existing_val_mask  = repo_class == "val"
    pool_mask          = ~existing_test_mask & ~existing_val_mask

    pool_sizes = df[pool_mask].groupby("repo_url").size().sort_values()  # ascending
    pool_repos = pool_sizes.index.tolist()

    if len(pool_repos) < n_cold_repos:
        logger.warning(
            "cold_start_split: only %d pool repos; using all as cold-start", len(pool_repos)
        )
        n_cold_repos = len(pool_repos)

    cold_repos = frozenset(pool_repos[:n_cold_repos])

    logger.info("cold_start_split: cold_repos (n=%d): %s", len(cold_repos), sorted(cold_repos))
    logger.info(
        "cold_start_split: mode=%s",
        "zero-shot" if few_shot_commits == 0 else f"few-shot({few_shot_commits})",
    )

    df = df.copy()

    def _assign(row) -> str:
        rc  = repo_class.loc[row.name]
        url = str(row["repo_url"] or "")
        if rc == "test":
            return "test"
        if rc == "val":
            return "val"
        if url in cold_repos:
            return "test"
        return "train"

    df["repo_split"] = df.apply(_assign, axis=1)

    if few_shot_commits > 0 and "author_date" in df.columns:
        cold_mask    = df["repo_url"].isin(cold_repos)
        cold_commits = df[cold_mask].sort_values("author_date")
        few_idx      = set(cold_commits.head(few_shot_commits).index)
        df.loc[list(few_idx), "repo_split"] = "train"
        logger.info("cold_start_split: %d few-shot commits promoted to train", len(few_idx))

    out = _base(df)
    out["repo_split"] = df["repo_split"]
    _log_split("cold_start_split", out)
    return out
