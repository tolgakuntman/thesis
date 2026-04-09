#!/usr/bin/env python3
"""
sample_matched_normals.py — v2 Normal Commit Sampling (Complexity-Matched)
with full Option A function/file/hunk extraction.

Resamples normal commits using complexity matching to eliminate the size shortcut
observed in GNN training (Herbold et al. IEEE TSE 2022).

Extraction (fresh PyDriller mode):
  For each sampled normal commit, extracts function/file/hunk/ownership records in the
  Option A format (code_before + code, pre/post snapshots per function).
  Outputs function_info_normal_v2.csv, file_info_normal_v2.csv,
  hunk_info_normal_v2.csv, ownership_normal_v2.csv alongside the sampling outputs.

Sampling protocol:
  - Anchor: all VCC commits (5,911)
  - Keep: all FC commits (2,862) as negatives, not matched
  - Match: up to 5 nearest normals per VCC in 6D feature space
  - Reuse cap: a normal commit can be matched to at most 5 different VCCs
  - Repo cap: if any repo > 40% of matched normals, apply soft cap + re-pass (1x)

Matching features (commit-level aggregates):
  loc_changed, num_files_changed, num_functions_changed,
  max_function_loc, max_function_complexity, sum_function_complexity

Standardization: mean/std computed on candidate pool only.

Usage (fast path — use existing v1 pool):
  python thesis/scripts/sample_matched_normals.py \\
      --icvulpp-root /path/to/ICVul++ \\
      --use-existing-pool \\
      --output-dir data_new/sampling_v2

Usage (fresh PyDriller, single test repo):
  python thesis/scripts/sample_matched_normals.py \\
      --icvulpp-root /path/to/ICVul++ \\
      --repo-cache /tmp/icvulpp_repo_cache \\
      --test-repo https://github.com/tensorflow/tensorflow \\
      --output-dir data_new/sampling_v2_test

Usage (fresh PyDriller, all repos):
  python thesis/scripts/sample_matched_normals.py \\
      --icvulpp-root /path/to/ICVul++ \\
      --repo-cache /tmp/icvulpp_repo_cache \\
      --workers 4 \\
      --output-dir data_new/sampling_v2

References:
  - Herbold et al. (IEEE TSE 2022): commit size as strongest confound
  - McIntosh & Kamei (IEEE TSE 2018): temporal stratification
  - Tantithamthavorn et al. (IEEE TSE 2019): 1:3-1:5 ratio recommendation
  - GraphSPD (S&P 2023): dual-snapshot +4-8 F1 on deletion-heavy CWEs
"""

import argparse
import ast
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lizard
import numpy as np
import pandas as pd
from git import Repo
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_CPP_EXTENSIONS = frozenset(
    '.c .h .ec .ecp .pgc .cpp .cxx .cc .pcc .hpp .hh .hxx'.split()
)

MATCHING_FEATURES = [
    'loc_changed',
    'num_files_changed',
    'num_functions_changed',
    'max_function_loc',
    'max_function_complexity',
    'sum_function_complexity',
]

FUZZY_MATCH_THRESHOLD = 0.70
FUZZY_THRESHOLD_LOW = 0.40
FUZZY_RENAME_THRESHOLD = 0.70
REFACTOR_METHOD_THRESHOLD = 50
OWNERSHIP_WINDOWS = [30, 90, 180]
DEV_ID_SALT = 'ICVul++_2026_ownership'
MIN_REACHABLE_COMMITS = 200
DEFAULT_DISTANCE_LADDER = [10.0, 15.0, 20.0, 50.0]

_SKIP_URL_SET: Set[str] = {
    'https://github.com/torvalds/linux',
    'https://github.com/mjg59/linux',
    'https://github.com/gregkh/linux',
    'https://github.com/ruscur/linux',
}

# v5-compatible column schemas
FUNCTION_COLUMNS = [
    'hash', 'commit_type', 'name', 'filename',
    'num_lines_of_code', 'complexity', 'token_count',
    'parameters', 'signature', 'start_line', 'end_line',
    'length', 'top_nesting_level',
    'code', 'code_before', 'loc_before', 'complexity_before', 'tokens_before',
    'function_change_type', 'extraction_method',
]

FILE_COLUMNS = [
    'hash', 'commit_type', 'filename', 'old_path', 'new_path', 'change_type',
    'diff', 'diff_parsed', 'num_lines_added', 'num_lines_deleted',
    'code_after', 'code_before', 'num_method_changed', 'num_lines_of_code',
    'complexity', 'token_count',
]

HUNK_COLUMNS = [
    'hash', 'commit_type', 'name', 'filename',
    'num_lines_of_code', 'complexity', 'token_count',
    'parameters', 'signature', 'start_line', 'end_line',
    'length', 'top_nesting_level',
    'code', 'before_change', 'function_change_type',
    'extraction_method', 'node_type',
]

OWNERSHIP_COLUMNS = [
    'file_id', 'commit_hash', 'commit_type',
    'file_path',
    'window_days', 'window_start', 'window_end',
    'dev_id', 'dev_email', 'ownership_ratio', 'lines_owned',
    'edits_in_window', 'lines_added_in_window', 'lines_deleted_in_window',
    'total_lines', 'total_devs',
]


def is_skipped(repo_url: str) -> bool:
    if repo_url in _SKIP_URL_SET:
        return True
    name = repo_url.rstrip('/').split('/')[-1].replace('.git', '').lower()
    return name == 'linux'


def is_ccpp(filename: str) -> bool:
    return Path(filename).suffix.lower() in C_CPP_EXTENSIONS


# ---------------------------------------------------------------------------
# Lizard helpers (matching v5 pattern)
# ---------------------------------------------------------------------------

def parse_functions_from_source(file_path: str, source: str) -> List[Dict]:
    """Parse all functions from source via Lizard, return list of dicts."""
    if not source:
        return []
    try:
        analysis = lizard.analyze_file.analyze_source_code(file_path, source)
        lines = source.split('\n')
        results = []
        for fn in analysis.function_list:
            start = max(0, fn.start_line - 1)
            end = min(len(lines), fn.end_line)
            results.append({
                'name': fn.name,
                'start_line': fn.start_line,
                'end_line': fn.end_line,
                'nloc': fn.nloc,
                'complexity': fn.cyclomatic_complexity,
                'token_count': fn.token_count,
                'parameters': fn.parameters,
                'signature': fn.long_name,
                'length': fn.end_line - fn.start_line + 1,
                'top_nesting_level': getattr(fn, 'top_nesting_level', 0),
                'code': '\n'.join(lines[start:end]),
            })
        return results
    except Exception:
        return []


def _unqualified(name: str) -> str:
    return name.split('::')[-1] if '::' in name else name


def find_function_by_name(target_name: str, candidates: List[Dict]) -> Tuple[Optional[Dict], str]:
    """Find matching function: exact → fuzzy_high → fuzzy_low."""
    for fn in candidates:
        if fn['name'] == target_name:
            return fn, 'exact'
    best_ratio, best_fn = 0.0, None
    for fn in candidates:
        ratio = SequenceMatcher(None, _unqualified(target_name), _unqualified(fn['name'])).ratio()
        if ratio > best_ratio:
            best_ratio, best_fn = ratio, fn
    if best_ratio >= FUZZY_MATCH_THRESHOLD and best_fn is not None:
        return best_fn, 'fuzzy_high'
    if best_ratio >= FUZZY_THRESHOLD_LOW and best_fn is not None:
        return best_fn, 'fuzzy_low'
    return None, 'none'


def file_aggregate_metrics(functions: List[Dict]) -> Tuple[int, int, int]:
    """Return (total_nloc, total_complexity, total_tokens) from parsed functions."""
    if not functions:
        return 0, 0, 0
    return (
        sum(f['nloc'] for f in functions),
        sum(f['complexity'] for f in functions),
        sum(f['token_count'] for f in functions),
    )


def hash_dev_id(email: str) -> str:
    norm = email.lower().strip() if email else ''
    return hashlib.sha256(f'{DEV_ID_SALT}:{norm}'.encode()).hexdigest()[:16]


def git_blame_ownership(
    repo_path: str,
    commit_hash: str,
    file_path: str,
) -> Optional[Dict[str, int]]:
    try:
        result = subprocess.run(
            ['git', 'blame', '--line-porcelain', commit_hash, '--', file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None
        ownership: Dict[str, int] = defaultdict(int)
        current_author = None
        for line in result.stdout.split('\n'):
            if line.startswith('author-mail '):
                current_author = line[12:].strip().strip('<>')
            elif line.startswith('\t') and current_author:
                ownership[current_author] += 1
        return dict(ownership)
    except subprocess.TimeoutExpired:
        logging.debug(f'git blame timed out for {file_path} at {commit_hash[:8]}')
        return None
    except Exception as exc:
        logging.debug(f'git blame error for {commit_hash[:8]} {file_path}: {exc}')
        return None


def git_log_activity(
    repo_path: str,
    file_path: str,
    end_date: datetime,
    window_days: int,
) -> Dict[str, Dict[str, int]]:
    try:
        since = (end_date - timedelta(days=window_days)).strftime('%Y-%m-%d')
        until = end_date.strftime('%Y-%m-%d')
        result = subprocess.run(
            [
                'git', 'log',
                f'--since={since}',
                f'--until={until}',
                '--numstat',
                '--format=COMMIT_BY:%ae',
                '--',
                file_path,
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return {}
        activity: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {'commits': 0, 'lines_added': 0, 'lines_deleted': 0}
        )
        current_author = None
        for line in result.stdout.strip().split('\n'):
            if line.startswith('COMMIT_BY:'):
                current_author = line[10:]
                if current_author:
                    activity[current_author]['commits'] += 1
            elif current_author and line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        activity[current_author]['lines_added'] += int(parts[0]) if parts[0] != '-' else 0
                        activity[current_author]['lines_deleted'] += int(parts[1]) if parts[1] != '-' else 0
                    except ValueError:
                        pass
        return {key: dict(value) for key, value in activity.items()}
    except Exception as exc:
        logging.debug(f'git log activity error for {file_path}: {exc}')
        return {}


def extract_ownership_for_commit(
    commit_hash: str,
    repo_path: str,
    file_rows: List['FileRecord'],
    author_date,
) -> List['OwnershipRecord']:
    if not file_rows:
        return []

    commit_date: Optional[datetime]
    if isinstance(author_date, datetime):
        commit_date = author_date
    else:
        commit_date = None
        try:
            commit_date = datetime.fromisoformat(str(author_date))
        except Exception:
            try:
                commit_date = pd.to_datetime(author_date, utc=True).to_pydatetime()
            except Exception:
                logging.debug(f'Cannot parse date {author_date} for ownership')
                return []

    rows: List[OwnershipRecord] = []
    for file_row in file_rows:
        file_path = file_row.new_path or file_row.filename
        if not file_path:
            continue

        ownership = git_blame_ownership(repo_path, commit_hash, file_path)
        if not ownership:
            continue
        total_lines = sum(ownership.values())
        if total_lines == 0:
            continue
        total_devs = len(ownership)

        for window_days in OWNERSHIP_WINDOWS:
            window_start = commit_date - timedelta(days=window_days)
            activity = git_log_activity(repo_path, file_path, commit_date, window_days)
            all_devs = set(ownership) | set(activity)
            for email in all_devs:
                lines_owned = ownership.get(email, 0)
                act = activity.get(email, {'commits': 0, 'lines_added': 0, 'lines_deleted': 0})
                if lines_owned == 0 and act['commits'] == 0:
                    continue
                rows.append(OwnershipRecord(
                    file_id=f'{commit_hash}_{file_path}',
                    commit_hash=commit_hash,
                    file_path=file_path,
                    window_days=window_days,
                    window_start=window_start.isoformat(),
                    window_end=commit_date.isoformat(),
                    dev_id=hash_dev_id(email),
                    dev_email=email,
                    ownership_ratio=round(lines_owned / total_lines, 4),
                    lines_owned=lines_owned,
                    edits_in_window=act['commits'],
                    lines_added_in_window=act['lines_added'],
                    lines_deleted_in_window=act['lines_deleted'],
                    total_lines=total_lines,
                    total_devs=total_devs,
                ))
    return rows


# ---------------------------------------------------------------------------
# Extraction dataclasses (v5-compatible schema)
# ---------------------------------------------------------------------------

@dataclass
class FunctionRecord:
    hash: str
    commit_type: str = 'normal'
    name: str = ''
    filename: str = ''
    num_lines_of_code: int = 0        # post-commit (0 for DELETE)
    complexity: int = 0               # post-commit (0 for DELETE)
    token_count: int = 0              # post-commit (0 for DELETE)
    parameters: str = ''
    signature: str = ''
    start_line: int = 0
    end_line: int = 0
    length: int = 0
    top_nesting_level: int = 0
    code: str = ''                    # post-commit body; '' for DELETE
    code_before: str = ''             # pre-commit body;  '' for ADD
    loc_before: int = 0               # 0 for ADD
    complexity_before: int = 0        # 0 for ADD
    tokens_before: int = 0            # 0 for ADD
    function_change_type: str = ''    # ADD, MODIFY, DELETE
    extraction_method: str = 'direct'


@dataclass
class FileRecord:
    hash: str
    commit_type: str = 'normal'
    filename: str = ''
    old_path: str = ''
    new_path: str = ''
    change_type: str = ''
    diff: str = ''
    diff_parsed: str = ''
    num_lines_added: int = 0
    num_lines_deleted: int = 0
    code_after: str = ''              # post-commit full file source
    code_before: str = ''            # pre-commit full file source
    num_method_changed: int = 0
    num_lines_of_code: int = 0
    complexity: int = 0
    token_count: int = 0


@dataclass
class HunkRecord:
    hash: str
    commit_type: str = 'normal'
    name: str = ''
    filename: str = ''
    num_lines_of_code: int = 0
    complexity: int = 0
    token_count: int = 0
    parameters: str = ''
    signature: str = ''
    start_line: int = 0
    end_line: int = 0
    length: int = 0
    top_nesting_level: int = 0
    code: str = ''
    before_change: bool = False
    function_change_type: str = 'MODIFY'
    extraction_method: str = 'hunk_fallback'
    node_type: str = 'hunk'


@dataclass
class OwnershipRecord:
    file_id: str
    commit_hash: str
    commit_type: str = 'normal'
    file_path: str = ''
    window_days: int = 0
    window_start: str = ''
    window_end: str = ''
    dev_id: str = ''
    dev_email: str = ''
    ownership_ratio: float = 0.0
    lines_owned: int = 0
    edits_in_window: int = 0
    lines_added_in_window: int = 0
    lines_deleted_in_window: int = 0
    total_lines: int = 0
    total_devs: int = 0


# ---------------------------------------------------------------------------
# Core extraction: one modified file → function + hunk records
# ---------------------------------------------------------------------------

def get_method_code(source_code: Optional[str], start: int, end: int) -> str:
    if not source_code:
        return ''
    try:
        return '\n'.join(source_code.split('\n')[int(start) - 1:int(end)])
    except Exception:
        return ''


def changed_methods_both(mf) -> Tuple[Set, Set]:
    new_methods = mf.methods or []
    old_methods = mf.methods_before or []
    diff_parsed = mf.diff_parsed or {}
    added = diff_parsed.get('added', [])
    deleted = diff_parsed.get('deleted', [])
    methods_after = {
        method for line, _ in added for method in new_methods
        if method.start_line <= line <= method.end_line
    }
    methods_before = {
        method for line, _ in deleted for method in old_methods
        if method.start_line <= line <= method.end_line
    }
    return methods_after, methods_before

def extract_records_from_modified_file(
    commit_hash: str,
    mf,
) -> Tuple[List[FunctionRecord], Optional[HunkRecord]]:
    """
    Given a PyDriller ModifiedFile, produce FunctionRecords (Option A) and
    optionally a HunkRecord fallback.

    Option A convention:
      ADD    → code=post-commit body, code_before='', metrics_before=0
      MODIFY → code=post-commit body, code_before=pre-commit body
      DELETE → code='', code_before=pre-commit body, post-commit metrics=0
    """
    fname = mf.filename or mf.new_path or mf.old_path or ''
    post_src = mf.source_code or ''
    pre_src = mf.source_code_before or ''
    records: List[FunctionRecord] = []

    changed_methods = mf.changed_methods or []
    if changed_methods:
        try:
            methods_after, methods_before = changed_methods_both(mf)
        except Exception:
            methods_after, methods_before = set(), set()

        after_by_name = {method.name: method for method in methods_after}
        before_by_name = {method.name: method for method in methods_before}
        is_refactor = len(changed_methods) > REFACTOR_METHOD_THRESHOLD

        after_only = {name: method for name, method in after_by_name.items() if name not in before_by_name}
        before_only = {name: method for name, method in before_by_name.items() if name not in after_by_name}
        both_names = {name for name in after_by_name if name in before_by_name}

        rename_pairs: Dict[str, str] = {}
        for before_name in list(before_only):
            best_score, best_after = 0.0, None
            for after_name in list(after_only):
                score = SequenceMatcher(None, before_name, after_name).ratio()
                if score > best_score:
                    best_score, best_after = score, after_name
            if best_score >= FUZZY_RENAME_THRESHOLD and best_after is not None:
                rename_pairs[before_name] = best_after
        for before_name, after_name in rename_pairs.items():
            before_only.pop(before_name, None)
            after_only.pop(after_name, None)

        if is_refactor:
            for method in after_by_name.values():
                pre_method = before_by_name.get(method.name)
                records.append(FunctionRecord(
                    hash=commit_hash,
                    name=method.name,
                    filename=fname,
                    num_lines_of_code=method.nloc,
                    complexity=method.complexity,
                    token_count=method.token_count,
                    parameters=str(method.parameters),
                    signature=method.long_name,
                    start_line=method.start_line,
                    end_line=method.end_line,
                    length=method.length,
                    top_nesting_level=method.top_nesting_level,
                    code=get_method_code(post_src, method.start_line, method.end_line),
                    code_before=get_method_code(pre_src, method.start_line, method.end_line),
                    loc_before=pre_method.nloc if pre_method else 0,
                    complexity_before=pre_method.complexity if pre_method else 0,
                    tokens_before=pre_method.token_count if pre_method else 0,
                    function_change_type='REFACTOR',
                    extraction_method='normal_direct',
                ))
        else:
            for name in both_names:
                method_after = after_by_name[name]
                method_before = before_by_name[name]
                records.append(FunctionRecord(
                    hash=commit_hash,
                    name=method_after.name,
                    filename=fname,
                    num_lines_of_code=method_after.nloc,
                    complexity=method_after.complexity,
                    token_count=method_after.token_count,
                    parameters=str(method_after.parameters),
                    signature=method_after.long_name,
                    start_line=method_after.start_line,
                    end_line=method_after.end_line,
                    length=method_after.length,
                    top_nesting_level=method_after.top_nesting_level,
                    code=get_method_code(post_src, method_after.start_line, method_after.end_line),
                    code_before=get_method_code(pre_src, method_before.start_line, method_before.end_line),
                    loc_before=method_before.nloc,
                    complexity_before=method_before.complexity,
                    tokens_before=method_before.token_count,
                    function_change_type='MODIFY',
                    extraction_method='normal_direct',
                ))

            for method in after_only.values():
                records.append(FunctionRecord(
                    hash=commit_hash,
                    name=method.name,
                    filename=fname,
                    num_lines_of_code=method.nloc,
                    complexity=method.complexity,
                    token_count=method.token_count,
                    parameters=str(method.parameters),
                    signature=method.long_name,
                    start_line=method.start_line,
                    end_line=method.end_line,
                    length=method.length,
                    top_nesting_level=method.top_nesting_level,
                    code=get_method_code(post_src, method.start_line, method.end_line),
                    code_before='',
                    loc_before=0,
                    complexity_before=0,
                    tokens_before=0,
                    function_change_type='ADD',
                    extraction_method='normal_direct',
                ))

            for method in before_only.values():
                records.append(FunctionRecord(
                    hash=commit_hash,
                    name=method.name,
                    filename=fname,
                    num_lines_of_code=0,
                    complexity=0,
                    token_count=0,
                    parameters=str(method.parameters),
                    signature=method.long_name,
                    start_line=0,
                    end_line=0,
                    length=0,
                    top_nesting_level=method.top_nesting_level,
                    code='',
                    code_before=get_method_code(pre_src, method.start_line, method.end_line),
                    loc_before=method.nloc,
                    complexity_before=method.complexity,
                    tokens_before=method.token_count,
                    function_change_type='DELETE',
                    extraction_method='normal_direct',
                ))

            for before_name, after_name in rename_pairs.items():
                method_after = after_by_name[after_name]
                method_before = before_by_name[before_name]
                records.append(FunctionRecord(
                    hash=commit_hash,
                    name=method_after.name,
                    filename=fname,
                    num_lines_of_code=method_after.nloc,
                    complexity=method_after.complexity,
                    token_count=method_after.token_count,
                    parameters=str(method_after.parameters),
                    signature=method_after.long_name,
                    start_line=method_after.start_line,
                    end_line=method_after.end_line,
                    length=method_after.length,
                    top_nesting_level=method_after.top_nesting_level,
                    code=get_method_code(post_src, method_after.start_line, method_after.end_line),
                    code_before=get_method_code(pre_src, method_before.start_line, method_before.end_line),
                    loc_before=method_before.nloc,
                    complexity_before=method_before.complexity,
                    tokens_before=method_before.token_count,
                    function_change_type='RENAME',
                    extraction_method='normal_direct',
                ))

    # --- Hunk fallback: C/C++ file with diff but no functions ---
    hunk: Optional[HunkRecord] = None
    diff_str = mf.diff or ''
    if not records and diff_str:
        diff_lines = diff_str.splitlines()
        added_lines = [l[1:] for l in diff_lines if l.startswith('+') and not l.startswith('+++')]
        capped_added_lines = added_lines[:200]
        hunk_code = '\n'.join(capped_added_lines)
        nloc = len(capped_added_lines)
        hunk = HunkRecord(
            hash=commit_hash,
            filename=fname,
            name=f'hunk:{fname}',
            num_lines_of_code=nloc,
            code=hunk_code,
            start_line=0,
            end_line=nloc,
            length=nloc,
        )

    return records, hunk


def extract_full_records_from_commit(
    commit_hash: str,
    commit,
) -> Tuple[Dict, List[FunctionRecord], List[FileRecord], List[HunkRecord]]:
    """
    Extract matching features + full function/file/hunk records from a PyDriller commit.

    Returns:
        feature_dict: matching feature dict (for NN matching)
        func_records: list of FunctionRecord (Option A schema)
        file_records: list of FileRecord
        hunk_records: list of HunkRecord (fallback only)
    """
    func_records: List[FunctionRecord] = []
    file_records: List[FileRecord] = []
    hunk_records: List[HunkRecord] = []

    ccpp_files = [mf for mf in commit.modified_files if is_ccpp(mf.filename or mf.new_path or '')]
    if not ccpp_files:
        return None, [], [], []

    loc_changed = (commit.insertions or 0) + (commit.deletions or 0)

    for mf in ccpp_files:
        fname = mf.filename or mf.new_path or mf.old_path or ''
        post_src = mf.source_code or ''
        pre_src = mf.source_code_before or ''

        # File record
        post_fns_for_metrics = parse_functions_from_source(fname, post_src)
        file_nloc, file_cpx, file_tok = file_aggregate_metrics(post_fns_for_metrics)
        fr = FileRecord(
            hash=commit_hash,
            filename=fname,
            old_path=mf.old_path or '',
            new_path=mf.new_path or '',
            change_type=str(mf.change_type.name) if mf.change_type else '',
            diff=mf.diff or '',
            diff_parsed=json.dumps(mf.diff_parsed or {}),
            num_lines_added=mf.added_lines or 0,
            num_lines_deleted=mf.deleted_lines or 0,
            code_after=post_src,
            code_before=pre_src,
            num_method_changed=len(mf.changed_methods or []),
            num_lines_of_code=file_nloc,
            complexity=file_cpx,
            token_count=file_tok,
        )
        file_records.append(fr)

        # Function + hunk records
        fns, hunk = extract_records_from_modified_file(commit_hash, mf)
        func_records.extend(fns)
        if hunk is not None:
            hunk_records.append(hunk)

    if not func_records and not hunk_records:
        return None, [], file_records, []

    # Aggregate commit-level matching features
    non_delete_func_records = [r for r in func_records if r.function_change_type != 'DELETE']
    all_post_nlocs = [r.num_lines_of_code for r in non_delete_func_records]
    all_post_cpx = [r.complexity for r in non_delete_func_records]
    n_funcs = len(non_delete_func_records)
    max_loc = max(all_post_nlocs) if all_post_nlocs else 0
    max_cpx = max(all_post_cpx) if all_post_cpx else 0
    sum_cpx = sum(all_post_cpx) if all_post_cpx else 0

    feature_dict = {
        'hash': commit_hash,
        'commit_type': 'normal',
        'loc_changed': loc_changed,
        'num_files_changed': len(ccpp_files),
        'num_functions_changed': n_funcs,
        'max_function_loc': max_loc,
        'max_function_complexity': max_cpx,
        'sum_function_complexity': sum_cpx,
    }
    return feature_dict, func_records, file_records, hunk_records


def extract_candidate_features_from_commit(
    commit_hash: str,
    commit,
) -> Dict:
    """Extract only commit-level matching features without writing full downstream records."""
    ccpp_files = [mf for mf in commit.modified_files if is_ccpp(mf.filename or mf.new_path or '')]
    if not ccpp_files:
        return None

    loc_changed = (commit.insertions or 0) + (commit.deletions or 0)
    func_count = 0
    max_loc = 0
    max_cpx = 0
    sum_cpx = 0
    qualifies = False

    for mf in ccpp_files:
        changed_methods = mf.changed_methods or []
        if changed_methods:
            qualifies = True
            try:
                methods_after, methods_before = changed_methods_both(mf)
            except Exception:
                methods_after, methods_before = set(), set()

            after_by_name = {method.name: method for method in methods_after}
            before_by_name = {method.name: method for method in methods_before}
            is_refactor = len(changed_methods) > REFACTOR_METHOD_THRESHOLD

            if is_refactor:
                metrics_methods = list(after_by_name.values())
            else:
                after_only = {name: method for name, method in after_by_name.items() if name not in before_by_name}
                before_only = {name: method for name, method in before_by_name.items() if name not in after_by_name}
                both_names = {name for name in after_by_name if name in before_by_name}

                rename_pairs: Dict[str, str] = {}
                for before_name in list(before_only):
                    best_score, best_after = 0.0, None
                    for after_name in list(after_only):
                        score = SequenceMatcher(None, before_name, after_name).ratio()
                        if score > best_score:
                            best_score, best_after = score, after_name
                    if best_score >= FUZZY_RENAME_THRESHOLD and best_after is not None:
                        rename_pairs[before_name] = best_after
                for before_name, after_name in rename_pairs.items():
                    before_only.pop(before_name, None)
                    after_only.pop(after_name, None)

                metrics_methods = (
                    [after_by_name[name] for name in both_names]
                    + list(after_only.values())
                    + [after_by_name[after_name] for after_name in rename_pairs.values()]
                )

            # Keep commit-level matching features aligned with extracted post-state
            # function records by excluding DELETE-only functions from the count.
            func_count += len(metrics_methods)
            for method in metrics_methods:
                max_loc = max(max_loc, int(getattr(method, 'nloc', 0) or 0))
                cpx = int(getattr(method, 'complexity', 0) or 0)
                max_cpx = max(max_cpx, cpx)
                sum_cpx += cpx

        elif mf.diff:
            qualifies = True

    if not qualifies:
        return None

    return {
        'hash': commit_hash,
        'commit_type': 'normal',
        'loc_changed': loc_changed,
        'num_files_changed': len(ccpp_files),
        'num_functions_changed': func_count,
        'max_function_loc': max_loc,
        'max_function_complexity': max_cpx,
        'sum_function_complexity': sum_cpx,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Load exclusions and build VCC feature table
# ---------------------------------------------------------------------------

def load_exclusions(mapping_csv: str) -> Tuple[Set[str], Set[str], pd.DataFrame]:
    gt = pd.read_csv(mapping_csv)
    gt['vcc_list'] = gt['vcc_hash'].apply(
        lambda v: ast.literal_eval(v)
        if pd.notna(v) and str(v).startswith('[')
        else ([v] if pd.notna(v) else [])
    )
    fc_hashes = set(gt['fc_hash'].dropna())
    vcc_hashes = set(h for lst in gt['vcc_list'] for h in lst if h)
    per_repo_vcc = (
        gt.groupby('repo_url')['vcc_list']
        .apply(lambda x: len(set(h for lst in x for h in lst if h)))
        .reset_index()
    )
    per_repo_vcc.columns = ['repo_url', 'vcc_count']
    return fc_hashes, vcc_hashes, per_repo_vcc


def load_repo_fc_hashes(mapping_csv: str) -> Dict[str, Set[str]]:
    gt = pd.read_csv(mapping_csv, usecols=['repo_url', 'fc_hash'])
    result: Dict[str, Set[str]] = {}
    for repo_url, group in gt.groupby('repo_url'):
        result[repo_url] = set(group['fc_hash'].dropna())
    return result


def build_vcc_features(icvulpp_root: str) -> pd.DataFrame:
    """Build VCC feature table from commit_info_full + function_info_new."""
    commit_csv = os.path.join(icvulpp_root, 'data/graph_data/commit_info_full.csv')
    func_csv = os.path.join(icvulpp_root, 'data/graph_data/function_info_new.csv')

    commit_df = pd.read_csv(
        commit_csv,
        usecols=['hash', 'repo_url', 'commit_type', 'num_lines_added', 'num_lines_deleted', 'num_files_changed'],
    )
    vcc_commits = commit_df[commit_df['commit_type'] == 'VCC'].copy()
    vcc_commits['loc_changed'] = vcc_commits['num_lines_added'] + vcc_commits['num_lines_deleted']

    func_df = pd.read_csv(func_csv, usecols=['hash', 'commit_label', 'num_lines_of_code', 'complexity'])
    vcc_funcs = func_df[func_df['commit_label'] == 'VCC']

    func_agg = vcc_funcs.groupby('hash').agg(
        num_functions_changed=('num_lines_of_code', 'count'),
        max_function_loc=('num_lines_of_code', 'max'),
        max_function_complexity=('complexity', 'max'),
        sum_function_complexity=('complexity', 'sum'),
    ).reset_index()

    vcc_features = vcc_commits[['hash', 'repo_url', 'loc_changed', 'num_files_changed']].merge(
        func_agg, on='hash', how='left'
    )
    func_cols = ['num_functions_changed', 'max_function_loc', 'max_function_complexity', 'sum_function_complexity']
    vcc_features[func_cols] = vcc_features[func_cols].fillna(0)
    logging.info(f'VCC feature table: {len(vcc_features)} rows')
    return vcc_features


# ---------------------------------------------------------------------------
# Phase 2a — Use existing v1 normal pool (no extraction)
# ---------------------------------------------------------------------------

def load_existing_candidate_pool(icvulpp_root: str) -> pd.DataFrame:
    """Build candidate pool from v1 normals. C/C++ filtered via file_info."""
    normal_dir = os.path.join(icvulpp_root, 'data/normal_commits_all')
    logging.info('Loading existing v1 normal commit pool...')

    commit_df = pd.read_csv(
        os.path.join(normal_dir, 'commit_info_normal_all.csv'),
        usecols=['hash', 'repo_url', 'num_lines_added', 'num_lines_deleted', 'num_files_changed'],
    )
    commit_df['loc_changed'] = commit_df['num_lines_added'] + commit_df['num_lines_deleted']

    file_df = pd.read_csv(os.path.join(normal_dir, 'file_info_normal_all.csv'), usecols=['hash'])
    ccpp_hashes = set(file_df['hash'].unique())
    commit_df = commit_df[commit_df['hash'].isin(ccpp_hashes)].copy()

    func_df = pd.read_csv(
        os.path.join(normal_dir, 'function_info_normal_all.csv'),
        usecols=['hash', 'num_lines_of_code', 'complexity'],
    )
    func_agg = func_df.groupby('hash').agg(
        num_functions_changed=('num_lines_of_code', 'count'),
        max_function_loc=('num_lines_of_code', 'max'),
        max_function_complexity=('complexity', 'max'),
        sum_function_complexity=('complexity', 'sum'),
    ).reset_index()

    candidates = commit_df[['hash', 'repo_url', 'loc_changed', 'num_files_changed']].merge(
        func_agg, on='hash', how='left'
    )
    func_cols = ['num_functions_changed', 'max_function_loc', 'max_function_complexity', 'sum_function_complexity']
    candidates[func_cols] = candidates[func_cols].fillna(0)
    candidates = candidates.drop_duplicates('hash').reset_index(drop=True)
    logging.info(f'Candidate pool (existing): {len(candidates)} unique C/C++ commits')
    return candidates


# ---------------------------------------------------------------------------
# Phase 2b — Fresh PyDriller enumeration with full extraction
# ---------------------------------------------------------------------------

def clone_or_get_repo(repo_url: str, cache_dir: str) -> Optional[str]:
    def _run_git(candidate_path: str, args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(
            ['git', *args],
            cwd=candidate_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _reachable_commit_count(candidate_path: str) -> int:
        return int(_run_git(candidate_path, ['rev-list', '--count', 'HEAD']).stdout.strip())

    def _is_complete_repo(candidate_path: str) -> bool:
        shallow = _run_git(candidate_path, ['rev-parse', '--is-shallow-repository']).stdout.strip()
        rev_count = _reachable_commit_count(candidate_path)
        return shallow == 'false' and rev_count >= MIN_REACHABLE_COMMITS

    def _repair_repo(candidate_path: str) -> bool:
        try:
            shallow = _run_git(candidate_path, ['rev-parse', '--is-shallow-repository']).stdout.strip()
            rev_count = _reachable_commit_count(candidate_path)
            if shallow == 'true' or rev_count < MIN_REACHABLE_COMMITS:
                depth_needed = max(MIN_REACHABLE_COMMITS - rev_count, MIN_REACHABLE_COMMITS)
                logging.info(
                    f'Cached repo is incomplete ({rev_count} reachable commits); '
                    f'deepening {candidate_path} by {depth_needed}'
                )
                _run_git(candidate_path, ['fetch', f'--deepen={depth_needed}', '--tags', 'origin'], timeout=900)
            else:
                _run_git(candidate_path, ['fetch', '--all', '--tags', '--quiet'], timeout=300)
            if _is_complete_repo(candidate_path):
                return True
            if _run_git(candidate_path, ['rev-parse', '--is-shallow-repository']).stdout.strip() == 'true':
                logging.info(f'Cached repo still shallow after deepen; unshallowing {candidate_path}')
                _run_git(candidate_path, ['fetch', '--unshallow', '--tags', 'origin'], timeout=900)
            return _is_complete_repo(candidate_path)
        except Exception as exc:
            logging.warning(f'Failed to repair cached repo {candidate_path}: {exc}')
            return False

    owner = repo_url.rstrip('/').split('/')[-2]
    name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    local_path = os.path.join(cache_dir, f'{owner}__{name}')

    candidate_paths = [local_path]
    if os.path.isdir(cache_dir):
        for entry in sorted(os.listdir(cache_dir)):
            entry_path = os.path.join(cache_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry_path == local_path:
                continue
            entry_norm = entry.lower()
            name_norm = name.lower()
            owner_norm = owner.lower()
            if (
                entry_norm == name_norm
                or entry_norm.endswith(f'__{name_norm}')
                or entry_norm == f'{owner_norm}__{name_norm}'
            ):
                candidate_paths.append(entry_path)

    for candidate_path in candidate_paths:
        if not os.path.exists(candidate_path):
            continue
        try:
            repo = Repo(candidate_path)
            _ = repo.git_dir
            # Guard against broken cached repos that still have a .git dir
            _run_git(candidate_path, ['rev-parse', '--verify', 'HEAD'], timeout=30)
            _run_git(candidate_path, ['log', '-1', '--format=%H'], timeout=30)
            if not _is_complete_repo(candidate_path):
                if not _repair_repo(candidate_path):
                    raise RuntimeError('cached repo is incomplete after repair')
            logging.info(f'Using cached repo: {candidate_path}')
            return candidate_path
        except Exception:
            logging.warning(f'Broken/corrupt clone at {candidate_path} — removing')
            shutil.rmtree(candidate_path, ignore_errors=True)

    logging.info(f'Cloning {repo_url} -> {local_path}')
    os.makedirs(cache_dir, exist_ok=True)
    try:
        Repo.clone_from(repo_url, local_path, multi_options=['--no-single-branch'])
        return local_path
    except Exception as e:
        logging.error(f'Clone failed for {repo_url}: {e}')
        shutil.rmtree(local_path, ignore_errors=True)
        return None


def enumerate_repo_candidates(
    repo_url: str,
    repo_path: str,
    all_excluded: Set[str],
    repo_fc_hashes: Set[str],
    with_ownership: bool = True,
    full_extract: bool = True,
    max_commits: Optional[int] = None,
) -> Tuple[List[Dict], List[FunctionRecord], List[FileRecord], List[HunkRecord], List[OwnershipRecord]]:
    """
    Enumerate C/C++ normal commits from a repo and extract full records.
    Returns (feature_rows, func_records, file_records, hunk_records, ownership_records).
    """
    from pydriller import Repository

    # git log for fast hash enumeration — pre-filtered to commits touching C/C++ files.
    # This avoids loading PyDriller diffs for commits that never touch C/C++ at all.
    _CCPP_GLOB_PATTERNS = [
        '*.c', '*.h', '*.cpp', '*.cxx', '*.cc', '*.hpp', '*.hh', '*.hxx',
        '*.ec', '*.ecp', '*.pgc', '*.pcc',
    ]
    try:
        result = subprocess.run(
            ['git', 'log', '--format=%H %aI', '--no-merges', '--first-parent', '--']
            + _CCPP_GLOB_PATTERNS,
            cwd=repo_path, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logging.error(f'git log failed for {repo_url}: {result.stderr.strip()}')
            return [], [], [], [], []
        all_commits_ordered = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                h = line.split(' ', 1)[0]
                all_commits_ordered.append(h)
    except Exception as e:
        logging.error(f'git log failed for {repo_url}: {e}')
        return [], [], [], [], []

    # ±1 adjacency exclusion around FC commits
    hash_to_idx = {h: i for i, h in enumerate(all_commits_ordered)}
    adjacent_excluded: Set[str] = set()
    for fc_hash in repo_fc_hashes:
        idx = hash_to_idx.get(fc_hash)
        if idx is None:
            continue
        if idx > 0:
            adjacent_excluded.add(all_commits_ordered[idx - 1])
        if idx < len(all_commits_ordered) - 1:
            adjacent_excluded.add(all_commits_ordered[idx + 1])

    full_excluded = all_excluded | adjacent_excluded
    candidate_hashes = [h for h in all_commits_ordered if h not in full_excluded]
    if max_commits:
        candidate_hashes = candidate_hashes[:max_commits]

    if not candidate_hashes:
        return [], [], [], [], []

    feature_rows: List[Dict] = []
    all_func_records: List[FunctionRecord] = []
    all_file_records: List[FileRecord] = []
    all_hunk_records: List[HunkRecord] = []
    all_ownership_records: List[OwnershipRecord] = []

    # PyDriller's only_commits path still walks large portions of history before filtering.
    # For smoke tests, traverse the requested hashes one-by-one so max_commits is a real cap.
    if max_commits:
        commit_iterables = (
            Repository(repo_path, single=commit_hash).traverse_commits()
            for commit_hash in candidate_hashes
        )
    else:
        commit_iterables = [Repository(repo_path, only_commits=candidate_hashes).traverse_commits()]

    for commit_iter in commit_iterables:
        for commit in commit_iter:
            try:
                if full_extract:
                    feature_dict, func_recs, file_recs, hunk_recs = extract_full_records_from_commit(
                        commit.hash, commit
                    )
                else:
                    feature_dict = extract_candidate_features_from_commit(commit.hash, commit)
                    func_recs, file_recs, hunk_recs = [], [], []
            except Exception as e:
                logging.debug(f'Extraction failed for {commit.hash}: {e}')
                continue

            if feature_dict is None:
                continue  # no C/C++ files

            feature_dict['repo_url'] = repo_url
            feature_dict['author_date'] = commit.author_date.isoformat() if commit.author_date else ''
            feature_rows.append(feature_dict)
            all_func_records.extend(func_recs)
            all_file_records.extend(file_recs)
            all_hunk_records.extend(hunk_recs)
            if with_ownership and file_recs:
                all_ownership_records.extend(
                    extract_ownership_for_commit(commit.hash, repo_path, file_recs, commit.author_date)
                )

    logging.info(
        f'{repo_url}: {len(feature_rows)} C/C++ candidates, '
        f'{len(all_func_records)} functions, {len(all_file_records)} files, '
        f'{len(all_hunk_records)} hunks, {len(all_ownership_records)} ownership rows'
    )
    return feature_rows, all_func_records, all_file_records, all_hunk_records, all_ownership_records


def append_records_to_csv(records: List, columns: List[str], path: str, lock: threading.Lock) -> None:
    """Thread-safe CSV append from dataclass list; writes header only on first write."""
    if not records:
        return
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows, columns=columns)
    with lock:
        write_header = not os.path.exists(path)
        df.to_csv(path, mode='a', header=write_header, index=False)


def write_csv_rows_append(rows: List[Dict], columns: List[str], path: str, lock: threading.Lock) -> None:
    """Thread-safe CSV append from plain dict list; writes header only on first write."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    with lock:
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        df.to_csv(path, mode='a', header=write_header, index=False)


_CANDIDATE_COLS = ['hash', 'commit_type', 'repo_url', 'author_date'] + MATCHING_FEATURES


def append_feature_rows(rows: List[Dict], path: str, lock: threading.Lock) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for col in _CANDIDATE_COLS:
        if col not in df.columns:
            df[col] = '' if col == 'author_date' else 0
    df = df[_CANDIDATE_COLS]
    with lock:
        write_header = not os.path.exists(path)
        df.to_csv(path, mode='a', header=write_header, index=False)


def repo_slug(repo_url: str) -> str:
    parts = repo_url.rstrip('/').replace('.git', '').split('/')
    if len(parts) >= 2:
        return f'{parts[-2]}__{parts[-1]}'
    return repo_url.rstrip('/').replace('/', '__')


def write_csv_rows(path: str, columns: List[str], rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def write_repo_shard_outputs(
    repo_url: str,
    output_dir: str,
    feat_rows: List[Dict],
    func_recs: List,
    file_recs: List,
    hunk_recs: List,
    ownership_recs: List,
    failed: bool,
) -> None:
    repo_dir = os.path.join(output_dir, 'per_repo', repo_slug(repo_url))
    os.makedirs(repo_dir, exist_ok=True)

    write_csv_rows(
        os.path.join(repo_dir, 'candidates_raw.csv'),
        _CANDIDATE_COLS,
        feat_rows,
    )
    write_csv_rows(
        os.path.join(repo_dir, 'function_info_normal_v2.csv'),
        FUNCTION_COLUMNS,
        [asdict(r) for r in func_recs],
    )
    write_csv_rows(
        os.path.join(repo_dir, 'file_info_normal_v2.csv'),
        FILE_COLUMNS,
        [asdict(r) for r in file_recs],
    )
    write_csv_rows(
        os.path.join(repo_dir, 'hunk_info_normal_v2.csv'),
        HUNK_COLUMNS,
        [asdict(r) for r in hunk_recs],
    )
    write_csv_rows(
        os.path.join(repo_dir, 'ownership_normal_v2.csv'),
        OWNERSHIP_COLUMNS,
        [asdict(r) for r in ownership_recs],
    )

    summary = {
        'repo_url': repo_url,
        'failed': failed,
        'candidate_rows': len(feat_rows),
        'function_rows': len(func_recs),
        'file_rows': len(file_recs),
        'hunk_rows': len(hunk_recs),
        'ownership_rows': len(ownership_recs),
        'written_at_epoch': int(time.time()),
    }
    with open(os.path.join(repo_dir, 'repo_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


def enumerate_fresh_candidates(
    repos: List[str],
    all_excluded: Set[str],
    repo_fc_map: Dict[str, Set[str]],
    repo_cache: str,
    output_dir: str,
    workers: int,
    with_ownership: bool = True,
    full_extract: bool = True,
    max_commits_per_repo: Optional[int] = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """Enumerate repos fresh via PyDriller. Checkpoints per-repo. Returns candidate DataFrame."""
    checkpoint_path = os.path.join(output_dir, 'done_repos_candidates.json')
    rows_path = os.path.join(output_dir, 'candidates_raw.csv')
    func_path = os.path.join(output_dir, 'function_info_normal_v2.csv')
    file_path = os.path.join(output_dir, 'file_info_normal_v2.csv')
    hunk_path = os.path.join(output_dir, 'hunk_info_normal_v2.csv')
    ownership_path = os.path.join(output_dir, 'ownership_normal_v2.csv')

    done_repos: Set[str] = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            done_repos = set(json.load(f))
        logging.info(f'Resuming: {len(done_repos)} repos already done')

    pending = [r for r in repos if r not in done_repos]

    checkpoint_lock = threading.Lock()
    feat_lock = threading.Lock()
    func_lock = threading.Lock()
    file_lock = threading.Lock()
    hunk_lock = threading.Lock()
    ownership_lock = threading.Lock()

    def _worker(repo_url: str):
        tmp_dir = None
        try:
            if no_cache:
                import tempfile
                scratch = os.environ.get('VSC_SCRATCH', tempfile.gettempdir())
                tmp_dir = tempfile.mkdtemp(prefix='icvul_repo_', dir=scratch)
                repo_path = clone_or_get_repo(repo_url, tmp_dir)
            else:
                repo_path = clone_or_get_repo(repo_url, repo_cache)
            if repo_path is None:
                return repo_url, [], [], [], [], [], True
            feat_rows, func_recs, file_recs, hunk_recs, ownership_recs = enumerate_repo_candidates(
                repo_url, repo_path, all_excluded,
                repo_fc_map.get(repo_url, set()),
                with_ownership=with_ownership,
                full_extract=full_extract,
                max_commits=max_commits_per_repo,
            )
            return repo_url, feat_rows, func_recs, file_recs, hunk_recs, ownership_recs, False
        except Exception as e:
            logging.error(f'Enumeration failed for {repo_url}: {e}')
            return repo_url, [], [], [], [], [], True
        finally:
            if no_cache and tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logging.info(f'Temp clone deleted: {os.path.basename(tmp_dir)}')

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, r): r for r in pending}
        for future in as_completed(futures):
            repo_url = futures[future]
            try:
                repo_url_ret, feat_rows, func_recs, file_recs, hunk_recs, ownership_recs, failed = future.result()
            except Exception as e:
                logging.error(f'Worker exception for {repo_url}: {e}')
                repo_url_ret = repo_url
                feat_rows, func_recs, file_recs, hunk_recs, ownership_recs, failed = [], [], [], [], [], True

            write_repo_shard_outputs(
                repo_url=repo_url_ret,
                output_dir=output_dir,
                feat_rows=feat_rows,
                func_recs=func_recs,
                file_recs=file_recs,
                hunk_recs=hunk_recs,
                ownership_recs=ownership_recs,
                failed=failed,
            )

            if not failed:
                append_feature_rows(feat_rows, rows_path, feat_lock)
                if full_extract:
                    append_records_to_csv(func_recs, FUNCTION_COLUMNS, func_path, func_lock)
                    append_records_to_csv(file_recs, FILE_COLUMNS, file_path, file_lock)
                    append_records_to_csv(hunk_recs, HUNK_COLUMNS, hunk_path, hunk_lock)
                    append_records_to_csv(ownership_recs, OWNERSHIP_COLUMNS, ownership_path, ownership_lock)

            with checkpoint_lock:
                done_repos.add(repo_url_ret)
                with open(checkpoint_path, 'w') as f:
                    json.dump(sorted(done_repos), f, indent=2)

            logging.info(f'{repo_url_ret}: {"FAILED" if failed else f"+{len(feat_rows)} candidates"}')

            # Delete repo from cache immediately after processing to free disk space
            cached = find_cached_repo_path(repo_url_ret, repo_cache)
            if cached and os.path.isdir(cached):
                shutil.rmtree(cached, ignore_errors=True)
                logging.info(f'Cache deleted: {os.path.basename(cached)}')

    if os.path.exists(rows_path):
        candidates = pd.read_csv(rows_path)
        candidates = candidates.drop_duplicates('hash').reset_index(drop=True)
        logging.info(f'Fresh enumeration complete: {len(candidates)} unique C/C++ candidates')
        return candidates
    else:
        logging.warning('No candidates found in fresh enumeration')
        return pd.DataFrame(columns=['hash', 'repo_url'] + MATCHING_FEATURES)


def extract_selected_normal_records(
    selected_candidates: pd.DataFrame,
    repo_cache: str,
    output_dir: str,
    workers: int,
    with_ownership: bool = True,
    no_cache: bool = False,
) -> None:
    """Extract full rows only for the selected matched normal commits."""
    if selected_candidates.empty:
        logging.info('No matched normals selected for deferred extraction.')
        return

    func_path      = os.path.join(output_dir, 'function_info_normal_v2.csv')
    file_path      = os.path.join(output_dir, 'file_info_normal_v2.csv')
    hunk_path      = os.path.join(output_dir, 'hunk_info_normal_v2.csv')
    ownership_path = os.path.join(output_dir, 'ownership_normal_v2.csv')
    commit_path    = os.path.join(output_dir, 'commit_info_normal_v2.csv')

    COMMIT_INFO_COLUMNS = [
        'hash', 'msg', 'author', 'author_date', 'author_timezone',
        'committer', 'committer_date', 'committer_timezone',
        'in_main_branch', 'merge', 'parents',
        'num_lines_deleted', 'num_lines_added', 'num_lines_changed', 'num_files_changed',
        'dmm_unit_size', 'dmm_unit_complexity', 'dmm_unit_interfacing',
        'commit_type', 'cwe_id', 'repo_url',
    ]

    for path, columns in [
        (func_path,      FUNCTION_COLUMNS),
        (file_path,      FILE_COLUMNS),
        (hunk_path,      HUNK_COLUMNS),
        (ownership_path, OWNERSHIP_COLUMNS),
        (commit_path,    COMMIT_INFO_COLUMNS),
    ]:
        write_csv_rows(path, columns, [])

    func_lock   = threading.Lock()
    file_lock   = threading.Lock()
    hunk_lock   = threading.Lock()
    ownership_lock = threading.Lock()
    commit_lock = threading.Lock()

    repo_groups = (
        selected_candidates[['hash', 'repo_url']]
        .drop_duplicates()
        .groupby('repo_url')['hash']
        .apply(list)
        .to_dict()
    )

    def _worker(repo_url: str, hashes: List[str]):
        import tempfile
        tmp_dir = None
        try:
            if no_cache:
                scratch = os.environ.get('VSC_SCRATCH', tempfile.gettempdir())
                tmp_dir = tempfile.mkdtemp(prefix='icvul_extract_', dir=scratch)
                repo_path = clone_or_get_repo(repo_url, tmp_dir)
            else:
                repo_path = clone_or_get_repo(repo_url, repo_cache)

            if repo_path is None:
                return repo_url, [], [], [], [], [], True

            from pydriller import Repository

            all_func_records: List[FunctionRecord] = []
            all_file_records: List[FileRecord] = []
            all_hunk_records: List[HunkRecord] = []
            all_ownership_records: List[OwnershipRecord] = []
            all_commit_rows: List[Dict] = []

            for commit_hash in hashes:
                try:
                    for commit in Repository(repo_path, single=commit_hash).traverse_commits():
                        feature_dict, func_recs, file_recs, hunk_recs = extract_full_records_from_commit(
                            commit.hash, commit
                        )
                        if feature_dict is None:
                            continue
                        all_func_records.extend(func_recs)
                        all_file_records.extend(file_recs)
                        all_hunk_records.extend(hunk_recs)
                        if with_ownership and file_recs:
                            all_ownership_records.extend(
                                extract_ownership_for_commit(commit.hash, repo_path, file_recs, commit.author_date)
                            )
                        # commit_info row — aligned with commit_info_full.csv schema
                        all_commit_rows.append({
                            'hash':               commit.hash,
                            'msg':                (commit.msg or '').replace('\n', ' ')[:500],
                            'author':             commit.author.name if commit.author else '',
                            'author_date':        str(commit.author_date),
                            'author_timezone':    str(commit.author_timezone),
                            'committer':          commit.committer.name if commit.committer else '',
                            'committer_date':     str(commit.committer_date),
                            'committer_timezone': str(commit.committer_timezone),
                            'in_main_branch':     commit.in_main_branch,
                            'merge':              commit.merge,
                            'parents':            str(commit.parents),
                            'num_lines_deleted':  commit.deletions or 0,
                            'num_lines_added':    commit.insertions or 0,
                            'num_lines_changed':  (commit.insertions or 0) + (commit.deletions or 0),
                            'num_files_changed':  len(commit.modified_files),
                            'dmm_unit_size':      commit.dmm_unit_size,
                            'dmm_unit_complexity': commit.dmm_unit_complexity,
                            'dmm_unit_interfacing': commit.dmm_unit_interfacing,
                            'commit_type':        'normal',
                            'cwe_id':             '',
                            'repo_url':           repo_url,
                        })
                except Exception as exc:
                    logging.debug(f'Deferred extraction failed for {repo_url} {commit_hash}: {exc}')
                    continue

            return repo_url, all_func_records, all_file_records, all_hunk_records, all_ownership_records, all_commit_rows, False

        except Exception as e:
            logging.error(f'Worker failed for {repo_url}: {e}')
            return repo_url, [], [], [], [], [], True
        finally:
            if no_cache and tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logging.info(f'Temp clone deleted: {os.path.basename(tmp_dir)}')

    logging.info(
        f'Deferred extraction: {len(selected_candidates)} matched rows across {len(repo_groups)} repos'
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker, repo_url, hashes): repo_url
            for repo_url, hashes in repo_groups.items()
        }
        for future in as_completed(futures):
            repo_url = futures[future]
            try:
                repo_url_ret, func_recs, file_recs, hunk_recs, ownership_recs, commit_rows, failed = future.result()
            except Exception as exc:
                logging.error(f'Deferred extraction worker exception for {repo_url}: {exc}')
                continue

            append_records_to_csv(func_recs, FUNCTION_COLUMNS, func_path, func_lock)
            append_records_to_csv(file_recs, FILE_COLUMNS, file_path, file_lock)
            append_records_to_csv(hunk_recs, HUNK_COLUMNS, hunk_path, hunk_lock)
            append_records_to_csv(ownership_recs, OWNERSHIP_COLUMNS, ownership_path, ownership_lock)
            if commit_rows:
                write_csv_rows_append(commit_rows, COMMIT_INFO_COLUMNS, commit_path, commit_lock)
            logging.info(
                f'Deferred extraction complete for {repo_url_ret}: '
                f'{len(func_recs)} functions, {len(file_recs)} files, '
                f'{len(hunk_recs)} hunks, {len(ownership_recs)} ownership rows, '
                f'{len(commit_rows)} commit rows'
            )

    # ── Schema alignment: rename commit_type → commit_label + add before_change ──
    # graph_builder.py filters function_info on `commit_label` (line 132) and uses
    # `before_change` as a file→function edge feature (line 537, default=False for normals).
    # commit_info_normal_v2.csv already uses commit_type (matching commit_info_full.csv).
    for _path, _renames, _add_cols in [
        (func_path,      {'commit_type': 'commit_label'}, {'before_change': False}),
        (file_path,      {'commit_type': 'commit_label'}, {}),
        (ownership_path, {'commit_type': 'commit_label'}, {}),
    ]:
        if os.path.exists(_path):
            _df = pd.read_csv(_path)
            if _df.empty:
                continue
            _df.rename(columns=_renames, inplace=True)
            for _col, _val in _add_cols.items():
                if _col not in _df.columns:
                    _df[_col] = _val
            _df.to_csv(_path, index=False)
            logging.info(f'Schema-aligned {os.path.basename(_path)}: renamed {_renames}, added {list(_add_cols)}')


def find_cached_repo_path(repo_url: str, cache_dir: str) -> Optional[str]:
    """Resolve an existing cached repo directory for a GitHub URL."""
    owner, name = repo_url.rstrip('/').replace('.git', '').split('/')[-2:]
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None
    patterns = {
        name.lower(),
        f'{owner}__{name}'.lower(),
    }
    for candidate in cache_path.iterdir():
        if not candidate.is_dir():
            continue
        candidate_name = candidate.name.lower()
        if candidate_name in patterns or candidate_name.endswith(f'__{name.lower()}'):
            if (candidate / '.git').exists() or (candidate / 'HEAD').exists():
                return str(candidate)
    return None


def order_active_repos(
    repo_urls: List[str],
    per_repo_vcc: pd.DataFrame,
    repo_cache: str,
) -> List[str]:
    """Prioritize cached, VCC-heavy repos first to improve partial-run diversity."""
    vcc_count_lookup = per_repo_vcc.set_index('repo_url')['vcc_count'].to_dict()
    ranked = sorted(
        repo_urls,
        key=lambda repo_url: (
            0 if find_cached_repo_path(repo_url, repo_cache) else 1,
            -int(vcc_count_lookup.get(repo_url, 0)),
            repo_url,
        ),
    )
    cached_count = sum(1 for repo_url in ranked if find_cached_repo_path(repo_url, repo_cache))
    logging.info(
        f'Active repo order: {cached_count}/{len(ranked)} cached repos prioritized before uncached repos'
    )
    return ranked


# ---------------------------------------------------------------------------
# Phase 3 — Standardize and build NN index
# ---------------------------------------------------------------------------

def compute_scaler_params(candidates: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    params = {}
    for feat in MATCHING_FEATURES:
        vals = candidates[feat].fillna(0).values
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        if std_v == 0.0:
            std_v = 1.0
        params[feat] = {'mean': mean_v, 'std': std_v}
    return params


def standardize(df: pd.DataFrame, scaler_params: Dict[str, Dict[str, float]]) -> np.ndarray:
    X = np.zeros((len(df), len(MATCHING_FEATURES)), dtype=float)
    for j, feat in enumerate(MATCHING_FEATURES):
        vals = df[feat].fillna(0).values.astype(float)
        X[:, j] = (vals - scaler_params[feat]['mean']) / scaler_params[feat]['std']
    return X


# ---------------------------------------------------------------------------
# Phase 4 — 5-NN matching with reuse cap
# ---------------------------------------------------------------------------

def run_matching(
    X_vccs: np.ndarray,
    vcc_hashes: List[str],
    X_candidates: np.ndarray,
    candidate_hashes: List[str],
    candidate_repo_urls: List[str],
    reuse_cap: int = 5,
    n_matches: int = 5,
    overfetch: int = 50,
    distance_penalties: Optional[np.ndarray] = None,
    max_distance: Optional[float] = None,
    distance_ladder: Optional[List[float]] = None,
    max_matches_per_repo_per_vcc: int = 2,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, int]]:
    """5-NN matching with reuse cap. Returns (assignments, reuse_count)."""
    nn_model = NearestNeighbors(
        n_neighbors=min(overfetch, len(candidate_hashes)),
        metric='euclidean',
        algorithm='ball_tree',
    )
    nn_model.fit(X_candidates)

    reuse_count: Dict[str, int] = {}
    assignments: Dict[str, List[Tuple[str, float]]] = {}

    for i, vcc_hash in enumerate(vcc_hashes):
        base_dists, idxs = nn_model.kneighbors(X_vccs[i].reshape(1, -1))
        base_dists = base_dists[0]
        idxs = idxs[0]
        ranked_dists = base_dists.copy()

        if distance_penalties is not None:
            penalties = distance_penalties[idxs]
            ranked_dists = ranked_dists * penalties
            order = np.argsort(ranked_dists)
            ranked_dists = ranked_dists[order]
            base_dists = base_dists[order]
            idxs = idxs[order]

        cutoffs = distance_ladder if distance_ladder else ([max_distance] if max_distance is not None else [None])
        matches = []
        repo_counts_for_vcc: Dict[str, int] = {}
        used_hashes: Set[str] = set()
        for cutoff in cutoffs:
            for base_dist, idx in zip(base_dists, idxs):
                if cutoff is not None and base_dist > cutoff:
                    continue
                cand_hash = candidate_hashes[idx]
                if cand_hash in used_hashes:
                    continue
                cand_repo = candidate_repo_urls[idx]
                if repo_counts_for_vcc.get(cand_repo, 0) >= max_matches_per_repo_per_vcc:
                    continue
                if reuse_count.get(cand_hash, 0) < reuse_cap:
                    matches.append((cand_hash, float(base_dist)))
                    used_hashes.add(cand_hash)
                    reuse_count[cand_hash] = reuse_count.get(cand_hash, 0) + 1
                    repo_counts_for_vcc[cand_repo] = repo_counts_for_vcc.get(cand_repo, 0) + 1
                    if len(matches) >= n_matches:
                        break
            if len(matches) >= n_matches:
                break
        assignments[vcc_hash] = matches

    vcc_with_full = sum(1 for m in assignments.values() if len(m) == n_matches)
    logging.info(
        f'Matching done: {vcc_with_full}/{len(vcc_hashes)} VCCs have {n_matches} matches, '
        f'{len(reuse_count)} unique normals selected'
    )
    return assignments, reuse_count


# ---------------------------------------------------------------------------
# Phase 5 — Repo concentration cap
# ---------------------------------------------------------------------------

def check_and_apply_repo_cap(
    assignments: Dict[str, List[Tuple[str, float]]],
    vcc_features: pd.DataFrame,
    candidates: pd.DataFrame,
    X_vccs: np.ndarray,
    vcc_hashes_list: List[str],
    X_candidates: np.ndarray,
    candidate_hashes_list: List[str],
    cap_threshold: float = 0.40,
    max_distance: Optional[float] = None,
    distance_ladder: Optional[List[float]] = None,
    max_matches_per_repo_per_vcc: int = 2,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, int], bool]:
    """Check repo concentration; re-run with distance penalty if > threshold (one pass)."""
    matched_normal_hashes = list({nh for matches in assignments.values() for nh, _ in matches})
    if not matched_normal_hashes:
        return assignments, {}, False

    cand_repo_lookup = candidates.set_index('hash')['repo_url'].to_dict()
    normal_repos = [cand_repo_lookup.get(h, '') for h in matched_normal_hashes]
    repo_counts = pd.Series(normal_repos).value_counts()
    total_matched = len(matched_normal_hashes)
    repo_shares = (repo_counts / total_matched).to_dict()
    max_share = max(repo_shares.values()) if repo_shares else 0.0

    logging.info(f'Repo concentration: max_share={max_share:.3f} (threshold={cap_threshold})')

    if max_share <= cap_threshold:
        logging.info('No repo concentration cap needed.')
        reuse_count: Dict[str, int] = {}
        for matches in assignments.values():
            for nh, _ in matches:
                reuse_count[nh] = reuse_count.get(nh, 0) + 1
        return assignments, reuse_count, False

    vcc_repo_counts = vcc_features['repo_url'].value_counts().to_dict()
    total_vccs = len(vcc_features)
    vcc_repo_shares = {r: c / total_vccs for r, c in vcc_repo_counts.items()}

    candidate_repo_list = candidates['repo_url'].tolist()
    penalties = np.ones(len(candidate_repo_list), dtype=float)
    for j, repo in enumerate(candidate_repo_list):
        if repo_shares.get(repo, 0.0) > cap_threshold:
            repo_vcc_share = vcc_repo_shares.get(repo, 0.001)
            max_share_for_repo = min(0.30, 3.0 * repo_vcc_share)
            actual_share = repo_shares[repo]
            penalties[j] = max(1.0, actual_share / max_share_for_repo)

    logging.info('Applying repo concentration penalties (one re-pass)...')
    assignments_new, reuse_count_new = run_matching(
        X_vccs=X_vccs,
        vcc_hashes=vcc_hashes_list,
        X_candidates=X_candidates,
        candidate_hashes=candidate_hashes_list,
        candidate_repo_urls=candidate_repo_list,
        distance_penalties=penalties,
        max_distance=max_distance,
        distance_ladder=distance_ladder,
        max_matches_per_repo_per_vcc=max_matches_per_repo_per_vcc,
    )
    return assignments_new, reuse_count_new, True


# ---------------------------------------------------------------------------
# Phase 6 — Write outputs
# ---------------------------------------------------------------------------

def write_outputs(
    assignments: Dict[str, List[Tuple[str, float]]],
    reuse_count: Dict[str, int],
    vcc_features: pd.DataFrame,
    candidates: pd.DataFrame,
    fc_commits: pd.DataFrame,
    scaler_params: Dict[str, Dict[str, float]],
    output_dir: str,
    cap_triggered: bool,
    max_distance: Optional[float],
    distance_ladder: Optional[List[float]],
    max_matches_per_repo_per_vcc: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cand_lookup = candidates.set_index('hash')[['repo_url']].to_dict('index')
    matched_normal_hashes = {nh for matches in assignments.values() for nh, _ in matches}

    # benchmark_manifest.csv
    manifest_rows = []
    for _, row in vcc_features.iterrows():
        manifest_rows.append({'hash': row['hash'], 'commit_type': 'VCC', 'repo_url': row['repo_url'], 'label': 1})
    for _, row in fc_commits.iterrows():
        manifest_rows.append({'hash': row['hash'], 'commit_type': 'FC', 'repo_url': row['repo_url'], 'label': 0})
    for nh in matched_normal_hashes:
        info = cand_lookup.get(nh, {})
        manifest_rows.append({'hash': nh, 'commit_type': 'normal', 'repo_url': info.get('repo_url', ''), 'label': 0})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(output_dir, 'benchmark_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False)
    logging.info(f'Wrote {len(manifest_df)} rows to {manifest_path}')

    # sampling_audit.csv
    audit_rows = []
    for vcc_hash, matches in assignments.items():
        vcc_repo_series = vcc_features.loc[vcc_features['hash'] == vcc_hash, 'repo_url']
        vcc_repo_url = vcc_repo_series.iloc[0] if len(vcc_repo_series) > 0 else ''
        for rank, (nh, dist) in enumerate(matches, start=1):
            info = cand_lookup.get(nh, {})
            normal_repo_url = info.get('repo_url', '')
            audit_rows.append({
                'vcc_hash': vcc_hash, 'normal_hash': nh, 'distance': dist,
                'match_rank': rank, 'vcc_repo_url': vcc_repo_url,
                'normal_repo_url': normal_repo_url,
                'same_repo': vcc_repo_url == normal_repo_url,
            })
    audit_df = pd.DataFrame(audit_rows)
    audit_path = os.path.join(output_dir, 'sampling_audit.csv')
    audit_df.to_csv(audit_path, index=False)
    logging.info(f'Wrote {len(audit_df)} rows to {audit_path}')

    # audit_summary.json
    match_counts = [len(m) for m in assignments.values()]
    distance_values = audit_df['distance'].astype(float).tolist() if not audit_df.empty else []
    reuse_hist: Dict[str, int] = {}
    for cnt in reuse_count.values():
        reuse_hist[str(cnt)] = reuse_hist.get(str(cnt), 0) + 1

    normal_repos_list = [cand_lookup.get(nh, {}).get('repo_url', '') for nh in matched_normal_hashes]
    repo_counts_series = pd.Series(normal_repos_list).value_counts()
    vcc_repo_counts = vcc_features['repo_url'].value_counts().to_dict()
    total_vccs = len(vcc_features)
    repo_concentration = {
        repo: {
            'normal_share': float(count / len(matched_normal_hashes)),
            'vcc_share': float(vcc_repo_counts.get(repo, 0) / total_vccs),
            'normal_count': int(count),
        }
        for repo, count in repo_counts_series.items()
        if float(count / len(matched_normal_hashes)) > 0.01
    }

    matched_cands = candidates[candidates['hash'].isin(matched_normal_hashes)]
    feature_distribution = {}
    for feat in MATCHING_FEATURES:
        vcc_vals = vcc_features[feat].fillna(0).values
        norm_vals = matched_cands[feat].fillna(0).values if len(matched_cands) > 0 else np.array([])
        feature_distribution[feat] = {
            'vcc_mean': float(np.mean(vcc_vals)) if len(vcc_vals) > 0 else 0.0,
            'vcc_std': float(np.std(vcc_vals)) if len(vcc_vals) > 0 else 0.0,
            'vcc_median': float(np.median(vcc_vals)) if len(vcc_vals) > 0 else 0.0,
            'normal_mean': float(np.mean(norm_vals)) if len(norm_vals) > 0 else 0.0,
            'normal_std': float(np.std(norm_vals)) if len(norm_vals) > 0 else 0.0,
            'normal_median': float(np.median(norm_vals)) if len(norm_vals) > 0 else 0.0,
        }

    summary = {
        'n_vcc': len(vcc_features),
        'n_fc': len(fc_commits),
        'n_matched_normals_unique': len(matched_normal_hashes),
        'n_matched_normals_assignments': len(audit_df),
        'vcc_with_5_matches': sum(1 for c in match_counts if c == 5),
        'vcc_with_less_than_5_matches': sum(1 for c in match_counts if c < 5),
        'max_distance': max_distance,
        'distance_ladder': distance_ladder,
        'max_matches_per_repo_per_vcc': max_matches_per_repo_per_vcc,
        'reuse_histogram': reuse_hist,
        'repo_concentration': repo_concentration,
        'concentration_cap_triggered': cap_triggered,
        'distance_summary': {
            'min': float(np.min(distance_values)) if distance_values else None,
            'median': float(np.median(distance_values)) if distance_values else None,
            'mean': float(np.mean(distance_values)) if distance_values else None,
            'p75': float(np.percentile(distance_values, 75)) if distance_values else None,
            'max': float(np.max(distance_values)) if distance_values else None,
        },
        'feature_distribution': feature_distribution,
        'scaler_params': scaler_params,
    }
    summary_path = os.path.join(output_dir, 'audit_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f'Wrote audit summary to {summary_path}')


# ---------------------------------------------------------------------------
# Phase 5.5 — Post-match ownership extraction
# ---------------------------------------------------------------------------

def extract_matched_ownership(
    assignments: Dict[str, List[Tuple[str, float]]],
    candidates: pd.DataFrame,
    file_info_path: str,
    repo_cache: str,
    output_dir: str,
    workers: int,
) -> None:
    """
    Run ownership extraction only for matched normal commits.

    Ownership is expensive (git blame + git log × 3 windows per file), so we
    defer it to after matching when we know which ~29k commits are actually
    used. This replaces the per-candidate ownership that --skip-ownership skips.

    Reads file paths from file_info_normal_v2.csv (written during enumeration).
    Writes to ownership_normal_v2.csv in output_dir.
    """
    matched_hashes = {nh for matches in assignments.values() for nh, _ in matches}
    if not matched_hashes:
        logging.warning('No matched normals — skipping post-match ownership.')
        return

    if not os.path.exists(file_info_path):
        logging.warning(f'file_info not found at {file_info_path} — skipping ownership.')
        return

    logging.info(f'Post-match ownership: loading file_info for {len(matched_hashes)} matched commits...')
    file_info = pd.read_csv(
        file_info_path,
        usecols=['hash', 'filename', 'new_path', 'old_path'],
    )
    file_info = file_info[file_info['hash'].isin(matched_hashes)]
    if file_info.empty:
        logging.warning('No file_info rows found for matched commits — skipping ownership.')
        return

    # Build lookup: hash → author_date and repo_url from candidates
    cand_lookup = candidates.set_index('hash')[['repo_url', 'author_date']].to_dict('index')

    # Group matched commits by repo_url so we open each repo once
    repo_to_commits: Dict[str, List[str]] = {}
    for h in matched_hashes:
        info = cand_lookup.get(h, {})
        repo_url = info.get('repo_url', '')
        if repo_url:
            repo_to_commits.setdefault(repo_url, []).append(h)

    ownership_path = os.path.join(output_dir, 'ownership_normal_v2.csv')
    write_lock = threading.Lock()
    total_repos = len(repo_to_commits)
    logging.info(f'Post-match ownership: {len(matched_hashes)} commits across {total_repos} repos')

    def _repo_ownership_worker(repo_url: str, commit_hashes: List[str]) -> int:
        repo_path = clone_or_get_repo(repo_url, repo_cache)
        if repo_path is None:
            logging.warning(f'Ownership: could not get repo {repo_url}')
            return 0

        repo_file_info = file_info[file_info['hash'].isin(commit_hashes)]
        rows_written = 0

        for commit_hash in commit_hashes:
            commit_info = cand_lookup.get(commit_hash, {})
            author_date = commit_info.get('author_date', '')
            if not author_date:
                continue

            commit_files = repo_file_info[repo_file_info['hash'] == commit_hash]
            if commit_files.empty:
                continue

            # Reconstruct minimal FileRecord-like objects for the ownership helper
            file_rows = [
                type('_FR', (), {
                    'filename': row.get('filename', ''),
                    'new_path': row.get('new_path', '') or row.get('filename', ''),
                })()
                for row in commit_files.to_dict('records')
            ]

            try:
                ownership_recs = extract_ownership_for_commit(
                    commit_hash, repo_path, file_rows, author_date
                )
            except Exception as e:
                logging.debug(f'Ownership failed for {commit_hash[:8]}: {e}')
                continue

            if ownership_recs:
                append_records_to_csv(ownership_recs, OWNERSHIP_COLUMNS, ownership_path, write_lock)
                rows_written += len(ownership_recs)

        return rows_written

    total_written = 0
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_repo_ownership_worker, repo_url, commit_hashes): repo_url
            for repo_url, commit_hashes in repo_to_commits.items()
        }
        for future in as_completed(futures):
            repo_url = futures[future]
            try:
                n = future.result()
            except Exception as e:
                logging.error(f'Ownership worker failed for {repo_url}: {e}')
                n = 0
            total_written += n
            done += 1
            if done % 10 == 0 or done == total_repos:
                logging.info(f'Ownership: {done}/{total_repos} repos done, {total_written} rows written')

    logging.info(f'Post-match ownership complete: {total_written} rows -> {ownership_path}')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='v2 Normal Commit Sampling (Complexity-Matched) + Option A extraction'
    )
    p.add_argument(
        '--icvulpp-root',
        default=str(Path(__file__).resolve().parents[2] / 'ICVul++'),
        help='Path to ICVul++ project root',
    )
    p.add_argument('--output-dir', default='data_new/sampling_v2')
    p.add_argument(
        '--use-existing-pool', action='store_true',
        help='Use existing v1 pool (data/normal_commits_all/) — fast, no extraction',
    )
    p.add_argument(
        '--skip-enumeration', action='store_true',
        help='Skip enumeration, load existing candidates_raw.csv checkpoint',
    )
    p.add_argument(
        '--stop-after-enumeration', action='store_true',
        help='Fresh mode only: stop after writing candidate/extraction outputs, skip matching',
    )
    p.add_argument(
        '--stop-after-matching', action='store_true',
        help='Stop after KNN matching and writing benchmark_manifest.csv/sampling_audit.csv. '
             'Skips deferred repo cloning and full extraction. Use this on Mac; run extraction separately on VSC.',
    )
    p.add_argument(
        '--smoke-test-repo', default=None,
        help='Limit deferred extraction to a single repo URL (for smoke testing the schema).',
    )
    p.add_argument(
        '--use-manifest', action='store_true',
        help='VSC extraction mode: read benchmark_manifest.csv from --output-dir directly, '
             'skip all enumeration and matching. No candidates_raw.csv or function_info_new.csv needed.',
    )
    p.add_argument(
        '--extract-repos', default=None,
        help='Comma-separated repo URLs to extract (shard filter for --use-manifest). '
             'Only commits from these repos are extracted.',
    )
    p.add_argument(
        '--commit-list', default=None,
        help='Path to a CSV with columns (hash, repo_url) listing exactly which commits to extract. '
             'Fastest VSC mode: skips enumeration, matching, and manifest filtering entirely. '
             'Use the per-shard CSVs in data_new/extraction_shards/shard_XX_commits.csv.',
    )
    p.add_argument(
        '--extract-all-candidates', action='store_true',
        help='Fresh mode: keep old behavior and extract full rows during enumeration instead of after matching',
    )
    p.add_argument('--repo-cache', default='/tmp/icvulpp_repo_cache')
    p.add_argument('--no-cache', action='store_true',
                   help='Clone each repo to a temp dir and delete immediately after processing. '
                        'Ignores --repo-cache entirely. Use on systems with limited disk space (e.g. VSC).')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--reuse-cap', type=int, default=5)
    p.add_argument('--n-matches', type=int, default=5)
    p.add_argument(
        '--exclude-no-vcc-repos', action='store_true',
        help='Remove candidate normals from repos that have zero VCC/FC commits in the dataset.',
    )
    p.add_argument(
        '--per-repo-cap-multiplier', type=int, default=0,
        help='Cap normals per repo at max(N × repo_vcc_count, 20). 0 = disabled.',
    )
    p.add_argument(
        '--max-distance', type=float, default=10.0,
        help='Hard cutoff on raw standardized NN distance; farther candidates are dropped',
    )
    p.add_argument(
        '--distance-ladder', default=','.join(str(int(x)) for x in DEFAULT_DISTANCE_LADDER),
        help='Comma-separated staged distance cutoffs used in order, e.g. 10,15,20,50',
    )
    p.add_argument(
        '--max-matches-per-repo-per-vcc', type=int, default=2,
        help='At most this many matched normals from the same repo for a single VCC',
    )
    p.add_argument(
        '--test-repo',
        help='Limit fresh enumeration to a single repo URL (for testing)',
    )
    p.add_argument(
        '--test-repos',
        help='Comma-separated list of repo URLs to limit fresh enumeration to (for testing)',
    )
    p.add_argument(
        '--max-commits-per-repo', type=int, default=None,
        help='Cap commits per repo in fresh mode (for faster testing)',
    )
    p.add_argument(
        '--skip-ownership', action='store_true',
        help='Skip ownership extraction in fresh mode',
    )
    p.add_argument('--log-level', default='INFO',
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    if args.max_distance is not None and args.max_distance <= 0:
        args.max_distance = None
    if args.distance_ladder:
        args.distance_ladder = [float(x.strip()) for x in args.distance_ladder.split(',') if x.strip()]
    else:
        args.distance_ladder = None

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'sample_matched_normals.log')
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    icvulpp = args.icvulpp_root
    mapping_csv = os.path.join(icvulpp, 'data/processed/cve_fc_vcc_mapping.csv')

    logging.info('=' * 60)
    logging.info('sample_matched_normals.py — v2 Complexity-Matched + Option A')
    logging.info(f'  icvulpp_root      : {icvulpp}')
    logging.info(f'  output_dir        : {args.output_dir}')
    logging.info(f'  use_existing_pool : {args.use_existing_pool}')
    logging.info(f'  test_repo         : {args.test_repo}')
    logging.info(f'  test_repos        : {args.test_repos}')
    logging.info(f'  ownership         : {not args.skip_ownership}')
    logging.info(f'  extract_all_cands : {args.extract_all_candidates}')
    logging.info(f'  stop_after_enum   : {args.stop_after_enumeration}')
    logging.info(f'  seed              : {args.seed}')
    logging.info(f'  max_distance      : {args.max_distance}')
    logging.info(f'  distance_ladder   : {args.distance_ladder}')
    logging.info(f'  repo_diversity_cap: {args.max_matches_per_repo_per_vcc}')
    logging.info('=' * 60)

    # ── commit-list fast path (VSC shard extraction) ────────────────────────────
    # Reads a pre-built CSV (hash, repo_url) directly — no candidates, no matching,
    # no manifest. Each SLURM shard gets its own shard_XX_commits.csv.
    if args.commit_list:
        if not os.path.exists(args.commit_list):
            logging.error(f'--commit-list file not found: {args.commit_list}')
            return
        selected = pd.read_csv(args.commit_list, usecols=['hash', 'repo_url'])
        logging.info(f'Commit-list mode: {len(selected)} commits across {selected["repo_url"].nunique()} repos')
        if selected.empty:
            logging.warning('Commit list is empty — nothing to extract.')
            return
        extract_selected_normal_records(
            selected_candidates=selected,
            repo_cache=args.repo_cache,
            output_dir=args.output_dir,
            workers=args.workers,
            with_ownership=not args.skip_ownership,
            no_cache=args.no_cache,
        )
        logging.info('=' * 60)
        logging.info('DONE (commit-list mode)')
        logging.info(f'  Commits extracted : {len(selected)}')
        logging.info(f'  Outputs in        : {args.output_dir}')
        logging.info('=' * 60)
        return

    # ── use-manifest fast path ──────────────────────────────────────────────────
    # Reads benchmark_manifest.csv directly — no candidates, no matching, no heavy
    # data files needed. Designed for VSC extraction shards.
    if args.use_manifest:
        manifest_path = os.path.join(args.output_dir, 'benchmark_manifest.csv')
        if not os.path.exists(manifest_path):
            logging.error(f'--use-manifest requires benchmark_manifest.csv in --output-dir: {manifest_path}')
            return
        manifest = pd.read_csv(manifest_path)
        normals = manifest[manifest['commit_type'] == 'normal'][['hash', 'repo_url']].copy()
        logging.info(f'Manifest loaded: {len(normals)} matched normal commits across {normals["repo_url"].nunique()} repos')
        if args.extract_repos:
            repo_list = [r.strip() for r in args.extract_repos.split(',') if r.strip()]
            normals = normals[normals['repo_url'].isin(repo_list)].copy()
            logging.info(f'Shard filter: {len(normals)} commits across {normals["repo_url"].nunique()} repos')
        if normals.empty:
            logging.warning('No commits to extract for this shard — exiting.')
            return
        extract_selected_normal_records(
            selected_candidates=normals,
            repo_cache=args.repo_cache,
            output_dir=args.output_dir,
            workers=args.workers,
            with_ownership=not args.skip_ownership,
            no_cache=args.no_cache,
        )
        n_done = len(normals)
        logging.info('=' * 60)
        logging.info('DONE (manifest mode)')
        logging.info(f'  Commits extracted : {n_done}')
        logging.info(f'  Outputs in        : {args.output_dir}')
        logging.info('=' * 60)
        return

    # Phase 1 — Load exclusions and VCC features
    fc_hashes, vcc_hashes, per_repo_vcc = load_exclusions(mapping_csv)
    repo_fc_map = load_repo_fc_hashes(mapping_csv)
    all_excluded = fc_hashes | vcc_hashes
    logging.info(f'Exclusions: {len(fc_hashes)} FC, {len(vcc_hashes)} VCC hashes')

    # vcc_features and fc_commits are only needed for Phase 3+ (matching).
    # Skip loading them when we will exit after enumeration.
    if not args.stop_after_enumeration:
        vcc_features = build_vcc_features(icvulpp)

        commit_full_path = os.path.join(icvulpp, 'data/graph_data/commit_info_full.csv')
        fc_commits = pd.read_csv(commit_full_path, usecols=['hash', 'repo_url', 'commit_type'])
        fc_commits = fc_commits[fc_commits['commit_type'] == 'FC'][['hash', 'repo_url']].copy()
        logging.info(f'FC commits: {len(fc_commits)}')
    else:
        vcc_features = None
        fc_commits = None

    if args.dry_run:
        print(f'VCC count: {len(vcc_features) if vcc_features is not None else "skipped (stop-after-enumeration)"}')
        print(f'FC count: {len(fc_commits) if fc_commits is not None else "skipped (stop-after-enumeration)"}')
        print(f'Mode: {"existing pool" if args.use_existing_pool else "fresh PyDriller"}')
        if args.test_repo:
            print(f'Test repo: {args.test_repo}')
        return

    # Phase 2 — Build candidate pool
    candidates_path = os.path.join(args.output_dir, 'candidates_raw.csv')

    if args.skip_enumeration and os.path.exists(candidates_path):
        logging.info(f'Loading checkpoint from {candidates_path}')
        candidates = pd.read_csv(candidates_path).drop_duplicates('hash').reset_index(drop=True)
        logging.info(f'Loaded {len(candidates)} candidates from checkpoint')

    elif args.use_existing_pool:
        candidates = load_existing_candidate_pool(icvulpp)
        n_before = len(candidates)
        candidates = candidates[~candidates['hash'].isin(all_excluded)].reset_index(drop=True)
        logging.info(f'Removed {n_before - len(candidates)} VCC/FC hashes, {len(candidates)} remain')
        candidates.to_csv(candidates_path, index=False)

    else:
        # Fresh PyDriller enumeration with Option A extraction
        active_repos = sorted(
            r for r in per_repo_vcc.loc[per_repo_vcc['vcc_count'] > 0, 'repo_url']
            if not is_skipped(r)
        )
        active_repos = order_active_repos(active_repos, per_repo_vcc, args.repo_cache)
        if args.test_repos:
            active_repos = [r.strip() for r in args.test_repos.split(',') if r.strip()]
            logging.info(f'TEST MODE: limiting to {len(active_repos)} repos from --test-repos')
        elif args.test_repo:
            active_repos = [args.test_repo]
            logging.info(f'TEST MODE: limiting to {args.test_repo}')

        candidates = enumerate_fresh_candidates(
            repos=active_repos,
            all_excluded=all_excluded,
            repo_fc_map=repo_fc_map,
            repo_cache=args.repo_cache,
            output_dir=args.output_dir,
            workers=args.workers,
            with_ownership=not args.skip_ownership,
            full_extract=args.extract_all_candidates,
            max_commits_per_repo=args.max_commits_per_repo,
            no_cache=args.no_cache,
        )
        n_before = len(candidates)
        candidates = candidates[~candidates['hash'].isin(all_excluded)].reset_index(drop=True)
        logging.info(f'Post-exclusion: {n_before - len(candidates)} removed, {len(candidates)} remain')
        candidates.to_csv(candidates_path, index=False)
        if args.stop_after_enumeration:
            logging.info('Stopping after enumeration as requested.')
            return

    logging.info(f'Candidate pool: {len(candidates)} commits')
    if candidates.empty:
        logging.error('No candidates — cannot run matching. Exiting.')
        return

    # Phase 2.5 — Optional candidate filtering
    if args.exclude_no_vcc_repos or args.per_repo_cap_multiplier > 0:
        vcc_repos = set(vcc_features['repo_url'].unique())
        vcc_repo_counts = vcc_features['repo_url'].value_counts().to_dict()

        if args.exclude_no_vcc_repos:
            before = len(candidates)
            dropped_repos = set(candidates['repo_url'].unique()) - vcc_repos
            candidates = candidates[candidates['repo_url'].isin(vcc_repos)].reset_index(drop=True)
            logging.info(
                f'[filter] Excluded {len(dropped_repos)} zero-VCC repos: '
                f'{before - len(candidates)} commits removed, {len(candidates)} remain'
            )

        if args.per_repo_cap_multiplier > 0:
            mult = args.per_repo_cap_multiplier
            def _per_repo_cap(df: pd.DataFrame) -> pd.DataFrame:
                repo = df['repo_url'].iloc[0]
                cap = max(mult * vcc_repo_counts.get(repo, 0), 20)
                return df.sample(min(len(df), cap), random_state=args.seed)
            before = len(candidates)
            candidates = candidates.groupby('repo_url', group_keys=False).apply(_per_repo_cap).reset_index(drop=True)
            logging.info(
                f'[filter] Per-repo cap ({mult}×VCC, min 20): {before - len(candidates)} removed, '
                f'{len(candidates)} remain'
            )

    # Phase 3 — Standardize
    for feat in MATCHING_FEATURES:
        candidates[feat] = candidates[feat].fillna(0)
        vcc_features[feat] = vcc_features[feat].fillna(0)

    scaler_params = compute_scaler_params(candidates)
    scaler_path = os.path.join(args.output_dir, 'matching_scaler_params.json')
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)

    X_candidates = standardize(candidates, scaler_params)
    X_vccs = standardize(vcc_features, scaler_params)
    candidate_hashes_list = candidates['hash'].tolist()
    candidate_repo_list = candidates['repo_url'].tolist()
    vcc_hashes_list = vcc_features['hash'].tolist()

    # Phase 4 — Matching
    logging.info(f'5-NN matching: {len(vcc_hashes_list)} VCCs x {len(candidate_hashes_list)} candidates')
    assignments, reuse_count = run_matching(
        X_vccs=X_vccs, vcc_hashes=vcc_hashes_list,
        X_candidates=X_candidates, candidate_hashes=candidate_hashes_list,
        candidate_repo_urls=candidate_repo_list,
        reuse_cap=args.reuse_cap, n_matches=args.n_matches,
        max_distance=args.max_distance,
        distance_ladder=args.distance_ladder,
        max_matches_per_repo_per_vcc=args.max_matches_per_repo_per_vcc,
    )

    # Phase 5 — Repo cap
    assignments, reuse_count, cap_triggered = check_and_apply_repo_cap(
        assignments=assignments, vcc_features=vcc_features, candidates=candidates,
        X_vccs=X_vccs, vcc_hashes_list=vcc_hashes_list,
        X_candidates=X_candidates, candidate_hashes_list=candidate_hashes_list,
        max_distance=args.max_distance,
        distance_ladder=args.distance_ladder,
        max_matches_per_repo_per_vcc=args.max_matches_per_repo_per_vcc,
    )

    # Phase 5.5 — Post-match ownership for eager-extraction mode only.
    # Deferred mode extracts ownership together with the selected matched commits.
    if (not args.skip_ownership) and args.extract_all_candidates:
        extract_matched_ownership(
            assignments=assignments,
            candidates=candidates,
            file_info_path=os.path.join(args.output_dir, 'file_info_normal_v2.csv'),
            repo_cache=args.repo_cache,
            output_dir=args.output_dir,
            workers=args.workers,
        )

    # Phase 6 — Outputs
    write_outputs(
        assignments=assignments, reuse_count=reuse_count,
        vcc_features=vcc_features, candidates=candidates,
        fc_commits=fc_commits, scaler_params=scaler_params,
        output_dir=args.output_dir, cap_triggered=cap_triggered,
        max_distance=args.max_distance,
        distance_ladder=args.distance_ladder,
        max_matches_per_repo_per_vcc=args.max_matches_per_repo_per_vcc,
    )

    if args.stop_after_matching:
        n_matched = len({nh for m in assignments.values() for nh, _ in m})
        logging.info(f'Stopping after matching as requested. {n_matched} normals matched.')
        logging.info(f'Outputs written to {args.output_dir}')
        return

    if (not args.use_existing_pool) and (not args.extract_all_candidates):
        matched_normal_hashes = sorted({nh for matches in assignments.values() for nh, _ in matches})
        selected_candidates = candidates[candidates['hash'].isin(matched_normal_hashes)][['hash', 'repo_url']].copy()
        if args.smoke_test_repo:
            selected_candidates = selected_candidates[selected_candidates['repo_url'] == args.smoke_test_repo].copy()
            logging.info(f'SMOKE TEST: limiting extraction to {len(selected_candidates)} commits from {args.smoke_test_repo}')
        extract_selected_normal_records(
            selected_candidates=selected_candidates,
            repo_cache=args.repo_cache,
            output_dir=args.output_dir,
            workers=args.workers,
            with_ownership=not args.skip_ownership,
            no_cache=args.no_cache,
        )

    n_matched = len({nh for m in assignments.values() for nh, _ in m})
    logging.info('=' * 60)
    logging.info('DONE')
    logging.info(f'  VCC                    : {len(vcc_features)}')
    logging.info(f'  FC                     : {len(fc_commits)}')
    logging.info(f'  Matched normals unique : {n_matched}')
    logging.info(f'  Total benchmark size   : {len(vcc_features) + len(fc_commits) + n_matched}')
    logging.info(f'  Repo cap triggered     : {cap_triggered}')
    logging.info(f'  Outputs in             : {args.output_dir}')
    logging.info('=' * 60)


if __name__ == '__main__':
    main()
