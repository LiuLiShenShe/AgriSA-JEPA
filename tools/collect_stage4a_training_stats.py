#!/usr/bin/env python
"""Collect the Stage 3B / 4A training cost comparison into one machine-readable file.

Reads the four run folders (the CSV logs and ``training_verification.yaml``) and
produces ``reports/stage4a_training_benchmark.json``:

* real optimizer updates and micro-steps,
* wall-clock time per arm and per update,
* measured step time (the compute-only timer inside the training loop),
* images per second at the effective batch size,
* peak GPU memory,
* the checkpoint inventory with SHA-256,
* the mask-generation overhead measured separately.

Usage:
    ./.venv/bin/python -m tools.collect_stage4a_training_stats \
        --runs-root /home/test/agrisa_runs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ARMS = {
    'B': 'stage3b_random_2000',
    'C1': 'stage4a_vegetation_2000',
    'C2': 'stage4a_boundary_2000',
    'C3': 'stage4a_shuffled_2000',
}

LOG_FILES = {
    'B': '/data/agrisa_jepa_runs/stream_gpu0.log',
    'C1': '/data/agrisa_jepa_runs/stream_gpu0.log',
    'C2': '/data/agrisa_jepa_runs/stream_gpu1.log',
    'C3': '/data/agrisa_jepa_runs/stream_gpu1.log',
}

ARM_RE = re.compile(r'^=== ARM (\S+) START (.+) ===$')
ARM_END_RE = re.compile(r'^=== ARM (\S+) EXIT=(\d+) (.+) ===$')


#: ``(path, size, mtime) -> sha256`` cache.  The checkpoints are 2.5 GB / 10 GB
#: each, and re-hashing them on every collector run costs minutes for an answer
#: that cannot have changed.
_SHA_CACHE_PATH = Path('.cache/checkpoint_sha256.json')


def _sha_cache() -> Dict[str, str]:
    if _SHA_CACHE_PATH.is_file():
        return json.loads(_SHA_CACHE_PATH.read_text(encoding='utf-8'))
    return {}


def _sha_cache_put(cache: Dict[str, str]) -> None:
    _SHA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SHA_CACHE_PATH.write_text(json.dumps(cache, indent=1) + '\n', encoding='utf-8')


def sha256_file(path: Path, cache: Optional[Dict[str, str]] = None) -> str:
    stat = path.stat()
    key = f'{path}|{stat.st_size}|{int(stat.st_mtime)}'
    if cache is not None and key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache is not None:
        cache[key] = value
        _sha_cache_put(cache)      # incremental: a later timeout keeps progress
    return value


def arm_windows(log_path: Path) -> Dict[str, Dict[str, str]]:
    """The start / exit timestamps of every arm inside a stream log."""
    if not log_path.is_file():
        return {}
    windows: Dict[str, Dict[str, str]] = {}
    current = None
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        start = ARM_RE.match(line.strip())
        end = ARM_END_RE.match(line.strip())
        if start:
            current = start.group(1)
            windows[current] = {'start': start.group(2)}
        elif end and current is not None:
            windows[current]['exit'] = end.group(3)
            windows[current]['exit_code'] = end.group(2)
            current = None
    return windows


def parse_iso(text: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(text).timestamp()


def read_csv(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    times = [float(r['time (ms)']) for r in rows if r.get('time (ms)')]
    losses = [float(r['loss']) for r in rows if r.get('loss')]
    mask_a = [float(r['mask-A']) for r in rows if r.get('mask-A')]
    mask_b = [float(r['mask-B']) for r in rows if r.get('mask-B')]
    epochs = sorted({int(r['epoch']) for r in rows})
    per_epoch = {}
    for r in rows:
        per_epoch[int(r['epoch'])] = per_epoch.get(int(r['epoch']), 0) + 1
    # every accumulation window contributes exactly one update, and the final
    # window of an epoch is short; `ceil(count / grad_accum)` per epoch is exact
    grad_accum = 8
    inferred_updates = sum(
        -(-count // grad_accum) for count in per_epoch.values())
    return {
        'micro_steps': len(rows),
        'inferred_optimizer_updates': inferred_updates,
        'micro_steps_per_epoch': per_epoch.get(epochs[-1]) if epochs else None,
        'epochs': epochs,
        'step_time_ms_mean': statistics.mean(times) if times else None,
        'step_time_ms_median': statistics.median(times) if times else None,
        'step_time_ms_p95': (sorted(times)[int(0.95 * (len(times) - 1))]
                             if times else None),
        'loss_first': losses[0] if losses else None,
        'loss_last': losses[-1] if losses else None,
        'context_patches_mean': statistics.mean(mask_a) if mask_a else None,
        'target_patches_mean': statistics.mean(mask_b) if mask_b else None,
    }


def read_verification(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    return payload if isinstance(payload, dict) else {}


def checkpoint_inventory(folder: Path, cache: Optional[Dict[str, str]] = None) -> List[Dict[str, object]]:
    items = []
    for path in sorted(folder.glob('*')):
        if path.is_file() and path.name.endswith(('.pth', '.pth.tar')):
            items.append({
                'file': path.name,
                'bytes': path.stat().st_size,
                'gib': round(path.stat().st_size / 2 ** 30, 2),
                'sha256': (sha256_file(path, cache)
                           if path.stat().st_size < 40 * 2 ** 30 else None),
            })
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', default='/home/test/agrisa_runs')
    parser.add_argument('--out', default='reports/stage4a_training_benchmark.json')
    args = parser.parse_args()

    root = Path(args.runs_root)
    windows: Dict[str, Dict[str, str]] = {}
    for log in set(LOG_FILES.values()):
        windows.update(arm_windows(Path(log)))

    sha_cache = _sha_cache()
    arms: Dict[str, object] = {}
    for arm, folder_name in ARMS.items():
        folder = root / folder_name
        csv_stats = read_csv(folder / 'jepa_r0.csv')
        verification = read_verification(folder / 'training_verification.yaml')
        window = windows.get(arm, {})
        wall = None
        if 'start' in window and 'exit' in window:
            wall = round(parse_iso(window['exit']) - parse_iso(window['start']), 1)

        micro = int(csv_stats.get('micro_steps') or 0)
        # the trainer writes ``optimizer_steps_this_run``; older Stage 3A runs
        # used the shorter key, so accept both and only then fall back to the
        # micro-step count.
        # ``micro // 8`` is wrong at an epoch boundary: the last accumulation
        # window of each epoch is short (785 micro-steps -> 99 updates, not 98),
        # so the recorded count is authoritative and the CSV only fills gaps.
        updates = int(verification.get('optimizer_steps_this_run')
                      or verification.get('optimizer_steps')
                      or csv_stats.get('optimizer_updates')
                      or micro // 8)
        diagnostics = verification.get('mask_diagnostics') or {}
        prior_seconds = diagnostics.get('prior_seconds')
        selection_seconds = diagnostics.get('selection_seconds')
        mask_seconds = None
        if isinstance(prior_seconds, (int, float)) and isinstance(selection_seconds, (int, float)):
            mask_seconds = round(prior_seconds + selection_seconds, 1)
        entry: Dict[str, object] = {
            'run_dir': str(folder),
            'optimizer_updates': updates,
            'micro_steps': micro,
            'wall_clock_seconds': wall,
            'seconds_per_optimizer_update': (round(wall / updates, 3)
                                             if wall and updates else None),
            'images_per_second': (round(updates * 128 / wall, 1)
                                  if wall and updates else None),
            'peak_memory_mib': verification.get('peak_memory_mib'),
            'step_time_ms_mean': verification.get('step_time_ms_mean'),
            'mask_mode': verification.get('mask_mode'),
            'mask_diagnostics': verification.get('mask_diagnostics'),
            'mask_overhead': {
                'prior_seconds': prior_seconds,
                'selection_seconds': selection_seconds,
                'total_seconds': mask_seconds,
                'fraction_of_wall_clock': (round(mask_seconds / wall, 4)
                                           if mask_seconds and wall else None),
                'milliseconds_per_micro_step': (round(1000.0 * mask_seconds / micro, 2)
                                                if mask_seconds and micro else None),
                'uniform_fallbacks': diagnostics.get('uniform_fallbacks'),
                'target_blocks': diagnostics.get('target_blocks'),
                'note': ('measured inside the collator in the main process; '
                         'with num_workers=0 it is serial CPU time next to the '
                         'GPU step, not additional GPU memory.'),
            },
            'saved_step_snapshots': verification.get('saved_step_snapshots'),
            'manifest_sha256': verification.get('manifest_sha256'),
            'pretrained_init_sha256': (verification.get('pretrained_init') or {}).get('sha256'),
            'git_commit': verification.get('git_commit'),
            'csv': csv_stats,
            'checkpoints': checkpoint_inventory(folder, sha_cache),
        }
        arms[arm] = entry

    payload = {
        'runs_root': str(root),
        'budget_optimizer_updates': 2000,
        'effective_batch': 128,
        'note': ('step_time_ms is the compute-only timer inside the training loop; '
                 'wall clock additionally contains the per-epoch checkpoint write '
                 'and the data loading.'),
        'arms': arms,
    }
    _sha_cache_put(sha_cache)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({arm: {k: v for k, v in entry.items()
                            if k in ('optimizer_updates', 'wall_clock_seconds',
                                     'seconds_per_optimizer_update',
                                     'images_per_second', 'peak_memory_mib',
                                     'step_time_ms_mean')}
                      for arm, entry in arms.items()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
