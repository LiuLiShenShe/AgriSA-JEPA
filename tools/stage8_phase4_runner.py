#!/usr/bin/env python
"""Stage 8 Phase 4: execute the frozen experiment registry, serial, GPU 1.

Reads ``reports/stage8_experiment_registry.csv`` (frozen in Phase 2) and runs
each executable row (status FROZEN or FROZEN_EXPLORATORY) in order.  Rules:

* never overwrite: a row whose ``output_dir`` already contains ``result.json``
  (gwhd: also ``results.json``) is skipped unless ``--force``;
* stopped / crashed runs resume: rows already producing their final artefact
  are skipped, the rest re-run from the top of the command;
* GPU 1 serial: each command runs with ``CUDA_VISIBLE_DEVICES=1`` and the next
  row starts only when the previous has exited;
* every command is echoed to ``stage8_phase4_wallclock.csv`` with start/end
  wall-clock and exit code, appended per run so nothing is lost on a crash;
* REUSE_STAGE7B rows are never executed (their results already exist).

A row whose command fails is recorded with its exit code and execution stops
(no silent continuation past a broken run).

Usage
-----
    ./.venv-uv/bin/python tools/stage8_phase4_runner.py \
        [--tasks gwhd_det,cropandweed_sem] [--backbones A_p1,B_p1] \
        [--roles primary,exploratory] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'reports'
REGISTRY = R / 'stage8_experiment_registry.csv'
WALLCLOCK = R / 'stage8_phase4_wallclock.csv'


def load_registry(path: Path) -> list:
    with path.open('r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def has_result(row: dict) -> bool:
    out = Path(row['output_dir'])
    for marker in ('result.json', 'results.json'):
        if (out / marker).is_file():
            return True
    return False


def prune_feature_cache(row: dict) -> None:
    """Free a fully-consumed feature cache once its last row completes.

    CropAndWeed caches are ~102 GB per backbone on the same NVMe that must
    hold the next backbone's extraction; without pruning the seven backbones
    need ~720 GB.  The prune refuses to touch any backbone with a pending
    row, so this only ever reclaims space that can no longer affect a result.
    """
    try:
        from tools.stage8_prune_feature_cache import prune_for_backbone
        prune_for_backbone(row['backbone'])
    except SystemExit as exc:
        print(f'!!! prune safety refusal for {row["backbone"]}: {exc}',
              flush=True)
        raise


def run_row(row: dict, force: bool) -> int:
    out = Path(row['output_dir'])
    if has_result(row) and not force:
        print(f'  SKIP (already has result): {row["output_dir"]}', flush=True)
        return -1
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = '1'
    print(f'\n===== {row["task_id"]} | {row["backbone"]} | '
          f'seed {row["downstream_seed"]} | {row["role"]} =====', flush=True)
    print(f'  CMD: {row["command"]}', flush=True)
    t0 = time.time()
    proc = subprocess.run(row['command'], shell=True, cwd=str(ROOT),
                          env=env)
    elapsed = round((time.time() - t0) / 60, 2)
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with WALLCLOCK.open('a', newline='', encoding='utf-8') as handle:
        w = csv.DictWriter(handle, fieldnames=list(row.keys()) + [
            'started_at', 'finished_at', 'wall_clock_minutes', 'exit_code'])
        w.writerow({**row, 'started_at': now, 'finished_at': now,
                    'wall_clock_minutes': elapsed, 'exit_code': proc.returncode})
    print(f'  exit={proc.returncode} wall={elapsed}min', flush=True)
    if proc.returncode == 0 and row['task_id'] == 'cropandweed_sem':
        prune_feature_cache(row)
    return proc.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', default=None,
                        help='comma-separated task_id filter')
    parser.add_argument('--backbones', default=None,
                        help='comma-separated backbone keys')
    parser.add_argument('--roles', default='frozen',
                        help='comma-separated: frozen,exploratory '
                             '(compared case-insensitively; FROZEN* statuses)')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    tasks = {s.lower() for s in (args.tasks or '').split(',') if s} if args.tasks else None
    backbones = {s for s in (args.backbones or '').split(',') if s} if args.backbones else None
    roles = {s.lower() for s in (args.roles or '').split(',') if s}

    rows = load_registry(REGISTRY)
    if not WALLCLOCK.is_file():
        with WALLCLOCK.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) + [
                'started_at', 'finished_at', 'wall_clock_minutes', 'exit_code'])
            writer.writeheader()

    executed = skipped = 0
    for row in rows:
        if row['status'] == 'REUSE_STAGE7B':
            print(f'  REUSE (Stage 7B): {row["task_id"]}/{row["backbone"]} '
                  f'seed {row["downstream_seed"]}', flush=True)
            continue
        if tasks and row['task_id'] not in tasks:
            continue
        if backbones and row['backbone'] not in backbones:
            continue
        status = row['status'].lower()
        if status not in roles:
            continue
        rc = run_row(row, args.force)
        if rc == -1:
            skipped += 1
            continue
        executed += 1
        if rc != 0:
            print(f'\n!!! RUN FAILED (exit {rc}): {row["output_dir"]} '
                  f'-- stopping', flush=True)
            return 2
    print(f'\nDONE: executed={executed} resumed_skipped={skipped}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())