#!/usr/bin/env python
"""Summarise a Stage 6 pretraining run from its own artefacts.

Everything reported here is read from files the run wrote: the console log, the
CSV training log, ``training_verification.yaml``, the source-exposure ledger and
the checkpoints.  Nothing is estimated unless the field says so.

Usage:
    ./.venv-uv/bin/python -m tools.summarize_stage6_pretraining \
        --arm B1 --run-dir runs/stage6_b1_natural_2000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = re.compile(
    r'PROGRESS step=(\d+)/(\d+) loss=([0-9.eE+-]+|nan|inf) lr=([0-9.eE+-]+) '
    r'wd=([0-9.eE+-]+) gpu=([0-9.]+)GB ms/step=([0-9]+|nan) '
    r'elapsed=([0-9.]+)min eta=([0-9.]+|inf)min')
MICRO = re.compile(r'^\[(\d+),\s*(\d+)\] loss: ([0-9.]+) ')
SKIP_REASONS = re.compile(r'^SKIPPED (.+?) -- (.*)$')


def parse_console(path: Path) -> Dict[str, object]:
    steps: List[Dict[str, object]] = []
    micro_rows = 0
    skips: List[str] = []
    errors: List[str] = []
    if not path.is_file():
        return {'steps': steps, 'micro_rows': micro_rows, 'skips': skips,
                'errors': [f'missing console log: {path}']}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        match = PROGRESS.search(line)
        if match:
            steps.append({
                'step': int(match.group(1)),
                'total': int(match.group(2)),
                'loss': float(match.group(3)),
                'lr': float(match.group(4)),
                'wd': float(match.group(5)),
                'gpu_gb': float(match.group(6)),
                'ms_per_step': (None if match.group(7) == 'nan' else float(match.group(7))),
                'elapsed_min': float(match.group(8)),
                'eta_min': (None if match.group(9) == 'inf' else float(match.group(9))),
            })
        elif MICRO.match(line):
            micro_rows += 1
        elif line.startswith('SKIPPED'):
            skips.append(line.strip())
        elif 'Traceback' in line or line.startswith('ERROR') or 'Error' in line:
            errors.append(line.strip())
    return {'steps': steps, 'micro_rows': micro_rows, 'skips': skips, 'errors': errors}


def parse_csv_log(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {'rows': 0, 'first_itr': None, 'last_itr': None, 'loss_min': None,
                'loss_max': None, 'loss_first': None, 'loss_last': None,
                'nan_rows': 0, 'time_ms_mean': None}
    rows: List[Dict[str, str]] = []
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        return {'rows': 0, 'first_itr': None, 'last_itr': None, 'loss_min': None,
                'loss_max': None, 'loss_first': None, 'loss_last': None,
                'nan_rows': 0, 'time_ms_mean': None}
    losses = [float(row['loss']) for row in rows]
    times = [float(row['time (ms)']) for row in rows]
    nonfinite = sum(1 for value in losses if value != value or value in (float('inf'), float('-inf')))
    return {
        'rows': len(rows),
        'first_itr': int(rows[0]['itr']),
        'last_itr': int(rows[-1]['itr']),
        'loss_min': min(losses),
        'loss_max': max(losses),
        'loss_first': losses[0],
        'loss_last': losses[-1],
        'nan_rows': nonfinite,
        'time_ms_mean': sum(times) / len(times),
    }


def load_verification(run_dir: Path) -> Dict[str, object]:
    path = run_dir / 'training_verification.yaml'
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def load_ledger(run_dir: Path) -> Dict[str, object]:
    summary = run_dir / 'source_exposure_ledger.summary.json'
    ledger = run_dir / 'source_exposure_ledger.csv'
    payload: Dict[str, object] = {}
    if summary.is_file():
        payload = json.loads(summary.read_text(encoding='utf-8'))
    if ledger.is_file():
        with ledger.open(newline='', encoding='utf-8') as handle:
            payload['ledger_rows'] = sum(1 for _ in handle) - 1
    return payload


def load_checkpoints(run_dir: Path, tag: str = 'jepa') -> Dict[str, object]:
    result: Dict[str, object] = {}
    snapshots = sorted(run_dir.glob(f'{tag}-step*.pth.tar'),
                       key=lambda path: int(re.search(r'step(\d+)', path.name).group(1)))
    result['step_snapshots'] = [
        {'step': int(re.search(r'step(\d+)', path.name).group(1)),
         'name': path.name, 'bytes': path.stat().st_size}
        for path in snapshots]
    latest = run_dir / f'{tag}-latest.pth.tar'
    if latest.is_file():
        result['latest'] = {'name': latest.name, 'bytes': latest.stat().st_size}
    encoders = sorted(run_dir.glob(f'{tag}-encoder-step*.pth'),
                      key=lambda path: int(re.search(r'step(\d+)', path.name).group(1)))
    result['encoder_snapshots'] = [
        {'step': int(re.search(r'step(\d+)', path.name).group(1)),
         'name': path.name, 'bytes': path.stat().st_size}
        for path in encoders]
    return result


def summarise(arm: str, run_dir: Path) -> Dict[str, object]:
    console = parse_console(run_dir / 'console.log')
    steps = console['steps']
    verification = load_verification(run_dir)
    csv_log = parse_csv_log(run_dir / f'{verification.get('write_tag', 'jepa')}_r0.csv'
                            if verification else run_dir / 'jepa_r0.csv')

    step_times = [s['ms_per_step'] for s in steps if s.get('ms_per_step') is not None]
    summary: Dict[str, object] = {
        'arm': arm,
        'run_dir': str(run_dir),
        'artefacts_present': {
            'console_log': (run_dir / 'console.log').is_file(),
            'csv_log': csv_log['rows'] > 0,
            'verification': bool(verification),
            'source_ledger': bool(load_ledger(run_dir)),
            'checkpoints': bool(load_checkpoints(run_dir).get('step_snapshots')),
        },
        'console': {
            'progress_lines': len(steps),
            'micro_log_lines': console['micro_rows'],
            'last_step': steps[-1] if steps else None,
            'errors': console['errors'][:10],
            'error_count': len(console['errors']),
            'skips': console['skips'],
        },
        'csv_log': csv_log,
        'verification': verification,
        'ledger': load_ledger(run_dir),
        'checkpoints': load_checkpoints(run_dir),
        'measured': {
            # -- median of the PROGRESS lines: robust to the one-off resume step
            'ms_per_update_median': (sorted(step_times)[len(step_times) // 2]
                                     if step_times else None),
            'ms_per_update_last': step_times[-1] if step_times else None,
            'updates_per_second_median': (1000.0 / sorted(step_times)[len(step_times) // 2]
                                          if step_times else None),
            'elapsed_min': steps[-1]['elapsed_min'] if steps else None,
            'peak_gpu_gb': max((s['gpu_gb'] for s in steps), default=None),
            'samples_per_second_estimate': (
                128.0 / (sorted(step_times)[len(step_times) // 2] / 1000.0)
                if step_times else None),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    payload = summarise(args.arm, run_dir)
    out = args.out or (ROOT / 'reports' / f'stage6_{args.arm.lower()}_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
