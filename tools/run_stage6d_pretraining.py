#!/usr/bin/env python
"""Run the four Stage 6D pretraining replicates, sequentially and guardedly.

Why a runner instead of four hand-typed commands:

* Stage 6's runs recorded the *base* commit (7ec8a6b) because the Stage 6 code
  was still uncommitted while they trained.  Stage 6D must not repeat that.
  This runner refuses a dirty tracked worktree, records HEAD, the full
  ``git status`` and the config SHA before launch, and writes the same facts
  into a per-run manifest afterwards.
* The protocol requires that a failed run stops the sequence, never silently
  fall through to the next run.
* GPU1 is the only permitted device.  The runner checks free memory on the
  requested physical GPU and refuses to start if it is occupied; it never kills
  another process and never switches to GPU0 on its own.

Usage::

    ./.venv/bin/python -m tools.run_stage6d_pretraining --dry-run
    ./.venv/bin/python -m tools.run_stage6d_pretraining --execute --device 1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from tools.stage6d_build_configs import (
    BASE_CONFIGS,
    ROOT,
    build_plan,
    config_path_for,
    sha256_file,
)

EXPECTED_MANIFEST_SHA = '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2'
EXPECTED_PRETRAINED_SHA = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'
EXPECTED_OPTIMIZER_STEPS = 2000
DEFAULT_MIN_FREE_MIB = 30000


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def git(args: Sequence[str]) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def git_status_porcelain() -> List[str]:
    return [line for line in git(['status', '--porcelain']).splitlines() if line]


def tracked_changes(lines: Sequence[str]) -> List[str]:
    """``??`` lines are untracked user files (e.g. CLAUDE.md); anything else is
    a tracked modification and blocks a traceable run."""
    return [line for line in lines if not line.startswith('??')]


def gpu_report(physical_index: int) -> Dict[str, object]:
    query = ('index,uuid,name,memory.used,memory.total,utilization.gpu')
    out = subprocess.check_output([
        'nvidia-smi', f'--id={physical_index}',
        f'--query-gpu={query}', '--format=csv,noheader,nounits',
    ], text=True).strip()
    parts = [part.strip() for part in out.split(',')]
    if len(parts) != 6:
        raise RuntimeError(f'unexpected nvidia-smi output: {out!r}')
    used = int(parts[3])
    total = int(parts[4])
    return {
        'physical_index': int(parts[0]),
        'uuid': parts[1],
        'name': parts[2],
        'memory_used_mib': used,
        'memory_total_mib': total,
        'memory_free_mib': total - used,
        'utilization_percent': int(parts[5]),
    }


def preflight(physical_index: int, min_free_mib: int,
              max_utilization_percent: int, allow_dirty: bool) -> Dict[str, object]:
    status_lines = git_status_porcelain()
    dirty = tracked_changes(status_lines)
    report: Dict[str, object] = {
        'git_head': git(['rev-parse', 'HEAD']),
        'git_status_porcelain': status_lines,
        'tracked_changes': dirty,
        'git_diff_check_ok': subprocess.run(
            ['git', 'diff', '--check'], cwd=ROOT).returncode == 0,
        'physical_gpu': gpu_report(physical_index),
        'min_free_mib': min_free_mib,
        'max_utilization_percent': max_utilization_percent,
        'problems': [],
    }
    if dirty and not allow_dirty:
        report['problems'].append(
            'tracked worktree is dirty; commit first so the run records a real '
            'code commit (Stage 6 could not do this)')
    if not report['git_diff_check_ok']:
        report['problems'].append('git diff --check failed')
    free = report['physical_gpu']['memory_free_mib']
    if free < min_free_mib:
        report['problems'].append(
            f'physical GPU{physical_index} has {free} MiB free < {min_free_mib} MiB; '
            'refusing to start (do not kill the other process)')
    util = report['physical_gpu']['utilization_percent']
    if util > max_utilization_percent:
        report['problems'].append(
            f'physical GPU{physical_index} utilization is {util}% '
            f'> {max_utilization_percent}%; another process is using it')
    return report


def verification_problems(run_dir: Path, seed: int, arm: str) -> List[str]:
    problems: List[str] = []
    verification_path = run_dir / 'training_verification.yaml'
    if not verification_path.is_file():
        return [f'missing {verification_path}']
    with verification_path.open('r', encoding='utf-8') as handle:
        verification = yaml.safe_load(handle)
    if verification.get('optimizer_steps_total') != EXPECTED_OPTIMIZER_STEPS:
        problems.append(f'optimizer_steps_total {verification.get("optimizer_steps_total")} '
                        f'!= {EXPECTED_OPTIMIZER_STEPS}')
    if verification.get('loss_finite') is not True:
        problems.append('loss_finite is not true')
    if verification.get('manifest_sha256') != EXPECTED_MANIFEST_SHA:
        problems.append(f'manifest_sha256 {verification.get("manifest_sha256")} '
                        f'!= {EXPECTED_MANIFEST_SHA}')
    pretrained = verification.get('pretrained_init') or {}
    if pretrained.get('sha256') != EXPECTED_PRETRAINED_SHA:
        problems.append(f'pretrained_init.sha256 {pretrained.get("sha256")} '
                        f'!= {EXPECTED_PRETRAINED_SHA}')
    if not (run_dir / 'jepa-encoder-step2000.pth').is_file():
        problems.append('missing jepa-encoder-step2000.pth')
    if not (run_dir / 'source_exposure_ledger.summary.json').is_file():
        problems.append('missing source_exposure_ledger.summary.json')
    params_path = run_dir / 'params-ijepa.yaml'
    if not params_path.is_file():
        problems.append('missing params-ijepa.yaml')
    else:
        with params_path.open('r', encoding='utf-8') as handle:
            params = yaml.safe_load(handle)
        if params['data'].get('seed') != seed:
            problems.append(f'data.seed {params["data"].get("seed")} != {seed}')
        if params['data'].get('augmentation_seed') != seed:
            problems.append(f'data.augmentation_seed {params["data"].get("augmentation_seed")} '
                            f'!= {seed}')
        if arm == 'B2' and params['data']['sampling'].get('seed') != seed:
            problems.append(f'data.sampling.seed {params["data"]["sampling"].get("seed")} '
                            f'!= {seed}')
    git_commit = verification.get('git_commit')
    if git_commit != git(['rev-parse', 'HEAD']):
        problems.append(f'verification git_commit {git_commit} != current HEAD '
                        f'{git(["rev-parse", "HEAD"])}')
    return problems


def run_manifest(run: Dict[str, object], run_dir: Path,
                 launch: Dict[str, object]) -> Dict[str, object]:
    outputs = []
    for pattern in ('jepa-step*.pth.tar', 'jepa-encoder-step*.pth',
                    'jepa-latest.pth.tar', 'jepa_r0.csv',
                    'training_verification.yaml', 'params-ijepa.yaml',
                    'source_exposure_ledger.csv',
                    'source_exposure_ledger.summary.json'):
        for path in sorted(run_dir.glob(pattern)):
            outputs.append({'path': str(path), 'bytes': path.stat().st_size,
                            'sha256': sha256_file(path)})
    return {
        'schema_version': 'stage6d_run_manifest/v1',
        'arm': run['arm'], 'run_id': run['run_id'], 'pair_id': run['pair_id'],
        'seed': run['seed'], 'launch': launch, 'outputs': outputs,
    }


def dry_run(plan: Dict[str, object], physical_index: int) -> int:
    print(json.dumps({
        'mode': 'dry-run',
        'physical_gpu': gpu_report(physical_index),
        'commands': [
            {
                'run_id': run['run_id'],
                'arm': run['arm'], 'seed': run['seed'],
                'config': str(config_path_for(str(run['arm']), int(run['seed'])).relative_to(ROOT)),
                'output_dir': run['output_dir'],
                'command': [sys.executable, 'main.py', '--fname',
                            str(config_path_for(str(run['arm']), int(run['seed'])).relative_to(ROOT)),
                            '--devices', f'cuda:{physical_index}'],
            }
            for run in plan['runs']
        ],
    }, indent=2, ensure_ascii=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='actually start training (default: dry-run)')
    parser.add_argument('--dry-run', action='store_true',
                        help='explicitly print the planned commands and exit')
    parser.add_argument('--device', type=int, default=1,
                        help='physical GPU index (project rule: 1)')
    parser.add_argument('--min-free-mib', type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument('--max-utilization-percent', type=int, default=5,
                        help='refuse to start if the GPU is busy above this')
    parser.add_argument('--run-id', action='append', default=None,
                        help='restrict to specific run ids (repeatable)')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='unsafe: run from a dirty tracked worktree')
    parser.add_argument('--runs-json', type=Path,
                        default=ROOT / 'reports/stage6d_pretraining_runs.json')
    args = parser.parse_args(argv)

    plan = build_plan()
    if not plan['ok']:
        print('REFUSING: config plan has problems', file=sys.stderr)
        for problem in plan['problems']:
            print(f'  - {problem}', file=sys.stderr)
        return 1

    runs = plan['runs']
    if args.run_id:
        wanted = set(args.run_id)
        runs = [run for run in runs if run['run_id'] in wanted]
        missing = wanted.difference({run['run_id'] for run in runs})
        if missing:
            print(f'unknown run ids: {sorted(missing)}', file=sys.stderr)
            return 1

    if not args.execute or args.dry_run:
        return dry_run({'runs': runs}, args.device)

    guard = preflight(args.device, args.min_free_mib,
                      args.max_utilization_percent, args.allow_dirty)
    print(json.dumps(guard, indent=2, ensure_ascii=False))
    if guard['problems']:
        print('REFUSING TO START:', file=sys.stderr)
        for problem in guard['problems']:
            print(f'  - {problem}', file=sys.stderr)
        return 1

    records: List[Dict[str, object]] = []
    for run in runs:
        arm = str(run['arm'])
        seed = int(run['seed'])
        run_id = str(run['run_id'])
        cfg_path = config_path_for(arm, seed)
        run_dir = Path(run['output_dir'])
        if run_dir.exists() and any(run_dir.iterdir()):
            print(f'REFUSING: output dir not empty: {run_dir}', file=sys.stderr)
            return 1
        run_dir.mkdir(parents=True, exist_ok=True)

        launch = {
            'schema_version': 'stage6d_launch/v1',
            'started_at': now_utc(),
            'git_head': guard['git_head'],
            'git_status_porcelain': guard['git_status_porcelain'],
            'config': str(cfg_path.relative_to(ROOT)),
            'config_sha256': sha256_file(cfg_path),
            'arm': arm, 'run_id': run_id, 'pair_id': run['pair_id'],
            'seed': seed,
            'physical_gpu': gpu_report(args.device),
            'command': [sys.executable, 'main.py', '--fname',
                        str(cfg_path.relative_to(ROOT)), '--devices', f'cuda:{args.device}'],
        }
        (run_dir / 'stage6d_launch.json').write_text(
            json.dumps(launch, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

        console_path = run_dir / 'console.log'
        started = time.time()
        print(f'--- {run_id} (seed {seed}, GPU{args.device}) ---', flush=True)
        with console_path.open('w', encoding='utf-8') as console:
            process = subprocess.run(
                launch['command'], cwd=ROOT, stdout=console,
                stderr=subprocess.STDOUT, text=True)
        wall_minutes = round((time.time() - started) / 60, 1)
        record: Dict[str, object] = {
            'arm': arm, 'run_id': run_id, 'pair_id': run['pair_id'],
            'seed': seed, 'status': 'completed' if process.returncode == 0 else 'failed',
            'returncode': process.returncode, 'wall_clock_minutes': wall_minutes,
            'started_at': launch['started_at'], 'finished_at': now_utc(),
            'config': launch['config'], 'config_sha256': launch['config_sha256'],
            'output_dir': str(run_dir),
            'checkpoint_sha256': None, 'failure_reason': None,
        }
        if process.returncode != 0:
            record['failure_reason'] = f'training exited {process.returncode}'
        else:
            problems = verification_problems(run_dir, seed, arm)
            if problems:
                record['status'] = 'failed'
                record['failure_reason'] = '; '.join(problems)
            else:
                ckpt = run_dir / 'jepa-encoder-step2000.pth'
                record['checkpoint'] = str(ckpt)
                record['checkpoint_sha256'] = sha256_file(ckpt)
                with (run_dir / 'training_verification.yaml').open('r', encoding='utf-8') as handle:
                    verification = yaml.safe_load(handle)
                record['optimizer_steps_total'] = verification.get('optimizer_steps_total')
                record['micro_steps_total'] = verification.get('micro_steps_total')
                record['peak_memory_mib'] = verification.get('peak_memory_mib')
                record['git_commit'] = verification.get('git_commit')
                record['manifest_sha256'] = verification.get('manifest_sha256')
                (run_dir / 'stage6d_run_manifest.json').write_text(
                    json.dumps(run_manifest(run, run_dir, launch), indent=2,
                               ensure_ascii=False) + '\n', encoding='utf-8')
        records.append(record)
        print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
        if record['status'] != 'completed':
            print('STOPPING after a failed run; fix the cause before continuing.',
                  file=sys.stderr)
            break

    # persist the run log (merge with any existing completed records)
    existing: List[Dict[str, object]] = []
    if args.runs_json.is_file():
        existing = json.loads(args.runs_json.read_text(encoding='utf-8'))
    by_id = {record['run_id']: record for record in existing}
    for record in records:
        by_id[record['run_id']] = record
    merged = sorted(by_id.values(), key=lambda record: record['run_id'])
    args.runs_json.parent.mkdir(parents=True, exist_ok=True)
    args.runs_json.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    csv_path = args.runs_json.with_suffix('.csv')
    fieldnames = ['run_id', 'arm', 'pair_id', 'seed', 'status', 'optimizer_steps_total',
                  'micro_steps_total', 'manifest_sha256', 'checkpoint_sha256',
                  'git_commit', 'wall_clock_minutes', 'peak_memory_mib',
                  'output_dir', 'failure_reason']
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for record in merged:
            writer.writerow(record)
    print(f'wrote {args.runs_json.relative_to(ROOT)} and {csv_path.relative_to(ROOT)}')
    return 0 if all(record['status'] == 'completed' for record in merged) else 1


if __name__ == '__main__':
    raise SystemExit(main())
