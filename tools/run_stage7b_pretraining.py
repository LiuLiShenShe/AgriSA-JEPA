#!/usr/bin/env python
"""Stage 7B: run the 3-arm x 2-seed paired schedule ablation (serial, GPU1).

Six runs, each exactly 2,000 optimizer updates from the official I-JEPA init:

* A_p1, A_p2 -- 2,000 horizon (schedule completes at 2,000)
* B_p1, B_p2 -- 50,000 horizon, interrupt at 2,000, warmup 100
* C_p1, C_p2 -- 50,000 horizon, interrupt at 2,000, warmup 2,500

Runs are serial (never parallel on the GPU).  Each run is verified against its
arm's schedule position at 2,000 (LR/WD/EMA), the frozen manifest/pretrained
SHAs, finite loss, EMA formula, and the step-2,000 full+encoder snapshot.

Writes ``reports/stage7b_pretraining_runs.json`` and ``.csv``.

Usage::

    ./.venv-uv/bin/python -m tools.run_stage7b_pretraining --dry-run
    ./.venv-uv/bin/python -m tools.run_stage7b_pretraining --execute --device 1
    ./.venv-uv/bin/python -m tools.run_stage7b_pretraining --execute --device 1 --only A_p1
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
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from tools.run_stage7a_pretraining import (
    expected_ema,
    expected_lr,
    expected_wd,
    preflight,
)
from tools.stage7a_build_configs import ROOT
from tools.stage7b_build_configs import ARMS, RUN_ROOT, SEEDS, build_plan
from tools.stage7b_protocol_check import STOP

R = ROOT / 'reports'


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def run_one(key: str, device: int) -> int:
    cfg_path = ROOT / 'configs' / f'stage7b_{key}.yaml'
    run_dir = RUN_ROOT / key
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    command = [sys.executable, 'main.py', '--fname',
               str(cfg_path.relative_to(ROOT)), '--devices', f'cuda:{device}']
    print(f'--- stage7b {key} (physical GPU{device}) ---', flush=True)
    console = run_dir / 'console.log'
    process = subprocess.run(command, cwd=ROOT, env=env, text=True)
    return process.returncode


def verify_run(key: str) -> List[str]:
    import math
    import torch
    arm, tag = key.split('_')
    spec = ARMS[arm]
    cfg = load_yaml(ROOT / 'configs' / f'stage7b_{key}.yaml')
    run_dir = RUN_ROOT / key
    problems: List[str] = []
    ver_path = run_dir / 'training_verification.yaml'
    if not ver_path.is_file():
        return [f'missing {ver_path}']
    v = load_yaml(ver_path)
    if v.get('optimizer_steps_total') != STOP:
        problems.append(f'steps {v.get("optimizer_steps_total")} != {STOP}')
    if v.get('max_optimizer_steps') != spec['max_optimizer_steps']:
        problems.append(f'horizon {v.get("max_optimizer_steps")} != '
                        f'{spec["max_optimizer_steps"]}')
    if v.get('interrupt_after_optimizer_steps') != spec['interrupt_after_optimizer_steps']:
        problems.append('interrupt mismatch')
    if v.get('loss_finite') is not True:
        problems.append('loss not finite')
    if v.get('ema_formula_verified') is not True:
        problems.append('ema not verified')
    if v.get('manifest_sha256') != '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2':
        problems.append('manifest SHA mismatch')
    if (v.get('pretrained_init') or {}).get('sha256') != \
            '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22':
        problems.append('pretrained SHA mismatch')
    if cfg['data'].get('seed') != SEEDS[tag]:
        problems.append(f'seed {cfg["data"].get("seed")} != {SEEDS[tag]}')
    # schedule position at the 2,000 stop
    horizon = spec['max_optimizer_steps']
    warmup = spec['warmup_steps']
    full = run_dir / 'jepa-step2000.pth.tar'
    if not full.is_file():
        problems.append('missing jepa-step2000.pth.tar')
    else:
        snap = torch.load(full, map_location='cpu', weights_only=False)
        if int(snap.get('optimizer_steps', -1)) != STOP:
            problems.append(f'snapshot optimizer_steps {snap.get("optimizer_steps")} != {STOP}')
        for label, got, want in (
                ('LR', snap.get('lr_at_snapshot'), expected_lr(STOP, horizon, warmup)),
                ('WD', snap.get('weight_decay_at_snapshot'),
                 expected_wd(STOP, horizon)),
                ('EMA', snap.get('ema_momentum_at_snapshot'),
                 expected_ema(STOP, horizon))):
            if got is None or not math.isclose(float(got), float(want), rel_tol=1e-6,
                                               abs_tol=1e-12):
                problems.append(f'{label}@snapshot {got} != {want}')
    if not (run_dir / 'jepa-encoder-step2000.pth').is_file():
        problems.append('missing encoder snapshot')
    summary_path = run_dir / 'source_exposure_ledger.summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if '0' not in summary.get('parts', {}):
            problems.append('ledger missing part 0')
    else:
        problems.append('missing ledger summary')
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--only', nargs='*', default=[])
    args = parser.parse_args(argv)

    plan = build_plan()
    keys = [k for k in plan['runs'] if not args.only or k in args.only]
    if not keys:
        print('no runs selected', file=sys.stderr)
        return 1

    if args.dry_run or not args.execute:
        print(json.dumps({
            'mode': 'dry-run', 'device': args.device, 'runs': keys,
            'configs': {k: f'configs/stage7b_{k}.yaml' for k in keys},
        }, indent=2, ensure_ascii=False))
        return 0

    guard = preflight(args.device, 30000, 5, False)
    print(json.dumps(guard, indent=2, ensure_ascii=False))
    if guard['problems']:
        print('REFUSING: clean, free GPU1 required', file=sys.stderr)
        return 1

    records = []
    for key in keys:
        run_dir = RUN_ROOT / key
        if (run_dir / 'training_verification.yaml').is_file():
            print(f'SKIP {key}: already has a verification record', flush=True)
            continue
        started = now_utc()
        t0 = time.time()
        rc = run_one(key, args.device)
        wall = (time.time() - t0) / 60.0
        if rc != 0:
            print(f'{key} training exited {rc}', file=sys.stderr)
            return 1
        problems = verify_run(key)
        if problems:
            print(f'{key} VERIFICATION FAILED:', file=sys.stderr)
            for problem in problems:
                print(f'  - {problem}', file=sys.stderr)
            return 1
        v = load_yaml(run_dir / 'training_verification.yaml')
        records.append({
            'run_id': key, 'arm': key.split('_')[0], 'tag': key.split('_')[1],
            'seed': int(cfg_seed(key)), 'started_at': started, 'finished_at': now_utc(),
            'wall_clock_minutes': round(wall, 1),
            'optimizer_steps_total': v.get('optimizer_steps_total'),
            'max_optimizer_steps': v.get('max_optimizer_steps'),
            'warmup_steps': int(load_yaml(ROOT / 'configs' / f'stage7b_{key}.yaml')
                                ['optimization']['warmup_steps']),
            'micro_steps_total': v.get('micro_steps_total'),
            'loss_first': v.get('loss_first'), 'loss_last': v.get('loss_last'),
            'loss_finite': v.get('loss_finite'),
            'ema_formula_verified': v.get('ema_formula_verified'),
            'peak_memory_mib': v.get('peak_memory_mib'),
            'step_time_ms_mean': v.get('step_time_ms_mean'),
            'git_commit': v.get('git_commit'),
            'realised_source_exposure': v.get('realised_source_exposure'),
        })

    R.mkdir(exist_ok=True)
    (R / 'stage7b_pretraining_runs.json').write_text(
        json.dumps({'schema_version': 'stage7b_pretraining_runs/v1',
                    'runs': records}, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    with (R / 'stage7b_pretraining_runs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records
                                else ['run_id'])
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({'ok': True, 'n_records': len(records)}, indent=2))
    return 0


def cfg_seed(key: str) -> str:
    return str(load_yaml(ROOT / 'configs' / f'stage7b_{key}.yaml')['data']['seed'])


if __name__ == '__main__':
    raise SystemExit(main())
