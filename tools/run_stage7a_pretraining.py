#!/usr/bin/env python
"""Run the Stage 7A 50k-schedule experiment in two guarded phases.

Phase 1 (``--stage1``)
    Initialise from the official I-JEPA ViT-H/14 checkpoint, train on the
    frozen B1 protocol under the **full 50,000-update schedule**, save
    snapshots at 10k and 20k, then stop at the pure interruption point
    ``interrupt_after_optimizer_steps=20000``.  The schedule horizon stays 50k,
    so the 20k checkpoint is a mid-run state of the 50k trajectory, not a model
    trained on a 20k schedule.

Phase 2 (``--stage2``)
    Resume the full ``jepa-step20000.pth.tar`` and continue to 50,000.  Only
    ``load_checkpoint`` / ``read_checkpoint`` change; the horizon, warmup,
    LR/WD/EMA and natural sampling stay identical.  This is NEVER run
    automatically: the 20k -> 50k decision is made from the PhenoBench probes
    through ``tools/stage7a_decision.py`` and requires an explicit call.

Guards mirror Stage 6D: refuse a dirty tracked worktree, refuse a busy GPU, and
never switch GPUs on their own.  Physical GPU 1 is the project default.

Usage::

    ./.venv-uv/bin/python -m tools.run_stage7a_pretraining --dry-run
    ./.venv-uv/bin/python -m tools.run_stage7a_pretraining --stage1 --execute --device 1
    ./.venv-uv/bin/python -m tools.run_stage7a_pretraining --stage2 --execute --device 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from tools.stage7a_build_configs import (
    INTERRUPT_AFTER,
    MAX_OPTIMIZER_STEPS,
    PRETRAINING_SEED,
    ROOT,
    RUN_ID,
    RUN_ROOT,
    SAVE_AT_STEPS,
    STAGE1_CONFIG,
    STAGE2_CONFIG,
    WARMUP_STEPS,
    sha256_file,
)

EXPECTED_MANIFEST_SHA = '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2'
EXPECTED_PRETRAINED_SHA = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'
DEFAULT_MIN_FREE_MIB = 30000
MIN_DISK_FREE_GIB = 60.0
RUN_DIR = Path(RUN_ROOT)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def git(args: Sequence[str]) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def git_status_porcelain() -> List[str]:
    return [line for line in git(['status', '--porcelain']).splitlines() if line]


def tracked_changes(lines: Sequence[str]) -> List[str]:
    return [line for line in lines if not line.startswith('??')]


def gpu_report(physical_index: int) -> Dict[str, object]:
    query = 'index,uuid,name,memory.used,memory.total,utilization.gpu'
    out = subprocess.check_output([
        'nvidia-smi', f'--id={physical_index}',
        f'--query-gpu={query}', '--format=csv,noheader,nounits',
    ], text=True).strip()
    parts = [part.strip() for part in out.split(',')]
    if len(parts) != 6:
        raise RuntimeError(f'unexpected nvidia-smi output: {out!r}')
    used, total = int(parts[3]), int(parts[4])
    return {
        'physical_index': int(parts[0]), 'uuid': parts[1], 'name': parts[2],
        'memory_used_mib': used, 'memory_total_mib': total,
        'memory_free_mib': total - used, 'utilization_percent': int(parts[5]),
    }


def disk_free_gib(path: Path) -> float:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize / 1024 ** 3


def preflight(physical_index: int, min_free_mib: int,
              max_utilization_percent: int, allow_dirty: bool) -> Dict[str, object]:
    status_lines = git_status_porcelain()
    dirty = tracked_changes(status_lines)
    report: Dict[str, object] = {
        'git_head': git(['rev-parse', 'HEAD']),
        'git_branch': git(['rev-parse', '--abbrev-ref', 'HEAD']),
        'git_status_porcelain': status_lines,
        'tracked_changes': dirty,
        'git_diff_check_ok': subprocess.run(
            ['git', 'diff', '--check'], cwd=ROOT).returncode == 0,
        'physical_gpu': gpu_report(physical_index),
        'disk_free_gib_data': round(disk_free_gib(Path('/data')), 1),
        'min_free_mib': min_free_mib,
        'max_utilization_percent': max_utilization_percent,
        'problems': [],
    }
    if dirty and not allow_dirty:
        report['problems'].append(
            'tracked worktree is dirty; commit first so the checkpoint records '
            'a real code commit')
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
    if report['disk_free_gib_data'] < MIN_DISK_FREE_GIB:
        report['problems'].append(
            f'/data has {report["disk_free_gib_data"]} GiB free '
            f'< {MIN_DISK_FREE_GIB} GiB needed for checkpoints')
    return report


def load_config(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------- #
# schedule mirror -- proves LR / WD / EMA do not reset or jump on resume
# --------------------------------------------------------------------------- #

def expected_lr(step: int, cfg: Mapping[str, object]) -> float:
    opt = cfg['optimization']
    horizon = int(opt['max_optimizer_steps'])
    warmup = int(opt['warmup_steps'])
    start, ref, final = float(opt['start_lr']), float(opt['lr']), float(opt['final_lr'])
    if step < warmup:
        return start + (step / warmup) * (ref - start)
    t_max = horizon - warmup
    progress = (step - warmup) / max(1, t_max)
    return max(final, final + (ref - final) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def expected_wd(step: int, cfg: Mapping[str, object]) -> float:
    opt = cfg['optimization']
    horizon = int(opt['max_optimizer_steps'])
    ref, final = float(opt['weight_decay']), float(opt['final_weight_decay'])
    progress = step / horizon
    wd = final + (ref - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(final, wd) if final <= ref else min(final, wd)


def expected_ema(step: int, cfg: Mapping[str, object]) -> float:
    """Momentum applied on the update that takes the run to ``step`` updates.

    ``momentum_scheduler`` yields ``ema[0] + i*(ema[1]-ema[0])/horizon`` for
    ``i = 0, 1, ...``; the k-th update consumes ``i = k-1``.
    """
    if step <= 0:
        return float(cfg['optimization']['ema'][0])
    opt = cfg['optimization']
    horizon = int(opt['max_optimizer_steps'])
    ema0, ema1 = float(opt['ema'][0]), float(opt['ema'][1])
    return ema0 + (step - 1) * (ema1 - ema0) / horizon


def _close(actual, expected, rtol=1e-6, atol=1e-12) -> bool:
    if actual is None:
        return False
    return math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def _read_verification(run_dir: Path) -> Dict[str, object]:
    with (run_dir / 'training_verification.yaml').open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _check_common(verification: Mapping[str, object]) -> List[str]:
    problems: List[str] = []
    if verification.get('loss_finite') is not True:
        problems.append('loss_finite is not true')
    if verification.get('ema_formula_verified') is not True:
        problems.append('ema_formula_verified is not true')
    if verification.get('manifest_sha256') != EXPECTED_MANIFEST_SHA:
        problems.append(f'manifest_sha256 {verification.get("manifest_sha256")} '
                        f'!= {EXPECTED_MANIFEST_SHA}')
    pretrained = verification.get('pretrained_init') or {}
    if pretrained.get('sha256') != EXPECTED_PRETRAINED_SHA:
        problems.append(f'pretrained_init.sha256 {pretrained.get("sha256")} '
                        f'!= {EXPECTED_PRETRAINED_SHA}')
    if verification.get('git_commit') != git(['rev-parse', 'HEAD']):
        problems.append(f'verification git_commit {verification.get("git_commit")} '
                        f'!= current HEAD {git(["rev-parse", "HEAD"])}')
    return problems


SNAPSHOT_STEPS = {
    'stage1': [10000, 20000],
    'stage2': [50000],
}


def _check_snapshot_files(run_dir: Path, steps: Sequence[int]) -> List[str]:
    problems: List[str] = []
    for step in steps:
        full = run_dir / f'jepa-step{step}.pth.tar'
        enc = run_dir / f'jepa-encoder-step{step}.pth'
        if not full.is_file():
            problems.append(f'missing {full.name}')
        if not enc.is_file():
            problems.append(f'missing {enc.name}')
    latest = run_dir / 'jepa-latest.pth.tar'
    if not latest.is_file():
        problems.append('missing jepa-latest.pth.tar')
    csv_path = run_dir / 'jepa_r0.csv'
    if not csv_path.is_file():
        problems.append('missing jepa_r0.csv')
    if not (run_dir / 'params-ijepa.yaml').is_file():
        problems.append('missing params-ijepa.yaml')
    if not (run_dir / 'source_exposure_ledger.csv').is_file():
        problems.append('missing source_exposure_ledger.csv')
    if not (run_dir / 'source_exposure_ledger.summary.json').is_file():
        problems.append('missing source_exposure_ledger.summary.json')
    return problems


def _check_snapshot_schedule(run_dir: Path, step: int, cfg: Mapping[str, object]
                             ) -> List[str]:
    """The full snapshot records LR/WD/EMA; verify they sit on the 50k schedule."""
    import torch
    problems: List[str] = []
    full = run_dir / f'jepa-step{step}.pth.tar'
    if not full.is_file():
        return [f'missing {full.name} for schedule audit']
    snap = torch.load(full, map_location='cpu', weights_only=False)
    if int(snap.get('optimizer_steps', -1)) != step:
        problems.append(f'{full.name}: optimizer_steps {snap.get("optimizer_steps")} != {step}')
    if not _close(snap.get('lr_at_snapshot'), expected_lr(step, cfg)):
        problems.append(f'{full.name}: lr {snap.get("lr_at_snapshot")} != '
                        f'schedule {expected_lr(step, cfg)}')
    if not _close(snap.get('weight_decay_at_snapshot'), expected_wd(step, cfg)):
        problems.append(f'{full.name}: wd {snap.get("weight_decay_at_snapshot")} != '
                        f'schedule {expected_wd(step, cfg)}')
    if not _close(snap.get('ema_momentum_at_snapshot'), expected_ema(step, cfg)):
        problems.append(f'{full.name}: ema {snap.get("ema_momentum_at_snapshot")} != '
                        f'schedule {expected_ema(step, cfg)}')
    return problems


def verify_stage1(run_dir: Path) -> List[str]:
    cfg = load_config(STAGE1_CONFIG)
    problems: List[str] = []
    verification_path = run_dir / 'training_verification.yaml'
    if not verification_path.is_file():
        return [f'missing {verification_path}']
    v = _read_verification(run_dir)
    if v.get('optimizer_steps_total') != INTERRUPT_AFTER:
        problems.append(f'optimizer_steps_total {v.get("optimizer_steps_total")} '
                        f'!= {INTERRUPT_AFTER}')
    if v.get('optimizer_steps_this_run') != INTERRUPT_AFTER:
        problems.append(f'optimizer_steps_this_run {v.get("optimizer_steps_this_run")} '
                        f'!= {INTERRUPT_AFTER}')
    if v.get('max_optimizer_steps') != MAX_OPTIMIZER_STEPS:
        problems.append(f'max_optimizer_steps {v.get("max_optimizer_steps")} '
                        f'!= {MAX_OPTIMIZER_STEPS}')
    if v.get('total_optimizer_steps_horizon') != MAX_OPTIMIZER_STEPS:
        problems.append(f'schedule horizon {v.get("total_optimizer_steps_horizon")} '
                        f'!= {MAX_OPTIMIZER_STEPS}')
    if v.get('interrupt_after_optimizer_steps') != INTERRUPT_AFTER:
        problems.append('interrupt_after_optimizer_steps mismatch')
    problems.extend(_check_common(v))
    problems.extend(_check_snapshot_files(run_dir, SNAPSHOT_STEPS['stage1']))
    for step in SNAPSHOT_STEPS['stage1']:
        problems.extend(_check_snapshot_schedule(run_dir, step, cfg))
    params = load_config(run_dir / 'params-ijepa.yaml')
    if params['data'].get('seed') != PRETRAINING_SEED:
        problems.append(f"data.seed {params['data'].get('seed')} != {PRETRAINING_SEED}")
    if params['optimization'].get('max_optimizer_steps') != MAX_OPTIMIZER_STEPS:
        problems.append('params max_optimizer_steps mismatch')
    return problems


def verify_stage2(run_dir: Path, expected_20k_sha: str) -> List[str]:
    cfg = load_config(STAGE2_CONFIG)
    problems: List[str] = []
    verification_path = run_dir / 'training_verification.yaml'
    if not verification_path.is_file():
        return [f'missing {verification_path}']
    v = _read_verification(run_dir)
    if v.get('optimizer_steps_total') != MAX_OPTIMIZER_STEPS:
        problems.append(f'optimizer_steps_total {v.get("optimizer_steps_total")} '
                        f'!= {MAX_OPTIMIZER_STEPS}')
    expected_this_run = MAX_OPTIMIZER_STEPS - INTERRUPT_AFTER
    if v.get('optimizer_steps_this_run') != expected_this_run:
        problems.append(f'optimizer_steps_this_run {v.get("optimizer_steps_this_run")} '
                        f'!= {expected_this_run}')
    if v.get('max_optimizer_steps') != MAX_OPTIMIZER_STEPS:
        problems.append('max_optimizer_steps mismatch')
    if v.get('interrupt_after_optimizer_steps') is not None:
        problems.append('stage2 still carries an interruption point')
    problems.extend(_check_common(v))
    problems.extend(_check_snapshot_files(run_dir, SNAPSHOT_STEPS['stage2']))
    for step in SNAPSHOT_STEPS['stage2']:
        problems.extend(_check_snapshot_schedule(run_dir, step, cfg))
    # the 20k source checkpoint must be unchanged since phase 1
    ckpt20 = run_dir / 'jepa-step20000.pth.tar'
    if not ckpt20.is_file():
        problems.append('missing jepa-step20000.pth.tar')
    elif expected_20k_sha and sha256_file(ckpt20) != expected_20k_sha:
        problems.append(f'jepa-step20000.pth.tar SHA changed since phase 1 '
                        f'({sha256_file(ckpt20)} != {expected_20k_sha})')
    # source ledger must contain both phases
    summary_path = run_dir / 'source_exposure_ledger.summary.json'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        parts = summary.get('parts', {})
        if str(INTERRUPT_AFTER) not in parts:
            problems.append(f'source ledger missing phase-2 part {INTERRUPT_AFTER} '
                            f'(parts={sorted(parts)})')
        if '0' not in parts:
            problems.append(f'source ledger missing phase-1 part 0 (parts={sorted(parts)})')
    else:
        problems.append('missing source_exposure_ledger.summary.json')
    return problems


def checkpoint_manifest(run_dir: Path) -> Dict[str, object]:
    entries = []
    for pattern in ('jepa-step*.pth.tar', 'jepa-encoder-step*.pth',
                    'jepa-latest.pth.tar'):
        for path in sorted(run_dir.glob(pattern)):
            entries.append({'path': str(path), 'bytes': path.stat().st_size,
                            'sha256': sha256_file(path)})
    return {
        'schema_version': 'stage7a_checkpoint_manifest/v1',
        'run_id': RUN_ID,
        'output_dir': str(run_dir),
        'git_commit': git(['rev-parse', 'HEAD']),
        'checkpoints': entries,
    }


def stage_records(run_dir: Path, verification: Mapping[str, object],
                  launch: Mapping[str, object], wall_minutes: float) -> Dict[str, object]:
    return {
        'schema_version': 'stage7a_training_record/v1',
        'run_id': RUN_ID,
        'git_commit': verification.get('git_commit'),
        'config': launch['config'],
        'config_sha256': launch['config_sha256'],
        'phase': launch['phase'],
        'physical_gpu': launch['physical_gpu'],
        'started_at': launch['started_at'],
        'finished_at': now_utc(),
        'wall_clock_minutes': round(wall_minutes, 1),
        'optimizer_steps_total': verification.get('optimizer_steps_total'),
        'optimizer_steps_this_run': verification.get('optimizer_steps_this_run'),
        'micro_steps_total': verification.get('micro_steps_total'),
        'micro_steps_this_run': verification.get('micro_steps_this_run'),
        'optimizer_steps_per_epoch': verification.get('optimizer_steps_per_epoch'),
        'max_optimizer_steps': verification.get('max_optimizer_steps'),
        'interrupt_after_optimizer_steps': verification.get('interrupt_after_optimizer_steps'),
        'total_optimizer_steps_horizon': verification.get('total_optimizer_steps_horizon'),
        'loss_first': verification.get('loss_first'),
        'loss_last': verification.get('loss_last'),
        'loss_finite': verification.get('loss_finite'),
        'ema_formula_verified': verification.get('ema_formula_verified'),
        'peak_memory_mib': verification.get('peak_memory_mib'),
        'step_time_ms_mean': verification.get('step_time_ms_mean'),
        'manifest_sha256': verification.get('manifest_sha256'),
        'pretrained_init': verification.get('pretrained_init'),
        'hardware': verification.get('hardware'),
        'sampling_profile': verification.get('sampling_profile'),
        'realised_source_exposure': verification.get('realised_source_exposure'),
        'saved_step_snapshots': verification.get('saved_step_snapshots'),
    }


def preflight_md(report: Mapping[str, object], phase: str) -> str:
    lines = ['# Stage 7A — 训练前验收（preflight）', '']
    lines.append(f'- 阶段：{phase}')
    lines.append(f'- 时间（UTC）：{now_utc()}')
    lines.append(f'- 分支：{report["git_branch"]}')
    lines.append(f'- HEAD：`{report["git_head"]}`')
    lines.append(f'- 工作树状态：{"干净" if not report["tracked_changes"] else "有变更"}')
    lines.append(f'- `git diff --check`：{"通过" if report["git_diff_check_ok"] else "失败"}')
    lines.append('')
    gpu = report['physical_gpu']
    lines.append('## GPU')
    lines.append('')
    lines.append(f'- 物理编号：{gpu["physical_index"]}（{gpu["name"]}，UUID `{gpu["uuid"]}`）')
    lines.append(f'- 显存：已用 {gpu["memory_used_mib"]} MiB / 共 {gpu["memory_total_mib"]} MiB，'
                 f'空闲 {gpu["memory_free_mib"]} MiB')
    lines.append(f'- 利用率：{gpu["utilization_percent"]}%')
    lines.append(f'- `/data` 剩余：{report["disk_free_gib_data"]} GiB')
    lines.append('')
    if report['problems']:
        lines.append('## 阻止启动的问题')
        lines.append('')
        for problem in report['problems']:
            lines.append(f'- {problem}')
    else:
        lines.append('验收通过，可以启动。')
    return '\n'.join(lines) + '\n'


def run_phase(phase: str, device: int, min_free_mib: int,
              max_utilization_percent: int, allow_dirty: bool) -> int:
    cfg_path = STAGE1_CONFIG if phase == 'stage1' else STAGE2_CONFIG
    guard = preflight(device, min_free_mib, max_utilization_percent, allow_dirty)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'stage7a_preflight.md').write_text(preflight_md(guard, phase),
                                                  encoding='utf-8')
    (RUN_DIR / f'stage7a_preflight_{phase}.json').write_text(
        json.dumps(guard, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(guard, indent=2, ensure_ascii=False))
    if guard['problems']:
        print('REFUSING TO START:', file=sys.stderr)
        for problem in guard['problems']:
            print(f'  - {problem}', file=sys.stderr)
        return 1

    if phase == 'stage1':
        existing = [p for p in RUN_DIR.iterdir() if p.name != 'stage7a_preflight_stage1.json']
        if existing:
            print(f'REFUSING: output dir not empty: {RUN_DIR}', file=sys.stderr)
            for path in existing[:10]:
                print(f'  - {path.name}', file=sys.stderr)
            return 1
    else:
        ckpt20 = RUN_DIR / 'jepa-step20000.pth.tar'
        if not ckpt20.is_file():
            print(f'REFUSING stage2: missing phase-1 checkpoint {ckpt20}', file=sys.stderr)
            return 1
        stage1_json = reports / 'stage7a_training_20k.json'
        if not stage1_json.is_file():
            print(f'REFUSING stage2: missing {stage1_json}', file=sys.stderr)
            return 1

    launch = {
        'schema_version': 'stage7a_launch/v1',
        'phase': phase,
        'started_at': now_utc(),
        'git_head': guard['git_head'],
        'git_status_porcelain': guard['git_status_porcelain'],
        'config': str(cfg_path.relative_to(ROOT)),
        'config_sha256': sha256_file(cfg_path),
        'physical_gpu': guard['physical_gpu'],
        'command': [sys.executable, 'main.py', '--fname',
                    str(cfg_path.relative_to(ROOT)), '--devices', f'cuda:{device}'],
    }
    (RUN_DIR / f'stage7a_launch_{phase}.json').write_text(
        json.dumps(launch, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    console_path = RUN_DIR / f'console_{phase}.log'
    started = time.time()
    print(f'--- {phase} (GPU{device}) ---', flush=True)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(device)
    with console_path.open('w', encoding='utf-8') as console:
        process = subprocess.run(launch['command'], cwd=ROOT, env=env,
                                 stdout=console, stderr=subprocess.STDOUT, text=True)
    wall_minutes = (time.time() - started) / 60.0
    if process.returncode != 0:
        print(f'{phase} training exited {process.returncode}', file=sys.stderr)
        return 1

    verification = _read_verification(RUN_DIR)
    if phase == 'stage1':
        problems = verify_stage1(RUN_DIR)
    else:
        prior = json.loads((reports / 'stage7a_training_20k.json').read_text(encoding='utf-8'))
        problems = verify_stage2(RUN_DIR, prior.get('checkpoint_20k_sha'))
    if problems:
        print(f'{phase} VERIFICATION FAILED:', file=sys.stderr)
        for problem in problems:
            print(f'  - {problem}', file=sys.stderr)
        return 1

    record = stage_records(RUN_DIR, verification, launch, wall_minutes)
    manifest = checkpoint_manifest(RUN_DIR)
    if phase == 'stage1':
        record['checkpoint_10k_sha'] = sha256_file(RUN_DIR / 'jepa-step10000.pth.tar')
        record['checkpoint_20k_sha'] = sha256_file(RUN_DIR / 'jepa-step20000.pth.tar')
        record['encoder_10k_sha'] = sha256_file(RUN_DIR / 'jepa-encoder-step10000.pth')
        record['encoder_20k_sha'] = sha256_file(RUN_DIR / 'jepa-encoder-step20000.pth')
        (reports / 'stage7a_training_20k.json').write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    else:
        record['checkpoint_50k_sha'] = sha256_file(RUN_DIR / 'jepa-step50000.pth.tar')
        record['encoder_50k_sha'] = sha256_file(RUN_DIR / 'jepa-encoder-step50000.pth')
        (reports / 'stage7a_training_50k.json').write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (reports / 'stage7a_checkpoint_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--stage1', action='store_true',
                       help='train to the 20k interruption point (default phase)')
    group.add_argument('--stage2', action='store_true',
                       help='resume the 20k checkpoint and continue to 50k')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', type=int, default=1,
                        help='physical GPU index (project rule: 1)')
    parser.add_argument('--min-free-mib', type=int, default=DEFAULT_MIN_FREE_MIB)
    parser.add_argument('--max-utilization-percent', type=int, default=5)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args(argv)

    phase = 'stage2' if args.stage2 else 'stage1'
    if not args.execute or args.dry_run:
        print(json.dumps({
            'mode': 'dry-run', 'phase': phase, 'device': args.device,
            'config': str((STAGE1_CONFIG if phase == 'stage1' else STAGE2_CONFIG).relative_to(ROOT)),
            'command': [sys.executable, 'main.py', '--fname',
                        str((STAGE1_CONFIG if phase == 'stage1'
                             else STAGE2_CONFIG).relative_to(ROOT)),
                        '--devices', f'cuda:{args.device}'],
            'output_dir': str(RUN_DIR),
        }, indent=2, ensure_ascii=False))
        return 0
    return run_phase(phase, args.device, args.min_free_mib,
                     args.max_utilization_percent, args.allow_dirty)


if __name__ == '__main__':
    raise SystemExit(main())
