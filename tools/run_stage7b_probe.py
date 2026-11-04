#!/usr/bin/env python
"""Stage 7B: PhenoBench frozen-encoder probe of the six ablation encoders.

Protocol is identical to Stage 6C/6D/7A: frozen ViT-H/14, 224x224 overlapping
tiles (stride 200), ``Linear(1280 -> 3)``, AdamW/1e-3, 10 epochs, probe seeds
0/1/2, val split only (772 images), official ``phenobench-eval`` cross-checked
against the self metric.

Probes each of the six ``jepa-encoder-step2000.pth`` checkpoints:
A_p1, A_p2, B_p1, B_p2, C_p1, C_p2.

Usage::

    ./.venv-uv/bin/python -m tools.run_stage7b_probe --dry-run
    ./.venv-uv/bin/python -m tools.run_stage7b_probe --execute --device 1
    ./.venv-uv/bin/python -m tools.run_stage7b_probe --execute --device 1 --only A_p1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from tools.stage7a_build_configs import ROOT
from tools.stage7b_build_configs import ARMS, RUN_ROOT, SEEDS

PHENOBENCH_ROOT = '/data/datasets/PhenoBench/PhenoBench-v110'
PHENOBENCH_MANIFEST = ROOT / 'reports/phenobench_file_manifest.sha256.txt'
OUT_ROOT = RUN_ROOT / 'phenobench'
CACHE_DIR = RUN_ROOT / 'phenobench_cache'
PROBE_SEEDS = (0, 1, 2)
EXPECTED_VAL_IMAGES = 772
PROTOCOL_VERSION = 'stage7b-phenobench-linear-probe-v1'
STOP = 2000


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def phenobench_manifest_sha() -> str:
    digest = hashlib.sha256()
    with PHENOBENCH_MANIFEST.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_path(key: str) -> Path:
    # the training configs write to RUN_ROOT/stage7b_{arm}_{tag} (see
    # tools/stage7b_build_configs.run_id), not RUN_ROOT/{arm}_{tag}
    return RUN_ROOT / f'stage7b_{key}' / f'jepa-encoder-step{STOP}.pth'


def new_records(only: Sequence[str] = ()) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for key in [f'{arm}_{tag}' for arm in ARMS for tag in SEEDS]:
        if only and key not in only:
            continue
        path = encoder_path(key)
        if not path.is_file():
            print(f'SKIP {key}: {path} not found', file=sys.stderr)
            continue
        arm, tag = key.split('_')
        records.append({
            'arm': arm, 'run_id': key, 'pair_id': tag.upper(),
            'seed': SEEDS[tag], 'traceable': True, 'phase': f'arm_{arm}',
            'checkpoint': str(path), 'checkpoint_sha256': sha256_file(path),
        })
    return records


def probe_label(record: Mapping[str, object]) -> str:
    return (f"stage7b_{record['run_id']}__{record['checkpoint_sha256'][:24]}")


def command_for(record: Mapping[str, object], device_index: int) -> List[str]:
    label = probe_label(record)
    out_dir = OUT_ROOT / str(record['run_id'])
    return [
        sys.executable, '-m', 'src.downstream.phenobench_probe',
        '--checkpoint', str(record['checkpoint']),
        '--arm-label', label,
        '--continued-training',
        (f"stage7b paired schedule ablation, arm {record['arm']}, "
         f"2,000 optimizer updates"),
        '--pretraining-data-version',
        'R3v3_license_verified train; 147277 images; audit status CONDITIONAL',
        '--stage', 'all',
        '--root', PHENOBENCH_ROOT,
        '--cache-dir', str(CACHE_DIR),
        '--out-dir', str(out_dir),
        '--device', 'cuda:0',
        '--seed', *[str(seed) for seed in PROBE_SEEDS],
    ], out_dir


def verify_probe_result(result_path: Path, checkpoint_sha: str) -> List[str]:
    problems: List[str] = []
    if not result_path.is_file():
        return [f'missing {result_path}']
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('backbone_checkpoint_sha256') != checkpoint_sha:
        problems.append('checkpoint SHA mismatch in result')
    seeds = [s.get('seed') for s in result.get('seeds', [])]
    if seeds != list(PROBE_SEEDS):
        problems.append(f'probe seeds {seeds} != {list(PROBE_SEEDS)}')
    if result.get('protocol', {}).get('val_images') != EXPECTED_VAL_IMAGES:
        problems.append('val_images mismatch')
    for seed in result.get('seeds', []):
        cross = seed.get('metric_cross_check') or {}
        if not cross.get('agrees'):
            problems.append(f"seed {seed.get('seed')}: official/self disagree")
    return problems


def build_manifest(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    return {
        'schema_version': 'stage7b_phenobench_manifest/v1',
        'phenobench_manifest_sha256': phenobench_manifest_sha(),
        'protocol_version': PROTOCOL_VERSION,
        'probe_seeds': list(PROBE_SEEDS),
        'note': ('3 schedule arms x 2 pretraining seeds, each exactly 2,000 '
                 'optimizer updates; arms are NOT independent seeds.'),
        'entries': [{
            'arm': r['arm'], 'run_id': r['run_id'], 'pair_id': r['pair_id'],
            'seed': r['seed'], 'phase': r['phase'], 'traceable': True,
            'checkpoint': r['checkpoint'],
            'checkpoint_sha256': r['checkpoint_sha256'],
            'result': str(OUT_ROOT / str(r['run_id']) / 'result.json'),
        } for r in records],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--only', nargs='*', default=[])
    parser.add_argument('--manifest-out', type=Path,
                        default=ROOT / 'reports/stage7b_phenobench_manifest.json')
    args = parser.parse_args(argv)

    records = new_records(tuple(args.only))
    if not records:
        print('no probeable encoders found', file=sys.stderr)
        return 1

    if args.dry_run or not args.execute:
        print(json.dumps({'mode': 'dry-run', 'device': args.device,
                          'records': [r['run_id'] for r in records],
                          'commands': [command_for(r, 0)[0] for r in records]},
                         indent=2, ensure_ascii=False))
        return 0

    query = subprocess.check_output([
        'nvidia-smi', f'--id={args.device}',
        '--query-gpu=memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    used, total, util = [int(part.strip()) for part in query.split(',')]
    if total - used < 30000 or util > 5:
        print(f'REFUSING: GPU{args.device} busy (free {total - used} MiB, '
              f'util {util}%)', file=sys.stderr)
        return 1

    for record in records:
        command, out_dir = command_for(record, 0)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'stage7b_probe_binding.json').write_text(json.dumps({
            'schema_version': 'stage7b_probe_binding/v1',
            'run_id': record['run_id'], 'arm': record['arm'],
            'pair_id': record['pair_id'], 'seed': record['seed'],
            'phase': record['phase'], 'checkpoint': record['checkpoint'],
            'checkpoint_sha256': record['checkpoint_sha256'],
            'phenobench_manifest_sha256': phenobench_manifest_sha(),
            'protocol_version': PROTOCOL_VERSION,
            'probe_seeds': list(PROBE_SEEDS), 'command': command,
            'started_at': now_utc(),
        }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = str(args.device)
        print(f'--- probe {record["run_id"]} on physical GPU{args.device} ---',
              flush=True)
        with (out_dir / 'console.log').open('w', encoding='utf-8') as console:
            process = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     stdout=console, stderr=subprocess.STDOUT)
        if process.returncode != 0:
            print(f'probe failed for {record["run_id"]}', file=sys.stderr)
            return 1
        problems = verify_probe_result(out_dir / 'result.json',
                                       str(record['checkpoint_sha256']))
        if problems:
            print(f'PROBE VERIFICATION FAILED ({record["run_id"]}):', file=sys.stderr)
            for problem in problems:
                print(f'  - {problem}', file=sys.stderr)
            return 1

    manifest = build_manifest(records)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {args.manifest_out.relative_to(ROOT)} with '
          f'{len(manifest["entries"])} entries')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
