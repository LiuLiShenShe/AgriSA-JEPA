#!/usr/bin/env python
"""Run the Stage 7A PhenoBench frozen-encoder probes on the 10k / 20k / (50k) checkpoints.

The protocol is byte-for-byte the Stage 6C/6D one: frozen ViT-H/14 encoder,
224x224 overlapping tiles, ``Linear(1280 -> 3)`` head, AdamW/1e-3, 10 epochs,
probe seeds 0/1/2, official ``phenobench-eval`` cross-checked against the self
metric, val split only (772 images).  The probe never touches the pretraining
schedule; it only answers whether more budget helps.

The purpose is DECISION SUPPORT, not a full multi-task benchmark: only the two
mandatory checkpoints (10k, 20k) are probed by default.  The 50k checkpoint is
probed only after the decision rule passes and phase 2 has finished.

Usage::

    ./.venv-uv/bin/python -m tools.run_stage7a_probe --dry-run
    ./.venv-uv/bin/python -m tools.run_stage7a_probe --execute --device 1
    ./.venv-uv/bin/python -m tools.run_stage7a_probe --include-50k --execute --device 1
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

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path('/data/agrisa_jepa_runs/stage7a_b1_natural_50k')
PHENOBENCH_ROOT = '/data/datasets/PhenoBench/PhenoBench-v110'
PHENOBENCH_MANIFEST = ROOT / 'reports/phenobench_file_manifest.sha256.txt'
ENCODER_FILES = {
    '10k': RUN_ROOT / 'jepa-encoder-step10000.pth',
    '20k': RUN_ROOT / 'jepa-encoder-step20000.pth',
    '50k': RUN_ROOT / 'jepa-encoder-step50000.pth',
}
PROBE_SEEDS = (0, 1, 2)
EXPECTED_VAL_IMAGES = 772
PROTOCOL_VERSION = 'stage7a-phenobench-linear-probe-v1'

BASELINE_A = {
    'arm': 'A', 'run_id': 'A_official', 'pair_id': '', 'seed': 0,
    'traceable': True,
    'result': 'runs/stage3a_phenobench/armA/result.json',
    'checkpoint': '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
    'checkpoint_sha256': '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22',
}

#: Stage 6D traceable B1 2k checkpoint (P1, seed 20261027), re-used, not re-run.
#: It was trained under a 2k TOTAL schedule, so it is a cross-schedule reference
#: for the 10k/20k trajectory, not part of the budget decision.
STAGE6D_2K_REFERENCE = {
    'arm': 'B1', 'run_id': 'stage6d_b1_natural_seed20261027_2k',
    'pair_id': 'ref_stage6d', 'seed': 20261027, 'traceable': True,
    'phase': 'ref_2k_stage6d',
    'result': ('/data/agrisa_jepa_runs/stage6d/phenobench/'
               'stage6d_b1_natural_seed20261027/result.json'),
    'checkpoint': ('/data/agrisa_jepa_runs/stage6d/stage6d_b1_natural_seed20261027/'
                   'jepa-encoder-step2000.pth'),
    'checkpoint_sha256': '1dc10ab68ab48b7e4de0bfa0cd444a1328d0104589484e4454d36cbe5f389106',
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def phenobench_manifest_sha() -> str:
    if not PHENOBENCH_MANIFEST.is_file():
        raise FileNotFoundError(PHENOBENCH_MANIFEST)
    return sha256_file(PHENOBENCH_MANIFEST)


def probe_label(run_id: str, checkpoint_sha: str, manifest_sha: str) -> str:
    digest = hashlib.sha256(
        f'{run_id}|{checkpoint_sha}|{manifest_sha}|{PROTOCOL_VERSION}'.encode('utf-8')
    ).hexdigest()
    return f'stage7a_{run_id}__{digest[:24]}'


def new_records(include_50k: bool) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for tag, path in ENCODER_FILES.items():
        if tag == '50k' and not include_50k:
            continue
        if not path.is_file():
            print(f'SKIP {tag}: {path} not found', file=sys.stderr)
            continue
        records.append({
            'arm': 'B1', 'run_id': f'stage7a_b1_natural_50k_{tag}',
            'pair_id': 'S7A', 'seed': 20261029,
            'traceable': True, 'phase': f'step{tag}',
            'checkpoint': str(path),
            'checkpoint_sha256': sha256_file(path),
        })
    return records


def verify_probe_result(result_path: Path, checkpoint_sha: str,
                        seeds: Sequence[int] = PROBE_SEEDS) -> List[str]:
    if not result_path.is_file():
        return [f'missing {result_path}']
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('backbone_checkpoint_sha256') != checkpoint_sha:
        return [f'{result_path}: checkpoint SHA {result.get("backbone_checkpoint_sha256")} '
                f'!= {checkpoint_sha}']
    observed = [entry.get('seed') for entry in result.get('seeds', [])]
    if observed != list(seeds):
        return [f'{result_path}: probe seeds {observed} != {list(seeds)}']
    if (result.get('protocol') or {}).get('val_images') != EXPECTED_VAL_IMAGES:
        return [f'{result_path}: val_images mismatch']
    for entry in result.get('seeds', []):
        seed = entry.get('seed')
        pred_dir = Path(entry.get('predictions_dir', ''))
        if not pred_dir.is_absolute():
            pred_dir = ROOT / pred_dir
        count = len(list(pred_dir.glob('*.png'))) if pred_dir.is_dir() else 0
        if count != EXPECTED_VAL_IMAGES:
            return [f'{result_path} seed {seed}: {count} predictions != {EXPECTED_VAL_IMAGES}']
        stdout = pred_dir.parents[1] / 'phenobench_eval_stdout.txt'
        if not stdout.is_file():
            return [f'{result_path} seed {seed}: missing {stdout}']
        if (entry.get('metric_cross_check') or {}).get('agrees') is not True:
            return [f'{result_path} seed {seed}: self/official cross-check did not agree']
    return []


def build_comparison_manifest(new_records: Sequence[Dict[str, object]],
                              out_root: Path) -> Dict[str, object]:
    entries = [dict(BASELINE_A), dict(STAGE6D_2K_REFERENCE)]
    for record in new_records:
        entries.append({
            'arm': record['arm'], 'run_id': record['run_id'],
            'pair_id': record['pair_id'], 'seed': record['seed'],
            'traceable': True, 'phase': record.get('phase'),
            'result': str(out_root / record['run_id'] / 'result.json'),
            'checkpoint': record['checkpoint'],
            'checkpoint_sha256': record['checkpoint_sha256'],
        })
    return {
        'schema_version': 'stage7a_comparison_manifest/v1',
        'phenobench_manifest_sha256': phenobench_manifest_sha(),
        'protocol_version': PROTOCOL_VERSION,
        'probe_seeds': list(PROBE_SEEDS),
        'note': ('Only the 10k/20k/50k checkpoints are one 50k-schedule trajectory; '
                 'they are NOT independent pretraining seeds.  The Stage 6D 2k '
                 'checkpoint is a cross-schedule reference (2k total schedule).'),
        'entries': entries,
    }


def command_for(record: Mapping[str, object], out_root: Path,
                cache_dir: Path, device: int) -> List[str]:
    checkpoint = str(record['checkpoint'])
    label = probe_label(str(record['run_id']), str(record['checkpoint_sha256']),
                        phenobench_manifest_sha())
    out_dir = out_root / str(record['run_id'])
    return [
        sys.executable, '-m', 'src.downstream.phenobench_probe',
        '--checkpoint', checkpoint,
        '--arm-label', label,
        '--continued-training',
        ('50,000-update 50k schedule (snapshotted at '
         f'{record["phase"]}); official I-JEPA objective; natural source sampling'),
        '--pretraining-data-version',
        'R3v3_license_verified train; 147277 images; audit status CONDITIONAL',
        '--stage', 'all',
        '--root', PHENOBENCH_ROOT,
        '--cache-dir', str(cache_dir),
        '--out-dir', str(out_dir),
        '--device', 'cuda:0',
        '--seed', *[str(seed) for seed in PROBE_SEEDS],
    ], out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--include-50k', action='store_true',
                        help='also probe the 50k checkpoint (after phase 2)')
    parser.add_argument('--manifest-only', action='store_true',
                        help='validate existing results and write the comparison manifest')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--out-root', type=Path,
                        default=Path('/data/agrisa_jepa_runs/stage7a_b1_natural_50k/phenobench'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/data/agrisa_jepa_runs/stage7a_b1_natural_50k/phenobench_cache'))
    parser.add_argument('--manifest-out', type=Path,
                        default=ROOT / 'reports/stage7a_comparison_manifest.json')
    args = parser.parse_args(argv)

    records = new_records(args.include_50k)
    if not records and not args.manifest_only:
        print('no probeable checkpoints found (run training first)', file=sys.stderr)
        return 1

    if args.dry_run or (not args.execute and not args.manifest_only):
        print(json.dumps({
            'mode': 'dry-run', 'device': args.device,
            'records': [r['run_id'] for r in records],
            'commands': [command_for(r, args.out_root, args.cache_dir, args.device)[0]
                         for r in records],
        }, indent=2, ensure_ascii=False))
        return 0

    if args.execute:
        query = subprocess.check_output([
            'nvidia-smi', f'--id={args.device}',
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        used, total, util = [int(part.strip()) for part in query.split(',')]
        free = total - used
        if free < 30000 or util > 5:
            print(f'REFUSING: physical GPU{args.device} free={free} MiB '
                  f'util={util}%; another process is using it', file=sys.stderr)
            return 1
        for record in records:
            command, out_dir = command_for(record, args.out_root, args.cache_dir, args.device)
            out_dir.mkdir(parents=True, exist_ok=True)
            binding = {
                'schema_version': 'stage7a_probe_binding/v1',
                'run_id': record['run_id'], 'arm': record['arm'],
                'pair_id': record['pair_id'], 'seed': record['seed'],
                'phase': record['phase'],
                'checkpoint': record['checkpoint'],
                'checkpoint_sha256': record['checkpoint_sha256'],
                'phenobench_manifest_sha256': phenobench_manifest_sha(),
                'protocol_version': PROTOCOL_VERSION,
                'probe_seeds': list(PROBE_SEEDS),
                'command': command, 'started_at': now_utc(),
            }
            (out_dir / 'stage7a_probe_binding.json').write_text(
                json.dumps(binding, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(args.device)
            print(f'--- probe {record["run_id"]} on physical GPU{args.device} ---', flush=True)
            with (out_dir / 'console.log').open('w', encoding='utf-8') as console:
                process = subprocess.run(command, cwd=ROOT, env=env,
                                         stdout=console, stderr=subprocess.STDOUT, text=True)
            if process.returncode != 0:
                print(f'probe failed for {record["run_id"]}', file=sys.stderr)
                return 1
            problems = verify_probe_result(out_dir / 'result.json',
                                           str(record['checkpoint_sha256']))
            if problems:
                print('PROBE VERIFICATION FAILED:', file=sys.stderr)
                for problem in problems:
                    print(f'  - {problem}', file=sys.stderr)
                return 1

    manifest = build_comparison_manifest(records, args.out_root)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {args.manifest_out.relative_to(ROOT)} with {len(manifest["entries"])} entries')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
