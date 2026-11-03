#!/usr/bin/env python
"""Run the Stage 6D PhenoBench frozen-encoder probes, one checkpoint at a time.

The scientific protocol is the Stage 6C one, unchanged: frozen ViT-H/14, 224x224
overlapping tiles, ``Linear(1280 -> 3)`` head, AdamW/1e-3, 10 epochs, probe
seeds 0/1/2, official ``phenobench-eval`` cross-checked against the self metric,
val split only (772 images).

What this wrapper adds:

* it refuses to probe a checkpoint whose on-disk SHA does not match the SHA the
  pretraining runner recorded;
* it binds the feature cache to run id + **full** checkpoint SHA + PhenoBench
  manifest SHA + protocol version (the probe seed is bound in the per-seed
  result record, since feature tensors are seed-independent);
* after each run it verifies that all 772 val predictions and the official
  evaluator stdout exist, and writes a prediction manifest;
* it assembles the comparison manifest (A + historical P0 + new runs) used by
  ``tools/stage6d_compare.py``.

Usage::

    ./.venv/bin/python -m tools.run_stage6d_probe --dry-run
    ./.venv/bin/python -m tools.run_stage6d_probe --execute --device 1
    ./.venv/bin/python -m tools.run_stage6d_probe --manifest-only
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
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
PHENOBENCH_ROOT = '/data/datasets/PhenoBench/PhenoBench-v110'
PHENOBENCH_MANIFEST = ROOT / 'reports/phenobench_file_manifest.sha256.txt'
PROBE_SEEDS = (0, 1, 2)
EXPECTED_VAL_IMAGES = 772
PROTOCOL_VERSION = 'stage6d-phenobench-linear-probe-v1'

#: Historical results reused, not re-run.  They are explicitly untraceable for
#: the Stage 6D code commit; the comparison reports them as a caveat.
HISTORICAL = (
    {'arm': 'A', 'run_id': 'A_official', 'pair_id': '', 'seed': 0,
     'traceable': True, 'result': 'runs/stage3a_phenobench/armA/result.json',
     'checkpoint': '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
     'checkpoint_sha256': '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'},
    {'arm': 'B1', 'run_id': 'stage6_b1_natural_2000', 'pair_id': 'P0_historical',
     'seed': 20261026, 'traceable': False,
     'result': 'runs/stage6c_phenobench/B1_natural/result.json',
     'checkpoint': 'runs/stage6_b1_natural_2000/jepa-encoder-step2000.pth',
     'checkpoint_sha256': '19796cebc65e29a07eafd7222f5a642d4ef9f28c33f2e4f4f2b65e34a2bfde6b'},
    {'arm': 'B2', 'run_id': 'stage6_b2_capped_no_backfill_2000',
     'pair_id': 'P0_historical', 'seed': 20261026, 'traceable': False,
     'result': 'runs/stage6c_phenobench/B2_capped/result.json',
     'checkpoint': 'runs/stage6_b2_capped_no_backfill_2000/jepa-encoder-step2000.pth',
     'checkpoint_sha256': '6e7a83c9156e556757016145fb804e2b67e7b94f5e8b45bda4ffd24cb5975a62'},
)


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
    """Short, filesystem-safe label that cryptographically binds all inputs."""
    digest = hashlib.sha256(
        f'{run_id}|{checkpoint_sha}|{manifest_sha}|{PROTOCOL_VERSION}'.encode('utf-8')
    ).hexdigest()
    return f'stage6d_{run_id}__{digest[:24]}'


def new_runs(runs_json: Path) -> List[Dict[str, object]]:
    if not runs_json.is_file():
        return []
    records = json.loads(runs_json.read_text(encoding='utf-8'))
    out = []
    for record in records:
        if record.get('status') != 'completed':
            continue
        out.append({
            'arm': record['arm'], 'run_id': record['run_id'],
            'pair_id': record['pair_id'], 'seed': record['seed'],
            'traceable': True, 'checkpoint': record['checkpoint'],
            'checkpoint_sha256': record['checkpoint_sha256'],
        })
    return out


def verify_probe_result(result_path: Path, checkpoint_sha: str,
                        seeds: Sequence[int] = PROBE_SEEDS) -> List[str]:
    problems: List[str] = []
    if not result_path.is_file():
        return [f'missing {result_path}']
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('backbone_checkpoint_sha256') != checkpoint_sha:
        problems.append(
            f'{result_path}: checkpoint SHA {result.get("backbone_checkpoint_sha256")} '
            f'!= {checkpoint_sha}')
    observed = [entry.get('seed') for entry in result.get('seeds', [])]
    if observed != list(seeds):
        problems.append(f'{result_path}: probe seeds {observed} != {list(seeds)}')
    if (result.get('protocol') or {}).get('val_images') != EXPECTED_VAL_IMAGES:
        problems.append(
            f'{result_path}: val_images {(result.get("protocol") or {}).get("val_images")} '
            f'!= {EXPECTED_VAL_IMAGES}')
    for entry in result.get('seeds', []):
        seed = entry.get('seed')
        pred_dir = Path(entry.get('predictions_dir', ''))
        if not pred_dir.is_absolute():
            pred_dir = ROOT / pred_dir
        count = len(list(pred_dir.glob('*.png'))) if pred_dir.is_dir() else 0
        if count != EXPECTED_VAL_IMAGES:
            problems.append(f'{result_path} seed {seed}: {count} predictions != '
                            f'{EXPECTED_VAL_IMAGES}')
        stdout = pred_dir.parents[1] / 'phenobench_eval_stdout.txt'
        if not stdout.is_file():
            problems.append(f'{result_path} seed {seed}: missing {stdout}')
        cross = entry.get('metric_cross_check') or {}
        if cross.get('agrees') is not True:
            problems.append(f'{result_path} seed {seed}: self/official cross-check '
                            'did not agree')
    return problems


def write_prediction_manifest(result_path: Path) -> Optional[Path]:
    """Record every prediction file and its SHA for one probe result."""
    result = json.loads(result_path.read_text(encoding='utf-8'))
    entries = []
    for entry in result.get('seeds', []):
        pred_dir = Path(entry['predictions_dir'])
        if not pred_dir.is_absolute():
            pred_dir = ROOT / pred_dir
        files = sorted(pred_dir.glob('*.png'))
        entries.append({
            'seed': entry['seed'],
            'predictions_dir': str(pred_dir),
            'n_files': len(files),
            'files': [{'name': path.name, 'bytes': path.stat().st_size,
                       'sha256': sha256_file(path)} for path in files],
        })
    out = result_path.parent / 'prediction_manifest.json'
    out.write_text(json.dumps({'schema_version': 'stage6d_prediction_manifest/v1',
                               'result': str(result_path),
                               'seeds': entries}, indent=2,
                              ensure_ascii=False) + '\n', encoding='utf-8')
    return out


def build_comparison_manifest(new_records: Sequence[Dict[str, object]],
                              out_root: Path) -> Dict[str, object]:
    entries = [dict(item) for item in HISTORICAL]
    for record in new_records:
        entries.append({
            'arm': record['arm'], 'run_id': record['run_id'],
            'pair_id': record['pair_id'], 'seed': record['seed'],
            'traceable': True,
            'result': str(out_root / str(record['run_id']) / 'result.json'),
            'checkpoint': record['checkpoint'],
            'checkpoint_sha256': record['checkpoint_sha256'],
        })
    return {
        'schema_version': 'stage6d_comparison_manifest/v1',
        'phenobench_manifest_sha256': phenobench_manifest_sha(),
        'protocol_version': PROTOCOL_VERSION,
        'probe_seeds': list(PROBE_SEEDS),
        'entries': entries,
    }


def command_for(record: Dict[str, object], out_root: Path, cache_dir: Path,
                device: int) -> List[str]:
    checkpoint = str(record['checkpoint'])
    label = probe_label(str(record['run_id']), str(record['checkpoint_sha256']),
                        phenobench_manifest_sha())
    out_dir = out_root / str(record['run_id'])
    return [
        sys.executable, '-m', 'src.downstream.phenobench_probe',
        '--checkpoint', checkpoint,
        '--arm-label', label,
        '--continued-training', (
            '2000 optimizer updates; official I-JEPA objective; '
            + ('natural source sampling' if record['arm'] == 'B1'
               else 'physical-source capped no-backfill sampling')),
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
    parser.add_argument('--manifest-only', action='store_true',
                        help='validate existing results and write the comparison manifest')
    parser.add_argument('--device', type=int, default=1,
                        help='physical GPU index (project rule: 1)')
    parser.add_argument('--runs-json', type=Path,
                        default=ROOT / 'reports/stage6d_pretraining_runs.json')
    parser.add_argument('--out-root', type=Path,
                        default=Path('/data/agrisa_jepa_runs/stage6d/phenobench'))
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('/data/agrisa_jepa_runs/stage6d/phenobench_cache'))
    parser.add_argument('--manifest-out', type=Path,
                        default=ROOT / 'reports/stage6d_comparison_manifest.json')
    parser.add_argument('--min-free-mib', type=int, default=30000)
    parser.add_argument('--max-utilization-percent', type=int, default=5)
    args = parser.parse_args(argv)

    records = new_runs(args.runs_json)
    if not records and not args.manifest_only:
        print('no completed new runs in ' + str(args.runs_json), file=sys.stderr)
        print('(the historical Stage 6C results are still available as P0)', file=sys.stderr)

    if args.dry_run or (not args.execute and not args.manifest_only):
        print(json.dumps({
            'mode': 'dry-run', 'device': args.device,
            'new_runs': [record['run_id'] for record in records],
            'commands': [command_for(record, args.out_root, args.cache_dir,
                                     args.device)[0] for record in records],
        }, indent=2, ensure_ascii=False))
        return 0

    if args.execute:
        # GPU guard
        query = subprocess.check_output([
            'nvidia-smi', f'--id={args.device}',
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        used, total, util = [int(part.strip()) for part in query.split(',')]
        free = total - used
        if free < args.min_free_mib or util > args.max_utilization_percent:
            print(f'REFUSING: physical GPU{args.device} free={free} MiB '
                  f'util={util}%; another process is using it', file=sys.stderr)
            return 1
        for record in records:
            command, out_dir = command_for(record, args.out_root, args.cache_dir,
                                           args.device)
            out_dir.mkdir(parents=True, exist_ok=True)
            binding = {
                'schema_version': 'stage6d_probe_binding/v1',
                'run_id': record['run_id'], 'arm': record['arm'],
                'pair_id': record['pair_id'], 'seed': record['seed'],
                'checkpoint': record['checkpoint'],
                'checkpoint_sha256': record['checkpoint_sha256'],
                'phenobench_manifest_sha256': phenobench_manifest_sha(),
                'protocol_version': PROTOCOL_VERSION,
                'probe_seeds': list(PROBE_SEEDS),
                'cache_label': probe_label(str(record['run_id']),
                                           str(record['checkpoint_sha256']),
                                           phenobench_manifest_sha()),
                'command': command,
                'started_at': now_utc(),
            }
            (out_dir / 'stage6d_probe_binding.json').write_text(
                json.dumps(binding, indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8')
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = str(args.device)
            print(f'--- probe {record["run_id"]} on physical GPU{args.device} ---',
                  flush=True)
            with (out_dir / 'console.log').open('w', encoding='utf-8') as console:
                process = subprocess.run(command, cwd=ROOT, env=env,
                                         stdout=console, stderr=subprocess.STDOUT,
                                         text=True)
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
            write_prediction_manifest(out_dir / 'result.json')

    # validate historical results too (they are reused, not re-run)
    for item in HISTORICAL:
        if item['arm'] == 'A':
            continue
        problems = verify_probe_result(ROOT / item['result'],
                                       item['checkpoint_sha256'])
        if problems:
            print('HISTORICAL RESULT VERIFICATION FAILED:', file=sys.stderr)
            for problem in problems:
                print(f'  - {problem}', file=sys.stderr)
            return 1

    manifest = build_comparison_manifest(records, args.out_root)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {args.manifest_out.relative_to(ROOT)} with '
          f'{len(manifest["entries"])} entries')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
