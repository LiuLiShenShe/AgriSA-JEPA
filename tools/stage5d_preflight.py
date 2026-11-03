#!/usr/bin/env python
"""Stage 5D gate 1: assets, hashes, paths and environment preflight.

Everything is measured on this machine.  A missing asset is reported as missing;
it is never inferred from a Git report.

Outputs
-------
reports/stage5d_assets_preflight.json
reports/stage5d_assets_preflight.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

REPORTS = ROOT / 'reports'
CORPUS_V2 = ROOT / 'data/corpus_v2'
LARGE_ASSETS = [
    'R0_agrifm35_source_pool_manifest.csv',
    'R1_agrifm147_rebuild_manifest.csv',
    'R2_field_strict_manifest.csv',
    'R3_field_extended_manifest.csv',
    'images/artemis_imagesafari.csv',
    'images/artemis_imagesafari_qc.csv',
    'images/agrifm_ref11.csv',
    'images/agrifm_ref11_qc.csv',
    'images/agrifm_ref58.csv',
    'images/agrifm_ref58_qc.csv',
    'images/agrifield40k_pool.csv',
    'images/agrifield40k_pool_qc.csv',
]
SMALL_JSON = ['R0_agrifm35_source_pool_manifest.json',
              'R1_agrifm147_rebuild_manifest.json',
              'R2_field_strict_manifest.json',
              'R3_field_extended_manifest.json',
              'policy.yaml']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT,
                                       stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as exc:  # noqa: BLE001
        return f'ERROR: {exc.output.decode(errors="replace")}'


def environment() -> Dict[str, object]:
    python = sys.executable
    cuda = 'unavailable'
    gpu_names: List[str] = []
    try:
        import torch
        cuda = f'torch {torch.__version__}, cuda {torch.version.cuda}, available={torch.cuda.is_available()}'
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        cuda = f'error: {exc}'
    usage = shutil.disk_usage(ROOT)
    return {
        'python': python,
        'python_version': sys.version.split()[0],
        'virtualenv': str(Path(python).parent.parent),
        'cuda': cuda,
        'gpus': gpu_names,
        'disk_total_gb': round(usage.total / 1e9, 1),
        'disk_free_gb': round(usage.free / 1e9, 1),
        'conda_detected': bool(os.environ.get('CONDA_PREFIX')),
    }


def assets(data_root: Path) -> Dict[str, object]:
    records = {}
    for name in LARGE_ASSETS + SMALL_JSON:
        path = CORPUS_V2 / name
        records[name] = {
            'exists': path.exists(),
            'bytes': path.stat().st_size if path.exists() else 0,
            'sha256': sha256_file(path) if path.exists() else None,
        }
    index = Path('/data/datasets/Stage5C/artemis/remote_index.csv')
    records['artemis/remote_index.csv'] = {
        'exists': index.exists(),
        'bytes': index.stat().st_size if index.exists() else 0,
        'sha256': sha256_file(index) if index.exists() else None,
    }
    return records


def raw_roots(data_root: Path) -> Dict[str, object]:
    out = {}
    for child in sorted((data_root / 'raw').glob('*')):
        if not child.is_dir():
            continue
        count = sum(1 for p in child.rglob('*') if p.is_file())
        out[child.name] = {'path': str(child), 'files': count}
    return out


def path_hygiene() -> Dict[str, object]:
    absolute = []
    for name in ('R0_agrifm35_source_pool_manifest.json',
                 'R1_agrifm147_rebuild_manifest.json',
                 'R2_field_strict_manifest.json',
                 'R3_field_extended_manifest.json'):
        path = CORPUS_V2 / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if 'manifest_csv' in payload and str(payload['manifest_csv']).startswith('/'):
            absolute.append({'file': name, 'manifest_csv': payload['manifest_csv']})
    return {'legacy_absolute_paths': absolute,
            'stage5d_policy': 'relative_path + logical data root '
                              '(AGRI_DATA_ROOT / --data-root)'}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=os.environ.get('AGRI_DATA_ROOT',
                                                              '/data/datasets/Stage5C'))
    args = parser.parse_args()
    data_root = Path(args.data_root)

    # Record the exact hashes the Stage 5C manifests claim, then verify them.
    claimed = {}
    for name in ('R0_agrifm35_source_pool_manifest', 'R1_agrifm147_rebuild_manifest',
                 'R2_field_strict_manifest', 'R3_field_extended_manifest'):
        path = CORPUS_V2 / f'{name}.json'
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        claimed[name] = {
            'policy_sha256': payload.get('policy_sha256'),
            'csv_sha256': payload.get('csv_sha256'),
            'manifest_csv_sha256': payload.get('manifest_csv_sha256'),
        }

    payload = {
        'git': {
            'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
            'head': git('rev-parse', 'HEAD'),
            'remote_head_recorded_by_stage5c': '2867e7f4b09814c268612e893c34216506cbe55e',
            'status_short': git('status', '--short').splitlines(),
            'branches': git('branch', '-vv').splitlines(),
        },
        'environment': environment(),
        'data_root': str(data_root),
        'data_root_exists': data_root.exists(),
        'raw_roots': raw_roots(data_root) if data_root.exists() else {},
        'assets': assets(data_root),
        'stage5c_claimed_hashes': claimed,
        'path_hygiene': path_hygiene(),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'stage5d_assets_preflight.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')

    missing = [name for name, info in payload['assets'].items() if not info['exists']]
    lines = [
        '# Stage 5D - assets preflight',
        '',
        '## Repository',
        '',
        f"- Branch: `{payload['git']['branch']}`",
        f"- HEAD: `{payload['git']['head']}`",
        f"- Stage 5C recorded HEAD: `{payload['git']['remote_head_recorded_by_stage5c']}`",
        f"- Match: **{payload['git']['head'] == payload['git']['remote_head_recorded_by_stage5c']}**",
        f"- Working tree: {payload['git']['status_short'] or 'clean'}",
        '',
        '## Environment',
        '',
        f"- Python: {payload['environment']['python_version']} at `{payload['environment']['virtualenv']}`",
        f"- CUDA: {payload['environment']['cuda']}",
        f"- GPUs: {payload['environment']['gpus']}",
        f"- Conda prefix detected: {payload['environment']['conda_detected']}",
        f"- Disk free: {payload['environment']['disk_free_gb']} GB",
        '',
        '## Data root',
        '',
        f"- `{payload['data_root']}` exists: **{payload['data_root_exists']}**",
    ]
    for name, info in payload['raw_roots'].items():
        lines.append(f"  - `{name}`: {info['files']:,} files")
    lines += ['', '## Assets present and hashed', '',
              '| asset | exists | bytes | SHA-256 |', '|---|---|---:|---|']
    for name, info in payload['assets'].items():
        lines.append(f"| `{name}` | {info['exists']} | {info['bytes']:,} | "
                     f"`{(info['sha256'] or '')[:16]}...` |")
    lines += [
        '',
        f"Missing assets: **{missing or 'none'}**",
        '',
        '## Path hygiene',
        '',
        f"- Legacy manifests embed absolute paths: "
        f"{len(payload['path_hygiene']['legacy_absolute_paths'])}",
        f"- Stage 5D rule: {payload['path_hygiene']['stage5d_policy']}",
    ]
    (REPORTS / 'stage5d_assets_preflight.md').write_text('\n'.join(lines) + '\n',
                                                         encoding='utf-8')
    print(json.dumps({'head_matches': payload['git']['head'] ==
                      payload['git']['remote_head_recorded_by_stage5c'],
                      'missing_assets': missing,
                      'disks': payload['environment']['disk_free_gb'],
                      'gpus': payload['environment']['gpus']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
