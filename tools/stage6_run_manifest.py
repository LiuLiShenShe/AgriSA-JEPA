#!/usr/bin/env python
"""Write a run manifest (inputs + outputs + SHA-256) for a Stage 6 run.

Usage:
    ./.venv-uv/bin/python -m tools.stage6_run_manifest \
        --arm B1 --run-dir runs/stage6_b1_natural_2000 \
        --out reports/stage6_run_manifest_b1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> Dict[str, object]:
    return {
        'path': str(path),
        'bytes': path.stat().st_size,
        'sha256': sha256_file(path),
    }


def collect(directory: Path, patterns: List[str]) -> List[Dict[str, object]]:
    seen: Dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.is_file():
                seen[str(path)] = path
    return [describe(path) for path in seen.values()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    params = {}
    params_path = run_dir / 'params-ijepa.yaml'
    if params_path.is_file():
        import yaml
        params = yaml.safe_load(params_path.read_text(encoding='utf-8'))

    inputs: List[Dict[str, object]] = []
    for key in ('manifest',):
        value = (params.get('data') or {}).get(key)
        if value:
            path = ROOT / value if not Path(value).is_absolute() else Path(value)
            if path.is_file():
                inputs.append({'role': key, **describe(path)})
    for key in ('pretrained_checkpoint',):
        value = (params.get('meta') or {}).get(key)
        if value and Path(value).is_file():
            inputs.append({'role': key, **describe(Path(value))})
    sampling = (params.get('data') or {}).get('sampling') or {}
    if sampling.get('profile'):
        path = ROOT / sampling['profile']
        if path.is_file():
            inputs.append({'role': 'sampling_profile', **describe(path)})
    config = ROOT / 'configs' / f'{args.arm.lower()}_placeholder'
    # -- record the config actually dumped by the run
    if params_path.is_file():
        inputs.append({'role': 'config_dumped_by_run', **describe(params_path)})

    outputs = collect(run_dir, [
        'jepa-step*.pth.tar', 'jepa-encoder-step*.pth', 'jepa-latest.pth.tar',
        'jepa_r0.csv', 'training_verification.yaml', 'stage6_sampling.json',
        'source_exposure_ledger.csv', 'source_exposure_ledger.summary.json',
    ])

    payload = {
        'arm': args.arm,
        'run_dir': str(run_dir),
        'inputs': inputs,
        'outputs': outputs,
        'output_bytes_total': sum(item['bytes'] for item in outputs),
        'note': 'sha256 computed from the bytes on disk at manifest-build time',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    print(f'{args.arm}: {len(inputs)} inputs, {len(outputs)} outputs, '
          f'{payload["output_bytes_total"]/1e9:.1f} GB -> {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
