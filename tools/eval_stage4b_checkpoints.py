#!/usr/bin/env python
"""Evaluate the Stage 4B G0-G3 checkpoints on PhenoBench.

The evaluation protocol is imported from the Stage 3A/4A driver
(``tools.eval_stage4a_checkpoints``): the same tiling, the same
``Linear(1280 -> 3)`` probe, the same optimizer/LR/epochs, the same probe seeds
0/1/2 and the same official ``phenobench-eval --task semantics`` call.  Feature
caches are bound to the checkpoint SHA-256, so no arm can silently reuse another
arm's features.

The reference evaluation points B_500, C1_500 and C2_500 are NOT recomputed;
they are the already-verified Stage 4A results and are linked into the Stage 4B
results directory by ``tools.prepare_stage4b_phenobench``.

Usage:
    ./.venv/bin/python -m tools.eval_stage4b_checkpoints --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from tools.eval_stage4a_checkpoints import (
    evaluate_one,
    extract_snapshot,
    protocol_from_config,
    torch_device,
)

#: arm -> (run directory under runs_root, config file)
ARMS: Dict[str, Tuple[str, str]] = {
    'G0': ('stage4b_geometry_only_500', 'configs/stage4b_geometry_only.yaml'),
    'G1': ('stage4b_vegetation_controlled_500',
           'configs/stage4b_vegetation_controlled.yaml'),
    'G2': ('stage4b_boundary_controlled_500',
           'configs/stage4b_boundary_controlled.yaml'),
    'G3': ('stage4b_shifted_boundary_500',
           'configs/stage4b_shifted_boundary.yaml'),
}

#: every evaluation point this round reads.  B_500/C1_500/C2_500 come from the
#: Stage 4A results and are linked, not recomputed.
LABELS = {
    'B_500': ('B', 500),
    'C1_500': ('C1', 500),
    'C2_500': ('C2', 500),
    'G0_500': ('G0', 500),
    'G1_500': ('G1', 500),
    'G2_500': ('G2', 500),
    'G3_500': ('G3', 500),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', default='/home/test/agrisa_runs')
    parser.add_argument('--root', default='/data/datasets/PhenoBench/PhenoBench-v110')
    parser.add_argument('--cache-dir', default='runs/stage4b_phenobench/cache')
    parser.add_argument('--out-dir', default='runs/stage4b_phenobench/results')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--only', default=None)
    parser.add_argument('--keep-cache', action='store_true')
    parser.add_argument('--phenobench-cache',
                        default='/home/test/agrisa_cache/phenobench-v110')
    args = parser.parse_args()

    if args.phenobench_cache:
        os.environ['AGRISA_PHENOBENCH_CACHE'] = str(Path(args.phenobench_cache).resolve())

    runs_root = Path(args.runs_root)
    root = Path(args.root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = protocol_from_config('configs/phenobench_armA_linear.yaml')
    print(f'evaluation protocol (identical for every arm): {json.dumps(protocol)}')

    wanted = set(args.only.split(',')) if args.only else None
    summary: List[Dict[str, object]] = []

    for arm, (run_dir, config) in ARMS.items():
        cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        step = int(cfg['optimization']['interrupt_after_optimizer_steps'])
        label = f'{arm}_{step}'
        if wanted is not None and label not in wanted:
            continue
        if not (runs_root / run_dir).is_dir():
            print(f'[{label}] run directory missing, skipping', flush=True)
            continue
        t0 = time.time()
        snapshot = extract_snapshot(runs_root, run_dir, step)
        payload = evaluate_one(
            label, snapshot, root, cache_dir, out_dir, args.device, protocol,
            (f"{cfg['optimization']['max_optimizer_steps']} optimizer updates "
             f"(interrupted at {step}), mask={cfg['mask'].get('mode')}/"
             f"{cfg['mask'].get('prior_mode')} beta={cfg['mask'].get('geometry_beta')}"),
            (f"agrifield40k-v1 (PROVISIONAL), {step} updates of a "
             f"{cfg['optimization']['max_optimizer_steps']}-update schedule"),
            args.keep_cache, args.feature_batch_size)
        summary.append({
            'label': label, 'arm': arm, 'optimizer_steps': step,
            'checkpoint': str(snapshot),
            'checkpoint_sha256': payload['backbone_checkpoint_sha256'],
            'aggregate': payload.get('aggregate'),
            'seconds': round(time.time() - t0, 1),
        })

    index = out_dir / 'index.json'
    merged: List[Dict[str, object]] = []
    if index.is_file():
        try:
            previous = json.loads(index.read_text(encoding='utf-8'))
            if isinstance(previous, list):
                merged = [entry for entry in previous
                          if isinstance(entry, dict) and 'label' in entry]
        except json.JSONDecodeError:
            merged = []
    labels = {entry['label'] for entry in merged}
    for entry in summary:
        if entry['label'] in labels:
            merged = [e for e in merged if e['label'] != entry['label']]
        merged.append(entry)
    merged.sort(key=lambda e: str(e['label']))
    index.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + '\n',
                     encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
