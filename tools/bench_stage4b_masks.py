#!/usr/bin/env python
"""Measure the Stage 4B mask samplers' cost and sampler health.

For every arm this tool reports, on real augmented crops at the training batch
size (16):

* wall-clock milliseconds per batch for the WHOLE collator call (prior
  computation included, unlike ``bench_stage4a_masks`` which timed placement
  only),
* the prior and selection components when the arm has a prior,
* images/second, and the overhead against arm B,
* guided selections, uniform fallbacks, invalid candidates and the effective
  fallback rate, so a sampler that silently degenerates to random is visible,
* the mean selected / candidate / expected overlap, which checks that the
  geometry penalty is active.

Usage:
    ./.venv/bin/python -m tools.bench_stage4b_masks \
        --num-batches 10 --device cpu --out reports/stage4b_mask_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator
from src.transforms import make_transforms

COLLATOR_KWARGS = dict(
    input_size=224, patch_size=14,
    enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
    allow_overlap=False)

ARMS = {
    'B': {'mode': 'random'},
    'U': {'mode': 'structure_guided', 'prior_mode': None,
          'score_mode': 'uniform', 'guidance_lambda': 0.0,
          'geometry_beta': 0.0, 'force_candidates': True},
    'G0': {'mode': 'geometry_controlled', 'prior_mode': None,
           'score_mode': 'uniform', 'guidance_lambda': 0.0,
           'geometry_beta': 3.5, 'geometry_control': 'linear'},
    'G1': {'mode': 'geometry_controlled', 'prior_mode': 'vegetation',
           'score_mode': 'prior', 'guidance_lambda': 6.0,
           'geometry_beta': 3.5, 'geometry_control': 'linear'},
    'G2': {'mode': 'geometry_controlled', 'prior_mode': 'boundary',
           'score_mode': 'prior', 'guidance_lambda': 6.0,
           'geometry_beta': 3.5, 'geometry_control': 'linear'},
    'G3': {'mode': 'geometry_controlled', 'prior_mode': 'shifted_boundary',
           'score_mode': 'prior', 'guidance_lambda': 6.0,
           'geometry_beta': 3.5, 'geometry_control': 'linear',
           'roll_shift': (8, 8)},
}


def make_collator(spec: dict, device) -> MaskCollator:
    spec = dict(spec)
    mode = spec.pop('mode')
    if mode == 'random':
        return MaskCollator(**COLLATOR_KWARGS)
    spec.setdefault('num_candidates', 16)
    spec.setdefault('base_seed', 20261019)
    spec.setdefault('geometry_beta', 3.5)
    spec.setdefault('geometry_control', 'linear')
    if device.type == 'cuda':
        spec['prior_device'] = device
    return StructureGuidedMaskCollator(**COLLATOR_KWARGS, **spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/frozen/agrifield40k-v1_train.csv')
    parser.add_argument('--image-cache',
                        default='/home/test/agrisa_cache/agrifield40k-v1-train')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--augmentation-seed', type=int, default=20261019)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='must match the training config: with workers the '
                             'collator runs in the worker, so prior_device stays CPU')
    parser.add_argument('--out', default='reports/stage4b_mask_benchmark.json')
    args = parser.parse_args()

    manifest = Path(args.manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != 'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435':
        raise SystemExit(f'wrong manifest: {digest}')
    rows = list(csv.DictReader(manifest.open(newline='', encoding='utf-8')))
    index = json.loads((Path(args.image_cache) / 'index.json').read_text(encoding='utf-8'))
    cache = np.load(str(Path(args.image_cache) / 'images.uint8.npy'), mmap_mode='r')
    order = {sid: i for i, sid in enumerate(index['sample_ids'])}
    transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0))
    device = torch.device(args.device)

    needed = (args.warmup + args.num_batches) * args.batch_size
    if len(rows) < needed:
        raise SystemExit(f'need {needed} rows, manifest has {len(rows)}')

    def batch_at(offset: int):
        items = []
        for k in range(args.batch_size):
            row = rows[offset + k]
            dataset_index = order[row['sample_id']]
            image = Image.fromarray(np.asarray(cache[dataset_index]), mode='RGB')
            seed_digest = hashlib.sha256(
                f'{args.augmentation_seed}:0:{dataset_index}'.encode()).digest()
            state = torch.get_rng_state()
            torch.manual_seed(int.from_bytes(seed_digest[:8], 'big') % (2 ** 63 - 1))
            try:
                tensor = transform(image)
            finally:
                torch.set_rng_state(state)
            items.append((tensor, row['sample_id']))
        return items

    results: Dict[str, dict] = {}
    for arm, spec in ARMS.items():
        if args.num_workers > 0 and spec.get('prior_device') == device:
            spec = dict(spec)
            spec.pop('prior_device', None)
        collator = make_collator(spec, device)
        timings: List[float] = []
        for b in range(args.warmup + args.num_batches):
            batch = batch_at(b * args.batch_size)
            collator._itr_counter.value = b
            t0 = time.perf_counter()
            collator(batch)
            elapsed = time.perf_counter() - t0
            if b >= args.warmup:
                timings.append(elapsed)
        mean = float(np.mean(timings))
        diagnostics = (collator.diagnostics_summary()
                       if hasattr(collator, 'diagnostics_summary')
                       else {'mask_mode': 'random'})
        selections = int(diagnostics.get('guided_selections', 0))
        fallbacks = int(diagnostics.get('uniform_fallbacks', 0))
        results[arm] = {
            'spec': {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in spec.items()},
            'batch_seconds_mean': round(mean, 6),
            'batch_seconds_std': round(float(np.std(timings)), 6),
            'batch_seconds_p50': round(float(np.median(timings)), 6),
            'images_per_second': round(args.batch_size / mean, 1),
            'prior_seconds_per_batch': round(
                float(diagnostics.get('prior_seconds', 0.0))
                / (args.warmup + args.num_batches), 6),
            'selection_seconds_per_batch': round(
                float(diagnostics.get('selection_seconds', 0.0))
                / (args.warmup + args.num_batches), 6),
            'guided_selections': selections,
            'uniform_fallbacks': fallbacks,
            'fallback_rate': ((fallbacks / (selections + fallbacks)
                               if selections + fallbacks else 1.0)
                              if 'guided_selections' in diagnostics else None),
            'invalid_candidate_rate': float(
                diagnostics.get('invalid_candidate_rate', 0.0)),
            'mean_selected_overlap': diagnostics.get('mean_selected_overlap'),
            'mean_candidate_overlap': diagnostics.get('mean_candidate_overlap'),
            'mean_expected_overlap': diagnostics.get('mean_expected_overlap'),
        }
        fallback = results[arm]['fallback_rate']
        fallback_text = 'n/a' if fallback is None else f'{fallback:.3f}'
        print(f'{arm}: {mean * 1000:8.2f} ms/batch  '
              f'({args.batch_size / mean:7.0f} img/s)  '
              f'fallback={fallback_text}  '
              f'invalid={results[arm]["invalid_candidate_rate"]:.3f}', flush=True)

    baseline = results['B']['batch_seconds_mean']
    for arm in results:
        results[arm]['overhead_vs_B_ms'] = round(
            1000.0 * (results[arm]['batch_seconds_mean'] - baseline), 3)
        results[arm]['overhead_vs_B_percent'] = round(
            100.0 * (results[arm]['batch_seconds_mean'] - baseline) / baseline, 2)

    payload = {
        'manifest': str(manifest),
        'manifest_sha256': digest,
        'batch_size': args.batch_size,
        'batches': args.num_batches,
        'warmup_batches': args.warmup,
        'device': args.device,
        'num_workers': args.num_workers,
        'note': ('whole-collator timing on real augmented crops; prior work is '
                 'included. With num_workers>0 the collator lives in a worker '
                 'process and the prior runs on CPU there.'),
        'arms': results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
