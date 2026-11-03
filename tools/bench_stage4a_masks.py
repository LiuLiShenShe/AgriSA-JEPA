#!/usr/bin/env python
"""Measure the mask-generation cost per batch for each arm, on the CPU only.

The wall-clock difference between the training arms is not a clean measurement
of the mask cost: arms C1/C3 and C2 ran concurrently on two GPUs, and the
evaluations shared host RAM and the NVMe bandwidth.  This tool times the mask
samplers in isolation, at the real batch size (16) and micro-batch geometry, so
the overhead reported in the benchmark is attributable to the sampler alone.

What it measures, per arm and per batch:

* ``official_seconds``  -- the parent ``MaskCollator`` alone,
* ``prior_seconds``     -- the structure prior computation,
* ``selection_seconds`` -- the candidate sampling and selection,
* ``total_seconds``     -- the whole collator call,
* ``images_per_second`` -- the batch rate the data pipeline would sustain.

The device is forced to CPU because the training loop is not running; the
samplers are pure Python/CPU work either way, so the relative comparison is
unchanged and the numbers are not polluted by another job.

Usage:
    ./.venv/bin/python -m tools.bench_stage4a_masks \
        --manifest data/frozen/agrifield40k-v1_train.csv \
        --root /data/datasets/AgriField-40K \
        --num-batches 20 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator
from src.masks.agri_structure import StructurePrior
from src.transforms import make_transforms


def collator_for(mode: str, prior_mode: str, lambda_: float, k: int):
    if mode == 'random':
        return MaskCollator(
            input_size=224,
            patch_size=14,
            enc_mask_scale=(0.85, 1.0),
            pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5),
            nenc=1, npred=4,
            min_keep=10,
            allow_overlap=False)
    return StructureGuidedMaskCollator(
        input_size=224,
        patch_size=14,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1, npred=4,
        min_keep=10,
        allow_overlap=False,
        guidance_lambda=lambda_,
        num_candidates=k,
        prior_mode=prior_mode,
        base_seed=0)


ARMS = [
    ('B', 'random', None),
    ('C1', 'structure_guided', 'vegetation'),
    ('C2', 'structure_guided', 'boundary'),
    ('C3', 'structure_guided', 'shuffled_boundary'),
]


def load_batch(paths: List[Path], ids: List[str], transform, seed: int):
    """A collator input identical in shape to what the DataLoader produces:
    a list of ``(tensor[C,H,W], sample_id)`` pairs."""
    items = []
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        for path, sample_id in zip(paths, ids):
            with Image.open(path) as handle:
                pil = handle.convert('RGB')
            items.append((transform(pil), sample_id))
    finally:
        torch.set_rng_state(state)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/frozen/agrifield40k-v1_train.csv')
    parser.add_argument('--root', default='/data/datasets/AgriField-40K')
    parser.add_argument('--num-batches', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--out', default='reports/stage4a_mask_benchmark.json')
    args = parser.parse_args()

    import csv as _csv
    with open(args.manifest, newline='', encoding='utf-8') as handle:
        rows = list(_csv.DictReader(handle))
    root = Path(args.root)
    paths = [root / r['path'] for r in rows]
    ids = [r['sample_id'] for r in rows]
    needed = (args.warmup + args.num_batches) * args.batch_size
    assert len(paths) >= needed, f'need {needed} samples, manifest has {len(paths)}'

    transform = make_transforms(
        crop_size=224, crop_scale=(0.3, 1.0),
        normalization=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    # one warmup per arm so lazy imports / cudnn autotuning never land in the
    # measured window
    for arm, mode, prior_mode in ARMS:
        collator = collator_for(mode, prior_mode, 6.0, 16)
        batch = load_batch(paths[:args.batch_size], ids[:args.batch_size],
                           transform, seed=0)
        collator(batch)

    results: Dict[str, Dict[str, float]] = {}
    for arm, mode, prior_mode in ARMS:
        collator = collator_for(mode, prior_mode, 6.0, 16)
        per_batch: List[float] = []
        prior: List[float] = []
        selection: List[float] = []
        for batch_index in range(args.warmup + args.num_batches):
            lo = batch_index * args.batch_size
            batch = load_batch(paths[lo:lo + args.batch_size],
                               ids[lo:lo + args.batch_size], transform, seed=0)
            collator._itr_counter.value = batch_index
            before = getattr(collator, 'diagnostics', None) or {}
            before_prior = before.get('prior_seconds', 0.0)
            before_selection = before.get('selection_seconds', 0.0)
            t0 = time.perf_counter()
            collator(batch)
            elapsed = time.perf_counter() - t0
            if batch_index >= args.warmup:
                per_batch.append(elapsed)
                # the collator's counters are cumulative, so take the delta
                diagnostics = getattr(collator, 'diagnostics', None) or {}
                if diagnostics:
                    prior.append(diagnostics.get('prior_seconds', 0.0) - before_prior)
                    selection.append(
                        diagnostics.get('selection_seconds', 0.0) - before_selection)
        mean = float(np.mean(per_batch))
        results[arm] = {
            'mask_mode': mode,
            'prior_mode': prior_mode,
            'batch_size': args.batch_size,
            'batches_measured': len(per_batch),
            'batch_seconds_mean': round(mean, 5),
            'batch_seconds_std': round(float(np.std(per_batch)), 5),
            'batch_seconds_p50': round(float(np.median(per_batch)), 5),
            'images_per_second': round(args.batch_size / mean, 1),
            'prior_seconds_mean': round(float(np.mean(prior)), 5) if prior else None,
            'selection_seconds_mean': (round(float(np.mean(selection)), 5)
                                       if selection else None),
            'micro_step_overhead_ms': (round(1000.0 * float(np.mean(prior + selection)), 3)
                                       if prior else None),
        }
        print(f'{arm}: {mean * 1000:.2f} ms/batch  '
              f'({args.batch_size / mean:.0f} img/s)', flush=True)

    baseline = results['B']['batch_seconds_mean']
    for arm in results:
        results[arm]['overhead_vs_B_ms'] = round(
            1000.0 * (results[arm]['batch_seconds_mean'] - baseline), 3)
        results[arm]['overhead_vs_B_percent'] = round(
            100.0 * (results[arm]['batch_seconds_mean'] - baseline) / baseline, 2)

    payload = {
        'manifest': args.manifest,
        'batch_size': args.batch_size,
        'batches': args.num_batches,
        'warmup_batches': args.warmup,
        'device': 'cpu',
        'guidance_lambda': 6.0,
        'num_candidates': 16,
        'note': ('timed in isolation; the wall-clock difference between the '
                 'training arms additionally contains two arms sharing two GPUs '
                 'and the evaluations sharing host RAM, so it is not attributed '
                 'to the mask sampler.'),
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
