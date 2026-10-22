#!/usr/bin/env python
"""Measure what the G3 circular shift preserves and what it destroys.

For a stratified sample of the frozen v1 training set this tool computes the C2
boundary prior and its (8, 8) circular roll, then reports:

* exact histogram preservation (the sorted value vectors must be identical),
* the per-image Pearson correlation with the unshifted map (how much alignment
  is left),
* the lag-1 autocorrelation of both maps (how much local continuity is kept),
* the artificial wrap-seam jump relative to interior jumps,
* the fraction of images whose shifted map is still accidentally well aligned
  (|r| > 0.5), which is the repetitive-texture failure mode the protocol asks
  about.

Usage:
    ./.venv/bin/python -m tools.audit_shifted_prior \
        --num-images 2000 --out reports/stage4b_shift_prior_stats.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from src.masks.agri_structure import (
    StructurePrior,
    circular_shift_patch_cells,
    lag1_autocorrelation,
    mean_grid_correlation,
)
from src.transforms import make_transforms

MANIFEST_SHA = 'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/frozen/agrifield40k-v1_train.csv')
    parser.add_argument('--image-cache',
                        default='/home/test/agrisa_cache/agrifield40k-v1-train')
    parser.add_argument('--num-images', type=int, default=2000)
    parser.add_argument('--shift', type=lambda s: tuple(int(x) for x in s.split(',')),
                        default=(8, 8))
    parser.add_argument('--augmentation-seed', type=int, default=20261019)
    parser.add_argument('--out', default='reports/stage4b_shift_prior_stats.json')
    args = parser.parse_args()

    manifest = Path(args.manifest)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if digest != MANIFEST_SHA:
        raise SystemExit(f'wrong manifest: {digest}')
    rows = list(csv.DictReader(manifest.open(newline='', encoding='utf-8')))
    if args.num_images > 0 and args.num_images < len(rows):
        # proportional source-stratified prefix sample (same rule as the mask
        # audit, so the shift statistics cover every agricultural source)
        import collections
        by_source = collections.defaultdict(list)
        for row in rows:
            by_source[row['source']].append(row)
        total = len(rows)
        picked = []
        counts = {s: int(args.num_images * len(v) / total)
                  for s, v in by_source.items()}
        remainder = args.num_images - sum(counts.values())
        for s in sorted(by_source, key=lambda s: -len(by_source[s])):
            if remainder <= 0:
                break
            counts[s] += 1
            remainder -= 1
        for s in sorted(by_source):
            picked.extend(by_source[s][:counts[s]])
        rows = picked

    index = json.loads((Path(args.image_cache) / 'index.json').read_text(encoding='utf-8'))
    if index['manifest_sha256'] != MANIFEST_SHA:
        raise SystemExit('image cache was built from a different manifest')
    cache = np.load(str(Path(args.image_cache) / 'images.uint8.npy'), mmap_mode='r')
    order = {sid: i for i, sid in enumerate(index['sample_ids'])}

    prior = StructurePrior(mode='boundary')
    transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0))
    rng = None
    records: List[Dict[str, float]] = []

    batch_size = 64
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        tensors = []
        for row in chunk:
            image = Image.fromarray(np.asarray(cache[order[row['sample_id']]]), mode='RGB')
            state = torch.get_rng_state()
            seed_digest = hashlib.sha256(
                f'{args.augmentation_seed}:0:{order[row["sample_id"]]}'.encode()).digest()
            torch.manual_seed(int.from_bytes(seed_digest[:8], 'big') % (2 ** 63 - 1))
            try:
                tensors.append(transform(image))
            finally:
                torch.set_rng_state(state)
        stacked = torch.stack(tensors)
        with torch.no_grad():
            original = prior.compute(stacked).map
            shifted, info = circular_shift_patch_cells(original, shift=args.shift)
        for i in range(original.shape[0]):
            records.append({
                'correlation': float(mean_grid_correlation(original[i][None],
                                                           shifted[i][None])),
                'lag1_original': lag1_autocorrelation(original[i][None]),
                'lag1_shifted': lag1_autocorrelation(shifted[i][None]),
                'value_multiset_preserved': float(info['value_multiset_preserved']),
                'interior_jump': float(info['interior_mean_abs_jump']),
                'seam_jump': float(info['seam_mean_abs_jump']),
                'source': chunk[i]['source'],
            })
        if (start // batch_size) % 8 == 0:
            print(f'  {len(records)}/{len(rows)}', flush=True)

    corr = np.array([r['correlation'] for r in records])
    lag_o = np.array([r['lag1_original'] for r in records])
    lag_s = np.array([r['lag1_shifted'] for r in records])
    seam = np.array([r['seam_jump'] for r in records])
    interior = np.array([r['interior_jump'] for r in records])

    by_source: Dict[str, Dict[str, float]] = {}
    for source in sorted(set(r['source'] for r in records)):
        mask = np.array([r['source'] == source for r in records])
        by_source[source] = {
            'n': int(mask.sum()),
            'mean_correlation': float(corr[mask].mean()),
            'mean_lag1_original': float(lag_o[mask].mean()),
            'mean_lag1_shifted': float(lag_s[mask].mean()),
            'mean_seam_to_interior_ratio': float(
                (seam[mask] / np.maximum(interior[mask], 1e-9)).mean()),
        }

    payload = {
        'manifest': str(manifest),
        'manifest_sha256': digest,
        'num_images': len(records),
        'shift': list(args.shift),
        'value_multiset_preserved_fraction': float(np.mean(
            [r['value_multiset_preserved'] for r in records])),
        'mean_pearson_correlation_with_original': float(corr.mean()),
        'median_pearson_correlation_with_original': float(np.median(corr)),
        'p10_correlation': float(np.percentile(corr, 10)),
        'p90_correlation': float(np.percentile(corr, 90)),
        'fraction_abs_correlation_above_0.5': float((np.abs(corr) > 0.5).mean()),
        'fraction_abs_correlation_above_0.8': float((np.abs(corr) > 0.8).mean()),
        'mean_lag1_autocorrelation_original': float(lag_o.mean()),
        'mean_lag1_autocorrelation_shifted': float(lag_s.mean()),
        'lag1_retention_ratio': float(lag_s.mean() / lag_o.mean()),
        'mean_interior_jump': float(interior.mean()),
        'mean_seam_jump': float(seam.mean()),
        'mean_seam_to_interior_ratio': float(
            (seam / np.maximum(interior, 1e-9)).mean()),
        'by_source': by_source,
        'note': ('the shift preserves the value histogram exactly and keeps most '
                 'lag-1 autocorrelation; it breaks alignment except on periodic '
                 'textures, where the residual correlation is reported rather '
                 'than hidden'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items()
                      if not isinstance(v, dict)}, indent=2))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
