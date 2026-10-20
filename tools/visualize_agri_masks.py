#!/usr/bin/env python
"""Visualise and measure the agricultural mask strategies.

For a sample of real AgriField images this tool renders, side by side:

    1. the augmented RGB crop the encoder actually sees,
    2. the vegetation (ExG) saliency,
    3. the structure/boundary map,
    4. the official random multi-block mask (arm B),
    5. the vegetation-guided mask (C1),
    6. the boundary-guided mask (C2),
    7. the shuffled-prior mask (C3),

and computes the statistics that decide whether the guidance actually changed
anything:

    * mean vegetation / boundary score inside the target blocks,
    * number of context patches the encoder can see,
    * patches per target block and the total / duplicated target coverage,
    * where the target blocks land (row/column occupancy),
    * how often the sampler fell back to the uniform official placement,
    * the candidate validity rate and the generation cost per batch.

All four methods are given the SAME block size and aspect ratio for a given
image, so the comparison isolates the placement rule.

The tool is read-only with respect to the dataset and writes images plus a JSON
statistics file.

Usage:
    ./.venv/bin/python -m tools.visualize_agri_masks \
        --manifest data/frozen/agrifield40k-v1_train.csv \
        --root /data/datasets/AgriField-40K \
        --num-images 24 --out-dir runs/stage4a_masks
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from src.masks.agri_structure import (
    StructurePrior,
    denormalize,
    lag1_autocorrelation,
    patch_pool,
)
from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator
from src.transforms import make_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

METHODS = ('B', 'C1', 'C2', 'C3')
PRIOR_FOR = {'C1': 'vegetation', 'C2': 'boundary', 'C3': 'shuffled_boundary'}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def pick_diverse_samples(rows: Sequence[Dict[str, str]], n: int,
                         root: Path) -> List[Dict[str, str]]:
    """Pick images that cover the interesting visual regimes.

    Stratified by source, then ordered by how much vegetation the image has, so
    the sample contains both vegetation-rich and vegetation-poor crops instead
    of 24 look-alikes.
    """
    by_source: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_source[row['source']].append(row)
    picked: List[Dict[str, str]] = []
    sources = sorted(by_source)
    per_source = max(1, n // max(len(sources), 1))
    for source in sources:
        candidates = by_source[source][:max(per_source * 4, 8)]
        scored = []
        for row in candidates:
            path = root / row['path']
            try:
                with Image.open(path) as handle:
                    small = np.asarray(handle.convert('RGB').resize((64, 64)),
                                       dtype=np.float32) / 255.0
            except Exception:
                continue
            exg = 2 * small[..., 1] - small[..., 0] - small[..., 2]
            scored.append((float(exg.mean()), row))
        scored.sort(key=lambda t: t[0])
        # take the most and the least vegetated examples of this source
        take = scored[:per_source // 2] + scored[-(per_source - per_source // 2):]
        picked.extend(row for _, row in take)
    return picked[:n]


# --------------------------------------------------------------------------- #
# Mask generation with a shared block size
# --------------------------------------------------------------------------- #

def make_collators(patch_size: int, guidance_lambda: float,
                   num_candidates: int, base_seed: int) -> Dict[str, MaskCollator]:
    common = dict(
        input_size=224, patch_size=patch_size,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=False)
    collators: Dict[str, MaskCollator] = {'B': MaskCollator(**common)}
    for method, prior in PRIOR_FOR.items():
        collators[method] = StructureGuidedMaskCollator(
            **common, prior_mode=prior, guidance_lambda=guidance_lambda,
            num_candidates=num_candidates, base_seed=base_seed)
    return collators


def block_sizes(collator: MaskCollator, seed: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """The target / context block sizes the official rules produce for a seed."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    p_size = collator._sample_block_size(generator=generator,
                                         scale=collator.pred_mask_scale,
                                         aspect_ratio_scale=collator.aspect_ratio)
    e_size = collator._sample_block_size(generator=generator,
                                         scale=collator.enc_mask_scale,
                                         aspect_ratio_scale=(1., 1.))
    return p_size, e_size


def place_masks(collator: MaskCollator, batch, p_size, e_size):
    """Place the masks of one method with externally supplied block sizes.

    Mirrors the official flow: ``npred`` target blocks per image, then ``nenc``
    context blocks constrained to avoid the target cells, then the batch-wide
    common-length truncation.
    """
    B = len(batch)
    masks_p_all, masks_e_all = [], []
    min_keep_pred = collator.height * collator.width
    min_keep_enc = collator.height * collator.width
    for _ in range(B):
        masks_p, complements = [], []
        for _ in range(collator.npred):
            mask, complement = collator._sample_block_mask(p_size)
            masks_p.append(mask)
            complements.append(complement)
            min_keep_pred = min(min_keep_pred, len(mask))
        masks_p_all.append(masks_p)
        for _ in range(collator.nenc):
            mask, _ = collator._sample_block_mask(e_size,
                                                  acceptable_regions=complements)
            masks_e_all.append(mask)
            min_keep_enc = min(min_keep_enc, len(mask))
    masks_p_all = [[m[:min_keep_pred] for m in group] for group in masks_p_all]
    masks_e_all = [[m[:min_keep_enc]] for m in masks_e_all]
    return masks_p_all, masks_e_all


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def colorize(mask: np.ndarray) -> Image.Image:
    """Green = target block, blue = context-visible cells."""
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[..., 1] = (mask['target'] * 200).astype(np.uint8)
    canvas[..., 2] = (mask['context'] * 200).astype(np.uint8)
    return Image.fromarray(canvas)


def mask_overlay(rgb: Image.Image, mask: dict, grid: int = 16,
                 patch: int = 14) -> Image.Image:
    """Paint the 16x16 patch masks onto the 224x224 crop (nearest upsample)."""
    base = np.asarray(rgb).astype(np.float32)
    target = np.kron(mask['target'].astype(np.float32), np.ones((patch, patch)))[..., None]
    context = np.kron(mask['context'].astype(np.float32), np.ones((patch, patch)))[..., None]
    out = base.copy()
    out = out * (1 - 0.45 * target) + np.array([0, 220, 0]) * (0.45 * target)
    out = out * (1 - 0.35 * context) + np.array([0, 80, 255]) * (0.35 * context)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def heat_image(x: np.ndarray) -> Image.Image:
    """Grey -> orange heat rendering of a [0,1] map."""
    v = (x.clip(0, 1) * 255).astype(np.uint8)
    rgb = np.stack([v, (v * 0.55).astype(np.uint8), np.zeros_like(v)], axis=-1)
    return Image.fromarray(rgb)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def mask_stats(target_masks: Sequence[torch.Tensor], context_masks: Sequence[torch.Tensor],
               vegetation: np.ndarray, boundary: np.ndarray,
               grid: int = 16) -> Dict[str, float]:
    """Geometry and structure-coverage statistics for one image."""
    target_cells = np.zeros((grid, grid), dtype=bool)
    for m in target_masks:
        flat = np.asarray(m).reshape(-1)
        target_cells[flat // grid, flat % grid] = True
    context_cells = np.zeros((grid, grid), dtype=bool)
    for m in context_masks:
        flat = np.asarray(m).reshape(-1)
        context_cells[flat // grid, flat % grid] = True
    union = target_cells | context_cells
    per_block_areas = [len(m) for m in target_masks]
    total_target_area = int(sum(per_block_areas))
    duplicated = total_target_area - int(target_cells.sum())
    return {
        'total_target_area': total_target_area,
        'duplicated_target_area': duplicated,
        'target_cells': int(target_cells.sum()),
        'context_cells': int(context_cells.sum()),
        'target_context_overlap': int((target_cells & context_cells).sum()),
        'patches_per_target': float(np.mean([len(m) for m in target_masks])),
        'mean_vegetation_in_target': float(vegetation[target_cells].mean())
        if target_cells.any() else 0.0,
        'mean_boundary_in_target': float(boundary[target_cells].mean())
        if target_cells.any() else 0.0,
        'mean_vegetation_in_context': float(vegetation[context_cells].mean())
        if context_cells.any() else 0.0,
        'mean_vegetation_overall': float(vegetation.mean()),
        'coverage_fraction': float(union.mean()),
        'target_row_centroid': float(target_cells.mean(axis=1).argmax())
        if target_cells.any() else -1.0,
        'target_col_centroid': float(target_cells.mean(axis=0).argmax())
        if target_cells.any() else -1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',
                        default='data/frozen/agrifield40k-v1_train.csv')
    parser.add_argument('--root', default='/data/datasets/AgriField-40K')
    parser.add_argument('--num-images', type=int, default=24)
    parser.add_argument('--out-dir', default='runs/stage4a_masks')
    parser.add_argument('--guidance-lambda', type=float, default=6.0)
    parser.add_argument('--num-candidates', type=int, default=16)
    parser.add_argument('--base-seed', type=int, default=20261019)
    parser.add_argument('--augmentation-seed', type=int, default=20261019)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / 'figures').mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    rows = read_manifest(Path(args.manifest))
    picked = pick_diverse_samples(rows, args.num_images, root)
    print(f'sampled {len(picked)} images from {len(set(r["source"] for r in picked))} sources')

    transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0))
    collators = make_collators(14, args.guidance_lambda, args.num_candidates,
                              args.base_seed)
    prior_vegetation = StructurePrior(mode='vegetation')
    prior_boundary = StructurePrior(mode='boundary')
    prior_shuffled = StructurePrior(mode='shuffled_boundary')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    per_image: List[Dict[str, object]] = []
    aggregate: Dict[str, List[float]] = collections.defaultdict(list)
    timing: Dict[str, List[float]] = collections.defaultdict(list)

    for i, row in enumerate(picked):
        path = root / row['path']
        with Image.open(path) as handle:
            pil = handle.convert('RGB')
        # the crop the training pipeline actually sees
        state = torch.get_rng_state()
        torch.manual_seed(args.augmentation_seed + i)
        try:
            tensor = transform(pil)
        finally:
            torch.set_rng_state(state)
        batch = [(tensor, row['sample_id'])]

        with torch.no_grad():
            veg_result = prior_vegetation.compute(tensor.unsqueeze(0).to(device),
                                                   sample_ids=[row['sample_id']],
                                                   base_seed=args.base_seed)
            bnd_result = prior_boundary.compute(tensor.unsqueeze(0).to(device),
                                                sample_ids=[row['sample_id']],
                                                base_seed=args.base_seed)
            shf_result = prior_shuffled.compute(tensor.unsqueeze(0).to(device),
                                                sample_ids=[row['sample_id']],
                                                base_seed=args.base_seed)
        veg = veg_result.map[0].cpu().numpy()
        bnd = bnd_result.map[0].cpu().numpy()

        p_size, e_size = block_sizes(collators['B'], args.base_seed + i)
        method_masks: Dict[str, Dict[str, np.ndarray]] = {}
        for method, collator in collators.items():
            t0 = time.time()
            if method == 'B':
                masks_p, masks_e = place_masks(collator, batch, p_size, e_size)
            else:
                # the guided collator needs its per-batch state prepared, and
                # EACH METHOD MUST USE ITS OWN PRIOR
                own = {'C1': veg_result, 'C2': bnd_result, 'C3': shf_result}[method]
                collator._prior_map = own.map.cpu()
                collator._prior_usable = own.usable.cpu()
                collator._cand_gen = torch.Generator()
                collator._cand_gen.manual_seed(args.base_seed + i)
                collator._target_calls = 0
                collator._batch_size = 1
                masks_p, masks_e = place_masks(collator, batch, p_size, e_size)
                collator._prior_map = None
                collator._prior_usable = None
            timing[method].append(time.time() - t0)

            def cells_of(m):
                flat = np.asarray(m).reshape(-1)
                return (flat // 16, flat % 16)

            target = np.zeros((16, 16), dtype=bool)
            for m in masks_p[0]:
                rows_i, cols_i = cells_of(m)
                target[rows_i, cols_i] = True
            context = np.zeros((16, 16), dtype=bool)
            for m in masks_e[0]:
                rows_i, cols_i = cells_of(m)
                context[rows_i, cols_i] = True
            method_masks[method] = {'target': target, 'context': context}
            stats = mask_stats(masks_p[0], masks_e[0], veg, bnd)
            for key, value in stats.items():
                aggregate[f'{method}.{key}'].append(value)

        # -- figure
        rgb = denormalize(tensor.unsqueeze(0))[0].permute(1, 2, 0).numpy()
        rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
        panels = [rgb_img, heat_image(veg), heat_image(bnd)]
        for method in METHODS:
            panels.append(mask_overlay(rgb_img, method_masks[method]))
        width, height = rgb_img.size
        canvas = Image.new('RGB', (width * len(panels), height), (0, 0, 0))
        for j, panel in enumerate(panels):
            canvas.paste(panel.resize((width, height)), (j * width, 0))
        canvas = canvas.resize((width * len(panels) // 2, height // 2))
        canvas.save(out_dir / 'figures' / f'{i:03d}_{row["source"]}.png')

        per_image.append({
            'index': i,
            'sample_id': row['sample_id'],
            'source': row['source'],
            'original_id': row['original_id'],
            'vegetation_usable': bool(veg_result.usable[0]),
            'vegetation_lag1_autocorrelation': lag1_autocorrelation(veg_result.map),
            'boundary_lag1_autocorrelation': lag1_autocorrelation(bnd_result.map),
            'mean_vegetation': float(veg.mean()),
            'mean_boundary': float(bnd.mean()),
            'target_block_size': list(p_size),
            'context_block_size': list(e_size),
        })

    summary = {
        'n_images': len(per_image),
        'manifest': args.manifest,
        'guidance_lambda': args.guidance_lambda,
        'num_candidates': args.num_candidates,
        'base_seed': args.base_seed,
        'method_statistics': {
            method: {key[len(method) + 1:]: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            } for key, values in aggregate.items() if key.startswith(method + '.')}
            for method in METHODS
        },
        'generation_seconds_per_image': {
            method: float(np.mean(values)) for method, values in timing.items()
        },
        'per_image': per_image,
    }
    (out_dir / 'mask_statistics.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps(summary['method_statistics'], indent=2))
    print(json.dumps(summary['generation_seconds_per_image'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
