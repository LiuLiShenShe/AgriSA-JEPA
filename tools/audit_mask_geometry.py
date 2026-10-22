#!/usr/bin/env python
"""Stage 4B-1 / 4B-2: full mask-geometry audit on real agricultural images.

This tool runs the *final* collator masks over a stratified sample of the frozen
v1 training manifest and reports, for every arm and every image:

* target geometry -- block area, deduplicated union, overlap multiplicity,
  block sizes, spatial spread, pre/post batch-truncation context length;
* context geometry -- visible patch count, fraction of the grid, context/target
  intersection;
* structure coverage -- how much vegetation / boundary prior mass the target
  union actually captures, and how much of that is merely *repeated* mass on
  cells that several target blocks predict.

The distinction that matters is ``mean_<prior>_target_union`` (independent
predicted cells) versus ``mean_<prior>_target_multiset`` (the average over the
4 blocks, counting a doubly predicted cell twice).  Stage 4A only reported the
former, so "each block contains more vegetation" and "the model predicts more
distinct vegetation" were conflated.

Every arm is given the SAME block sizes and the SAME augmented crop per image,
so the comparison isolates the placement rule.  The crop is generated exactly
as the training pipeline does (RandomResizedCrop(224, scale=(0.3, 1.0)) with
the per-sample derived seed from ``augmentation_seed``), which is what makes
the prior aligned with what the encoder sees.

Usage:
    ./.venv/bin/python -m tools.audit_mask_geometry \
        --manifest data/frozen/agrifield40k-v1_train.csv \
        --root /data/datasets/AgriField-40K \
        --image-cache /home/test/agrisa_cache/agrifield40k-v1-train \
        --num-images 1000 --out runs/stage4b_mask_audit/audit_1000.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from src.masks.agri_structure import (
    PATCH_GRID,
    StructurePrior,
    denormalize,
    lag1_autocorrelation,
)
from src.masks.geometry import (
    bootstrap_ci,
    clustered_bootstrap_ci,
    paired_delta,
    summarize,
    vegetation_bins,
)
from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator
from src.transforms import make_transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GRID = 16

#: arm -> (mask mode kwargs, human label).  The geometry-control kwargs are
#: injected by the caller so a beta sweep reuses this table verbatim.
BASE_ARMS: Dict[str, dict] = {
    'B': {'mode': 'random', 'label': 'official random multi-block'},
    'C1': {'mode': 'structure_guided', 'prior_mode': 'vegetation',
           'guidance_lambda': 6.0, 'geometry_beta': 0.0,
           'label': 'Stage4A vegetation guidance'},
    'C2': {'mode': 'structure_guided', 'prior_mode': 'boundary',
           'guidance_lambda': 6.0, 'geometry_beta': 0.0,
           'label': 'Stage4A boundary guidance'},
    'C3': {'mode': 'structure_guided', 'prior_mode': 'shuffled_boundary',
           'guidance_lambda': 6.0, 'geometry_beta': 0.0,
           'label': 'Stage4A shuffled boundary (patch permutation)'},
    'U': {'mode': 'structure_guided', 'prior_mode': None, 'score_mode': 'uniform',
          'guidance_lambda': 0.0, 'geometry_beta': 0.0, 'force_candidates': True,
          'label': 'uniform candidate sampling (lambda=0, beta=0)'},
    'G0': {'mode': 'geometry_controlled', 'prior_mode': None, 'score_mode': 'uniform',
           'guidance_lambda': 0.0, 'label': 'geometry control, no structure prior'},
    'G1': {'mode': 'geometry_controlled', 'prior_mode': 'vegetation',
           'guidance_lambda': 6.0, 'label': 'geometry-controlled vegetation'},
    'G2': {'mode': 'geometry_controlled', 'prior_mode': 'boundary',
           'guidance_lambda': 6.0, 'label': 'geometry-controlled boundary'},
    'G3': {'mode': 'geometry_controlled', 'prior_mode': 'shifted_boundary',
           'guidance_lambda': 6.0, 'label': 'geometry-controlled circularly shifted boundary'},
}

COLLATOR_KWARGS = dict(
    input_size=224, patch_size=14,
    enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
    allow_overlap=False)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def stratified_sample(rows: Sequence[Dict[str, str]], n: int,
                      seed: int = 20261022) -> List[Dict[str, str]]:
    """Deterministic, source-stratified sample of the manifest.

    ``n <= 0`` or ``n >= len(rows)`` selects the whole training set.  Otherwise
    each source contributes proportionally to its share of the corpus, with the
    largest-remainder rule and a hard cap at the source size; if small sources
    cap out, the shortfall is redistributed to the sources that still have
    rows.  Within a source the first rows in manifest order are taken, which is
    a stable rule applied identically to every arm.
    """
    if n <= 0 or n >= len(rows):
        return list(rows)
    by_source: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_source[row['source']].append(row)
    sources = sorted(by_source)
    total = len(rows)
    quotas = {s: n * len(by_source[s]) / total for s in sources}
    counts = {s: min(int(quotas[s]), len(by_source[s])) for s in sources}
    # largest remainder first, skipping capped sources
    order = sorted(sources, key=lambda s: (-(quotas[s] - int(quotas[s])), s))
    remaining = n - sum(counts.values())
    while remaining > 0:
        progressed = False
        for s in order:
            if remaining <= 0:
                break
            if counts[s] < len(by_source[s]):
                counts[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    picked: List[Dict[str, str]] = []
    for s in sources:
        picked.extend(by_source[s][:counts[s]])
    return picked


def make_collator(arm: str, geometry_beta: float, num_candidates: int,
                  guidance_lambda: Optional[float] = None,
                  roll_shift: Tuple[int, int] = (8, 8),
                  geometry_control: str = 'deviation') -> MaskCollator:
    spec = dict(BASE_ARMS[arm])
    mode = spec.pop('mode')
    spec.pop('label', None)
    if guidance_lambda is not None:
        spec['guidance_lambda'] = guidance_lambda
    kwargs = dict(COLLATOR_KWARGS)
    if mode == 'random':
        return MaskCollator(**kwargs)
    spec.setdefault('geometry_beta', geometry_beta)
    spec['geometry_control'] = geometry_control
    spec['num_candidates'] = num_candidates
    spec['base_seed'] = 20261022
    spec['roll_shift'] = roll_shift
    return StructureGuidedMaskCollator(**kwargs, **spec)


def augmented_crop(pil: Image.Image, augmentation_seed: int, epoch: int,
                   index: int) -> torch.Tensor:
    """Exactly the crop the training pipeline hands the collator."""
    digest = hashlib.sha256(
        f'{augmentation_seed}:{epoch}:{index}'.encode('utf-8')).digest()
    derived = int.from_bytes(digest[:8], 'big') % (2 ** 63 - 1)
    state = torch.get_rng_state()
    torch.manual_seed(derived)
    try:
        transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0))
        return transform(pil)
    finally:
        torch.set_rng_state(state)


def place_masks(collator, batch, p_size, e_size):
    """Place the masks of one arm with externally supplied block sizes.

    Mirrors the official flow: ``npred`` target blocks per image, then ``nenc``
    context blocks constrained to avoid the target cells, then the batch-wide
    common-length truncation.  Returns the masks *and* the pre-truncation
    context lengths so the audit can tell the two stages apart.
    """
    B = len(batch)
    masks_p_all, masks_e_all, pre_trunc = [], [], []
    min_keep_pred = GRID * GRID
    min_keep_enc = GRID * GRID
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
            pre_trunc.append(len(mask))
            min_keep_enc = min(min_keep_enc, len(mask))
    masks_p_all = [[m[:min_keep_pred] for m in group] for group in masks_p_all]
    masks_e_all = [[m[:min_keep_enc]] for m in masks_e_all]
    return masks_p_all, masks_e_all, pre_trunc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/frozen/agrifield40k-v1_train.csv')
    parser.add_argument('--root', default='/data/datasets/AgriField-40K')
    parser.add_argument('--image-cache',
                        default='/home/test/agrisa_cache/agrifield40k-v1-train')
    parser.add_argument('--num-images', type=int, default=1000)
    parser.add_argument('--arms', default='B,C1,C2,C3,U,G0,G1,G2,G3')
    parser.add_argument('--geometry-beta', type=float, default=2.0)
    parser.add_argument('--geometry-control', choices=('linear', 'deviation'),
                        default='deviation')
    parser.add_argument('--num-candidates', type=int, default=16)
    parser.add_argument('--guidance-lambda', type=float, default=6.0)
    parser.add_argument('--roll-shift', type=lambda s: tuple(int(x) for x in s.split(',')),
                        default=(8, 8))
    parser.add_argument('--augmentation-seed', type=int, default=20261019)
    parser.add_argument('--shuffle-seed', type=int, default=20261019,
                        help='shuffle the selected images before batching (mixes '
                             'sources across a batch, as the training sampler '
                             'does); set <0 to keep manifest order')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--per-image-format', choices=('json', 'csv', 'none'),
                        default='json',
                        help='json embeds every image in the payload; csv writes a '
                             'long-format sidecar (gzipped when the name ends .gz); '
                             'none keeps only aggregates')
    parser.add_argument('--out', default='runs/stage4b_mask_audit/audit.json')
    args = parser.parse_args()

    manifest = Path(args.manifest)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = read_manifest(manifest)
    if manifest_sha != 'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435':
        raise SystemExit(f'refusing to audit: manifest sha256 is {manifest_sha}')
    picked = stratified_sample(rows, args.num_images)
    if args.shuffle_seed is not None and args.shuffle_seed >= 0:
        rng = np.random.default_rng(args.shuffle_seed)
        rng.shuffle(picked)
    print(f'auditing {len(picked)} images across '
          f'{len(set(r["source"] for r in picked))} sources', flush=True)

    cache_array = None
    if args.image_cache:
        index = json.loads((Path(args.image_cache) / 'index.json').read_text(encoding='utf-8'))
        if index.get('manifest_sha256') != manifest_sha:
            raise SystemExit('image cache was built from a different manifest')
        cache_array = np.load(str(Path(args.image_cache) / 'images.uint8.npy'),
                              mmap_mode='r')

    id_to_index = {r['sample_id']: i for i, r in enumerate(rows)}
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    collators = {a: make_collator(a, args.geometry_beta, args.num_candidates,
                                  args.guidance_lambda, args.roll_shift,
                                  args.geometry_control)
                 for a in arms}
    prior_vegetation = StructurePrior(mode='vegetation')
    prior_boundary = StructurePrior(mode='boundary')
    prior_shuffled = StructurePrior(mode='shuffled_boundary')
    prior_shifted = StructurePrior(mode='shifted_boundary', roll_shift=args.roll_shift)

    records: List[Dict[str, object]] = []
    timing: Dict[str, List[float]] = collections.defaultdict(list)
    t_start = time.time()

    def process_chunk(chunk: List[Tuple[int, Dict[str, str]]],
                      batch_index: int) -> None:
        tensors, batches = [], []
        for index, row in chunk:
            if cache_array is not None:
                pil = Image.fromarray(np.asarray(cache_array[id_to_index[row['sample_id']]]),
                                      mode='RGB')
            else:
                with Image.open(Path(args.root) / row['path']) as handle:
                    pil = handle.convert('RGB')
            tensor = augmented_crop(pil, args.augmentation_seed, 0, index)
            tensors.append(tensor)
            batches.append((tensor, row['sample_id']))

        with torch.no_grad():
            stacked = torch.stack(tensors)
            ids = [b[1] for b in batches]
            veg_result = prior_vegetation.compute(stacked, sample_ids=ids)
            bnd_result = prior_boundary.compute(stacked, sample_ids=ids)
            shuffled_result = prior_shuffled.compute(stacked, sample_ids=ids)
            shifted_result = prior_shifted.compute(stacked, sample_ids=ids)
            veg, bnd = veg_result.map.numpy(), bnd_result.map.numpy()
            shuffled, shifted = shuffled_result.map.numpy(), shifted_result.map.numpy()
            shifted_info = shifted_result.diagnostics

        # -- OFFICIAL BATCH SEMANTICS ------------------------------------- #
        # The training collator samples ONE target size and ONE context size per
        # batch, then truncates every context mask to the batch-wide minimum
        # length.  The audit must reproduce both, otherwise its context counts
        # are pre-truncation values that never reach the network.
        generator = torch.Generator()
        generator.manual_seed(20261022 + batch_index)
        p_size = collators[arms[0]]._sample_block_size(
            generator=generator, scale=collators[arms[0]].pred_mask_scale,
            aspect_ratio_scale=collators[arms[0]].aspect_ratio)
        e_size = collators[arms[0]]._sample_block_size(
            generator=generator, scale=collators[arms[0]].enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        own_priors = {
            'C1': (veg, veg_result.usable.numpy()),
            'G1': (veg, veg_result.usable.numpy()),
            'C2': (bnd, bnd_result.usable.numpy()),
            'G2': (bnd, bnd_result.usable.numpy()),
            'C3': (shuffled, shuffled_result.usable.numpy()),
            'G3': (shifted, shifted_result.usable.numpy()),
        }
        per_arm: Dict[str, object] = {}
        for arm_index, (arm, collator) in enumerate(collators.items()):
            t0 = time.time()
            collator._itr_counter.value = batch_index
            collator._batch_size = len(chunk)
            collator._target_calls = 0
            collator._current_union = None
            if isinstance(collator, StructureGuidedMaskCollator):
                if collator.prior is not None and arm in own_priors:
                    maps, usable = own_priors[arm]
                    collator._prior_map = torch.from_numpy(maps).float()
                    collator._prior_usable = torch.from_numpy(usable).bool()
                    collator._prior_diagnostics = {}
                else:
                    collator._prior_map = None
                    collator._prior_usable = None
                # same candidate seed the collator derives from its step counter
                collator._cand_gen = torch.Generator()
                collator._cand_gen.manual_seed(batch_index + 1_000_003)
            # the context block is drawn from the global RNG; give every
            # (batch, arm) an independent, reproducible stream
            state = torch.get_rng_state()
            torch.manual_seed(20261022 + batch_index * 101 + arm_index)
            try:
                masks_p, masks_e, pre_trunc = place_masks(
                    collator, batches, p_size, e_size)
            finally:
                torch.set_rng_state(state)
            timing[arm].append(time.time() - t0)
            per_arm[arm] = {'masks_p': masks_p, 'masks_e': masks_e,
                            'pre_trunc': pre_trunc}

        for position, (index, row) in enumerate(chunk):
            record: Dict[str, object] = {
                'index': index,
                'sample_id': row['sample_id'],
                'source': row['source'],
                'original_id': row.get('original_id', ''),
                'batch_index': batch_index,
                'target_block_size': list(p_size),
                'context_block_size': list(e_size),
                'mean_vegetation': float(veg[position].mean()),
                'mean_boundary': float(bnd[position].mean()),
                'vegetation_lag1': lag1_autocorrelation(
                    torch.from_numpy(veg[position])[None]),
                'boundary_lag1': lag1_autocorrelation(
                    torch.from_numpy(bnd[position])[None]),
                'shuffled_lag1': lag1_autocorrelation(
                    torch.from_numpy(shuffled[position])[None]),
                'shifted_lag1': lag1_autocorrelation(
                    torch.from_numpy(shifted[position])[None]),
                'prior_diagnostics': {
                    'shifted': {
                        'pearson_correlation_with_original':
                            shifted_info.get('shift_pearson_correlation_with_original'),
                        'lag1_shifted': shifted_info.get('lag1_autocorrelation_shifted'),
                        'lag1_original': shifted_info.get('lag1_autocorrelation_original'),
                        'seam_jump': shifted_info.get('seam_mean_abs_jump'),
                    },
                    'shuffled_value_multiset_preserved': True,
                },
                'arms': {},
            }
            for arm in arms:
                entry = per_arm[arm]
                record['arms'][arm] = _arm_geometry(
                    entry['masks_p'][position], entry['masks_e'][position],
                    veg[position], bnd[position], [entry['pre_trunc'][position]])
            records.append(record)

    def _arm_geometry(masks_p, masks_e, veg_map, bnd_map, pre_trunc):
        from src.masks.geometry import geometry_record
        return geometry_record(
            masks_p, masks_e,
            prior_maps={'vegetation': veg_map, 'boundary': bnd_map},
            pre_truncation_context_lengths=pre_trunc)

    batch_size = args.batch_size
    chunk: List[Tuple[int, Dict[str, str]]] = []
    batch_index = 0
    for row in picked:
        # the dataset index (manifest position) is what the training pipeline
        # uses to derive the augmentation seed, so the audit must use it too
        chunk.append((id_to_index[row['sample_id']], row))
        if len(chunk) >= batch_size:
            process_chunk(chunk, batch_index)
            batch_index += 1
            chunk = []
            if len(records) % 200 < batch_size:
                print(f'  {len(records)}/{len(picked)} images '
                      f'({(time.time() - t_start) / max(len(records), 1):.3f} s/img)',
                      flush=True)
    if chunk:
        process_chunk(chunk, batch_index)

    total_seconds = time.time() - t_start
    print(f'audited {len(records)} images in {total_seconds:.0f}s '
          f'({total_seconds / max(len(records), 1):.3f} s/image)', flush=True)

    # ---------------- aggregate ---------------------------------------- #
    metrics = sorted({k for r in records for k in r['arms'][arms[0]]})
    aggregate: Dict[str, object] = {}
    for arm in arms:
        per_source: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
        overall: Dict[str, Dict[str, float]] = {}
        for metric in metrics:
            values = [r['arms'][arm][metric] for r in records]
            overall[metric] = summarize(values)
            overall[metric]['ci_low'], overall[metric]['ci_high'] = (
                bootstrap_ci(values, n_boot=args.bootstrap, seed=1)['low'],
                bootstrap_ci(values, n_boot=args.bootstrap, seed=1)['high'])
            sources = [str(r['source']) for r in records]
            overall[metric]['clustered_ci_low'], overall[metric]['clustered_ci_high'] = (
                clustered_bootstrap_ci(sources, values, n_boot=200, seed=2)['low'],
                clustered_bootstrap_ci(sources, values, n_boot=200, seed=2)['high'])
            groups: Dict[str, List[float]] = collections.defaultdict(list)
            for r, v in zip(records, values):
                groups[str(r['source'])].append(v)
            for source, group in groups.items():
                per_source[source][metric] = summarize(group)
        aggregate[arm] = {
            'overall': overall,
            'by_source': per_source,
            'source_balanced': {
                metric: float(np.mean([per_source[s][metric]['mean']
                                       for s in per_source if metric in per_source[s]]))
                for metric in metrics if any(metric in per_source[s] for s in per_source)
            },
            'generation_seconds_per_image': float(np.mean(timing[arm])),
            'generation_seconds_p90': float(np.percentile(timing[arm], 90)),
        }

    # vegetation-coverage strata (source-balanced as well as raw)
    bins = vegetation_bins([float(r['mean_vegetation']) for r in records])
    strata: Dict[str, object] = collections.defaultdict(dict)
    for arm in arms:
        by_bin: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
        for label in sorted(set(bins)):
            idx = [i for i, b in enumerate(bins) if b == label]
            for metric in ('target_cells', 'context_cells', 'duplicated_target_area',
                           'mean_vegetation_target_union', 'mean_boundary_target_union'):
                by_bin[label][metric] = summarize(
                    [records[i]['arms'][arm][metric] for i in idx])
        strata[arm] = by_bin

    # paired differences against the reference arm
    reference = 'B'
    comparisons: Dict[str, object] = {}
    for arm in arms:
        if arm == reference:
            continue
        block: Dict[str, object] = {}
        for metric in metrics:
            block[metric] = paired_delta(
                [r['arms'][arm][metric] for r in records],
                [r['arms'][reference][metric] for r in records],
                n_boot=args.bootstrap, seed=3)
        comparisons[f'{arm}-{reference}'] = block

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- per-image sidecar --------------------------------- #
    sidecar = None
    if args.per_image_format == 'csv':
        sidecar = out.with_suffix('').with_name(out.stem + '_per_image.csv.gz')
        import gzip
        with gzip.open(sidecar, 'wt', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            header = ['sample_id', 'source', 'original_id', 'index',
                      'target_block_size', 'context_block_size',
                      'mean_vegetation', 'mean_boundary',
                      'vegetation_lag1', 'boundary_lag1',
                      'shuffled_lag1', 'shifted_lag1']
            for arm in arms:
                header += [f'{arm}:{m}' for m in metrics]
            writer.writerow(header)
            for record in records:
                row = [record['sample_id'], record['source'],
                       record['original_id'], record['index'],
                       'x'.join(map(str, record['target_block_size'])),
                       'x'.join(map(str, record['context_block_size'])),
                       record['mean_vegetation'], record['mean_boundary'],
                       record['vegetation_lag1'], record['boundary_lag1'],
                       record['shuffled_lag1'], record['shifted_lag1']]
                for arm in arms:
                    row += [record['arms'][arm].get(m, '') for m in metrics]
                writer.writerow(row)
        print(f'wrote {sidecar} ({sidecar.stat().st_size / 1e6:.1f} MB)')

    payload: Dict[str, object] = {
        'manifest': str(manifest),
        'manifest_sha256': manifest_sha,
        'n_images': len(records),
        'sampling': ('source-stratified prefix sample; the first n/source rows of '
                     'each source in manifest order; identical sample for every arm'),
        'augmentation_seed': args.augmentation_seed,
        'epoch': 0,
        'batch_size': args.batch_size,
        'arms': arms,
        'arm_labels': {a: BASE_ARMS[a]['label'] for a in arms},
        'geometry_beta': args.geometry_beta,
        'geometry_control': args.geometry_control,
        'num_candidates': args.num_candidates,
        'guidance_lambda': args.guidance_lambda,
        'roll_shift': list(args.roll_shift),
        'collator_kwargs': {k: list(v) if isinstance(v, tuple) else v
                            for k, v in COLLATOR_KWARGS.items()},
        'source_counts': dict(collections.Counter(str(r['source']) for r in records)),
        'audit_seconds': total_seconds,
        'aggregate': aggregate,
        'vegetation_strata': strata,
        'comparisons_vs_B': comparisons,
        'per_image_path': str(sidecar) if sidecar else None,
    }
    if args.per_image_format == 'json':
        payload['per_image'] = records
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')

    # ---------------- console summary --------------------------------- #
    key_metrics = ['total_target_area', 'target_cells', 'duplicated_target_area',
                   'context_cells', 'target_overlap_fraction', 'target_center_spread',
                   'mean_vegetation_target_union', 'mean_boundary_target_union',
                   'mean_vegetation_target_multiset', 'mean_boundary_target_multiset',
                   'vegetation_union_mass_fraction', 'boundary_union_mass_fraction',
                   'vegetation_repeat_mass_ratio', 'boundary_repeat_mass_ratio']
    print(f'\n{"arm":<6}{"area":>9}{"union":>9}{"dupl":>9}{"ctx":>9}'
          f'{"ovl_frac":>10}{"spread":>9}{"vegU":>8}{"bndU":>8}'
          f'{"vegRep":>8}{"bndRep":>8}{"s/img":>8}')
    for arm in arms:
        a = aggregate[arm]['overall']
        def g(metric):
            return a.get(metric, {}).get('mean', float('nan'))
        print(f'{arm:<6}{g("total_target_area"):>9.1f}{g("target_cells"):>9.2f}'
              f'{g("duplicated_target_area"):>9.2f}{g("context_cells"):>9.2f}'
              f'{g("target_overlap_fraction"):>10.4f}{g("target_center_spread"):>9.2f}'
              f'{g("mean_vegetation_target_union"):>8.4f}'
              f'{g("mean_boundary_target_union"):>8.4f}'
              f'{g("vegetation_repeat_mass_ratio"):>8.3f}'
              f'{g("boundary_repeat_mass_ratio"):>8.3f}'
              f'{aggregate[arm]["generation_seconds_per_image"]:>8.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
