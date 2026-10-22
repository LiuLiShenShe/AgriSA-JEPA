#!/usr/bin/env python
"""PhenoBench semantic segmentation with a frozen ViT-H backbone and linear probe.

This is the Stage 3A downstream evaluation.  It answers one question under one
fixed protocol:

    Do the features of an agricultural continued-pretrained I-JEPA ViT-H/14
    segment agricultural images better than the official ImageNet I-JEPA
    ViT-H/14 they started from?

Pipeline
--------
1. Encode every PhenoBench 224x224 tile with the frozen backbone (cached).
2. Train an identical ``Linear(1280 -> 3)`` probe on the official train split.
3. Predict every val image at full 1024x1024 resolution (tile -> stitch).
4. Score with a self-implemented per-class IoU AND the official
   ``phenobench-eval --task semantics``; the two are cross-checked.

Both arms share every setting: tile grid, normalization, probe, optimizer, LR,
epochs, seed, batch size, label mapping, evaluation.  Only the backbone weights
differ.  The script refuses to run if the two arms' protocols differ.

Usage
-----
    # cache features for one arm (GPU), then train/evaluate (CPU is fine)
    ./.venv/bin/python -m src.downstream.phenobench_probe --arm A --stage features
    ./.venv/bin/python -m src.downstream.phenobench_probe --arm A --stage train_eval

    # or everything in one go
    ./.venv/bin/python -m src.downstream.phenobench_probe --arm B --stage all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.downstream.features import (
    arm_config,
    build_encoder,
    cache_key,
    checkpoint_config,
    extract_tile_features,
    sha256_file,
)
from src.downstream.phenobench_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    PhenoBenchSemantic,
    dataset_summary,
    list_split_files,
    load_image_and_label,
    map_partial_to_parent,
)
from src.downstream.probe import (
    IoUAccumulator,
    LinearSegmentationProbe,
    patch_class_counts,
    train_probe,
)
from src.downstream.tiling import (
    TILE_SIZE,
    covered_mask,
    n_tiles_per_axis,
    stitch_logits,
    tile_coords,
    tile_image,
    tile_label,
)

DEFAULT_ROOT = '/data/datasets/PhenoBench/PhenoBench-v110'
DEFAULT_CACHE = 'runs/stage3a_phenobench/cache'


# --------------------------------------------------------------------------- #
# Feature caching
# --------------------------------------------------------------------------- #

def _cache_paths(cache_dir: Path, key: str, split: str, name: str) -> Path:
    return cache_dir / key / split / f'{Path(name).stem}.npy'


def write_cache_manifest(cache_dir: Path, cfg: Dict[str, object]) -> None:
    """Record which weights a feature cache belongs to.

    A cache directory without this manifest is refused, and a manifest whose
    checkpoint hash differs from the requested one is refused too, so a cache
    can never be reused for different weights.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'cache_manifest.json'
    expected = {
        'label': cfg['label'],
        'checkpoint': cfg['checkpoint'],
        'checkpoint_sha256': cfg['checkpoint_sha256'],
        'tiling': 'overlapping 224x224, stride 200, 5x5 per 1024 image',
        'encoder': 'ViT-H/14 frozen, 16x16 tokens, embed 1280',
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('checkpoint_sha256') != expected['checkpoint_sha256']:
            raise RuntimeError(
                f'Feature cache {cache_dir} belongs to checkpoint '
                f"{existing.get('checkpoint_sha256')} but "
                f'{expected["checkpoint_sha256"]} was requested. Refusing to '
                f'reuse it.')
        return
    manifest_path.write_text(json.dumps(expected, indent=2) + '\n', encoding='utf-8')


def verify_cache_manifest(cache_dir: Path, cfg: Dict[str, object]) -> None:
    """Assert that a cache directory belongs to the requested weights."""
    manifest_path = cache_dir / 'cache_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Feature cache {cache_dir} has no cache_manifest.json; it cannot '
            f'be verified against the requested checkpoint.')
    existing = json.loads(manifest_path.read_text(encoding='utf-8'))
    if existing.get('checkpoint_sha256') != cfg['checkpoint_sha256']:
        raise RuntimeError(
            f'Feature cache {cache_dir} was built from '
            f"{existing.get('checkpoint')} "
            f"({existing.get('checkpoint_sha256')}), not from "
            f"{cfg['checkpoint']} ({cfg['checkpoint_sha256']}). Refusing to reuse it.")


def resolve_arm_config(arm: Optional[str], checkpoint: Optional[str],
                       arm_label: Optional[str],
                       continued_training: str = 'unknown',
                       data_version: Optional[str] = None) -> Dict[str, object]:
    """Either a legacy arm (``A``/``B``) or an explicit checkpoint."""
    if checkpoint:
        label = arm_label or Path(checkpoint).stem
        return checkpoint_config(checkpoint, label,
                                 continued_training=continued_training,
                                 data_version=data_version)
    if not arm:
        raise ValueError('either --arm or --checkpoint must be given')
    cfg = dict(arm_config(arm))
    cfg['checkpoint_sha256'] = sha256_file(cfg['checkpoint'])
    return cfg


def cache_split_features(
    arm: str,
    root: Path,
    cache_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int = 16,
    limit: Optional[int] = None,
    overwrite: bool = False,
    checkpoint: Optional[str] = None,
    arm_label: Optional[str] = None,
) -> Dict[str, object]:
    """Extract and cache ``[n_tiles,256,1280]`` fp16 features for one split."""
    cfg = resolve_arm_config(arm, checkpoint, arm_label)
    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    encoder = build_encoder(cfg['checkpoint'], device)
    names = list_split_files(root, split)
    if limit is not None:
        names = names[:limit]

    out_dir = cache_dir / key / split
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cache_manifest(cache_dir / key, cfg)
    written, skipped, t0 = 0, 0, time.time()
    for i, name in enumerate(names):
        out_path = _cache_paths(cache_dir, key, split, name)
        if out_path.is_file() and not overwrite:
            skipped += 1
            continue
        image, _ = load_image_and_label(root, split, name)
        tiles = tile_image(image, tile_coords(image.shape[0], image.shape[1]))
        feats = extract_tile_features(
            encoder, tiles, device, batch_size=batch_size, use_bfloat16=True)
        np.save(out_path, feats.numpy().astype(np.float16))
        written += 1
        if (i + 1) % 50 == 0:
            print(f'  [{arm}/{split}] {i + 1}/{len(names)} '
                  f'({(time.time() - t0):.0f}s)', flush=True)
    return {
        'arm': cfg['arm'], 'label': cfg['label'], 'split': split,
        'images': len(names),
        'written': written, 'skipped_existing': skipped,
        'seconds': round(time.time() - t0, 1),
        'cache_key': key,
        'cache_dir': str(cache_dir / key / split),
        'checkpoint': cfg['checkpoint'],
        'checkpoint_sha256': cfg['checkpoint_sha256'],
    }


def load_cached_split(cache_dir: Path, key: str, split: str,
                      names: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    for name in names:
        path = _cache_paths(cache_dir, key, split, name)
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing cached features for {arm}/{split}/{name}: {path}\n'
                f'Run the `features` stage for this arm first.')
        out[name] = np.load(path)
    return out


# --------------------------------------------------------------------------- #
# Probe train / evaluate
# --------------------------------------------------------------------------- #

def build_training_tensors(
    root: Path,
    names: Sequence[str],
    features: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    """Assemble cached tokens and per-patch class counts.

    The probe loss is computed from patch-level class histograms rather than a
    materialised ``[N,224,224]`` label tensor.  That is mathematically identical
    to the nearest-upsampled pixel loss (see ``probe.patch_ce_loss``) and keeps
    memory at ~6 GB instead of ~10 GB for the training split.
    """
    token_list, count_list = [], []
    for name in names:
        image, raw_label = load_image_and_label(root, 'train', name)
        label = map_partial_to_parent(raw_label)
        tiles = tile_coords(image.shape[0], image.shape[1])
        tokens = torch.from_numpy(features[name])   # keep fp16; cast per batch
        if tokens.shape[0] != len(tiles):
            raise ValueError(
                f'{name}: {tokens.shape[0]} cached tiles, expected {len(tiles)}')
        token_list.append(tokens)
        counts = patch_class_counts(tile_label(label, tiles), num_classes=NUM_CLASSES)
        count_list.append(torch.from_numpy(counts))
    return {
        'tokens': torch.cat(token_list, dim=0),
        'counts': torch.cat(count_list, dim=0),
    }


def predict_split(
    probe: LinearSegmentationProbe,
    root: Path,
    split: str,
    names: Sequence[str],
    features: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Full-resolution prediction per image; overlaps are logit-averaged.

    The tile grid covers every pixel, so no pixel is left unpredicted and the
    self-metric (full image) and the official metric score the same region.
    """
    predictions: Dict[str, np.ndarray] = {}
    for name in names:
        image, _ = load_image_and_label(root, split, name)
        tiles = tile_coords(image.shape[0], image.shape[1])
        tokens = torch.from_numpy(features[name].astype(np.float32)).to(device)
        tile_logits = probe.predict_tile_logits(tokens)
        predictions[name] = stitch_logits(
            tile_logits, image.shape[0], image.shape[1], tiles)
    return predictions


def write_prediction_pngs(predictions: Dict[str, np.ndarray], out_dir: Path) -> None:
    """Write official-format PNG predictions (uint8 ids 0/1/2) for phenobench-eval."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in predictions.items():
        arr = np.where(pred < 0, 0, pred).astype(np.uint8)
        Image.fromarray(arr, mode='L').save(out_dir / name)


def score_predictions(
    root: Path,
    split: str,
    names: Sequence[str],
    predictions: Dict[str, np.ndarray],
    ignore_border: bool = False,
) -> Dict[str, object]:
    """Self-implemented per-class IoU over the full-resolution prediction.

    Scored over the whole image, matching ``phenobench-eval --task semantics``,
    which also scores every pixel.  ``ignore_border`` is retained only for
    diagnostics and is off by default.
    """
    acc = IoUAccumulator(NUM_CLASSES)
    for name in names:
        _, raw_label = load_image_and_label(root, split, name)
        label = map_partial_to_parent(raw_label)
        pred = predictions[name]
        valid = covered_mask(label.shape[0], label.shape[1]) if ignore_border else None
        acc.update(pred, label, valid)
    ious = acc.compute()
    return {
        'per_class_iou_percent': {CLASS_NAMES[c]: round(100.0 * float(v), 4)
                                  for c, v in ious.items()},
        'mIoU_percent': round(100.0 * float(np.mean([float(v) for v in ious.values()])), 4),
        'pixels_intersection': {CLASS_NAMES[c]: int(acc.intersection[c]) for c in range(NUM_CLASSES)},
        'pixels_union': {CLASS_NAMES[c]: int(acc.union[c]) for c in range(NUM_CLASSES)},
        'scored_region': 'full 1024x1024' if not ignore_border else 'tile-covered interior',
    }


def run_official_eval(root: Path, prediction_dir: Path, split: str,
                      work_dir: Path) -> Dict[str, object]:
    """Run ``phenobench-eval --task semantics`` and parse its output."""
    cmd = [sys.executable, '-m', 'phenobench.evaluation.phenobench_eval',
           '--task', 'semantics', '--phenobench_dir', str(root),
           '--prediction_dir', str(prediction_dir), '--split', split]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / 'phenobench_eval_stdout.txt').write_text(
        proc.stdout + '\n' + proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(
            f'phenobench-eval failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}')
    parsed: Dict[str, float] = {}
    for line in proc.stdout.splitlines():
        for key in ('IoU (crop)', 'IoU (weed)', 'IoU (soil)', 'mIoU'):
            if line.strip().startswith(key):
                parsed[key] = float(line.split(':')[1].strip())
    return {'command': ' '.join(cmd), 'stdout': proc.stdout, 'metrics': parsed}


def cross_check_metrics(self_metrics: Dict[str, object],
                        official_metrics: Dict[str, float],
                        tolerance_pp: float = 0.01) -> Dict[str, object]:
    """Compare the self-implemented IoU against the official evaluator.

    Both must score the same region.  A mismatch above ``tolerance_pp``
    percentage points is an error, not a rounding difference: the protocol says
    investigate, never pick the more favourable of the two.
    """
    ours = self_metrics['per_class_iou_percent']
    pairs = [
        ('crop', 'IoU (crop)', ours['crop']),
        ('weed', 'IoU (weed)', ours['weed']),
        ('soil', 'IoU (soil)', ours['soil']),
        ('mIoU', 'mIoU', self_metrics['mIoU_percent']),
    ]
    rows = []
    for name, key, value in pairs:
        official = float(official_metrics.get(key, float('nan')))
        ours_value = float(value)
        rows.append({
            'metric': name,
            'self_percent': ours_value,
            'official_percent': official,
            'difference_pp': round(ours_value - official, 6),
        })
    max_diff = max(abs(r['difference_pp']) for r in rows)
    return {
        'rows': rows,
        'max_abs_difference_pp': round(float(max_diff), 6),
        'tolerance_pp': tolerance_pp,
        'agrees': bool(max_diff <= tolerance_pp),
        'note': ('The official evaluator scores every pixel; the self metric now '
                 'scores the same region, so any gap beyond rounding is a bug.'),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def train_and_evaluate(
    arm: str,
    root: Path,
    cache_dir: Path,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seeds: Sequence[int],
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    run_official: bool = True,
    checkpoint: Optional[str] = None,
    arm_label: Optional[str] = None,
    continued_training: str = 'unknown',
    data_version: Optional[str] = None,
) -> Dict[str, object]:
    """Train and evaluate one probe per seed, then aggregate.

    Cached features are loaded once and shared by every seed, so a multi-seed
    run costs only the extra probe training (seconds per seed), not another
    feature pass.
    """
    cfg = resolve_arm_config(arm, checkpoint, arm_label,
                              continued_training=continued_training,
                              data_version=data_version)
    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    train_names = list_split_files(root, 'train')
    val_names = list_split_files(root, 'val')
    if train_limit is not None:
        train_names = train_names[:train_limit]
    if val_limit is not None:
        val_names = val_names[:val_limit]

    verify_cache_manifest(cache_dir / key, cfg)
    print(f'[{cfg["arm"]}] loading cached train features ({len(train_names)} images)', flush=True)
    train_feats = load_cached_split(cache_dir, key, 'train', train_names)
    tensors = build_training_tensors(root, train_names, train_feats)
    del train_feats

    print(f'[{cfg["arm"]}] loading cached val features ({len(val_names)} images)', flush=True)
    val_feats = load_cached_split(cache_dir, key, 'val', val_names)

    per_seed: List[Dict[str, object]] = []
    for seed in seeds:
        print(f'[{arm}] seed {seed}: training probe', flush=True)
        probe = LinearSegmentationProbe().to(device)
        t0 = time.time()
        result = train_probe(
            probe, tensors['tokens'], tensors['counts'],
            epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, seed=seed, device=device, verbose=True)
        train_seconds = time.time() - t0

        print(f'[{arm}] seed {seed}: predicting {len(val_names)} val images', flush=True)
        predictions = predict_split(probe, root, 'val', val_names, val_feats, device)
        metrics = score_predictions(root, 'val', val_names, predictions)

        seed_dir = out_dir / f'seed{seed}'
        pred_dir = seed_dir / 'predictions' / 'semantics'
        write_prediction_pngs(predictions, pred_dir)

        official = None
        cross_check = None
        if run_official:
            print(f'[{arm}] seed {seed}: official phenobench-eval', flush=True)
            official = run_official_eval(root, seed_dir / 'predictions', 'val', seed_dir)
            cross_check = cross_check_metrics(metrics, official['metrics'])
            if not cross_check['agrees']:
                raise RuntimeError(
                    f'Self metric and official metric disagree by '
                    f"{cross_check['max_abs_difference_pp']} pp "
                    f"(> {cross_check['tolerance_pp']}). Investigate before "
                    f'reporting; do not pick the better one.')

        per_seed.append({
            'seed': seed,
            'train_updates': result['updates'],
            'loss_history': result['loss_history'],
            'final_train_loss': result['final_loss'],
            'train_seconds': round(train_seconds, 1),
            'self_metrics': metrics,
            'official_metrics': official['metrics'] if official else None,
            'metric_cross_check': cross_check,
            'predictions_dir': str(pred_dir),
        })
        peak = (round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
                if device.type == 'cuda' else None)
        per_seed[-1]['peak_gpu_memory_mib'] = peak
        print(json.dumps({'seed': seed, 'self_metrics': metrics,
                          'official_metrics': official['metrics'] if official else None},
                         indent=2, ensure_ascii=False), flush=True)

    payload = {
        'arm': cfg['arm'],
        'arm_label': cfg['label'],
        'cache_key': key,
        'backbone_checkpoint': cfg['checkpoint'],
        'backbone_checkpoint_sha256': sha256_file(cfg['checkpoint']),
        'continued_training': cfg['continued_training'],
        'pretraining_data_version': cfg['data_version'],
        'protocol': {
            'backbone': 'ViT-H/14, frozen, 224x224 tiles, 16x16 tokens, embed 1280',
            'head': 'Linear(1280 -> 3) over patch tokens, nearest upsample x14',
            'tiling': ('overlapping 224x224, stride 200, 5x5 per 1024 image, '
                       'no padding, every pixel covered, overlaps logit-averaged'),
            'optimizer': 'AdamW',
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs': epochs,
            'batch_size': batch_size,
            'seeds': list(seeds),
            'loss': 'patch cross-entropy, exact equivalent of nearest-upsampled pixel CE',
            'train_images': len(train_names),
            'val_images': len(val_names),
        },
        'probe_parameters': 1280 * 3 + 3,
        'seeds': per_seed,
        'aggregate': aggregate_seeds(per_seed),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'result.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(payload['aggregate'], indent=2, ensure_ascii=False))
    return payload


def aggregate_seeds(per_seed: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Mean / std of the official metrics across seeds."""
    keys = {'soil': 'IoU (soil)', 'crop': 'IoU (crop)',
            'weed': 'IoU (weed)', 'mIoU': 'mIoU'}
    agg: Dict[str, object] = {'n_seeds': len(per_seed), 'official': {}, 'self': {}}
    for name, key in keys.items():
        values = [s['official_metrics'][key] for s in per_seed
                  if s.get('official_metrics') and key in s['official_metrics']]
        if values:
            agg['official'][name] = {
                'mean': round(float(np.mean(values)), 4),
                'std': round(float(np.std(values)), 4),
                'values': [round(float(v), 4) for v in values],
            }
        self_values = [s['self_metrics']['per_class_iou_percent'][name]
                       if name != 'mIoU' else s['self_metrics']['mIoU_percent']
                       for s in per_seed]
        agg['self'][name] = {
            'mean': round(float(np.mean(self_values)), 4),
            'std': round(float(np.std(self_values)), 4),
            'values': [round(float(v), 4) for v in self_values],
        }
    agg['final_train_loss'] = {
        'mean': round(float(np.mean([s['final_train_loss'] for s in per_seed])), 6),
        'values': [round(float(s['final_train_loss']), 6) for s in per_seed],
    }
    return agg



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['A', 'B'], default=None,
                        help="legacy arms; 'A' is the official checkpoint")
    parser.add_argument('--checkpoint', default=None,
                        help='evaluate an arbitrary backbone checkpoint '
                             '(e.g. a Stage 3B/4A encoder snapshot) instead of --arm')
    parser.add_argument('--arm-label', default=None,
                        help='label for --checkpoint; also the feature-cache key')
    parser.add_argument('--continued-training', default='unknown')
    parser.add_argument('--pretraining-data-version', default=None)
    parser.add_argument('--stage', choices=['features', 'train_eval', 'all'],
                        default='all')
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, nargs='+', default=[0],
                        help='probe seed(s); pass several for a stability estimate')
    parser.add_argument('--train-limit', type=int, default=None)
    parser.add_argument('--val-limit', type=int, default=None)
    parser.add_argument('--feature-limit', type=int, default=None)
    parser.add_argument('--overwrite-features', action='store_true')
    parser.add_argument('--no-official', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir or f'runs/stage3a_phenobench/arm{args.arm}')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if not args.arm and not args.checkpoint:
        raise SystemExit('either --arm or --checkpoint must be given')

    if args.stage in ('features', 'all'):
        for split in ('train', 'val'):
            report = cache_split_features(
                args.arm, root, cache_dir, split, device,
                batch_size=args.feature_batch_size,
                limit=args.feature_limit,
                overwrite=args.overwrite_features,
                checkpoint=args.checkpoint, arm_label=args.arm_label)
            print(json.dumps(report, indent=2), flush=True)

    if args.stage in ('train_eval', 'all'):
        payload = train_and_evaluate(
            args.arm, root, cache_dir, out_dir, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            weight_decay=args.weight_decay, seeds=args.seed,
            train_limit=args.train_limit, val_limit=args.val_limit,
            run_official=not args.no_official,
            checkpoint=args.checkpoint, arm_label=args.arm_label,
            continued_training=args.continued_training,
            data_version=args.pretraining_data_version)
        print(json.dumps({'result': str(out_dir / 'result.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())