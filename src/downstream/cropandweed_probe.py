#!/usr/bin/env python
"""Stage 8: CropAndWeed semantic segmentation, frozen ViT-H + linear probe.

Identical pipeline to ``phenobench_probe`` but evaluated on CropAndWeed with
the official CropsOrWeed9 mapping (9 classes).  Soil (0) and Vegetation (255)
have no CropsOrWeed9 mapping and are ignore for both the loss and the mIoU.

Split (see ``cropandweed_data``): train = full Application Set (ave_*); eval =
one image per Experimental-Set recording session (vwg_*), seeded at
``--sampling-seed`` so the eval set is reproducible.  The chosen eval names are
written to ``out_dir/split.csv`` on first use and every subsequent run with the
same out_dir must select the identical names, or it errors.

Pipeline (per backbone)
-----------------------
1. cache frozen-encoder tile features for train and eval (cache key binds label
   + full checkpoint SHA; a cache without a matching manifest is refused);
2. train ``Linear(1280 -> 9)`` probes, one per ``--seed``;
3. predict every eval image full-resolution (tile -> stitch, overlaps
   logit-averaged);
4. score 9-class mIoU and per-class IoU (ignore pixels excluded).

Usage
-----
    CUDA_VISIBLE_DEVICES=1 ./.venv-uv/bin/python -m src.downstream.cropandweed_probe \
        --checkpoint /data/.../jepa-encoder-step2000.pth --label A_p1 \
        --out-dir /data/agrisa_jepa_runs/stage8/cropandweed_sem/A_p1_seed0 \
        --seed 0 --stage all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.downstream.cropandweed_data import (CLASS_NAMES, IGNORE_ID,
                                             NUM_CLASSES,
                                             build_split, load_image_and_mask,
                                             split_counts, write_split_csv)
from src.downstream.features import (build_encoder, cache_key,
                                     checkpoint_config, extract_tile_features,
                                     sha256_file)
from src.downstream.probe import (IoUAccumulator, LinearSegmentationProbe,
                                  patch_class_counts, train_probe)
from src.downstream.tiling import (tile_coords, tile_image, tile_label,
                                   stitch_logits)

DEFAULT_IMAGES_ROOT = '/data/datasets/cropandweed/images'
DEFAULT_ANNOTATION_ROOT = ('/fj/jepa-二维图像任务/data/holdout_raw/'
                           'cropandweed_annotations/labelIds/CropAndWeed')
DEFAULT_CACHE = 'runs/stage8_cropandweed/cache'


# --------------------------------------------------------------------------- #
# Feature caching (same contract as phenobench_probe)
# --------------------------------------------------------------------------- #

def _cache_path(cache_dir: Path, key: str, split: str, name: str) -> Path:
    return cache_dir / key / split / f'{Path(name).stem}.npy'


def write_cache_manifest(cache_dir: Path, cfg: Dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'cache_manifest.json'
    expected = {
        'label': cfg['label'],
        'checkpoint': cfg['checkpoint'],
        'checkpoint_sha256': cfg['checkpoint_sha256'],
        'tiling': 'overlapping 224x224 (CropAndWeed 1080x1920 -> 5x9 tiles)',
        'encoder': 'ViT-H/14 frozen, 16x16 tokens, embed 1280',
        'mapping': 'CropsOrWeed9 (datasets.py verbatim)',
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('checkpoint_sha256') != expected['checkpoint_sha256']:
            raise RuntimeError(
                f'Feature cache {cache_dir} belongs to checkpoint '
                f"{existing.get('checkpoint_sha256')} but "
                f'{expected["checkpoint_sha256"]} was requested; refusing.')
        return
    manifest_path.write_text(json.dumps(expected, indent=2) + '\n',
                             encoding='utf-8')


def verify_cache_manifest(cache_dir: Path, cfg: Dict[str, object]) -> None:
    manifest_path = cache_dir / 'cache_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'{cache_dir} has no cache_manifest.json; cannot verify weights.')
    existing = json.loads(manifest_path.read_text(encoding='utf-8'))
    if existing.get('checkpoint_sha256') != cfg['checkpoint_sha256']:
        raise RuntimeError(
            f'{cache_dir} was built from {existing.get("checkpoint")} '
            f'({existing.get("checkpoint_sha256")}), not {cfg["checkpoint"]} '
            f'({cfg["checkpoint_sha256"]}); refusing to reuse it.')


def cache_split_features(
    cfg: Dict[str, object],
    images_root: Path,
    annotation_root: Path,
    split: Dict[str, List[str]],
    part: str,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    """Extract and cache ``[n_tiles,256,1280]`` fp16 features for one split."""
    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    out_dir = cache_dir / key / part
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cache_manifest(cache_dir / key, cfg)
    encoder = build_encoder(cfg['checkpoint'], device)
    names = split[part]
    # probe expects feature arrays keyed by image name
    written, skipped, t0 = 0, 0, time.time()
    repo = {}
    for i, name in enumerate(names):
        out_path = _cache_path(cache_dir, key, part, name)
        if out_path.is_file():
            skipped += 1
            repo[name] = out_path
            continue
        image, _ = load_image_and_mask(images_root, annotation_root, name)
        tiles = tile_image(image, tile_coords(image.shape[0], image.shape[1]))
        feats = extract_tile_features(encoder, tiles, device,
                                      batch_size=batch_size, use_bfloat16=True)
        np.save(out_path, feats.numpy().astype(np.float16))
        written += 1
        repo[name] = out_path
        if (i + 1) % 50 == 0:
            print(f'  [{cfg["label"]}/{part}] {i + 1}/{len(names)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return {'label': cfg['label'], 'split': part, 'images': len(names),
            'written': written, 'skipped_existing': skipped,
            'seconds': round(time.time() - t0, 1), 'cache_key': key}


def load_cached_split(cache_dir: Path, key: str, part: str,
                      names: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    for name in names:
        path = _cache_path(cache_dir, key, part, name)
        if not path.is_file():
            raise FileNotFoundError(f'missing cached features for {part}/{name}')
        out[name] = np.load(path)
    return out


# --------------------------------------------------------------------------- #
# Probe train / evaluate
# --------------------------------------------------------------------------- #

def build_training_tensors(
    images_root: Path,
    annotation_root: Path,
    names: Sequence[str],
    features: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor]:
    """Cached tokens + per-patch class counts over the non-ignore labels."""
    token_list, count_list = [], []
    for name in names:
        image, label = load_image_and_mask(images_root, annotation_root, name)
        tiles = tile_coords(image.shape[0], image.shape[1])
        tokens = torch.from_numpy(features[name])
        if tokens.shape[0] != len(tiles):
            raise ValueError(f'{name}: {tokens.shape[0]} tiles, expected {len(tiles)}')
        counts = patch_class_counts(tile_label(label, tiles),
                                    num_classes=NUM_CLASSES)
        # unmapped pixels (IGNORE_ID=-1) are counted in no class; their patch
        # pixel count is excluded because patch_class_counts only tallies
        # classes 0..C-1.
        token_list.append(tokens)
        count_list.append(torch.from_numpy(counts))
    return {'tokens': torch.cat(token_list, dim=0),
            'counts': torch.cat(count_list, dim=0)}


def predict_split(
    probe: LinearSegmentationProbe,
    images_root: Path,
    annotation_root: Path,
    names: Sequence[str],
    features: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    predictions: Dict[str, np.ndarray] = {}
    for name in names:
        image, _ = load_image_and_mask(images_root, annotation_root, name)
        tiles = tile_coords(image.shape[0], image.shape[1])
        tokens = torch.from_numpy(features[name].astype(np.float32)).to(device)
        tile_logits = probe.predict_tile_logits(tokens)
        predictions[name] = stitch_logits(
            tile_logits, image.shape[0], image.shape[1], tiles)
    return predictions


def score_predictions(
    images_root: Path,
    annotation_root: Path,
    names: Sequence[str],
    predictions: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """9-class IoU over the full image; IGNORE_ID pixels excluded."""
    acc = IoUAccumulator(NUM_CLASSES)
    for name in names:
        _, label = load_image_and_mask(images_root, annotation_root, name)
        pred = predictions[name]
        valid = label != IGNORE_ID
        acc.update(pred, label, valid)
    ious = acc.compute()
    return {
        'per_class_iou_percent': {
            CLASS_NAMES[c]: round(100.0 * float(ious[c]), 4)
            for c in range(NUM_CLASSES)},
        'mIoU_percent': round(100.0 * float(np.mean(list(ious.values()))), 4),
        'scored_region': 'full resolution, Soil/Vegetation excluded',
    }


def train_and_evaluate(
    cfg: Dict[str, object],
    images_root: Path,
    annotation_root: Path,
    split: Dict[str, List[str]],
    cache_dir: Path,
    out_dir: Path,
    device: torch.device,
    seeds: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
) -> Dict[str, object]:
    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    verify_cache_manifest(cache_dir / key, cfg)

    train_names = split['train']
    eval_names = split['eval']
    print(f'[{cfg["label"]}] loading train features ({len(train_names)})', flush=True)
    train_feats = load_cached_split(cache_dir, key, 'train', train_names)
    tensors = build_training_tensors(images_root, annotation_root,
                                     train_names, train_feats)
    del train_feats
    print(f'[{cfg["label"]}] loading eval features ({len(eval_names)})', flush=True)
    eval_feats = load_cached_split(cache_dir, key, 'eval', eval_names)

    per_seed = []
    for seed in seeds:
        print(f'[{cfg["label"]}] seed {seed}', flush=True)
        probe = LinearSegmentationProbe(num_classes=NUM_CLASSES)
        t0 = time.time()
        result = train_probe(probe, tensors['tokens'], tensors['counts'],
                             epochs=epochs, batch_size=batch_size, lr=lr,
                             weight_decay=weight_decay, seed=seed,
                             device=device, verbose=True)
        predictions = predict_split(probe, images_root, annotation_root,
                                    eval_names, eval_feats, device)
        metrics = score_predictions(images_root, annotation_root,
                                    eval_names, predictions)
        per_seed.append({
            'seed': seed,
            'final_train_loss': result['final_loss'],
            'train_updates': result['updates'],
            'train_seconds': round(time.time() - t0, 1),
            'self_metrics': metrics,
        })
        print(json.dumps({'seed': seed, 'mIoU_percent': metrics['mIoU_percent']},
                         indent=2), flush=True)

    payload = {
        'label': cfg['label'],
        'cache_key': key,
        'backbone_checkpoint': cfg['checkpoint'],
        'backbone_checkpoint_sha256': sha256_file(cfg['checkpoint']),
        'continued_training': cfg['continued_training'],
        'pretraining_data_version': cfg['data_version'],
        'protocol': {
            'backbone': 'ViT-H/14 frozen, 224x224 tiles, 16x16 tokens, embed 1280',
            'head': 'Linear(1280 -> 9) over patch tokens, nearest upsample x14',
            'mapping': 'CropsOrWeed9 (official datasets.py); Soil 0 and '
                       'Vegetation 255 are ignore for loss and mIoU',
            'tiling': 'overlapping 224x224, no padding, every pixel covered, '
                      'overlaps logit-averaged',
            'optimizer': 'AdamW',
            'lr': lr, 'weight_decay': weight_decay, 'epochs': epochs,
            'batch_size': batch_size, 'seeds': list(seeds),
            'loss': 'patch cross-entropy over non-ignore pixels',
            'train_images': len(train_names),
            'eval_images': len(eval_names),
            'split_seed': split.get('_seed'),
            'split_counts': split_counts(split),
        },
        'seeds': per_seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'result.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', required=True,
                        help='backbone label; also the feature-cache key')
    parser.add_argument('--continued-training', default='Stage 7B continued '
                        'pretraining (unknown)')
    parser.add_argument('--pretraining-data-version', default='agrifield40k-v1')
    parser.add_argument('--images-root', default=DEFAULT_IMAGES_ROOT)
    parser.add_argument('--annotation-root', default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument('--sampling-seed', type=int, default=20261107)
    parser.add_argument('--per-session', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--overwrite-features', action='store_true')
    args = parser.parse_args()

    if args.per_session != 1 and args.sampling_seed == 20261107:
        raise SystemExit('changing per-session sampling from the frozen default '
                         'requires an explicit decision; pass a different '
                         '--per-session AND re-freeze the protocol first')

    images_root = Path(args.images_root)
    annotation_root = Path(args.annotation_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)

    split_csv = out_dir / 'split.csv'
    if split_csv.is_file() and not args.overwrite_features:
        raise SystemExit(f'{split_csv} already exists; refusing to overwrite.')
    split = build_split(images_root, args.sampling_seed, args.per_session)
    split['_seed'] = args.sampling_seed
    write_split_csv(split, split_csv)

    cfg = checkpoint_config(args.checkpoint, args.label,
                            continued_training=args.continued_training,
                            data_version=args.pretraining_data_version)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    for part in ('train', 'eval'):
        report = cache_split_features(
            cfg, images_root, annotation_root, split, part, cache_dir,
            device, args.feature_batch_size)
        print(json.dumps(report, indent=2), flush=True)
    train_and_evaluate(cfg, images_root, annotation_root, split, cache_dir,
                       out_dir, device, args.seed, args.epochs, args.lr,
                       args.batch_size, args.weight_decay)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())