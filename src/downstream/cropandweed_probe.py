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
1. cache frozen-encoder tile features for train and eval, plus per-image
   per-patch class counts (cache key binds label + full checkpoint SHA; a
   cache without a matching manifest is refused);
2. train ``Linear(1280 -> 9)`` probes, one per ``--seed``, by **streaming the
   cached features chunk-by-chunk** — the train split is ~3,044 images x 45
   tiles = 136,980 tiles (~86 GB fp16), which does not fit in host RAM
   (the first attempt was OOM-killed at 125 GB);
3. predict every eval image full-resolution (tile -> stitch, overlaps
   logit-averaged), also streamed one image at a time;
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
                                             read_split_csv, split_counts,
                                             write_split_csv)
from src.downstream.features import (build_encoder, cache_key,
                                     checkpoint_config, extract_tile_features,
                                     sha256_file)
from src.downstream.probe import (IoUAccumulator, LinearSegmentationProbe,
                                  patch_class_counts, patch_ce_loss)
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


def _counts_cache_path(cache_dir: Path, key: str, split: str, name: str) -> Path:
    return cache_dir / key / split / f'{Path(name).stem}.counts.npy'


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
        image, label = load_image_and_mask(images_root, annotation_root, name)
        tiles = tile_image(image, tile_coords(image.shape[0], image.shape[1]))
        feats = extract_tile_features(encoder, tiles, device,
                                      batch_size=batch_size, use_bfloat16=True)
        np.save(out_path, feats.numpy().astype(np.float16))
        # per-image patch class counts [n_tiles, 16, 16, 9] int32; the mask is
        # already in hand, so caching it here avoids re-tiling labels later.
        grid = tile_coords(label.shape[0], label.shape[1])
        counts = patch_class_counts(tile_label(label, grid),
                                    num_classes=NUM_CLASSES)
        np.save(_counts_cache_path(cache_dir, key, part, name), counts)
        written += 1
        repo[name] = out_path
        if (i + 1) % 50 == 0:
            print(f'  [{cfg["label"]}/{part}] {i + 1}/{len(names)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return {'label': cfg['label'], 'split': part, 'images': len(names),
            'written': written, 'skipped_existing': skipped,
            'seconds': round(time.time() - t0, 1), 'cache_key': key}


# --------------------------------------------------------------------------- #
# Probe train / evaluate (streaming, memory-bounded)
# --------------------------------------------------------------------------- #
#
# The train split is 3,044 images x 45 tiles = 136,980 tile windows.  Held in
# RAM as fp16 that is ~86 GB, and close to twice that during a torch.cat of the
# per-image arrays — the first run of this task was OOM-killed (exit 137) on a
# 125 GB host before training even started.  Training and evaluation therefore
# stream from the per-image .npy cache: a handful of images is loaded to the
# device, consumed by mini-batch gradient steps, and dropped, so peak memory
# stays at one chunk (~0.25 GB) plus one image for prediction (~29 MB).

TRAIN_IMAGES_PER_CHUNK = 8


def ensure_train_counts(cache_dir: Path, key: str, images_root: Path,
                        annotation_root: Path,
                        names: Sequence[str]) -> Dict[str, object]:
    """Backfill per-image ``*.counts.npy`` for caches extracted before counts
    existed (the OFFICIAL cache predates this change).

    Counts are ``[n_tiles, 16, 16, 9]`` int32 histograms from the annotation
    masks; recomputing them needs no encoder, so backfilling the existing
    cache costs minutes instead of the ~30 min a re-extraction would need.
    """
    written, skipped = 0, 0
    for name in names:
        out = _counts_cache_path(cache_dir, key, 'train', name)
        if out.is_file():
            skipped += 1
            continue
        _, label = load_image_and_mask(images_root, annotation_root, name)
        grid = tile_coords(label.shape[0], label.shape[1])
        counts = patch_class_counts(tile_label(label, grid),
                                    num_classes=NUM_CLASSES)
        np.save(out, counts)
        written += 1
    return {'images': len(names), 'written': written,
            'skipped_existing': skipped}


def _load_chunk(cache_dir: Path, key: str, part: str, names: Sequence[str],
                device: torch.device) -> tuple:
    """Load one chunk of images' features + counts onto ``device``."""
    token_list, count_list = [], []
    for name in names:
        tokens = torch.from_numpy(
            np.load(_cache_path(cache_dir, key, part, name)))
        counts = torch.from_numpy(
            np.load(_counts_cache_path(cache_dir, key, part, name)))
        if tokens.shape[0] != counts.shape[0]:
            raise ValueError(f'{part}/{name}: {tokens.shape[0]} tile features '
                             f'but {counts.shape[0]} count rows')
        token_list.append(tokens)
        count_list.append(counts)
    # tokens stay fp16; the probe math casts per mini-batch exactly as
    # train_probe does, so results are bit-comparable with the other tasks.
    return (torch.cat(token_list).to(device),
            torch.cat(count_list).to(device))


def train_probe_lazy(
    probe: LinearSegmentationProbe,
    cache_dir: Path,
    key: str,
    names: Sequence[str],
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: torch.device = torch.device('cpu'),
    images_per_step: int = TRAIN_IMAGES_PER_CHUNK,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train the probe by streaming image chunks from the feature cache.

    Deterministic given (seed, cache contents): each epoch shuffles the image
    order with a torch.Generator seeded at ``seed``, then shuffles the tiles
    inside each chunk with the same generator before consuming them in
    mini-batches of ``batch_size`` tiles.  This is the same SGD as
    :func:`train_probe` (same loss, batch size, optimizer, epochs); only the
    sample ordering differs (image-level shuffle instead of one global tile
    shuffle, which is what keeps disk IO local to one chunk at a time).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe.to(device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n_images = len(names)
    history, updates = [], 0
    for epoch in range(epochs):
        img_order = torch.randperm(n_images, generator=generator)
        total_loss, total_pixels = 0.0, 0.0
        for c in range(0, n_images, images_per_step):
            chunk_names = [names[int(i)] for i in img_order[c:c + images_per_step]]
            tokens, counts = _load_chunk(cache_dir, key, 'train',
                                         chunk_names, device)
            tile_order = torch.randperm(tokens.shape[0], generator=generator)
            for b in range(0, tokens.shape[0], batch_size):
                idx = tile_order[b:b + batch_size]
                logits = probe(tokens[idx].float())
                loss = patch_ce_loss(logits, counts[idx].float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                updates += 1
                pixels = float(counts[idx].sum().item())
                total_loss += float(loss.item()) * pixels
                total_pixels += pixels
            del tokens, counts
        epoch_loss = total_loss / max(total_pixels, 1.0)
        history.append(epoch_loss)
        if verbose:
            print(f'  epoch {epoch + 1}/{epochs} loss {epoch_loss:.4f}',
                  flush=True)
    probe.eval()
    return {'loss_history': history, 'updates': updates,
            'final_loss': history[-1] if history else None}


def evaluate_split_lazy(
    probe: LinearSegmentationProbe,
    images_root: Path,
    annotation_root: Path,
    cache_dir: Path,
    key: str,
    names: Sequence[str],
    device: torch.device,
) -> Dict[str, object]:
    """Per-image lazy prediction and full-resolution 9-class IoU scoring.

    Peak memory is one image's cached features (~29 MB fp16) plus the
    stitched prediction; no per-image prediction is retained.
    """
    acc = IoUAccumulator(NUM_CLASSES)
    for name in names:
        tokens = torch.from_numpy(
            np.load(_cache_path(cache_dir, key, 'eval', name))
            .astype(np.float32)).to(device)
        _, label = load_image_and_mask(images_root, annotation_root, name)
        tiles = tile_coords(label.shape[0], label.shape[1])
        if tokens.shape[0] != len(tiles):
            raise ValueError(f'{name}: {tokens.shape[0]} tiles, expected '
                             f'{len(tiles)}')
        tile_logits = probe.predict_tile_logits(tokens)
        pred = stitch_logits(tile_logits, label.shape[0], label.shape[1],
                             tiles)
        valid = label != IGNORE_ID
        acc.update(pred, label, valid)
        del tokens
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
    counts_report = ensure_train_counts(cache_dir, key, images_root,
                                        annotation_root, train_names)
    if counts_report['written']:
        print(f'[{cfg["label"]}] backfilled train counts: '
              f'{counts_report["written"]} written / '
              f'{counts_report["skipped_existing"]} cached', flush=True)
    print(f'[{cfg["label"]}] streaming train ({len(train_names)} imgs '
          f'x ~45 tiles, chunked) + eval ({len(eval_names)} imgs)',
          flush=True)

    per_seed = []
    for seed in seeds:
        print(f'[{cfg["label"]}] seed {seed}', flush=True)
        probe = LinearSegmentationProbe(num_classes=NUM_CLASSES)
        t0 = time.time()
        result = train_probe_lazy(probe, cache_dir, key, train_names,
                                  epochs=epochs, batch_size=batch_size,
                                  lr=lr, weight_decay=weight_decay,
                                  seed=seed, device=device, verbose=True)
        metrics = evaluate_split_lazy(probe, images_root, annotation_root,
                                      cache_dir, key, eval_names, device)
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
            # deviation note (frozen-protocol rule): the train split (86 GB
            # fp16) cannot fit in host RAM, so training streams the per-image
            # feature cache in chunks; image-level shuffle per epoch replaces
            # the global tile shuffle of train_probe.  All arms share this
            # procedure, so intra-task comparisons are unaffected.
            'training_loader': (f'chunked streaming, {TRAIN_IMAGES_PER_CHUNK} '
                                'images/chunk, image-level seeded shuffle '
                                'per epoch'),
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
    parser.add_argument('--limit-images', type=int, default=None,
                        help='smoke-test only: truncate each split to N images')
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
    fresh = build_split(images_root, args.sampling_seed, args.per_session)
    if split_csv.is_file() and not args.overwrite_features:
        # Resume support: the split is deterministic in (sampling_seed,
        # per_session), so an existing split.csv must contain exactly the
        # same names.  Any mismatch is a protocol violation, not a restart
        # decision.
        existing = read_split_csv(split_csv)
        if (existing['train'] != fresh['train']
                or existing['eval'] != fresh['eval']):
            raise SystemExit(
                f'{split_csv} does not match a fresh seeded split; refusing.')
        print(f'  reusing existing split.csv '
              f'({len(fresh["train"])} train / {len(fresh["eval"])} eval)',
              flush=True)
        split = fresh
    else:
        split = fresh
        write_split_csv(split, split_csv)
    if args.limit_images:
        split['train'] = split['train'][:args.limit_images]
        split['eval'] = split['eval'][:args.limit_images]
    split['_seed'] = args.sampling_seed

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