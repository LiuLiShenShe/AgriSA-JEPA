#!/usr/bin/env python
"""Stage 8: PlantDoc 28-class classification, frozen ViT-H + linear head.

Split: the official repository ships ``train/{28 class dirs}`` and
``test/{28 class dirs}``; there is no official val split, so a val set is
carved from train by stratified class sampling at 15% (seed 20261107, frozen in
the Phase 2 protocol).  Test is evaluated once at the end.

Exactly the same image transform and head as ``deepweeds_probe``: RGB, resize
short side 224, center crop, mean-pooled patch tokens, one
``Linear(1280 -> 28)``.  Primary metric is macro-F1 on test.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 ./.venv-uv/bin/python -m src.downstream.plantdoc_probe \
        --checkpoint ... --label A_p1 \
        --out-dir /data/agrisa_jepa_runs/stage8/plantdoc_cls/A_p1_seed0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from src.downstream.classifier_probe import (LinearClassificationProbe,
                                             evaluate_classifier,
                                             train_classifier)
from src.downstream.features import (build_encoder, cache_key,
                                     checkpoint_config, extract_tile_features,
                                     sha256_file)

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent /
                   'data/holdout_raw/plantdoc_extract/PlantDoc-Dataset-master')
DEFAULT_CACHE = 'runs/stage8_plantdoc/cache'
VAL_FRACTION = 0.15
VAL_SEED = 20261107


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def discover(root: Path) -> Tuple[List[str], Dict[str, int]]:
    """``(image_names, class_of_image)`` from train/test class directories."""
    names, class_of = [], {}
    for split_name in ('train', 'test'):
        split_dir = root / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for img in sorted(p for p in class_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in
                              {'.jpg', '.jpeg', '.png'}):
                rel = f'{split_name}/{class_dir.name}/{img.name}'
                names.append(rel)
                class_of[rel] = class_dir.name
    return names, class_of


def build_split(root: Path) -> Dict[str, List[str]]:
    names, class_of = discover(root)
    classes = sorted(set(class_of.values()))
    class_id = {c: i for i, c in enumerate(classes)}
    train_all = [n for n in names if n.startswith('train/')]
    test = [n for n in names if n.startswith('test/')]
    if not train_all or not test:
        raise ValueError('no train/test images discovered')
    # train and test must expose the same class set: the head sees exactly
    # the classes it is scored on.
    train_classes = {class_of[n] for n in train_all}
    test_classes = {class_of[n] for n in test}
    if train_classes != test_classes:
        raise ValueError(
            f'train/test class mismatch: only-in-train {sorted(train_classes - test_classes)}, '
            f'only-in-test {sorted(test_classes - train_classes)}')
    # stratified 15% val carve from train
    from collections import defaultdict
    by_class: Dict[str, List[str]] = defaultdict(list)
    for n in train_all:
        by_class[class_of[n]].append(n)
    rng = np.random.RandomState(VAL_SEED)
    val, train = [], []
    for c, members in by_class.items():
        members = sorted(members)
        n_val = max(1, round(VAL_FRACTION * len(members)))
        chosen = set(rng.choice(len(members), size=n_val, replace=False).tolist())
        for i, m in enumerate(members):
            (val if i in chosen else train).append(m)
    return {'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}


def load_image_tile(root: Path, rel: str) -> np.ndarray:
    path = root / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        im.load()
        rgb = im.convert('RGB')
    w, h = rgb.size
    scale = 224.0 / min(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    rgb = rgb.resize((new_w, new_h), Image.BICUBIC)
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    return np.asarray(rgb.crop((left, top, left + 224, top + 224)), dtype=np.uint8)


def labels_vector(root: Path, names: Sequence[str]) -> torch.Tensor:
    _, class_of = discover(root)
    classes = sorted(set(class_of.values()))
    class_id = {c: i for i, c in enumerate(classes)}
    return torch.tensor([class_id[class_of[n]] for n in names], dtype=torch.int64)


# --------------------------------------------------------------------------- #
# Feature cache (identical contract to deepweeds_probe)
# --------------------------------------------------------------------------- #

def _cache_path(cache_dir: Path, key: str, part: str, name: str) -> Path:
    safe = name.replace('/', '__')
    return cache_dir / key / part / f'{Path(safe).stem}.npy'


def write_cache_manifest(cache_dir: Path, cfg: Dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'cache_manifest.json'
    expected = {
        'label': cfg['label'],
        'checkpoint': cfg['checkpoint'],
        'checkpoint_sha256': cfg['checkpoint_sha256'],
        'transform': 'resize short side 224 (BICUBIC) + center crop 224, 1 tile',
        'encoder': 'ViT-H/14 frozen, 16x16 tokens, embed 1280',
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('checkpoint_sha256') != expected['checkpoint_sha256']:
            raise RuntimeError(f'cache {cache_dir} belongs to '
                               f'{existing.get("checkpoint_sha256")}, not '
                               f'{expected["checkpoint_sha256"]}; refusing.')
        return
    manifest_path.write_text(json.dumps(expected, indent=2) + '\n', encoding='utf-8')


def verify_cache_manifest(cache_dir: Path, cfg: Dict[str, object]) -> None:
    manifest_path = cache_dir / 'cache_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'{cache_dir} has no cache_manifest.json')
    existing = json.loads(manifest_path.read_text(encoding='utf-8'))
    if existing.get('checkpoint_sha256') != cfg['checkpoint_sha256']:
        raise RuntimeError(f'{cache_dir} was built for '
                           f'{existing.get("checkpoint_sha256")}, not '
                           f'{cfg["checkpoint_sha256"]}; refusing.')


def cache_split_features(cfg: Dict[str, object], root: Path, names: Sequence[str],
                         part: str, cache_dir: Path, device: torch.device,
                         batch_size: int) -> Dict[str, object]:
    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    out_dir = cache_dir / key / part
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cache_manifest(cache_dir / key, cfg)
    encoder = build_encoder(cfg['checkpoint'], device)
    written = skipped = 0
    t0 = time.time()
    for i, name in enumerate(names):
        out_path = _cache_path(cache_dir, key, part, name)
        if out_path.is_file():
            skipped += 1
            continue
        tile = load_image_tile(root, name)
        feats = extract_tile_features(encoder, tile[None], device,
                                      batch_size=batch_size, use_bfloat16=True)
        np.save(out_path, feats.numpy().astype(np.float16))
        written += 1
        if (i + 1) % 500 == 0:
            print(f'  [{cfg["label"]}/{part}] {i + 1}/{len(names)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return {'label': cfg['label'], 'split': part, 'images': len(names),
            'written': written, 'skipped_existing': skipped,
            'seconds': round(time.time() - t0, 1), 'cache_key': key}


def load_cached_split(cache_dir: Path, key: str, part: str,
                      names: Sequence[str]) -> torch.Tensor:
    out = []
    for name in names:
        path = _cache_path(cache_dir, key, part, name)
        if not path.is_file():
            raise FileNotFoundError(f'missing cached features for {part}/{name}')
        out.append(torch.from_numpy(np.load(path).astype(np.float16)))
    return torch.cat(out, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--continued-training',
                        default='Stage 7B continued pretraining (unknown)')
    parser.add_argument('--pretraining-data-version', default='agrifield40k-v1')
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--limit-images', type=int, default=None,
                        help='smoke-test only: truncate each split to N images')
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    split = build_split(root)
    if args.limit_images:
        for part in split:
            split[part] = split[part][:args.limit_images]
    cfg = checkpoint_config(args.checkpoint, args.label,
                            continued_training=args.continued_training,
                            data_version=args.pretraining_data_version)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for part in ('train', 'val', 'test'):
        report = cache_split_features(cfg, root, split[part], part, cache_dir,
                                      device, args.feature_batch_size)
        print(json.dumps(report, indent=2), flush=True)

    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    verify_cache_manifest(cache_dir / key, cfg)
    train_tokens = load_cached_split(cache_dir, key, 'train', split['train'])
    val_tokens = load_cached_split(cache_dir, key, 'val', split['val'])
    test_tokens = load_cached_split(cache_dir, key, 'test', split['test'])
    train_labels = labels_vector(root, split['train'])
    val_labels = labels_vector(root, split['val'])
    test_labels = labels_vector(root, split['test'])

    per_seed = []
    for seed in args.seed:
        print(f'[{cfg["label"]}] seed {seed}: training classifier', flush=True)
        probe = LinearClassificationProbe(num_classes=28)
        t0 = time.time()
        history = train_classifier(probe, train_tokens, train_labels,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   lr=args.lr, weight_decay=args.weight_decay,
                                   seed=seed, device=device, verbose=True)
        val_metrics = evaluate_classifier(probe, val_tokens, val_labels, device)
        test_metrics = evaluate_classifier(probe, test_tokens, test_labels, device)
        per_seed.append({
            'seed': seed,
            'final_train_loss': history['final_loss'],
            'train_seconds': round(time.time() - t0, 1),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
        })
        print(json.dumps({'seed': seed,
                          'test_macro_f1': test_metrics['macro_f1']}, indent=2),
              flush=True)

    payload = {
        'label': cfg['label'],
        'cache_key': key,
        'backbone_checkpoint': cfg['checkpoint'],
        'backbone_checkpoint_sha256': sha256_file(cfg['checkpoint']),
        'continued_training': cfg['continued_training'],
        'pretraining_data_version': cfg['data_version'],
        'protocol': {
            'backbone': 'ViT-H/14 frozen; 224x224 center crop, 16x16 tokens',
            'head': 'Linear(1280 -> 28) on mean-pooled patch tokens',
            'transform': 'RGB, resize short side 224 BICUBIC, center crop',
            'optimizer': 'AdamW',
            'lr': args.lr, 'weight_decay': args.weight_decay,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'seeds': list(args.seed),
            'loss': 'cross-entropy',
            'metric': 'macro-F1 (primary), accuracy (secondary)',
            'val_fraction': VAL_FRACTION, 'val_seed': VAL_SEED,
            'splits': {part: len(names) for part, names in split.items()},
        },
        'seeds': per_seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'result.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())