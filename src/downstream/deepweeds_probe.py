#!/usr/bin/env python
"""Stage 8: DeepWeeds 9-class classification, frozen ViT-H + linear head.

Split (frozen in Phase 2): fold 0 of the official 5-fold CV, used verbatim from
``data/holdout_raw/deepweeds_labels/``:

* train ``train_subset0.csv`` + ``val_subset0.csv`` (10,501 + 3,501),
* test ``test_subset0.csv`` (3,507), evaluated once at the end.

Every image is transformed identically: RGB, resize short side to 224
(BICUBIC), center crop 224x224, then the frozen encoder's 16x16 patch tokens
are mean-pooled and classified by a single ``Linear(1280 -> 9)``.  Primary
metric is macro-F1 on test; accuracy is secondary.

The encoder features are cached per (label, checkpoint SHA) exactly like the
segmentation probes, and a cache without a matching manifest is refused.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 ./.venv-uv/bin/python -m src.downstream.deepweeds_probe \
        --checkpoint ... --label A_p1 \
        --out-dir /data/agrisa_jepa_runs/stage8/deepweeds_cls/A_p1_seed0 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from src.downstream.classifier_probe import (LinearClassificationProbe,
                                             evaluate_classifier,
                                             train_classifier)
from src.downstream.features import (build_encoder, cache_key,
                                     checkpoint_config, extract_tile_features,
                                     sha256_file)

DEFAULT_LABELS = str(Path(__file__).resolve().parent.parent /
                     'data/holdout_raw/deepweeds_labels')
DEFAULT_IMAGES = '/data/datasets/DeepWeeds/images'
DEFAULT_CACHE = 'runs/stage8_deepweeds/cache'
NUM_CLASSES = 9

# official fold-0 split files
TRAIN_CSV = 'train_subset0.csv'
VAL_CSV = 'val_subset0.csv'
TEST_CSV = 'test_subset0.csv'


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def read_names(labels_root: Path, csv_name: str) -> List[str]:
    path = labels_root / csv_name
    with path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        names = [row['Filename'] for row in reader]
    if len(names) != len(set(names)):
        raise ValueError(f'{csv_name}: duplicate filenames')
    return names


def build_split(labels_root: Path) -> Dict[str, List[str]]:
    train = read_names(labels_root, TRAIN_CSV)
    val = read_names(labels_root, VAL_CSV)
    test = read_names(labels_root, TEST_CSV)
    known = set(train) | set(val) | set(test)
    with (labels_root / 'labels.csv').open('r', encoding='utf-8') as handle:
        all_names = {row['Filename'] for row in csv.DictReader(handle)}
    if known != all_names:
        missing = sorted(all_names - known)
        raise ValueError(f'fold-0 does not cover all labels.csv names '
                         f'({len(missing)} missing, e.g. {missing[:3]})')
    overlap = (set(train) & (set(val) | set(test))) | (set(val) & set(test))
    if overlap:
        raise ValueError(f'fold-0 splits overlap: {sorted(overlap)[:5]}')
    return {'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}


def load_image_tile(images_root: Path, name: str) -> np.ndarray:
    """RGB-uint8 ``[224,224,3]`` tile: resize short side 224, center crop."""
    path = images_root / name
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


def labels_vector(labels_root: Path, names: Sequence[str]) -> torch.Tensor:
    by_name = {}
    with (labels_root / 'labels.csv').open('r', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            by_name[row['Filename']] = int(row['Label'])
    out = torch.tensor([by_name[n] for n in names], dtype=torch.int64)
    if int(out.min()) < 0 or int(out.max()) >= NUM_CLASSES:
        raise ValueError(f'label outside 0..{NUM_CLASSES - 1}')
    return out


# --------------------------------------------------------------------------- #
# Feature extraction with cache (1 tile per image)
# --------------------------------------------------------------------------- #

def _cache_path(cache_dir: Path, key: str, part: str, name: str) -> Path:
    return cache_dir / key / part / f'{Path(name).stem}.npy'


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
            raise RuntimeError(
                f'cache {cache_dir} belongs to '
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


def cache_split_features(cfg: Dict[str, object], images_root: Path,
                         names: Sequence[str], part: str, cache_dir: Path,
                         device: torch.device, batch_size: int) -> Dict[str, object]:
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
        tile = load_image_tile(images_root, name)
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


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--continued-training',
                        default='Stage 7B continued pretraining (unknown)')
    parser.add_argument('--pretraining-data-version', default='agrifield40k-v1')
    parser.add_argument('--labels', default=DEFAULT_LABELS)
    parser.add_argument('--images-root', default=DEFAULT_IMAGES)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--feature-batch-size', type=int, default=16)
    args = parser.parse_args()

    labels_root = Path(args.labels)
    images_root = Path(args.images_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    split = build_split(labels_root)
    cfg = checkpoint_config(args.checkpoint, args.label,
                            continued_training=args.continued_training,
                            data_version=args.pretraining_data_version)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for part in ('train', 'val', 'test'):
        report = cache_split_features(cfg, images_root, split[part], part,
                                      cache_dir, device, args.feature_batch_size)
        print(json.dumps(report, indent=2), flush=True)

    key = cache_key(cfg['label'], cfg['checkpoint_sha256'])
    verify_cache_manifest(cache_dir / key, cfg)
    train_tokens = load_cached_split(cache_dir, key, 'train', split['train'])
    val_tokens = load_cached_split(cache_dir, key, 'val', split['val'])
    test_tokens = load_cached_split(cache_dir, key, 'test', split['test'])
    train_labels = labels_vector(labels_root, split['train'])
    val_labels = labels_vector(labels_root, split['val'])
    test_labels = labels_vector(labels_root, split['test'])

    per_seed = []
    for seed in args.seed:
        print(f'[{cfg["label"]}] seed {seed}: training classifier', flush=True)
        probe = LinearClassificationProbe(num_classes=NUM_CLASSES)
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
            'head': 'Linear(1280 -> 9) on mean-pooled patch tokens',
            'transform': 'RGB, resize short side 224 BICUBIC, center crop',
            'optimizer': 'AdamW',
            'lr': args.lr, 'weight_decay': args.weight_decay,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'seeds': list(args.seed),
            'loss': 'cross-entropy',
            'metric': 'macro-F1 (primary), accuracy (secondary)',
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