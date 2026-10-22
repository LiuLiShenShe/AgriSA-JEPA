#!/usr/bin/env python
"""Build a memory-mapped cache of the PhenoBench images and semantic labels.

Why
---
The downstream evaluation reads every PhenoBench image and label from the
spinning disk that holds the dataset.  With 1,407 train + 772 val images this
happens once per evaluated checkpoint -- nine times in this round -- and the
random reads dominate the wall clock.

The labels do not depend on the backbone weights at all, and the images do not
either, so caching them once is strictly a storage decision: the bytes are
verified against the source files and the pixels are identical.

Layout
------
    <out>/train/images.uint8.npy    [N, 1024, 1024, 3] uint8
    <out>/train/labels.int16.npy    [N, 1024, 1024]    int16
    <out>/train/index.json          {"names": [...], "sha256": {...}}
    <out>/val/...

The name order is the official ``sorted()`` order used by
``src.downstream.phenobench_data.list_split_files``.

Usage:
    ./.venv/bin/python -m tools.build_phenobench_cache \
        --root /data/datasets/PhenoBench/PhenoBench-v110 \
        --out /home/test/agrisa_cache/phenobench-v110
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def split_names(root: Path, split: str) -> List[str]:
    image_dir = root / split / 'images'
    return sorted(p.name for p in image_dir.iterdir() if p.suffix == '.png')


def build_split(root: Path, split: str, out: Path) -> Dict[str, object]:
    names = split_names(root, split)
    first = root / split / 'images' / names[0]
    with Image.open(first) as handle:
        width, height = handle.size

    target = out / split
    target.mkdir(parents=True, exist_ok=True)
    images_path = target / 'images.uint8.npy'
    labels_path = target / 'labels.int16.npy'
    images = np.lib.format.open_memmap(images_path, mode='w+', dtype=np.uint8,
                                       shape=(len(names), height, width, 3))
    labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype=np.int16,
                                       shape=(len(names), height, width))
    digests: Dict[str, str] = {}
    t0 = time.time()
    for index, name in enumerate(names):
        image_path = root / split / 'images' / name
        label_path = root / split / 'semantics' / name
        with Image.open(image_path) as handle:
            images[index] = np.asarray(handle.convert('RGB'), dtype=np.uint8)
        with Image.open(label_path) as handle:
            labels[index] = np.asarray(handle, dtype=np.int16)
        digests[name] = sha256_file(image_path)
        if (index + 1) % 200 == 0 or index + 1 == len(names):
            rate = (index + 1) / max(time.time() - t0, 1e-9)
            print(f'  {split} {index + 1}/{len(names)}  {rate:.1f} img/s', flush=True)
    images.flush()
    labels.flush()
    del images, labels

    payload = {
        'split': split,
        'count': len(names),
        'width': width,
        'height': height,
        'names': names,
        'image_sha256': digests,
        'images_sha256': sha256_file(images_path),
        'labels_sha256': sha256_file(labels_path),
        'build_seconds': round(time.time() - t0, 1),
    }
    (target / 'index.json').write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/data/datasets/PhenoBench/PhenoBench-v110')
    parser.add_argument('--out', default='/home/test/agrisa_cache/phenobench-v110')
    parser.add_argument('--splits', default='train,val')
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits.split(','):
        summary[split] = build_split(root, split.strip(), out)
    (out / 'summary.json').write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'image_sha256'}
                    for k, v in summary.items()}, indent=1) + '\n', encoding='utf-8')
    for split, payload in summary.items():
        print(f'{split}: {payload["count"]} images, '
              f'{payload["build_seconds"]}s, '
              f'{payload["images_sha256"][:16]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
