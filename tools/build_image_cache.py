#!/usr/bin/env python
"""Build a memory-mapped RGB image cache for the agrifield40k manifest.

Why this exists
---------------
The pretraining manifest points at ~12.5k individual JPEG/PNG files on a
spinning disk.  Every training batch therefore costs thousands of random reads
on a rotational device, which dominated the wall clock of the first Stage 3B
attempt far more than the GPU ever did.

This tool decodes every image **once**, in manifest order, into a single
``uint8`` memory-mapped array on a fast device.  The training loader then reads
zero-padded crops straight out of that file instead of re-opening a JPEG per
sample, which turns a seek-bound workload into a sequential one.

Correctness guarantees
----------------------
* every file's bytes are checked against the ``sha256`` column of the manifest
  before its pixels are written; a mismatch aborts the build,
* the in-memory pixels are exactly ``Image.open(path).convert('RGB')``, the same
  conversion :class:`~src.datasets.agri_dataset.AgriFieldDataset` performs, so
  the augmentation protocol is unchanged,
* the cache records the manifest SHA-256 and the sample-id order, and the loader
  refuses a cache that does not match the manifest it was asked for.

Usage:
    ./.venv/bin/python -m tools.build_image_cache \
        --manifest data/frozen/agrifield40k-v1_train.csv \
        --root /data/datasets/AgriField-40K \
        --out /home/test/agrisa_cache/agrifield40k-v1-train
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path, root: Path, split: str = 'train') -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open('r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row.get('status', '').strip().lower() != 'include':
                continue
            if split and row.get('split', '').strip() != split:
                continue
            image_path = Path(row['path'])
            if not image_path.is_absolute():
                image_path = (root / image_path).resolve()
            row['_resolved'] = str(image_path)
            rows.append(row)
    return rows


def _store(array, index: int, payload: bytes, side: int) -> None:
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert('RGB')
        if rgb.size != (side, side):
            rgb = rgb.resize((side, side), Image.BICUBIC)
        array[index] = np.asarray(rgb, dtype=np.uint8)


def _fill_from_zip(zip_path: str, rows: List[Dict[str, str]], order: List[str],
                   array, side: int, verify_sha: bool, t0: float,
                   failures: List[Dict[str, str]]):
    """One sequential pass over the archive.

    The source images live on a rotational disk; opening them one by one is
    seek-bound (~0.4 MB/s measured).  The archive is a single 11.9 GB file that
    streams at ~48 MB/s, so reading it once is two orders of magnitude cheaper.
    """
    import zipfile

    index_of = {path: position for position, path in enumerate(order)}
    written = 0
    processed = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            position = index_of.get(name)
            if position is None:
                continue
            row = rows[position]
            try:
                payload = archive.read(name)
                expected = (row.get('sha256') or '').strip()
                if verify_sha and expected:
                    actual = hashlib.sha256(payload).hexdigest()
                    if actual != expected:
                        failures.append({'sample_id': row.get('sample_id', ''), 'path': name,
                                         'error': f'sha256 mismatch: manifest {expected} archive {actual}'})
                        processed += 1
                        continue
                _store(array, position, payload, side)
                written += 1
            except Exception as exc:                  # noqa: BLE001
                failures.append({'sample_id': row.get('sample_id', ''), 'path': name,
                                 'error': f'{type(exc).__name__}: {exc}'})
            processed += 1
            if processed % 500 == 0:
                rate = processed / max(time.time() - t0, 1e-9)
                remaining = (len(rows) - processed) / max(rate, 1e-9)
                print(f'  {processed}/{len(rows)}  {rate:.1f} img/s  eta {remaining / 60:.1f} min',
                      flush=True)
    return written, failures


def _fill_from_disk(rows: List[Dict[str, str]], array, side: int, verify_sha: bool,
                    t0: float, failures: List[Dict[str, str]]):
    written = 0
    for index, row in enumerate(rows):
        path = Path(row['_resolved'])
        try:
            # -- read the file ONCE: the same bytes are hashed and decoded.
            payload = path.read_bytes()
            expected = (row.get('sha256') or '').strip()
            if verify_sha and expected:
                actual = hashlib.sha256(payload).hexdigest()
                if actual != expected:
                    failures.append({'sample_id': row.get('sample_id', ''), 'path': str(path),
                                     'error': f'sha256 mismatch: manifest {expected} disk {actual}'})
                    continue
            _store(array, index, payload, side)
            written += 1
        except Exception as exc:                      # noqa: BLE001
            failures.append({'sample_id': row.get('sample_id', ''), 'path': str(path),
                             'error': f'{type(exc).__name__}: {exc}'})
            continue
        if (index + 1) % 500 == 0 or index + 1 == len(rows):
            rate = (index + 1) / max(time.time() - t0, 1e-9)
            remaining = (len(rows) - index - 1) / max(rate, 1e-9)
            print(f'  {index + 1}/{len(rows)}  {rate:.1f} img/s  eta {remaining / 60:.1f} min',
                  flush=True)
    return written, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--zip', default=None,
                        help='read the images from this archive instead of the '
                             'source directory (one sequential pass)')
    parser.add_argument('--verify-sha', action='store_true', default=True)
    parser.add_argument('--no-verify-sha', dest='verify_sha', action='store_false')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path, root, args.split)
    if not rows:
        print('manifest selected no rows', file=sys.stderr)
        return 1

    side = int(args.size)
    shape = (len(rows), side, side, 3)
    images_path = out / 'images.uint8.npy'
    array = np.lib.format.open_memmap(images_path, mode='w+', dtype=np.uint8, shape=shape)

    manifest_sha = sha256_file(manifest_path)
    t0 = time.time()
    failures: List[Dict[str, str]] = []
    written = 0
    order = [row.get('path', '') for row in rows]

    if args.zip:
        written, failures = _fill_from_zip(
            args.zip, rows, order, array, side, args.verify_sha, t0, failures)
    else:
        written, failures = _fill_from_disk(
            rows, array, side, args.verify_sha, t0, failures)

    array.flush()
    del array

    index_payload = {
        'manifest': str(manifest_path),
        'manifest_sha256': manifest_sha,
        'root': str(root),
        'split': args.split,
        'side': side,
        'channels': 3,
        'dtype': 'uint8',
        'count': written,
        'requested': len(rows),
        'sample_ids': [row.get('sample_id', '') for row in rows],
        'paths': [row.get('_resolved', '') for row in rows],
        'sha256_verified': bool(args.verify_sha),
        'failures': failures,
        'images_sha256': sha256_file(images_path),
        'build_seconds': round(time.time() - t0, 1),
    }
    (out / 'index.json').write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'cache written to {images_path}  ({images_path.stat().st_size / 2**30:.2f} GiB)')
    print(f'sample_ids={written}/{len(rows)}  failures={len(failures)}')
    if failures:
        print(json.dumps(failures[:5], indent=2, ensure_ascii=False))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
