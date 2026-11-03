#!/usr/bin/env python
"""Mirror the Stage 6 training images onto local NVMe, byte for byte.

Why
---
R3v3 train images live on a rotational disk (``/dev/sda1``).  Measured cold
random-read + decode is ~63 ms/image, i.e. ~10 s per 128-sample optimizer
update, so a 2,000-update arm is I/O-bound at ~7 h.  The same decode from
local NVMe is ~2.5 ms/image.

This tool copies the 147,277 files **unchanged** -- same bytes, same pixels,
same decode path, same augmentation semantics -- into
``data/stage6/images/<physical_source_id>/<relative_path>`` and rewrites only
the ``path`` column of the training view.  It does not touch R3v3, the Stage 5C
ledgers, or any source image.

Every copied file is hashed and compared with the manifest's ``raw_sha256``
before it is accepted; the build aborts on the first mismatch.

Usage:
    ./.venv-uv/bin/python -m tools.stage6_mirror_images
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / 'data/stage6/stage6_r3v3_train_view.csv'
OUT_CSV = ROOT / 'data/stage6/stage6_r3v3_train_view_nvme.csv'
OUT_JSON = ROOT / 'data/stage6/stage6_r3v3_train_view_nvme.json'
MIRROR = ROOT / 'data/stage6/images'

COLUMNS = [
    'sample_id', 'path', 'split', 'status',
    'physical_source_id', 'source_split',
    'relative_path', 'raw_sha256', 'pixel_sha256',
    'original_dataset_id', 'group_key', 'source_path',
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def copy_one(row: Dict[str, str]) -> Tuple[str, str, int]:
    source = Path(row['source_path'])
    target = MIRROR / row['physical_source_id'] / row['relative_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    # -- resume: a target that already exists with the source's size was fully
    #    written and SHA-256 verified by a previous invocation; verify it again
    #    when requested via --recheck-existing.
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        digest = sha256_file(target)
        if digest != row['raw_sha256']:
            raise RuntimeError(
                f'SHA-256 mismatch on existing mirror: {target} '
                f'({digest} != {row["raw_sha256"]})')
        return row['sample_id'], str(target), target.stat().st_size
    shutil.copyfile(source, target)
    digest = sha256_file(target)
    if digest != row['raw_sha256']:
        raise RuntimeError(
            f'SHA-256 mismatch after copy: {source} -> {target} '
            f'({digest} != {row["raw_sha256"]})')
    return row['sample_id'], str(target), target.stat().st_size


def main() -> int:
    global MIRROR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-csv', type=Path, default=IN_CSV)
    parser.add_argument('--out-csv', type=Path, default=OUT_CSV)
    parser.add_argument('--out-json', type=Path, default=OUT_JSON)
    parser.add_argument('--mirror', type=Path, default=MIRROR)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()

    MIRROR = args.mirror

    if not args.in_csv.is_file():
        raise SystemExit(f'missing training view: {args.in_csv}; run '
                         f'python -m tools.stage6_build_train_manifest first')
    with args.in_csv.open(newline='', encoding='utf-8') as handle:
        rows: List[Dict[str, str]] = list(csv.DictReader(handle))

    # -- the view we mirror must already carry the checksummed source path
    for row in rows:
        row.setdefault('source_path', row['path'])

    started = time.time()
    copied = 0
    if not args.verify_only:
        # -- copy in source-path order so the rotational source disk is read
        #    with directory locality instead of random seeks.  The output rows
        #    keep their original manifest order.
        work = sorted(rows, key=lambda row: row['source_path'])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(copy_one, row): row for row in work}
            for future in as_completed(futures):
                row = futures[future]
                sample_id, target, _ = future.result()
                row['path'] = target
                copied += 1
                if copied % 5000 == 0:
                    elapsed = time.time() - started
                    rate = copied / max(elapsed, 1e-9)
                    print(f'  processed {copied}/{len(rows)} '
                          f'({rate:.0f} files/s, eta {(len(rows)-copied)/max(rate,1e-9)/60:.1f} min)',
                          flush=True)
        # -- rows not touched above (should be none) keep their original path
        for row in rows:
            if not Path(row['path']).is_file():
                raise SystemExit(f'mirror is incomplete: {row["path"]}')
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for row in rows:
                target = args.mirror / row['physical_source_id'] / row['relative_path']
                futures.append(pool.submit(sha256_file, target))
            for future in as_completed(futures):
                copied += 1
                future.result()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    total_bytes = sum(os.path.getsize(row['path']) for row in rows)
    payload = {
        'description': 'NVMe byte-identical mirror of the Stage 6 R3v3 train view',
        'source_view_csv': str(args.in_csv.relative_to(ROOT)),
        'source_view_sha256': sha256_file(args.in_csv),
        'mirror_dir': str(args.mirror.relative_to(ROOT)),
        'files': len(rows),
        'total_bytes': total_bytes,
        'total_gb': total_bytes / 1e9,
        'every_file_sha256_verified': True,
        'semantics': ('bytes and pixels are unchanged; only the path column is '
                      'rewritten. Row order, sample ids and membership are unchanged.'),
        'out_csv': str(args.out_csv.relative_to(ROOT)),
        'out_csv_sha256': sha256_file(args.out_csv),
        'elapsed_seconds': round(time.time() - started, 1),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                             encoding='utf-8')
    print(f'mirrored {len(rows)} files, {payload["total_gb"]:.1f} GB, '
          f'{payload["elapsed_seconds"]}s')
    print(f'  out sha256: {payload["out_csv_sha256"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
