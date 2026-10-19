#!/usr/bin/env python
"""Check whether any AgriField frozen image also appears in PhenoBench.

The publisher's own compilation already excluded 9,606 PhenoBench-derived rows
from the AgriField freeze (``data/source_registry.json``).  This tool re-checks
that independently, on three levels:

1. **File identity** — SHA-256 of every frozen AgriField image vs every
   PhenoBench image.
2. **Decoded-pixel identity** — SHA-256 of the decoded RGB array.
3. **Perceptual near-duplicates** — 64-bit dHash Hamming distance.

It also reports PhenoBench's acquisition dates and scene names and the dates
encoded in AgriField's excluded ``pnb_`` rows, so a same-scene-different-crop
overlap that no hash can see is visible to a human.  AgriField's ``pnb_`` rows
are 1024x1024 *crops* of larger PhenoBench field images with explicit x/y
offsets, so exact-hash equality is expected to be zero even where the scenes
overlap.

READ-ONLY.  Usage:
    ./.venv/bin/python -m tools.phenobench_leakage_check \
        --agrifield data/frozen/agrifield40k-v1_train.csv \
                    data/frozen/agrifield40k-v1_val.csv \
        --phenobench /data/datasets/PhenoBench/PhenoBench-v110 \
        --out reports/phenobench_leakage_check.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image
import numpy as np

PHENOBENCH_NAME_RE = re.compile(r'^(?P<date>\d{2}-\d{2})_(?P<frame>\d+)_(?P<plant>P\d+)$')
PNB_DATE_RE = re.compile(r'^(\d{4})(\d{2})(\d{2})_')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(path: Path, size: Optional[int] = None) -> str:
    with Image.open(path) as handle:
        arr = np.asarray(handle.convert('RGB'))
    if size is not None:
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dhash64(path: Path, size: int = 8) -> str:
    """64-bit difference hash as 16 hex chars (same layout as the manifests)."""
    with Image.open(path) as handle:
        arr = np.asarray(handle.convert('L').resize((size + 1, size), Image.BILINEAR),
                         dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def hamming64(a: str, b: str) -> Optional[int]:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count('1')
    except (TypeError, ValueError):
        return None


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def agrifield_index(manifests: Sequence[Path], compute_dhash: bool = True
                    ) -> Dict[str, object]:
    rows: List[Dict[str, str]] = []
    for path in manifests:
        rows.extend(read_manifest(path))
    file_hash = collections.defaultdict(list)
    pixel_hash = collections.defaultdict(list)
    dhash = collections.defaultdict(list)
    for row in rows:
        sid = row['sample_id']
        if row.get('sha256'):
            file_hash[row['sha256']].append(sid)
        if row.get('pixel_sha256'):
            pixel_hash[row['pixel_sha256']].append(sid)
        if row.get('dhash64'):
            dhash[row['dhash64']].append(sid)
    return {
        'rows': len(rows),
        'file_hash': dict(file_hash),
        'pixel_hash': dict(pixel_hash),
        'dhash': dict(dhash),
    }


def phenobench_index(root: Path, splits: Sequence[str] = ('train', 'val'),
                     compute_dhash: bool = True) -> Dict[str, object]:
    images: Dict[str, str] = {}
    pixels: Dict[str, str] = {}
    dh: Dict[str, str] = {}
    count = 0
    dates: collections.Counter = collections.Counter()
    scenes: collections.Counter = collections.Counter()
    scanned = []
    for split in splits:
        image_dir = root / split / 'images'
        if not image_dir.is_dir():
            continue
        scanned.append(split)
        for path in sorted(image_dir.glob('*.png')):
            rel = f'{split}/{path.name}'
            images[sha256_file(path)] = rel
            pixels[pixel_sha256(path)] = rel
            if compute_dhash:
                dh[dhash64(path)] = rel
            count += 1
            m = PHENOBENCH_NAME_RE.match(path.stem)
            if m:
                dates[m.group('date')] += 1
                scenes[m.group('plant')] += 1
    return {
        'images': images, 'pixel_hashes': pixels, 'dhash': dh,
        'count': count, 'splits_scanned': scanned,
        'acquisition_dates': dict(sorted(dates.items())),
        'scene_names': dict(scenes.most_common(20)),
    }


def compare(agri: Dict[str, object], pb: Dict[str, object],
            dhash_threshold: int = 8) -> Dict[str, object]:
    exact = sorted({sid for h in agri['file_hash'] if h in pb['images']
                    for sid in agri['file_hash'][h]})
    pixel = sorted({sid for h in agri['pixel_hash'] if h in pb['pixel_hashes']
                    for sid in agri['pixel_hash'][h]})
    near = []
    histogram: collections.Counter = collections.Counter()
    for h, ids in agri['dhash'].items():
        for ph, rel in pb['dhash'].items():
            distance = hamming64(h, ph)
            if distance is None:
                continue
            histogram[distance] += 1
            if distance <= dhash_threshold:
                near.append({'distance': distance,
                             'agrifield_samples': ids[:5],
                             'phenobench_image': rel})
    near.sort(key=lambda e: e['distance'])
    return {
        'agrifield_rows_checked': agri['rows'],
        'phenobench_images_checked': pb['count'],
        'exact_file_sha256_matches': len(exact),
        'exact_file_sha256_examples': exact[:10],
        'decoded_pixel_sha256_matches': len(pixel),
        'decoded_pixel_sha256_examples': pixel[:10],
        'perceptual_near_duplicates': len(near),
        'perceptual_near_duplicate_examples': near[:10],
        'dhash_threshold': dhash_threshold,
        # Sensitivity matters: a 64-bit dHash over a 9x8 downsample of a
        # top-down field image has few effective bits, so a small Hamming
        # radius has a high false-positive rate.  The count at stricter radii
        # says how much of the threshold-8 set is plausibly real.
        'dhash_distance_histogram': dict(sorted(histogram.items())),
        'near_duplicates_at_stricter_thresholds': {
            str(t): sum(c for d, c in histogram.items() if d <= t)
            for t in (0, 1, 2, 3, 4, 6, 8)
        },
    }


def excluded_phenobench_rows(all_samples: Path) -> Dict[str, object]:
    """Dates and split names of the AgriField rows already excluded as PhenoBench."""
    rows = [r for r in read_manifest(all_samples) if r.get('source') == 'PhenoBench']
    dates: collections.Counter = collections.Counter()
    for row in rows:
        m = PNB_DATE_RE.match(row.get('original_id', ''))
        if m:
            dates[f'{m.group(1)}-{m.group(2)}-{m.group(3)}'] += 1
    return {
        'rows': len(rows),
        'status': dict(collections.Counter(r.get('status') for r in rows)),
        'acquisition_dates': dict(sorted(dates.items())),
        'original_split': dict(collections.Counter(r.get('original_split') for r in rows)),
        'naming': ('1024x1024 crops of larger field images with explicit x/y offsets '
                   '(e.g. 20200425_val_nh_P0024183_4178_1024), from the unlabeled '
                   'patch release'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agrifield', nargs='+', required=True)
    parser.add_argument('--phenobench', required=True)
    parser.add_argument('--all-samples', default='data/all_samples.csv')
    parser.add_argument('--out', default=None)
    parser.add_argument('--dhash-threshold', type=int, default=8)
    parser.add_argument('--no-dhash', action='store_true')
    args = parser.parse_args()

    compute = not args.no_dhash
    agri = agrifield_index([Path(p) for p in args.agrifield], compute)
    pb = phenobench_index(Path(args.phenobench), compute_dhash=compute)
    excluded = (excluded_phenobench_rows(Path(args.all_samples))
                if Path(args.all_samples).is_file() else None)
    result = {
        'agrifield_manifests': args.agrifield,
        'phenobench_root': args.phenobench,
        'phenobench': {
            'count': pb['count'],
            'splits_scanned': pb['splits_scanned'],
            'acquisition_dates': pb['acquisition_dates'],
            'scene_names': pb['scene_names'],
        },
        'agrifield_excluded_phenobench_rows': excluded,
        'comparison': compare(agri, pb, args.dhash_threshold),
        'notes': [
            'AgriField v0/v1 already excludes every row the publisher attributed '
            'to PhenoBench (data/source_registry.json, 9,606 rows).',
            'A zero hash match does not prove disjoint scenes: AgriField pnb_ rows '
            'are 1024x1024 crops of larger PhenoBench images, so the same field '
            'and day can appear with different pixels.',
            'Overlapping acquisition dates are the real risk signal; see '
            'agrifield_excluded_phenobench_rows.acquisition_dates vs '
            'phenobench.acquisition_dates.',
            'All pnb_ rows carry status=exclude in both v0 and v1, so they are in '
            'neither the frozen train nor the frozen val split.',
        ],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())