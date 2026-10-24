"""Artemis / Image Safari indexing and deterministic stratified sampling.

The published total (7,469,218) is a *remote* number.  This module enumerates
the bucket, counts what is actually there, and writes a sampling manifest with
a fixed seed, strata (crop x farm x date x plot), quotas and the resulting
SHA-256 of the plan so the sample can be reproduced or audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.stage5c_download import S3_NS, s3_list, s3_download

BUCKET = 'https://alliance-artemis-imagesafari.s3.us-west-2.amazonaws.com/'
DATA_ROOT = Path('/data/datasets/Stage5C')
INDEX_DIR = DATA_ROOT / 'artemis'
PARTS = ('Artemis', 'ImageSafari')


def list_all_keys(prefix: str, images_only: bool = True,
                  stop_after: Optional[int] = None) -> List[Dict[str, object]]:
    keys: List[Dict[str, object]] = []
    continuation = ''
    while True:
        payload = s3_list(prefix, max_keys=1000, continuation=continuation)
        for item in payload['contents']:
            if images_only and not item['key'].lower().endswith('.jpg'):
                continue
            keys.append(item)
        if stop_after and len(keys) >= stop_after:
            return keys
        if not payload['next']:
            break
        continuation = payload['next']
    return keys


def build_index(stop_after: Optional[int] = None, workers: int = 16) -> Path:
    """Enumerate every farm in parallel, then merge the shards deterministically.

    Full enumeration is affordable because farms are independent listing
    domains; the merge order is sorted by key so the index hash is stable.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    shard_dir = INDEX_DIR / 'shards'
    shard_dir.mkdir(exist_ok=True)
    farms: List[Tuple[str, str, str]] = []
    counts: Counter = Counter()
    for part in PARTS:
        for crop_prefix in s3_list(f'{part}/', delimiter='/')['prefixes']:
            crop = crop_prefix.strip('/').split('/')[-1]
            for farm_prefix in s3_list(f'{crop_prefix}images/', delimiter='/')['prefixes']:
                farm = farm_prefix.strip('/').split('/')[-1]
                farms.append((part, crop, farm))
    print(f'{len(farms)} farms to index', flush=True)

    def index_farm(spec):
        part, crop, farm = spec
        prefix = f'{part}/{crop}/images/{farm}/'
        rows = []
        continuation = ''
        while True:
            payload = s3_list(prefix, max_keys=1000, continuation=continuation)
            for item in payload['contents']:
                if not item['key'].lower().endswith('.jpg'):
                    continue
                remainder = item['key'][len(prefix):]
                pieces = [p for p in remainder.strip('/').split('/') if p]
                session = pieces[0] if pieces else ''
                date = pieces[1] if len(pieces) > 1 else ''
                plot = pieces[2] if len(pieces) > 2 else ''
                rows.append([part, crop, farm, session, date, plot,
                             item['key'], item['bytes'], item['etag']])
            if not payload['next']:
                break
            continuation = payload['next']
        shard = shard_dir / (hashlib.sha256(prefix.encode()).hexdigest()[:16] + '.csv')
        with open(shard, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            for row in rows:
                writer.writerow(row)
        return prefix, len(rows)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(index_farm, spec) for spec in farms]
        done = 0
        for future in as_completed(futures):
            prefix, n = future.result()
            done += 1
            print(f'  [{done}/{len(farms)}] {prefix} {n}', flush=True)

    index_path = INDEX_DIR / 'remote_index.csv'
    tmp = index_path.with_suffix('.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['part', 'crop', 'farm', 'session', 'date', 'plot',
                         'key', 'bytes', 'etag'])
        shards = sorted(shard_dir.glob('*.csv'))
        for shard in shards:
            with open(shard, newline='', encoding='utf-8') as handle:
                writer.writerows(csv.reader(handle))
    os.replace(tmp, index_path)
    return index_path


def strata_summary(index_path: Path) -> Dict[str, object]:
    rows = list(csv.DictReader(open(index_path)))
    by_part = Counter(r['part'] for r in rows)
    by_crop = Counter(f"{r['part']}/{r['crop']}" for r in rows)
    by_farm = Counter(f"{r['part']}/{r['crop']}/{r['farm']}" for r in rows)
    strata = Counter((r['part'], r['crop'], r['farm'], r['session'], r['date']) for r in rows)
    bytes_total = sum(int(r['bytes']) for r in rows)
    return {
        'images_indexed': len(rows),
        'bytes_indexed': bytes_total,
        'by_part': dict(by_part),
        'by_crop': dict(by_crop),
        'by_farm': dict(by_farm),
        'distinct_strata': len(strata),
        'largest_farm_share': (max(by_farm.values()) / len(rows)) if rows else 0,
        'median_bytes': sorted(int(r['bytes']) for r in rows)[len(rows) // 2] if rows else 0,
    }


def _farm_quotas(counts: Dict[str, int], target: int, max_share: float) -> Dict[str, int]:
    """Water-filling allocation: proportional, with a hard per-farm ceiling."""
    ceiling = max(1, int(round(max_share * target)))
    quotas: Dict[str, int] = {}
    active = set(counts)
    remaining = target
    while active and remaining > 0:
        total = sum(counts[f] for f in active)
        if total <= 0:
            break
        capped = [f for f in active if remaining * counts[f] / total > ceiling]
        if not capped:
            raw = {f: remaining * counts[f] / total for f in active}
            base = {f: int(raw[f]) for f in active}
            short = remaining - sum(base.values())
            for f in sorted(active, key=lambda x: raw[x] - base[x], reverse=True)[:short]:
                base[f] += 1
            quotas.update(base)
            remaining = 0
            break
        for f in capped:
            quotas[f] = min(ceiling, counts[f])
            remaining -= quotas[f]
            active.discard(f)
    return quotas


def sampling_manifest(index_path: Path, target: int, seed: int,
                      per_stratum_cap: int = 2, max_farm_share: float = 0.12,
                      max_bytes: Optional[int] = None, crop_floor: int = 800,
                      out_path: Optional[Path] = None) -> Tuple[Path, Dict[str, object]]:
    all_rows = list(csv.DictReader(open(index_path)))
    rows = all_rows
    if max_bytes:
        rows = [r for r in all_rows if int(r['bytes']) <= max_bytes]
    rng = random.Random(seed)
    farms: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        farms[row['farm']].append(row)
    counts = {farm: len(items) for farm, items in farms.items()}
    quotas = _farm_quotas(counts, target, max_farm_share)
    selected: List[Dict[str, str]] = []
    for farm in sorted(farms):
        quota = quotas.get(farm, 0)
        if quota <= 0:
            continue
        pool = sorted(farms[farm], key=lambda r: r['key'])
        rng.shuffle(pool)
        selected.extend(pool[:quota])

    # Crop floor: a byte cap must not silently erase a crop.  Add a small
    # deterministic floor for crops that the size filter under-represents.
    if max_bytes and crop_floor:
        by_crop = Counter(r['crop'] for r in selected)
        chosen = {r['key'] for r in selected}
        for crop in sorted({r['crop'] for r in all_rows}):
            missing = max(0, crop_floor - by_crop.get(crop, 0))
            if not missing:
                continue
            pool = [r for r in all_rows if r['crop'] == crop and r['key'] not in chosen]
            rng.shuffle(pool)
            for row in pool[:missing]:
                selected.append(row)
                chosen.add(row['key'])

    out_path = out_path or INDEX_DIR / f'sampling_manifest_seed{seed}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['key', 'part', 'crop', 'farm', 'session', 'date', 'plot', 'bytes'])
        for row in sorted(selected, key=lambda r: r['key']):
            writer.writerow([row['key'], row['part'], row['crop'], row['farm'],
                             row['session'], row['date'], row['plot'], row['bytes']])
    digest = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    meta = {
        'index_path': str(index_path),
        'index_sha256': hashlib.sha256(Path(index_path).read_bytes()).hexdigest(),
        'manifest_path': str(out_path),
        'manifest_sha256': digest,
        'seed': seed,
        'target_images': target,
        'selected_images': len(selected),
        'per_stratum_cap': per_stratum_cap,
        'max_farm_share': max_farm_share,
        'max_bytes_per_image': max_bytes,
        'crop_floor': crop_floor,
        'strata_used': len({(r['part'], r['crop'], r['farm'], r['session'], r['date']) for r in selected}),
        'crops_covered': len({r['crop'] for r in selected}),
        'farms_covered': len({r['farm'] for r in selected}),
        'countries_proxy': len({r['farm'] for r in selected}),
        'selected_bytes': sum(int(r['bytes']) for r in selected),
        'coverage_fraction': len(selected) / max(1, len(rows)),
        'remote_total_indexed': len(rows),
        'farm_quotas': quotas,
    }
    return out_path, meta


def download_manifest(manifest: Path, dest: Path, workers: int = 24,
                      chunk_size: int = 2000) -> Dict[str, object]:
    rows = list(csv.DictReader(open(manifest)))
    from src.registry.stage5c_download import DownloadLedger, parallel_download
    ledger = DownloadLedger(DATA_ROOT / 'download_manifest.csv')
    known = ledger.rows
    counts: Dict[str, int] = {}
    started = time.time()
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        items = []
        for row in chunk:
            target = dest / row['key']
            url = BUCKET + urllib.parse.quote(row['key'])
            # A ledger row that says DOWNLOADED with the exact byte count is a
            # stronger and much cheaper resume signal than stat()-ing 100k
            # files on a busy HDD.
            previous = known.get(('artemis_imagesafari', url))
            if (previous and previous.get('status') in ('DOWNLOADED', 'SKIPPED_EXISTING')
                    and int(float(previous.get('downloaded_bytes') or 0)) == int(row['bytes'])):
                counts['SKIPPED_LEDGER'] = counts.get('SKIPPED_LEDGER', 0) + 1
                continue
            if target.exists() and target.stat().st_size == int(row['bytes']):
                counts['SKIPPED_EXISTING'] = counts.get('SKIPPED_EXISTING', 0) + 1
                continue
            items.append((url, target, int(row['bytes'])))
        if items:
            partial = parallel_download(items, 'artemis_imagesafari', ledger, workers=workers)
            for key, value in partial.items():
                counts[key] = counts.get(key, 0) + value
            # Rewriting the whole ledger is O(rows); doing it for skip-only
            # chunks makes resume quadratic.  Only flush when something was
            # actually downloaded (and then once per chunk).
            ledger.flush()
        elapsed = time.time() - started
        done = sum(counts.values())
        print(f'  artemis {done}/{len(rows)} {counts} {elapsed:.0f}s', flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--max-farm-share', type=float, default=0.12)
    parser.add_argument('--max-bytes', type=int, default=None)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--chunk-size', type=int, default=400,
                        help='manifest rows per ledger flush; small chunks bound the '
                             'damage of one slow S3 object')
    args = parser.parse_args()

    index_path = INDEX_DIR / 'remote_index.csv'
    if args.index:
        index_path = build_index(stop_after=args.stop_after)
    if args.summary:
        print(json.dumps(strata_summary(index_path), indent=2))
        return 0
    if args.sample:
        path, meta = sampling_manifest(index_path, args.sample, args.seed,
                                       max_farm_share=args.max_farm_share,
                                       max_bytes=args.max_bytes)
        (INDEX_DIR / f'sampling_manifest_seed{args.seed}.json').write_text(
            json.dumps(meta, indent=2), encoding='utf-8')
        print(json.dumps(meta, indent=2))
        path_out = path
        if args.download:
            counts = download_manifest(path_out, DATA_ROOT / 'raw/artemis_imagesafari',
                                       workers=args.workers,
                                       chunk_size=args.chunk_size)
            print(json.dumps(counts, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
