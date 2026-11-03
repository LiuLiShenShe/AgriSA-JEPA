#!/usr/bin/env python
"""Download every legally reachable Stage 5C source, then verify the archives.

Usage::

    python tools/stage5c_download_all_available.py --list
    python tools/stage5c_download_all_available.py --source agrifm_ref1
    python tools/stage5c_download_all_available.py --all --workers 24

Nothing is deleted and nothing is fetched twice: an archive whose size matches
the ledger is reported as ``SKIPPED_EXISTING``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.stage5c_download import (DownloadLedger, dataverse_files,
                                           download_file, download_kaggle,
                                           mendeley_files, parallel_download,
                                           remote_size, sha256_file)
from src.registry.stage5c_source_registry import ACCESS, all_rows

DATA_ROOT = Path('/data/datasets/Stage5C')
ARCHIVES = DATA_ROOT / 'archives'
RAW = DATA_ROOT / 'raw'
LEDGER = DATA_ROOT / 'download_manifest.csv'

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def source_row(source_id: str):
    for row in all_rows():
        ref = row.original_ref_ids[0] if row.original_ref_ids else ''
        if row.physical_source_id == source_id or ref == source_id:
            return row
    raise KeyError(source_id)


def plan_for(source_id: str):
    row = source_row(source_id)
    ref = row.original_ref_ids[0] if row.original_ref_ids else source_id
    return ACCESS.get(ref)


def extract_zip(archive: Path, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / f'.extracted.{archive.name}'
    if marker.exists():
        return {'status': 'SKIPPED_EXISTING', 'dest': str(dest), 'archive': archive.name}
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            return {'status': 'CORRUPT', 'bad_member': bad}
        bundle.extractall(dest)
    marker.write_text(archive.name + '\n', encoding='utf-8')
    return {'status': 'EXTRACTED', 'dest': str(dest)}


def count_images(root: Path) -> int:
    return sum(1 for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def run_source(source_id: str, ledger: DownloadLedger, workers: int = 24) -> dict:
    row = source_row(source_id)
    plan = plan_for(source_id)
    method = row.access_method
    out = {'source_id': row.physical_source_id, 'method': method, 'actions': []}
    if plan is None or method in ('none', 'kaggle_dsv', 'kaggle_competition'):
        out['actions'].append({'status': 'BLOCKED', 'reason': row.failure_reason or 'no access method'})
        return out

    if method == 'local_reuse':
        root = Path(row.local_root)
        out['actions'].append({'status': 'LOCAL_PRESENT' if root.exists() else 'LOCAL_MISSING',
                               'root': str(root), 'images': count_images(root) if root.exists() else 0})
        return out

    if method == 'kaggle_dataset':
        archive = ARCHIVES / row.physical_source_id / (plan.archive_name or f'{row.physical_source_id}.zip')
        record = download_kaggle(plan.ref, archive, expected_bytes=plan.expected_bytes)
        record['source_id'] = row.physical_source_id
        ledger.record(row.physical_source_id, record)
        out['actions'].append(record)
        if record['status'] in ('DOWNLOADED', 'SKIPPED_EXISTING'):
            extraction = extract_zip(archive, RAW / row.physical_source_id)
            out['actions'].append(extraction)
            out['images'] = count_images(RAW / row.physical_source_id)
        return out

    if method == 'mendeley_zip':
        files = mendeley_files(plan.ref)
        archive = ARCHIVES / row.physical_source_id / (plan.archive_name or f'{row.physical_source_id}.zip')
        items = [(f['url'], archive, f['bytes']) for f in files[:1]]
        counts = parallel_download(items, row.physical_source_id, ledger, workers=1)
        out['actions'].append(counts)
        if archive.exists():
            extraction = extract_zip(archive, RAW / row.physical_source_id)
            out['actions'].append(extraction)
            out['images'] = count_images(RAW / row.physical_source_id)
        return out

    if method == 'dataverse':
        files = dataverse_files(plan.ref)
        items = []
        for entry in files:
            if not entry['name'].lower().endswith(('.zip', '.7z', '.tar', '.gz')):
                continue
            dest = ARCHIVES / row.physical_source_id / entry['name']
            items.append((entry['url'], dest, entry['bytes']))
        counts = parallel_download(items, row.physical_source_id, ledger, workers=2)
        out['actions'].append(counts)
        for url, dest, _ in items:
            if dest.exists():
                if dest.suffix.lower() == '.zip':
                    out['actions'].append(extract_zip(dest, RAW / row.physical_source_id))
                else:
                    out['actions'].append({'status': 'UNEXTRACTED',
                                           'file': str(dest), 'note': 'non-zip archive recorded'})
        out['images'] = count_images(RAW / row.physical_source_id)
        return out

    if method == 'mendeley':
        files = [f for f in mendeley_files(plan.ref) if f['url']]
        dest_root = RAW / row.physical_source_id
        items = [(f['url'], dest_root / f['name'], f['bytes']) for f in files]
        counts: dict = {}
        batch = 500
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            partial = parallel_download(chunk, row.physical_source_id, ledger, workers=workers)
            for key, value in partial.items():
                counts[key] = counts.get(key, 0) + value
            ledger.flush()
            print(f'  {row.physical_source_id}: {start + len(chunk)}/{len(items)} {counts}', flush=True)
        out['actions'].append(counts)
        out['images'] = count_images(dest_root)
        return out

    out['actions'].append({'status': 'NO_HANDLER', 'method': method})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    rows = all_rows()
    available = [(r.original_ref_ids[0] if r.original_ref_ids else r.physical_source_id,
                  r.physical_source_id, r.access_method)
                 for r in rows if r.access_method in
                 ('kaggle_dataset', 'dataverse', 'mendeley', 'mendeley_zip', 'local_reuse')]
    if args.list:
        for ref, phys, method in available:
            print(f'{ref:14s} {phys:24s} {method}')
        return 0

    targets = []
    if args.all:
        targets = [phys for _, phys, _ in available]
    elif args.source:
        wanted = set(args.source)
        for ref, phys, _ in available:
            if ref in wanted or phys in wanted:
                targets.append(phys)
    if not targets:
        parser.error('nothing to do: pass --source or --all')

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = DownloadLedger(LEDGER)
    results = []
    for phys in targets:
        print(f'==> {phys}', flush=True)
        result = run_source(phys, ledger, workers=args.workers)
        results.append(result)
        ledger.flush()
        print(json.dumps(result, default=str)[:500], flush=True)
    ledger.flush()
    out_path = ROOT / 'reports/stage5c_download_run.json'
    previous = []
    if out_path.exists():
        try:
            previous = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            previous = []
    out_path.write_text(json.dumps(previous + results, indent=2, default=str), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
