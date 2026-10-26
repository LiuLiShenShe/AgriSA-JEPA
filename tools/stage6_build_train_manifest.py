#!/usr/bin/env python
"""Build the Stage 6 *training view* of the frozen R3v3 train split.

Why this exists
---------------
``AgriFieldDataset`` expects a manifest with ``sample_id``, ``path``,
``split`` and ``status`` columns (the Stage 3/4 ``agrifield40k`` schema).
The frozen Stage 5E.1 artefact
``data/corpus_v4/R3v3_license_verified_train.csv`` uses a different schema
(``image_id``, ``relative_path``, ``split``, no ``status``) and does not carry
an absolute path.

This tool writes a *derived view*, not a new corpus:

* it never edits, reorders or rewrites the R3v3 CSV;
* row order is preserved exactly, so manifest row ``i`` is R3v3 row ``i``;
* membership is unchanged (all 147,277 ``stage5e_train_eligible`` rows);
* absolute paths come from the versioned Stage 5C per-source ledgers
  (``raw_path``), joined on ``image_id``;
* every row's file is checked to exist before the view is written.

The two ``TRAIN_ONLY_GROUP_UNKNOWN`` rows are part of the train split (the
R3v3 manifest counts them under ``train`` as ``train_only_group_unknown``);
they are labelled ``split=train`` here and their original value is preserved
in ``source_split``.

Usage:
    ./.venv-uv/bin/python -m tools.stage6_build_train_manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv'
VAL_CSV = ROOT / 'data/corpus_v4/R3v3_license_verified_val.csv'
LEDGER_DIR = ROOT / 'data/corpus_v3/images'
OUT_CSV = ROOT / 'data/stage6/stage6_r3v3_train_view.csv'
OUT_JSON = ROOT / 'data/stage6/stage6_r3v3_train_view.json'

EXPECTED_TRAIN_SHA = '9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08'
EXPECTED_VAL_SHA = 'f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686'
EXPECTED_TRAIN_ROWS = 147_277
EXPECTED_VAL_ROWS = 22_047

OUT_COLUMNS = [
    'sample_id', 'path', 'split', 'status',
    'physical_source_id', 'source_split',
    'relative_path', 'raw_sha256', 'pixel_sha256',
    'original_dataset_id', 'group_key',
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_ledger_paths(ledger_dir: Path) -> Dict[str, str]:
    parts: List[pd.DataFrame] = []
    for path in sorted(ledger_dir.glob('*.csv')):
        frame = pd.read_csv(path, usecols=['image_id', 'raw_path'], dtype=str,
                            low_memory=False)
        parts.append(frame)
    if not parts:
        raise SystemExit(f'no ledger CSV under {ledger_dir}')
    ledger = pd.concat(parts, ignore_index=True).drop_duplicates('image_id')
    return dict(zip(ledger['image_id'], ledger['raw_path']))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, default=TRAIN_CSV)
    parser.add_argument('--val-csv', type=Path, default=VAL_CSV)
    parser.add_argument('--ledger-dir', type=Path, default=LEDGER_DIR)
    parser.add_argument('--out-csv', type=Path, default=OUT_CSV)
    parser.add_argument('--out-json', type=Path, default=OUT_JSON)
    args = parser.parse_args()

    train_sha = sha256_file(args.train_csv)
    val_sha = sha256_file(args.val_csv)
    if train_sha != EXPECTED_TRAIN_SHA:
        raise SystemExit(f'train CSV SHA mismatch: {train_sha} != {EXPECTED_TRAIN_SHA}')
    if val_sha != EXPECTED_VAL_SHA:
        raise SystemExit(f'val CSV SHA mismatch: {val_sha} != {EXPECTED_VAL_SHA}')

    train = pd.read_csv(args.train_csv, dtype=str, keep_default_na=False)
    val = pd.read_csv(args.val_csv, dtype=str, keep_default_na=False)
    if len(train) != EXPECTED_TRAIN_ROWS:
        raise SystemExit(f'train rows {len(train)} != {EXPECTED_TRAIN_ROWS}')
    if len(val) != EXPECTED_VAL_ROWS:
        raise SystemExit(f'val rows {len(val)} != {EXPECTED_VAL_ROWS}')

    # -- every train row must be declared train-eligible and holdout-clean
    ineligible = train[train['stage5e_train_eligible'].str.lower() != 'true']
    if len(ineligible):
        raise SystemExit(f'{len(ineligible)} train rows are not stage5e_train_eligible')
    holdup = train[train['holdout_status'] != 'OK']
    if len(holdup):
        raise SystemExit(f'{len(holdup)} train rows are not holdout OK')

    # -- split disjointness must be exact: no train id may appear in val
    train_ids = set(train['image_id'])
    val_ids = set(val['image_id'])
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(f'{len(overlap)} sample ids occur in both splits')
    if len(train_ids) != len(train):
        raise SystemExit('duplicate image_id within the train CSV')

    ledger = load_ledger_paths(args.ledger_dir)
    missing_ledger = train['image_id'][~train['image_id'].isin(ledger)]
    if len(missing_ledger):
        raise SystemExit(f'{len(missing_ledger)} train ids have no ledger raw_path')

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, str]] = []
    missing_files: List[str] = []
    suffix_mismatch: List[str] = []
    split_counter: Counter = Counter()
    for row in train.itertuples(index=False):
        raw_path = ledger[row.image_id]
        rel = row.relative_path
        if not raw_path.replace('\\', '/').endswith(rel):
            suffix_mismatch.append(row.image_id)
        if not Path(raw_path).is_file():
            missing_files.append(raw_path)
        split_counter[row.split] += 1
        out_rows.append({
            'sample_id': row.image_id,
            'path': raw_path,
            'split': 'train',
            'status': 'include',
            'physical_source_id': row.physical_source_id,
            'source_split': row.split,
            'relative_path': rel,
            'raw_sha256': row.raw_sha256,
            'pixel_sha256': row.pixel_sha256,
            'original_dataset_id': row.original_dataset_id,
            'group_key': row.group_key,
        })

    if missing_files:
        preview = '\n'.join(missing_files[:10])
        raise SystemExit(f'{len(missing_files)} train images are missing on disk:\n{preview}')
    if suffix_mismatch:
        raise SystemExit(f'{len(suffix_mismatch)} ledger paths do not end with relative_path')

    with args.out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    by_source = Counter(row['physical_source_id'] for row in out_rows)
    payload = {
        'description': 'Stage 6 derived training view of the frozen R3v3 train split',
        'derived_from': {
            'train_csv': str(args.train_csv.relative_to(ROOT)),
            'train_csv_sha256': train_sha,
            'val_csv': str(args.val_csv.relative_to(ROOT)),
            'val_csv_sha256': val_sha,
            'ledger_dir': str(args.ledger_dir.relative_to(ROOT)),
            'ledger_files': sorted(p.name for p in args.ledger_dir.glob('*.csv')),
            'path_source': 'Stage 5C ledger raw_path, joined on image_id',
            'row_order': 'identical to the R3v3 train CSV (no reordering)',
        },
        'membership': {
            'rows': len(out_rows),
            'unique_sample_ids': len(train_ids),
            'train_val_overlap': 0,
            'all_files_exist': True,
            'source_split_counts': dict(split_counter),
            'split_column': 'all rows labelled split=train; original value kept in source_split',
        },
        'by_source': dict(sorted(by_source.items())),
        'out_csv': str(args.out_csv.relative_to(ROOT)),
        'out_csv_sha256': sha256_file(args.out_csv),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                             encoding='utf-8')
    print(f'wrote {args.out_csv} ({len(out_rows)} rows)')
    print(f'  out sha256 : {payload["out_csv_sha256"]}')
    print(f'  by source  : {dict(sorted(by_source.items()))}')
    print(f'  source splits: {dict(split_counter)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
