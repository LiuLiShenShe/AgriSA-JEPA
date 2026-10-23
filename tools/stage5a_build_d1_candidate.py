#!/usr/bin/env python
"""Build the D1 candidate corpus manifest.

Inputs (never modified)
----------------------
``data/frozen/agrifield40k-v1_train.csv`` and ``..._val.csv``

New sources must be ingested first with ``stage5a_ingest_source.py``.

Outputs
-------
``data/frozen/agri-corpus-d1-candidate_train.csv``
``data/frozen/agri-corpus-d1-candidate_val.csv``
``data/frozen/agri-corpus-d1-candidate_manifest.csv``
``data/frozen/agri-corpus-d1-candidate_manifest.json``

Status is always ``CANDIDATE``.  Nothing here is approved training data; the
manifest records that 14,743 pending-audit images remain outside the corpus
and that perceptual-review candidates are unresolved.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry.freeze import (filter_candidate, load_ingested, load_v1_train,
                                 write_corpus)

ROOT = Path(__file__).resolve().parent.parent
AGRI_ROOT = Path('/data/datasets/AgriField-40K/agrifield40k')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1-train', default=str(ROOT / 'data/frozen/agrifield40k-v1_train.csv'))
    parser.add_argument('--v1-val', default=str(ROOT / 'data/frozen/agrifield40k-v1_val.csv'))
    parser.add_argument('--agrifield-root', default=str(AGRI_ROOT))
    parser.add_argument('--ingested', nargs='*', default=[],
                        help='optional source_id=path.csv pairs, e.g. plantvillage=/path/manifest.csv')
    parser.add_argument('--out-prefix', default=str(ROOT / 'data/frozen/agri-corpus-d1-candidate'))
    args = parser.parse_args()

    train_rows = load_v1_train(Path(args.v1_train), Path(args.agrifield_root))
    val_rows = load_v1_train(Path(args.v1_val), Path(args.agrifield_root))
    for row in val_rows:
        row.split_role = 'pretrain_val'

    added: Counter = Counter()
    for item in args.ingested:
        source_id, _, path = item.partition('=')
        if not source_id or not path:
            raise SystemExit(f'--ingested expects source_id=path.csv, got {item!r}')
        new_rows = filter_candidate(load_ingested(Path(path), source_id))
        for row in new_rows:
            row.split_role = 'pretrain_train'
        train_rows.extend(new_rows)
        added[source_id] = len(new_rows)

    train_kept = filter_candidate(train_rows)
    val_kept = filter_candidate(val_rows)

    prefix = Path(args.out_prefix)
    digest_train = write_corpus(prefix.parent / (prefix.name + '_train.csv'),
                                train_kept)
    digest_val = write_corpus(prefix.parent / (prefix.name + '_val.csv'), val_kept)

    manifest = {
        'data_version': 'agri-corpus-d1-candidate',
        'status': 'CANDIDATE',
        'approved': False,
        'base_version': 'agrifield40k-v1 (PROVISIONAL)',
        'new_sources': dict(added),
        'new_sources_images': sum(added.values()),
        'v1_train_images': len(train_kept),
        'v1_val_images': len(val_kept),
        'total_images': len(train_kept) + len(val_kept),
        'sha256_canonical_csv_train': digest_train,
        'sha256_canonical_csv_val': digest_val,
        'unresolved': {
            'agrifield40k_pending_audit_images': 14743,
            'pending_manual_duplicate_candidates': 784,
            'plantvillage_perceptual_candidates': 12436,
        },
        'notes': ('Every new row carries source, licence, attribution, original '
                  'path and SHA-256.  Approval requires a human decision, which '
                  'has not been made yet.'),
    }
    out = prefix.parent / (prefix.name + '_manifest.csv')
    with open(out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['key', 'value'])
        for key, value in manifest.items():
            writer.writerow([key, json.dumps(value, ensure_ascii=False)])
    (prefix.parent / (prefix.name + '_manifest.json')).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in (
        'status', 'v1_train_images', 'v1_val_images', 'new_sources_images',
        'total_images')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
