#!/usr/bin/env python
"""Stage 5E.1 - evidence reconciliation tool.

Rebuilds every Stage 5E artifact that depended on a stale claim, from the real
files on this machine:

* ``reports/stage5e1_pool_upstream_licence_audit.json`` - per-upstream licence
  evidence grades for ``agrifield40k_pool``.
* ``data/corpus_v4/holdout_crossmatch.csv`` - regenerated so each of the 14
  pHash candidates carries the verdict from
  ``reports/stage5e_hash_collision_review.json``.
* ``reports/stage5e1_evidence_reconciliation.md`` / ``.json`` / gates.

It never edits the ledger, never rewrites R3v3, and never trains.

Usage
-----
    .venv-uv/bin/python tools/stage5e1_reconcile.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.registry import stage5e_licence as lic
from src.registry.stage5e1_licence_evidence import (EVIDENCE_ARCHIVED,
                                                    EVIDENCE_BACKED_GRADES,
                                                    EVIDENCE_CARRIED,
                                                    EVIDENCE_UNRESOLVED,
                                                    EVIDENCE_URL_ONLY,
                                                    build_pool_audit, grade_source)

V4 = ROOT / 'data/corpus_v4'
REPORTS = ROOT / 'reports'


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    import subprocess
    proc = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip()


def load_ledger() -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in sorted(glob.glob(str(ROOT / 'data/corpus_v3/images/*.csv')))]
    return pd.concat(frames, ignore_index=True)


def pool_upstream_counts(ledger: pd.DataFrame, accepted_ids: set) -> Dict[str, Dict[str, int]]:
    train = pd.read_csv(V4 / 'R3v3_license_verified_train.csv', usecols=['image_id'], dtype=str)
    train_ids = set(train['image_id'])
    out: Dict[str, Dict[str, int]] = {}
    pool = ledger[ledger['image_id'].isin(accepted_ids) &
                  (ledger['physical_source_id'] == 'agrifield40k_pool')]
    for upstream, group in pool.groupby('upstream_dataset'):
        out[str(upstream)] = {'accepted': int(len(group)),
                              'train': int(group['image_id'].isin(train_ids).sum())}
    return out


def source_counts(ledger: pd.DataFrame, accepted_ids: set, train_ids: set) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    accepted = ledger[ledger['image_id'].isin(accepted_ids)]
    for source, group in accepted.groupby('physical_source_id'):
        out[str(source)] = {'accepted': int(len(group)),
                            'train': int(group['image_id'].isin(train_ids).sum())}
    return out


def rebuild_crossmatch(review: Dict[str, object]) -> Dict[str, object]:
    """Re-emit holdout_crossmatch.csv with the review verdict per candidate."""
    path = V4 / 'holdout_crossmatch.csv'
    rows = list(csv.DictReader(open(path)))
    verdicts = {(str(r['dataset_id']), str(r['holdout_id']), str(r['corpus_id'])): r
                for r in review['rows']}
    matched = 0
    unmatched: List[str] = []
    pending = 0
    for row in rows:
        if row.get('conclusion') != 'CANDIDATE_REQUIRES_REVIEW':
            continue
        key = (row['dataset_id'], row['holdout_id'], row['corpus_id'])
        verdict = verdicts.get(key)
        if verdict is None:
            unmatched.append('|'.join(key))
            continue
        matched += 1
        row['embedding_confirmed'] = 'RUN' if verdict.get('holdout_decoded') and \
            verdict.get('corpus_decoded') else 'NOT_RUN'
        row['conclusion'] = 'VERDICT_' + str(verdict['verdict'])
        row['verdict'] = str(verdict['verdict'])
        row['same_pixel_sha256'] = str(bool(verdict.get('same_pixel_sha256'))).lower()
        row['mean_absolute_difference'] = f"{float(verdict['mean_absolute_difference']):.6f}"
        row['pearson_correlation'] = f"{float(verdict['pearson_correlation']):.6f}"
        row['review_basis'] = ('embedding/pixel comparison in '
                               'reports/stage5e_hash_collision_review.json')
    for row in rows:
        if row.get('conclusion') == 'CANDIDATE_REQUIRES_REVIEW':
            pending += 1
    if unmatched:
        raise SystemExit('crossmatch/review mismatch: ' + '; '.join(unmatched))
    fieldnames = list(rows[0].keys())
    for extra in ('verdict', 'same_pixel_sha256', 'mean_absolute_difference',
                  'pearson_correlation', 'review_basis'):
        if extra not in fieldnames:
            fieldnames.append(extra)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {'candidates_matched': matched, 'near_candidates_still_pending_review': pending,
            'path': str(path), 'sha256': sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    ledger = load_ledger()
    train_ids = set(pd.read_csv(V4 / 'R3v3_license_verified_train.csv',
                                usecols=['image_id'], dtype=str)['image_id'])
    val_ids = set(pd.read_csv(V4 / 'R3v3_license_verified_val.csv',
                              usecols=['image_id'], dtype=str)['image_id'])

    # ---- 1. licence evidence grades -------------------------------------- #
    accepted_ids = train_ids | val_ids
    src_counts = source_counts(ledger, accepted_ids, train_ids)
    source_rows: List[Dict[str, object]] = []
    for source_id in sorted(src_counts):
        row = grade_source(source_id, src_counts[source_id]['accepted'])
        row['accepted_images'] = src_counts[source_id]['accepted']
        row['train_images'] = src_counts[source_id]['train']
        source_rows.append(row)

    # ---- 2. pool upstream audit ------------------------------------------ #
    pool_audit = build_pool_audit(pool_upstream_counts(ledger, accepted_ids))

    # ---- 3. crossmatch reconciliation ------------------------------------ #
    review = json.loads((REPORTS / 'stage5e_hash_collision_review.json').read_text())
    crossmatch_info = rebuild_crossmatch(review)

    pool_row = next(r for r in source_rows if r['source_id'] == 'agrifield40k_pool')
    summary = pool_audit['summary']
    pool_row['grade'] = (EVIDENCE_CARRIED if summary['upstreams_unresolved'] == 0
                         else EVIDENCE_UNRESOLVED)
    pool_row['affected_accepted_images'] = summary['accepted_images_unresolved']
    pool_row['affected_train_images'] = summary['train_images_unresolved']
    pool_row['notes'] = ('Pool is split per upstream. %d upstreams unresolved with %d accepted '
                         '(%d train) images.' % (summary['upstreams_unresolved'],
                                                 summary['accepted_images_unresolved'],
                                                 summary['train_images_unresolved']))
    if 'accepted_images' in pool_row:
        pool_row['affected_accepted_images'] = pool_row['accepted_images']
        pool_row['affected_train_images'] = pool_row['train_images']

    graded = [r for r in source_rows if r['accepted_images'] > 0]
    unresolved_sources = [r for r in graded if r['grade'] in (EVIDENCE_UNRESOLVED, EVIDENCE_URL_ONLY)]

    licence_summary = {
        'grades': {r['source_id']: r['grade'] for r in source_rows},
        'evidence_backed_sources': sorted(r['source_id'] for r in graded
                                          if r['grade'] in EVIDENCE_BACKED_GRADES),
        'newly_archived_sources': sorted(r['source_id'] for r in source_rows
                                         if r['grade'] == EVIDENCE_ARCHIVED),
        'carried_forward_sources': sorted(r['source_id'] for r in graded
                                          if r['grade'] == EVIDENCE_CARRIED),
        'registry_url_only_sources': sorted(r['source_id'] for r in graded
                                            if r['grade'] == EVIDENCE_URL_ONLY),
        'unresolved_sources': sorted(r['source_id'] for r in graded
                                     if r['grade'] == EVIDENCE_UNRESOLVED),
        'gate_semantics': ('A source may be admitted to training only when its research-use '
                           'status is VERIFIED or RESTRICTED. The evidence grade describes how '
                           'strong the *documentary basis* is and never upgrades a status on '
                           'its own. Any accepted source without evidence-backed grade keeps '
                           'the Stage 6 status at CONDITIONAL.'),
    }
    (REPORTS / 'stage5e1_licence_evidence.json').write_text(
        json.dumps({'summary': licence_summary, 'sources': source_rows,
                    'pool_upstream_audit': pool_audit}, indent=2, ensure_ascii=False),
        encoding='utf-8')

    # Per-upstream CSV so the audit can be diffed row by row.
    upstream_rows = pool_audit['upstreams']
    csv_path = REPORTS / 'stage5e1_pool_upstream_licence_audit.csv'
    fields = ['upstream_dataset', 'publisher', 'year', 'version', 'pool_declared_license',
              'upstream_asset_license', 'upstream_asset_license_grade', 'accepted_images',
              'train_images', 'identity_basis', 'evidence_file', 'evidence_sha256', 'notes']
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(upstream_rows)
    (REPORTS / 'stage5e1_pool_upstream_licence_audit.json').write_text(
        json.dumps(pool_audit, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- 4. counts ------------------------------------------------------- #
    index = pd.read_csv(V4 / 'holdout_file_index.csv', dtype=str)
    per_dataset = {}
    for dataset_id, group in index.groupby('dataset_id'):
        per_dataset[str(dataset_id)] = {
            'files': int(len(group)),
            'decoded': int((group['decode_ok'].astype(str).str.lower() == 'true').sum()),
            'with_raw_hash': int((group['raw_sha256'].fillna('').astype(str).str.strip() != '').sum()),
            'with_pixel_hash': int((group['pixel_sha256'].fillna('').astype(str).str.strip() != '').sum()),
            'with_phash': int((group['phash'].fillna('').astype(str).str.strip() != '').sum()),
        }

    evidence = json.loads((REPORTS / 'stage5e_holdout_evidence.json').read_text())
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())

    counts = {
        'ledger_rows': int(len(ledger)),
        'accepted': len(train_ids) + len(val_ids),
        'train': len(train_ids),
        'val': len(val_ids),
        'pretraining_train_count': len(train_ids),
        'pretraining_validation_count': len(val_ids),
        'agri147k_target': 147285,
        'pretraining_train_minus_target': len(train_ids) - 147285,
        'accepted_minus_target': (len(train_ids) + len(val_ids)) - 147285,
    }

    print(json.dumps({'per_dataset': per_dataset, 'crossmatch': crossmatch_info,
                      'licence': licence_summary, 'counts': counts,
                      'pool_summary': summary}, indent=2, ensure_ascii=False))

    # Persist the raw facts the gates/report tools read.
    (REPORTS / 'stage5e1_facts.json').write_text(json.dumps({
        'per_dataset': per_dataset,
        'crossmatch': crossmatch_info,
        'licence': licence_summary,
        'counts': counts,
        'pool_summary': summary,
        'sources': source_rows,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
