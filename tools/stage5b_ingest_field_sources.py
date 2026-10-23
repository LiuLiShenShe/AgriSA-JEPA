#!/usr/bin/env python
"""Stage 5B Gate 4: ingest the newly downloaded sources.

For every source this writes:

* ``data/registry/ingested/<source>_stage5b_manifest.csv`` -- per-image hashes,
  decode result, duplicate status and capture-group key;
* ``reports/stage5b_ingest_<source>.json`` -- counts and duplicate evidence.

Cross-source screening uses the frozen v1 manifests plus the full AgriField-40K
sample index, so an image that is already in the corpus (or in one of its
excluded rows) is reported rather than silently included.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.ingest import (IngestConfig, file_sha256, ingest,
                                 write_manifest_csv)

DATA_ROOT = Path('/data/datasets/Stage5B')

SOURCES = {
    'sesame_weed': {
        'source_name': 'Crop and Weed Detection Data with Bounding Boxes (Sesame)',
        'root': DATA_ROOT / 'sesame_weed/extracted/agri_data/data',
        'archive': DATA_ROOT / 'sesame_weed/sesame_weed.zip',
        'license': 'CC0-1.0',
        'attribution': 'Ravirajsinh Dabhi (Kaggle), CC0',
        'source_url': ('https://www.kaggle.com/datasets/ravirajsinh45/'
                       'crop-and-weed-detection-data-with-bounding-boxes'),
    },
    'applebbch81': {
        'source_name': 'AppleBBCH81 (orchard apple fruit, BBCH 81-85)',
        'root': DATA_ROOT / 'applebbch81/extracted',
        'archive': DATA_ROOT / 'applebbch81/applebbch81.zip',
        'license': 'CC-BY-4.0',
        'attribution': ('Kodors et al. (2024), Drones 8(12):734, '
                        'https://doi.org/10.3390/drones8120734'),
        'source_url': 'https://doi.org/10.3390/drones8120734',
    },
    'weedcrop': {
        'source_name': 'WeedCrop Image Dataset (Ronin-derived)',
        'root': DATA_ROOT / 'weedcrop/extracted',
        'archive': DATA_ROOT / 'weedcrop/weedcrop.zip',
        'license': 'CC0-1.0',
        'attribution': ('Derived from Sudars et al. (2020), Data in Brief 31:105833'),
        'source_url': ('https://www.kaggle.com/datasets/vinayakshanawad/'
                       'weedcrop-image-dataset'),
    },
}


def load_reference(include_plantvillage: bool = False):
    """Reference rows for cross-source screening.

    ``corpus_status`` records whether the reference row is *in* an audited
    corpus (``include``) or is an upstream row the project has deliberately
    not adopted (``pending_audit``).  The distinction matters: a byte match to
    an unadopted upstream row proves provenance, but it does not make the new
    image a duplicate of an image that is in the corpus.
    """
    rows = []
    for rel in ('data/all_samples.csv',):
        path = ROOT / rel
        if not path.is_file():
            continue
        with open(path, newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row.get('sha256'):
                    row['corpus_status'] = row.get('status', '') or 'unknown'
                    rows.append(row)
    if include_plantvillage:
        path = ROOT / 'data/registry/ingested/plantvillage_manifest.csv'
        if path.is_file():
            with open(path, newline='', encoding='utf-8') as handle:
                for row in csv.DictReader(handle):
                    if row.get('sha256'):
                        rows.append({'sample_id': row['sample_id'],
                                     'source': 'plantvillage',
                                     'corpus_status': 'candidate',
                                     'sha256': row['sha256'],
                                     'pixel_sha256': row['pixel_sha256'],
                                     'phash64': row['phash64'],
                                     'dhash64': row['dhash64']})
    return rows


def reclassify_upstream_matches(rows, reference) -> int:
    """Turn matches against unadopted upstream rows into provenance evidence.

    A match is only a *duplicate* when the reference row is already in an
    adopted corpus.  Matching an ``pending_audit`` row means the new source
    confirms that upstream row's mapping; the image is still new to the corpus.
    """
    status_by_id = {r.get('sample_id'): r.get('corpus_status', '')
                    for r in reference if r.get('sample_id')}
    changed = 0
    for row in rows:
        if row.get('duplicate_status') != 'cross_source_duplicate':
            continue
        ref_id = str(row.get('duplicate_of', '')).removeprefix('existing:')
        status = status_by_id.get(ref_id, '')
        if status and status != 'include':
            row['duplicate_status'] = 'upstream_pending_match'
            changed += 1
    return changed


def group_key_for(source_id: str, rel_path: str, default: str) -> str:
    """Capture-group key from the source's own naming scheme.

    A group is the unit that must never straddle train/val.  When a source
    publishes no session/plot metadata the conservative choice is one group per
    image -- that over-splits rather than leaks.
    """
    name = Path(rel_path).name
    if source_id == 'applebbch81':
        # DSC_<tree>_<session>_<view>.jpg -> one group per tree
        parts = name.split('_')
        if len(parts) >= 2 and parts[0].upper().startswith('DSC'):
            return f'tree_{parts[1]}'
        return default
    if source_id == 'weedcrop':
        # <original>_jpg.rf.<hash>.jpg -> the three augmentations share a group
        return name.split('_jpg')[0] or default
    if source_id == 'sesame_weed':
        # no session metadata is published; one image per group, documented
        return Path(name).stem
    return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='append', default=[],
                        help='source id (repeatable); default: all')
    parser.add_argument('--include-plantvillage-reference', action='store_true')
    args = parser.parse_args()

    ids = args.source or list(SOURCES)
    reference = load_reference(args.include_plantvillage_reference)
    print(f'reference rows for cross-source screening: {len(reference)}')

    written = []
    for source_id in ids:
        spec = SOURCES[source_id]
        root = Path(spec['root'])
        if not root.is_dir():
            print(f'SKIP {source_id}: root missing {root}')
            continue
        archive = Path(spec['archive'])
        config = IngestConfig(
            source_id=source_id,
            source_name=spec['source_name'],
            root=root,
            out_csv=ROOT / f'data/registry/ingested/{source_id}_stage5b_manifest.csv',
            license=spec['license'],
            attribution=spec['attribution'],
            source_url=spec['source_url'],
            source_archive_sha256=file_sha256(archive) if archive.is_file() else '',
        )
        result = ingest(config, reference_rows=reference)
        upstream_reclassified = reclassify_upstream_matches(result.rows, reference)
        for row in result.rows:
            row['group_key'] = group_key_for(source_id, str(row.get('rel_path', '')),
                                             str(row.get('group_key', '')))
        result.audit['unique_group_keys'] = len({r['group_key'] for r in result.rows})
        write_manifest_csv(Path(config.out_csv), result.rows)
        audit = dict(result.audit)
        audit['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
        audit['archive'] = str(archive)
        audit['archive_sha256'] = config.source_archive_sha256
        audit['upstream_pending_reclassified'] = upstream_reclassified
        audit['cross_source_duplicates_in_adopted_corpus'] = (
            audit['cross_source_exact_matches'] - upstream_reclassified)
        audit['images_duplicate_free'] = sum(
            1 for r in result.rows
            if not str(r.get('duplicate_status', '')).startswith('exact_')
            and r.get('duplicate_status') != 'cross_source_duplicate')
        audit_path = ROOT / f'reports/stage5b_ingest_{source_id}.json'
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n',
                              encoding='utf-8')
        written.append({
            'source_id': source_id,
            'files_found': audit['files_found'],
            'files_decoded': audit['files_decoded'],
            'decode_failures': audit['decode_failures'],
            'exact_duplicate_pairs': audit['exact_duplicate_pairs'],
            'perceptual_candidate_pairs': audit['perceptual_candidate_pairs'],
            'cross_source_exact_matches': audit['cross_source_exact_matches'],
            'cross_source_candidates': audit['cross_source_candidates'],
            'images_duplicate_free': audit['images_duplicate_free'],
            'unique_group_keys': audit['unique_group_keys'],
            'manifest': str(config.out_csv),
        })
        print(json.dumps(written[-1], indent=2, ensure_ascii=False))

    summary = {'sources': written,
               'reference_rows': len(reference),
               'generated_at_utc': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'reports/stage5b_ingest_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
