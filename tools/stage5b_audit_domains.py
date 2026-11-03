#!/usr/bin/env python
"""Stage 5B Gate 4: source- and image-level acquisition-domain audit.

Stage 5A recorded 63 registry rows covering 59 distinct datasets but never
recorded *where the camera was*.  This tool applies the per-source ledger in
``src/registry/source_domains.py`` to every registry row, then runs an
objective image-level triage over the sources actually on disk.

Image triage is explicitly *not* a domain verdict: it produces a suggested
label, a quality verdict and a review queue.  Model-assisted inspection is
reserved for the boundary cases and must be marked as such in
``field_image_review.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.registry.domain_policy import admission, summarise_labels
from src.registry.field_filter import (QualityGate, quality_gate, quality_metrics,
                                       suggest_domain)
from src.registry.ingest import IMAGE_SUFFIXES
from src.registry.source_domains import (SAME_PHYSICAL_DATASET, SOURCE_DOMAINS,
                                         label_for)

SOURCES_CSV = ROOT / 'data/registry/agricultural_sources.csv'
HOLDOUT_CSV = ROOT / 'data/registry/downstream_holdout_sources.csv'
OUT_REVIEW = ROOT / 'data/registry/field_source_review.csv'
OUT_IMAGE_REVIEW = ROOT / 'data/registry/field_image_review.csv'
OUT_JSON = ROOT / 'reports/stage5b_domain_audit.json'

#: How many images per source are triaged for the queue (stratified by file size).
TRIAGE_PER_SOURCE = int(__import__('os').environ.get('STAGE5B_TRIAGE_PER_SOURCE', '60'))


def load_csv(path: Path):
    if not path.is_file():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def build_source_review(registry_rows, holdout_ids):
    rows = []
    for row in registry_rows:
        sid = row['source_id']
        entry = label_for(sid)
        duplicate_of = SAME_PHYSICAL_DATASET.get(sid, '')
        rows.append({
            'source_id': sid,
            'official_name': row.get('official_name', ''),
            'catalog_groups': row.get('catalog_groups', ''),
            'registry_state': row.get('state', ''),
            'images_reported': row.get('images_reported', ''),
            'license': row.get('license', ''),
            'acquisition_domain': entry['acquisition_domain'],
            'domain_evidence_type': entry['evidence_type'],
            'domain_confidence': entry['confidence'],
            'platform': entry['platform'],
            'domain_review_status': 'reviewed' if entry['evidence_type'] in (
                'official_documentation',) else 'unreviewed',
            'domain_review_reason': entry['reason'],
            'same_physical_dataset_as': duplicate_of,
            'counted_as_independent': 'no' if duplicate_of else 'yes',
            'downstream_holdout': 'yes' if sid in holdout_ids else 'no',
            'mixed_domain': 'yes' if 'mixed' in entry['reason'].lower() or
                            entry['evidence_type'] == 'none' else 'no',
        })
    return rows


def stratified_sample(paths, n, seed=0):
    """Take n paths spread over file size, not just the first n alphabetically."""
    if len(paths) <= n:
        return paths
    ordered = sorted(paths, key=lambda p: (p.stat().st_size, str(p)))
    rng = np.random.default_rng(seed)
    picks = set()
    step = len(ordered) / n
    for i in range(n):
        picks.add(int(i * step))
    return [ordered[i] for i in sorted(picks)]


def triage_source(source_id, root: Path, gate: QualityGate, limit: int,
                  prefixes=()):
    records = []
    if not root.is_dir():
        return records
    paths = [p for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
             and (not prefixes or p.name.startswith(tuple(prefixes)))]
    for path in stratified_sample(paths, limit):
        try:
            with Image.open(path) as image:
                image.load()
                metrics = quality_metrics(image)
        except Exception as exc:  # noqa: BLE001 - triage must never abort
            records.append({
                'sample_id': f'{source_id}:{path.name}',
                'source_id': source_id,
                'rel_path': str(path.relative_to(root)),
                'width': '', 'height': '', 'mode': '', 'format': '',
                'mean_brightness': '', 'laplacian_variance': '',
                'border_fraction': '', 'border_luma': '', 'vegetation_fraction': '',
                'suggested_domain': 'UNKNOWN',
                'suggestion_reason': f'decode failure: {type(exc).__name__}',
                'quality_passed': 'false',
                'quality_reason': f'decode failure: {type(exc).__name__}',
                'domain_review_status': 'pending_human_review',
                'domain_evidence_type': 'none',
                'domain_confidence': 'unknown',
                'reviewer': '', 'reviewed_at': '',
                'domain_review_reason': 'undecodable image; needs manual handling',
            })
            continue
        suggestion, why = suggest_domain(metrics, path.name)
        passed, quality_reason = quality_gate(metrics, gate)
        records.append({
            'sample_id': f'{source_id}:{path.name}',
            'source_id': source_id,
            'rel_path': str(path.relative_to(root)),
            'width': metrics['width'], 'height': metrics['height'],
            'mode': metrics['mode'], 'format': image.format or '',
            'mean_brightness': metrics['mean_brightness'],
            'laplacian_variance': metrics['laplacian_variance'],
            'border_fraction': metrics['border_fraction'],
            'border_luma': metrics['border_luma'],
            'vegetation_fraction': metrics['vegetation_fraction'],
            'suggested_domain': suggestion,
            'suggestion_reason': why,
            'quality_passed': 'true' if passed else 'false',
            'quality_reason': quality_reason,
            'domain_review_status': 'pending_human_review',
            'domain_evidence_type': 'model_assisted_inspection',
            'domain_confidence': 'unknown',
            'reviewer': '', 'reviewed_at': '',
            'domain_review_reason': ('heuristic triage only; a human or documented '
                                     'evidence must confirm the final label'),
        })
    return records


#: Local roots for the sources actually downloaded (Stage 5A + 5B).
#: ``prefixes`` restricts a shared AgriField-40K directory to one source's rows.
LOCAL_ROOTS = {
    'lucas_vision': {'root': Path('/data/datasets/AgriField-40K/agrifield40k'),
                     'prefixes': ('lvb',)},
    'acre_crop_weed': {'root': Path('/data/datasets/AgriField-40K/agrifield40k'),
                       'prefixes': ('acw_',)},
    'radish_wheat': {'root': Path('/data/datasets/AgriField-40K/agrifield40k'),
                     'prefixes': ('rwd_',)},
    'sorghum_weed': {'root': Path('/data/datasets/AgriField-40K/agrifield40k'),
                     'prefixes': ('swd_',)},
    'vegann': {'root': Path('/data/datasets/AgriField-40K/agrifield40k'),
               'prefixes': ('vgn_',)},
    'inat_weeds': {'root': Path('/data/datasets/AgriField-40K/agrifield40k'),
                   'prefixes': ('inat_',)},
    'sesame_weed': {'root': Path('/data/datasets/Stage5B/sesame_weed/extracted/agri_data/data'),
                    'prefixes': ()},
    'applebbch81': {'root': Path('/data/datasets/Stage5B/applebbch81/extracted'),
                    'prefixes': ()},
    'weedcrop': {'root': Path('/data/datasets/Stage5B/weedcrop/extracted'),
                 'prefixes': ()},
    'plantvillage': {'root': Path('/data/datasets/PlantVillage/extracted/'
                                  'plantvillage_deeplearning_paper_dataset/color'),
                     'prefixes': ()},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--triage-per-source', type=int, default=TRIAGE_PER_SOURCE)
    parser.add_argument('--limit-sources', default='')
    args = parser.parse_args()

    registry_rows = load_csv(SOURCES_CSV)
    holdout_rows = load_csv(HOLDOUT_CSV)
    holdout_ids = {r['source_id'] for r in holdout_rows}

    source_review = build_source_review(registry_rows, holdout_ids)
    with open(OUT_REVIEW, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_review[0].keys()))
        writer.writeheader()
        writer.writerows(source_review)

    limit_ids = set(args.limit_sources.split(',')) if args.limit_sources else None
    image_rows = []
    for source_id, spec in LOCAL_ROOTS.items():
        if limit_ids and source_id not in limit_ids:
            continue
        image_rows.extend(triage_source(source_id, Path(spec['root']), QualityGate(),
                                        args.triage_per_source,
                                        tuple(spec.get('prefixes', ()))))
    if image_rows:
        with open(OUT_IMAGE_REVIEW, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(image_rows)

    domain_counts = Counter(r['acquisition_domain'] for r in source_review)
    suggestions = Counter(r['suggested_domain'] for r in image_rows)
    quality_pass = Counter(r['quality_passed'] for r in image_rows)
    independent = [r for r in source_review if r['counted_as_independent'] == 'yes']

    audit = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'registry_rows': len(registry_rows),
        'independent_datasets': len(independent),
        'same_physical_dataset_pairs': {
            'rows': sorted(SAME_PHYSICAL_DATASET.items()),
            'count': len(SAME_PHYSICAL_DATASET),
        },
        'source_domain_counts': dict(domain_counts),
        'source_domain_counts_independent_only': dict(Counter(
            r['acquisition_domain'] for r in independent)),
        'image_triage': {
            'sources_triaged': sorted({r['source_id'] for r in image_rows}),
            'images_triaged': len(image_rows),
            'suggested_domain_counts': dict(suggestions),
            'quality_gate': dict(quality_pass),
        },
        'admission_examples': {
            'field_insitu_documented': admission('FIELD_IN_SITU',
                                                 'official_documentation', 'high',
                                                 'reviewed'),
            'controlled_lab': admission('CONTROLLED_LAB', 'official_documentation',
                                        'high', 'reviewed'),
            'unknown_none': admission('UNKNOWN', 'none', 'unknown', 'unreviewed'),
            'model_only_at_freeze': admission('FIELD_IN_SITU',
                                              'model_assisted_inspection', 'medium',
                                              'model_assisted_pending_human',
                                              level='freeze'),
        },
        'note': ('Suggested domains come from image statistics plus filenames. They '
                 'order the review queue; they are not expert confirmations.'),
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'admission_examples'},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
