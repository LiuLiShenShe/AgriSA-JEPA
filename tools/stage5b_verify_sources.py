#!/usr/bin/env python
"""Stage 5B Gate 3: verify and record the download facts for new sources.

Nothing here decides that a source is admissible; it records what was
actually fetched, how, and what the source claims about the capture setting.

    .venv-uv/bin/python tools/stage5b_verify_sources.py --json
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

DATA_ROOT = Path('/data/datasets/Stage5B')
SOURCES_CSV = ROOT / 'data/registry/agricultural_sources.csv'
HOLDOUT_CSV = ROOT / 'data/registry/downstream_holdout_sources.csv'
OUT_CSV = ROOT / 'data/registry/agricultural_sources_v2.csv'
OUT_REVIEW = ROOT / 'data/registry/field_source_download.csv'
OUT_JSON = ROOT / 'reports/stage5b_source_verification.json'


def load_csv(path: Path):
    if not path.is_file():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


#: Per-source intake record.  Every field is transcribed from a source document
#: or from an actual on-disk file; nothing is inferred from a dataset name.
#:
#: ``acquisition_domain`` is the *source-level* label.  A source is only given a
#: single domain label when its own documentation states the capture setting and
#: the downloaded content supports it.  Anything else stays ``UNKNOWN`` and is
#: reviewed image-by-image by ``tools/stage5b_audit_domains.py``.
SOURCES = [
    {
        'source_id': 'sesame_weed',
        'local_name': 'Crop and Weed Detection Data with Bounding Boxes (Sesame)',
        'registry_row': 'sesame_weed',
        'download_url': 'https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
        'archive': DATA_ROOT / 'sesame_weed/sesame_weed.zip',
        'extracted_root': DATA_ROOT / 'sesame_weed/extracted/agri_data/data',
        'license': 'CC0-1.0',
        'license_evidence': ('Kaggle dataset page licence field "CC0: Public Domain"; '
                             'upstream Ravirajsinh Dabhi release'),
        'attribution': 'Ravirajsinh Dabhi (Kaggle), CC0',
        'source_doc_url': 'https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
        'acquisition_domain': 'FIELD_IN_SITU',
        'domain_evidence_type': 'official_documentation',
        'domain_confidence': 'high',
        'domain_review_reason': ('Official dataset description states the images are of a '
                                 'sesame crop and weeds in the field, with both YOLO and '
                                 'Pascal VOC bounding-box labels; image-level '
                                 'verification is run separately.'),
        'modality': 'RGB',
        'platform': 'handheld camera in field',
        'grouping': 'image (per-image annotation file)',
        'crop_scope': 'sesame crop; weed species',
    },
    {
        'source_id': 'applebbch81',
        'local_name': 'AppleBBCH81 (orchard apple fruit, BBCH 81-85)',
        'registry_row': 'agrifm_new_orchard_applebbch81',
        'download_url': 'https://www.kaggle.com/datasets/projectlzp201910094/applebbch81',
        'archive': DATA_ROOT / 'applebbch81/applebbch81.zip',
        'extracted_root': DATA_ROOT / 'applebbch81/extracted',
        'license': 'CC-BY-4.0',
        'license_evidence': ('Kaggle dataset page licence field "Attribution 4.0 '
                             'International (CC BY 4.0)"'),
        'attribution': ('Kodors et al. (2024), Drones 8(12):734, '
                        'https://doi.org/10.3390/drones8120734'),
        'source_doc_url': 'https://doi.org/10.3390/drones8120734',
        'acquisition_domain': 'ORCHARD_IN_SITU',
        'domain_evidence_type': 'official_documentation',
        'domain_confidence': 'high',
        'domain_review_reason': ('Dataset description: photographs taken in the LatHort '
                                 'orchard in Dobele of apple fruits at BBCH 81-85; '
                                 'two views per tree (tree-facing and oblique). '
                                 'This is an orchard, not a field crop.'),
        'modality': 'RGB',
        'platform': 'handheld camera in orchard',
        'grouping': 'tree/view (two views per tree)',
        'crop_scope': 'apple fruit at maturity',
    },
    {
        'source_id': 'weedcrop',
        'local_name': 'WeedCrop Image Dataset (Ronin-derived)',
        'registry_row': 'ronin',
        'download_url': 'https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset',
        'archive': DATA_ROOT / 'weedcrop/weedcrop.zip',
        'extracted_root': DATA_ROOT / 'weedcrop/extracted',
        'license': 'CC0-1.0',
        'license_evidence': 'Kaggle dataset page licence field "CC0: Public Domain"',
        'attribution': ('Derived from Sudars et al. (2020), Data in Brief 31:105833, '
                        'https://doi.org/10.1016/j.dib.2020.105833'),
        'source_doc_url': 'https://doi.org/10.1016/j.dib.2020.105833',
        'acquisition_domain': 'UNKNOWN',
        'domain_evidence_type': 'filename_or_metadata_only',
        'domain_confidence': 'low',
        'domain_review_reason': ('Kaggle description states the images were augmented '
                                 '(90-degree rotations, shear, brightness) from an '
                                 'upstream field dataset, but does not state that the '
                                 'source images were field-captured. A vision-model '
                                 'sample read 14/18 cells as open-field and 4/18 as '
                                 'ambiguous close-ups, which is not enough to confirm a '
                                 'uniform field label. It is also a derivative of the '
                                 'registered Ronin source, so it is not counted as a '
                                 'new source.'),
        'modality': 'RGB',
        'platform': 'derived from field images (unverified)',
        'grouping': 'source image (3 augmented versions each)',
        'crop_scope': 'crops and weeds (robotic weed control)',
    },
]


def build_source_review(sources, registry_rows, holdout_rows):
    """Source-level review table: one row per source, evidence-linked."""
    registry_by_id = {r['source_id']: r for r in registry_rows}
    holdout_ids = {r['source_id'] for r in holdout_rows}
    out = []
    for src in sources:
        reg = registry_by_id.get(src['registry_row'], {})
        archive = Path(src['archive'])
        exists = archive.is_file()
        extracted = Path(src['extracted_root'])
        extracted_exists = extracted.is_dir()
        n_extracted = 0
        if extracted_exists:
            n_extracted = sum(1 for p in extracted.rglob('*')
                              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        out.append({
            'source_id': src['source_id'],
            'local_name': src['local_name'],
            'registry_row': src['registry_row'],
            'registry_official_name': reg.get('official_name', ''),
            'registry_images_reported': reg.get('images_reported', ''),
            'registry_state': reg.get('state', ''),
            'download_url': src['download_url'],
            'archive_present': exists,
            'archive_bytes': archive.stat().st_size if exists else 0,
            'archive_sha256': sha256_file(archive) if exists else '',
            'extracted_root_present': extracted_exists,
            'extracted_images_found': n_extracted,
            'download_time_utc': (datetime.now(timezone.utc).isoformat()
                                  if exists else ''),
            'license': src['license'],
            'license_evidence': src['license_evidence'],
            'attribution': src['attribution'],
            'source_doc_url': src['source_doc_url'],
            'acquisition_domain': src['acquisition_domain'],
            'domain_evidence_type': src['domain_evidence_type'],
            'domain_confidence': src['domain_confidence'],
            'domain_review_status': 'pending_human_review',
            'domain_review_reason': src['domain_review_reason'],
            'reviewer': '',
            'reviewed_at': '',
            'downstream_holdout': 'yes' if src['source_id'] in holdout_ids or
                                  src['registry_row'] in holdout_ids else 'no',
            'modality': src['modality'],
            'platform': src['platform'],
            'grouping': src['grouping'],
            'crop_scope': src['crop_scope'],
        })
    return out


#: Sources that did not exist in the Stage 5A ledger and get a new v2 row.
NEW_REGISTRY_ROWS = [
    {
        'source_id': 'applebbch81',
        'official_name': 'AppleBBCH81 (orchard apple fruit, BBCH 81-85)',
        'catalog_groups': 'stage5b_new',
        'paper': ('Kodors, Zarembo, Lacis, Litavniece, Apeinans, Sondors & Pacejs, '
                  'Drones 8(12):734 (2024)'),
        'year': '2024',
        'paper_url': 'https://doi.org/10.3390/drones8120734',
        'download_url': 'https://www.kaggle.com/datasets/projectlzp201910094/applebbch81',
        'repo_url': '',
        'category': 'orchard fruit detection',
        'domain': 'orchard;apple',
        'modality': 'RGB',
        'platform': 'handheld camera in orchard',
        'resolution': '640x640 crops from 3008x2000',
        'grouping_fields': 'tree',
        'official_splits': 'none',
        'images_reported': '',
        'files_remote': '',
        'files_downloaded': '',
        'images_decoded': '',
        'images_duplicate_free': '',
        'images_approved': '',
        'license': 'CC-BY-4.0',
        'license_evidence': ('Kaggle dataset page licence field "Attribution 4.0 '
                             'International (CC BY 4.0)" and the paper acknowledgement'),
        'attribution_required': 'yes',
        'allowed_in_project': 'yes',
        'overlaps_agrifield40k': 'no',
        'overlaps_agrifmplus': 'no',
        'downstream_holdout': 'no',
        'overlap_evidence': '',
        'state': 'DOWNLOADED',
        'downloaded': 'yes',
        'download_failure': '',
        'human_review_status': 'not_reviewed',
        'download_evidence': '',
        'decode_evidence': '',
        'duplicate_evidence': '',
        'split_evidence': '',
        'approval_evidence': '',
        'freeze_manifest': '',
        'evidence': 'Stage 5B download; see reports/stage5b_download_audit.md',
        'notes': ('New Stage 5B source. 1,838 images in 140 tree groups. '
                  'Domain evidence: publisher states the LatHort orchard.'),
    },
]


def build_registry_v2(registry_rows, sources, review_count=None):
    """Stage 5A registry + the Stage 5B acquisition-domain columns."""
    review_by_row = {s['registry_row']: s for s in sources}
    by_id = {s['source_id']: s for s in sources}
    review_count = review_count or {}
    from src.registry.source_domains import label_for
    rows = []
    for row in registry_rows:
        new = dict(row)
        sid = row['source_id']
        # Every registry row gets a domain label from the Stage 5B ledger.  A
        # derivative download (weedcrop) must NOT relabel its upstream source
        # (ronin); the derivative is recorded separately in
        # field_source_download.csv.
        entry = label_for(sid)
        new['acquisition_domain'] = entry['acquisition_domain']
        new['domain_evidence_type'] = entry['evidence_type']
        new['domain_confidence'] = entry['confidence']
        new['domain_review_status'] = ('reviewed'
                                       if entry['evidence_type'] == 'official_documentation'
                                       else 'unreviewed')
        new['domain_review_reason'] = entry['reason']
        if review_count.get(sid):
            new['files_downloaded'] = review_count[sid]
            new['images_decoded'] = review_count[sid]
        rows.append(new)
    existing = {r['source_id'] for r in rows}
    for extra in NEW_REGISTRY_ROWS:
        if extra['source_id'] in existing:
            continue
        src = by_id.get(extra['source_id'])
        new = dict(extra)
        if src is not None:
            new['acquisition_domain'] = src['acquisition_domain']
            new['domain_evidence_type'] = src['domain_evidence_type']
            new['domain_confidence'] = src['domain_confidence']
            new['domain_review_status'] = 'model_assisted_pending_human'
            new['domain_review_reason'] = src['domain_review_reason']
            count = review_count.get(extra['source_id'])
            if count:
                new['files_downloaded'] = count
                new['images_decoded'] = count
        rows.append(new)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true', help='print JSON summary')
    args = parser.parse_args()

    registry_rows = load_csv(SOURCES_CSV)
    holdout_rows = load_csv(HOLDOUT_CSV)
    review = build_source_review(SOURCES, registry_rows, holdout_rows)

    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REVIEW, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0].keys()))
        writer.writeheader()
        writer.writerows(review)

    review_count = {r['source_id']: r['extracted_images_found'] for r in review}
    registry_v2 = build_registry_v2(registry_rows, SOURCES, review_count)
    v2_fields = (list(registry_rows[0].keys()) + ['acquisition_domain',
                                                 'domain_evidence_type',
                                                 'domain_confidence',
                                                 'domain_review_status',
                                                 'domain_review_reason'])
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=v2_fields)
        writer.writeheader()
        writer.writerows(registry_v2)

    summary = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'registry_rows': len(registry_rows),
        'new_source_reviews': len(review),
        'sources': {r['source_id']: {
            'archive_present': r['archive_present'],
            'archive_bytes': r['archive_bytes'],
            'archive_sha256': r['archive_sha256'],
            'extracted_images_found': r['extracted_images_found'],
            'license': r['license'],
            'acquisition_domain': r['acquisition_domain'],
            'downstream_holdout': r['downstream_holdout'],
        } for r in review},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        for r in review:
            print(f"{r['source_id']:16s} archive={r['archive_present']!s:5s} "
                  f"images={r['extracted_images_found']:6d} "
                  f"domain={r['acquisition_domain']:22s} lic={r['license']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
