"""Build a versioned, hashed corpus manifest from audited sources.

The D1 candidate is *not* a replacement for ``agrifield40k-v1``.  It is a new
manifest that:

1. keeps only rows that passed the automatic gates (decode, exact-duplicate
   removal, no holdout source, licence recorded);
2. never overwrites a frozen v0/v1 file;
3. records perceptual-review candidates instead of deleting them;
4. is written with a SHA-256 over a canonical serialisation.

A manifest carrying ``status=candidate`` is deliberately distinct from
``freeze_status=frozen``: no human approval has happened.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

MANIFEST_FIELDS = [
    'sample_id', 'source_id', 'source', 'path', 'sha256', 'pixel_sha256',
    'width', 'height', 'group_key', 'license', 'attribution', 'split_role',
    'review_status', 'duplicate_status', 'origin_manifest',
]

#: Sources that must never enter a pretraining manifest.
HOLDOUT_SOURCE_IDS = {
    'phenobench', 'gwhd_2021', 'gwhd_instance_seg', 'plantdoc', 'minneapple',
    'roboflow_disease_detection', 'roboflow_crop_detection', 'plantseg',
    'multiclass_weeds', 'deepweeds', 'growliflower', 'cropandweed',
}


@dataclass
class CorpusRow:
    sample_id: str
    source_id: str
    source: str
    path: str
    sha256: str
    pixel_sha256: str
    width: int
    height: int
    group_key: str
    license: str
    attribution: str
    split_role: str
    review_status: str
    duplicate_status: str
    origin_manifest: str

    def to_dict(self) -> Dict[str, object]:
        return {f: getattr(self, f) for f in MANIFEST_FIELDS}


def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_v1_train(path: Path, agrifield_root: Path) -> List[CorpusRow]:
    rows: List[CorpusRow] = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row.get('status') != 'include':
                continue
            rows.append(CorpusRow(
                sample_id=row['sample_id'],
                source_id=row.get('source', '').strip(),
                source=row.get('source', ''),
                path=str(agrifield_root / row['path']),
                sha256=row.get('sha256', ''),
                pixel_sha256=row.get('pixel_sha256', ''),
                width=_int(row.get('width')),
                height=_int(row.get('height')),
                group_key=row.get('original_id') or row.get('sample_id'),
                license=row.get('license', ''),
                attribution=row.get('attribution', ''),
                split_role='pretrain_train',
                review_status='inherited_v1_confirmed',
                duplicate_status='v1_screened',
                origin_manifest=str(path),
            ))
    return rows


def load_ingested(path: Path, source_id: str) -> List[CorpusRow]:
    rows: List[CorpusRow] = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            rows.append(CorpusRow(
                sample_id=row['sample_id'],
                source_id=source_id,
                source=row.get('source_name', source_id),
                path=row.get('path', ''),
                sha256=row.get('sha256', ''),
                pixel_sha256=row.get('pixel_sha256', ''),
                width=_int(row.get('width')),
                height=_int(row.get('height')),
                group_key=row.get('group_key', ''),
                license=row.get('license', ''),
                attribution=row.get('attribution', ''),
                split_role='pretrain_train',
                review_status='pending_perceptual_review',
                duplicate_status=row.get('duplicate_status', ''),
                origin_manifest=str(path),
            ))
    return rows


def filter_candidate(rows: Iterable[CorpusRow],
                     holdout_sources: Optional[Set[str]] = None,
                     allow_sources: Optional[Set[str]] = None) -> List[CorpusRow]:
    """Apply the automatic gates; never silently include a holdout source."""
    holdout = HOLDOUT_SOURCE_IDS if holdout_sources is None else holdout_sources
    kept: List[CorpusRow] = []
    seen_sha: Set[str] = set()
    seen_pixel: Set[str] = set()
    for row in rows:
        if row.source_id in holdout:
            continue
        if allow_sources is not None and row.source_id not in allow_sources:
            continue
        if not row.sha256 or not row.pixel_sha256:
            continue
        if row.duplicate_status.startswith('exact_') or \
                row.duplicate_status == 'cross_source_duplicate':
            continue
        if row.sha256 in seen_sha or row.pixel_sha256 in seen_pixel:
            continue
        seen_sha.add(row.sha256)
        seen_pixel.add(row.pixel_sha256)
        kept.append(row)
    return kept


def canonical_csv(rows: Sequence[CorpusRow]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_dict())
    return buffer.getvalue()


def write_corpus(path: Path, rows: Sequence[CorpusRow],
                 extra: Optional[Dict[str, object]] = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_csv(rows)
    path.write_text(text, encoding='utf-8')
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    composition: Dict[str, Dict[str, int]] = {}
    for row in rows:
        entry = composition.setdefault(row.source_id, {'images': 0, 'bytes': 0})
        entry['images'] += 1
    manifest = {
        'status': 'CANDIDATE',
        'approved': False,
        'images': len(rows),
        'sources': len(composition),
        'composition': composition,
        'sha256_canonical_csv': digest,
        'rows_with_pending_perceptual_review': sum(
            1 for r in rows if r.review_status == 'pending_perceptual_review'),
        'holdout_sources_excluded': sorted(HOLDOUT_SOURCE_IDS),
    }
    if extra:
        manifest.update(extra)
    (path.with_suffix('.json') if path.suffix == '.csv'
     else Path(str(path) + '.json')).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return digest
