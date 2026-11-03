"""Build the Field-D1 manifest from the audited v1 corpus plus new sources.

The output keeps every column the existing ``AgriFieldDataset`` loader needs
(``sample_id``, ``path``, ``split``, ``status``) and adds the Stage 5B domain
columns.  Frozen v0/v1 files are read-only inputs and are never rewritten.

Design rules encoded here:

* a holdout source is dropped even if it was explicitly whitelisted;
* an image whose domain/evidence/review combination fails the policy gate is
  dropped (and counted, never silently ignored);
* a capture group (``group_key``) is never split across train and val;
* the manifest SHA-256 is taken over a canonical CSV serialisation, so two
  builds from the same input bytes produce the same digest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.registry.domain_policy import admission

#: Superset of the v1 manifest columns plus the Stage 5B domain columns.
FIELD_FIELDS: List[str] = [
    'sample_id', 'path', 'source_prefix', 'source', 'original_id', 'split',
    'width', 'height', 'mode', 'format', 'bytes', 'sha256', 'pixel_sha256',
    'phash64', 'dhash64', 'decode_ok', 'license', 'attribution', 'source_url',
    'status', 'reason', 'component',
    'acquisition_domain', 'domain_evidence_type', 'domain_confidence',
    'domain_review_status', 'domain_review_reason', 'quality_status',
    'duplicate_status', 'origin_manifest',
]

#: Columns a manifest must have to be loadable by ``AgriFieldDataset``.
LOADER_REQUIRED = ('sample_id', 'path', 'split', 'status')


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical_csv(rows: Sequence[Mapping[str, object]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELD_FIELDS, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, '') for field in FIELD_FIELDS})
    return buffer.getvalue()


def load_v1_manifest(path: Path, split: str) -> List[Dict[str, object]]:
    """Read a frozen v1 split; only ``status == include`` rows are returned."""
    rows: List[Dict[str, object]] = []
    with open(path, newline='', encoding='utf-8') as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get('status', '')).strip() != 'include':
                continue
            rows.append({
                'sample_id': raw['sample_id'],
                'path': raw['path'],
                'source_prefix': raw.get('source_prefix', ''),
                'source': raw.get('source', ''),
                'original_id': raw.get('original_id', ''),
                'split': split,
                'width': raw.get('width', ''),
                'height': raw.get('height', ''),
                'mode': raw.get('mode', ''),
                'format': raw.get('format', ''),
                'bytes': raw.get('bytes', ''),
                'sha256': raw.get('sha256', ''),
                'pixel_sha256': raw.get('pixel_sha256', ''),
                'phash64': raw.get('phash64', ''),
                'dhash64': raw.get('dhash64', ''),
                'decode_ok': raw.get('decode_ok', ''),
                'license': raw.get('license', ''),
                'attribution': raw.get('attribution', ''),
                'source_url': raw.get('source_url', ''),
                'status': 'include',
                'reason': raw.get('reason', ''),
                'component': raw.get('component', raw.get('original_id', '')),
                'origin_manifest': str(path),
            })
    return rows


def load_ingested(path: Path, source_id: str, source_label: str,
                  domain: str, evidence_type: str, confidence: str,
                  review_status: str, review_reason: str,
                  license: str, attribution: str, source_url: str,
                  origin_manifest: str = '') -> List[Dict[str, object]]:
    """Map an ingest manifest row to the Field-D1 schema (no split yet)."""
    rows: List[Dict[str, object]] = []
    with open(path, newline='', encoding='utf-8') as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get('decode_ok', '')).lower() not in ('true', '1'):
                continue
            rows.append({
                'sample_id': raw['sample_id'],
                'path': raw.get('path', ''),
                'source_prefix': source_id,
                'source': source_label,
                'original_id': raw.get('group_key', ''),
                'split': '',
                'width': raw.get('width', ''),
                'height': raw.get('height', ''),
                'mode': raw.get('mode', ''),
                'format': raw.get('format', ''),
                'bytes': raw.get('bytes', ''),
                'sha256': raw.get('sha256', ''),
                'pixel_sha256': raw.get('pixel_sha256', ''),
                'phash64': raw.get('phash64', ''),
                'dhash64': raw.get('dhash64', ''),
                'decode_ok': 'true',
                'license': license,
                'attribution': attribution,
                'source_url': source_url,
                'status': 'include',
                'reason': '',
                'component': raw.get('group_key', raw.get('sample_id', '')),
                'acquisition_domain': domain,
                'domain_evidence_type': evidence_type,
                'domain_confidence': confidence,
                'domain_review_status': review_status,
                'domain_review_reason': review_reason,
                'quality_status': raw.get('duplicate_status', 'unique'),
                'duplicate_status': raw.get('duplicate_status', 'unique'),
                'origin_manifest': origin_manifest or str(path),
            })
    return rows


def assign_group_split(rows: Sequence[Mapping[str, object]], val_fraction: float = 0.2,
                       seed: int = 20261023) -> Dict[str, str]:
    """Deterministically assign whole capture groups to ``train``/``val``.

    Groups are ordered by a salted hash, so the split does not depend on
    directory order and is stable across machines.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError('val_fraction must be in [0, 1)')
    groups = sorted({str(r.get('component') or r.get('sample_id')) for r in rows})
    if not groups:
        return {}
    if val_fraction == 0.0:
        return {group: 'train' for group in groups}
    ranked = sorted(groups, key=lambda g: hashlib.sha256(
        f'{seed}|{g}'.encode('utf-8')).hexdigest())
    n_val = max(1, int(round(len(ranked) * val_fraction)))
    val_groups = set(ranked[:n_val])
    return {group: ('val' if group in val_groups else 'train') for group in groups}


def apply_gate(rows: Sequence[Mapping[str, object]], holdout_ids: Set[str],
               level: str = 'candidate') -> Tuple[List[Dict[str, object]],
                                                  Dict[str, int], List[Dict[str, str]]]:
    """Drop holdout, policy-failing and exact-duplicate rows; return rejections."""
    kept: List[Dict[str, object]] = []
    rejected: List[Dict[str, str]] = []
    reasons: Counter = Counter()
    seen_sha: Set[str] = set()
    seen_pixel: Set[str] = set()
    for row in rows:
        source_id = str(row.get('source_prefix') or row.get('source_id') or '')
        if source_id in holdout_ids:
            reasons['holdout_source'] += 1
            rejected.append({'sample_id': str(row.get('sample_id')),
                             'reason': 'holdout_source'})
            continue
        duplicate = str(row.get('duplicate_status', 'unique'))
        if duplicate.startswith('exact_') or duplicate == 'cross_source_duplicate':
            reasons['exact_duplicate'] += 1
            rejected.append({'sample_id': str(row.get('sample_id')),
                             'reason': duplicate})
            continue
        if row.get('acquisition_domain'):
            admitted, why = admission(
                str(row['acquisition_domain']),
                str(row.get('domain_evidence_type', 'none')),
                str(row.get('domain_confidence', 'unknown')),
                str(row.get('domain_review_status', 'unreviewed')),
                level=level)
            if not admitted:
                reasons['domain_policy'] += 1
                rejected.append({'sample_id': str(row.get('sample_id')),
                                 'reason': why})
                continue
        sha = str(row.get('sha256') or '')
        pixel = str(row.get('pixel_sha256') or '')
        if not sha or not pixel:
            reasons['missing_hash'] += 1
            rejected.append({'sample_id': str(row.get('sample_id')),
                             'reason': 'missing_hash'})
            continue
        if sha in seen_sha or pixel in seen_pixel:
            reasons['exact_duplicate'] += 1
            rejected.append({'sample_id': str(row.get('sample_id')),
                             'reason': 'duplicate_within_manifest'})
            continue
        seen_sha.add(sha)
        seen_pixel.add(pixel)
        kept.append(dict(row))
    return kept, dict(reasons), rejected


def write_field_corpus(out_prefix: Path,
                       rows: Sequence[Mapping[str, object]],
                       extra_manifest: Optional[Mapping[str, object]] = None
                       ) -> Dict[str, object]:
    """Write ``<prefix>_train.csv``, ``<prefix>_val.csv`` and ``<prefix>_manifest.json``."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train = [dict(r) for r in rows if str(r.get('split')) == 'train']
    val = [dict(r) for r in rows if str(r.get('split')) == 'val']
    digests: Dict[str, str] = {}
    for name, subset in (('_train.csv', train), ('_val.csv', val)):
        text = canonical_csv(subset)
        path = Path(str(out_prefix) + name)
        path.write_text(text, encoding='utf-8')
        digests[path.name] = _digest(text)

    def _composition(column: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = defaultdict(lambda: {'images': 0})
        for row in rows:
            key = str(row.get(column) or 'UNKNOWN')
            out[key]['images'] += 1
        return dict(sorted(out.items()))

    train_groups = {str(r.get('component')) for r in train}
    val_groups = {str(r.get('component')) for r in val}
    manifest: Dict[str, object] = {
        'data_version': 'agri-field-d1-candidate',
        'status': 'CANDIDATE',
        'approved': False,
        'images': {'train': len(train), 'val': len(val), 'total': len(rows)},
        'sources': len({str(r.get('source_prefix')) for r in rows}),
        'composition': {
            'train': _composition('source_prefix'),
        },
        'acquisition_domain': _composition('acquisition_domain'),
        'split_integrity': {
            'groups': len(train_groups | val_groups),
            'train_groups': len(train_groups),
            'val_groups': len(val_groups),
            'cross_split_groups': len(train_groups & val_groups),
            'train_val_disjoint': not (train_groups & val_groups),
        },
        'sha256': digests,
        'loader_contract': list(LOADER_REQUIRED),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = Path(str(out_prefix) + '_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
                             encoding='utf-8')
    return manifest


def read_field_corpus(path: Path) -> List[Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))
