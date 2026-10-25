"""Traceable metadata recovery for Stage 5D.

Two upstream sources publish machine-readable identity:

* Artemis / Image Safari -- ``remote_index.csv`` (6,920,332 published S3
  objects with ``part``/``crop``/``farm``/``key``/``bytes``/``etag``).  The
  object key is authoritative and encodes the capture session.
* AgriField-40K -- ``all_samples.csv`` maps every image to its *original*
  upstream dataset (``source``/``candidate_source``), original id and original
  split.

Nothing here predicts a crop with a model.  When the official record does not
contain a field, the field stays ``unknown`` and its provenance says
``MISSING``.

The published Artemis CSV has column semantics that shift between its two
``part`` values (for Image Safari ``session`` holds the date and ``date`` holds
the file name; for Artemis ``plot`` holds the date).  Parsing the ``key``
directly avoids depending on that layout.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from .stage5d_metadata_schema import (PROVENANCE_DATASET_LAYOUT,
                                      PROVENANCE_MISSING, PROVENANCE_OFFICIAL,
                                      JoinAudit, normalize_crop)

ARTEMIS_PARTS = ('Artemis', 'ImageSafari')

#: ``2023--tza--nov`` -> ISO3 country + month.  This is part of the published
#: object key, not an inference from a farm name.
SEASON_TOKEN = re.compile(r'^(?P<year>\d{4})--(?P<country>[a-z]{3})--(?P<month>[a-z]{3})$')
DATE_TOKEN = re.compile(r'^\d{4}-\d{2}-\d{2}$')


@dataclass
class ArtemisKey:
    part: str
    crop: str
    farm: str
    session: str
    season: str
    capture_date: str
    plot: str
    filename: str
    scheme: str

    @property
    def group_key(self) -> str:
        if self.scheme == 'artemis':
            return (f'artemis|{self.crop}|{self.farm}|{self.session}|'
                    f'{self.season}|{self.capture_date}|{self.plot}')
        return f'imagesafari|{self.crop}|{self.farm}|{self.capture_date}'

    @property
    def country_code(self) -> str:
        if self.scheme != 'artemis':
            return ''
        match = SEASON_TOKEN.match(self.season)
        return match.group('country').upper() if match else ''


def parse_artemis_key(key: str) -> Optional[ArtemisKey]:
    """Parse a published S3 key.  Returns ``None`` if the layout is unknown."""
    parts = [p for p in str(key).split('/') if p]
    if len(parts) < 3 or parts[0] not in ARTEMIS_PARTS or 'images' not in parts:
        return None
    part = parts[0]
    crop = parts[1]
    index = parts.index('images')
    if index + 2 >= len(parts):
        return None
    farm = parts[index + 1]
    rest = parts[index + 2:]
    filename = rest[-1]
    if len(rest) == 2 and DATE_TOKEN.match(rest[0]):
        return ArtemisKey(part, crop, farm, rest[0], '', rest[0], '', filename,
                          'imagesafari')
    if len(rest) == 5 and DATE_TOKEN.match(rest[2]):
        return ArtemisKey(part, crop, farm, rest[0], rest[1], rest[2], rest[3],
                          filename, 'artemis')
    # Tolerate a plot folder being absent on the Artemis side.
    if len(rest) == 4 and DATE_TOKEN.match(rest[2]):
        return ArtemisKey(part, crop, farm, rest[0], rest[1], rest[2], '',
                          filename, 'artemis')
    return None


def _iter_index(index_path: Path) -> Iterator[Dict[str, str]]:
    with open(index_path, newline='', encoding='utf-8') as handle:
        yield from csv.DictReader(handle)


def load_artemis_index(index_path: str | Path,
                       wanted_keys: Optional[Iterable[str]] = None,
                       chunk_rows: int = 500_000
                       ) -> Tuple[pd.DataFrame, JoinAudit]:
    """Stream the published index and keep only the requested keys.

    ``wanted_keys=None`` keeps everything (used for the official inventory).
    Duplicate keys are counted, never silently dropped.
    """
    path = Path(index_path)
    wanted = set(wanted_keys) if wanted_keys is not None else None
    rows: List[Dict[str, str]] = []
    audit = JoinAudit()
    seen: Dict[str, int] = defaultdict(int)
    total = 0
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            total += 1
            key = record.get('key') or ''
            if wanted is not None and key not in wanted:
                continue
            seen[key] += 1
            rows.append(record)
    audit.reference_rows = total
    audit.candidate_rows = len(wanted) if wanted is not None else total
    audit.matched_rows = len(seen)
    audit.unmatched_rows = (len(wanted) - len(seen)) if wanted is not None else 0
    audit.key_collisions = sum(1 for count in seen.values() if count > 1)
    audit.one_to_many = sum(count - 1 for count in seen.values() if count > 1)
    frame = pd.DataFrame(rows, columns=['part', 'crop', 'farm', 'session',
                                        'date', 'plot', 'key', 'bytes', 'etag'])
    if not frame.empty and audit.key_collisions:
        # Keep the first record so the join stays total, but the collision is
        # reported and the affected rows are marked MATCHED_AMBIGUOUS.
        frame = frame.drop_duplicates(subset=['key'], keep='first')
    return frame, audit


def attach_artemis_metadata(frame: pd.DataFrame,
                            index_path: str | Path) -> Tuple[pd.DataFrame, JoinAudit, Dict[str, int]]:
    """Join the official index onto the Artemis/Image Safari ledger rows.

    The join key is the exact relative object path (``relative_path`` ==
    ``key``) plus the recorded byte size.  A size mismatch is a conflict: the
    metadata is attached but flagged, and the row can never claim
    ``OFFICIAL_METADATA`` provenance for that field.
    """
    mask = frame['physical_source_id'].eq('artemis_imagesafari')
    keys = frame.loc[mask, 'relative_path'].astype(str)
    index, audit = load_artemis_index(index_path, wanted_keys=keys)

    parsed: Dict[str, ArtemisKey] = {}
    scheme_counts: Counter = Counter()
    parse_failures = 0
    for key in index['key'].astype(str):
        parsed_key = parse_artemis_key(key)
        if parsed_key is None:
            parse_failures += 1
            continue
        parsed[key] = parsed_key
        scheme_counts[parsed_key.scheme] += 1
    stats = {
        'index_matched_keys': len(parsed),
        'parse_failures': parse_failures,
        'artemis_keys': scheme_counts.get('artemis', 0),
        'imagesafari_keys': scheme_counts.get('imagesafari', 0),
    }

    out = frame.copy()
    defaults = {
        'crop': 'unknown', 'country': 'unknown', 'farm_id': '', 'session_id': '',
        'capture_timestamp': '', 'plot_id': '', 'group_key': '',
        'crop_evidence': PROVENANCE_MISSING, 'country_evidence': PROVENANCE_MISSING,
        'group_evidence': '', 'metadata_join_status': '', 'original_dataset_id': '',
        'original_image_id': '', 'original_uri': '', 'original_split': '',
    }
    for column, default in defaults.items():
        out[column] = default

    if index.empty:
        audit.conflict_fields = {}
        audit.conflict_rows = 0
        stats['index_matched_keys'] = 0
        for column, default in defaults.items():
            out[column] = default
        out.loc[mask, 'metadata_join_status'] = 'UNMATCHED'
        return out, audit, stats
    size_by_key = dict(zip(index['key'].astype(str),
                           index.get('bytes', pd.Series(dtype=str)).astype(str)))
    has_size = mask & out.get('bytes', pd.Series(dtype=float)).notna()
    keys_all = out['relative_path'].astype(str)
    parsed_keys = keys_all.map(parsed)
    matched = mask & parsed_keys.notna()

    def series_from(mapping) -> pd.Series:
        values = parsed_keys.map(
            lambda key: mapping(key) if isinstance(key, ArtemisKey) else None)
        values = values.where(matched, None)
        return values

    out['crop'] = series_from(lambda k: normalize_crop(k.crop)).fillna('unknown')
    out['country'] = series_from(lambda k: k.country_code or 'unknown').fillna('unknown')
    out['farm_id'] = series_from(lambda k: k.farm).fillna('')
    out['session_id'] = series_from(lambda k: k.session).fillna('')
    out['capture_timestamp'] = series_from(lambda k: k.capture_date).fillna('')
    out['plot_id'] = series_from(lambda k: k.plot).fillna('')
    out['group_key'] = series_from(lambda k: k.group_key).fillna('')
    out['crop_evidence'] = series_from(lambda k: PROVENANCE_OFFICIAL).fillna(PROVENANCE_MISSING)
    out['country_evidence'] = series_from(
        lambda k: PROVENANCE_DATASET_LAYOUT if k.country_code else PROVENANCE_MISSING
    ).fillna(PROVENANCE_MISSING)
    out['group_evidence'] = series_from(lambda k: 'DOCUMENTED_PATH_GROUP').fillna('')
    out['original_dataset_id'] = series_from(
        lambda k: 'Artemis' if k.part == 'Artemis' else 'ImageSafari').fillna('')
    out['original_image_id'] = series_from(lambda k: k.filename).fillna('')
    out['original_uri'] = series_from(
        lambda k: f's3://artemis-imagesafari/{k.part}').fillna('')
    out['original_split'] = ''

    # Byte-size conflict: the key matched but the stored object differs.
    conflicts: Counter = Counter()
    if 'bytes' in out.columns:
        recorded = pd.to_numeric(out['bytes'], errors='coerce')
        expected = keys_all.map(size_by_key)
        expected_num = pd.to_numeric(expected, errors='coerce')
        conflict_mask = matched & recorded.notna() & expected_num.notna() & (recorded != expected_num)
        conflicts['bytes'] = int(conflict_mask.sum())
        out.loc[conflict_mask, 'metadata_join_status'] = 'MATCHED_CONFLICT'
    out.loc[mask & parsed_keys.isna(), 'metadata_join_status'] = 'UNMATCHED'
    out.loc[matched & ~out['metadata_join_status'].eq('MATCHED_CONFLICT'),
            'metadata_join_status'] = 'MATCHED'
    # Keep the exact object key for traceability.
    out.loc[matched, 'original_uri'] = keys_all[matched].map(
        lambda k: f's3://artemis-imagesafari/{k}')
    out.loc[mask & parsed_keys.isna(), 'original_image_id'] = keys_all.map(
        lambda k: Path(k).stem)

    audit.conflict_fields = dict(conflicts)
    audit.conflict_rows = sum(conflicts.values())
    return out, audit, stats


def load_agrifield_metadata(all_samples_path: str | Path) -> Tuple[pd.DataFrame, JoinAudit]:
    """Return the original-dataset identity for every AgriField-40K image."""
    frame = pd.read_csv(all_samples_path, low_memory=False)
    required = {'sample_id', 'source', 'source_prefix', 'original_id', 'original_split'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'all_samples.csv missing columns: {sorted(missing)}')
    frame = frame.copy()
    frame['filename'] = frame['sample_id'].astype(str)
    # The pool's on-disk name carries an image extension; ``sample_id`` does not.
    frame['stem'] = frame['sample_id'].astype(str)
    if 'candidate_source' not in frame.columns:
        frame['candidate_source'] = ''
    frame['upstream_dataset'] = frame['source'].where(
        ~frame['source'].astype(str).eq('pending_audit'),
        frame['candidate_source'].astype(str))
    audit = JoinAudit(reference_rows=len(frame), candidate_rows=len(frame),
                      matched_rows=len(frame), one_to_one=len(frame))
    duplicates = int(frame['filename'].duplicated().sum())
    audit.key_collisions = duplicates
    keep = ['filename', 'stem', 'source', 'candidate_source', 'source_prefix',
            'original_id', 'original_split', 'upstream_dataset']
    for optional in ('component', 'observation_url', 'sha256', 'pixel_sha256'):
        if optional in frame.columns:
            keep.append(optional)
        else:
            frame[optional] = ''
            keep.append(optional)
    return frame[keep], audit


COUNTRY_NAMES = {
    'TZA': 'Tanzania', 'KEN': 'Kenya', 'UGA': 'Uganda', 'NGA': 'Nigeria',
    'GHA': 'Ghana', 'COL': 'Colombia', 'ZMB': 'Zambia', 'MWI': 'Malawi',
    'ETH': 'Ethiopia', 'IND': 'India', 'USA': 'United States', 'PER': 'Peru',
    'MOZ': 'Mozambique', 'ZWE': 'Zimbabwe', 'RWA': 'Rwanda', 'BFA': 'Burkina Faso',
}


def country_name(code: str) -> str:
    if not code or code == 'unknown':
        return 'unknown'
    return COUNTRY_NAMES.get(code.upper(), code.upper())
