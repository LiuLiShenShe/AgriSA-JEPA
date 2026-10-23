"""Data-gate schema for the Stage 5A agricultural source registry.

The registry is a *ledger*, not a wish list.  Every row carries the evidence
that justifies its state, and the state machine below is strictly ordered: a
row can only advance one gate at a time, and every hop requires an explicit
machine-checkable predicate (see ``advance``).  Nothing reaches ``FROZEN``
without a SHA-256 of the frozen manifest.

Field groups follow ``reports/stage5a_data_source_catalog.md``:

* identity      -- source_id, official name, paper, URLs
* counts        -- reported / remote / downloaded / decoded / approved
* content       -- modality, domain, platform, resolution, grouping fields
* legal         -- licence, attribution, allowed_in_project
* provenance    -- evidence, human review state, failure reason
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- #
# Ordered gate states
# --------------------------------------------------------------------------- #

STATES: List[str] = [
    'DISCOVERED',
    'URL_VERIFIED',
    'LICENSE_VERIFIED',
    'DOWNLOADED',
    'DECODED',
    'DUPLICATE_SCREENED',
    'SPLIT_CHECKED',
    'APPROVED',
    'FROZEN',
]

BLOCKED = 'BLOCKED'

#: Fields that must be non-empty before a row may enter a given state.
#: ``ALL`` values are checked on top of the per-state requirements.
REQUIRED_FOR_STATE: Dict[str, Sequence[str]] = {
    'DISCOVERED': ('source_id', 'official_name', 'paper', 'year'),
    'URL_VERIFIED': ('paper_url', 'download_url'),
    'LICENSE_VERIFIED': ('license', 'license_evidence'),
    'DOWNLOADED': ('download_evidence', 'files_downloaded'),
    'DECODED': ('decode_evidence', 'images_decoded'),
    'DUPLICATE_SCREENED': ('duplicate_evidence', 'images_duplicate_free'),
    'SPLIT_CHECKED': ('split_evidence', 'downstream_holdout'),
    'APPROVED': ('approval_evidence', 'human_review_status'),
    'FROZEN': ('freeze_manifest',),
}

#: The five *distinct* quantities the task requires.  They are intentionally
#: separate columns so "147,285 images" can never be read as "on this machine".
COUNT_FIELDS = (
    'images_reported',      # number stated in the source paper
    'files_remote',         # number of files the remote archive lists
    'files_downloaded',     # number of files actually on this machine
    'images_decoded',       # files that decode as an image
    'images_approved',      # rows that passed every audit and were accepted
)

SOURCE_FIELDS: List[str] = [
    # identity
    'source_id',
    'official_name',
    'catalog_groups',
    'paper',
    'year',
    'paper_url',
    'download_url',
    'repo_url',
    # content
    'category',
    'domain',
    'modality',
    'platform',
    'resolution',
    'grouping_fields',
    'official_splits',
    # counts
    'images_reported',
    'files_remote',
    'files_downloaded',
    'images_decoded',
    'images_duplicate_free',
    'images_approved',
    # legal
    'license',
    'license_evidence',
    'attribution_required',
    'allowed_in_project',
    # overlap / holdout
    'overlaps_agrifield40k',
    'overlaps_agrifmplus',
    'downstream_holdout',
    'overlap_evidence',
    # state / provenance
    'state',
    'downloaded',
    'download_failure',
    'human_review_status',
    'download_evidence',
    'decode_evidence',
    'duplicate_evidence',
    'split_evidence',
    'approval_evidence',
    'freeze_manifest',
    'evidence',
    'notes',
]

INT_FIELDS = set(COUNT_FIELDS) | {'files_remote', 'files_downloaded'}


class RegistryError(ValueError):
    """Raised when a registry row violates the gate schema."""


def _int_or_blank(value) -> Optional[int]:
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise RegistryError('boolean is not a count')
    if isinstance(value, float):
        if not value.is_integer():
            raise RegistryError(f'non-integral count: {value}')
        return int(value)
    if isinstance(value, str):
        text = value.strip().replace(',', '')
        if not text:
            return None
        if not text.isdigit():
            raise RegistryError(f'non-numeric count: {value!r}')
        return int(text)
    if isinstance(value, int):
        return value
    raise RegistryError(f'unsupported count type: {type(value).__name__}')


def validate_row(row: Mapping[str, object]) -> Dict[str, object]:
    """Validate and normalise one source row.

    Missing optional fields become ``''``; count fields become ``int`` or
    ``None``; illegal ``state`` / ``allowed_in_project`` values raise.
    """
    unknown = set(row) - set(SOURCE_FIELDS)
    if unknown:
        raise RegistryError(f'unknown registry fields: {sorted(unknown)}')
    out: Dict[str, object] = {}
    for field in SOURCE_FIELDS:
        value = row.get(field, '')
        if field in INT_FIELDS:
            value = _int_or_blank(value)
        elif value is None:
            value = ''
        out[field] = value
    if not str(out['state']):
        out['state'] = STATES[0]      # DISCOVERED
    if not str(out['allowed_in_project']):
        out['allowed_in_project'] = 'pending'
    state = out['state']
    if state not in STATES and state != BLOCKED:
        raise RegistryError(f'illegal state {state!r}')
    allowed = out['allowed_in_project']
    if allowed not in ('', 'yes', 'no', 'pending'):
        raise RegistryError(f'allowed_in_project must be yes/no/pending, got {allowed!r}')
    if not str(out['source_id']):
        raise RegistryError('source_id is required')
    return out


def missing_for_state(row: Mapping[str, object], state: str) -> List[str]:
    """Fields still empty for ``state`` (``BLOCKED`` needs none)."""
    if state == BLOCKED:
        return []
    return [f for f in REQUIRED_FOR_STATE[state] if str(row.get(f, '') or '').strip() == '']


def advance(row: Mapping[str, object], target: str) -> Dict[str, object]:
    """Advance one gate, refusing skips and refusing to satisfy a gate with blanks."""
    row = validate_row(row)
    current = row['state']
    if current == BLOCKED or target == BLOCKED:
        raise RegistryError('unblock a BLOCKED row explicitly before advancing')
    if target not in STATES or current not in STATES:
        raise RegistryError(f'unknown transition {current!r} -> {target!r}')
    if STATES.index(target) != STATES.index(current) + 1:
        raise RegistryError(
            f'states must advance one gate at a time: {current!r} -> {target!r}')
    missing = missing_for_state(row, target)
    if missing:
        raise RegistryError(
            f'cannot enter {target}: missing evidence fields {missing}')
    row = dict(row)
    row['state'] = target
    return row


def can_freeze(row: Mapping[str, object]) -> bool:
    """A row may freeze only at APPROVED with a positive approved count."""
    row = validate_row(row)
    return (row['state'] == 'APPROVED'
            and bool(row['images_approved'])
            and int(row['images_approved']) > 0
            and row['allowed_in_project'] == 'yes'
            and row['human_review_status'] in ('reviewed', 'auto+reviewed'))


# --------------------------------------------------------------------------- #
# CSV / JSON IO
# --------------------------------------------------------------------------- #

def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(validate_row(row))


def read_csv(path: Path) -> List[Dict[str, object]]:
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return [validate_row(row) for row in rows]


def sha256_rows(rows: Sequence[Mapping[str, object]]) -> str:
    """Stable SHA-256 over the canonical CSV serialisation of ``rows``."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SOURCE_FIELDS, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow(validate_row(row))
    return hashlib.sha256(buffer.getvalue().encode('utf-8')).hexdigest()


def write_manifest(path: Path, rows: Sequence[Mapping[str, object]],
                   extra: Optional[Mapping[str, object]] = None) -> str:
    payload = {
        'rows': len(rows),
        'sha256_canonical_csv': sha256_rows(rows),
        'states': {s: sum(1 for r in rows if r['state'] == s) for s in STATES + [BLOCKED]},
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                    encoding='utf-8')
    return payload['sha256_canonical_csv']
