"""Acquisition-domain policy for the Stage 5B field corpus.

The Stage 5A registry records *whether* a dataset exists and is licensed.  It
does not record *where the camera was*.  This module adds that second axis and
turns it into an executable admission rule, so "the dataset name contains
crop" can never be mistaken for "the images were taken in a field".

Two admission levels are deliberately distinct:

``candidate``
    Enough evidence to enter ``agri-field-d1-candidate`` (CANDIDATE version).
    Model-assisted inspection is allowed, but the label is marked as such.

``freeze``
    Enough evidence to enter a FROZEN version.  Requires documented capture
    setting *and* a review that is not model-assisted-only.

Nothing in this module decides that an image is a field image.  It decides
which combinations of domain, evidence and review state are *admissible*, and
refuses every other combination with a reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

POLICY_VERSION = 'stage5b-field-domain-policy-v1'

DOMAINS: Tuple[str, ...] = (
    'FIELD_IN_SITU',
    'ORCHARD_IN_SITU',
    'PROTECTED_CULTIVATION',
    'CONTROLLED_LAB',
    'UAV_AERIAL',
    'SATELLITE_REMOTE',
    'UNKNOWN',
)

#: Domains admitted to the strict Field-D1 corpus (candidate and frozen).
FIELD_DOMAINS: Tuple[str, ...] = ('FIELD_IN_SITU', 'ORCHARD_IN_SITU')

#: Domains that are recorded but never admitted to strict Field-D1.
EXCLUDED_BY_DEFAULT: Tuple[str, ...] = (
    'CONTROLLED_LAB', 'PROTECTED_CULTIVATION',
    'UAV_AERIAL', 'SATELLITE_REMOTE', 'UNKNOWN',
)

EVIDENCE_TYPES: Tuple[str, ...] = (
    'official_documentation',      # the source's own docs state the capture setting
    'manual_inspection',           # a reviewer looked at images and recorded why
    'model_assisted_inspection',   # an image model described a sample; not a verdict
    'filename_or_metadata_only',   # inferred from paths / EXIF / folder names
    'none',
)

CONFIDENCES: Tuple[str, ...] = ('high', 'medium', 'low', 'unknown')

REVIEW_STATUSES: Tuple[str, ...] = (
    'reviewed',                     # a human recorded domain_review_reason
    'model_assisted_pending_human', # model evidence only, still needs a human
    'pending_human_review',
    'unreviewed',
)

#: Evidence types strong enough for a CANDIDATE version.
_CANDIDATE_EVIDENCE = {
    'official_documentation', 'manual_inspection', 'model_assisted_inspection',
}
#: Evidence types strong enough for a FROZEN version (no model-only labels).
_FREEZE_EVIDENCE = {'official_documentation', 'manual_inspection'}


class DomainPolicyError(ValueError):
    """Raised when a domain label or admission request is malformed."""


def default_policy() -> Dict[str, object]:
    """The serialisable policy written to ``data/registry/field_domain_policy.json``."""
    return {
        'policy_version': POLICY_VERSION,
        'domains': list(DOMAINS),
        'field_domains': list(FIELD_DOMAINS),
        'excluded_by_default': list(EXCLUDED_BY_DEFAULT),
        'evidence_types': list(EVIDENCE_TYPES),
        'confidences': list(CONFIDENCES),
        'review_statuses': list(REVIEW_STATUSES),
        'rules': {
            'candidate': {
                'allowed_domains': list(FIELD_DOMAINS),
                'allowed_evidence': sorted(_CANDIDATE_EVIDENCE),
                'allowed_confidence': ['high', 'medium'],
                'allowed_review_status': ['reviewed', 'model_assisted_pending_human'],
                'note': ('A CANDIDATE row may rest on documented capture setting or '
                         'model-assisted sample inspection, but the review status '
                         'must say which.'),
            },
            'freeze': {
                'allowed_domains': list(FIELD_DOMAINS),
                'allowed_evidence': sorted(_FREEZE_EVIDENCE),
                'allowed_confidence': ['high', 'medium'],
                'allowed_review_status': ['reviewed'],
                'note': ('Freezing requires a non-model evidence type and a human '
                         'review record; a model prediction is never a confirmation.'),
            },
        },
        'quality_gate': {
            'min_short_side_px': 224,
            'min_short_side_rationale': ('Agri-FM+ excludes sources below 224x224; the '
                                         'side is measured before any resize and images '
                                         'are never upscaled to pass the gate.'),
            'exclude_grayscale': True,
            'max_blur_laplacian_variance_below': 0.0,
            'blur_note': ('blur is reported as a distribution; only an explicit '
                          'per-source threshold may reject images'),
        },
        'holdout_rule': {
            'enforced': True,
            'note': ('A source listed in downstream_holdout_sources.csv is excluded '
                     'from every pretraining manifest, whatever its domain label.'),
        },
    }


def write_policy(path: Path, policy: Optional[Mapping[str, object]] = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(policy or default_policy())
    text = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    path.write_text(text, encoding='utf-8')
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_policy(path: Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_domain(domain: str) -> str:
    if domain not in DOMAINS:
        raise DomainPolicyError(
            f'unknown acquisition_domain {domain!r}; expected one of {list(DOMAINS)}')
    return domain


def validate_evidence(evidence: str) -> str:
    if evidence not in EVIDENCE_TYPES:
        raise DomainPolicyError(
            f'unknown domain_evidence_type {evidence!r}; expected {list(EVIDENCE_TYPES)}')
    return evidence


def validate_confidence(confidence: str) -> str:
    if confidence not in CONFIDENCES:
        raise DomainPolicyError(
            f'unknown domain_confidence {confidence!r}; expected {list(CONFIDENCES)}')
    return confidence


def validate_review_status(status: str) -> str:
    if status not in REVIEW_STATUSES:
        raise DomainPolicyError(
            f'unknown domain_review_status {status!r}; expected {list(REVIEW_STATUSES)}')
    return status


def admission(domain: str, evidence_type: str, confidence: str,
              review_status: str, level: str = 'candidate') -> Tuple[bool, str]:
    """Return ``(admitted, reason)`` for one image or source label.

    The reason is written to be quoted verbatim in an audit report.
    """
    validate_domain(domain)
    validate_evidence(evidence_type)
    validate_confidence(confidence)
    validate_review_status(review_status)
    if level not in ('candidate', 'freeze'):
        raise DomainPolicyError(f'unknown admission level {level!r}')
    rules = default_policy()['rules'][level]  # type: ignore[index]

    if domain not in rules['allowed_domains']:  # type: ignore[index]
        if domain == 'CONTROLLED_LAB':
            return False, 'controlled/laboratory imagery is excluded from Field-D1'
        if domain == 'PROTECTED_CULTIVATION':
            return False, ('protected cultivation is registered separately and excluded '
                           'from strict Field-D1 by default')
        if domain in ('UAV_AERIAL', 'SATELLITE_REMOTE'):
            return False, 'aerial/remote-sensing imagery is outside the near-ground scope'
        if domain == 'UNKNOWN':
            return False, 'an unverified acquisition domain never enters the main corpus'
        return False, f'domain {domain} is not admitted to Field-D1'
    if evidence_type not in rules['allowed_evidence']:  # type: ignore[index]
        return False, (f'evidence {evidence_type!r} is not sufficient for a '
                       f'{level.upper()} row')
    if confidence not in rules['allowed_confidence']:  # type: ignore[index]
        return False, f'confidence {confidence!r} is too weak for a {level.upper()} row'
    if review_status not in rules['allowed_review_status']:  # type: ignore[index]
        return False, (f'review status {review_status!r} does not satisfy the '
                       f'{level.upper()} gate')
    return True, f'admitted at {level}: {domain}/{evidence_type}/{confidence}/{review_status}'


def is_holdout(source_id: str, holdout_ids: Iterable[str]) -> bool:
    return source_id in set(holdout_ids)


def summarise_labels(rows: Sequence[Mapping[str, object]]) -> Dict[str, int]:
    """Count rows per domain (blank counts as UNKNOWN, never as FIELD)."""
    counts: Dict[str, int] = {d: 0 for d in DOMAINS}
    for row in rows:
        domain = str(row.get('acquisition_domain') or 'UNKNOWN')
        counts[domain if domain in counts else 'UNKNOWN'] += 1
    return counts
