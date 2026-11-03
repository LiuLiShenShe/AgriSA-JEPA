"""Exclusive, conserving exclusion accounting for Stage 5D.

Stage 5C reported a funnel whose last row did not equal the published manifest
(8 images for R2, 899 for R3) because the funnel predicates and the selection
boolean were two different expressions.  This module removes the possibility:
there is exactly one decision function, and every count in every report is a
``value_counts`` of its output.

Each image gets:

* ``primary_reason`` - one frozen, mutually exclusive reason (priority order
  below);
* ``secondary_reasons`` - every other check that also failed, so a licence
  problem is never hidden behind a quality problem;
* ``bucket`` - the accounting class used by the conservation identity.

Priority order is a policy decision, not an accident.  Holdout and licence come
first because those are hard external constraints; modality comes next because
a non-visible image cannot be repaired by quality filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

#: (reason, bucket, human label).  Order is the frozen priority order.
PRIMARY_ORDER: Sequence[Tuple[str, str, str]] = (
    ('HOLDOUT_EXCLUDED', 'holdout', 'registered downstream holdout'),
    ('LICENCE_BLOCKED', 'licence_blocked', 'licence unverified or conflicting'),
    ('ARCHIVE_ONLY', 'archive_only', 'archive-only source'),
    ('MODALITY_NOT_VISIBLE_RGB', 'rejected', 'documented non-visible modality'),
    ('MODALITY_UNKNOWN', 'quarantined', 'modality cannot be evidenced'),
    ('DECODE_FAILED', 'rejected', 'file did not decode'),
    ('BELOW_MIN_RESOLUTION', 'rejected', 'short side below policy minimum'),
    ('BASIC_QC_REJECT', 'rejected', 'deterministic basic-quality rejection'),
    ('BASIC_QC_QUARANTINE', 'quarantined', 'basic-quality ambiguity'),
    ('IQA_REJECT', 'rejected', 'reference-free IQA below reject threshold'),
    ('IQA_QUARANTINE', 'quarantined', 'reference-free IQA in the grey band'),
    ('IQA_MISSING', 'quarantined', 'reference-free IQA not measured'),
    ('DOMAIN_REJECT', 'rejected', 'acquisition domain excluded by policy'),
    ('DOMAIN_QUARANTINE', 'quarantined', 'acquisition domain uncertain'),
    ('EXACT_DUPLICATE', 'duplicate_removed', 'byte/pixel identical copy'),
    ('NEAR_DUPLICATE', 'duplicate_removed', 'confirmed near duplicate'),
    ('DUPLICATE_UNCERTAIN', 'quarantined', 'near-duplicate candidate unresolved'),
    ('GIANT_CLUSTER', 'quarantined', 'texture-collision cluster'),
)

PRIMARY_INDEX = {reason: index for index, (reason, _, _) in enumerate(PRIMARY_ORDER)}
BUCKET_BY_REASON = {reason: bucket for reason, bucket, _ in PRIMARY_ORDER}
ACCEPTED = 'ACCEPTED'


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(('true', '1', 'yes', 'y'))


def decide_frame(frame: pd.DataFrame, *, iqa_status_col: str = 'iqa_gate_status',
                 iqa_reason_col: str = 'iqa_reasons',
                 basic_status_col: str = 'basic_status',
                 basic_reason_col: str = 'basic_reasons',
                 near_dup_col: str = 'near_dup_status') -> pd.DataFrame:
    """Attach decision columns.  Every row is charged to exactly one reason."""
    out = frame.copy()

    holdout = out.get('holdout_status', pd.Series('OK', index=out.index)) \
        .astype(str).str.upper().eq('HOLDOUT')
    licence_blocked = ~_as_bool(out.get('license_ok', pd.Series(True, index=out.index)))
    archive = _as_bool(out.get('archive_only', pd.Series(False, index=out.index)))
    modality_status = out['modality_status'].astype(str)
    modality_unknown = modality_status.eq('UNKNOWN_MODALITY')
    modality_not_visible = modality_status.eq('NOT_VISIBLE_RGB')

    decode_ok = _as_bool(out['decode_ok'])
    below_res = out['short_side'].fillna(0).astype(float) < 224

    basic = out[basic_status_col].astype(str).str.upper()
    basic_reject = basic.eq('REJECT')
    basic_quarantine = basic.eq('QUARANTINE')

    iqa = out[iqa_status_col].astype(str).str.upper()
    iqa_reject = iqa.eq('REJECT')
    iqa_quarantine = iqa.eq('QUARANTINE')
    iqa_missing = iqa.eq('MISSING')

    domain = out.get('domain_status', pd.Series('OK', index=out.index)).astype(str)
    domain_field = out['acquisition_domain'].isin(('FIELD_IN_SITU', 'ORCHARD_IN_SITU'))
    domain_quarantine = domain.eq('QUARANTINE_DOMAIN') | (
        out['acquisition_domain'].astype(str).eq('UNKNOWN'))
    domain_reject = ~domain_field & ~domain_quarantine

    exact_dup = _as_bool(out.get('exact_duplicate', pd.Series(False, index=out.index)))
    near = out.get(near_dup_col, pd.Series('OK', index=out.index)).astype(str).str.upper()
    near_confirmed = near.eq('CONFIRMED_NEAR_DUPLICATE')
    near_uncertain = near.eq('UNCERTAIN_CANDIDATE')
    giant = near.eq('GIANT_CLUSTER')

    conditions: List[Tuple[str, pd.Series]] = [
        ('HOLDOUT_EXCLUDED', holdout),
        ('LICENCE_BLOCKED', licence_blocked),
        ('ARCHIVE_ONLY', archive),
        ('MODALITY_NOT_VISIBLE_RGB', modality_not_visible),
        ('MODALITY_UNKNOWN', modality_unknown),
        ('DECODE_FAILED', ~decode_ok),
        ('BELOW_MIN_RESOLUTION', below_res),
        ('BASIC_QC_REJECT', basic_reject),
        ('BASIC_QC_QUARANTINE', basic_quarantine),
        ('IQA_REJECT', iqa_reject),
        ('IQA_QUARANTINE', iqa_quarantine),
        ('IQA_MISSING', iqa_missing),
        ('DOMAIN_REJECT', domain_reject),
        ('DOMAIN_QUARANTINE', domain_quarantine),
        ('EXACT_DUPLICATE', exact_dup),
        ('NEAR_DUPLICATE', near_confirmed),
        ('DUPLICATE_UNCERTAIN', near_uncertain),
        ('GIANT_CLUSTER', giant),
    ]

    reasons_in_order = [reason for reason, _ in conditions]
    fail_matrix = np.column_stack([mask.to_numpy(dtype=bool) for _, mask in conditions])
    any_failed = fail_matrix.any(axis=1)
    first_index = np.where(any_failed, fail_matrix.argmax(axis=1), -1)
    primary = np.where(any_failed,
                       np.array(reasons_in_order, dtype=object)[np.clip(first_index, 0, None)],
                       ACCEPTED)
    secondary: List[str] = []
    for row_index in range(fail_matrix.shape[0]):
        if not any_failed[row_index]:
            secondary.append('')
            continue
        first = int(first_index[row_index])
        secondary.append(';'.join(reasons_in_order[i] for i in range(len(reasons_in_order))
                                  if i != first and fail_matrix[row_index, i]))
    out['primary_reason'] = primary
    out['secondary_reasons'] = secondary
    out['decision'] = ['ACCEPTED' if reason == ACCEPTED else 'EXCLUDED'
                       for reason in primary]
    out['bucket'] = [BUCKET_BY_REASON.get(reason, 'accepted') for reason in primary]
    out['selected'] = primary == ACCEPTED
    return out


def conservation(frame: pd.DataFrame, input_col: str = 'image_id') -> Dict[str, object]:
    """Return the conservation identity for the frame.

    ``N_input == N_accepted + N_rejected + N_quarantined + N_holdout +
    N_licence_blocked + N_archive_only + N_duplicate_removed`` and the reason
    counts must add up to the same total.
    """
    total = int(len(frame))
    buckets = frame['bucket'].value_counts().to_dict()
    reasons = frame['primary_reason'].value_counts().to_dict()
    bucket_total = int(sum(buckets.values()))
    reason_total = int(sum(reasons.values()))
    unique = int(frame[input_col].nunique()) if input_col in frame else total
    return {
        'n_input': total,
        'n_unique_ids': unique,
        'n_accepted': int(buckets.get('accepted', 0)),
        'n_rejected': int(buckets.get('rejected', 0)),
        'n_quarantined': int(buckets.get('quarantined', 0)),
        'n_holdout': int(buckets.get('holdout', 0)),
        'n_licence_blocked': int(buckets.get('licence_blocked', 0)),
        'n_archive_only': int(buckets.get('archive_only', 0)),
        'n_duplicate_removed': int(buckets.get('duplicate_removed', 0)),
        'bucket_total': bucket_total,
        'reason_total': reason_total,
        'by_bucket': buckets,
        'by_primary_reason': reasons,
        'conserved': bool(bucket_total == total == reason_total == unique),
    }


def funnel_table(frame: pd.DataFrame, version: str) -> List[Dict[str, object]]:
    """Single-cause funnel: each row is charged at its primary-reason step."""
    total = int(len(frame))
    remaining = total
    records: List[Dict[str, object]] = [{
        'version': version, 'step': 'input', 'reason': 'all ledger rows',
        'input_images': total, 'excluded_this_step': 0, 'remaining_images': total,
    }]
    counts = frame['primary_reason'].value_counts().to_dict()
    for reason, bucket, label in PRIMARY_ORDER:
        excluded = int(counts.get(reason, 0))
        remaining -= excluded
        records.append({
            'version': version, 'step': reason, 'reason': label,
            'input_images': remaining + excluded,
            'excluded_this_step': excluded,
            'remaining_images': remaining,
        })
    records.append({
        'version': version, 'step': 'ACCEPTED', 'reason': 'strict eligible',
        'input_images': remaining, 'excluded_this_step': 0,
        'remaining_images': remaining,
    })
    return records


def per_source_conservation(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-source bucket counts; totals must equal the global totals."""
    table = (frame.groupby(['physical_source_id', 'primary_reason'])
             .size().unstack(fill_value=0))
    table['total'] = table.sum(axis=1)
    return table.reset_index()


def assert_conserved(frame: pd.DataFrame) -> None:
    report = conservation(frame)
    if not report['conserved']:
        raise AssertionError(f'funnel is not conserved: {report}')
