"""Version construction and programmatic freezing for Stage 5C.

The versions are intentionally layered:

``R0_AGRI147_SOURCE_POOL``
    Everything actually downloaded (plus verified local reuse), including
    holdouts, laboratory and UAV images.  Audit only.

``R1_AGRI147_REBUILD``
    Protocol-informed reconstruction: only rules the Agri-FM+ paper discloses
    (224 px short side, scene screening, verifiable licence) plus an explicit
    Laplacian approximation where the paper gives no threshold.

``R2_FIELD_STRICT``
    R1 + automated multi-stage QC + metadata/visual domain evidence + exact
    dedup + confirmed near-dedup + holdout/licence isolation.

``R3_FIELD_EXTENDED``
    R2 from the 35-source pool plus verified extension sources (Artemis,
    AgriField pool, sesame, apple orchard), all passed through the same gates.

AUTO_FROZEN is a statement about *programmatic* gates only.  It is never
HUMAN_VERIFIED.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

POLICY_VERSION = 'stage5c-policy-v1'

FIELD_DOMAINS = {'FIELD_IN_SITU', 'ORCHARD_IN_SITU'}
HOLDOUT_IDS = {
    'phenobench', 'gwhd_2021', 'plantdoc', 'minneapple',
    'roboflow_disease_detection', 'roboflow_crop_detection', 'plantseg',
    'multiclass_weeds', 'gwhd_instance_seg', 'deepweeds', 'growliflower',
    'cropandweed',
}


def policy_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_csv_sha(df: pd.DataFrame, fields: Sequence[str]) -> str:
    payload = df[list(fields)].to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def iqa_accept(row: pd.Series, accept_min: float = 45.0, reject_max: float = 25.0) -> Tuple[bool, str]:
    value = row.get('iqa_musiq')
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float('nan')
    # NaN compares False to everything, which would silently turn a missing
    # measurement into a pass.  Treat non-finite values as missing.
    if not math.isfinite(score):
        # A missing IQA score is a missing gate result, not a pass.  Sources
        # must be enriched by the automated IQA step before they can enter
        # R2/R3; otherwise the gate would silently accept unmeasured images.
        return False, 'IQA_MISSING'
    if score < reject_max:
        return False, 'IQA_REJECT'
    if score < accept_min:
        return False, 'IQA_QUARANTINE'
    return True, 'IQA_PASS'


def classify_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Assign per-row flags used by every version."""
    out = df.copy()
    for column, default in (('near_dup_unconfirmed', False),
                            ('dup_representative', True),
                            ('giant_cluster_suspect', False),
                            ('crop_label_if_metadata', ''),
                            ('group_known', True),
                            ('group_key', ''),
                            ('train_split', 'train'),
                            ('in_pool', True)):
        if column not in out.columns:
            out[column] = default
    out['in_pool'] = out['in_pool'].astype(str).str.lower().isin(['true', '1', 'yes'])
    out['decode_ok_b'] = out['decode_ok'].astype(str).str.lower().isin(['true', '1', 'yes'])
    out['holdout_b'] = out['holdout_flag'].astype(str).str.lower().eq('yes')
    out['archive_b'] = out['archive_only'].astype(str).str.lower().eq('yes')
    out['license_ok'] = out['license_status'].astype(str).str.startswith('ACCEPTED')
    out['domain_field'] = out['acquisition_domain'].isin(FIELD_DOMAINS)
    out['domain_ok'] = out['domain_field'] & ~out['domain_status'].astype(str).eq('QUARANTINE_DOMAIN')
    out['qc_accept'] = out['qc_status'].astype(str).eq('AUTO_ACCEPT')
    p0_accept = out.get('p0_status', pd.Series('', index=out.index)).astype(str).eq('AUTO_ACCEPT')
    # A licence that could not be verified is a metadata gap, not evidence of a
    # restriction.  For the *reconstruction* layer (R1) only, such rows are
    # re-admitted when licence was the sole P0 failure.  R2/R3 still require a
    # positively accepted licence.  Rows with a genuine licence conflict
    # (LICENSE_BLOCKED) never re-enter.
    soft_licence = out['license_status'].astype(str).eq('RESEARCH_ONLY_UNVERIFIED')
    only_licence_failed = out.get(
        'p0_reason_codes', pd.Series('', index=out.index)).fillna('').astype(str).apply(
            lambda value: bool(value) and set(p for p in value.split(';') if p)
            <= {'LICENSE_NOT_ACCEPTED'})
    p0_accept = p0_accept | (soft_licence & only_licence_failed & out['decode_ok_b'])
    out['p0_accept'] = p0_accept
    # R1 admits unverified-licence rows (audit/reconstruction only); R2/R3 do not.
    out['license_soft_ok'] = out['license_ok'] | soft_licence
    iqa = out.apply(lambda r: iqa_accept(r)[0], axis=1)
    out['iqa_ok'] = iqa
    out['p1_accept'] = out['qc_accept'] & out['iqa_ok']
    out['selected_for_R1'] = (
        out['in_pool'] & out['p0_accept'] & out['license_soft_ok']
        & ~out['holdout_b'] & ~out['archive_b'] & out['decode_ok_b'])
    out['selected_for_R2'] = (
        out['in_pool'] & out['selected_for_R1'] & out['license_ok']
        & out['p1_accept'] & out['domain_ok']
        & ~out['near_dup_unconfirmed']
        & out['dup_representative'] & ~out['giant_cluster_suspect'])
    # R3 = R2 plus extension sources that pass exactly the same strict gates.
    # Extensions are never allowed to back-fill R1/R2: those two layers are the
    # Agri-FM+ reconstruction and must stay inside the published source pool.
    extension_ok = (
        ~out['in_pool'] & out['decode_ok_b'] & out['license_ok']
        & out['p1_accept'] & out['domain_ok']
        & ~out['holdout_b'] & ~out['archive_b']
        & ~out['near_dup_unconfirmed']
        & out['dup_representative'] & ~out['giant_cluster_suspect'])
    out['selected_for_R3'] = out['selected_for_R2'] | extension_ok
    return out


def funnel(df: pd.DataFrame, version: str, strict: bool = True) -> List[Dict[str, object]]:
    """Single-cause funnel; every image is charged to exactly one step.

    ``strict=True`` is the R2/R3 pipeline (P1 + domain + dedup).  ``strict=False``
    is the R1 reconstruction pipeline: only rules the paper discloses (P0) plus
    licence/holdout safety, with no field-domain requirement.
    """
    if strict:
        steps = [
            ('downloaded', lambda d: d['decode_ok_b'] | ~d['decode_ok_b'], 'on-disk file'),
            ('decoded', lambda d: d['decode_ok_b'], 'PIL decode'),
            ('min_resolution', lambda d: d['short_side'].fillna(0).astype(float) >= 224, 'short side >= 224'),
            ('basic_qc', lambda d: ~d['qc_status'].astype(str).eq('AUTO_REJECT'), 'blur/exposure/blank'),
            ('iqa', lambda d: d['iqa_ok'], 'MUSIQ + NIQE'),
            ('domain', lambda d: d['domain_ok'], 'field / orchard evidence'),
            ('exact_dedup', lambda d: d['dup_representative'], 'byte/pixel identity'),
            ('near_dedup', lambda d: ~d['near_dup_unconfirmed'] & ~d['giant_cluster_suspect'], 'confirmed collapse / uncertainty isolation'),
            ('holdout_license_archive', lambda d: ~d['holdout_b'] & d['license_ok'] & ~d['archive_b'], 'holdout / licence / archive-only'),
        ]
    else:
        steps = [
            ('downloaded', lambda d: d['decode_ok_b'] | ~d['decode_ok_b'], 'on-disk file'),
            ('decoded', lambda d: d['decode_ok_b'], 'PIL decode'),
            ('paper_rules_p0', lambda d: d['p0_accept'], 'paper-disclosed rules only'),
            ('license', lambda d: d['license_soft_ok'], 'licence not blocked'),
            ('holdout_archive', lambda d: ~d['holdout_b'] & ~d['archive_b'], 'holdout / archive-only'),
        ]
    total = len(df)
    remaining = pd.Series(True, index=df.index)
    records: List[Dict[str, object]] = []
    cumulative = total
    for name, predicate, note in steps:
        passed = predicate(df)
        newly_excluded = remaining & ~passed
        cumulative = int((remaining & passed).sum())
        records.append({
            'version': version,
            'step': name,
            'input_images': int(remaining.sum()),
            'excluded_this_step': int(newly_excluded.sum()),
            'remaining_images': cumulative,
            'reason': note,
        })
        remaining = remaining & passed
    records.append({
        'version': version, 'step': 'strict_eligible', 'input_images': int(remaining.sum()),
        'excluded_this_step': 0, 'remaining_images': int(remaining.sum()),
        'reason': 'final strict set',
    })
    return records


def by_source_composition(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby('physical_source_id')
    frame = grouped.agg(
        downloaded=('image_id', 'count'),
        decoded=('decode_ok_b', 'sum'),
        basic_qc_pass=('qc_accept', 'sum'),
        iqa_pass=('iqa_ok', 'sum'),
        domain_pass=('domain_ok', 'sum'),
        holdout_removed=('holdout_b', 'sum'),
        license_blocked=('license_ok', lambda s: int((~s).sum())),
        r1=('selected_for_R1', 'sum'),
        r2=('selected_for_R2', 'sum'),
        r3=('selected_for_R3', 'sum'),
    ).reset_index()
    return frame


def manifest_payload(version: str, df: pd.DataFrame, policy_path: Path,
                     status: str, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    known = df[df['group_known'].astype(str).str.lower().isin(['true', '1'])] \
        if 'group_known' in df.columns else df.iloc[0:0]
    counts = {
        'images': int(len(df)),
        'sources': int(df['physical_source_id'].nunique()),
        'groups': int(known['group_key'].nunique()),
        'groups_unknown_images': int(len(df) - len(known)),
        'train': int((df['train_split'] == 'train').sum()),
        'val': int((df['train_split'] == 'val').sum()),
    }
    payload = {
        'version': version,
        'status': status,
        'auto_status_meaning': ('programmatic gates satisfied; NOT human-verified'
                                if status == 'AUTO_FROZEN' else status),
        'policy_version': POLICY_VERSION,
        'policy_sha256': policy_sha256(policy_path),
        'counts': counts,
        'by_source': df['physical_source_id'].value_counts().to_dict(),
        'by_domain': df['acquisition_domain'].value_counts().to_dict(),
        'by_crop': df['crop_label_if_metadata'].fillna('').replace('', 'unknown').value_counts().to_dict(),
        'max_source_share': float(df['physical_source_id'].value_counts(normalize=True).max()) if len(df) else 0.0,
        'cross_split_groups': int(known.groupby('group_key')['train_split'].nunique().gt(1).sum()) if len(known) else 0,
        'holdout_images': int(df['holdout_b'].sum()),
        'license_blocked_images': int((~df['license_ok']).sum()),
        'csv_sha256': canonical_csv_sha(df, ['image_id', 'physical_source_id', 'train_split']),
    }
    if extra:
        payload.update(extra)
    return payload


def _known_cross_split(df: pd.DataFrame) -> int:
    if 'group_known' not in df.columns or len(df) == 0:
        return 0
    known = df[df['group_known'].astype(str).str.lower().isin(['true', '1'])]
    if not len(known):
        return 0
    return int(known.groupby('group_key')['train_split'].nunique().gt(1).sum())


def freeze_gates(df: pd.DataFrame) -> Dict[str, bool]:
    return {
        'no_holdout': bool(df['holdout_b'].sum() == 0),
        'no_license_blocked': bool((~df['license_ok']).sum() == 0),
        'groups_disjoint': bool(_known_cross_split(df) == 0),
        'no_duplicate_representatives': bool(df['dup_representative'].all()),
        'no_unconfirmed_near_duplicates': bool(~df['near_dup_unconfirmed'].any()),
        'no_giant_clusters': bool(~df['giant_cluster_suspect'].any()),
        'all_qc_labelled': bool(df['qc_status'].astype(str).str.len().gt(0).all()),
        'all_domain_labelled': bool(df['acquisition_domain'].astype(str).str.len().gt(0).all()),
    }
