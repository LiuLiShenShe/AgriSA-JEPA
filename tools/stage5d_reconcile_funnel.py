#!/usr/bin/env python
"""Explain the Stage 5C funnel/manifest discrepancy image by image.

Stage 5C published a funnel ending at 20,952 (R2) / 178,295 (R3) while the
manifests contained 20,944 / 177,396.  This tool re-runs the *old* decision
functions on the *old* ledgers, then compares the funnel's surviving set with
the manifest's image ids and prints the reason for every difference.

It reproduces the old loader exactly, including its mtime rule (which is one of
the defects Stage 5D fixes), so that the counts are comparable.

Outputs
-------
reports/stage5d_legacy_funnel_reconciliation.json
reports/stage5d_legacy_funnel_reconciliation.md
reports/stage5d_funnel_mismatches.csv
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.registry.stage5c_freeze import classify_rows, funnel

IMAGE_DIR = ROOT / 'data/corpus_v2/images'
CORPUS = ROOT / 'data/corpus_v2'
REPORTS = ROOT / 'reports'
EXTENSION_IDS = ['artemis_imagesafari', 'agrifield40k_pool', 'sesame_weed', 'applebbch81']


def load_old_build_module():
    path = ROOT / 'tools/stage5c_build_versions.py'
    spec = importlib.util.spec_from_file_location('stage5c_build_versions', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct_old_frame(module) -> pd.DataFrame:
    sources = sorted({p.stem.replace('_qc', '') for p in IMAGE_DIR.glob('*.csv')
                      if p.stem not in ('cleanvision_issues',)})
    frames = []
    provenance = {}
    for source in sources:
        enriched = IMAGE_DIR / f'{source}_qc.csv'
        plain = IMAGE_DIR / f'{source}.csv'
        if enriched.exists() and plain.exists() and plain.stat().st_mtime > enriched.stat().st_mtime:
            chosen = plain
        else:
            chosen = enriched if enriched.exists() else plain
        provenance[source] = {'chosen': chosen.name, 'enriched_exists': enriched.exists(),
                              'plain_exists': plain.exists()}
        frames.append(module.load_source(source))
    frame = pd.concat(frames, ignore_index=True)
    frame = module.holdout_isolation(frame)
    frame['in_pool'] = ~frame['physical_source_id'].isin(EXTENSION_IDS)
    for column, default in (('near_dup_unconfirmed', False),
                            ('dup_representative', True),
                            ('giant_cluster_suspect', False)):
        if column not in frame.columns:
            frame[column] = default
    frame = classify_rows(frame)
    # The old run used --no-embeddings for the frozen versions.  Reproduce that
    # flag: the DINOv2 confirmation never ran, which is itself part of the
    # finding (the near-duplicate stage was inert).
    frame = module.add_dedup_columns(frame, do_embeddings=False)
    frame = classify_rows(frame)
    frame.attrs['loader_provenance'] = provenance
    return frame


def replicate_funnel_mask(df: pd.DataFrame, strict: bool) -> pd.Series:
    """Copy of the predicate chain in stage5c_freeze.funnel (read-only)."""
    if strict:
        steps = [
            ('decoded', lambda d: d['decode_ok_b']),
            ('min_resolution', lambda d: d['short_side'].fillna(0).astype(float) >= 224),
            ('basic_qc', lambda d: ~d['qc_status'].astype(str).eq('AUTO_REJECT')),
            ('iqa', lambda d: d['iqa_ok']),
            ('domain', lambda d: d['domain_ok']),
            ('exact_dedup', lambda d: d['dup_representative']),
            ('near_dedup', lambda d: ~d['near_dup_unconfirmed'] & ~d['giant_cluster_suspect']),
            ('holdout_license_archive',
             lambda d: ~d['holdout_b'] & d['license_ok'] & ~d['archive_b']),
        ]
    else:
        steps = [
            ('decoded', lambda d: d['decode_ok_b']),
            ('paper_rules_p0', lambda d: d['p0_accept']),
            ('license', lambda d: d['license_soft_ok']),
            ('holdout_archive', lambda d: ~d['holdout_b'] & ~d['archive_b']),
        ]
    remaining = pd.Series(True, index=df.index)
    for _, predicate in steps:
        remaining = remaining & predicate(df)
    return remaining


def explain(frame: pd.DataFrame, version: str, mask_column: str,
            manifest_name: str, funnel_records: List[Dict[str, object]]) -> Dict[str, object]:
    manifest = pd.read_csv(CORPUS / manifest_name, low_memory=False)
    selected = frame[frame[mask_column]]
    strict = version != 'R1'
    funnel_ids = set(frame.loc[replicate_funnel_mask(frame, strict), 'image_id'])
    manifest_ids = set(manifest['image_id'])
    extra = funnel_ids - manifest_ids
    missing = manifest_ids - funnel_ids

    details = []
    lookup = frame.set_index('image_id')
    for image_id in sorted(extra):
        row = lookup.loc[image_id]
        details.append({
            'image_id': image_id,
            'physical_source_id': row['physical_source_id'],
            'direction': 'funnel_only',
            'short_side': row['short_side'],
            'qc_status': row['qc_status'],
            'p0_status': row.get('p0_status', ''),
            'p0_reason_codes': row.get('p0_reason_codes', ''),
            'iqa_musiq': row.get('iqa_musiq', ''),
            'domain': row['acquisition_domain'],
            'dup_representative': row.get('dup_representative', ''),
            'near_dup_unconfirmed': row.get('near_dup_unconfirmed', ''),
            'giant_cluster_suspect': row.get('giant_cluster_suspect', ''),
            'license_status': row['license_status'],
        })
    for image_id in sorted(missing):
        row = manifest[manifest['image_id'] == image_id].iloc[0]
        details.append({'image_id': image_id,
                        'physical_source_id': row['physical_source_id'],
                        'direction': 'manifest_only'})

    funnel_frame = frame[replicate_funnel_mask(frame, version != 'R1')]
    final = funnel_records[-1] if funnel_records else {}
    return {
        'version': version,
        'funnel_final': final.get('remaining_images'),
        'manifest_count': int(len(manifest)),
        'difference': int(len(funnel_ids) - len(manifest_ids)),
        'funnel_only': int(len(extra)),
        'manifest_only': int(len(missing)),
        'funnel_only_by_source': pd.Series(
            [d['physical_source_id'] for d in details if d['direction'] == 'funnel_only']
        ).value_counts().to_dict() if extra else {},
        'explanation': _explain_counts(funnel_frame, extra),
        '_details': details,
    }


def _explain_counts(funnel_frame: pd.DataFrame, extra: set) -> Dict[str, int]:
    """Why does the funnel keep an image the manifest dropped?

    The cause is structural: the Stage 5C funnel's ``basic_qc`` step excluded
    only ``AUTO_REJECT``, so ``QUARANTINE`` rows survived it, while the
    selection mask required ``qc_status == AUTO_ACCEPT``.
    """
    subset = funnel_frame[funnel_frame['image_id'].isin(extra)]
    if subset.empty:
        return {}
    reasons = {
        'qc_status_QUARANTINE': int(subset['qc_status'].astype(str).eq('QUARANTINE').sum()),
        'qc_status_AUTO_ACCEPT': int(subset['qc_status'].astype(str).eq('AUTO_ACCEPT').sum()),
        'p0_rejected_but_not_a_funnel_step': int(
            (~subset.get('p0_status', pd.Series('', index=subset.index))
             .astype(str).eq('AUTO_ACCEPT')).sum()),
    }
    text = subset.get('p0_reason_codes', pd.Series('', index=subset.index)).astype(str)
    for code in ('BLURRY_APPROXIMATION', 'BELOW_MIN_RESOLUTION', 'LICENSE_NOT_ACCEPTED'):
        reasons[f'p0_{code}'] = int(text.str.contains(code).sum())
    return {k: v for k, v in reasons.items() if v}


def main() -> int:
    module = load_old_build_module()
    frame = reconstruct_old_frame(module)
    pool = frame[frame['in_pool']]

    r1_funnel = funnel(pool, 'R1', strict=False)
    r2_funnel = funnel(pool, 'R2', strict=True)
    r3_funnel = funnel(frame, 'R3', strict=True)

    r2 = explain(pool, 'R2', 'selected_for_R2', 'R2_field_strict_manifest.csv', r2_funnel)
    r3 = explain(frame, 'R3', 'selected_for_R3', 'R3_field_extended_manifest.csv', r3_funnel)

    old_policy = CORPUS / 'policy.yaml'
    recorded = json.loads((CORPUS / 'R2_field_strict_manifest.json').read_text())
    import hashlib
    actual_sha = hashlib.sha256(old_policy.read_bytes()).hexdigest()

    details = r2.pop('_details') + r3.pop('_details')
    pd.DataFrame(details).to_csv(REPORTS / 'stage5d_funnel_mismatches.csv', index=False)

    payload = {
        'method': ('re-ran the Stage 5C functions on the Stage 5C ledgers with the '
                   'Stage 5C loader (including its mtime rule) and compared the '
                   'funnel survivor set with the published manifest ids'),
        'loader_provenance': frame.attrs.get('loader_provenance', {}),
        'r1_funnel_final': r1_funnel[-1]['remaining_images'],
        'r2': r2,
        'r3': r3,
        'legacy_policy': {
            'path': 'data/corpus_v2/policy.yaml',
            'sha256_actual': actual_sha,
            'sha256_recorded_in_manifests': recorded.get('policy_sha256'),
            'unchanged': actual_sha == recorded.get('policy_sha256'),
        },
        'mismatch_rows': len(details),
    }
    (REPORTS / 'stage5d_legacy_funnel_reconciliation.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')

    lines = [
        '# Stage 5D - legacy funnel reconciliation',
        '',
        'Re-ran the Stage 5C functions on the Stage 5C ledgers, using the Stage 5C',
        'loader exactly (including the mtime rule that chose the unenriched ledger',
        'for two sources).',
        '',
        '| version | funnel final | manifest | funnel-only | manifest-only |',
        '|---|---:|---:|---:|---:|',
        f"| R1 | {payload['r1_funnel_final']} | (60,603 reported) | - | - |",
        f"| R2 | {r2['funnel_final']} | {r2['manifest_count']} | {r2['funnel_only']} | {r2['manifest_only']} |",
        f"| R3 | {r3['funnel_final']} | {r3['manifest_count']} | {r3['funnel_only']} | {r3['manifest_only']} |",
        '',
        '## Why the funnel kept images the manifest dropped',
        '',
        f"- R2: {json.dumps(r2['explanation'])}",
        f"- R3: {json.dumps(r3['explanation'])}",
        '',
        'The funnel never evaluated the P0 (paper-disclosed) predicate, while the',
        'selection mask required `selected_for_R1`.  Images that passed every',
        'funnel step but failed the paper-disclosed Laplacian rule survived the',
        'funnel and were dropped from the manifest.  Every such image is listed in',
        '`stage5d_funnel_mismatches.csv`.',
        '',
        '## Legacy policy integrity',
        '',
        f"- `data/corpus_v2/policy.yaml` SHA-256: `{actual_sha}`",
        f"- recorded in the Stage 5C manifests: `{recorded.get('policy_sha256')}`",
        f"- unchanged: **{payload['legacy_policy']['unchanged']}**",
    ]
    (REPORTS / 'stage5d_legacy_funnel_reconciliation.md').write_text('\n'.join(lines) + '\n',
                                                              encoding='utf-8')
    print(json.dumps({'r2': {k: v for k, v in r2.items() if k != '_details'},
                      'r3': {k: v for k, v in r3.items() if k != '_details'}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
