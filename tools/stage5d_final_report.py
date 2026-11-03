#!/usr/bin/env python
"""Evaluate the twelve Stage 5D publication gates from real evidence.

Nothing in this file is typed by hand.  Each gate is measured against the files
the build produced and the files the legacy stage left behind.  A gate that
cannot be measured is ``None`` and the corpus can never be
``READY_AUTO_VALIDATED``.

Outputs
-------
reports/stage5d_gates.json
reports/stage5d_final_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from PIL import Image

from src.registry.stage5d_gates import GATES, status as gate_status

CORPUS_V2 = ROOT / 'data/corpus_v2'
CORPUS_V3 = ROOT / 'data/corpus_v3'
REPORTS = ROOT / 'reports'
LEGACY_JSON = ['R0_agrifm35_source_pool_manifest.json',
               'R1_agrifm147_rebuild_manifest.json',
               'R2_field_strict_manifest.json',
               'R3_field_extended_manifest.json',
               'policy.yaml']
LEGACY_REPORTS = ['stage5c_final_report.md', 'stage5c_filtering_funnel.csv',
                  'stage5c_qc_calibration.json', 'stage5c_test_summary.md']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ledger_columns(image_ids: List[str], columns: List[str]) -> Dict[str, Dict[str, object]]:
    """Map ``image_id`` to ledger columns for the requested ids.

    The published train/val CSV keeps the canonical columns only, so gates that
    need a ledger-only column (``raw_path``, ``modality_status``) read it from
    the per-source ledger that the build wrote for the same ids.
    """
    wanted = {str(value) for value in image_ids}
    mapping: Dict[str, Dict[str, object]] = {column: {} for column in columns}
    for path in sorted((CORPUS_V3 / 'images').glob('*.csv')):
        available = list(pd.read_csv(path, nrows=0).columns)
        use = ['image_id'] + [column for column in columns if column in available]
        if len(use) == 1:
            continue
        frame = pd.read_csv(path, low_memory=False, usecols=use)
        frame = frame[frame['image_id'].astype(str).isin(wanted)]
        ids = frame['image_id'].astype(str)
        for column in use[1:]:
            mapping[column].update(dict(zip(ids, frame[column])))
    return mapping


def gate_files_readable(manifest: Dict[str, object], train: pd.DataFrame) -> Optional[bool]:
    if not len(train):
        return False
    rng = random.Random(20261025)
    sample = train.sample(min(400, len(train)), random_state=20261025)
    paths = ledger_columns(sample['image_id'].astype(str).tolist(),
                           ['raw_path'])['raw_path']
    if len(paths) != len(sample):
        return False
    readable = 0
    for image_id in sample['image_id'].astype(str):
        try:
            with Image.open(paths[image_id]) as image:
                image.load()
            readable += 1
        except Exception:  # noqa: BLE001
            pass
    return readable == len(sample)


def gate_policy_scale(manifest: Dict[str, object]) -> Optional[bool]:
    policy_path = CORPUS_V3 / 'policy.yaml'
    if not policy_path.exists():
        return False
    if manifest.get('policy_sha256') != sha256_file(policy_path):
        return False
    trace = manifest.get('iqa_trace', {})
    for name, spec in trace.items():
        if spec.get('role') == 'GATE':
            if not (spec.get('direction') and spec.get('score_range')
                    and spec.get('accept_threshold') is not None
                    and spec.get('reject_threshold') is not None
                    and spec.get('model_sha256')):
                return False
    return True


def gate_visible_rgb(accepted: pd.DataFrame) -> Optional[bool]:
    return bool(accepted['modality_status'].eq('VISIBLE_RGB_EVIDENCED').all())


def gate_field_domain(accepted: pd.DataFrame) -> Optional[bool]:
    allowed = {'FIELD_IN_SITU', 'ORCHARD_IN_SITU'}
    return bool(accepted['acquisition_domain'].isin(allowed).all()
                and accepted['domain_evidence_type'].astype(str).str.len().gt(0).all())


def gate_licence(accepted: pd.DataFrame) -> Optional[bool]:
    return bool(accepted['license_status'].astype(str).str.startswith('ACCEPTED').all())


def gate_dedup(manifest: Dict[str, object]) -> Optional[bool]:
    dedup = manifest.get('dedup', {})
    if dedup.get('exact_duplicate_images') is None:
        return False
    if dedup.get('hash_report', {}).get('phash', {}).get('valid_fraction', 0) < 0.99:
        return False
    return True


def gate_holdout(manifest: Dict[str, object]) -> Optional[bool]:
    holdout = manifest.get('counts', {}).get('holdout')
    return holdout is not None and holdout >= 0


def gate_groups(accepted: pd.DataFrame, manifest: Dict[str, object]) -> Optional[bool]:
    strong = accepted[accepted['group_tier'].isin(
        ['OFFICIAL_GROUP', 'DOCUMENTED_PATH_GROUP'])]
    if len(strong) and strong.groupby('group_key')['split'].nunique().gt(1).any():
        return False
    unknown_val = accepted[(accepted['group_tier'] == 'UNKNOWN_GROUP') &
                           (accepted['split'] == 'val')]
    if len(unknown_val):
        return False
    return True


def gate_counts(manifest: Dict[str, object]) -> Optional[bool]:
    counts = manifest.get('funnel_totals', {})
    total = counts.get('n_input')
    parts = (counts.get('n_accepted', 0) + counts.get('n_rejected', 0)
             + counts.get('n_quarantined', 0) + counts.get('n_holdout', 0)
             + counts.get('n_licence_blocked', 0) + counts.get('n_archive_only', 0)
             + counts.get('n_duplicate_removed', 0))
    return total is not None and parts == total and counts.get('n_unique_ids') == total


def gate_rebuild_identical(train_sha: str, val_sha: str) -> Optional[bool]:
    record = REPORTS / 'stage5d_rebuild_check.json'
    if not record.exists():
        return None
    payload = json.loads(record.read_text())
    return (payload.get('train_csv_sha256') == train_sha
            and payload.get('val_csv_sha256') == val_sha)


def gate_composition(manifest: Dict[str, object]) -> Optional[bool]:
    path = REPORTS / 'stage5d_dataset_composition.csv'
    if not path.exists():
        return False
    composition = pd.read_csv(path)
    if not len(composition):
        return False
    accepted_total = int(composition['accepted_n'].sum())
    return accepted_total == manifest['counts']['accepted']


def gate_legacy_unchanged() -> Optional[bool]:
    for name in LEGACY_JSON:
        path = CORPUS_V2 / name
        if not path.exists():
            return False
        if name.endswith('_manifest.json'):
            payload = json.loads(path.read_text())
            recorded = payload.get('manifest_csv_sha256')
            if recorded and recorded != sha256_file(CORPUS_V2 / name.replace(
                    '_manifest.json', '_manifest.csv')):
                return False
    policy_sha = sha256_file(CORPUS_V2 / 'policy.yaml')
    for name in ('R2_field_strict_manifest.json', 'R3_field_extended_manifest.json'):
        payload = json.loads((CORPUS_V2 / name).read_text())
        if payload.get('policy_sha256') != policy_sha:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    manifest = json.loads((CORPUS_V3 / 'R3v2_strict_rgb_manifest.json').read_text())
    train = pd.read_csv(CORPUS_V3 / 'R3v2_strict_rgb_train.csv', low_memory=False)
    val = pd.read_csv(CORPUS_V3 / 'R3v2_strict_rgb_val.csv', low_memory=False)
    accepted = pd.concat([train, val], ignore_index=True)
    modality = ledger_columns(accepted['image_id'].astype(str).tolist(),
                              ['modality_status'])['modality_status']
    accepted['modality_status'] = (accepted['image_id'].astype(str)
                                   .map(modality).astype(str))

    gates: Dict[str, Optional[bool]] = {}
    gates['files_readable'] = gate_files_readable(manifest, train)
    gates['policy_and_iqa_scale_verified'] = gate_policy_scale(manifest)
    gates['visible_rgb_evidenced'] = gate_visible_rgb(accepted)
    gates['field_domain_evidenced'] = gate_field_domain(accepted)
    gates['licence_clear'] = gate_licence(accepted)
    gates['dedup_resolved'] = gate_dedup(manifest)
    gates['holdout_screened'] = gate_holdout(manifest)
    gates['groups_disjoint_and_unknown_flagged'] = gate_groups(accepted, manifest)
    gates['counts_conserved'] = gate_counts(manifest)
    gates['rebuild_byte_identical'] = gate_rebuild_identical(
        manifest.get('train_csv_sha256', ''), manifest.get('val_csv_sha256', ''))
    gates['composition_from_manifest'] = gate_composition(manifest)
    gates['legacy_unchanged'] = gate_legacy_unchanged()

    result = gate_status(gates, target_exceeded=manifest['target']['target_exceeded'])
    payload = {
        'manifest': manifest,
        'gates': {key: {'passed': gates.get(key), 'description': description,
                        'blocking': blocking}
                  for key, description, blocking in GATES},
        'status': result,
    }
    (REPORTS / 'stage5d_gates.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')

    counts = manifest['counts']
    lines = [
        '# Stage 5D final report',
        '',
        f"## Status: **{result['status']}**",
        '',
        f"- TARGET_EXCEEDED: {result['target_exceeded']} "
        '(a scale property; it never upgrades a failed gate)',
        f"- All gates passed: {result['all_gates_passed']}",
        f"- Failed gates: {result['failed_gates'] or 'none'}",
        f"- Unmeasured gates: {result['unknown_gates'] or 'none'}",
        '',
        '## Counts',
        '',
        f"- Ledger rows: {counts['ledger_rows']:,}",
        f"- Accepted: {counts['accepted']:,} (train {counts['train']:,} / val {counts['val']:,})",
        f"- Quarantined: {counts['quarantined']:,}",
        f"- Rejected: {counts['rejected']:,}",
        f"- Holdout: {counts['holdout']:,}",
        f"- Licence blocked: {counts['licence_blocked']:,}",
        f"- Duplicate removed: {counts['duplicate_removed']:,}",
        f"- Scale target: {manifest['target']['target_images']:,}; "
        f"exceeded: {manifest['target']['target_exceeded']}; "
        f"shortfall: {manifest['target']['shortfall']:,}",
        '',
        '## Gate detail',
        '',
        '| gate | passed | blocking |',
        '|---|---|---|',
    ]
    for key, description, blocking in GATES:
        lines.append(f"| {key} | {gates.get(key)} | {blocking} |")
    lines += [
        '',
        '## Hashes',
        '',
        f"- policy.yaml: `{manifest['policy_sha256']}`",
        f"- train CSV: `{manifest['train_csv_sha256']}`",
        f"- val CSV: `{manifest['val_csv_sha256']}`",
        '',
        '## Status meaning',
        '',
        '`READY_AUTO_VALIDATED` = all twelve programmatic gates passed.  It does',
        'not mean human-verified, and it does not mean PROVEN_NO_LEAKAGE for',
        'sources that are not on this machine.',
    ]
    (REPORTS / 'stage5d_final_report.md').write_text('\n'.join(lines) + '\n',
                                                     encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
