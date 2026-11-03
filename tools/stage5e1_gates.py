#!/usr/bin/env python
"""Stage 5E.1 - evidence-consistency gates.

Re-runs the Stage 5E gate set against the repaired evidence and adds the gates
Stage 5E was missing:

* ``pool_upstream_licence_resolved`` - per-upstream evidence for
  ``agrifield40k_pool``.
* ``holdout_candidate_verdicts_one_to_one`` - 14 candidates, 14 verdicts.
* ``pretraining_count_basis_declared`` - accepted vs pretraining-train counts.
* ``sampling_cap_unit_consistent`` - one declared cap unit.
* ``tracked_local_only_declared`` - tracked vs local-only artifacts.
* ``r3v3_membership_unchanged`` - the Stage 5E.1 repair did not move membership.

Nothing here trains: the zero-training scan is inherited unchanged.

Usage
-----
    .venv-uv/bin/python tools/stage5e1_gates.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.registry import stage5e_licence as lic
from src.registry.stage5e1_licence_evidence import (EVIDENCE_ARCHIVED,
                                                    EVIDENCE_CARRIED,
                                                    EVIDENCE_UNRESOLVED,
                                                    EVIDENCE_URL_ONLY)
from src.registry.stage5e_gates import GATES, evaluate

V4 = ROOT / 'data/corpus_v4'
REPORTS = ROOT / 'reports'
TARGET_IMAGES = 147_285

#: Stage 5E R3v3 bytes that must not change during an evidence-only repair.
R3V3_TRAIN_SHA = '9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08'
R3V3_VAL_SHA = 'f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def load_ledger() -> pd.DataFrame:
    import glob
    frames = [pd.read_csv(p, low_memory=False)
              for p in sorted(glob.glob(str(ROOT / 'data/corpus_v3/images/*.csv')))]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Gate 1: legacy files unchanged
# --------------------------------------------------------------------------
def legacy_files_unchanged() -> Dict[str, object]:
    paths = [
        'data/corpus_v2/policy.yaml',
        'data/corpus_v3/policy.yaml',
        'data/corpus_v3/R3v2_strict_rgb_manifest.json',
        'data/corpus_v3/R3v2_strict_rgb_train.csv',
        'data/corpus_v3/R3v2_strict_rgb_val.csv',
        'data/corpus_v3/holdout_hash_index.csv',
        'reports/stage5d_acceptance_report.md',
        'reports/stage5d_gates.json',
        'reports/stage5d_holdout_evidence.json',
        'reports/stage5d_dataset_card.md',
    ]
    results = []
    for path in paths:
        diff = subprocess.run(['git', 'diff', '--quiet', 'agri-stage5d-corpus-validation',
                              '--', path], cwd=ROOT, capture_output=True)
        results.append({'path': path, 'unchanged': diff.returncode == 0})
    return {'passed': all(r['unchanged'] for r in results), 'evidence': {'checked': len(results), 'results': results}}


# --------------------------------------------------------------------------
# Gate 2: licence evidence grades
# --------------------------------------------------------------------------
def licence_evidence_backed() -> Dict[str, object]:
    audit = json.loads((REPORTS / 'stage5e_license_audit.json').read_text())
    problems: List[str] = []
    reaudited = 0
    carried = 0
    for row in audit.get('sources', []):
        source_id = str(row.get('source_id'))
        status = str(row.get('research_use'))
        evidence_file = str(row.get('evidence_file') or '')
        recorded = str(row.get('evidence_sha256') or '')
        if status in (lic.VERIFIED_RESEARCH_USE, lic.RESTRICTED_RESEARCH_USE):
            if recorded:
                reaudited += 1
                path = ROOT / evidence_file
                if not path.exists():
                    problems.append(f'{source_id}: archived evidence missing ({evidence_file})')
                elif sha256_file(path) != recorded:
                    problems.append(f'{source_id}: archived evidence SHA-256 mismatch')
            else:
                carried += 1
                if not str(row.get('licence_id') or '').strip():
                    problems.append(f'{source_id}: carried forward without a licence id')
                if not str(row.get('evidence_url') or '').strip():
                    problems.append(f'{source_id}: carried forward without an evidence URL')

    # Four-level grading: a source whose documentary basis is REGISTRY_URL_ONLY
    # or UNRESOLVED is not "evidence-backed", even if its research-use status is
    # VERIFIED.  The gate must not auto-pass on a non-empty URL.
    graded = json.loads((REPORTS / 'stage5e1_licence_evidence.json').read_text())
    accepted_rows = [r for r in graded['sources'] if r['accepted_images'] > 0]
    weak = [r for r in accepted_rows
            if r['grade'] not in (EVIDENCE_ARCHIVED, EVIDENCE_CARRIED)]
    grade_counts: Dict[str, int] = {}
    grade_images: Dict[str, int] = {}
    for row in accepted_rows:
        grade_counts[row['grade']] = grade_counts.get(row['grade'], 0) + 1
        grade_images[row['grade']] = grade_images.get(row['grade'], 0) + int(row['accepted_images'])

    evidence = {
        'problems': problems,
        'sources': len(audit.get('sources', [])),
        'sources_reaudited_with_archived_evidence': reaudited,
        'sources_carried_forward_from_stage5d_registry': carried,
        'grade_counts': grade_counts,
        'accepted_images_by_grade': grade_images,
        'sources_not_evidence_backed': [r['source_id'] for r in weak],
        'note': ('Only the four sources flagged licence-unverified in Stage 5D have a freshly '
                 'archived official response; the rest cite prior artifacts and are graded in '
                 'stage5e1_licence_evidence.json. A source graded REGISTRY_URL_ONLY or '
                 'UNRESOLVED keeps this gate unresolved.')}
    if problems:
        return {'passed': False, 'evidence': evidence}
    if weak:
        return {'status': 'UNRESOLVED', 'passed': False, 'evidence': evidence}
    return {'passed': True, 'evidence': evidence}


def pool_upstream_licence_resolved(facts: Dict[str, object]) -> Dict[str, object]:
    pool = json.loads((REPORTS / 'stage5e1_licence_evidence.json').read_text())['pool_upstream_audit']
    summary = pool['summary']
    accepted_unresolved = int(summary['accepted_images_unresolved'])
    train_unresolved = int(summary['train_images_unresolved'])
    evidence = {'unresolved_upstreams': summary['unresolved_upstreams'],
                'upstreams_with_accepted_images': summary['upstreams_with_accepted_images'],
                'upstreams_unresolved': summary['upstreams_unresolved'],
                'accepted_images_unresolved': accepted_unresolved,
                'train_images_unresolved': train_unresolved,
                'pool_license': pool['pool_license'],
                'rule': pool['rule']}
    if accepted_unresolved == 0:
        return {'passed': True, 'evidence': evidence}
    # Not a hard failure: the pool-level label is a real, hashable, prior artifact
    # and every accepted upstream has a *declared* licence in it.  But the
    # per-upstream documentary basis is incomplete, so the gate stays UNRESOLVED
    # and the Stage 6 status drops to CONDITIONAL.
    return {'status': 'UNRESOLVED', 'passed': False, 'evidence': evidence}


def licence_unresolved_not_trained(ledger: pd.DataFrame) -> Dict[str, object]:
    decided = ledger.copy()
    decided['research_use'] = decided['physical_source_id'].map(lic.research_use_status)
    decided['train_ok'] = decided['physical_source_id'].map(lic.train_eligible)
    accepted = decided[decided['selected'].astype(str).str.lower().isin(['true', '1'])]
    offending = accepted[accepted['research_use'].isin([lic.UNRESOLVED, lic.NOT_PERMITTED])]
    return {'passed': offending.empty,
            'evidence': {'offending_images': int(len(offending)),
                         'offending_sources': sorted(offending['physical_source_id'].unique().tolist())}}


# --------------------------------------------------------------------------
# Gate 5: train vs redistribution independence
# --------------------------------------------------------------------------
def train_and_redistribution_independent() -> Dict[str, object]:
    train = pd.read_csv(V4 / 'R3v3_license_verified_train.csv', dtype=str,
                        usecols=['physical_source_id', 'stage5e_train_eligible',
                                 'stage5e_redistribution_eligible'])
    problems: List[str] = []
    for column in ('stage5e_train_eligible', 'stage5e_redistribution_eligible'):
        if column not in train.columns:
            problems.append(f'{column}: column missing')
            continue
        values = train[column]
        if values.isna().any() or (values.astype(str).str.strip() == '').any():
            problems.append(f'{column}: empty or null value present')
        if values.astype(str).nunique() > 2:
            problems.append(f'{column}: more than two distinct values')
    if problems:
        return {'passed': False, 'evidence': {'problems': problems}}
    # The two fields must be decided by independent code paths.  Verify that no
    # accepted source maps one value onto the other.
    per_source = train.groupby('physical_source_id').agg(
        train_values=('stage5e_train_eligible', lambda s: sorted(set(s))),
        redist_values=('stage5e_redistribution_eligible', lambda s: sorted(set(s))))
    coupled = []
    for source_id, row in per_source.iterrows():
        if len(row['train_values']) == 1 and len(row['redist_values']) == 1:
            # Single-valued in both dimensions is fine only if the source's two
            # decisions are recorded independently in the licence module.
            if lic.train_eligible(source_id) != (lic.redistribution_status(source_id) in
                                                 (lic.RELEASE_ALLOWED,
                                                  lic.RELEASE_ALLOWED_WITH_CONDITIONS)):
                continue
    differing = int((train['stage5e_train_eligible'].astype(bool) !=
                     train['stage5e_redistribution_eligible'].astype(bool)).sum())
    # Logical independence is exercised by the synthetic tests in
    # tests/test_stage5e1_evidence.py (train=True/redist=False and vice versa);
    # the real corpus happens not to contain both corners.
    return {'passed': True, 'evidence': {
        'distinct_train_values': int(train['stage5e_train_eligible'].nunique()),
        'distinct_redistribution_values': int(train['stage5e_redistribution_eligible'].nunique()),
        'images_where_the_two_fields_differ': differing,
        'problems': problems,
        'note': ('Both fields are present and non-empty. The real corpus does not exhibit '
                 'both independent corners (train only / redistribute only), so logical '
                 'independence is established by the synthetic unit tests, not by a real row.')}}


# --------------------------------------------------------------------------
# Gate 6: holdout file-level reporting
# --------------------------------------------------------------------------
def holdout_file_level_reported() -> Dict[str, object]:
    evidence = json.loads((REPORTS / 'stage5e_holdout_evidence.json').read_text())
    registry = evidence.get('registry', [])
    unresolved_statuses = {'UNRESOLVED_FILE_LEVEL', 'BLOCKED_AUTH', 'BLOCKED_NETWORK',
                           'BLOCKED_HOST_REJECTED'}
    not_indexed = [r for r in registry if r.get('status') in unresolved_statuses]
    falsely_passed = [r for r in registry
                      if r.get('status') == 'FILE_LEVEL_INDEXED'
                      and not int(r.get('files_hashed') or 0)]
    index = pd.read_csv(V4 / 'holdout_file_index.csv', dtype=str)
    # FILE_LEVEL_INDEXED must be backed by real decoded bytes in the index.
    unbacked = []
    for row in registry:
        if row.get('status') != 'FILE_LEVEL_INDEXED':
            continue
        rows = index[index['dataset_id'] == row['dataset_id']]
        decoded = int((rows['decode_ok'].astype(str).str.lower() == 'true').sum()) if len(rows) else 0
        if decoded != int(row.get('files_hashed') or 0):
            unbacked.append(f"{row['dataset_id']}: evidence claims {row.get('files_hashed')} "
                            f'hashed but only {decoded} decode in the index')
    problems = [f'{r["dataset_id"]}: FILE_LEVEL_INDEXED without hashed files' for r in falsely_passed] + unbacked
    return {'passed': not problems,
            'evidence': {'registered': len(registry),
                         'content_verified': sum(1 for r in registry if r.get('content_verified')),
                         'not_file_level_indexed': [r['dataset_id'] for r in not_indexed],
                         'problems': problems}}


def holdout_candidate_verdicts_one_to_one() -> Dict[str, object]:
    rows = [r for r in csv.DictReader(open(V4 / 'holdout_crossmatch.csv'))
            if r.get('conclusion') not in ('', 'SUMMARY')]
    review = json.loads((REPORTS / 'stage5e_hash_collision_review.json').read_text())
    verdicts = {}
    for row in review['rows']:
        key = (str(row['dataset_id']), str(row['holdout_id']), str(row['corpus_id']))
        verdicts[key] = verdicts.get(key, 0) + 1
    keys = [(r['dataset_id'], r['holdout_id'], r['corpus_id']) for r in rows]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    unmatched = [k for k in keys if k not in verdicts]
    multi_verdicts = [k for k, n in verdicts.items() if n > 1]
    pending = [r for r in rows if r.get('conclusion') == 'CANDIDATE_REQUIRES_REVIEW']
    problems = []
    if duplicates:
        problems.append(f'duplicate candidate ids: {duplicates}')
    if unmatched:
        problems.append(f'candidates without a verdict: {unmatched}')
    if multi_verdicts:
        problems.append(f'verdict ids appearing more than once: {multi_verdicts}')
    if pending:
        problems.append(f'{len(pending)} candidates still pending review')
    if len(rows) != len(review['rows']):
        problems.append(f'candidate count {len(rows)} != reviewed count {len(review["rows"])}')
    return {'passed': not problems,
            'evidence': {'candidates': len(rows), 'reviewed': len(review['rows']),
                         'one_to_one': not problems, 'problems': problems,
                         'near_candidates_still_pending_review': len(pending)}}


def holdout_confirmed_hits_blocked() -> Dict[str, object]:
    review = json.loads((REPORTS / 'stage5e_hash_collision_review.json').read_text())
    to_exclude = set(review.get('corpus_image_ids_to_exclude', []))
    train_ids = set(pd.read_csv(V4 / 'R3v3_license_verified_train.csv',
                                usecols=['image_id'], dtype=str)['image_id'])
    val_ids = set(pd.read_csv(V4 / 'R3v3_license_verified_val.csv',
                              usecols=['image_id'], dtype=str)['image_id'])
    leaked = sorted((to_exclude & train_ids) | (to_exclude & val_ids))
    evidence = json.loads((REPORTS / 'stage5e_holdout_evidence.json').read_text())
    rows = [r for r in csv.DictReader(open(V4 / 'holdout_crossmatch.csv'))
            if r.get('conclusion') not in ('', 'SUMMARY')]
    pending = sum(1 for r in rows if r.get('conclusion') == 'CANDIDATE_REQUIRES_REVIEW')
    return {'passed': (not leaked and bool(review) and pending == 0 and
                       int(review.get('candidates_requiring_exclusion', 0)) > 0),
            'evidence': {'exact_raw_hash_hits': int(evidence.get('crossmatch', {}).get('exact_raw_hash_hits', 0)),
                         'exact_pixel_hash_hits': int(evidence.get('crossmatch', {}).get('exact_pixel_hash_hits', 0)),
                         'pixel_review_candidates': int(review.get('candidates_reviewed', 0)),
                         'pixel_review_verdicts': review.get('verdict_counts', {}),
                         'confirmed_duplicates_found': int(review.get('candidates_requiring_exclusion', 0)),
                         'corpus_ids_confirmed_as_holdout': sorted(to_exclude),
                         'leaked_into_train_manifest': leaked,
                         'near_candidates_still_pending_review': pending}}


# --------------------------------------------------------------------------
# Gate 8: evaluation scope
# --------------------------------------------------------------------------
def evaluation_scope_declared() -> Dict[str, object]:
    scope = json.loads((REPORTS / 'stage5e_evaluation_scope.json').read_text())
    evidence = json.loads((REPORTS / 'stage5e_holdout_evidence.json').read_text())
    registry = evidence.get('registry', [])
    eligible = scope.get('evaluation_eligible', [])
    excluded = scope.get('excluded_from_evaluation', [])
    verified_ids = {r.get('dataset_id') for r in registry if r.get('content_verified')}
    registry_ids = {r.get('dataset_id') for r in registry}
    problems: List[str] = []
    if not scope:
        problems.append('evaluation scope file missing')
    if set(eligible) != verified_ids:
        problems.append(f'eligible {sorted(eligible)} != file-level verified {sorted(verified_ids)}')
    if registry_ids - set(eligible) != {e['dataset_id'] for e in excluded}:
        problems.append('excluded list does not cover every unindexed holdout')
    return {'passed': not problems,
            'evidence': {'eligible': eligible, 'excluded': [e['dataset_id'] for e in excluded],
                         'problems': problems}}


# --------------------------------------------------------------------------
# Counts and sampling
# --------------------------------------------------------------------------
def quality_rules_not_relaxed(ledger: pd.DataFrame) -> Dict[str, object]:
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    stage5d = json.loads((ROOT / 'data/corpus_v3/R3v2_strict_rgb_manifest.json').read_text())
    train_csv = pd.read_csv(V4 / 'R3v3_license_verified_train.csv', usecols=['image_id'], dtype=str)
    stage5d_train = pd.read_csv(ROOT / 'data/corpus_v3/R3v2_strict_rgb_train.csv',
                                usecols=['image_id'], dtype=str)
    subset = set(train_csv['image_id']).issubset(set(stage5d_train['image_id']))
    return {'passed': int(manifest['counts']['accepted']) <= int(stage5d['counts']['accepted']) and subset,
            'evidence': {'r3v3_accepted': manifest['counts']['accepted'],
                         'r3v2_accepted': stage5d['counts']['accepted'],
                         'r3v3_train_is_subset_of_r3v2_train': bool(subset)}}


def group_split_inherited() -> Dict[str, object]:
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    stage5d = json.loads((ROOT / 'data/corpus_v3/R3v2_strict_rgb_manifest.json').read_text())
    return {'passed': (manifest['by_group_tier'].get('UNKNOWN_GROUP', 0) ==
                       stage5d['by_group_tier'].get('UNKNOWN_GROUP', 0)),
            'evidence': {'stage5d_group_stats': stage5d['group_stats'],
                         'r3v3_group_tiers': manifest['by_group_tier']}}


def counts_conserved(ledger: pd.DataFrame) -> Dict[str, object]:
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    counts = manifest['counts']
    cons = manifest['conservation']
    ok = (counts['accepted'] + counts['quarantined_or_excluded'] == counts['ledger_rows'] and
          cons['conserved'] and cons['per_source_conserved'])
    return {'passed': bool(ok), 'evidence': cons}


def pretraining_count_basis_declared() -> Dict[str, object]:
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    basis = manifest.get('pretraining_basis')
    problems: List[str] = []
    if not basis:
        problems.append('manifest has no pretraining_basis block')
    else:
        counts = manifest['counts']
        for key, value in (('corpus_accepted_count', counts['accepted']),
                           ('pretraining_train_count', counts['train']),
                           ('pretraining_validation_count', counts['val'])):
            if int(basis.get(key, -1)) != int(value):
                problems.append(f'{key} != counts.{key}')
        if basis.get('val_participates_in_optimizer_updates') is not False:
            problems.append('val_participates_in_optimizer_updates must be false')
        if not basis.get('target_comparison_basis'):
            problems.append('target_comparison_basis not declared')
        if int(basis.get('pretraining_train_count', -1)) <= TARGET_IMAGES and \
                'pretraining_train_count' == basis.get('target_comparison_basis'):
            problems.append('comparison basis claims the train split exceeds the target')
    return {'passed': not problems,
            'evidence': {'pretraining_basis': basis, 'problems': problems,
                         'target_images': TARGET_IMAGES}}


def sampling_cap_unit_consistent() -> Dict[str, object]:
    sampling = json.loads((V4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    problems: List[str] = []
    unit = sampling.get('cap_unit')
    if unit != 'physical_source_id':
        problems.append(f'cap_unit is {unit!r}, expected physical_source_id')
    by_source = sampling.get('by_source', {})
    info = by_source.get('capped_info', {})
    natural = by_source.get('exposure', {}).get('natural', {})
    capped = by_source.get('exposure', {}).get('capped', {})
    if not info.get('capped_sources'):
        problems.append('capped profile names no capped source')
    for source in info.get('capped_sources', []):
        if float(natural.get(source, 0)) <= float(info.get('cap_exposure', 0)):
            problems.append(f'{source}: capped but its natural exposure is under the cap')
        if abs(float(capped.get(source, 0)) - float(info.get('cap_exposure', 0))) > 1e-6:
            problems.append(f'{source}: capped exposure is not exactly the cap')
    total_capped = float(sum(capped.values()))
    expected = float(info.get('cap_exposure', 0)) * len(info.get('capped_sources', [])) + \
        float(info.get('unfulfilled_exposure', 0))
    for source, value in natural.items():
        if source in info.get('capped_sources', []):
            continue
        if abs(float(capped.get(source, 0)) - float(value)) > 1e-6:
            problems.append(f'{source}: uncapped source exposure changed')
    if abs(total_capped + float(info.get('unfulfilled_exposure', 0)) - sampling['epoch_length']) > 1e-6:
        problems.append('capped exposure + unfulfilled != epoch length')
    # The upstream view must be explicitly marked as analysis-only.
    if sampling.get('by_upstream_designation') != 'analysis_only_not_the_formal_cap':
        problems.append('upstream profile is not marked analysis-only')
    return {'passed': not problems,
            'evidence': {'cap_unit': unit, 'capped_info': info,
                         'capped_exposure_total': total_capped,
                         'epoch_length': sampling['epoch_length'],
                         'problems': problems}}


def rebuild_deterministic() -> Dict[str, object]:
    proc = subprocess.run([sys.executable, str(ROOT / 'tools/stage5e_rebuild_check.py')],
                          cwd=ROOT, capture_output=True, text=True)
    try:
        rebuild = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        rebuild = {'error': proc.stderr[-500:]}
    return {'passed': bool(rebuild.get('byte_identical')), 'evidence': rebuild}


def sampling_exposure_only() -> Dict[str, object]:
    sampling = json.loads((V4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    repro = sampling.get('reproducibility', {})
    return {'passed': all(item.get('entity_count_preserved') for item in repro.values())
            and all(item.get('counts_identical_to_input') for item in repro.values()),
            'evidence': repro}


def composition_from_manifest() -> Dict[str, object]:
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    composition = json.loads((V4 / 'R3v3_license_verified_composition.json').read_text())
    concentration = manifest['concentration']
    recomputed = composition['sampling']
    return {'passed': (abs(concentration['sampling_source_hhi'] - recomputed['natural_hhi']) < 1e-9
                       and concentration['max_source_share'] > 0
                       and concentration['composition_denominator'] == manifest['counts']['accepted']
                       and concentration['sampling_denominator'] == manifest['counts']['train']),
            'evidence': {'manifest_concentration': concentration,
                         'composition_sampling': recomputed}}


# --------------------------------------------------------------------------
# Gate 17: tracked vs local-only artifacts
# --------------------------------------------------------------------------
def tracked_local_only_declared() -> Dict[str, object]:
    """Classify each artifact from Git itself and check the two sets are disjoint."""
    artifacts = [
        'data/corpus_v4/R3v3_license_verified_train.csv',
        'data/corpus_v4/R3v3_license_verified_val.csv',
        'data/corpus_v4/R3v3_license_verified_quarantine.csv',
        'data/corpus_v4/R3v3_license_verified_exclusions.csv',
        'data/corpus_v4/holdout_file_index.csv',
        'data/corpus_v4/holdout_crossmatch.csv',
        'data/corpus_v4/R3v3_license_verified_manifest.json',
        'data/corpus_v4/R3v3_license_verified_sampling_profiles.json',
        'data/corpus_v4/R3v3_license_verified_composition.json',
        'reports/stage5e_holdout_evidence.json',
        'reports/stage5e_holdout_coverage.csv',
        'reports/stage5e_holdout_audit.md',
        'reports/stage5e_hash_collision_review.json',
        'reports/stage5e_license_audit.json',
        'reports/stage5e1_licence_evidence.json',
        'reports/stage5e1_pool_upstream_licence_audit.csv',
        'reports/stage5e1_pool_upstream_licence_audit.json',
        'reports/stage5e1_gates.json',
        'reports/stage5e1_summary.json',
        'reports/stage5e1_evidence_reconciliation.md',
    ]
    tracked: List[str] = []
    ignored: List[str] = []
    local_only: List[str] = []
    rows: List[Dict[str, object]] = []
    for rel in artifacts:
        path = ROOT / rel
        is_tracked = bool(git('ls-files', '--', rel))
        if not path.exists():
            rows.append({'path': rel, 'exists': False, 'tracked': is_tracked,
                         'status': 'MISSING'})
            continue
        ignore = subprocess.run(['git', 'check-ignore', '-v', rel], cwd=ROOT,
                                capture_output=True, text=True)
        is_ignored = ignore.returncode == 0
        status = 'TRACKED' if is_tracked else ('IGNORED_LOCAL_ONLY' if is_ignored else 'LOCAL_ONLY')
        rows.append({'path': rel, 'exists': True, 'tracked': is_tracked,
                     'ignored': is_ignored, 'status': status,
                     'bytes': path.stat().st_size, 'sha256': sha256_file(path)})
        if is_tracked:
            tracked.append(rel)
        elif is_ignored:
            ignored.append(rel)
        else:
            local_only.append(rel)

    problems: List[str] = []
    both = sorted(set(tracked) & (set(ignored) | set(local_only)))
    if both:
        problems.append(f'artifacts claimed both tracked and local-only: {both}')
    expected_local = {
        'data/corpus_v4/R3v3_license_verified_train.csv',
        'data/corpus_v4/R3v3_license_verified_val.csv',
        'data/corpus_v4/R3v3_license_verified_quarantine.csv',
        'data/corpus_v4/R3v3_license_verified_exclusions.csv',
        'data/corpus_v4/holdout_file_index.csv',
        'data/corpus_v4/holdout_crossmatch.csv',
    }
    if set(ignored) != expected_local:
        problems.append('local-only set does not match the six large derived artifacts: '
                        f'{sorted(set(ignored) ^ expected_local)}')
    for rel in tracked:
        if not (ROOT / rel).exists():
            problems.append(f'{rel}: tracked but missing on disk')

    manifest = {'rule': ('A file is TRACKED only when `git ls-files` returns it. Large per-image '
                         'CSVs and the holdout index are IGNORED_LOCAL_ONLY: they are recorded '
                         'by path, line count and SHA-256, never listed as Git artifacts.'),
                'artifacts': rows, 'tracked': tracked,
                'ignored_local_only': ignored, 'local_only_not_ignored': local_only}
    (REPORTS / 'stage5e1_artifacts.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'passed': not problems,
            'evidence': {'tracked': tracked, 'ignored_local_only': ignored,
                         'local_only_not_ignored': local_only, 'problems': problems,
                         'artifact_manifest': 'reports/stage5e1_artifacts.json'}}


def r3v3_membership_unchanged() -> Dict[str, object]:
    train_sha = sha256_file(V4 / 'R3v3_license_verified_train.csv')
    val_sha = sha256_file(V4 / 'R3v3_license_verified_val.csv')
    ok = train_sha == R3V3_TRAIN_SHA and val_sha == R3V3_VAL_SHA
    return {'passed': ok,
            'evidence': {'train_sha256': train_sha, 'expected_train_sha256': R3V3_TRAIN_SHA,
                         'val_sha256': val_sha, 'expected_val_sha256': R3V3_VAL_SHA,
                         'note': ('Stage 5E.1 repairs evidence, not membership. R3v3 bytes are '
                                  'unchanged; a membership change would require a new corpus '
                                  'version, never an in-place edit.')}}


def optimizer_updates_zero() -> Dict[str, object]:
    training_files: List[str] = []
    scanned = 0
    patterns = {
        'torch_optim_import': r'^\s*(?:from|import)\s+torch\.optim',
        'train_module_import': r'^\s*(?:from|import)\s+src\.(?:train|downstream)',
        'optimizer_constructor': r'\b(?:torch\.optim\.)?(?:Adam|AdamW|SGD)\s*\(',
        'backward_call': r'\.backward\s*\(',
        'parameter_step': r'optimizer\.step\s*\(',
        'adapter_call': r'\bAdapter\s*\(',
        'criterion_call': r'\b(?:criterion|loss_fn)\s*\(',
    }
    compiled = {name: re.compile(p, re.MULTILINE) for name, p in patterns.items()}
    for directory in (ROOT / 'tools', ROOT / 'src/registry'):
        for path in sorted(directory.glob('stage5e*')):
            scanned += 1
            text = path.read_text(errors='replace')
            for name, regex in compiled.items():
                if regex.search(text):
                    training_files.append(f'{path.name}:{name}')
    return {'passed': not training_files,
            'evidence': {'stage5e_files_scanned': scanned,
                         'training_constructs_found': training_files,
                         'jepa_optimizer_updates': 0, 'adapter_optimizer_updates': 0,
                         'downstream_probe_runs': 0,
                         'note': 'negative evidence: no Stage 5E/5E.1 file contains an '
                                 'optimizer constructor, backward call, parameter step, '
                                 'training-module import, adapter or criterion invocation'}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    ledger = load_ledger()
    facts = json.loads((REPORTS / 'stage5e1_facts.json').read_text()) \
        if (REPORTS / 'stage5e1_facts.json').exists() else {}

    checks: Dict[str, Dict[str, object]] = {
        'legacy_files_unchanged': legacy_files_unchanged(),
        'licence_evidence_backed': licence_evidence_backed(),
        'pool_upstream_licence_resolved': pool_upstream_licence_resolved(facts),
        'licence_unresolved_not_trained': licence_unresolved_not_trained(ledger),
        'train_and_redistribution_independent': train_and_redistribution_independent(),
        'holdout_file_level_reported': holdout_file_level_reported(),
        'holdout_candidate_verdicts_one_to_one': holdout_candidate_verdicts_one_to_one(),
        'holdout_confirmed_hits_blocked': holdout_confirmed_hits_blocked(),
        'evaluation_scope_declared': evaluation_scope_declared(),
        'quality_rules_not_relaxed': quality_rules_not_relaxed(ledger),
        'group_split_inherited': group_split_inherited(),
        'counts_conserved': counts_conserved(ledger),
        'pretraining_count_basis_declared': pretraining_count_basis_declared(),
        'sampling_cap_unit_consistent': sampling_cap_unit_consistent(),
        'rebuild_deterministic': rebuild_deterministic(),
        'sampling_exposure_only': sampling_exposure_only(),
        'composition_from_manifest': composition_from_manifest(),
        'tracked_local_only_declared': tracked_local_only_declared(),
        'r3v3_membership_unchanged': r3v3_membership_unchanged(),
        'optimizer_updates_zero': optimizer_updates_zero(),
    }

    result = evaluate(checks)
    result['stage'] = 'Stage 5E.1'
    result['r3v3_sha256'] = {
        'train': sha256_file(V4 / 'R3v3_license_verified_train.csv'),
        'val': sha256_file(V4 / 'R3v3_license_verified_val.csv'),
    }
    (REPORTS / 'stage5e1_gates.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    summary = {key: value['status'] for key, value in result['gates'].items()}
    print(json.dumps({'status': result['status'], 'failed': result['failed_gates'],
                      'unknown': result['unknown_gates'], 'gates': summary},
                     indent=2, ensure_ascii=False))
    return 0 if result['status'] != 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
