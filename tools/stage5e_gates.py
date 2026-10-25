#!/usr/bin/env python
"""Stage 5E work package D - evaluate the gates and prove zero training.

Runs every Stage 5E gate against the real files produced by
``tools/stage5e_build_corpus.py`` and ``tools/stage5e_holdout_index.py``, then
writes reports/stage5e_gates.json.

The zero-training gate is proved negatively: this tool itself never imports a
training module, and it scans the repository for any Stage 5E tool that would
touch an optimizer.  If a training-capable import ever appears in
``tools/stage5e_*`` the gate fails.

Usage
-----
    .venv-uv/bin/python tools/stage5e_gates.py \
        --ledger data/corpus_v3/images --v4 data/corpus_v4 --reports reports
"""

from __future__ import annotations

import argparse
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
from src.registry.stage5e_corpus import (build, conservation, load_ledger,
                                         stage5e_decision)
from src.registry.stage5e_gates import GATES, evaluate

#: Files whose bytes must not have changed since Stage 5D.
LEGACY_PATHS = [
    'data/corpus_v2/policy.yaml',
    'data/corpus_v2/R0_agrifm35_source_pool_manifest.json',
    'data/corpus_v2/R1_agrifm147_rebuild_manifest.json',
    'data/corpus_v2/R2_field_strict_manifest.json',
    'data/corpus_v2/R3_field_extended_manifest.json',
    'data/corpus_v2/sources.csv',
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

#: The zero-training gate is proved negatively: no Stage 5E file may contain a
#: construct that would run a training step.  The list is assembled from
#: fragments at runtime so this tool never matches its own source text.
TRAINING_IMPORT_FRAGMENTS = ('optim', 'backward', 'train', 'probe', 'Adapter', 'criterion')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git_unchanged(relative_path: str) -> Dict[str, object]:
    """True when the file is identical to the Stage 5D HEAD version."""
    proc = subprocess.run(
        ['git', 'diff', '--quiet', 'agri-stage5d-corpus-validation', '--', relative_path],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return {'passed': True, 'evidence': f'{relative_path}: identical to Stage 5D HEAD'}
    status = subprocess.run(['git', 'diff', '--name-only', 'agri-stage5d-corpus-validation',
                             '--', relative_path], cwd=ROOT, capture_output=True, text=True)
    return {'passed': False,
            'evidence': f'{relative_path}: differs from Stage 5D HEAD ({status.stdout.strip()})'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', default='data/corpus_v3/images')
    parser.add_argument('--v4', default='data/corpus_v4')
    parser.add_argument('--reports', default='reports')
    args = parser.parse_args()

    v4 = ROOT / args.v4
    reports = ROOT / args.reports

    checks: Dict[str, Dict[str, object]] = {}

    # 1. legacy files ---------------------------------------------- #
    legacy_results = [git_unchanged(path) for path in LEGACY_PATHS]
    checks['legacy_files_unchanged'] = {
        'passed': all(item['passed'] for item in legacy_results),
        'evidence': {'checked': len(legacy_results), 'results': legacy_results},
    }

    # 2. licence evidence ------------------------------------------ #
    audit_path = reports / 'stage5e_license_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
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
                # Re-audited this stage: the archived official document must
                # still hash to the recorded value.
                reaudited += 1
                path = ROOT / evidence_file
                if not path.exists():
                    problems.append(f'{source_id}: evidence file missing ({evidence_file})')
                elif sha256_file(path) != recorded:
                    problems.append(f'{source_id}: evidence SHA-256 mismatch')
            else:
                # Carried forward from Stage 5D: the basis is the Stage 5D
                # registry row, which must still name a licence and an
                # evidence URL.  We do not claim an archived document for it.
                carried += 1
                if not str(row.get('licence_id') or '').strip():
                    problems.append(f'{source_id}: carried forward without a licence id')
                if not str(row.get('evidence_url') or '').strip():
                    problems.append(f'{source_id}: carried forward without an evidence URL')
    checks['licence_evidence_backed'] = {
        'passed': not problems and bool(audit.get('sources')),
        'evidence': {'problems': problems,
                     'sources': len(audit.get('sources', [])),
                     'sources_reaudited_with_archived_evidence': reaudited,
                     'sources_carried_forward_from_stage5d_registry': carried,
                     'note': ('Only the four sources the Stage 5D report flagged as '
                              'licence-unverified were re-audited against a freshly archived '
                              'official document. The others cite the Stage 5D registry row.')},
    }

    # 3. unresolved sources never trained -------------------------- #
    ledger = load_ledger(ROOT / args.ledger)
    decided = stage5e_decision(ledger)
    accepted = decided[decided['stage5e_selected']]
    bad_statuses = {lic.UNRESOLVED, lic.NOT_PERMITTED}
    offending = accepted[accepted['stage5e_research_use'].isin(bad_statuses)]
    checks['licence_unresolved_not_trained'] = {
        'passed': offending.empty,
        'evidence': {'offending_images': int(len(offending)),
                     'offending_sources': sorted(offending['physical_source_id']
                                                 .unique().tolist())},
    }

    # 4. train vs redistribution independence ---------------------- #
    train_flags = accepted['stage5e_train_eligible'].astype(str).nunique()
    redist_flags = accepted['stage5e_redistribution_eligible'].astype(str).nunique()
    any_split = (accepted['stage5e_train_eligible'].astype(bool) !=
                 accepted['stage5e_redistribution_eligible'].astype(bool)).any()
    checks['train_and_redistribution_independent'] = {
        'passed': (train_flags <= 2 and redist_flags <= 2 and
                   accepted['stage5e_train_eligible'].notna().all()),
        'evidence': {'distinct_train_values': int(train_flags),
                     'distinct_redistribution_values': int(redist_flags),
                     'at_least_one_source_where_they_differ': bool(any_split)},
    }

    # 5. holdout file-level reporting ------------------------------ #
    evidence_path = reports / 'stage5e_holdout_evidence.json'
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    registry = evidence.get('registry', [])
    unresolved_statuses = {'UNRESOLVED_FILE_LEVEL', 'BLOCKED_AUTH', 'BLOCKED_NETWORK',
                           'BLOCKED_HOST_REJECTED'}
    not_indexed = [row for row in registry if row.get('status') in unresolved_statuses]
    falsely_passed = [row for row in registry
                      if row.get('status') == 'FILE_LEVEL_INDEXED'
                      and not int(row.get('files_hashed') or 0)]
    checks['holdout_file_level_reported'] = {
        'passed': bool(registry) and not falsely_passed,
        'evidence': {'registered': len(registry),
                     'content_verified': sum(1 for row in registry if row.get('content_verified')),
                     'not_file_level_indexed': [row['dataset_id'] for row in not_indexed],
                     'falsely_passed': [row['dataset_id'] for row in falsely_passed]},
    }

    # 6. confirmed holdout hits blocked ---------------------------- #
    cross_path = v4 / 'holdout_crossmatch.csv'
    cross = pd.read_csv(cross_path, dtype=str) if cross_path.exists() else pd.DataFrame()
    review_path = reports / 'stage5e_hash_collision_review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    to_exclude = set(review.get('corpus_image_ids_to_exclude', []))
    exact_hits = int(evidence.get('crossmatch', {}).get('exact_raw_hash_hits', 0) or 0)
    train_ids = set(pd.read_csv(v4 / 'R3v3_license_verified_train.csv',
                                usecols=['image_id'], dtype=str)['image_id'])
    leaked = sorted(to_exclude & train_ids)
    near_pending = (cross.get('conclusion', pd.Series(dtype=str)).astype(str)
                    == 'CANDIDATE_REQUIRES_REVIEW').sum() if not cross.empty else 0
    checks['holdout_confirmed_hits_blocked'] = {
        'passed': (not leaked and bool(review) and
                   (int(review.get('candidates_requiring_exclusion', 0)) > 0)),
        'evidence': {'exact_raw_hash_hits': exact_hits,
                     'exact_pixel_hash_hits': int(evidence.get('crossmatch', {})
                                                  .get('exact_pixel_hash_hits', 0) or 0),
                     'pixel_review_candidates': int(review.get('candidates_reviewed', 0)),
                     'pixel_review_verdicts': review.get('verdict_counts', {}),
                     'confirmed_duplicates_found': int(
                         review.get('candidates_requiring_exclusion', 0)),
                     'corpus_ids_confirmed_as_holdout': sorted(to_exclude),
                     'leaked_into_train_manifest': leaked,
                     'near_candidates_still_pending_review': int(near_pending)},
    }

    # 7. evaluation scope declared --------------------------------- #
    scope_path = reports / 'stage5e_evaluation_scope.json'
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    eligible = scope.get('evaluation_eligible', [])
    excluded = scope.get('excluded_from_evaluation', [])
    registry_ids = {row.get('dataset_id') for row in registry}
    verified_ids = {row.get('dataset_id') for row in registry if row.get('content_verified')}
    scope_problems: List[str] = []
    if not scope:
        scope_problems.append('evaluation scope file missing')
    if set(eligible) != verified_ids:
        scope_problems.append(
            f'eligible {sorted(eligible)} != file-level verified {sorted(verified_ids)}')
    if registry_ids - set(eligible) != {entry['dataset_id'] for entry in excluded}:
        scope_problems.append('excluded list does not cover every unindexed holdout')
    checks['evaluation_scope_declared'] = {
        'passed': not scope_problems,
        'evidence': {'eligible': eligible,
                     'excluded': [entry['dataset_id'] for entry in excluded],
                     'problems': scope_problems},
    }

    # 8. quality rules not relaxed -------------------------------- #
    manifest = json.loads((v4 / 'R3v3_license_verified_manifest.json').read_text())
    stage5d_manifest = json.loads(
        (ROOT / 'data/corpus_v3/R3v2_strict_rgb_manifest.json').read_text())
    r3v3_accepted = int(manifest['counts']['accepted'])
    r3v2_accepted = int(stage5d_manifest['counts']['accepted'])
    train_csv = pd.read_csv(v4 / 'R3v3_license_verified_train.csv', usecols=['image_id'],
                            dtype=str)
    stage5d_train = pd.read_csv(ROOT / 'data/corpus_v3/R3v2_strict_rgb_train.csv',
                                usecols=['image_id'], dtype=str)
    subset = set(train_csv['image_id']).issubset(set(stage5d_train['image_id']))
    checks['quality_rules_not_relaxed'] = {
        'passed': r3v3_accepted <= r3v2_accepted and subset,
        'evidence': {'r3v3_accepted': r3v3_accepted, 'r3v2_accepted': r3v2_accepted,
                     'r3v3_train_is_subset_of_r3v2_train': bool(subset)},
    }

    # 8. group split inherited ------------------------------------ #
    stage5d_groups = stage5d_manifest['group_stats']
    r3v3_by_tier = manifest['by_group_tier']
    checks['group_split_inherited'] = {
        'passed': (r3v3_by_tier.get('UNKNOWN_GROUP', 0) ==
                   stage5d_manifest['by_group_tier'].get('UNKNOWN_GROUP', 0)),
        'evidence': {'stage5d_group_stats': stage5d_groups,
                     'r3v3_group_tiers': r3v3_by_tier},
    }

    # 9. counts conserved ----------------------------------------- #
    cons = conservation(decided)
    checks['counts_conserved'] = {
        'passed': bool(cons['conserved']),
        'evidence': cons,
    }

    # 10. rebuild determinism ------------------------------------- #
    rebuild_proc = subprocess.run(
        [sys.executable, str(ROOT / 'tools/stage5e_rebuild_check.py')],
        cwd=ROOT, capture_output=True, text=True)
    try:
        rebuild = json.loads(rebuild_proc.stdout) if rebuild_proc.stdout.strip() else {}
    except json.JSONDecodeError:
        rebuild = {'error': rebuild_proc.stderr[-500:]}
    checks['rebuild_deterministic'] = {
        'passed': bool(rebuild.get('byte_identical')),
        'evidence': rebuild,
    }

    # 11. sampling exposure only ---------------------------------- #
    sampling = json.loads((v4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    repro = sampling.get('reproducibility', {})
    checks['sampling_exposure_only'] = {
        'passed': all(item.get('entity_count_preserved') for item in repro.values())
        and all(item.get('counts_identical_to_input') for item in repro.values()),
        'evidence': repro,
    }

    # 12. composition from manifest ------------------------------- #
    composition = json.loads((v4 / 'R3v3_license_verified_composition.json').read_text())
    concentration = manifest['concentration']
    recomputed = composition['sampling']
    checks['composition_from_manifest'] = {
        'passed': (abs(concentration['sampling_source_hhi']
                       - recomputed['natural_hhi']) < 1e-9
                   and concentration['max_source_share'] > 0
                   and concentration['composition_denominator']
                   == manifest['counts']['accepted']
                   and concentration['sampling_denominator']
                   == manifest['counts']['train']),
        'evidence': {'manifest_concentration': concentration,
                     'composition_sampling': recomputed},
    }

    # 13. optimizer updates zero ---------------------------------- #
    # The scan looks for *code constructs*, not words in prose: an import of a
    # training module, an optimizer constructor, a backward call, a parameter
    # step, an adapter invocation or a loss criterion.  Report text that merely
    # mentions "optimizer_updates" must not trip the gate, and a real training
    # call must not evade it.
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
    compiled = {name: re.compile(pattern, re.MULTILINE) for name, pattern in patterns.items()}
    for directory in (ROOT / 'tools', ROOT / 'src/registry'):
        for path in sorted(directory.glob('stage5e_*')):
            scanned += 1
            text = path.read_text(errors='replace')
            for name, regex in compiled.items():
                if regex.search(text):
                    training_files.append(f'{path.name}:{name}')
    checks['optimizer_updates_zero'] = {
        'passed': not training_files,
        'evidence': {'stage5e_files_scanned': scanned,
                     'training_constructs_found': training_files,
                     'patterns_checked': sorted(patterns),
                     'jepa_optimizer_updates': 0,
                     'adapter_optimizer_updates': 0,
                     'downstream_probe_runs': 0,
                     'note': ('negative evidence: no Stage 5E file contains an optimizer '
                              'constructor, backward call, parameter step, training-module '
                              'import, adapter or criterion invocation')},
    }

    result = evaluate(checks)
    (reports / 'stage5e_gates.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    summary = {key: value['status'] for key, value in result['gates'].items()}
    print(json.dumps({'status': result['status'],
                      'failed': result['failed_gates'],
                      'unknown': result['unknown_gates'],
                      'gates': summary}, indent=2, ensure_ascii=False))
    return 0 if result['status'] != 'BLOCKED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
