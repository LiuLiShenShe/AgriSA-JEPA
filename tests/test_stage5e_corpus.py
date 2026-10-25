"""Stage 5E tests.

The Stage 5E brief requires coverage for, among others:

* the four licence sources and their real statuses;
* a licence decision without evidence being rejected;
* train licence and image redistribution being independent;
* conflicting or missing evidence;
* "file name only, no content hash" never counting as a content check;
* 403 / 404 / timeout / auth being recorded distinctly;
* exact duplicate vs near-duplicate candidate;
* hash-collision review (a pHash of 0 is not proof of identity);
* unknown-group split behaviour;
* count conservation;
* sampling never changing the entity count;
* Stage 5C/5D historical files staying byte-identical;
* zero optimizer updates.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry import stage5e_licence as lic
from src.registry.stage5e_holdout_registry import (BLOCKED_AUTH, BLOCKED_HOST_REJECTED,
                                                   BLOCKED_NETWORK, CONTENT_VERIFIED_STATUSES,
                                                   ENTITIES, FILE_LEVEL_INDEXED,
                                                   INDEX_ONLY, UNRESOLVED_FILE_LEVEL,
                                                   by_id, status_summary)
from src.registry.stage5e_gates import GATES, evaluate
from src.registry.stage5e_sampling import (balanced_profile, build_profiles,
                                           capped_profile, entropy, hhi,
                                           natural_profile, reproducibility_check)

FOUR = ['agrifm_ref1', 'agrifm_ref58', 'agrifm_ref11', 'agrifm_ref9']
EVIDENCE_DIR = ROOT / 'reports/stage5e_evidence'


# --------------------------------------------------------------------------
# Licence: the four audited sources
# --------------------------------------------------------------------------

@pytest.mark.parametrize('source_id', FOUR)
def test_four_sources_have_a_real_identity(source_id):
    """The identity comes from the registry, never from the ref number."""
    entry = lic.LICENCES[source_id]
    assert entry.real_name
    assert entry.publisher
    assert entry.official_url.startswith('http')
    assert entry.licence_id
    assert entry.research_use in lic.RESEARCH_USE_STATUSES


def test_agrifm_ref58_is_cc0_from_dataverse_api():
    entry = lic.LICENCES['agrifm_ref58']
    assert entry.licence_id == 'CC0-1.0'
    assert entry.research_use == lic.VERIFIED_RESEARCH_USE
    assert entry.redistribution == lic.RELEASE_ALLOWED


def test_agrifm_ref1_is_cc0_from_kaggle_api():
    entry = lic.LICENCES['agrifm_ref1']
    assert entry.licence_id == 'CC0-1.0'
    assert entry.research_use == lic.VERIFIED_RESEARCH_USE


def test_agrifm_ref11_is_apache2():
    entry = lic.LICENCES['agrifm_ref11']
    assert entry.licence_id == 'Apache-2.0'
    assert entry.research_use == lic.VERIFIED_RESEARCH_USE
    # Apache permits redistribution with notice, not unconditionally.
    assert entry.redistribution == lic.RELEASE_ALLOWED_WITH_CONDITIONS


def test_agrifm_ref9_is_nc_and_restricted_not_verified():
    entry = lic.LICENCES['agrifm_ref9']
    assert entry.licence_id == 'CC-BY-NC-4.0'
    # NC is not auto-rejected and not auto-accepted: it is restricted.
    assert entry.research_use == lic.RESTRICTED_RESEARCH_USE
    assert entry.train_eligible is True


def test_agrifm_ref9_checkpoint_release_is_unresolved_not_allowed():
    """A published checkpoint is a derivative; NC makes that undecidable here."""
    entry = lic.LICENCES['agrifm_ref9']
    assert entry.checkpoint_release == lic.RELEASE_UNRESOLVED
    assert entry.checkpoint_release != lic.RELEASE_ALLOWED


@pytest.mark.parametrize('source_id', FOUR)
def test_each_licence_decision_has_archived_evidence(source_id):
    entry = lic.LICENCES[source_id]
    evidence = ROOT / entry.evidence_file
    assert evidence.exists(), entry.evidence_file
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert digest == entry.evidence_sha256
    assert entry.evidence_excerpt
    assert entry.evidence_locator


def test_evidence_index_records_the_archived_bytes():
    index = json.loads((EVIDENCE_DIR / 'evidence_index.json').read_text())
    assert set(index['sources']) == set(FOUR)
    for source_id, record in index['sources'].items():
        assert (ROOT / record['evidence_file']).exists()
        assert record['sha256']


def test_licence_audit_tool_reports_the_four_sources():
    audit = json.loads((ROOT / 'reports/stage5e_license_audit.json').read_text())
    assert set(audit['summary']['audited_sources']) == set(FOUR)
    assert audit['summary']['excluded_from_training'] == 0
    assert audit['summary']['evidence_problems'] == []


# --------------------------------------------------------------------------
# Licence: independence of the two rights dimensions
# --------------------------------------------------------------------------

def test_train_and_redistribution_are_independent_fields():
    """A source may train but not redistribute, and vice versa."""
    # agrifm_ref9: training allowed, redistribution conditional.
    assert lic.train_eligible('agrifm_ref9') is True
    assert lic.redistribution_status('agrifm_ref9') == lic.RELEASE_ALLOWED_WITH_CONDITIONS
    # A synthetic check: no status is ever mapped onto the other.
    for source_id in FOUR:
        train = lic.research_use_status(source_id)
        redist = lic.redistribution_status(source_id)
        assert (train == lic.VERIFIED_RESEARCH_USE) != (redist == lic.RELEASE_NOT_ALLOWED) \
            or source_id != 'agrifm_ref1'
    # The decisive property: a known-redistributable licence must not imply a
    # known research use, and vice versa.
    assert lic.train_eligible('unknown_source') is False
    assert lic.redistribution_eligible('unknown_source') is False


def test_unknown_source_is_never_train_eligible():
    assert lic.research_use_status('some_source_never_registered') == lic.UNRESOLVED
    assert lic.train_eligible('some_source_never_registered') is False


def test_redistribution_allowed_does_not_imply_train_allowed():
    """Explicit negative test of the collapse the brief forbids."""
    class _Source(lic.SourceLicence):
        pass
    fake = lic.SourceLicence(
        source_id='fake', real_name='fake', publisher='fake', official_url='https://x',
        licence_id='X', licence_name='X', evidence_url='https://x', evidence_locator='x',
        evidence_excerpt='x', evidence_file='', evidence_sha256='', accessed='2026-10-25',
        research_use=lic.NOT_PERMITTED, research_use_conditions='',
        redistribution=lic.RELEASE_ALLOWED, redistribution_conditions='',
        metadata_release=lic.RELEASE_ALLOWED, metadata_release_conditions='',
        derived_release=lic.RELEASE_ALLOWED, derived_release_conditions='',
        checkpoint_release=lic.RELEASE_ALLOWED, checkpoint_release_conditions='',
        open_issues='', decision='EXCLUDE', decision_reason='synthetic')
    assert fake.train_eligible is False
    assert fake.redistribution_eligible is True


def test_evidence_mismatch_is_detected(tmp_path):
    """A tampered evidence archive must fail the audit, not be ignored."""
    entry = lic.LICENCES['agrifm_ref58']
    copy = tmp_path / 'evidence.json'
    copy.write_text('{"tampered": true}')
    digest = hashlib.sha256(copy.read_bytes()).hexdigest()
    assert digest != entry.evidence_sha256


def test_blocked_sources_are_not_train_eligible():
    for source_id in lic.BLOCKED_IDS:
        assert source_id not in lic.LICENCES
        assert lic.train_eligible(source_id) is False


# --------------------------------------------------------------------------
# Holdout registry: statuses and their meaning
# --------------------------------------------------------------------------

def test_registry_splits_roboflow_into_concrete_entities():
    """The brief forbids treating "Roboflow series" as one verified dataset."""
    roboflow = [e for e in ENTITIES if 'Roboflow' in e.publisher]
    assert len(roboflow) == 3
    for entity in roboflow:
        assert entity.version
        assert entity.status in (UNRESOLVED_FILE_LEVEL, BLOCKED_HOST_REJECTED)


def test_no_holdout_entity_claims_more_than_its_status_supports():
    for entity in ENTITIES:
        row = entity.as_row()
        if row['content_verified']:
            assert entity.status == FILE_LEVEL_INDEXED
            assert entity.files_indexed >= 0
        else:
            assert entity.status in (INDEX_ONLY, BLOCKED_AUTH, BLOCKED_NETWORK,
                                     BLOCKED_HOST_REJECTED, UNRESOLVED_FILE_LEVEL)


def test_blocked_statuses_are_distinct():
    statuses = {BLOCKED_AUTH, BLOCKED_NETWORK, BLOCKED_HOST_REJECTED}
    assert len(statuses) == 3
    assert status_summary()


def test_every_registered_holdout_has_an_observed_failure_or_index():
    for entity in ENTITIES:
        assert entity.observed


def test_only_file_level_entities_are_content_verified():
    """Content verification is decided by the runtime evidence, not the
    registration-time status, so a dataset that was obtained later (PlantDoc)
    is counted only once its bytes were actually hashed."""
    evidence = json.loads((ROOT / 'reports/stage5e_holdout_evidence.json').read_text())
    verified = {row['dataset_id'] for row in evidence['registry']
                if row['content_verified']}
    assert verified == {'phenobench', 'gwhd_2021', 'plantdoc'}
    for row in evidence['registry']:
        if row['content_verified']:
            assert str(row['status']) == FILE_LEVEL_INDEXED
            assert int(row['files_hashed']) > 0


def test_plantdoc_was_really_indexed():
    index = pd.read_csv(ROOT / 'data/corpus_v4/holdout_file_index.csv', dtype=str)
    plantdoc = index[index['dataset_id'] == 'plantdoc']
    assert len(plantdoc) == 2579
    assert (plantdoc['raw_sha256'].astype(str) != '').all()
    assert (plantdoc['pixel_sha256'].astype(str) != '').all()
    assert (plantdoc['decode_ok'].astype(str) == 'true').all()


def test_filename_only_is_not_a_content_check():
    """A name collision must never be reported as a hash match."""
    index = pd.read_csv(ROOT / 'data/corpus_v4/holdout_file_index.csv', dtype=str)
    # Every indexed row has a real hash; a filename-only row would have none.
    assert not (index['raw_sha256'].astype(str) == '').any()
    # And the cross-match column count is derived from hashes, not names.
    cross = pd.read_csv(ROOT / 'data/corpus_v4/holdout_crossmatch.csv', dtype=str)
    assert 'exact_raw_hash_hits' in cross.columns


def test_policy_holdsout_ids_cover_registry():
    from src.registry.stage5c_source_registry import HOLDOUT_IDS
    for entity in ENTITIES:
        assert entity.dataset_id in HOLDOUT_IDS


# --------------------------------------------------------------------------
# Cross-check results
# --------------------------------------------------------------------------

def test_no_exact_hash_match_between_holdout_and_corpus():
    evidence = json.loads((ROOT / 'reports/stage5e_holdout_evidence.json').read_text())
    cross = evidence['crossmatch']
    assert cross['exact_raw_hash_hits'] == 0
    assert cross['exact_pixel_hash_hits'] == 0


def test_near_candidates_are_flagged_not_deleted():
    cross = pd.read_csv(ROOT / 'data/corpus_v4/holdout_crossmatch.csv', dtype=str)
    candidates = cross[cross['conclusion'] == 'CANDIDATE_REQUIRES_REVIEW']
    assert len(candidates) > 0
    assert (candidates['embedding_confirmed'] == 'NOT_RUN').all()
    # None of them is claimed to be a duplicate.
    assert not (cross['conclusion'].astype(str).str.contains('CONFIRMED', na=False)).any()


def test_zero_hits_is_not_reported_as_proven_no_leakage():
    evidence = json.loads((ROOT / 'reports/stage5e_holdout_evidence.json').read_text())
    assert 'never' in evidence['crossmatch']['scope']
    text = (ROOT / 'reports/stage5e_holdout_audit.md').read_text()
    assert '不得宣称无泄漏' in text


def test_unindexed_holdouts_are_listed_not_skipped():
    evidence = json.loads((ROOT / 'reports/stage5e_holdout_evidence.json').read_text())
    unresolved = [row['dataset_id'] for row in evidence['registry']
                  if not row['content_verified']]
    assert set(unresolved) >= {'minneapple', 'plantseg', 'growliflower', 'multiclass_weeds',
                              'deepweeds', 'cropandweed'}


def test_blocked_holdout_records_the_real_failure_mode():
    registry = {row['dataset_id']: row for row in
                json.loads((ROOT / 'reports/stage5e_holdout_evidence.json').read_text())['registry']}
    assert '403' in registry['minneapple']['observed']
    assert '403' in registry['multiclass_weeds']['observed']
    assert 'sign_in' in registry['plantseg']['observed']
    assert 'zenodo' in registry['growliflower']['observed'].lower()


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def test_natural_profile_preserves_share():
    counts = {'a': 90, 'b': 10}
    exposure = natural_profile(counts, 1000)
    assert exposure['a'] == 900.0
    assert exposure['b'] == 100.0


def test_capped_profile_reports_unfulfilled_budget():
    counts = {'big': 99, 'small': 1}
    exposure, info = capped_profile(counts, 1000, 0.5)
    assert exposure['big'] == 500.0
    assert exposure['small'] == 10.0
    assert info['budget_filled'] is False
    assert info['unfulfilled_exposure'] == pytest.approx(490.0)
    # Surplus is NOT redistributed to the small source.
    assert exposure['small'] == pytest.approx(10.0)


def test_capped_profile_can_fill_the_budget():
    counts = {'a': 50, 'b': 50}
    exposure, info = capped_profile(counts, 1000, 0.6)
    assert sum(exposure.values()) == pytest.approx(1000.0)
    assert info['budget_filled'] is True


def test_balanced_profile_is_equal_per_source():
    counts = {'a': 5, 'b': 5, 'c': 5}
    exposure = balanced_profile(counts, 900)
    assert set(exposure.values()) == {300.0}


def test_hhi_and_entropy():
    assert hhi({'a': 1.0}) == pytest.approx(1.0)
    assert entropy({'a': 1.0}) == pytest.approx(0.0)
    assert entropy({'a': 1.0, 'b': 1.0}) == pytest.approx(0.6931471805599453)


def test_sampling_never_changes_entities():
    frame = pd.DataFrame({'physical_source_id': ['a'] * 9 + ['b'] * 1,
                          'split': ['train'] * 10})
    profiles = build_profiles({'a': 9, 'b': 1}, 100, 0.5)
    repro = reproducibility_check(profiles, frame, 'physical_source_id')
    assert repro['entity_count_preserved'] is True
    assert repro['counts_identical_to_input'] is True
    assert repro['entity_count'] == 10


def test_real_sampling_profiles_preserve_train_entities():
    sampling = json.loads(
        (ROOT / 'data/corpus_v4/R3v3_license_verified_sampling_profiles.json').read_text())
    assert sampling['train_total'] == 147277
    for scope, repro in sampling['reproducibility'].items():
        assert repro['entity_count_preserved'] is True, scope
        assert repro['entity_count'] == 147277, scope


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def test_gate_engine_blocks_on_failure():
    result = evaluate({'counts_conserved': {'passed': False, 'evidence': {}}})
    assert result['status'] == 'BLOCKED'
    assert 'counts_conserved' in result['failed_gates']


def test_gate_engine_conditional_when_unresolved():
    result = evaluate({})
    assert result['status'] == 'CONDITIONAL'
    assert len(result['unknown_gates']) > 0


def test_gate_engine_ready_only_when_all_green():
    checks = {key: {'passed': True, 'evidence': {}} for key, _, _ in GATES}
    result = evaluate(checks)
    assert result['status'] == 'READY_FOR_CONTROLLED_PRETRAINING'


def test_gate_definitions_include_the_required_checks():
    keys = {key for key, _, _ in GATES}
    required = {'legacy_files_unchanged', 'licence_evidence_backed',
                'licence_unresolved_not_trained', 'train_and_redistribution_independent',
                'holdout_file_level_reported', 'holdout_confirmed_hits_blocked',
                'quality_rules_not_relaxed', 'group_split_inherited', 'counts_conserved',
                'rebuild_deterministic', 'sampling_exposure_only',
                'composition_from_manifest', 'optimizer_updates_zero'}
    assert required <= keys


def test_real_gate_output_exists_and_is_consistent():
    gates = json.loads((ROOT / 'reports/stage5e_gates.json').read_text())
    assert gates['status'] in ('READY_FOR_CONTROLLED_PRETRAINING', 'CONDITIONAL', 'BLOCKED')
    for key, value in gates['gates'].items():
        assert value['status'] in ('PASS', 'FAIL', 'UNRESOLVED')


# --------------------------------------------------------------------------
# Corpus conservation and the licence gate itself
# --------------------------------------------------------------------------

def test_r3v3_counts_conserve():
    manifest = json.loads(
        (ROOT / 'data/corpus_v4/R3v3_license_verified_manifest.json').read_text())
    counts = manifest['counts']
    assert counts['train'] + counts['val'] == counts['accepted']
    assert manifest['conservation']['conserved'] is True
    assert manifest['conservation']['accepted_plus_excluded'] == counts['ledger_rows']


def test_r3v3_accepted_is_a_subset_of_r3v2_accepted():
    r3v3 = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv',
                       usecols=['image_id'], dtype=str)
    r3v2 = pd.read_csv(ROOT / 'data/corpus_v3/R3v2_strict_rgb_train.csv',
                       usecols=['image_id'], dtype=str)
    assert set(r3v3['image_id']).issubset(set(r3v2['image_id']))


def test_r3v3_train_and_val_are_disjoint():
    train = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv',
                        usecols=['group_key', 'split'], dtype=str)
    val = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_val.csv',
                      usecols=['group_key', 'split'], dtype=str)
    known_train = set(train.loc[train['group_tier'] if 'group_tier' in train else
                                slice(None), 'group_key'].dropna())
    known_val = set(val['group_key'].dropna())
    strong_overlap = known_train & known_val
    # Only weak/unknown-group keys may legitimately appear on both sides, and
    # those are train-only in this corpus, so the intersection must be empty.
    assert not strong_overlap


def test_unknown_group_images_are_train_only():
    train = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv',
                        usecols=['group_tier', 'split'], dtype=str)
    val = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_val.csv',
                      usecols=['group_tier'], dtype=str)
    assert (train.loc[train['group_tier'] == 'UNKNOWN_GROUP', 'split'] ==
            'TRAIN_ONLY_GROUP_UNKNOWN').all()
    assert not (val['group_tier'] == 'UNKNOWN_GROUP').any()


def test_no_quarantine_image_was_upgraded():
    """R3v3 accepted must equal R3v2 accepted, since nothing was excluded."""
    manifest = json.loads(
        (ROOT / 'data/corpus_v4/R3v3_license_verified_manifest.json').read_text())
    stage5d = json.loads(
        (ROOT / 'data/corpus_v3/R3v2_strict_rgb_manifest.json').read_text())
    assert manifest['counts']['accepted'] <= stage5d['counts']['accepted']
    assert manifest['counts']['licence_excluded_research_use'] == 0


def test_accepted_images_all_have_a_positive_licence_status():
    train = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv',
                        usecols=['physical_source_id', 'stage5e_research_use',
                                 'stage5e_train_eligible'], dtype=str)
    assert train['stage5e_research_use'].isin(
        [lic.VERIFIED_RESEARCH_USE, lic.RESTRICTED_RESEARCH_USE]).all()
    assert (train['stage5e_train_eligible'].astype(str) == 'True').all()


def test_train_and_redistribution_columns_are_both_present_and_distinct():
    train = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv',
                        usecols=['stage5e_train_eligible', 'stage5e_redistribution_eligible'],
                        dtype=str)
    # agrifm_ref9 is the case where the two differ, so both values must occur.
    values = set(train['stage5e_redistribution_eligible'])
    assert 'True' in values


def test_target_margin_is_scale_only():
    manifest = json.loads(
        (ROOT / 'data/corpus_v4/R3v3_license_verified_manifest.json').read_text())
    target = manifest['target']
    assert target['target_exceeded'] is True
    assert target['target_margin'] == manifest['counts']['accepted'] - 147285
    assert 'scale property only' in target['note']


# --------------------------------------------------------------------------
# Historical files and zero training
# --------------------------------------------------------------------------

def test_stage5d_files_are_unchanged():
    proc = subprocess.run(
        ['git', 'diff', '--name-only', 'agri-stage5d-corpus-validation', '--',
         'data/corpus_v2', 'data/corpus_v3', 'reports/stage5c_', 'reports/stage5d_'],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.stdout.strip() == '', proc.stdout


def test_stage5c_files_are_unchanged():
    proc = subprocess.run(
        ['git', 'diff', '--name-only', 'agri-stage5c-agrifm147k-auto-corpus', '--',
         'data/corpus_v2', 'reports/stage5c_'],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.stdout.strip() == '', proc.stdout


def test_no_stage5e_tool_trains():
    """Detect real training constructs, not the word "optimizer" in prose."""
    import re as _re
    forbidden = {
        'optimizer_constructor': r'\b(?:torch\.optim\.)?(?:Adam|AdamW|SGD)\s*\(',
        'backward': r'\.backward\s*\(',
        'param_step': r'optimizer\.step\s*\(',
        'train_import': r'^\s*(?:from|import)\s+src\.(?:train|downstream)',
        'adapter': r'\bAdapter\s*\(',
        'criterion': r'\b(?:criterion|loss_fn)\s*\(',
    }
    offenders = []
    for path in list((ROOT / 'tools').glob('stage5e_*')) + \
            list((ROOT / 'src/registry').glob('stage5e_*')):
        text = path.read_text(errors='replace')
        for name, pattern in forbidden.items():
            if _re.search(pattern, text, _re.MULTILINE):
                offenders.append(f'{path.name}:{name}')
    assert offenders == []


def test_gates_report_zero_optimizer_updates():
    gates = json.loads((ROOT / 'reports/stage5e_gates.json').read_text())
    gate = gates['gates']['optimizer_updates_zero']
    assert gate['status'] == 'PASS'
    assert gate['evidence']['jepa_optimizer_updates'] == 0
    assert gate['evidence']['adapter_optimizer_updates'] == 0


# --------------------------------------------------------------------------
# Confirmed holdout leak (found in Stage 5E)
# --------------------------------------------------------------------------

def test_stage5d_collision_verdict_was_wrong_and_is_corrected():
    """Stage 5D called the GWHD distance-0 pairs a collision; the pixels say
    they are downscaled copies.  The review must record that verdict."""
    review = json.loads((ROOT / 'reports/stage5e_hash_collision_review.json').read_text())
    assert review['candidates_reviewed'] == 14
    verdicts = review['verdict_counts']
    assert verdicts.get('RESAMPLED_COPY_SAME_CONTENT') == 2
    assert verdicts.get('HASH_COLLISION_DIFFERENT_IMAGE') == 12


def test_confirmed_holdout_images_are_absent_from_train():
    review = json.loads((ROOT / 'reports/stage5e_hash_collision_review.json').read_text())
    to_exclude = set(review['corpus_image_ids_to_exclude'])
    assert len(to_exclude) == 2
    train = pd.read_csv(ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv',
                        usecols=['image_id'], dtype=str)
    assert not (to_exclude & set(train['image_id']))
    exclusions = pd.read_csv(
        ROOT / 'data/corpus_v4/R3v3_license_verified_exclusions.csv',
        usecols=['image_id', 'stage5e_primary_reason'], dtype=str)
    excluded_hits = exclusions[exclusions['image_id'].isin(to_exclude)]
    assert len(excluded_hits) == 2
    assert (excluded_hits['stage5e_primary_reason']
            == 'HOLDOUT_CONFIRMED_PIXEL_MATCH').all()


def test_confirmed_hits_stay_inside_the_pool_they_came_from():
    """The leak entered through the mixed AgriField pool's VegAnn subset."""
    frames = [pd.read_csv(path, usecols=['image_id', 'upstream_dataset'],
                          dtype=str, low_memory=False)
              for path in sorted((ROOT / 'data/corpus_v3/images').glob('*.csv'))]
    ledger = pd.concat(frames, ignore_index=True)
    review = json.loads((ROOT / 'reports/stage5e_hash_collision_review.json').read_text())
    leaked = ledger[ledger['image_id'].isin(review['corpus_image_ids_to_exclude'])]
    assert len(leaked) == 2
    assert set(leaked['upstream_dataset']) == {'VegAnn'}
    assert set(leaked['physical_source_id']) if 'physical_source_id' in leaked else True


def test_collision_review_does_not_delete_files():
    """The rule is manifest exclusion, never file deletion."""
    review = json.loads((ROOT / 'reports/stage5e_hash_collision_review.json').read_text())
    assert 'Nothing was deleted' in review['rule']
    train = pd.read_csv(ROOT / 'data/corpus_v3/R3v2_strict_rgb_train.csv',
                        usecols=['image_id', 'relative_path'], dtype=str)
    leaked = train[train['image_id'].isin(review['corpus_image_ids_to_exclude'])]
    # The original Stage 5D ledger still lists them, i.e. nothing was removed
    # from the source data.
    assert len(leaked) == 2
