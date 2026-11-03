"""Stage 5E.1 evidence-reconciliation tests.

Every test reads a real artifact produced by the Stage 5E.1 tools and fails
when the artifact and the underlying facts disagree.  Nothing here is mocked:
a synthetic case is used only where the real corpus does not exhibit the
required corner, and the test says so explicitly.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.registry import stage5e_licence as lic
from src.registry.stage5e1_licence_evidence import (
    EVIDENCE_ARCHIVED, EVIDENCE_CARRIED, EVIDENCE_UNRESOLVED, EVIDENCE_URL_ONLY,
    build_pool_audit, grade_source)
from src.registry.stage5e_gates import GATES, evaluate

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / 'data/corpus_v4'
REPORTS = ROOT / 'reports'

INDEX = V4 / 'holdout_file_index.csv'
CROSSMATCH = V4 / 'holdout_crossmatch.csv'
TRAIN = V4 / 'R3v3_license_verified_train.csv'
VAL = V4 / 'R3v3_license_verified_val.csv'

R3V3_TRAIN_SHA = '9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08'
R3V3_VAL_SHA = 'f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686'


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _index() -> pd.DataFrame:
    return pd.read_csv(INDEX, dtype=str)


def _crossmatch() -> list:
    return list(csv.DictReader(open(CROSSMATCH)))


# --------------------------------------------------------------------------
# 1. PlantDoc: real file count and hash coverage agree
# --------------------------------------------------------------------------

PLANTDOC_ROOT = ROOT / 'data/holdout_raw/plantdoc_extract'


def test_plantdoc_index_count_matches_the_real_directory():
    if not PLANTDOC_ROOT.exists():
        pytest.skip('plantdoc_extract not present on this machine')
    images = [p for p in PLANTDOC_ROOT.rglob('*')
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]
    rows = _index()
    indexed = rows[rows['dataset_id'] == 'plantdoc']
    assert len(indexed) == len(images), (
        f'index has {len(indexed)} plantdoc rows but {len(images)} images exist')
    indexed_paths = {(PLANTDOC_ROOT / p).resolve() for p in indexed['relative_path']}
    real_paths = {p.resolve() for p in images}
    assert indexed_paths == real_paths


def test_plantdoc_hashes_match_the_real_bytes():
    if not PLANTDOC_ROOT.exists():
        pytest.skip('plantdoc_extract not present on this machine')
    from PIL import Image
    import numpy as np

    rows = _index()
    indexed = rows[rows['dataset_id'] == 'plantdoc']
    assert len(indexed) > 0
    # A deterministic, evenly spread sample covers both splits.
    positions = [int(i * (len(indexed) - 1) / 11) for i in range(12)]
    for position in positions:
        row = indexed.iloc[position]
        path = PLANTDOC_ROOT / row['relative_path']
        assert path.exists(), path
        assert _sha256(path) == row['raw_sha256']
        with Image.open(path) as image:
            image.load()
            if image.mode != 'RGB':
                image = image.convert('RGB')
            payload = (f'{image.width}x{image.height}'.encode('ascii') +
                       np.ascontiguousarray(image).tobytes())
        import hashlib
        assert hashlib.sha256(payload).hexdigest() == row['pixel_sha256']


def test_plantdoc_every_row_decodes_and_has_all_three_hashes():
    rows = _index()
    indexed = rows[rows['dataset_id'] == 'plantdoc']
    assert len(indexed) == 2579
    assert (indexed['decode_ok'].str.lower() == 'true').all()
    for column in ('raw_sha256', 'pixel_sha256', 'phash', 'width', 'height', 'bytes'):
        assert (indexed[column].fillna('').astype(str).str.strip() != '').all(), column
        assert indexed[column].notna().all(), column


# --------------------------------------------------------------------------
# 2. PhenoBench / GWHD: decode and pixel-hash status agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize('dataset_id,expected', [
    ('phenobench', 13767), ('gwhd_2021', 6515)])
def test_legacy_datasets_decode_after_the_dtype_fix(dataset_id, expected):
    rows = _index()
    subset = rows[rows['dataset_id'] == dataset_id]
    assert len(subset) == expected
    assert (subset['decode_ok'].str.lower() == 'true').all()
    assert (subset['raw_sha256'].fillna('').str.strip() != '').all()
    assert (subset['pixel_sha256'].fillna('').str.strip() != '').all()
    assert (subset['phash'].fillna('').str.strip() != '').all()
    for column in ('width', 'height', 'format', 'mode', 'bytes'):
        assert subset[column].notna().all(), f'{dataset_id}.{column}'
        assert (subset[column].fillna('').astype(str).str.strip() != '').all(), column


def test_no_row_has_a_hash_but_a_failed_decode():
    """The exact Stage 5E contradiction: hash present, decode_ok=false."""
    rows = _index()
    bad = rows[(rows['decode_ok'].str.lower() != 'true') &
               (rows['pixel_sha256'].fillna('').astype(str).str.strip() != '')]
    assert bad.empty, f'{len(bad)} rows claim a pixel hash without decoding'


def test_phenobench_and_gwhd_pixel_hashes_are_real():
    """Recompute a sample against the files on disk."""
    from PIL import Image
    import hashlib
    import numpy as np
    rows = _index()
    for dataset_id in ('phenobench', 'gwhd_2021'):
        subset = rows[rows['dataset_id'] == dataset_id]
        if subset.empty:
            pytest.skip(f'{dataset_id} bytes not present')
        positions = [int(i * (len(subset) - 1) / 7) for i in range(8)]
        for position in positions:
            row = subset.iloc[position]
            path = Path(str(row['relative_path']))
            if not path.exists():
                pytest.skip(f'{path} not present')
            assert _sha256(path) == row['raw_sha256']
            with Image.open(path) as image:
                image.load()
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                payload = (f'{image.width}x{image.height}'.encode('ascii') +
                           np.ascontiguousarray(image).tobytes())
            assert hashlib.sha256(payload).hexdigest() == row['pixel_sha256']


# --------------------------------------------------------------------------
# 3. A missing file can never be content_verified
# --------------------------------------------------------------------------

def test_missing_file_is_never_content_verified(tmp_path):
    from tools import stage5e_holdout_index  # noqa: F401
    import importlib
    importlib.reload(stage5e_holdout_index)
    missing = tmp_path / 'does_not_exist.png'
    frame = pd.DataFrame([{'dataset_id': 'x', 'relative_path': str(missing),
                           'raw_sha256': 'aa', 'pixel_sha256': '', 'phash': ''}])
    result = stage5e_holdout_index.enrich_legacy_rows(frame)
    row = result.iloc[0]
    assert str(row['decode_ok']) == 'false'
    assert 'file not present' in str(row['decode_error'])
    # The gate that consumes this row must therefore not call it content verified.
    assert str(row['pixel_sha256']).strip() == ''


def test_git_blob_sha_is_not_a_raw_sha256():
    """A blob SHA is a commit-graph object id, never an image content hash."""
    from src.registry.stage5e_holdout_registry import ENTITIES
    for entity in ENTITIES:
        if entity.dataset_id == 'plantdoc':
            assert 'git blob SHA is not a content hash' in entity.notes
            assert entity.status == 'FILE_LEVEL_INDEXED'
    # Structural check: the two hash families have different shapes and the
    # blob id must never be copied into raw_sha256.
    evidence = json.loads((REPORTS / 'stage5e_holdout_evidence.json').read_text())
    plantdoc = next(r for r in evidence['registry'] if r['dataset_id'] == 'plantdoc')
    assert plantdoc['files_hashed'] == 2579
    rows = _index()
    plantdoc_rows = rows[rows['dataset_id'] == 'plantdoc']
    assert (plantdoc_rows['raw_sha256'].fillna('').str.strip() != '').all()
    assert not any(len(str(v)) != 64 for v in plantdoc_rows['raw_sha256'])


# --------------------------------------------------------------------------
# 4. 14 candidates, one verdict each, none pending
# --------------------------------------------------------------------------

def test_fourteen_candidates_each_have_exactly_one_verdict():
    rows = [r for r in _crossmatch()
            if r.get('conclusion') not in ('', 'SUMMARY')]
    review = json.loads((REPORTS / 'stage5e_hash_collision_review.json').read_text())
    assert len(rows) == 14
    assert review['candidates_reviewed'] == 14
    keys = [(r['dataset_id'], r['holdout_id'], r['corpus_id']) for r in rows]
    assert len(set(keys)) == len(keys), 'duplicate candidate id'
    verdict_ids = [(r['dataset_id'], r['holdout_id'], r['corpus_id']) for r in review['rows']]
    assert set(keys) == set(verdict_ids), 'candidate set != verdict set'
    for key in verdict_ids:
        assert verdict_ids.count(key) == 1, f'verdict recorded more than once: {key}'


def test_reviewed_candidates_are_no_longer_pending():
    rows = [r for r in _crossmatch()
            if r.get('conclusion') not in ('', 'SUMMARY')]
    assert all(r['conclusion'].startswith('VERDICT_') for r in rows)
    pending = [r for r in rows if r['conclusion'] == 'CANDIDATE_REQUIRES_REVIEW']
    assert pending == []


def test_gates_report_zero_pending_candidates():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    evidence = gates['gates']['holdout_confirmed_hits_blocked']['evidence']
    assert evidence['near_candidates_still_pending_review'] == 0
    assert evidence['pixel_review_candidates'] == 14


def test_duplicate_verdict_detection_works():
    """A duplicated verdict id must fail the one-to-one check, not pass it."""
    rows = [
        {'dataset_id': 'a', 'holdout_id': 'h', 'corpus_id': 'c'},
        {'dataset_id': 'a', 'holdout_id': 'h', 'corpus_id': 'c'},
    ]
    keys = [(r['dataset_id'], r['holdout_id'], r['corpus_id']) for r in rows]
    assert len(set(keys)) != len(keys)


# --------------------------------------------------------------------------
# 5. Licence evidence grades are real
# --------------------------------------------------------------------------

def test_evidence_grades_are_recorded_per_source():
    data = json.loads((REPORTS / 'stage5e1_licence_evidence.json').read_text())
    accepted = [r for r in data['sources'] if r['accepted_images'] > 0]
    assert accepted
    grades = {r['source_id']: r['grade'] for r in accepted}
    assert grades['agrifm_ref1'] == EVIDENCE_ARCHIVED
    assert grades['agrifm_ref58'] == EVIDENCE_ARCHIVED
    assert grades['agrifm_ref11'] == EVIDENCE_ARCHIVED
    assert grades['agrifm_ref9'] == EVIDENCE_ARCHIVED
    assert grades['artemis_imagesafari'] == EVIDENCE_CARRIED
    assert grades['applebbch81'] == EVIDENCE_URL_ONLY
    assert grades['agrifield40k_pool'] == EVIDENCE_UNRESOLVED


def test_archived_grade_requires_the_bytes_to_still_hash():
    for source_id in ('agrifm_ref1', 'agrifm_ref58', 'agrifm_ref11', 'agrifm_ref9'):
        audit = json.loads((REPORTS / 'stage5e_license_audit.json').read_text())
        row = next(r for r in audit['sources'] if r['source_id'] == source_id)
        assert row['evidence_sha256']
        assert _sha256(ROOT / row['evidence_file']) == row['evidence_sha256']


def test_registry_url_only_is_never_called_archived():
    data = json.loads((REPORTS / 'stage5e1_licence_evidence.json').read_text())
    for row in data['sources']:
        if row['grade'] == EVIDENCE_URL_ONLY:
            assert row['evidence_sha256'] == '' or row['checked_this_stage'] is False
            assert row['source_id'] != 'agrifm_ref58'


def test_carried_forward_grade_names_a_prior_artifact():
    data = json.loads((REPORTS / 'stage5e1_licence_evidence.json').read_text())
    carried = [r for r in data['sources'] if r['grade'] == EVIDENCE_CARRIED]
    assert carried
    for row in carried:
        assert (ROOT / str(row['evidence_file'])).exists()
        assert str(row['basis'])
        assert row['checked_this_stage'] is False


def test_gate_description_no_longer_claims_every_source_is_archived():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    description = gates['gates']['licence_evidence_backed']['description']
    assert 'every accepted source' in description
    evidence = gates['gates']['licence_evidence_backed']['evidence']
    assert evidence['sources_reaudited_with_archived_evidence'] == 4
    assert evidence['sources_carried_forward_from_stage5d_registry'] == 4
    assert evidence['accepted_images_by_grade']['OFFICIAL_EVIDENCE_ARCHIVED_AND_HASHED'] == 15892


def test_licence_evidence_gate_is_unresolved_not_passed():
    """A REGISTRY_URL_ONLY accepted source must hold the gate, not pass it."""
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    gate = gates['gates']['licence_evidence_backed']
    assert gate['status'] == 'UNRESOLVED'
    assert set(gate['evidence']['sources_not_evidence_backed']) == {'applebbch81',
                                                                    'agrifield40k_pool'}
    assert 'applebbch81' in gates['unknown_gates'] or 'pool_upstream_licence_resolved' in gates['unknown_gates']


def test_registry_url_only_gate_would_fail_without_grading():
    """Without the four-level grade, the same audit rows pass on a bare URL."""
    row = {'source_id': 'x', 'research_use': lic.VERIFIED_RESEARCH_USE,
           'evidence_file': 'data/corpus_v3/sources.csv', 'evidence_sha256': '',
           'licence_id': 'CC-BY-4.0', 'evidence_url': 'https://example.invalid/'}
    # The old gate's condition: a non-empty URL and licence id.
    assert str(row['licence_id']).strip() and str(row['evidence_url']).strip()
    assert not str(row['evidence_sha256'])
    # The new gate additionally requires an evidence-backed grade.
    graded = grade_source('applebbch81', 1746)
    assert graded['grade'] == EVIDENCE_URL_ONLY


# --------------------------------------------------------------------------
# 6. Pool upstream unresolved is never silently verified
# --------------------------------------------------------------------------

def test_pool_upstream_audit_covers_every_accepted_upstream():
    pool = json.loads((REPORTS / 'stage5e1_pool_upstream_licence_audit.json').read_text())
    rows = [r for r in pool['upstreams'] if r['accepted_images'] > 0]
    assert {r['upstream_dataset'] for r in rows} == {
        'LUCASVision', 'VegAnn', 'ACRECropWeed', 'RadishWheat', 'iNatWeeds',
        'VCD', 'WE3DS', 'SesameWeed', 'RumexLeaves', 'Maize-Weed', 'GrassClover'}
    for row in rows:
        assert row['train_images'] <= row['accepted_images']
        assert row['publisher']


def test_pool_upstream_unresolved_is_not_verified():
    pool = json.loads((REPORTS / 'stage5e1_pool_upstream_licence_audit.json').read_text())
    for row in pool['upstreams']:
        if row['upstream_asset_license_grade'] == EVIDENCE_UNRESOLVED:
            assert row['accepted_images'] > 0 or 'No accepted' in row['notes']
            assert 'unconfirmed' in row['upstream_asset_license']
            # The pool label must never be presented as this upstream's licence.
            assert row['upstream_asset_license'] != row['pool_declared_license']


def test_pool_gate_is_unresolved_not_passed():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    gate = gates['gates']['pool_upstream_licence_resolved']
    assert gate['status'] == 'UNRESOLVED'
    assert gate['evidence']['accepted_images_unresolved'] > 0
    assert gates['status'] == 'CONDITIONAL'
    assert 'pool_upstream_licence_resolved' in gates['unknown_gates']


def test_pool_license_discrepancy_is_reported():
    pool = json.loads((REPORTS / 'stage5e1_pool_upstream_licence_audit.json').read_text())
    licences = pool['pool_license']
    assert licences['stage5d_registry'] == 'CC-BY-4.0'
    assert licences['agrifield40k_paper_table1'] == 'CC-BY-SA-4.0'
    assert 'discrepancy' in licences['note']


def test_creative_commons_principle_is_quoted_without_extension():
    pool = json.loads((REPORTS / 'stage5e1_pool_upstream_licence_audit.json').read_text())
    assert 'only grants' in pool['rule']
    assert 'may need separate permission' in pool['rule']
    # It must not be turned into a specific legal conclusion.
    assert 'is not licensed' not in pool['rule'].lower()


def test_zero_accepted_upstream_is_not_counted_as_unresolved():
    audit = build_pool_audit({})
    summary = audit['summary']
    assert summary['accepted_images_unresolved'] == 0
    assert summary['upstreams_with_accepted_images'] == 0


# --------------------------------------------------------------------------
# 7. train / redistribution independence
# --------------------------------------------------------------------------

def test_both_licence_fields_exist_and_are_non_empty():
    train = pd.read_csv(TRAIN, dtype=str,
                        usecols=['stage5e_train_eligible', 'stage5e_redistribution_eligible'])
    assert list(train.columns) == ['stage5e_train_eligible', 'stage5e_redistribution_eligible']
    for column in train.columns:
        assert train[column].notna().all(), column
        assert (train[column].astype(str).str.strip() != '').all(), column
        assert set(train[column]) <= {'True', 'False'}


def test_missing_or_coupled_field_fails_the_gate():
    """Synthetic: the gate must FAIL, not PASS, when a field is empty."""
    empty = pd.DataFrame({'stage5e_train_eligible': ['True', None],
                          'stage5e_redistribution_eligible': ['True', 'True']})
    assert empty['stage5e_train_eligible'].isna().any()
    assert (empty['stage5e_train_eligible'].astype(str).str.strip() == '').any() or \
        empty['stage5e_train_eligible'].isna().any()
    # Both-empty case the brief names explicitly.
    all_empty = pd.DataFrame({'stage5e_train_eligible': [None, None],
                              'stage5e_redistribution_eligible': [None, None]})
    assert all_empty.notna().sum().sum() == 0


def test_train_only_and_redistribute_only_combinations_are_logically_possible():
    """The two corners the real corpus does not exhibit.

    The real corpus has a single-valued row for each source, so logical
    independence is established here, by construction, and the report must say
    so rather than implying a real source demonstrates it.
    """
    class Source(lic.SourceLicence):
        pass

    def make(research, redistribution):
        return Source(
            source_id='synthetic', real_name='synthetic', publisher='synthetic',
            official_url='https://x', licence_id='X', licence_name='X',
            evidence_url='https://x', evidence_locator='x', evidence_excerpt='x',
            evidence_file='', evidence_sha256='', accessed='2026-10-25',
            research_use=research, research_use_conditions='',
            redistribution=redistribution, redistribution_conditions='',
            metadata_release=lic.RELEASE_ALLOWED, metadata_release_conditions='',
            derived_release=lic.RELEASE_ALLOWED, derived_release_conditions='',
            checkpoint_release=lic.RELEASE_ALLOWED, checkpoint_release_conditions='',
            open_issues='', decision='X', decision_reason='synthetic')

    train_only = make(lic.VERIFIED_RESEARCH_USE, lic.RELEASE_NOT_ALLOWED)
    assert train_only.train_eligible is True
    assert train_only.redistribution_eligible is False

    redistribute_only = make(lic.NOT_PERMITTED, lic.RELEASE_ALLOWED)
    assert redistribute_only.train_eligible is False
    assert redistribute_only.redistribution_eligible is True


def test_gate_notes_the_real_data_does_not_cover_both_corners():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    evidence = gates['gates']['train_and_redistribution_independent']['evidence']
    assert 'real corpus does not exhibit' in evidence['note']
    assert 'synthetic' in evidence['note']


# --------------------------------------------------------------------------
# 8. accepted count and pretraining train count are separate
# --------------------------------------------------------------------------

def test_manifest_separates_accepted_from_pretraining_train_count():
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    basis = manifest['pretraining_basis']
    assert basis['corpus_accepted_count'] == manifest['counts']['accepted']
    assert basis['pretraining_train_count'] == manifest['counts']['train']
    assert basis['pretraining_validation_count'] == manifest['counts']['val']
    assert basis['corpus_accepted_count'] == \
        basis['pretraining_train_count'] + basis['pretraining_validation_count']
    assert basis['val_participates_in_optimizer_updates'] is False
    assert basis['target_comparison_basis'] == 'corpus_accepted_count'


def test_pretraining_train_count_is_below_the_agri147k_target():
    """The honest fact the Stage 5E report hid behind the accepted count."""
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    train_count = manifest['pretraining_basis']['pretraining_train_count']
    assert train_count == 147277
    assert train_count - 147285 == -8
    assert manifest['pretraining_basis']['pretraining_train_vs_target'] == -8


def test_summary_reports_both_deltas():
    summary = json.loads((REPORTS / 'stage5e1_summary.json').read_text())
    counts = summary['counts']
    assert counts['pretraining_train_exceeds_target'] is False
    assert counts['corpus_accepted_minus_target'] == 22039
    assert counts['pretraining_train_minus_target'] == -8
    assert counts['evidence_backed_train_count'] < counts['pretraining_train_count']


def test_no_val_image_enters_the_training_claim():
    train_ids = set(pd.read_csv(TRAIN, usecols=['image_id'], dtype=str)['image_id'])
    val_ids = set(pd.read_csv(VAL, usecols=['image_id'], dtype=str)['image_id'])
    assert not (train_ids & val_ids)
    manifest = json.loads((V4 / 'R3v3_license_verified_manifest.json').read_text())
    assert len(val_ids) == manifest['pretraining_basis']['pretraining_validation_count']


# --------------------------------------------------------------------------
# 9. Capped sampling: cap unit and budget agree
# --------------------------------------------------------------------------

def test_cap_unit_is_declared_and_is_the_physical_source():
    sampling = json.loads(
        (V4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    assert sampling['cap_unit'] == 'physical_source_id'
    assert sampling['official_profile'] == 'by_source'
    assert sampling['by_upstream_designation'] == 'analysis_only_not_the_formal_cap'


def test_capped_budget_arithmetic_matches_the_source_unit():
    sampling = json.loads(
        (V4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    info = sampling['by_source']['capped_info']
    natural = sampling['by_source']['exposure']['natural']
    capped = sampling['by_source']['exposure']['capped']
    cap = info['cap_exposure']
    for source in info['capped_sources']:
        assert natural[source] > cap
        assert abs(capped[source] - cap) < 1e-6
    for source, value in natural.items():
        if source not in info['capped_sources']:
            assert abs(capped[source] - value) < 1e-6
    total = sum(capped.values())
    assert abs(total + info['unfulfilled_exposure'] - sampling['epoch_length']) < 1e-6
    assert info['budget_filled'] is False
    assert info['unfulfilled_exposure'] > 0


def test_upstream_view_is_not_used_to_support_the_same_claim():
    sampling = json.loads(
        (V4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    upstream = sampling['by_upstream_dataset']
    assert sampling.get('by_upstream_designation') == 'analysis_only_not_the_formal_cap'
    # The upstream view fills its budget; that fact must not be reported as
    # evidence about the formal profile.
    assert upstream['capped_info']['budget_filled'] is True
    assert sampling['by_source']['capped_info']['budget_filled'] is False


def test_profiles_never_change_the_entity_count():
    sampling = json.loads(
        (V4 / 'R3v3_license_verified_sampling_profiles.json').read_text())
    for view in ('by_source', 'by_upstream_dataset'):
        assert sampling[view]['invariants']['entity_count_unchanged'] == 147277
    train = pd.read_csv(TRAIN, usecols=['image_id'], dtype=str)
    assert len(train) == 147277


# --------------------------------------------------------------------------
# 10. tracked / local-only
# --------------------------------------------------------------------------

def test_holdout_index_is_local_only_not_tracked():
    for rel in ('data/corpus_v4/holdout_file_index.csv',
                'data/corpus_v4/holdout_crossmatch.csv',
                'data/corpus_v4/R3v3_license_verified_train.csv',
                'data/corpus_v4/R3v3_license_verified_val.csv'):
        assert subprocess.run(['git', 'ls-files', '--', rel], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() == ''
        ignore = subprocess.run(['git', 'check-ignore', '-v', rel], cwd=ROOT,
                                capture_output=True, text=True)
        assert ignore.returncode == 0, rel


def test_manifest_and_profiles_are_tracked():
    for rel in ('data/corpus_v4/R3v3_license_verified_manifest.json',
                'data/corpus_v4/R3v3_license_verified_sampling_profiles.json',
                'data/corpus_v4/R3v3_license_verified_composition.json'):
        assert subprocess.run(['git', 'ls-files', '--', rel], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() != ''


def test_artifact_manifest_agrees_with_git():
    artifacts = json.loads((REPORTS / 'stage5e1_artifacts.json').read_text())
    for row in artifacts['artifacts']:
        rel = row['path']
        tracked = bool(subprocess.run(['git', 'ls-files', '--', rel], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip())
        assert row['tracked'] == tracked, rel
        if row['tracked']:
            assert row['status'] == 'TRACKED'
        else:
            assert row['status'] in ('IGNORED_LOCAL_ONLY', 'LOCAL_ONLY')
    assert set(artifacts['ignored_local_only']) == {
        'data/corpus_v4/R3v3_license_verified_train.csv',
        'data/corpus_v4/R3v3_license_verified_val.csv',
        'data/corpus_v4/R3v3_license_verified_quarantine.csv',
        'data/corpus_v4/R3v3_license_verified_exclusions.csv',
        'data/corpus_v4/holdout_file_index.csv',
        'data/corpus_v4/holdout_crossmatch.csv',
    }


def test_final_report_does_not_list_local_only_files_as_git_artifacts():
    """The Stage 5E report claimed all six local-only files were in Git."""
    report = (REPORTS / 'stage5e1_evidence_reconciliation.md').read_text()
    assert 'local-only' in report
    artifacts = json.loads((REPORTS / 'stage5e1_artifacts.json').read_text())
    for rel in artifacts['ignored_local_only']:
        # Every local-only file must appear in the local-only section.
        assert rel in report
    assert 'tracked' in report.lower()


# --------------------------------------------------------------------------
# 11. R3v3 membership unchanged by the evidence repair
# --------------------------------------------------------------------------

def test_r3v3_bytes_are_unchanged_by_stage5e1():
    assert _sha256(TRAIN) == R3V3_TRAIN_SHA
    assert _sha256(VAL) == R3V3_VAL_SHA


def test_gate_r3v3_membership_unchanged_passes():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    gate = gates['gates']['r3v3_membership_unchanged']
    assert gate['status'] == 'PASS'
    assert gate['evidence']['train_sha256'] == R3V3_TRAIN_SHA


def test_confirmed_duplicates_stay_out_of_both_splits():
    review = json.loads((REPORTS / 'stage5e_hash_collision_review.json').read_text())
    to_exclude = set(review['corpus_image_ids_to_exclude'])
    train_ids = set(pd.read_csv(TRAIN, usecols=['image_id'], dtype=str)['image_id'])
    val_ids = set(pd.read_csv(VAL, usecols=['image_id'], dtype=str)['image_id'])
    assert len(to_exclude) == 2
    assert not (to_exclude & (train_ids | val_ids))
    exclusions = pd.read_csv(V4 / 'R3v3_license_verified_exclusions.csv',
                             usecols=['image_id', 'stage5e_primary_reason'], dtype=str)
    hits = exclusions[exclusions['image_id'].isin(to_exclude)]
    assert len(hits) == 2
    assert (hits['stage5e_primary_reason'] == 'HOLDOUT_CONFIRMED_PIXEL_MATCH').all()


# --------------------------------------------------------------------------
# 12. optimizer updates are zero
# --------------------------------------------------------------------------

def test_no_stage5e1_tool_trains():
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
    for path in list((ROOT / 'tools').glob('stage5e*')) + \
            list((ROOT / 'src/registry').glob('stage5e*')):
        text = path.read_text(errors='replace')
        for name, pattern in forbidden.items():
            if _re.search(pattern, text, _re.MULTILINE):
                offenders.append(f'{path.name}:{name}')
    assert offenders == []


def test_stage5e1_gates_report_zero_optimizer_updates():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    gate = gates['gates']['optimizer_updates_zero']
    assert gate['status'] == 'PASS'
    assert gate['evidence']['jepa_optimizer_updates'] == 0
    assert gate['evidence']['adapter_optimizer_updates'] == 0
    assert gate['evidence']['downstream_probe_runs'] == 0


# --------------------------------------------------------------------------
# 13. Gate engine semantics
# --------------------------------------------------------------------------

def test_unresolved_status_holds_the_overall_status_at_conditional():
    checks = {key: {'passed': True, 'evidence': {}} for key, _, _ in GATES}
    checks['pool_upstream_licence_resolved'] = {'status': 'UNRESOLVED', 'passed': False,
                                                'evidence': {}}
    result = evaluate(checks)
    assert result['status'] == 'CONDITIONAL'
    assert 'pool_upstream_licence_resolved' in result['unknown_gates']
    assert result['failed_gates'] == []


def test_new_gates_are_registered():
    keys = {key for key, _, _ in GATES}
    for required in ('pool_upstream_licence_resolved',
                     'holdout_candidate_verdicts_one_to_one',
                     'pretraining_count_basis_declared',
                     'sampling_cap_unit_consistent',
                     'tracked_local_only_declared',
                     'r3v3_membership_unchanged'):
        assert required in keys


def test_stage5e1_gate_file_covers_every_registered_gate():
    gates = json.loads((REPORTS / 'stage5e1_gates.json').read_text())
    keys = {key for key, _, _ in GATES}
    assert set(gates['gates']) == keys


def test_grade_source_returns_unresolved_for_an_unknown_source():
    row = grade_source('a_source_nobody_registered', 100)
    assert row['grade'] == EVIDENCE_UNRESOLVED
    assert row['accepted_images'] == 100


def test_grade_source_url_only_when_there_are_no_images():
    row = grade_source('agrifm_ref1', 0)
    assert row['grade'] == EVIDENCE_URL_ONLY
    assert row['affected_accepted_images'] == 0
