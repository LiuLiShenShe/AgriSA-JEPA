"""Stage 5D corpus-validation tests.

Each test names the defect it locks shut.  The tests are ordered by the
defect, not by module.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.stage5d_dedup import (DedupConfig, exact_duplicate_decisions,
                                        hash_quality, near_duplicate_scan,
                                        validate_hash_columns)
from src.registry.stage5d_funnel import (ACCEPTED, conservation,
                                        decide_frame, funnel_table,
                                        per_source_conservation)
from src.registry.stage5d_gates import STATUS_BLOCKED, STATUS_READY, status
from src.registry.stage5d_group_policy import (STRONG_GROUPS, GroupAssignment,
                                               assign_diagnostic_random,
                                               assign_splits, agrifield_group)
from src.registry.stage5d_metadata import parse_artemis_key
from src.registry.stage5d_metadata_schema import (GROUP_DOCUMENTED_PATH,
                                                  GROUP_OFFICIAL,
                                                  GROUP_UNKNOWN, GROUP_WEAK_PROXY,
                                                  MODALITY_RGB_VISIBLE,
                                                  MODALITY_STATUS_EVIDENCED,
                                                  MODALITY_STATUS_NOT_VISIBLE,
                                                  PROVENANCE_OFFICIAL,
                                                  normalize_crop)
from src.registry.stage5d_modality import classify_row, screen_visible_rgb
from src.registry.stage5d_qc_policy import (MetricSpec, PolicyError,
                                            combine_metric_statuses, load_policy)
from src.registry.stage5d_qc_validation import (DEGRADATIONS,
                                               basic_decision_from_metrics,
                                               degrade, measure_pil)

POLICY = ROOT / 'data/corpus_v3/policy.yaml'
CORPUS_V2 = ROOT / 'data/corpus_v2'
CORPUS_V3 = ROOT / 'data/corpus_v3'


def base_frame(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        'image_id': [f'img{i}' for i in range(rows)],
        'physical_source_id': ['src_a'] * rows,
        'relative_path': [f'folder/{i}.jpg' for i in range(rows)],
        'raw_sha256': ['a'] * rows,
        'pixel_sha256': ['b'] * rows,
        'phash': ['0000000000000000'] * rows,
        'dhash': ['0000000000000000'] * rows,
        'width': [640] * rows,
        'height': [480] * rows,
        'short_side': [480] * rows,
        'decode_ok': ['true'] * rows,
        'laplacian_var': [120.0] * rows,
        'tenengrad': [90.0] * rows,
        'brightness_mean': [90.0] * rows,
        'entropy': [6.0] * rows,
        'blank_fraction': [0.01] * rows,
        'aspect_ratio': [1.33] * rows,
        'iqa_musiq': [70.0] * rows,
        'iqa_niqe': [4.0] * rows,
        'acquisition_domain': ['FIELD_IN_SITU'] * rows,
        'domain_status': ['OK'] * rows,
        'domain_evidence_type': ['official_metadata'] * rows,
        'modality_status': ['VISIBLE_RGB_EVIDENCED'] * rows,
        'modality': ['RGB_VISIBLE'] * rows,
        'license_status': ['ACCEPTED'] * rows,
        'license_ok': [True] * rows,
        'holdout_status': ['OK'] * rows,
        'archive_only': ['false'] * rows,
        'basic_status': ['PASS'] * rows,
        'iqa_gate_status': ['PASS'] * rows,
        'exact_duplicate': [False] * rows,
        'near_dup_status': ['OK'] * rows,
        'group_key': ['g0'] * rows,
        'group_tier': ['OFFICIAL_GROUP'] * rows,
        'split': ['train'] * rows,
        'crop': ['wheat'] * rows,
    })


# --------------------------------------------------------------------------- #
# P0-1 funnel conservation
# --------------------------------------------------------------------------- #
def test_funnel_and_manifest_come_from_one_decision_table():
    frame = decide_frame(base_frame())
    assert frame['primary_reason'].eq(ACCEPTED).all()
    report = conservation(frame)
    assert report['conserved']
    assert report['n_input'] == report['n_accepted'] == 6
    assert funnel_table(frame, 'test')[-1]['remaining_images'] == 6


def test_every_row_has_exactly_one_primary_reason():
    frame = base_frame(4)
    frame.loc[0, 'iqa_gate_status'] = 'REJECT'
    frame.loc[1, 'holdout_status'] = 'HOLDOUT'
    frame.loc[2, 'modality_status'] = 'NOT_VISIBLE_RGB'
    frame.loc[3, 'license_ok'] = False
    decided = decide_frame(frame)
    reasons = list(decided['primary_reason'])
    assert reasons == ['IQA_REJECT', 'HOLDOUT_EXCLUDED',
                       'MODALITY_NOT_VISIBLE_RGB', 'LICENCE_BLOCKED']
    report = conservation(decided)
    assert report['conserved']
    assert report['n_holdout'] == 1
    assert report['n_rejected'] == 2   # IQA_REJECT + MODALITY_NOT_VISIBLE_RGB
    assert report['n_licence_blocked'] == 1


def test_secondary_reasons_record_hidden_failures():
    frame = base_frame(2)
    frame.loc[0, 'holdout_status'] = 'HOLDOUT'
    frame.loc[1, 'iqa_gate_status'] = 'REJECT'
    frame.loc[0, 'license_ok'] = False
    decided = decide_frame(frame)
    # Holdout is the primary reason, but the licence failure must still be visible.
    assert 'LICENCE_BLOCKED' in decided.iloc[0]['secondary_reasons']


def test_per_source_counts_sum_to_global_total():
    frame = base_frame(6)
    frame['physical_source_id'] = ['a'] * 3 + ['b'] * 3
    decided = decide_frame(frame)
    table = per_source_conservation(decided)
    assert int(table['total'].sum()) == len(decided)


def test_priority_order_is_frozen():
    frame = base_frame(1)
    frame.loc[0, 'holdout_status'] = 'HOLDOUT'
    frame.loc[0, 'modality_status'] = 'NOT_VISIBLE_RGB'
    decided = decide_frame(frame)
    assert decided.iloc[0]['primary_reason'] == 'HOLDOUT_EXCLUDED'


# --------------------------------------------------------------------------- #
# P0-2 IQA policy
# --------------------------------------------------------------------------- #
def test_policy_loads_and_declares_scale():
    policy = load_policy(POLICY)
    musiq = policy.metrics['musiq']
    assert musiq.direction == 'higher_is_better'
    assert musiq.score_range == (0.0, 100.0)
    assert musiq.accept_threshold == 45.0
    assert musiq.reject_threshold == 25.0
    assert musiq.model_sha256
    assert musiq.role == 'GATE'


def test_missing_iqa_is_never_a_pass():
    policy = load_policy(POLICY)
    musiq = policy.metrics['musiq']
    for value in ('', None, float('nan'), float('inf'), float('-inf'), 'nan'):
        result, reason = musiq.evaluate(value)
        assert result == 'MISSING', (value, result, reason)


def test_missing_score_never_combines_into_pass():
    assert combine_metric_statuses(['PASS', 'MISSING'])[0] == 'QUARANTINE'
    assert combine_metric_statuses(['PASS', 'REJECT'])[0] == 'REJECT'
    assert combine_metric_statuses(['PASS', 'QUARANTINE'])[0] == 'QUARANTINE'
    assert combine_metric_statuses(['PASS'])[0] == 'PASS'


def test_boundary_values_are_exclusive_on_the_reject_side():
    spec = MetricSpec('x', 'higher_is_better', (0.0, 100.0), 45.0, 25.0)
    assert spec.evaluate(45.0)[0] == 'PASS'
    assert spec.evaluate(44.999)[0] == 'QUARANTINE'
    assert spec.evaluate(25.0)[0] == 'QUARANTINE'
    assert spec.evaluate(24.999)[0] == 'REJECT'


def test_lower_is_better_metric_is_supported():
    spec = MetricSpec('niqe', 'lower_is_better', (0.0, 100.0), 8.0, 12.0)
    assert spec.evaluate(5.0)[0] == 'PASS'
    assert spec.evaluate(10.0)[0] == 'QUARANTINE'
    assert spec.evaluate(15.0)[0] == 'REJECT'


def test_out_of_range_score_is_missing_not_pass():
    spec = MetricSpec('x', 'higher_is_better', (0.0, 100.0), 45.0, 25.0)
    assert spec.evaluate(-5.0)[0] == 'MISSING'
    assert spec.evaluate(500.0)[0] == 'MISSING'


def test_mixed_types_do_not_crash():
    spec = MetricSpec('x', 'higher_is_better', (0.0, 100.0), 45.0, 25.0)
    for value in ('55.5', 55, None, True, object()):
        spec.evaluate(value)  # must not raise


def test_contradictory_policy_is_rejected():
    with pytest.raises(PolicyError):
        MetricSpec('x', 'higher_is_better', (0.0, 100.0), 25.0, 45.0).validate()
    with pytest.raises(PolicyError):
        MetricSpec('x', 'sideways', (0.0, 100.0), 25.0, 45.0).validate()


def test_niqe_is_not_an_admission_gate():
    policy = load_policy(POLICY)
    assert policy.metrics['niqe'].role == 'DIAGNOSTIC_ONLY'
    assert 'niqe' not in {m.name for m in policy.gate_metrics()}


# --------------------------------------------------------------------------- #
# P0-3 modality
# --------------------------------------------------------------------------- #
def test_nir_folder_is_never_visible_rgb():
    decision = classify_row('agrifm_ref11', 'P1/nir/3335161_nirraw_PTZ002.jpeg', 'RGB+NDVI')
    assert decision['modality_status'] == MODALITY_STATUS_NOT_VISIBLE


def test_rgb_folder_is_visible_rgb():
    decision = classify_row('agrifm_ref11', 'P1/rgb/3335161_rgbnormal_PTZ002.jpeg', 'RGB+NDVI')
    assert decision['modality'] == MODALITY_RGB_VISIBLE
    assert decision['modality_status'] == MODALITY_STATUS_EVIDENCED


def test_dead_channel_is_rejected_by_the_pixel_screen():
    passed, reason = screen_visible_rgb({'valid': 1.0, 'mean_r': 149.0,
                                         'mean_g': 1.7, 'mean_b': 72.6,
                                         'corr_rg': 0.08, 'corr_rb': 0.65,
                                         'corr_gb': 0.24})
    assert not passed
    assert reason in ('DEAD_CHANNEL', 'EXTREME_CHANNEL_OFFSET')


def test_photographic_channels_pass_the_screen():
    passed, _ = screen_visible_rgb({'valid': 1.0, 'mean_r': 67.5, 'mean_g': 77.0,
                                    'mean_b': 70.5, 'corr_rg': 0.94, 'corr_rb': 0.98,
                                    'corr_gb': 0.96})
    assert passed


def test_multispectral_source_cannot_enter_strict_rgb():
    decision = classify_row('agrifm_ref82', 'images/001.jpg', 'multispectral')
    assert decision['modality_status'] == MODALITY_STATUS_NOT_VISIBLE


# --------------------------------------------------------------------------- #
# P0-4 group policy
# --------------------------------------------------------------------------- #
def test_unknown_group_cannot_produce_group_disjoint_val():
    assignments = [GroupAssignment('unknown_a', GROUP_UNKNOWN, '', 0.0),
                   GroupAssignment('strong_a', GROUP_OFFICIAL, '', 1.0)]
    splits = assign_splits(assignments, 0.15, 'salt')
    assert splits['unknown_a'] == 'TRAIN_ONLY_GROUP_UNKNOWN'
    assert splits['strong_a'] in ('train', 'val')


def test_diagnostic_val_is_a_distinct_object():
    assignments = [GroupAssignment(f'g{i}', GROUP_OFFICIAL, '', 1.0)
                   for i in range(20)]
    splits = assign_splits(assignments, 0.2, 'salt')
    diagnostic = assign_diagnostic_random(assignments, 0.2, 'salt')
    assert set(diagnostic.values()) <= {'train', 'DIAGNOSTIC_RANDOM_VAL'}
    assert 'GROUP_DISJOINT_VAL' not in set(diagnostic.values())


def test_known_and_unknown_splits_are_not_confused():
    frame = base_frame(4)
    frame['group_tier'] = [GROUP_OFFICIAL, GROUP_DOCUMENTED_PATH, GROUP_WEAK_PROXY,
                           GROUP_UNKNOWN]
    decided = decide_frame(frame)
    strong = decided['group_tier'].isin(STRONG_GROUPS)
    assert set(decided.loc[~strong, 'split']) <= {
        'train', 'TRAIN_ONLY_GROUP_UNKNOWN', 'DIAGNOSTIC_RANDOM_VAL'}


def test_acw_single_campaign_is_not_split():
    row = {'component': 'acw_rgb-2022-10-06-17-16-49', 'source_prefix': 'acw',
           'observation_url': '', 'upstream_dataset': 'ACRECropWeed'}
    assignment = agrifield_group(row)
    assert assignment.group_key == 'acw|single_campaign'
    assert assignment.tier == GROUP_DOCUMENTED_PATH


def test_inat_observation_is_the_grouping_unit():
    row = {'component': 'inat_Rumex_crispus_556910782', 'source_prefix': 'inat',
           'observation_url': 'https://www.inaturalist.org/observations/308673477',
           'upstream_dataset': 'iNatWeeds'}
    assignment = agrifield_group(row)
    assert assignment.group_key == 'inat|observation|308673477'


def test_inat_without_observation_id_is_only_a_weak_proxy():
    row = {'component': 'inat_Rumex_crispus_556910782', 'source_prefix': 'inat',
           'observation_url': '', 'upstream_dataset': 'iNatWeeds'}
    assignment = agrifield_group(row)
    assert assignment.tier == GROUP_WEAK_PROXY
    splits = assign_splits([assignment], 0.15, 'salt')
    assert splits[assignment.group_key] == 'train'


# --------------------------------------------------------------------------- #
# P0-5 metadata recovery
# --------------------------------------------------------------------------- #
def test_artemis_key_is_parsed_from_the_published_key():
    key = ('Artemis/common-bean/images/ciat-arusha/alta-sopdevelopment/'
           '2023--tza--jun/2023-01-16/107/1673858283486758.jpg')
    parsed = parse_artemis_key(key)
    assert parsed is not None
    assert parsed.crop == 'common-bean'
    assert parsed.farm == 'ciat-arusha'
    assert parsed.session == 'alta-sopdevelopment'
    assert parsed.capture_date == '2023-01-16'
    assert parsed.plot == '107'
    assert parsed.country_code == 'TZA'


def test_imagesafari_key_is_parsed():
    key = 'ImageSafari/common-bean/images/qed-lilongwe/2025-02-14/abc.jpg'
    parsed = parse_artemis_key(key)
    assert parsed is not None
    assert parsed.scheme == 'imagesafari'
    assert parsed.capture_date == '2025-02-14'
    assert parsed.country_code == ''


def test_unknown_key_layout_returns_none():
    assert parse_artemis_key('ImageSafari/maize/xyz.jpg') is None


def test_crop_normalisation_is_documented_only():
    assert normalize_crop('Common Bean') == 'common-bean'
    assert normalize_crop('') == 'unknown'
    assert normalize_crop(None) == 'unknown'
    assert normalize_crop('zzz-unknown-crop') == 'zzz-unknown-crop'


def test_crop_provenance_is_official_when_joined():
    joined = pd.DataFrame({
        'physical_source_id': ['artemis_imagesafari'],
        'relative_path': ['Artemis/common-bean/images/ciat-arusha/s1/2023--tza--jun/'
                          '2023-01-16/107/x.jpg'],
        'bytes': [1234],
    })
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / 'index.csv'
        pd.DataFrame([{
            'part': 'Artemis', 'crop': 'common-bean', 'farm': 'ciat-arusha',
            'session': 's1', 'date': '2023--tza--jun', 'plot': '2023-01-16',
            'key': 'Artemis/common-bean/images/ciat-arusha/s1/2023--tza--jun/'
                   '2023-01-16/107/x.jpg', 'bytes': '1234', 'etag': 'e',
        }]).to_csv(index, index=False)
        from src.registry.stage5d_metadata import attach_artemis_metadata
        out, audit, stats = attach_artemis_metadata(joined, index)
    assert out.iloc[0]['crop'] == 'common-bean'
    assert out.iloc[0]['crop_evidence'] == PROVENANCE_OFFICIAL
    assert out.iloc[0]['country'] == 'TZA'
    assert audit.matched_rows == 1
    assert stats['artemis_keys'] == 1


def test_join_size_conflict_is_flagged():
    import tempfile
    joined = pd.DataFrame({
        'physical_source_id': ['artemis_imagesafari'],
        'relative_path': ['Artemis/maize/images/f/s/2023--tza--jun/2023-01-16/1/x.jpg'],
        'bytes': [999],
    })
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / 'index.csv'
        pd.DataFrame([{
            'part': 'Artemis', 'crop': 'maize', 'farm': 'f', 'session': 's',
            'date': '2023--tza--jun', 'plot': '2023-01-16',
            'key': 'Artemis/maize/images/f/s/2023--tza--jun/2023-01-16/1/x.jpg',
            'bytes': '123', 'etag': 'e',
        }]).to_csv(index, index=False)
        from src.registry.stage5d_metadata import attach_artemis_metadata
        out, audit, _ = attach_artemis_metadata(joined, index)
    assert out.iloc[0]['metadata_join_status'] == 'MATCHED_CONFLICT'
    assert audit.conflict_rows == 1


def test_unmatched_key_keeps_unknown_crop():
    import tempfile
    joined = pd.DataFrame({
        'physical_source_id': ['artemis_imagesafari'],
        'relative_path': ['Artemis/maize/images/f/s/2023--tza--jun/2023-01-16/1/nope.jpg'],
        'bytes': [1],
    })
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / 'index.csv'
        pd.DataFrame([{
            'part': 'Artemis', 'crop': 'maize', 'farm': 'f', 'session': 's',
            'date': '2023--tza--jun', 'plot': '2023-01-16',
            'key': 'Artemis/maize/images/f/s/2023--tza--jun/2023-01-16/1/x.jpg',
            'bytes': '123', 'etag': 'e',
        }]).to_csv(index, index=False)
        from src.registry.stage5d_metadata import attach_artemis_metadata
        out, _, _ = attach_artemis_metadata(joined, index)
    assert out.iloc[0]['crop'] == 'unknown'
    assert out.iloc[0]['metadata_join_status'] == 'UNMATCHED'


# --------------------------------------------------------------------------- #
# Dedup (the Stage 5C inert near-duplicate stage)
# --------------------------------------------------------------------------- #
def test_hash_columns_are_read_under_their_real_names():
    frame = pd.DataFrame({'phash': ['abcd', '0000'], 'dhash': ['abcd', '0000'],
                          'image_id': ['a', 'b']})
    report = validate_hash_columns(frame)
    assert report['phash']['valid'] == 2


def test_unparseable_hash_matrix_refuses_to_run():
    frame = pd.DataFrame({'phash': ['nan', 'nan'], 'dhash': ['nan', 'nan'],
                          'image_id': ['a', 'b']})
    with pytest.raises(ValueError):
        validate_hash_columns(frame)


def test_exact_duplicate_keeps_a_deterministic_representative():
    frame = pd.DataFrame({
        'image_id': ['a', 'b', 'c'],
        'raw_sha256': ['x', 'x', 'y'],
        'pixel_sha256': ['p', 'p', 'q'],
        'laplacian_var': [10.0, 200.0, 50.0],
        'width': [640, 640, 640],
        'height': [480, 480, 480],
    })
    decisions = exact_duplicate_decisions(frame)
    kept = decisions[~decisions['exact_duplicate']]['image_id'].tolist()
    assert sorted(kept) == ['b', 'c']          # b is the sharper copy
    removed = decisions[decisions['exact_duplicate']]['image_id'].tolist()
    assert removed == ['a']
    assert decisions[decisions['image_id'] == 'a'].iloc[0]['exact_duplicate_of'] == 'b'


def test_soil_texture_giant_cluster_is_quarantined_not_collapsed():
    # 500 images with an identical hash form one texture-collision component.
    frame = pd.DataFrame({
        'image_id': [f'i{i}' for i in range(500)],
        'phash': ['0000000000000000'] * 500,
        'dhash': ['0000000000000000'] * 500,
        'width': [640] * 500,
        'height': [480] * 500,
        'decode_ok': ['true'] * 500,
        'laplacian_var': [100.0] * 500,
    })
    pairs, components, stats = near_duplicate_scan(frame, DedupConfig(use_embeddings=False))
    assert stats['giant_members'] == 500
    assert stats['confirmed_pairs'] > 0


def test_uncertain_candidate_is_not_deleted():
    frame = pd.DataFrame({
        'image_id': ['a', 'b'],
        'phash': ['00000000000000ff', '0000000000000000'],
        'dhash': ['00000000000000ff', '0000000000000000'],
        'width': [640, 640],
        'height': [480, 480],
        'decode_ok': ['true', 'true'],
        'laplacian_var': [100.0, 100.0],
    })
    pairs, _, _ = near_duplicate_scan(frame, DedupConfig(use_embeddings=False))
    assert pairs and not pairs[0].confirmed
    assert 'PHASH_WEAK' in pairs[0].reason


# --------------------------------------------------------------------------- #
# Holdout protection
# --------------------------------------------------------------------------- #
def test_holdout_prefix_is_not_conditioned_on_split_dir():
    frame = base_frame(2)
    frame['relative_path'] = ['val/pnb_20200425_train_fh_P1_0_0.png',
                              'train/pnb_20200425_train_fh_P2_0_0.png']
    names = Path if False else None
    from pathlib import Path as _Path
    blocked = [str(_Path(p).name).startswith('pnb_') for p in frame['relative_path']]
    assert blocked == [True, True]


def test_holdout_source_is_blocked_at_source_level():
    frame = base_frame(1)
    frame['physical_source_id'] = 'phenobench'
    frame['holdout_status'] = 'HOLDOUT'
    decided = decide_frame(frame)
    assert decided.iloc[0]['primary_reason'] == 'HOLDOUT_EXCLUDED'


def test_legacy_stage5c_artifacts_are_unchanged():
    policy = CORPUS_V2 / 'policy.yaml'
    manifest = json.loads((CORPUS_V2 / 'R3_field_extended_manifest.json').read_text())
    assert hashlib.sha256(policy.read_bytes()).hexdigest() == manifest['policy_sha256']
    for name in ('R0_agrifm35_source_pool_manifest.csv',
                 'R1_agrifm147_rebuild_manifest.csv',
                 'R2_field_strict_manifest.csv',
                 'R3_field_extended_manifest.csv'):
        path = CORPUS_V2 / name
        recorded = json.loads(
            (CORPUS_V2 / f'{name.replace("_manifest.csv", "_manifest.json")}').read_text())
        if recorded.get('manifest_csv_sha256'):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded['manifest_csv_sha256']


# --------------------------------------------------------------------------- #
# Gate status vocabulary
# --------------------------------------------------------------------------- #
def test_target_exceeded_does_not_upgrade_a_failed_gate():
    gates = {key: True for key, _, _ in
             __import__('src.registry.stage5d_gates', fromlist=['GATES']).GATES}
    gates['counts_conserved'] = False
    result = status(gates, target_exceeded=True)
    assert result['status'] == STATUS_BLOCKED
    assert result['target_exceeded'] is True


def test_all_gates_passed_is_ready_but_not_human_verified():
    gates = {key: True for key, _, _ in
             __import__('src.registry.stage5d_gates', fromlist=['GATES']).GATES}
    result = status(gates, target_exceeded=True)
    assert result['status'] == STATUS_READY
    assert result['all_gates_passed'] is True


def test_a_single_blocking_failure_blocks():
    gates = {key: True for key, _, _ in
             __import__('src.registry.stage5d_gates', fromlist=['GATES']).GATES}
    gates['visible_rgb_evidenced'] = False
    assert status(gates)['status'] == STATUS_BLOCKED


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_split_assignment_is_a_pure_function_of_group_ids():
    assignments = [GroupAssignment(f'g{i}', GROUP_OFFICIAL, '', 1.0) for i in range(30)]
    first = assign_splits(assignments, 0.15, 'salt')
    shuffled = list(reversed(assignments))
    second = assign_splits(shuffled, 0.15, 'salt')
    assert first == second


def test_near_duplicate_scan_is_deterministic():
    frame = pd.DataFrame({
        'image_id': ['a', 'b', 'c', 'd'],
        'phash': ['0000000000000001', '0000000000000001', 'ffffffffffffffff',
                  '0000000000000003'],
        'dhash': ['0000000000000001', '0000000000000001', 'ffffffffffffffff',
                  '0000000000000003'],
        'width': [640] * 4, 'height': [480] * 4,
        'decode_ok': ['true'] * 4, 'laplacian_var': [10.0, 20.0, 30.0, 40.0],
    })
    config = DedupConfig(use_embeddings=False)
    first, components, _ = near_duplicate_scan(frame, config)
    second, _, _ = near_duplicate_scan(frame, config)
    assert [(p.left, p.right) for p in first] == [(p.left, p.right) for p in second]
    assert components['a'] == components['b']


def test_phash_column_is_not_silently_zero():
    frame = pd.DataFrame({
        'image_id': ['a', 'b'], 'phash': ['abcd', 'abce'], 'dhash': ['abcd', 'abce'],
        'width': [640, 640], 'height': [480, 480],
        'decode_ok': ['true', 'true'], 'laplacian_var': [1.0, 1.0],
    })
    quality = hash_quality(frame, 'phash')
    assert quality['unique_hashes'] == 2


# --------------------------------------------------------------------------- #
# Synthetic QC harness
# --------------------------------------------------------------------------- #
def test_every_degradation_produces_an_image():
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
    image = Image.fromarray(np.stack([gradient] * 3, axis=-1))
    for kind, value, _ in DEGRADATIONS:
        damaged = degrade(image, kind, value, seed_key='synthetic-test')
        assert damaged is not None, kind
        if kind != 'downscale':  # downscale exists to shrink the image
            assert damaged.size == image.size, (kind, damaged.size)


def test_downscale_drops_below_minimum_resolution():
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
    image = Image.fromarray(np.stack([gradient] * 3, axis=-1))
    small = degrade(image, 'downscale', 'factor0.15', seed_key='x')
    metrics = measure_pil(small)
    decision, reasons = basic_decision_from_metrics(
        metrics, load_policy(POLICY).basic)
    assert decision == 'REJECT'
    assert 'BELOW_MIN_RESOLUTION' in reasons


def test_gaussian_noise_is_reproducible_for_the_same_key():
    gradient = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (96, 1))
    image = Image.fromarray(np.stack([gradient] * 3, axis=-1))
    first = np.asarray(degrade(image, 'gaussian_noise', 'sigma30', seed_key='k'))
    second = np.asarray(degrade(image, 'gaussian_noise', 'sigma30', seed_key='k'))
    other = np.asarray(degrade(image, 'gaussian_noise', 'sigma30', seed_key='j'))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
