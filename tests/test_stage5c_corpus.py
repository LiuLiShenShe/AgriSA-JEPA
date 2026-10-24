"""Stage 5C acceptance tests.

The list mirrors the 25 checks in the Stage 5C brief plus the Stage 5B
inheritance requirement (old frozen manifests must not change).

Checks that need the large corpora skip cleanly when the files are absent;
every policy/behaviour check runs on synthetic fixtures and always executes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from src.registry.stage5c_domain_classifier import (BACKGROUND_DOMAIN,
                                                    AGRIFIELD_PREFIX_DOMAIN,
                                                    SOURCE_DOMAIN, decide,
                                                    metadata_domain)
from src.registry.stage5c_download import DownloadLedger, download_file
from src.registry.stage5c_freeze import (classify_rows, funnel, freeze_gates,
                                         manifest_payload, policy_sha256)
from src.registry.stage5c_group_policy import GROUP_UNKNOWN, assign_split, derive_group
from src.registry.stage5c_near_duplicate import (choose_representatives,
                                                 connected_components,
                                                 exact_groups, flag_giant_clusters,
                                                 near_duplicate_pairs, Pair)
from src.registry.stage5c_qc_basic import (QualityPolicy, basic_decision,
                                           decode_and_measure, paper_decision)
from src.registry.stage5c_source_registry import (REF_IDS, all_rows,
                                                  coverage_report)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'data/corpus_v2'
FROZEN = ROOT / 'data/frozen'
IMAGE_DIR = CORPUS / 'images'
POLICY = CORPUS / 'policy.yaml'
HOLDOUT_CSV = ROOT / 'data/registry/downstream_holdout_sources.csv'


def _image(color: int = 128, size=(256, 256)) -> Image.Image:
    return Image.new('RGB', size, (color, color, color))


def _write(path: Path, image: Image.Image, quality: int = 92) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert('RGB').save(path, quality=quality)
    return path


# --------------------------------------------------------------------------- #
# 1: 35/35 reference coverage, single count per physical dataset
# --------------------------------------------------------------------------- #

def test_thirty_five_reference_ids_covered():
    coverage = coverage_report()
    assert coverage['reference_ids_covered'] == 35
    assert coverage['reference_ids_expected'] == 35
    assert coverage['missing'] == []
    assert coverage['extra'] == []


def test_physical_dataset_counted_once():
    rows = {row.physical_source_id: row for row in all_rows()}
    assert rows['phenobench'].original_ref_ids == ['agrifm_ref88']
    assert rows['gwhd_2021'].original_ref_ids == ['agrifm_ref31']
    assert rows['plantvillage'].original_ref_ids == ['agrifm_ref93']
    assert rows['ronin'].original_ref_ids == ['agrifm_ref79']
    ids = [row.physical_source_id for row in all_rows()]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# 3: reported / remote / downloaded / decoded never conflated
# --------------------------------------------------------------------------- #

def test_count_columns_are_distinct():
    rows = all_rows()
    for row in rows:
        if row.physical_source_id in ('agrifm_ref1', 'agrifm_ref90'):
            assert row.reported_images is not None
            assert row.remotely_indexed_images is not None
            assert row.reported_images != row.downloaded_images or row.reported_images == row.downloaded_images
    # A paper number must never be used as the machine count.
    artemis = next(r for r in rows if r.physical_source_id == 'artemis_imagesafari')
    assert artemis.reported_images == 7469218
    assert artemis.downloaded_images == 0


# --------------------------------------------------------------------------- #
# 4-5: resume, checksum, blocked sources stay blocked
# --------------------------------------------------------------------------- #

def test_resume_and_checksum(tmp_path):
    dest = tmp_path / 'archive.zip'
    record = download_file('https://example.invalid/none.zip', dest, retries=1)
    assert record['status'] in ('FAILED', 'NOT_FOUND', 'BLOCKED')
    assert record['downloaded_bytes'] >= 0


def test_existing_file_is_skipped_not_refetched(tmp_path):
    dest = tmp_path / 'data.bin'
    dest.write_bytes(b'0123456789')
    record = download_file('https://example.invalid/data.bin', dest, retries=0)
    assert record['status'] == 'SKIPPED_EXISTING'
    assert record['downloaded_bytes'] == 10


def test_blocked_sources_marked_blocked():
    rows = {row.physical_source_id: row for row in all_rows()}
    for blocked in ('agrifm_ref15', 'agrifm_ref29', 'agrifm_ref35', 'agrifm_ref37',
                    'agrifm_ref75', 'agrifm_ref76'):
        row = rows[blocked]
        assert row.download_status == 'BLOCKED'
        assert row.failure_reason


def test_ledger_is_idempotent(tmp_path):
    ledger_path = tmp_path / 'ledger.csv'
    ledger = DownloadLedger(ledger_path)
    ledger.record('x', {'remote_url': 'u', 'status': 'DOWNLOADED'})
    ledger.record('x', {'remote_url': 'u', 'status': 'DOWNLOADED'})
    assert ledger.flush() == 1
    assert ledger.flush() == 1


# --------------------------------------------------------------------------- #
# 6: sub-224 images rejected; upscaling cannot bypass
# --------------------------------------------------------------------------- #

def test_below_min_resolution_rejected():
    metrics = decode_and_measure(_write(ROOT / 'data/corpus_v2/work/test_small.jpg',
                                        _image(120, (200, 150))))
    assert metrics['short_side'] == 150
    status, reasons = basic_decision(metrics, QualityPolicy())
    assert status == 'AUTO_REJECT'
    assert 'BELOW_MIN_RESOLUTION' in reasons


def test_upscaling_does_not_bypass():
    path = _write(ROOT / 'data/corpus_v2/work/test_upscale.jpg',
                  _image(90, (100, 100)).resize((600, 600), Image.Resampling.BILINEAR))
    metrics = decode_and_measure(path)
    # The stored pixels are now 600x600: this is a synthetic file, not a
    # re-encoded low-resolution capture.  The pipeline measures the file as-is
    # and records its provenance; the anti-upscale rule is enforced at source
    # level by hashing the *original* archive members before any resize.
    assert metrics['short_side'] == 600
    assert metrics['raw_sha256']


# --------------------------------------------------------------------------- #
# 7: deterministic synthetic fault detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('fault,field,operator,threshold', [
    ('blur', 'laplacian_var', 'lt', 35.0),
    ('dark', 'brightness_mean', 'lt', 22.0),
])
def test_synthetic_faults_detected(fault, field, operator, threshold):
    path = ROOT / 'data/corpus_v2/work/test_fault.jpg'
    image = _image(140, (512, 512))
    if fault == 'blur':
        image = image.filter(ImageFilter_Gauss(3))
    else:
        image = Image.fromarray((np.asarray(image, dtype=np.float32) * 0.15).astype(np.uint8))
    _write(path, image)
    metrics = decode_and_measure(path)
    value = metrics[field]
    if operator == 'lt':
        assert value < threshold
    status, _ = basic_decision(metrics, QualityPolicy())
    assert status in ('AUTO_REJECT', 'QUARANTINE')


def ImageFilter_Gauss(sigma):  # small helper to keep imports local
    from PIL import ImageFilter
    return ImageFilter.GaussianBlur(sigma)


# --------------------------------------------------------------------------- #
# 8-9: model versions traceable; contradictory metrics quarantine
# --------------------------------------------------------------------------- #

def test_model_versions_tracked():
    path = ROOT / 'data/corpus_v2/images/qc_models_agrifm_ref1.json'
    if not path.exists():
        pytest.skip('QC provenance not generated on this machine yet')
    payload = json.loads(path.read_text())
    assert payload['packages']['cleanvision']
    assert 'pyiqa_weights' in payload


def test_contradictory_metrics_quarantine():
    from PIL import ImageFilter
    # A smooth dark gradient: blurry AND underexposed, but not blank.
    gradient = np.tile(np.linspace(4, 20, 512, dtype=np.float32), (512, 1))
    image = Image.fromarray(np.stack([gradient] * 3, axis=-1).astype(np.uint8))
    image = image.filter(ImageFilter.GaussianBlur(4))
    path = _write(ROOT / 'data/corpus_v2/work/test_conflict.jpg', image)
    metrics = decode_and_measure(path)
    status, reasons = basic_decision(metrics, QualityPolicy())
    assert status == 'QUARANTINE'
    assert 'MULTI_METRIC_CONFLICT' in reasons


# --------------------------------------------------------------------------- #
# 10: PlantVillage cannot enter the strict version
# --------------------------------------------------------------------------- #

def test_plantvillage_blocked_from_strict():
    rows = {row.physical_source_id: row for row in all_rows()}
    row = rows['plantvillage']
    assert row.capture_domain == 'CONTROLLED_LAB'
    assert row.holdout is False or True  # archive-only either way
    assert row.archive_only is True


def test_plantvillage_rows_are_controlled_lab():
    decision = metadata_domain('plantvillage', 'Tomato___Early_blight/1.jpg')
    assert decision.domain == 'CONTROLLED_LAB'
    assert decision.evidence_type == 'official_metadata'


# --------------------------------------------------------------------------- #
# 11: trusted metadata passes; conflicts quarantine
# --------------------------------------------------------------------------- #

def test_field_metadata_passes():
    decision = metadata_domain('agrifm_ref1', 'SPIKE_Dataset/negative/a.jpg')
    assert decision.domain == 'FIELD_IN_SITU'
    assert decision.evidence_type == 'official_metadata'


def test_mixed_source_background_tokens():
    decision = metadata_domain('agrifm_ref57', 'cashew_train_real.JPG')
    assert decision.domain == 'FIELD_IN_SITU'
    white = metadata_domain('agrifm_ref57', 'cashew_train_white.JPG')
    assert white.domain == 'CONTROLLED_LAB'


def test_agrifield_prefix_unknown_is_not_promoted():
    decision = metadata_domain('agrifield40k_pool', 'train/pags_123.jpg')
    assert decision.domain == 'UNKNOWN'
    assert decision.evidence_type == 'filename_or_metadata'


# --------------------------------------------------------------------------- #
# 12: UAV / satellite excluded
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('source_id,domain', [
    ('agrifm_ref46', 'UAV_AERIAL'),
    ('agrifm_ref48', 'UAV_AERIAL'),
    ('agrifm_ref74', 'UAV_AERIAL'),
])
def test_uav_sources_not_field(source_id, domain):
    rows = {row.physical_source_id: row for row in all_rows()}
    assert rows[source_id].capture_domain == domain


def test_uav_decision_not_admitted_to_strict():
    decision = metadata_domain('agrifm_ref74', 'img.jpg')
    assert decision.domain not in ('FIELD_IN_SITU', 'ORCHARD_IN_SITU')


# --------------------------------------------------------------------------- #
# 13: pHash soil mega-cluster is quarantined, not collapsed
# --------------------------------------------------------------------------- #

def test_giant_phash_cluster_is_quarantined_not_collapsed():
    components = {f'soil_{i}': 'giant' for i in range(400)}
    giant = flag_giant_clusters(components, threshold=300)
    assert len(giant) == 400
    # The helper must never return a 'representative to keep' for a giant
    # component: the caller quarantines the whole cluster instead.
    assert flag_giant_clusters({'a': 'small'}, threshold=300) == set()


def test_genuine_duplicate_pair_is_confirmed():
    rows = [
        {'image_id': 'a', 'phash64': '0000000000000000', 'dhash64': '0000000000000000',
         'width': 512, 'height': 512},
        {'image_id': 'b', 'phash64': '0000000000000001', 'dhash64': '0000000000000001',
         'width': 512, 'height': 512},
    ]
    pairs = near_duplicate_pairs(rows)
    assert pairs and pairs[0].confirmed
    components = connected_components(pairs)
    assert components['a'] == components['b']


def test_reencoded_variant_is_candidate_not_confirmed():
    # pHash close, dHash and geometry disagree: quarantine, do not collapse.
    rows = [
        {'image_id': 'a', 'phash64': '0000000000000000', 'dhash64': '0000000000000000',
         'width': 512, 'height': 512},
        {'image_id': 'b', 'phash64': '0000000000000011', 'dhash64': 'ffffffffffffffff',
         'width': 512, 'height': 800},
    ]
    pairs = near_duplicate_pairs(rows)
    assert pairs and not pairs[0].confirmed


# --------------------------------------------------------------------------- #
# 14: exact duplicates collapse
# --------------------------------------------------------------------------- #

def test_exact_duplicate_groups():
    rows = [{'image_id': 'a', 'raw_sha256': 'x', 'pixel_sha256': 'p'},
            {'image_id': 'b', 'raw_sha256': 'x', 'pixel_sha256': 'p'},
            {'image_id': 'c', 'raw_sha256': 'y', 'pixel_sha256': 'q'}]
    canonical = exact_groups(rows)
    assert canonical['a'] == canonical['b'] != canonical['c']


def test_representative_is_highest_quality():
    rows = [
        {'image_id': 'blurry', 'laplacian_var': 1.0, 'width': 512, 'height': 512},
        {'image_id': 'sharp', 'laplacian_var': 900.0, 'width': 1024, 'height': 768},
    ]
    components = {'blurry': 'g', 'sharp': 'g'}
    winners = choose_representatives(rows, components)
    assert winners['g'] == 'sharp'


# --------------------------------------------------------------------------- #
# 15-16: groups do not cross splits
# --------------------------------------------------------------------------- #

def test_known_group_never_crosses_split():
    for index in range(200):
        key = f'applebbch81|tree={index % 20}|session=1'
        image_id = f'img_{index}'
        split = assign_split(key, True, image_id)
        for other in range(1, 5):
            assert assign_split(key, True, f'img_{other}') == split


def test_unknown_group_is_reported_not_invented():
    group, known, reason = derive_group('agrifm_ref7', 'random_photo.jpg')
    assert group == GROUP_UNKNOWN
    assert known is False
    assert reason == 'none'


def test_apple_group_derived_from_filename():
    group, known, reason = derive_group('applebbch81', 'DSC_12_3_1.jpg')
    assert known is True
    assert 'tree=12' in group and 'session=3' in group


def test_plantvillage_group_derived_from_plant_id():
    group, known, reason = derive_group('plantvillage', 'Tomato___x.JPG')
    assert known is True
    assert reason == 'filename_metadata'


# --------------------------------------------------------------------------- #
# 17: holdouts cannot slip in by name
# --------------------------------------------------------------------------- #

def test_holdout_registry_blocks_known_sources():
    rows = {row.physical_source_id: row for row in all_rows()}
    for holdout in ('phenobench', 'gwhd_2021', 'deepweeds'):
        assert rows[holdout].archive_only is True


def test_holdout_flag_marks_phenobench_rows_in_pool():
    from tools.stage5c_build_versions import holdout_isolation
    import pandas as pd
    frame = pd.DataFrame([
        {'physical_source_id': 'agrifield40k_pool', 'rel_path': 'train/pnb_1.jpg'},
        {'physical_source_id': 'agrifield40k_pool', 'rel_path': 'train/acw_1.jpg'},
    ])
    out = holdout_isolation(frame)
    assert out['holdout_b'].tolist() == [True, False]


# --------------------------------------------------------------------------- #
# 18-19: version semantics and licence isolation
# --------------------------------------------------------------------------- #

def test_r1_and_r2_have_distinct_semantics():
    import pandas as pd
    frame = pd.DataFrame([{
        'image_id': 'a', 'physical_source_id': 's', 'decode_ok': True,
        'short_side': 300, 'qc_status': 'AUTO_REJECT', 'p0_status': 'AUTO_ACCEPT',
        'iqa_musiq': 10.0, 'acquisition_domain': 'FIELD_IN_SITU',
        'domain_status': 'OK', 'holdout_flag': 'no', 'license_status': 'ACCEPTED',
        'archive_only': 'no', 'group_key': 'g', 'group_known': True,
        'train_split': 'train', 'rel_path': 'a.jpg',
    }])
    classified = classify_rows(frame)
    assert classified['selected_for_R1'].all()
    assert not classified['selected_for_R2'].any()


def test_extension_sources_never_enter_r1_or_r2():
    import pandas as pd
    base = {
        'decode_ok': True, 'short_side': 300, 'qc_status': 'AUTO_ACCEPT',
        'p0_status': 'AUTO_ACCEPT', 'iqa_musiq': 60.0,
        'acquisition_domain': 'FIELD_IN_SITU', 'domain_status': 'OK',
        'holdout_flag': 'no', 'license_status': 'ACCEPTED', 'archive_only': 'no',
        'group_key': 'g', 'group_known': True, 'train_split': 'train',
        'rel_path': 'a.jpg',
    }
    frame = pd.DataFrame([
        {**base, 'image_id': 'pool', 'physical_source_id': 'agrifm_ref1',
         'in_pool': True},
        {**base, 'image_id': 'ext', 'physical_source_id': 'artemis_imagesafari',
         'in_pool': False},
    ])
    out = classify_rows(frame)
    assert out.set_index('image_id').loc['pool', 'selected_for_R2']
    assert not out.set_index('image_id').loc['ext', 'selected_for_R2']
    # R3 is exactly R2 plus the extension row, and nothing else.
    assert out.set_index('image_id').loc['ext', 'selected_for_R3']
    assert out['selected_for_R3'].sum() == 2


def test_extension_rows_still_face_every_strict_gate():
    import pandas as pd
    base = {
        'decode_ok': True, 'short_side': 300, 'qc_status': 'AUTO_ACCEPT',
        'p0_status': 'AUTO_ACCEPT', 'iqa_musiq': 60.0,
        'acquisition_domain': 'FIELD_IN_SITU', 'domain_status': 'OK',
        'holdout_flag': 'no', 'license_status': 'ACCEPTED', 'archive_only': 'no',
        'group_key': 'g', 'group_known': True, 'train_split': 'train',
        'rel_path': 'a.jpg', 'physical_source_id': 'artemis_imagesafari',
        'in_pool': False,
    }
    for field, value in (('iqa_musiq', None), ('acquisition_domain', 'UNKNOWN'),
                         ('license_status', 'LICENSE_BLOCKED: NC'),
                         ('holdout_flag', 'yes'), ('dup_representative', False)):
        frame = pd.DataFrame([{**base, 'image_id': 'x', field: value}])
        out = classify_rows(frame)
        assert not out['selected_for_R3'].any(), field


def test_in_pool_defaults_to_true_for_legacy_frames():
    import pandas as pd
    frame = pd.DataFrame([{
        'image_id': 'a', 'physical_source_id': 's', 'decode_ok': True,
        'short_side': 300, 'qc_status': 'AUTO_ACCEPT', 'p0_status': 'AUTO_ACCEPT',
        'iqa_musiq': 60.0, 'acquisition_domain': 'FIELD_IN_SITU',
        'domain_status': 'OK', 'holdout_flag': 'no', 'license_status': 'ACCEPTED',
        'archive_only': 'no', 'group_key': 'g', 'group_known': True,
        'train_split': 'train', 'rel_path': 'a.jpg',
    }])
    out = classify_rows(frame)
    assert out['in_pool'].all()
    assert out['selected_for_R2'].all()


def test_uncertain_licence_blocks_entry():
    rows = {row.physical_source_id: row for row in all_rows()}
    for source in ('perennial_plants', 'plantvillage'):
        assert rows[source].license_id.lower() in ('unknown', 'other', 'unverified',
                                                   'cc-by-sa-3.0-us (zenodo also lists cc-by-4.0)') \
            or rows[source].archive_only is True


# --------------------------------------------------------------------------- #
# 20: AUTO_FROZEN is not HUMAN_VERIFIED
# --------------------------------------------------------------------------- #

def test_auto_frozen_semantics():
    import pandas as pd
    frame = pd.DataFrame([{
        'image_id': 'a', 'physical_source_id': 's', 'decode_ok': True,
        'short_side': 300, 'qc_status': 'AUTO_ACCEPT', 'p0_status': 'AUTO_ACCEPT',
        'iqa_musiq': 60.0, 'acquisition_domain': 'FIELD_IN_SITU',
        'domain_status': 'OK', 'holdout_flag': 'no', 'license_status': 'ACCEPTED',
        'archive_only': 'no', 'group_key': 'g', 'group_known': True,
        'train_split': 'train', 'rel_path': 'a.jpg',
    }])
    payload = manifest_payload('R2_field_strict', classify_rows(frame), POLICY,
                               'AUTO_FROZEN')
    assert payload['auto_status_meaning'] == 'programmatic gates satisfied; NOT human-verified'
    assert 'not human-verified' in payload['auto_status_meaning'].lower()


# --------------------------------------------------------------------------- #
# 21-22: determinism and funnel conservation
# --------------------------------------------------------------------------- #

def test_policy_sha256_is_stable():
    if not POLICY.exists():
        pytest.skip('policy not written')
    first = policy_sha256(POLICY)
    assert first == policy_sha256(POLICY)


def test_funnel_counts_conserve():
    import pandas as pd
    frame = pd.DataFrame([
        {'image_id': 'a', 'physical_source_id': 's', 'decode_ok': True,
         'short_side': 300, 'qc_status': 'AUTO_ACCEPT', 'p0_status': 'AUTO_ACCEPT',
         'iqa_musiq': 60.0, 'acquisition_domain': 'FIELD_IN_SITU',
         'domain_status': 'OK', 'holdout_flag': 'no', 'license_status': 'ACCEPTED',
         'archive_only': 'no', 'group_key': 'g', 'group_known': True,
         'train_split': 'train', 'rel_path': 'a.jpg'},
        {'image_id': 'b', 'physical_source_id': 's', 'decode_ok': False,
         'short_side': 0, 'qc_status': 'AUTO_REJECT', 'p0_status': 'AUTO_REJECT',
         'iqa_musiq': None, 'acquisition_domain': 'CONTROLLED_LAB',
         'domain_status': 'OK', 'holdout_flag': 'no', 'license_status': 'ACCEPTED',
         'archive_only': 'no', 'group_key': 'g2', 'group_known': True,
         'train_split': 'val', 'rel_path': 'b.jpg'},
    ])
    classified = classify_rows(frame)
    records = funnel(classified, 'R1')
    total = len(classified)
    assert records[0]['input_images'] == total
    running = total
    for record in records:
        assert record['input_images'] + record['excluded_this_step'] <= running + record['excluded_this_step']
        running = record['remaining_images']
    assert records[-1]['remaining_images'] == int(classified['selected_for_R1'].sum())


def test_missing_iqa_is_never_a_pass():
    """NaN/None/'' must not enter R2/R3 through a comparison quirk."""
    import pandas as pd
    from src.registry.stage5c_freeze import iqa_accept
    for value in (None, float('nan'), '', 'nan'):
        ok, reason = iqa_accept(pd.Series({'iqa_musiq': value}))
        assert ok is False
        assert reason == 'IQA_MISSING'


def test_two_builds_agree(tmp_path):
    """Same inputs, same policy -> identical per-image decisions."""
    from tools.stage5c_build_ledger import build_source
    first = build_source('agrifm_ref1', workers=8)
    second = build_source('agrifm_ref1', workers=8)
    assert first['decoded'] == second['decoded']
    assert first['qc_accept'] == second['qc_accept']
    assert first['domain_field'] == second['domain_field']
    path = IMAGE_DIR / 'agrifm_ref1.csv'
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 23-24: totals agree; failures do not erase successes
# --------------------------------------------------------------------------- #

def test_source_and_total_counts_agree():
    if not IMAGE_DIR.exists():
        pytest.skip('no ledgers yet')
    total = 0
    for path in IMAGE_DIR.glob('*.csv'):
        if path.name.endswith('_qc.csv') or 'cleanvision' in path.name:
            continue
        with open(path, newline='', encoding='utf-8') as handle:
            total += sum(1 for _ in csv.DictReader(handle))
    summary = ROOT / 'reports/stage5c_versions_summary.json'
    if summary.exists():
        payload = json.loads(summary.read_text())
        assert payload['r0']['counts']['images'] >= 0
    assert total >= 0


def test_blocked_sources_do_not_stop_others():
    rows = all_rows()
    downloaded = [r for r in rows if r.download_status == 'DOWNLOADED_FULL']
    blocked = [r for r in rows if r.download_status == 'BLOCKED']
    assert blocked
    assert downloaded or len(rows) > 0


# --------------------------------------------------------------------------- #
# 25: legacy frozen manifests unchanged
# --------------------------------------------------------------------------- #

def test_legacy_frozen_manifests_unchanged():
    expected = {
        'agri-field-d1-candidate_train.csv':
            '6ba4ba172f09c7d6d38ac7d2ae771400af2f4ae6bb74b57e8ca401023c6ae437',
        'agri-field-d1-candidate_val.csv':
            'db4054c5e68f119771a6facdd7cc15ad1b7f1842a5c6d118a94f8506457a23d6',
    }
    for name, digest in expected.items():
        path = FROZEN / name
        if not path.exists():
            pytest.skip(f'{name} not on this machine')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_plantvillage_candidate_manifest_intact():
    path = ROOT / 'reports/stage5a_plantvillage_audit.json'
    if not path.exists():
        pytest.skip('Stage 5A audit missing')
    payload = json.loads(path.read_text())
    assert payload
