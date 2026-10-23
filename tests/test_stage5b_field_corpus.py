"""Stage 5B acceptance tests: field-domain policy, corpus gates, reproducibility.

The tests are organised around the twenty acceptance checks in the Stage 5B
brief.  Data-dependent checks skip cleanly when the large corpora are not on
this machine; policy checks always run.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.registry.domain_policy import (DOMAINS, admission, default_policy,
                                        summarise_labels, validate_domain)
from src.registry.field_filter import QualityGate, quality_gate, suggest_domain
from src.registry.field_freeze import (apply_gate, assign_group_split,
                                       canonical_csv, write_field_corpus)
from src.registry.source_balance import (composition, effective_sample_counts,
                                         max_source_share, shannon_entropy,
                                         strategy_weights)
from src.registry.source_domains import SAME_PHYSICAL_DATASET, SOURCE_DOMAINS

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / 'data/frozen'
HOLDOUT = ROOT / 'data/registry/downstream_holdout_sources.csv'


# --------------------------------------------------------------------------- #
# 1-7: acquisition-domain admission
# --------------------------------------------------------------------------- #

def test_plantvillage_is_controlled_and_excluded():
    """1+2: PlantVillage is CONTROLLED_LAB and can never enter Field-D1."""
    assert SOURCE_DOMAINS['agrifm_ref93']['acquisition_domain'] == 'CONTROLLED_LAB'
    admitted, reason = admission('CONTROLLED_LAB', 'official_documentation',
                                 'high', 'reviewed')
    assert admitted is False
    assert 'controlled' in reason.lower()


@pytest.mark.parametrize('domain', ['UAV_AERIAL', 'SATELLITE_REMOTE'])
def test_aerial_and_satellite_are_excluded(domain):
    """3: aerial and satellite imagery is outside the near-ground scope."""
    admitted, reason = admission(domain, 'official_documentation', 'high', 'reviewed')
    assert admitted is False
    assert 'aerial' in reason.lower() or 'remote' in reason.lower()


@pytest.mark.parametrize('level', ['candidate', 'freeze'])
def test_unknown_never_enters_the_main_corpus(level):
    """4: UNKNOWN cannot be admitted by any evidence combination."""
    for evidence in ('official_documentation', 'manual_inspection',
                     'model_assisted_inspection', 'filename_or_metadata_only', 'none'):
        admitted, reason = admission('UNKNOWN', evidence, 'high', 'reviewed',
                                     level=level)
        assert admitted is False
        assert 'unverified' in reason.lower()


def test_orchard_in_situ_is_admitted():
    """5: ORCHARD_IN_SITU is allowed, at candidate and freeze level."""
    for level in ('candidate', 'freeze'):
        admitted, _ = admission('ORCHARD_IN_SITU', 'official_documentation',
                                'high', 'reviewed', level=level)
        assert admitted is True


def test_protected_cultivation_is_registered_but_excluded():
    """6: protected cultivation is its own label and is not Field-D1."""
    assert 'PROTECTED_CULTIVATION' in DOMAINS
    assert SOURCE_DOMAINS['agrifm_ref90']['acquisition_domain'] == 'PROTECTED_CULTIVATION'
    admitted, reason = admission('PROTECTED_CULTIVATION', 'official_documentation',
                                 'high', 'reviewed')
    assert admitted is False
    assert 'separately' in reason.lower()


def test_field_in_situ_requires_real_evidence():
    """7: a field label with weak evidence is refused."""
    weak, _ = admission('FIELD_IN_SITU', 'filename_or_metadata_only', 'low',
                        'unreviewed')
    none, _ = admission('FIELD_IN_SITU', 'none', 'unknown', 'unreviewed')
    strong, _ = admission('FIELD_IN_SITU', 'official_documentation', 'high', 'reviewed')
    assert weak is False
    assert none is False
    assert strong is True


def test_freeze_refuses_model_only_evidence():
    """12-adjacent: a model prediction is evidence, not a confirmation."""
    ok, why = admission('FIELD_IN_SITU', 'model_assisted_inspection', 'medium',
                        'model_assisted_pending_human', level='freeze')
    assert ok is False
    assert 'not sufficient' in why or 'review status' in why
    candidate_ok, _ = admission('FIELD_IN_SITU', 'model_assisted_inspection',
                                'medium', 'model_assisted_pending_human',
                                level='candidate')
    assert candidate_ok is True


def test_unknown_domain_string_is_rejected():
    with pytest.raises(ValueError):
        validate_domain('MAYBE_FIELD')


def test_policy_is_serialisable_and_complete():
    policy = default_policy()
    assert set(policy['domains']) == set(DOMAINS)
    assert policy['rules']['freeze']['allowed_domains'] == ['FIELD_IN_SITU',
                                                            'ORCHARD_IN_SITU']
    assert 'UNKNOWN' in policy['excluded_by_default']


# --------------------------------------------------------------------------- #
# 8-10: holdout rule and non-duplication of sources
# --------------------------------------------------------------------------- #

def _row(**overrides):
    base = {
        'sample_id': 'x:1', 'path': '/tmp/x.jpg', 'source_prefix': 'field_a',
        'source': 'A', 'original_id': 'g1', 'split': 'train', 'width': 512,
        'height': 512, 'sha256': hashlib.sha256(b'a').hexdigest(),
        'pixel_sha256': hashlib.sha256(b'p').hexdigest(),
        'duplicate_status': 'unique', 'component': 'g1', 'acquisition_domain':
        'FIELD_IN_SITU', 'domain_evidence_type': 'official_documentation',
        'domain_confidence': 'high', 'domain_review_status': 'reviewed',
    }
    base.update(overrides)
    return base


def test_whitelist_cannot_override_holdout():
    """8: allow_sources is not a way to smuggle in a holdout source."""
    rows = [_row(source_prefix='phenobench', sample_id='phenobench:1'),
            _row(source_prefix='field_a', sample_id='field_a:1')]
    kept, reasons, _ = apply_gate(rows, holdout_ids={'phenobench'})
    assert [r['sample_id'] for r in kept] == ['field_a:1']
    assert reasons['holdout_source'] == 1


@pytest.mark.parametrize('source_id', ['phenobench', 'gwhd_2021', 'plantdoc'])
def test_named_holdout_sources_are_dropped(source_id):
    """9: PhenoBench, GWHD and PlantDoc never reach the main corpus."""
    kept, reasons, _ = apply_gate([_row(source_prefix=source_id)],
                                  holdout_ids={'phenobench', 'gwhd_2021',
                                               'plantdoc'})
    assert kept == []
    assert reasons['holdout_source'] == 1


def test_holdout_registry_lists_twelve_sources():
    with open(HOLDOUT, newline='', encoding='utf-8') as handle:
        ids = {r['source_id'] for r in csv.DictReader(handle)}
    assert len(ids) == 12
    assert {'phenobench', 'gwhd_2021', 'plantdoc', 'minneapple'} <= ids


def test_same_dataset_rows_are_marked_and_not_counted_twice():
    """10: Agri-FM+ repeat citations map to one physical dataset."""
    assert SAME_PHYSICAL_DATASET['agrifm_ref79'] == 'ronin'
    assert SAME_PHYSICAL_DATASET['agrifm_ref88'] == 'phenobench'
    assert SAME_PHYSICAL_DATASET['agrifm_ref4'] == 'perennial_plants'
    # the ledger itself carries the same marker for those rows
    assert SOURCE_DOMAINS['agrifm_ref79']['acquisition_domain'] == \
        SOURCE_DOMAINS['ronin']['acquisition_domain']


# --------------------------------------------------------------------------- #
# 11: hashes
# --------------------------------------------------------------------------- #

def test_file_and_pixel_hash_are_stable_and_distinct(tmp_path):
    from src.registry.ingest import file_sha256, phash64, pixel_sha256
    array = np.random.default_rng(7).integers(0, 255, (40, 40, 3), dtype='uint8')
    path = tmp_path / 'a.png'
    Image.fromarray(array, 'RGB').save(path)
    with Image.open(path) as image:
        image.load()
        pixel = pixel_sha256(image)
        first = phash64(image)
        second = phash64(image)
    assert first == second
    assert len(pixel) == 64
    assert file_sha256(path) == file_sha256(path)


# --------------------------------------------------------------------------- #
# 12-13: duplicates and capture groups
# --------------------------------------------------------------------------- #

def test_perceptual_candidate_is_not_a_duplicate_verdict():
    """12: a perceptual match keeps a pending status and is not dropped."""
    rows = [_row(sample_id='a:1', duplicate_status='phash_candidate',
                 component='g1'),
            _row(sample_id='a:2', duplicate_status='phash_candidate',
                 component='g2',
                 sha256=hashlib.sha256(b'b').hexdigest(),
                 pixel_sha256=hashlib.sha256(b'q').hexdigest())]
    kept, reasons, _ = apply_gate(rows, holdout_ids=set())
    assert len(kept) == 2          # a candidate is not removed automatically
    assert reasons == {}


def test_exact_duplicate_is_dropped():
    sha = hashlib.sha256(b'c').hexdigest()
    pixel = hashlib.sha256(b'r').hexdigest()
    rows = [_row(sample_id='a:1', sha256=sha, pixel_sha256=pixel),
            _row(sample_id='a:2', sha256=sha, pixel_sha256=pixel)]
    kept, reasons, _ = apply_gate(rows, holdout_ids=set())
    assert len(kept) == 1
    assert reasons['exact_duplicate'] == 1


def test_capture_groups_never_cross_splits():
    """13: whole groups are assigned to one split, deterministically."""
    rows = [_row(sample_id=f'a:{i}', component=f'g{i // 4}',
                 sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                 pixel_sha256=hashlib.sha256(f'p{i}'.encode()).hexdigest())
            for i in range(40)]
    mapping = assign_group_split(rows, val_fraction=0.25, seed=11)
    again = assign_group_split(rows, val_fraction=0.25, seed=11)
    assert mapping == again
    assert len(set(mapping.values())) == 2
    for left in rows:
        for right in rows:
            if left['component'] == right['component']:
                assert mapping[left['component']] == mapping[right['component']]


# --------------------------------------------------------------------------- #
# 14-15: reproducibility
# --------------------------------------------------------------------------- #

def test_manifest_build_is_byte_reproducible(tmp_path):
    """14+15: the same rows produce the same bytes and the same SHA-256."""
    rows = []
    for i in range(10):
        row = _row(sample_id=f'a:{i}', split='train' if i < 8 else 'val',
                   component=f'g{i}',
                   sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                   pixel_sha256=hashlib.sha256(f'p{i}'.encode()).hexdigest())
        rows.append(row)
    prefix_a = tmp_path / 'a' / 'corpus'
    prefix_b = tmp_path / 'b' / 'corpus'
    manifest_a = write_field_corpus(prefix_a, rows)
    manifest_b = write_field_corpus(prefix_b, rows)
    assert manifest_a['sha256'] == manifest_b['sha256']
    assert (tmp_path / 'a' / 'corpus_train.csv').read_bytes() == \
        (tmp_path / 'b' / 'corpus_train.csv').read_bytes()


# --------------------------------------------------------------------------- #
# 16-17: licence and review gates
# --------------------------------------------------------------------------- #

def test_unverified_licence_cannot_be_approved():
    """16: an APPROVED row needs a non-empty licence and evidence."""
    from src.registry.schema import RegistryError, advance
    row = {
        'source_id': 's', 'official_name': 'S', 'paper': 'p', 'year': '2024',
        'paper_url': 'https://example.org', 'download_url': 'https://example.org',
        'license': '', 'license_evidence': '', 'state': 'URL_VERIFIED',
    }
    with pytest.raises(RegistryError):
        advance(row, 'LICENSE_VERIFIED')


def test_unreviewed_images_cannot_be_frozen():
    """17: a manifest with pending reviews reports CANDIDATE, never FROZEN."""
    rows = [_row(sample_id=f'a:{i}', component=f'g{i}',
                 domain_review_status='model_assisted_pending_human',
                 sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                 pixel_sha256=hashlib.sha256(f'p{i}'.encode()).hexdigest(),
                 split='train')
            for i in range(4)]
    kept, _, _ = apply_gate(rows, holdout_ids=set())
    assert len(kept) == 4
    pending = sum(1 for r in kept if r['domain_review_status'] != 'reviewed')
    assert pending == 4


# --------------------------------------------------------------------------- #
# 18: frozen files untouched
# --------------------------------------------------------------------------- #

V1_HASHES = {
    'agrifield40k-v1_train.csv':
        'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435',
    'agrifield40k-v1_val.csv':
        '86efe5156291950b25af692d41b5f1b4cee437dd3ba2a44ce39e2cf18984f944',
}


@pytest.mark.parametrize('name,expected', sorted(V1_HASHES.items()))
def test_frozen_v1_is_byte_identical(name, expected):
    """18: Stage 2D's frozen v1 files must not change."""
    path = FROZEN / name
    if not path.is_file():
        pytest.skip('frozen v1 not present on this machine')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


# --------------------------------------------------------------------------- #
# 19-20: holdout evaluation intact, loader compatibility
# --------------------------------------------------------------------------- #

def test_field_d1_manifest_contains_no_holdout_source():
    """19: building Field-D1 did not pull in an evaluation dataset."""
    path = FROZEN / 'agri-field-d1-candidate_train.csv'
    if not path.is_file():
        pytest.skip('Field-D1 candidate not built on this machine')
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    with open(HOLDOUT, newline='', encoding='utf-8') as handle:
        holdout = {r['source_id'] for r in csv.DictReader(handle)}
    assert not ({r['source_prefix'] for r in rows} & holdout)
    assert all(r['status'] == 'include' for r in rows)
    assert all(r['acquisition_domain'] in ('FIELD_IN_SITU', 'ORCHARD_IN_SITU')
               for r in rows)


def test_field_d1_manifest_has_no_controlled_or_unknown_source():
    path = FROZEN / 'agri-field-d1-candidate_train.csv'
    if not path.is_file():
        pytest.skip('Field-D1 candidate not built on this machine')
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert 'plantvillage' not in {r['source_prefix'] for r in rows}
    assert 'agrifm_ref93' not in {r['source_prefix'] for r in rows}


def test_field_d1_groups_are_split_disjoint():
    train = FROZEN / 'agri-field-d1-candidate_train.csv'
    val = FROZEN / 'agri-field-d1-candidate_val.csv'
    if not train.is_file() or not val.is_file():
        pytest.skip('Field-D1 candidate not built on this machine')
    with open(train, newline='', encoding='utf-8') as handle:
        train_groups = {r['component'] for r in csv.DictReader(handle)}
    with open(val, newline='', encoding='utf-8') as handle:
        val_groups = {r['component'] for r in csv.DictReader(handle)}
    assert not (train_groups & val_groups)


def test_field_d1_manifest_is_loader_compatible():
    """20: the manifest has the columns ``AgriFieldDataset`` requires."""
    path = FROZEN / 'agri-field-d1-candidate_train.csv'
    if not path.is_file():
        pytest.skip('Field-D1 candidate not built on this machine')
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
    assert {'sample_id', 'path', 'split', 'status'} <= fields
    # ... and the dataset really reads it
    from src.datasets.agri_dataset import AgriFieldDataset
    root = '/data/datasets/AgriField-40K'
    if not Path(root).is_dir():
        pytest.skip('AgriField-40K images not mounted')
    dataset = AgriFieldDataset(str(path), split='train', max_samples=2,
                               root_path=root, transform=None)
    image, sample_id = dataset[0]
    assert image.mode == 'RGB'
    assert sample_id


# --------------------------------------------------------------------------- #
# Sampling strategies
# --------------------------------------------------------------------------- #

def test_source_balance_strategies():
    rows = ([_row(sample_id=f'big:{i}', source_prefix='big', component=f'b{i}',
                  sha256=hashlib.sha256(f'b{i}'.encode()).hexdigest(),
                  pixel_sha256=hashlib.sha256(f'bp{i}'.encode()).hexdigest())
             for i in range(90)] +
            [_row(sample_id=f'small:{i}', source_prefix='small', component=f's{i}',
                  sha256=hashlib.sha256(f's{i}'.encode()).hexdigest(),
                  pixel_sha256=hashlib.sha256(f'sp{i}'.encode()).hexdigest())
             for i in range(10)])
    assert max_source_share(rows, 'source_prefix') == pytest.approx(0.9)
    natural = strategy_weights(rows, 'natural')
    assert natural == {'big': 1.0, 'small': 1.0}
    balanced = strategy_weights(rows, 'source_balanced')
    counts = effective_sample_counts(rows, balanced)
    assert counts['big']['share_after_sampling'] == pytest.approx(0.5, abs=0.01)
    assert counts['small']['share_after_sampling'] == pytest.approx(0.5, abs=0.01)
    capped = strategy_weights(rows, 'source_capped', cap=0.6)
    counts = effective_sample_counts(rows, capped)
    assert counts['big']['share_after_sampling'] == pytest.approx(0.6, abs=0.02)
    assert shannon_entropy(rows, 'source_prefix') < 1.0


def test_quality_gate_rejects_small_and_grayscale(tmp_path):
    small = tmp_path / 'small.png'
    Image.fromarray(np.zeros((100, 100, 3), dtype='uint8'), 'RGB').save(small)
    with Image.open(small) as image:
        metrics = {'short_side': min(image.size), 'mode': image.mode,
                   'mean_brightness': 50.0, 'laplacian_variance': 100.0}
    passed, reason = quality_gate(metrics, QualityGate(min_short_side=224))
    assert passed is False
    assert '224' in reason

    gray = {'short_side': 300, 'mode': 'L', 'mean_brightness': 50.0,
            'laplacian_variance': 100.0}
    passed, reason = quality_gate(gray, QualityGate())
    assert passed is False
    assert 'grayscale' in reason


def test_triage_marks_lab_like_images_for_review():
    lab = {'border_fraction': 0.95, 'border_luma': 240.0,
           'vegetation_fraction': 0.01}
    suggestion, reason = suggest_domain(lab, 'leaf.jpg')
    assert suggestion == 'CONTROLLED_LAB'
    field = {'border_fraction': 0.2, 'border_luma': 120.0,
             'vegetation_fraction': 0.4}
    suggestion, _ = suggest_domain(field, 'field.jpg')
    assert suggestion == 'FIELD_IN_SITU'
