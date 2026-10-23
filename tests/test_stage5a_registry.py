"""Stage 5A tests: registry schema, state machine, and source catalog."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.registry.catalog import all_sources
from src.registry.schema import (BLOCKED, STATES, advance, can_freeze,
                                 missing_for_state, validate_row)

ROOT = Path(__file__).resolve().parent.parent


def base_row(**kw):
    row = validate_row({
        'source_id': 'test_src',
        'official_name': 'Test Dataset',
        'paper': 'Test et al. 2026',
        'year': '2026',
        'paper_url': 'https://example.org/paper',
        'download_url': 'https://example.org/data',
        'license': 'CC-BY-4.0',
        'license_evidence': 'page licence field',
    })
    row.update(kw)
    return validate_row(row)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

def test_states_are_ordered_and_unique():
    assert len(set(STATES)) == len(STATES)
    assert STATES[0] == 'DISCOVERED'
    assert STATES[-1] == 'FROZEN'
    assert BLOCKED not in STATES


def test_validate_row_fills_defaults():
    row = base_row()
    for field in ('images_reported', 'files_remote', 'files_downloaded',
                  'images_decoded', 'images_approved'):
        assert row[field] is None
    assert row['state'] == 'DISCOVERED'
    assert row['allowed_in_project'] == 'pending'


def test_validate_row_rejects_unknown_field():
    with pytest.raises(ValueError):
        validate_row({'source_id': 'x', 'not_a_field': 'y'})


def test_validate_row_parses_comma_numbers():
    row = base_row(images_reported='147,285')
    assert row['images_reported'] == 147285


def test_validate_row_rejects_non_numeric_count():
    with pytest.raises(ValueError):
        base_row(images_reported='about 1000')


def test_validate_row_rejects_fractional_count():
    with pytest.raises(ValueError):
        base_row(files_remote=12.5)


def test_validate_row_rejects_illegal_state():
    with pytest.raises(ValueError):
        base_row(state='APPROVED_PLEASE')


def test_validate_row_rejects_illegal_allowed():
    with pytest.raises(ValueError):
        base_row(allowed_in_project='maybe')


def test_advance_requires_one_gate_at_a_time():
    row = base_row()
    with pytest.raises(ValueError):
        advance(row, 'LICENSE_VERIFIED')


def test_advance_refuses_missing_evidence():
    row = base_row()
    row = advance(row, 'URL_VERIFIED')
    row = advance(row, 'LICENSE_VERIFIED')
    with pytest.raises(ValueError, match='download_evidence|files_downloaded'):
        advance(row, 'DOWNLOADED')


def test_advance_chain_to_split_checked():
    row = base_row(state='DISCOVERED')
    row = advance(row, 'URL_VERIFIED')
    assert row['state'] == 'URL_VERIFIED'
    row = advance(row, 'LICENSE_VERIFIED')
    assert row['state'] == 'LICENSE_VERIFIED'
    row.update(download_evidence='archive sha256 verified', files_downloaded=10)
    row = advance(row, 'DOWNLOADED')
    row.update(decode_evidence='all decode', images_decoded=10)
    row = advance(row, 'DECODED')
    row.update(duplicate_evidence='screened', images_duplicate_free=10)
    row = advance(row, 'DUPLICATE_SCREENED')
    row.update(split_evidence='no holdout overlap', downstream_holdout='no')
    row = advance(row, 'SPLIT_CHECKED')
    row.update(approval_evidence='curator approved', human_review_status='reviewed',
               allowed_in_project='yes', images_approved=10)
    row = advance(row, 'APPROVED')
    assert row['state'] == 'APPROVED'


def test_freeze_requires_positive_approved_count_and_review():
    row = base_row(state='APPROVED', allowed_in_project='yes',
                   human_review_status='reviewed')
    assert not can_freeze(row)
    row['images_approved'] = 100
    assert can_freeze(row)
    row['allowed_in_project'] = 'pending'
    assert not can_freeze(row)


def test_freeze_refuses_blocked_row():
    row = base_row(state='BLOCKED')
    assert not can_freeze(row)


def test_missing_for_state_blocked_needs_nothing():
    assert missing_for_state(base_row(state='BLOCKED'), 'BLOCKED') == []


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #

def test_catalog_has_all_17_agrifield_sources():
    rows = all_sources()
    agrifield = [r for r in rows if 'agrifield40k' in str(r['catalog_groups'])]
    assert len(agrifield) == 17


def test_catalog_has_35_agrifm_sources():
    rows = all_sources()
    fm = [r for r in rows if 'agrifmplus' in str(r['catalog_groups'])]
    ids = {r['source_id'] for r in fm}
    # 17 AgriField rows may also carry the agrifmplus tag when the paper cites
    # the same dataset; the 35 Table-1 refs are the ones with ref ids.
    refs = [r for r in fm if str(r['source_id']).startswith('agrifm_ref')]
    assert len(refs) == 35, sorted(r['source_id'] for r in refs)


def test_every_row_has_distinct_source_id():
    rows = all_sources()
    ids = [r['source_id'] for r in rows]
    assert len(set(ids)) == len(ids)


def test_every_row_validates():
    for row in all_sources():
        validate_row(row)


def test_agrifm_rows_carry_ref_number_and_url():
    rows = all_sources()
    refs = [r for r in rows if str(r['source_id']).startswith('agrifm_ref')]
    for row in refs:
        assert 'ref [' in str(row['paper'])
        assert str(row['download_url']).startswith('http'), row['source_id']


def test_phenobench_and_gwhd_are_marked_holdout():
    rows = {r['source_id']: r for r in all_sources()}
    assert rows['phenobench']['downstream_holdout'] == 'yes'
    assert rows['gwhd_2021']['downstream_holdout'] == 'yes'
    assert rows['gwhd_instance_seg']['downstream_holdout'] == 'yes'


def test_gwhd_detection_and_instance_seg_are_distinct_rows():
    rows = {r['source_id']: r for r in all_sources()}
    assert rows['gwhd_2021']['images_reported'] == 6514
    assert rows['gwhd_instance_seg']['images_reported'] == 582


def test_counts_are_separate_columns_not_derived():
    row = next(r for r in all_sources() if r['source_id'] == 'agrifm_ref10')
    # 40,457 images is only the paper's number; nothing else may be filled.
    assert row['images_reported'] == 40457
    assert row['files_downloaded'] is None
    assert row['images_decoded'] is None


def test_downloaded_rows_carry_local_evidence():
    rows = {r['source_id']: r for r in all_sources()}
    for sid in ('acre_crop_weed', 'lucas_vision', 'vegann', 'sorghum_weed',
                'radish_wheat', 'inat_weeds'):
        row = rows[sid]
        assert row['downloaded'] == 'yes', sid
        assert row['download_evidence'], sid
        assert row['decode_evidence'], sid
        assert row['split_evidence'], sid
        assert row['state'] == 'SPLIT_CHECKED', sid


def test_pending_audit_sources_are_not_marked_confirmed():
    rows = {r['source_id']: r for r in all_sources()}
    for sid in ('grass_clover', 'we3ds', 'maize_weed', 'rumex_leaves'):
        assert rows[sid]['state'] != 'APPROVED'
        assert rows[sid]['state'] != 'FROZEN'


# --------------------------------------------------------------------------- #
# built artefacts
# --------------------------------------------------------------------------- #

def test_registry_csv_exists_and_parses():
    path = ROOT / 'data/registry/agricultural_sources.csv'
    assert path.is_file()
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 60
    for row in rows:
        assert row['source_id']
        assert row['state'] in STATES + [BLOCKED]


def test_benchmark_registry_covers_agrifm_eight_benchmarks():
    path = ROOT / 'data/registry/agricultural_benchmarks.csv'
    assert path.is_file()
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    tasks = {row['task'] for row in rows}
    assert 'object_detection' in tasks
    assert 'semantic_segmentation' in tasks
    assert 'instance_segmentation' in tasks
    assert len(rows) == 12


def test_benchmark_registry_separates_gwhd_tasks():
    path = ROOT / 'data/registry/agricultural_benchmarks.csv'
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    det = next(r for r in rows if r['benchmark_id'] == 'gwhd2021_det')
    inst = next(r for r in rows if r['benchmark_id'] == 'gwhd_instance_seg')
    assert det['dataset'] != inst['dataset']
    assert det['metric'] != inst['metric']
    assert det['images_reported'] != inst['images_reported']


def test_holdout_registry_contains_phenobench_and_gwhd():
    path = ROOT / 'data/registry/downstream_holdout_sources.csv'
    with open(path, newline='', encoding='utf-8') as handle:
        ids = {row['source_id'] for row in csv.DictReader(handle)}
    for sid in ('phenobench', 'gwhd_2021', 'gwhd_instance_seg', 'deepweeds'):
        assert sid in ids


def test_source_manifest_records_no_frozen_rows():
    path = ROOT / 'data/registry/agricultural_sources_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    assert manifest['states']['FROZEN'] == 0
    assert manifest['sha256_canonical_csv']


def test_only_holdouts_may_be_approved():
    """APPROVED means accepted for its intended use.

    GWHD is approved **as the evaluation set**; it must never be counted as
    approved pretraining data, so every APPROVED row has to be a holdout and
    no row may be FROZEN.
    """
    path = ROOT / 'data/registry/agricultural_sources.csv'
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    approved = [r for r in rows if r['state'] == 'APPROVED']
    for row in approved:
        assert row['downstream_holdout'] == 'yes', row['source_id']
        assert int(row['images_approved']) > 0, row['source_id']
    assert not [r for r in rows if r['state'] == 'FROZEN']


def test_overlap_candidates_record_all_members():
    path = ROOT / 'data/registry/source_overlap_candidates.csv'
    with open(path, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    groups = {row['overlap_group'] for row in rows}
    assert {'phenobench', 'perennial_plants', 'ronin', 'gwhd_2021',
            'deepweeds'} <= groups
