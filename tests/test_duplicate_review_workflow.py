"""Tests for the Stage 2C duplicate-review workflow tool.

CPU-only. Every test uses ``tmp_path`` and never touches the real
``data/duplicate_review.csv`` or ``data/frozen/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools import duplicate_review_workflow as drw


FIELDS = [
    'edge_id', 'sample_a', 'sample_b', 'kind', 'phash_distance', 'dhash_distance',
    'cross_split', 'split_a', 'split_b', 'source_a', 'source_b', 'path_a', 'path_b',
    'in_train_manifest', 'in_val_manifest', 'review_status', 'reviewer',
    'review_notes', 'reviewed_at', 'isolation_category', 'thumb_a', 'thumb_b',
]


def _row(edge_id, a, b, cross='False', in_train='false', in_val='false',
         status=drw.PENDING, reviewer='', notes='', reviewed_at=''):
    return {
        'edge_id': edge_id, 'sample_a': a, 'sample_b': b, 'kind': 'perceptual_candidate',
        'phash_distance': '1', 'dhash_distance': '2', 'cross_split': cross,
        'split_a': 'train', 'split_b': 'val' if cross == 'True' else 'train',
        'source_a': 'iNatWeeds', 'source_b': 'iNatWeeds', 'path_a': f'{a}.jpg',
        'path_b': f'{b}.jpg', 'in_train_manifest': in_train, 'in_val_manifest': in_val,
        'review_status': status, 'reviewer': reviewer, 'review_notes': notes,
        'reviewed_at': reviewed_at, 'isolation_category': 'same_split_screening',
        'thumb_a': '', 'thumb_b': '',
    }


def _write_csv(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\r\n')
        w.writeheader()
        w.writerows(rows)


def _verdict(status='not_duplicate', reviewer='Dr A. Reviewer',
             notes='looked at both images; different scenes', reviewed_at='2026-10-19T10:00:00'):
    return {'review_status': status, 'reviewer': reviewer,
            'review_notes': notes, 'reviewed_at': reviewed_at}


# --- record: validation ----------------------------------------------------

def test_record_rejects_blank_reviewer():
    err = drw.validate_verdict('edge-0001', _verdict(reviewer=''))
    assert err is not None
    assert 'reviewer is empty' in err


@pytest.mark.parametrize('sentinel', ['ai', 'AI', 'Claude', 'automation', 'gpt', 'bot'])
def test_record_rejects_reserved_ai_reviewer(sentinel):
    err = drw.validate_verdict('edge-0001', _verdict(reviewer=sentinel))
    assert err is not None
    assert 'sentinel' in err


def test_record_rejects_missing_fields_and_bad_status():
    v = _verdict()
    v.pop('review_notes')
    assert 'missing required field' in drw.validate_verdict('edge-9', v)
    assert 'invalid review_status' in drw.validate_verdict('edge-9', _verdict(status='maybe'))
    assert 'ISO-8601' in drw.validate_verdict('edge-9', _verdict(reviewed_at='yesterday'))


def test_record_accepts_fully_attributed_verdict():
    rows = [_row('edge-0001', 'a', 'b')]
    errors = drw.apply_verdicts(rows, {'edge-0001': _verdict()})
    assert errors == []
    assert rows[0]['review_status'] == 'not_duplicate'
    assert rows[0]['reviewer'] == 'Dr A. Reviewer'


def test_record_is_all_or_nothing():
    rows = [_row('edge-0001', 'a', 'b'), _row('edge-0002', 'c', 'd')]
    errors = drw.apply_verdicts(rows, {
        'edge-0001': _verdict(),
        'edge-0002': _verdict(reviewer=''),   # bad -> whole batch fails
    })
    assert errors
    # Nothing applied because one record was invalid.
    assert all(r['review_status'] == drw.PENDING for r in rows)
    assert all(r['reviewer'] == '' for r in rows)


def test_record_cli_rejects_blank_reviewer_and_writes_nothing(tmp_path, capsys):
    csv_path = tmp_path / 'duplicate_review.csv'
    _write_csv(csv_path, [_row('edge-0001', 'a', 'b')])
    vfile = tmp_path / 'v.json'
    vfile.write_text(json.dumps({'edge-0001': _verdict(reviewer='')}), encoding='utf-8')

    rc = drw.main(['--review-csv', str(csv_path), 'record', '--verdicts', str(vfile)])
    assert rc == 1
    # The original CSV is untouched (still pending, blank reviewer).
    back = drw._load_rows(csv_path)
    assert back[0]['review_status'] == drw.PENDING
    assert back[0]['reviewer'] == ''


def test_record_cli_accepts_and_writes_back(tmp_path):
    csv_path = tmp_path / 'duplicate_review.csv'
    _write_csv(csv_path, [_row('edge-0001', 'a', 'b')])
    vfile = tmp_path / 'v.json'
    vfile.write_text(json.dumps({'edge-0001': _verdict()}), encoding='utf-8')

    rc = drw.main(['--review-csv', str(csv_path), 'record', '--verdicts', str(vfile)])
    assert rc == 0
    back = drw._load_rows(csv_path)
    assert back[0]['review_status'] == 'not_duplicate'
    assert back[0]['reviewer'] == 'Dr A. Reviewer'


# --- check: consistency ----------------------------------------------------

def test_check_flags_inconsistent_row():
    rows = [
        _row('edge-0001', 'a', 'b', status='confirmed_duplicate', reviewer=''),  # unattributed
        _row('edge-0002', 'c', 'd', status=drw.PENDING, reviewer='ghost'),       # half-state
        _row('edge-0003', 'e', 'f', status='not_a_verdict'),                     # unknown
    ]
    issues = drw.check_rows(rows)
    joined = '\n'.join(issues)
    assert 'unattributed verdict' in joined
    assert 'reviewer set but status still' in joined
    assert 'unknown verdict' in joined


def test_check_clean_rows_have_no_issues():
    rows = [_row('edge-0001', 'a', 'b', status='not_duplicate',
                 reviewer='Dr A', notes='n', reviewed_at='2026-10-19T10:00:00')]
    assert drw.check_rows(rows) == []


# --- impact: classification ------------------------------------------------

def test_impact_classifies_synthetic_rows():
    frozen = {'a', 'b', 'c', 'frozen_only'}
    rows = [
        # not_duplicate with one endpoint frozen, one isolated -> returnable.
        _row('edge-0001', 'a', 'isolated_x', status='not_duplicate',
             reviewer='Dr A', notes='n', reviewed_at='2026-10-19T10:00:00'),
        # confirmed duplicate with both endpoints frozen -> must remove.
        _row('edge-0002', 'b', 'c', status='confirmed_duplicate',
             reviewer='Dr B', notes='n', reviewed_at='2026-10-19T10:00:00'),
        # confirmed duplicate but one endpoint outside frozen -> no frozen removal.
        _row('edge-0003', 'frozen_only', 'outsider', status='confirmed_duplicate',
             reviewer='Dr C', notes='n', reviewed_at='2026-10-19T10:00:00'),
        # pending -> ignored.
        _row('edge-0004', 'a', 'b'),
    ]
    cls = drw.classify_impact(rows, frozen)
    assert cls['return_candidates'] == {'a', 'isolated_x'}
    assert cls['return_in_frozen'] == {'a'}
    assert cls['return_outside_frozen'] == {'isolated_x'}
    # edge-0002 removes b and c; edge-0003 must NOT add frozen_only.
    assert cls['remove_from_frozen'] == {'b', 'c'}
    assert 'frozen_only' not in cls['remove_from_frozen']


def test_impact_cli_dry_run_does_not_touch_frozen(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / 'duplicate_review.csv'
    _write_csv(csv_path, [
        _row('edge-0001', 'a', 'b', status='confirmed_duplicate',
             reviewer='Dr A', notes='n', reviewed_at='2026-10-19T10:00:00'),
    ])
    # Point frozen-set loading at synthetic manifests in tmp_path.
    train_f = tmp_path / 'train.csv'
    val_f = tmp_path / 'val.csv'
    with train_f.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id'])
        w.writeheader()
        w.writerows([{'sample_id': 'a'}, {'sample_id': 'b'}])
    with val_f.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['sample_id'])
        w.writeheader()
    monkeypatch.setattr(drw, 'FROZEN_TRAIN', train_f)
    monkeypatch.setattr(drw, 'FROZEN_VAL', val_f)

    out_md = tmp_path / 'impact.md'
    rc = drw.main(['--review-csv', str(csv_path), 'impact', '--out', str(out_md)])
    assert rc == 0
    text = out_md.read_text(encoding='utf-8')
    assert 'INVALIDATED' in text
    # Frozen manifests were not modified.
    assert {r['sample_id'] for r in drw._load_rows(train_f)} == {'a', 'b'}
