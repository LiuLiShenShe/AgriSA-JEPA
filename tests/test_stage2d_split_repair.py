"""Stage 2D split-repair tests.

Covers the two structural leaks the v1 freeze fixes, plus the properties the
freeze record depends on: determinism, byte-level hashes, train/val disjointness,
and conservative handling of observations with no id.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from tools.build_split_v1 import (
    FIELDNAMES,
    assign_inat_observations,
    build_v1,
    observation_id,
)

REPO = Path(__file__).resolve().parents[1]
V0_TRAIN = REPO / 'data/frozen/agrifield40k-v0_train.csv'
V0_VAL = REPO / 'data/frozen/agrifield40k-v0_val.csv'
V1_TRAIN = REPO / 'data/frozen/agrifield40k-v1_train.csv'
V1_VAL = REPO / 'data/frozen/agrifield40k-v1_val.csv'
V1_MANIFEST = REPO / 'data/frozen/agrifield40k-v1_freeze_manifest.json'


def read(path):
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def row(sample_id, source, original_id='', observation_url='', split='train'):
    return {
        'sample_id': sample_id,
        'path': f'agrifield40k/{split}/{sample_id}.jpg',
        'source_prefix': source[:3].lower(),
        'source': source,
        'original_id': original_id,
        'observation_url': observation_url,
        'split': split,
        'status': 'include',
        'sha256': f'hash-{sample_id}',
    }


# --------------------------------------------------------------------------- #
# ACRECropWeed
# --------------------------------------------------------------------------- #

class TestACRECropWeed:

    def test_whole_source_lands_in_train_only(self):
        v0_train = [row(f'acw_{i}', 'ACRECropWeed', f'rgb-2022-10-06-17-16-{i:02d}')
                    for i in range(6)]
        v0_val = [row(f'acw_v{i}', 'ACRECropWeed', f'rgb-2022-10-06-17-39-{i:02d}',
                      split='val') for i in range(4)]
        train, val, summary = build_v1(v0_train, v0_val)
        assert {r['source'] for r in val} == set()
        assert sum(1 for r in train if r['source'] == 'ACRECropWeed') == 10
        assert summary['acw']['val_images_moved_to_train'] == 4

    def test_no_acw_in_frozen_val(self):
        val = read(V1_VAL)
        assert not [r for r in val if r['source'] == 'ACRECropWeed']

    def test_frozen_v1_moves_exactly_120(self):
        manifest = json.loads(V1_MANIFEST.read_text())
        assert manifest['changes_from_v0']['moved_counts'][
            'ACRECropWeed:val->train'] == 120

    def test_acw_never_creates_independent_sessions(self):
        # the tool must not split ACW by date, however the timestamps look
        v0_train = [row('a', 'ACRECropWeed', 'rgb-2022-10-06-17-16-49')]
        v0_val = [row('b', 'ACRECropWeed', 'rgb-2022-10-10-17-30-48', split='val')]
        train, val, _ = build_v1(v0_train, v0_val)
        assert [r['sample_id'] for r in train] == ['a', 'b']
        assert val == []


# --------------------------------------------------------------------------- #
# iNatWeeds
# --------------------------------------------------------------------------- #

class TestINatWeeds:

    def test_observation_id_parsing(self):
        assert observation_id(
            {'observation_url': 'https://www.inaturalist.org/observations/308673477'}
        ) == '308673477'
        assert observation_id({'observation_url': ''}) is None
        assert observation_id({}) is None

    def test_no_observation_straddles_train_and_val(self):
        train = read(V1_TRAIN)
        val = read(V1_VAL)
        splits = {}
        for name, rows in (('train', train), ('val', val)):
            for r in rows:
                if r['source'] != 'iNatWeeds':
                    continue
                obs = observation_id(r)
                if obs is None:
                    continue
                splits.setdefault(obs, set()).add(name)
        crossing = {k: v for k, v in splits.items() if len(v) > 1}
        assert crossing == {}
        # the 31 v0 crossings are gone, not merely relabelled
        assert len(splits) == 1227

    def test_observation_with_missing_id_goes_to_train(self):
        rows = [
            row('x1', 'iNatWeeds', observation_url=''),
            row('x2', 'iNatWeeds',
                observation_url='https://www.inaturalist.org/observations/1'),
        ]
        assignment, stats = assign_inat_observations(rows, 0.5, seed=0)
        assert stats['missing_observation_id_images'] == 1
        assert 'x1' not in [r['sample_id'] for r in rows if observation_id(r)]
        # the missing-id row is never assigned to val by the assignment map
        assert all(observation_id(r) is None or assignment[observation_id(r)] in
                   ('train', 'val') for r in rows)

    def test_assignment_is_deterministic(self):
        rows = [row(f'inat_{i}', 'iNatWeeds',
                    observation_url=f'https://www.inaturalist.org/observations/{i}')
                for i in range(200)]
        a, _ = assign_inat_observations(rows, 0.2, seed=7)
        b, _ = assign_inat_observations(list(reversed(rows)), 0.2, seed=7)
        assert a == b, 'assignment must not depend on input row order'

    def test_assignment_keeps_val_fraction(self):
        rows = []
        for i in range(100):
            for j in range(i % 3 + 1):
                rows.append(row(f'inat_{i}_{j}', 'iNatWeeds',
                                observation_url=f'https://www.inaturalist.org/observations/{i}'))
        assignment, stats = assign_inat_observations(rows, 0.2, seed=3)
        val_images = sum(
            sum(1 for r in rows if observation_id(r) == obs)
            for obs, split in assignment.items() if split == 'val')
        assert stats['assigned_val_images'] == val_images
        assert val_images == stats['target_val_images']
        assert 0.15 <= val_images / len(rows) <= 0.25

    def test_v1_val_keeps_278_inat_images(self):
        val = [r for r in read(V1_VAL) if r['source'] == 'iNatWeeds']
        assert len(val) == 278


# --------------------------------------------------------------------------- #
# Version-level invariants
# --------------------------------------------------------------------------- #

class TestV1Version:

    def test_v1_preserves_the_v0_sample_set(self):
        v0 = {r['sample_id'] for r in read(V0_TRAIN)} | {
            r['sample_id'] for r in read(V0_VAL)}
        v1 = {r['sample_id'] for r in read(V1_TRAIN)} | {
            r['sample_id'] for r in read(V1_VAL)}
        assert v0 == v1
        assert len(v1) == 15592

    def test_v1_counts(self):
        assert len(read(V1_TRAIN)) == 12567
        assert len(read(V1_VAL)) == 3025

    def test_train_val_disjoint(self):
        train = {r['sample_id'] for r in read(V1_TRAIN)}
        val = {r['sample_id'] for r in read(V1_VAL)}
        assert train & val == set()

    def test_no_phenobench_rows(self):
        for path in (V1_TRAIN, V1_VAL):
            assert not [r for r in read(path) if r['source'] == 'PhenoBench']

    def test_manifest_records_hashes_and_provisional_status(self):
        manifest = json.loads(V1_MANIFEST.read_text())
        assert manifest['data_version'] == 'agrifield40k-v1'
        assert manifest['freeze_status'] == 'provisional'
        assert manifest['gate']['source_license_gate_met'] is False
        assert set(manifest['sha256']) == {
            'agrifield40k-v1_train.csv', 'agrifield40k-v1_val.csv'}

    def test_manifest_hashes_match_files(self):
        import hashlib
        manifest = json.loads(V1_MANIFEST.read_text())
        for name, expected in manifest['sha256'].items():
            actual = hashlib.sha256((REPO / 'data/frozen' / name).read_bytes()).hexdigest()
            assert actual == expected, f'{name} changed after freezing'

    def test_build_is_deterministic(self):
        v0_train, v0_val = read(V0_TRAIN), read(V0_VAL)
        t1, v1, s1 = build_v1(v0_train, v0_val)
        t2, v2, s2 = build_v1(v0_train, v0_val)
        assert [r['sample_id'] for r in t1] == [r['sample_id'] for r in t2]
        assert [r['sample_id'] for r in v1] == [r['sample_id'] for r in v2]
        assert s1 == s2

    def test_v0_is_untouched(self):
        import hashlib
        expected = {
            'agrifield40k-v0_train.csv':
                'd8a179b81f2052c6d66d2f43d26505652e95b1090560d13c1e9d46efe3b27037',
            'agrifield40k-v0_val.csv':
                '4a7338d9f105164dd97fbf5719adef4b03d0e61e45d337cea07f24fbc1905a45',
        }
        for name, digest in expected.items():
            actual = hashlib.sha256((REPO / 'data/frozen' / name).read_bytes()).hexdigest()
            assert actual == digest

    def test_output_columns_are_stable(self):
        header = next(csv.reader(V1_TRAIN.open(newline='', encoding='utf-8')))
        assert header == FIELDNAMES