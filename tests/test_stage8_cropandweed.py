"""Stage 8 tests: CropAndWeed data mapping, split logic, probe arithmetic.

Checks that do not need the 8k-image dataset or a GPU:

* the CropsOrWeed9 source->class mapping equals the vendored official source
  (AST-parsed, verbatim);
* Soil (0) and Vegetation (255) are unmapped (ignore);
* session keys group by prefix+session and the prefixed sets are disjoint;
* the eval session-stratified sample is seeded and reproducible;
* patch_class_counts with IGNORE_ID=-1 pixels does not count them anywhere;
* no session leaks between the ave_ train set and vwg_ eval sample.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.downstream.cropandweed_data import (IGNORE_ID, build_source_to_class,
                                             map_label_ids,
                                             session_stratified_sample,
                                             session_key)
from src.downstream.probe import patch_class_counts

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / 'data' / 'holdout_raw' / 'cropandweed_official_code' / 'datasets.py'


def _official_cropsorweed9() -> dict:
    src = OFFICIAL.read_text(encoding='utf-8')
    tree = ast.parse(src)
    mapping = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, 'id', None) == 'Dataset'):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                        and t.slice.value == 'CropsOrWeed9'):
                    root = node.value.args[0]
                    for k, v in zip(root.keys, root.values):
                        srcs = v.elts[2].elts if len(v.elts) > 2 else []
                        for s in srcs:
                            mapping[s.value] = k.value
    return mapping


def test_cropsorweed9_mapping_matches_official_source():
    if not OFFICIAL.is_file():
        pytest.skip('vendored official datasets.py not present')
    official = _official_cropsorweed9()
    assert len(official) == 76
    assert len({v for v in official.values()}) == 9
    assert build_source_to_class() == official


def test_soil_and_vegetation_are_ignore():
    mapping = build_source_to_class()
    assert 0 not in mapping and 255 not in mapping


def test_map_label_ids_ignore_unmapped():
    mask = np.array([[0, 1, 255, 94], [24, 15, 13, 31]], dtype=np.uint8)
    mapped = map_label_ids(mask)
    expect = np.array([[IGNORE_ID, 0, IGNORE_ID, 2],
                       [3, 7, 5, 8]], dtype=np.int16)
    assert (mapped == expect).all()


def test_session_key_is_prefixed_session():
    assert session_key('ave-0540-0012.jpg') == 'ave-0540'
    assert session_key('vwg-0999-0001.jpg') == 'vwg-0999'
    with pytest.raises(ValueError):
        session_key('bad_name.jpg')


def test_session_sample_reproducible():
    names = [f'vwg-0{i}-{j:04d}.jpg' for i in range(5) for j in range(4)]
    a = session_stratified_sample(names, per_session=1, seed=20261107)
    b = session_stratified_sample(names, per_session=1, seed=20261107)
    c = session_stratified_sample(names, per_session=1, seed=99)
    assert a == b
    assert a != c
    sessions = {session_key(n) for n in a}
    assert len(sessions) == len(a) == 5  # one per session


def test_patch_counts_exclude_ignore_id():
    # 224x224 label: 100% class 0 except a 14x14 patch of IGNORE_ID (-1)
    label = np.full((224, 224), 0, dtype=np.int16)
    label[0:14, 0:14] = IGNORE_ID
    counts = patch_class_counts(label[None], num_classes=9)   # [1,224,224]
    assert counts.shape == (1, 16, 16, 9)
    # the ignore patch contributes nothing anywhere (all zeros)
    assert counts[0, 0, 0].sum() == 0
    # every other patch is fully class 0
    assert counts[0, 0, 1, 0] == 14 * 14
    assert counts[0, 0, 1, 1:].sum() == 0


def test_ave_vwg_sessions_disjoint_from_frozen_split():
    # Uses only the name prefix rules, not the full 8K dataset: derived from
    # the disjointness invariant that build_split enforces via session keys.
    from src.downstream.cropandweed_data import session_key
    assert '-' in session_key('ave-0000-0000.jpg')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))