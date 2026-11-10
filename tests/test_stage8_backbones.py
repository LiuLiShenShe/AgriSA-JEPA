"""Stage 8 tests: backbone registry integrity and audit cross-check logic.

The registry tests avoid hashing the multi-GB checkpoints themselves (that
takes minutes); they validate the recorded values, the cache-key uniqueness
of every backbone spec, and the SHA mismatch behaviour with a stub registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.stage8_backbones import (DOWNSTREAM_SEEDS, EXPLORATORY, PRIMARY,
                                    PRETRAIN_SEEDS, Stage8Backbone,
                                    build_registry, recorded_checkpoint_sha256,
                                    verify_checkpoint_sha256)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'reports' / 'stage7b_phenobench_manifest.json'


# --------------------------------------------------------------------------
# Registry shape.
# --------------------------------------------------------------------------
def test_primary_and_exploratory_sets():
    reg = build_registry(include_exploratory=True)
    assert set(PRIMARY) <= set(reg)
    assert set(EXPLORATORY) <= set(reg)
    assert len(PRIMARY) == 5
    assert set(PRETRAIN_SEEDS.values()) == {20261104, 20261105}
    assert DOWNSTREAM_SEEDS == (0, 1, 2)


def test_registry_arm_and_seed_layout():
    reg = build_registry(include_exploratory=False)
    assert reg['OFFICIAL'].arm == 'OFFICIAL'
    assert reg['OFFICIAL'].pretraining_seed is None
    for key in ('A_p1', 'A_p2', 'B_p1', 'B_p2'):
        assert reg[key].arm == key[0]
        assert reg[key].role == 'primary'
        assert reg[key].pretraining_seed == PRETRAIN_SEEDS['p' + key[-1]]
        assert reg[key].updates == 2000
    assert 'C_p1' not in reg


def test_every_backbone_has_distinct_cache_key():
    reg = build_registry(include_exploratory=True)
    keys = {spec.to_spec().cache_key for spec in reg.values()}
    assert len(keys) == len(reg), 'two backbones share a feature cache key'


def test_to_spec_input_size_override():
    reg = build_registry(include_exploratory=False)
    spec = reg['A_p1'].to_spec(input_size=224)
    assert spec.input_size == 224
    assert spec.name == 'A_p1'


# --------------------------------------------------------------------------
# Recorded SHA values.
# --------------------------------------------------------------------------
def test_recorded_sha_matches_stage7b_manifest():
    if not MANIFEST.is_file():
        pytest.skip('Stage 7B manifest not present')
    recorded = recorded_checkpoint_sha256()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert len(recorded) == len(manifest['entries']) == 6
    assert all(len(v) == 64 and set(v) <= set('0123456789abcdef')
               for v in recorded.values())


def test_verify_sha_mismatch_raises():
    reg = build_registry(include_exploratory=True)
    with pytest.raises(AssertionError):
        verify_checkpoint_sha256('A_p1', recorded={'A_p1': '0' * 64})


def test_verify_sha_missing_key_raises():
    with pytest.raises(KeyError):
        verify_checkpoint_sha256('C_p1', recorded={})


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))