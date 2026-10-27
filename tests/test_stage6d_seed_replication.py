"""Stage 6D seed-replication tests.

These tests are entirely synthetic: they exercise the config generator, the
probe-key isolation rule and the cross-checkpoint aggregation on small JSON
fixtures.  They do not read checkpoints or run training, so they are fast and
deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.downstream.features import cache_key, stage6d_probe_key
from tools import stage6d_build_configs as cfgmod
from tools.check_stage6d_protocol import check_frozen, check_pair, run_checks
from tools.run_stage6d_probe import (
    build_comparison_manifest,
    probe_label,
    verify_probe_result,
)
from tools.stage6d_compare import (
    build_payload,
    checkpoint_mean_miou,
    paired_b1_b2,
    read_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# synthetic result.json helpers
# --------------------------------------------------------------------------- #

def _result(arm_label: str, sha: str, per_seed, *, val_images: int = 772,
            seeds=(0, 1, 2)):
    """per_seed: sequence of (soil, crop, weed, mIoU)."""
    return {
        'arm_label': arm_label,
        'backbone_checkpoint_sha256': sha,
        'protocol': {'val_images': val_images},
        'seeds': [
            {
                'seed': seed,
                'official_metrics': {
                    'IoU (crop)': crop, 'IoU (weed)': weed, 'mIoU': miou,
                },
                'self_metrics': {
                    'per_class_iou_percent': {'soil': soil},
                },
            }
            for seed, (soil, crop, weed, miou) in zip(seeds, per_seed, strict=True)
        ],
    }


def _write(tmp_path: Path, name: str, payload) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding='utf-8')
    return str(path)


def _sha(char: str) -> str:
    return char * 64


# --------------------------------------------------------------------------- #
# config plan
# --------------------------------------------------------------------------- #

def test_config_plan_has_paired_seeds_and_unique_run_ids():
    plan = cfgmod.build_plan()
    assert plan['ok'], plan['problems']
    run_ids = [run['run_id'] for run in plan['runs']]
    assert len(run_ids) == len(set(run_ids))
    seeds = [run['seed'] for run in plan['runs']]
    assert all(seed != cfgmod.HISTORICAL_SEED for seed in seeds)
    # each pair has one B1 and one B2 with the same seed
    by_pair = {}
    for run in plan['runs']:
        by_pair.setdefault(run['pair_id'], {})[run['arm']] = run['seed']
    for pair_id, arms in by_pair.items():
        assert set(arms) == {'B1', 'B2'}, pair_id
        assert len(set(arms.values())) == 1, pair_id


def test_config_plan_changes_only_allowed_fields():
    plan = cfgmod.build_plan()
    assert plan['ok'], plan['problems']
    for run in plan['runs']:
        arm = run['arm']
        seed = run['seed']
        cfg = run['config_payload']
        assert cfg['data']['seed'] == seed
        assert cfg['data']['augmentation_seed'] == seed
        if arm == 'B2':
            assert cfg['data']['sampling']['seed'] == seed
        # protocol fields stay frozen relative to the Stage 6 base
        import yaml
        base = yaml.safe_load(cfgmod.BASE_CONFIGS[arm].read_text(encoding='utf-8'))
        assert cfgmod.assert_only_allowed_changes(arm, base, cfg) == []


def test_config_plan_detects_historical_seed_collision(monkeypatch):
    monkeypatch.setattr(cfgmod, 'PAIRS',
                        ({'pair_id': 'PX', 'seed': cfgmod.HISTORICAL_SEED},))
    plan = cfgmod.build_plan()
    assert not plan['ok']
    assert any('historical seed' in problem for problem in plan['problems'])


def test_generated_configs_match_plan():
    assert cfgmod.main(['--check']) == 0


# --------------------------------------------------------------------------- #
# probe key isolation
# --------------------------------------------------------------------------- #

def test_stage6d_probe_key_binds_every_component():
    base = stage6d_probe_key('run1', _sha('a'), _sha('b'), 0)
    assert base == stage6d_probe_key('run1', _sha('a'), _sha('b'), 0)
    assert base != stage6d_probe_key('run2', _sha('a'), _sha('b'), 0)
    assert base != stage6d_probe_key('run1', _sha('c'), _sha('b'), 0)
    assert base != stage6d_probe_key('run1', _sha('a'), _sha('d'), 0)
    assert base != stage6d_probe_key('run1', _sha('a'), _sha('b'), 1)
    assert base != stage6d_probe_key('run1', _sha('a'), _sha('b'), 0,
                                     protocol_version='v2')
    # full SHA is preserved, not truncated
    assert _sha('a') in base and _sha('b') in base


def test_stage6d_probe_key_rejects_short_sha():
    with pytest.raises(ValueError):
        stage6d_probe_key('run1', 'deadbeef', _sha('b'), 0)
    with pytest.raises(ValueError):
        stage6d_probe_key('run1', _sha('a'), 'deadbeef', 0)


def test_legacy_cache_key_still_short():
    assert cache_key('label', _sha('a')) == 'label__' + 'a' * 16


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #

def test_checkpoint_mean_is_probe_seed_mean(tmp_path):
    payload = _result('B1', _sha('1'), [(97.0, 80.0, 30.0, 69.0),
                                        (97.5, 81.0, 31.0, 69.83),
                                        (98.0, 82.0, 32.0, 70.67)])
    path = tmp_path / 'r.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    mean = checkpoint_mean_miou(payload)
    assert mean['crop'] == pytest.approx((80 + 81 + 82) / 3)
    assert mean['soil'] == pytest.approx((97 + 97.5 + 98) / 3)
    assert len(mean['_seed_values']['crop']) == 3


def test_aggregation_uses_checkpoint_means_not_probe_seeds(tmp_path):
    # ckpt 1: probe seeds spread 60/70/80 -> mean 70
    # ckpt 2: probe seeds spread 70/80/90 -> mean 80
    # The arm's std must be std([70, 80]) = 5, NOT std of the 6 probe values.
    A = _write(tmp_path, 'a.json', _result('A', _sha('0'), [(0, 50, 0, 50)] * 3))
    p1 = _write(tmp_path, 'b1a.json',
                _result('B1', _sha('1'), [(0, 60, 0, 60), (0, 70, 0, 70),
                                          (0, 80, 0, 80)]))
    p2 = _write(tmp_path, 'b1b.json',
                _result('B1', _sha('2'), [(0, 70, 0, 70), (0, 80, 0, 80),
                                          (0, 90, 0, 90)]))
    specs = [
        {'arm': 'A', 'run_id': 'A', 'pair_id': '', 'seed': 0,
         'traceable': True, 'result': A},
        {'arm': 'B1', 'run_id': 'r1', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': p1},
        {'arm': 'B1', 'run_id': 'r2', 'pair_id': 'P2', 'seed': 20261028,
         'traceable': True, 'result': p2},
    ]
    payload = build_payload(specs)
    summary = payload['arm_summaries']['B1']
    assert summary['n_checkpoints'] == 2
    assert summary['mIoU']['mean'] == pytest.approx(75.0)
    assert summary['mIoU']['std'] == pytest.approx(5.0)
    all_probe = [60, 70, 80, 70, 80, 90]
    assert summary['mIoU']['std'] != pytest.approx(float(np.std(all_probe)))


def test_compare_refuses_missing_probe_seed(tmp_path):
    payload = _result('B1', _sha('1'), [(0, 80, 0, 70), (0, 80, 0, 70)],
                      seeds=(0, 1))
    with pytest.raises(ValueError, match='probe seeds'):
        checkpoint_mean_miou(payload)


def test_compare_refuses_wrong_val_count(tmp_path):
    payload = _result('B1', _sha('1'),
                      [(0, 80, 0, 70), (0, 80, 0, 70), (0, 80, 0, 70)],
                      val_images=771)
    with pytest.raises(ValueError, match='val_images'):
        checkpoint_mean_miou(payload)


def test_paired_b1_b2_uses_same_seed(tmp_path):
    A = _write(tmp_path, 'a.json', _result('A', _sha('0'),
                                           [(0, 90, 0, 80)] * 3))
    B1 = _write(tmp_path, 'b1.json', _result('B1', _sha('1'),
                                             [(0, 91, 0, 81)] * 3))
    B2 = _write(tmp_path, 'b2.json', _result('B2', _sha('2'),
                                             [(0, 92, 0, 82)] * 3))
    specs = [
        {'arm': 'A', 'run_id': 'A', 'pair_id': '', 'seed': 0,
         'traceable': True, 'result': A},
        {'arm': 'B1', 'run_id': 'b1', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': B1},
        {'arm': 'B2', 'run_id': 'b2', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': B2},
    ]
    payload = build_payload(specs)
    pair = payload['paired_b1_b2']['per_pair'][0]
    assert pair['mIoU']['B2_minus_B1'] == pytest.approx(1.0)


def test_paired_b1_b2_rejects_mismatched_seed(tmp_path):
    B1 = _write(tmp_path, 'b1.json', _result('B1', _sha('1'), [(0, 91, 0, 81)] * 3))
    B2 = _write(tmp_path, 'b2.json', _result('B2', _sha('2'), [(0, 92, 0, 82)] * 3))
    specs = [
        {'arm': 'B1', 'run_id': 'b1', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': B1},
        {'arm': 'B2', 'run_id': 'b2', 'pair_id': 'P1', 'seed': 20261028,
         'traceable': True, 'result': B2},
    ]
    with pytest.raises(ValueError, match='paired by seed'):
        paired_b1_b2(read_checkpoints(specs))


def test_sensitivity_subset_excludes_historical(tmp_path):
    A = _write(tmp_path, 'a.json', _result('A', _sha('0'), [(0, 90, 0, 80)] * 3))
    b1_hist = _write(tmp_path, 'b1h.json', _result('B1', _sha('1'), [(0, 91, 0, 81)] * 3))
    b2_hist = _write(tmp_path, 'b2h.json', _result('B2', _sha('2'), [(0, 91, 0, 81)] * 3))
    b1_new = _write(tmp_path, 'b1n.json', _result('B1', _sha('3'), [(0, 92, 0, 82)] * 3))
    b2_new = _write(tmp_path, 'b2n.json', _result('B2', _sha('4'), [(0, 92, 0, 82)] * 3))
    specs = [
        {'arm': 'A', 'run_id': 'A', 'pair_id': '', 'seed': 0,
         'traceable': True, 'result': A},
        {'arm': 'B1', 'run_id': 'hist', 'pair_id': 'P0_historical',
         'seed': 20261026, 'traceable': False, 'result': b1_hist},
        {'arm': 'B2', 'run_id': 'hist', 'pair_id': 'P0_historical',
         'seed': 20261026, 'traceable': False, 'result': b2_hist},
        {'arm': 'B1', 'run_id': 'new', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': b1_new},
        {'arm': 'B2', 'run_id': 'new', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': b2_new},
    ]
    primary = build_payload(specs)
    sensitivity = build_payload(specs, subset_traceable=True)
    assert primary['n_checkpoints']['B1'] == 2
    assert sensitivity['n_checkpoints']['B1'] == 1
    assert sensitivity['n_checkpoints']['B2'] == 1
    # historical pair must not enter the paired B1/B2 table
    assert all(row['pair_id'] != 'P0_historical'
               for row in primary['paired_b1_b2']['per_pair'])


def test_run_id_uniqueness_and_arm_identity():
    plan = cfgmod.build_plan()
    for run in plan['runs']:
        assert run['config_payload']['meta']['run_id'] == run['run_id']
        assert run['config_payload']['meta']['pair_id'] == run['pair_id']
        assert run['config_payload']['meta']['pretraining_seed'] == run['seed']
        assert run['config_payload']['meta']['arm'] == run['arm']


# --------------------------------------------------------------------------- #
# protocol checker
# --------------------------------------------------------------------------- #

def test_protocol_check_passes_on_generated_configs():
    report = run_checks()
    assert report['ok'], report['problems']
    assert len(report['runs']) == 4


def test_protocol_check_detects_lr_change():
    plan = cfgmod.build_plan()
    run = plan['runs'][0]
    cfg = json.loads(json.dumps(run['config_payload']))
    cfg['optimization']['lr'] = 1e-3
    problems = check_frozen(run['arm'], cfg)
    assert any('optimization.lr' in problem for problem in problems)


def test_protocol_check_detects_manifest_sha_change():
    plan = cfgmod.build_plan()
    run = plan['runs'][0]
    cfg = json.loads(json.dumps(run['config_payload']))
    cfg['data']['manifest_sha256'] = '0' * 64
    problems = check_frozen(run['arm'], cfg)
    assert any('data.manifest_sha256' in problem for problem in problems)


def test_check_pair_rejects_seed_change():
    plan = cfgmod.build_plan()
    b1 = json.loads(json.dumps(plan['runs'][0]['config_payload']))
    b2 = json.loads(json.dumps(plan['runs'][1]['config_payload']))
    b2['data']['seed'] = 123
    b2['data']['augmentation_seed'] = 123
    problems = check_pair(b1, b2, 'P1')
    assert any('data.seed' in problem for problem in problems)


def test_check_pair_allows_sampling_only():
    plan = cfgmod.build_plan()
    b1 = plan['runs'][0]['config_payload']
    b2 = plan['runs'][1]['config_payload']
    assert check_pair(b1, b2, 'P1') == []


# --------------------------------------------------------------------------- #
# probe runner
# --------------------------------------------------------------------------- #

def test_probe_label_binds_inputs():
    a = probe_label('run1', _sha('a'), _sha('b'))
    assert a == probe_label('run1', _sha('a'), _sha('b'))
    assert a != probe_label('run2', _sha('a'), _sha('b'))
    assert a != probe_label('run1', _sha('c'), _sha('b'))
    assert a != probe_label('run1', _sha('a'), _sha('d'))
    assert len(a) < 200


def test_probe_verify_rejects_missing_predictions(tmp_path):
    pred_root = tmp_path / 'run' / 'seed0' / 'predictions' / 'semantics'
    pred_root.mkdir(parents=True)
    (pred_root.parents[1] / 'phenobench_eval_stdout.txt').write_text('x')
    result = {
        'backbone_checkpoint_sha256': _sha('a'),
        'protocol': {'val_images': 772},
        'seeds': [
            {'seed': seed, 'predictions_dir': str(pred_root),
             'metric_cross_check': {'agrees': True}}
            for seed in (0, 1, 2)
        ],
    }
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    problems = verify_probe_result(path, _sha('a'))
    assert any('predictions != 772' in problem for problem in problems)


def test_probe_verify_rejects_wrong_checkpoint_sha(tmp_path):
    result = {
        'backbone_checkpoint_sha256': _sha('b'),
        'protocol': {'val_images': 772},
        'seeds': [],
    }
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    problems = verify_probe_result(path, _sha('a'))
    assert any('checkpoint SHA' in problem for problem in problems)


def test_comparison_manifest_includes_historical_and_new(tmp_path):
    new = [{'arm': 'B1', 'run_id': 'stage6d_b1_natural_seed20261027',
            'pair_id': 'P1', 'seed': 20261027, 'traceable': True,
            'checkpoint': 'runs/x.pth', 'checkpoint_sha256': _sha('a')}]
    manifest = build_comparison_manifest(new, tmp_path)
    arms = [entry['arm'] for entry in manifest['entries']]
    assert arms.count('A') == 1
    assert arms.count('B1') == 2  # historical + new
    assert arms.count('B2') == 1
    assert any(entry['run_id'] == 'stage6d_b1_natural_seed20261027'
               for entry in manifest['entries'])
    assert manifest['probe_seeds'] == [0, 1, 2]


def test_compare_accepts_manifest_dict(tmp_path):
    A = _write(tmp_path, 'a.json', _result('A', _sha('0'), [(0, 90, 0, 80)] * 3))
    B1 = _write(tmp_path, 'b1.json', _result('B1', _sha('1'), [(0, 91, 0, 81)] * 3))
    manifest = {'entries': [
        {'arm': 'A', 'run_id': 'A', 'pair_id': '', 'seed': 0,
         'traceable': True, 'result': A},
        {'arm': 'B1', 'run_id': 'b1', 'pair_id': 'P1', 'seed': 20261027,
         'traceable': True, 'result': B1},
    ]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    # the CLI path is what accepts a dict manifest; exercise the same parsing
    from tools.stage6d_compare import main as compare_main
    prefix = tmp_path / 'out'
    assert compare_main(['--manifest', str(path), '--out-prefix', str(prefix)]) == 0
    assert (tmp_path / 'out.md').is_file()
    assert (tmp_path / 'out.json').is_file()
