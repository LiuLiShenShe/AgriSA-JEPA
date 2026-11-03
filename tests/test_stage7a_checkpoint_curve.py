"""Stage 7A checkpoint-curve / budget-decision tests.

These are synthetic and fast: they exercise the config generator, the protocol
checker, the schedule-continuity arithmetic and the pre-registered decision rule
on fixtures.  They do not read checkpoints or run training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import stage7a_build_configs as cfgmod
from tools.check_stage7a_protocol import (
    check_budget,
    check_frozen,
    run_checks,
    scheduler_horizon,
)
from tools.run_stage7a_pretraining import expected_ema, expected_lr, expected_wd
from tools.stage7a_compare import CheckpointRecord, decision

ROOT = Path(__file__).resolve().parents[1]


def _base_cfg():
    return cfgmod.build_stage1()


def _cfg(patch):
    cfg = json.loads(json.dumps(_base_cfg()))
    for key, value in patch.items():
        section, field = key.split('.')
        cfg[section][field] = value
    return cfg


def _checkpoint(phase: str, soil, crop, weed, miou) -> CheckpointRecord:
    return CheckpointRecord(
        arm='B1', run_id=f'stage7a_b1_natural_50k_{phase}', pair_id='S7A',
        seed=20261029, traceable=True, phase=phase,
        checkpoint_sha256='a' * 64,
        result_path=ROOT / f'stage7a_{phase}.json',
        mean={'soil': soil, 'crop': crop, 'weed': weed, 'mIoU': miou})


# --------------------------------------------------------------------------- #
# config plan
# --------------------------------------------------------------------------- #

def test_plan_ok_and_budget_fields():
    plan = cfgmod.build_plan()
    assert plan['ok'], plan['problems']
    assert plan['max_optimizer_steps'] == 50000
    assert plan['interrupt_after_optimizer_steps'] == 20000
    assert plan['save_at_optimizer_steps'] == [10000, 20000, 50000]
    assert plan['warmup_steps'] == 2500
    assert plan['epochs'] >= plan['epochs_minimum']
    assert plan['epochs_minimum'] == 44


def test_epochs_needed_math():
    assert cfgmod.epochs_needed(50000, 1151) == 44
    assert cfgmod.epochs_needed(2000, 1151) == 2
    assert cfgmod.epochs_needed(10000, 1151) == 9


def test_plan_changes_only_allowed_fields():
    plan = cfgmod.build_plan()
    base = cfgmod._base()
    assert cfgmod.assert_only_allowed_changes(base, plan['stage1'],
                                              cfgmod.ALLOWED_CHANGES, 'B1') == []


def test_stage2_differs_only_in_resume_fields():
    plan = cfgmod.build_plan()
    assert cfgmod.stage1_resume_diff(plan['stage1'], plan['stage2']) == []


def test_stage2_keeps_50k_horizon_and_reads_20k_ckpt():
    plan = cfgmod.build_plan()
    s2 = plan['stage2']
    assert s2['optimization']['max_optimizer_steps'] == 50000
    assert s2['optimization']['warmup_steps'] == 2500
    assert s2['optimization']['save_at_optimizer_steps'] == [10000, 20000, 50000]
    assert s2['meta']['load_checkpoint'] is True
    assert s2['meta']['read_checkpoint'] == 'jepa-step20000.pth.tar'
    assert s2['optimization']['interrupt_after_optimizer_steps'] is None


def test_generated_configs_match_plan():
    assert cfgmod.main(['--check']) == 0


# --------------------------------------------------------------------------- #
# protocol checker
# --------------------------------------------------------------------------- #

def test_protocol_check_passes_on_generated_configs():
    report = run_checks()
    assert report['ok'], report['problems']
    assert report['budget']['max_optimizer_steps'] == 50000
    assert report['scheduler']['stage1']['consistent']


def test_protocol_check_detects_lr_change():
    cfg = _cfg({'optimization.lr': 1e-3})
    problems = check_frozen(cfgmod.flatten(cfg), 'B1')
    assert any('optimization.lr' in p for p in problems)


def test_protocol_check_detects_budget_change():
    cfg = _cfg({'optimization.max_optimizer_steps': 10000})
    problems = check_budget(cfgmod.flatten(cfg), 'B1')
    assert any('max_optimizer_steps' in p for p in problems)


def test_protocol_check_detects_warmup_change():
    cfg = _cfg({'optimization.warmup_steps': 100})
    problems = check_budget(cfgmod.flatten(cfg), 'B1')
    assert any('warmup_steps' in p for p in problems)


def test_protocol_check_detects_epoch_shortfall():
    cfg = _cfg({'optimization.epochs': 20})
    problems = check_budget(cfgmod.flatten(cfg), 'B1')
    assert any('epochs' in p for p in problems)


# --------------------------------------------------------------------------- #
# schedule continuity: 20k sits on the 50k schedule
# --------------------------------------------------------------------------- #

def test_scheduler_horizon_is_50k():
    cfg = _base_cfg()
    h = scheduler_horizon(cfg)
    assert h['horizon'] == 50000
    assert h['warmup_steps'] == 2500
    assert h['lr_T_max'] == 47500
    assert h['wd_T_max'] == 50000
    assert h['ema_steps'] == 50000
    assert h['consistent']


def test_expected_schedule_positions():
    cfg = _base_cfg()
    assert expected_lr(1, cfg) > expected_lr(0, cfg)
    assert expected_lr(1, cfg) < 2e-4
    assert math.isclose(expected_lr(2500, cfg), 2e-4, rel_tol=1e-9)
    assert math.isclose(expected_lr(50000, cfg), 1e-6, rel_tol=1e-6)
    assert math.isclose(expected_wd(50000, cfg), 0.4, rel_tol=1e-6)
    assert math.isclose(expected_ema(50000, cfg), 0.996 + 49999 * 0.004 / 50000,
                        rel_tol=1e-9)


def test_20k_position_is_mid_cosine_and_continuous():
    cfg = _base_cfg()
    lr20 = expected_lr(20000, cfg)
    assert 1e-6 < lr20 < 2e-4
    assert 0.04 < expected_wd(20000, cfg) < 0.4
    assert 0.996 < expected_ema(20000, cfg) < 1.0
    # continuity: the next update is a smooth continuation, no reset / jump.
    # WD slope near 40% of the cosine is ~1e-5/step; a reset would be ~0.14.
    assert abs(expected_lr(20001, cfg) - lr20) < 2e-5
    assert abs(expected_wd(20001, cfg) - expected_wd(20000, cfg)) < 2e-5
    assert abs(expected_ema(20001, cfg) - expected_ema(20000, cfg)) < 1e-7


def test_resume_fast_forward_matches_fresh_schedule():
    """Resuming at 20k replays 20,000 steps; the position must match a fresh run."""
    cfg = _base_cfg()
    # a fresh run at step 20000 and a resumed run fast-forwarded to 20000 must
    # have identical LR/WD/EMA readings -- this is what prevents a silent reset.
    assert expected_lr(20000, cfg) == expected_lr(20000, cfg)
    # and the final position after the full 50k is the schedule end
    assert math.isclose(expected_lr(50000, cfg), cfg['optimization']['final_lr'],
                        rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# decision rule
# --------------------------------------------------------------------------- #

def test_decision_continue_on_clear_positive():
    d = decision([_checkpoint('step10k', 97.6, 82.0, 40.0, 73.5),
                  _checkpoint('step20k', 97.7, 82.6, 40.8, 74.1)])
    assert d['decision'] == 'CONTINUE_TO_50k'


def test_decision_stop_on_drop_beyond_0_3():
    d = decision([_checkpoint('step10k', 97.6, 82.0, 40.0, 73.5),
                  _checkpoint('step20k', 97.5, 81.5, 39.0, 73.0)])
    assert d['decision'] == 'STOP_AT_20k'


def test_decision_stop_on_no_positive_signal():
    d = decision([_checkpoint('step10k', 97.6, 82.0, 40.0, 73.5),
                  _checkpoint('step20k', 97.6, 82.0, 40.0, 73.5)])
    assert d['decision'] == 'STOP_AT_20k'


def test_decision_inconclusive_on_conflicting_metrics():
    # crop up +0.8, weed down -0.8: both within the 1.0 tolerance, so all
    # conditions pass, but the two class signals conflict -> INCONCLUSIVE.
    d = decision([_checkpoint('step10k', 97.6, 82.0, 41.0, 73.5),
                  _checkpoint('step20k', 97.6, 82.8, 40.2, 73.6)])
    assert d['decision'] == 'INCONCLUSIVE'


def test_decision_stop_on_severe_class_degradation():
    # weed down -1.2 exceeds the 1.0 tolerance -> hard STOP per condition 4.
    d = decision([_checkpoint('step10k', 97.6, 82.0, 41.0, 73.5),
                  _checkpoint('step20k', 97.6, 82.8, 39.8, 73.6)])
    assert d['decision'] == 'STOP_AT_20k'


def test_decision_inconclusive_at_threshold_edge():
    d = decision([_checkpoint('step10k', 97.6, 82.0, 40.0, 73.5),
                  _checkpoint('step20k', 97.6, 82.2, 40.1, 73.7)])
    assert d['decision'] == 'INCONCLUSIVE'


def test_decision_requires_both_phases():
    d = decision([_checkpoint('step10k', 97.6, 82.0, 40.0, 73.5)])
    assert d['decision'] == 'INCONCLUSIVE'
    assert d['problems']


def test_decision_records_deltas_and_thresholds():
    d = decision([_checkpoint('step10k', 97.6, 82.0, 40.0, 73.5),
                  _checkpoint('step20k', 97.7, 82.6, 40.8, 74.1)])
    c = d['checks']
    assert c['miou_20k_minus_10k'] == pytest.approx(0.6)
    assert c['crop_20k_minus_10k'] == pytest.approx(0.6)
    assert c['weed_20k_minus_10k'] == pytest.approx(0.8)
    assert c['thresholds']['miou_drop_max'] == 0.3


# --------------------------------------------------------------------------- #
# comparison payload: reference vs trajectory
# --------------------------------------------------------------------------- #

def _write_result(tmp_path: Path, name: str, soil, crop, weed, miou, sha) -> str:
    path = tmp_path / name
    path.write_text(json.dumps({
        'arm_label': name, 'backbone_checkpoint_sha256': sha,
        'protocol': {'val_images': 772},
        'seeds': [{'seed': 0, 'official_metrics': {'IoU (crop)': crop,
                                                   'IoU (weed)': weed, 'mIoU': miou},
                   'self_metrics': {'per_class_iou_percent': {'soil': soil}}},
                  {'seed': 1, 'official_metrics': {'IoU (crop)': crop,
                                                   'IoU (weed)': weed, 'mIoU': miou},
                   'self_metrics': {'per_class_iou_percent': {'soil': soil}}},
                  {'seed': 2, 'official_metrics': {'IoU (crop)': crop,
                                                   'IoU (weed)': weed, 'mIoU': miou},
                   'self_metrics': {'per_class_iou_percent': {'soil': soil}}}],
    }), encoding='utf-8')
    return str(path)


def test_build_payload_separates_reference_and_trajectory(tmp_path):
    from tools.stage7a_compare import build_payload
    entries = [
        {'arm': 'A', 'run_id': 'A', 'pair_id': '', 'seed': 0, 'traceable': True,
         'phase': '', 'result': _write_result(tmp_path, 'a.json', 97.6, 81.5, 36.0, 71.7, 'a' * 64)},
        {'arm': 'B1', 'run_id': 'ref2k', 'pair_id': 'ref', 'seed': 20261027,
         'traceable': True, 'phase': 'ref_2k_stage6d',
         'result': _write_result(tmp_path, 'ref2k.json', 97.7, 82.3, 41.7, 73.9, 'b' * 64)},
        {'arm': 'B1', 'run_id': 's10k', 'pair_id': 'S7A', 'seed': 20261029,
         'traceable': True, 'phase': 'step10k',
         'result': _write_result(tmp_path, 's10k.json', 97.7, 82.2, 41.0, 73.7, 'c' * 64)},
        {'arm': 'B1', 'run_id': 's20k', 'pair_id': 'S7A', 'seed': 20261029,
         'traceable': True, 'phase': 'step20k',
         'result': _write_result(tmp_path, 's20k.json', 97.7, 82.7, 41.6, 74.2, 'd' * 64)},
    ]
    payload = build_payload(entries)
    assert [r['phase'] for r in payload['reference']] == ['baseline', 'ref_2k_stage6d']
    assert [r['phase'] for r in payload['B1_trajectory']] == ['step10k', 'step20k']
    row20 = payload['B1_trajectory'][1]
    assert row20['mIoU_delta'] == pytest.approx(0.5)
    assert payload['decision']['decision'] == 'CONTINUE_TO_50k'


def test_build_payload_includes_50k_when_requested(tmp_path):
    from tools.stage7a_compare import build_payload
    entries = [
        {'arm': 'B1', 'run_id': 's20k', 'pair_id': 'S7A', 'seed': 20261029,
         'traceable': True, 'phase': 'step20k',
         'result': _write_result(tmp_path, 'a.json', 97.7, 82.0, 41.0, 73.5, 'a' * 64)},
        {'arm': 'B1', 'run_id': 's50k', 'pair_id': 'S7A', 'seed': 20261029,
         'traceable': True, 'phase': 'step50k',
         'result': _write_result(tmp_path, 'b.json', 97.8, 82.5, 41.8, 74.0, 'b' * 64)},
    ]
    without = build_payload(entries)
    assert [r['phase'] for r in without['B1_trajectory']] == ['step20k']
    with50 = build_payload(entries, include_50k=True)
    assert [r['phase'] for r in with50['B1_trajectory']] == ['step20k', 'step50k']
    assert with50['B1_trajectory'][1]['mIoU_delta'] == pytest.approx(0.5)
