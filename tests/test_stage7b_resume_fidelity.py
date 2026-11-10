#!/usr/bin/env python
"""Stage 7B strict resume-fidelity tests.

Two layers:

* **Unit tests** (fast, no GPU): the gradient-accumulation normalization rule
  (``accumulation_divisor``), partial-window windowing
  (``accumulation_window``), and the pure trace / ledger / state-dict
  comparison logic used by the GPU experiment.
* **GPU integration test** (skipped unless ``STAGE7B_RUN_GPU=1``): runs
  ``tools.stage7b_resume_fidelity`` and asserts a continuous run and an
  interrupted-and-resumed run produce the identical data stream, counts and
  (within tolerance) final state.

The GPU test needs a free GPU1 and a clean worktree; it is NOT part of the
normal fast suite.  Run it explicitly with::

    STAGE7B_RUN_GPU=1 ./.venv-uv/bin/python -m pytest tests/test_stage7b_resume_fidelity.py -k gpu
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.train import accumulation_divisor, accumulation_window
from tools.stage7b_resume_fidelity import (
    INTERRUPT,
    MAX_STEPS,
    ROOT,
    compare_ledgers,
    compare_state_dicts,
    compare_traces,
    read_ledger,
    read_trace,
)


# --------------------------------------------------------------------------- #
# Unit tests: gradient accumulation normalization (3.3)
# --------------------------------------------------------------------------- #
class TestAccumulationDivisor:
    def test_full_window_uses_grad_accum(self):
        assert accumulation_divisor(8, 8) == 8

    def test_partial_window_uses_actual_size(self):
        # the Stage 7B bug: this returned 8 before the fix, halving the
        # gradient of a 4-micro-batch epoch-end window
        assert accumulation_divisor(4, 8) == 4
        assert accumulation_divisor(6, 8) == 6
        assert accumulation_divisor(1, 8) == 1

    def test_rejects_bad_sizes(self):
        with pytest.raises(ValueError):
            accumulation_divisor(9, 8)
        with pytest.raises(ValueError):
            accumulation_divisor(0, 8)
        with pytest.raises(ValueError):
            accumulation_divisor(4, 0)


class TestAccumulationWindowPartial:
    def test_ipe_not_divisible(self):
        # ipe=190, grad_accum=8 -> 23 full windows + one partial window (6)
        # full window [0..7]: update fires at itr=7
        assert accumulation_window(0, 190, 8) == (0, 8, False)
        assert accumulation_window(7, 190, 8) == (0, 8, True)
        assert accumulation_window(8, 190, 8) == (8, 8, False)
        # partial window [184..189]: size 6, update at itr=189
        assert accumulation_window(184, 190, 8) == (184, 6, False)
        assert accumulation_window(189, 190, 8) == (184, 6, True)

    def test_divisible(self):
        # ipe=192 -> 24 full windows, no partial
        assert accumulation_window(191, 192, 8) == (184, 8, True)


# --------------------------------------------------------------------------- #
# Unit tests: pure comparison logic used by the GPU experiment
# --------------------------------------------------------------------------- #
def _line(**kw):
    base = {'epoch': 0, 'itr': 0, 'micro_step': 1, 'optimizer_step': 0,
            'window_size': 8, 'lr': 1e-4, 'wd': 0.04,
            'imgs_sha': 'a' * 16, 'mask_enc_sha': 'b' * 16,
            'mask_pred_sha': 'c' * 16, 'sample_ids': ['s0', 's1']}
    base.update(kw)
    return base


class TestCompareTraces:
    def test_identical(self):
        a = [_line(itr=i, micro_step=i + 1, optimizer_step=i // 8)
             for i in range(16)]
        assert compare_traces(a, a)['ok'] is True

    def test_detects_off_by_one(self):
        # The old off-by-one re-processed the last consumed micro-batch on
        # resume, so B emits ONE EXTRA trained micro-batch (a duplicate of
        # itr=11 with a restarted micro-step counter).  The length mismatch is
        # the primary signal; the duplicated itr/micro_step pair is secondary.
        a = [_line(itr=i, micro_step=i + 1, optimizer_step=i // 8)
             for i in range(12)]
        b = [_line(itr=i, micro_step=i + 1, optimizer_step=i // 8)
             for i in range(12)]
        b.append(_line(itr=11, micro_step=13, optimizer_step=1))  # duplicate itr=11
        result = compare_traces(a, b)
        assert result['ok'] is False
        assert result['n_a'] == 12 and result['n_b'] == 13

    def test_detects_lr_drift(self):
        a = [_line(itr=i) for i in range(4)]
        b = [_line(itr=i) for i in range(4)]
        b[2]['lr'] = 1.00001e-4
        result = compare_traces(a, b, rtol=1e-9)
        assert result['ok'] is False
        assert result['float_mismatch']['lr'] >= 1


class TestCompareLedgers:
    def test_identical(self):
        a = [{'micro_step': str(i + 1), 'sample_id': f's{i}',
              'physical_source_id': 'src'} for i in range(5)]
        assert compare_ledgers(a, list(a))['ok'] is True

    def test_detects_different_stream(self):
        a = [{'micro_step': str(i + 1), 'sample_id': f's{i}',
              'physical_source_id': 'src'} for i in range(5)]
        b = [{'micro_step': str(i + 1), 'sample_id': f's{i + 1}',
              'physical_source_id': 'src'} for i in range(5)]
        assert compare_ledgers(a, b)['ok'] is False


class TestCompareStateDicts:
    def test_identical_tensors(self):
        import torch
        a = {'w': torch.randn(4, 4)}
        assert compare_state_dicts(a, a)['ok'] is True

    def test_within_tolerance(self):
        import torch
        a = {'w': torch.ones(4, 4)}
        b = {'w': torch.ones(4, 4) + 1e-4}
        result = compare_state_dicts(a, b, rtol=1e-3, atol=1e-5)
        # scale=1, tol=1e-5+1e-3=1.01e-3 > 1e-4 -> ok
        assert result['ok'] is True

    def test_beyond_tolerance(self):
        import torch
        a = {'w': torch.zeros(4, 4)}
        b = {'w': torch.ones(4, 4)}
        result = compare_state_dicts(a, b)
        assert result['ok'] is False


# --------------------------------------------------------------------------- #
# GPU integration (skipped in the normal suite)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.environ.get('STAGE7B_RUN_GPU'),
                    reason='set STAGE7B_RUN_GPU=1 to run the GPU fidelity experiment')
def test_gpu_resume_fidelity():
    import subprocess
    report_path = ROOT / 'reports' / 'stage7b_resume_test.json'
    if report_path.exists():
        report_path.unlink()
    result = subprocess.run(
        ['python', '-m', 'tools.stage7b_resume_fidelity', '--execute', '--device', '1'],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report_path.read_text(encoding='utf-8'))
    assert payload['ok'] is True, json.dumps(payload, ensure_ascii=False, indent=2)
    for scenario, run in payload['scenarios'].items():
        assert run['trace']['ok'], f"{scenario} trace differs: {run['trace']['problems']}"
        assert run['ledger']['ok'], f"{scenario} ledger differs: {run['ledger']['problems']}"
        assert run['checkpoint']['ok'], f"{scenario} weights differ: {run['checkpoint']['problems']}"
        assert run['counts']['optimizer_steps_total']['a'] == MAX_STEPS
        assert run['counts']['optimizer_steps_total']['b'] == MAX_STEPS


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
