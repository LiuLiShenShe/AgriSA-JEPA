"""Tests for the Stage 2C exact optimizer-step budget and mid-epoch resume.

CPU-only. The training loop itself is not run here (it needs the dataloader and
a GPU); these tests cover the pure functions the loop relies on and the
arithmetic that a mid-epoch budget stop has to get right.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.train import (
    accumulation_window,
    capture_rng_state,
    epochs_needed_for_optimizer_steps,
    restore_rng_state,
)


class TestEpochsNeededForOptimizerSteps:

    def test_ceiling_division(self):
        assert epochs_needed_for_optimizer_steps(500, 98) == 6
        assert epochs_needed_for_optimizer_steps(98, 98) == 1
        assert epochs_needed_for_optimizer_steps(97, 98) == 1

    def test_exact_multiple_is_exact(self):
        assert epochs_needed_for_optimizer_steps(196, 98) == 2
        assert epochs_needed_for_optimizer_steps(30000, 98) == 307

    def test_single_step_budget(self):
        assert epochs_needed_for_optimizer_steps(1, 98) == 1

    @pytest.mark.parametrize('bad', [0, -1])
    def test_rejects_non_positive_budget(self, bad):
        with pytest.raises(ValueError, match='must be >= 1'):
            epochs_needed_for_optimizer_steps(bad, 98)

    def test_rejects_non_positive_per_epoch(self):
        with pytest.raises(ValueError, match='optimizer_steps_per_epoch must be >= 1'):
            epochs_needed_for_optimizer_steps(10, 0)


class TestAccumulationWindow:

    def test_first_micro_step_of_a_window_is_not_an_update(self):
        window_start, window_size, do_update = accumulation_window(0, ipe=777, grad_accum=8)
        assert window_start == 0
        assert window_size == 8
        assert do_update is False

    def test_last_micro_step_of_a_window_is_an_update(self):
        window_start, window_size, do_update = accumulation_window(7, ipe=777, grad_accum=8)
        assert window_start == 0
        assert window_size == 8
        assert do_update is True

    def test_second_window(self):
        window_start, window_size, do_update = accumulation_window(8, ipe=777, grad_accum=8)
        assert (window_start, window_size, do_update) == (8, 8, False)
        window_start, window_size, do_update = accumulation_window(15, ipe=777, grad_accum=8)
        assert (window_start, window_size, do_update) == (8, 8, True)

    def test_partial_final_window_still_updates(self):
        # 777 micro-steps, accum 8 -> final window has only 777-776=1 micro-step
        window_start, window_size, do_update = accumulation_window(776, ipe=777, grad_accum=8)
        assert window_start == 776
        assert window_size == 1
        assert do_update is True

    def test_exactly_divisible_epoch_has_no_partial_window(self):
        window_start, window_size, do_update = accumulation_window(7, ipe=8, grad_accum=8)
        assert (window_start, window_size, do_update) == (0, 8, True)

    def test_grad_accum_one_updates_every_step(self):
        for itr in range(4):
            assert accumulation_window(itr, ipe=10, grad_accum=1) == (itr, 1, True)

    def test_window_is_pure_in_itr_ipe_accum(self):
        # the property a mid-epoch resume depends on: no hidden state
        for itr in range(20):
            assert accumulation_window(itr, 20, 4) == accumulation_window(itr, 20, 4)

    def test_rejects_zero_grad_accum(self):
        with pytest.raises(ValueError, match='grad_accum must be >= 1'):
            accumulation_window(0, 10, 0)

    def test_updates_per_epoch_count_matches_windows(self):
        # the invariant src/train.py relies on: ceil(ipe/grad_accum) updates
        for ipe, grad_accum in [(777, 8), (80, 8), (16, 8), (1, 8), (100, 3), (9, 8)]:
            updates = sum(1 for itr in range(ipe)
                         if accumulation_window(itr, ipe, grad_accum)[2])
            assert updates == -(-ipe // grad_accum)


class TestRngStateRoundTrip:

    def test_restore_reproduces_random_draws(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        before = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        restore_rng_state(before)
        replayed = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert replayed == pytest.approx(expected)

    def test_restore_after_mutation_is_exact_for_torch(self):
        torch.manual_seed(1234)
        before = capture_rng_state()
        torch.manual_seed(9999)  # perturb
        restore_rng_state(before)
        first = torch.rand(3)
        restore_rng_state(before)
        assert torch.equal(first, torch.rand(3))

    def test_numpy_legacy_state_survives(self):
        np.random.seed(7)
        before = capture_rng_state()
        np.random.seed(11)
        restore_rng_state(before)
        first = np.random.rand(4)
        restore_rng_state(before)
        assert np.array_equal(first, np.random.rand(4))

    def test_capture_and_restore_are_inverses_on_python_random(self):
        random.seed(3)
        state = capture_rng_state()
        draws = [random.randint(0, 1_000_000) for _ in range(5)]
        restore_rng_state(state)
        assert [random.randint(0, 1_000_000) for _ in range(5)] == draws


class TestBudgetArithmetic:
    def test_500_updates_with_777_micro_steps_stops_mid_epoch(self):
        ipe, grad_accum = 777, 8
        updates_per_epoch = -(-ipe // grad_accum)
        budget = 500
        epochs_needed = epochs_needed_for_optimizer_steps(budget, updates_per_epoch)
        assert epochs_needed == 6
        # update 500 lands inside epoch 6 (0-indexed epoch 5), not at its end
        updates_before_last_epoch = updates_per_epoch * (epochs_needed - 1)
        assert updates_before_last_epoch < budget <= updates_per_epoch * epochs_needed
        # micro-steps consumed in that final epoch at the budget stop
        updates_in_last = budget - updates_before_last_epoch
        micro_steps_in_last = updates_in_last * grad_accum
        assert micro_steps_in_last < ipe, 'stop must be mid-epoch for this case'

    def test_500_updates_with_80_micro_steps_needs_50_epochs(self):
        # the Stage 2 smoke config: 80 images, batch 1 -> 80 micro-steps
        ipe, grad_accum = 80, 8
        updates_per_epoch = -(-ipe // grad_accum)
        assert epochs_needed_for_optimizer_steps(500, updates_per_epoch) == 50

    def test_budget_equal_to_one_epoch_is_not_mid_epoch(self):
        ipe, grad_accum = 777, 8
        updates_per_epoch = -(-ipe // grad_accum)
        assert epochs_needed_for_optimizer_steps(updates_per_epoch, updates_per_epoch) == 1

    def test_resume_does_not_restart_at_zero(self):
        # the bug this guards: resume must restore the recorded update count, not
        # start_epoch * updates_per_epoch (they differ once a stop is mid-epoch)
        completed = 500
        per_epoch = 98
        start_epoch = epochs_needed_for_optimizer_steps(completed, per_epoch)  # 6
        epoch_itr = completed % per_epoch  # updates into that epoch
        epoch_itr = epoch_itr if epoch_itr else per_epoch
        # the reconstruction rule used by src/train.py
        reconstructed = (start_epoch - 1) * per_epoch + epoch_itr
        assert reconstructed == completed, (
            f'resume reconstruction {reconstructed} must equal recorded {completed}')
        # and the naive epoch-based position is wrong for this case
        assert start_epoch * per_epoch != completed
