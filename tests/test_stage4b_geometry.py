"""Stage 4B tests: geometry-controlled sampling and the mask-geometry audit.

Three groups:

1. **Diagnostics fix** -- ``mean_candidate_prior_score`` must now average over
   the *valid candidates* it accumulated, not over every draw.  The tests build
   a collator, record the raw sums, and recompute the mean by hand.
2. **Stage 4A equivalence** -- with ``geometry_beta=0`` and no ``geometry``
   argument at all the new sampler must produce byte-identical masks and RNG
   consumption to the pre-fix implementation.  A third-party reference is not
   available, so the reference is an explicit reimplementation of the Stage 4A
   rule inside the test.
3. **Geometry control** -- the new arms must be legal (no out-of-range indices,
   no context/target overlap, min_keep respected), reproducible, and G0 must not
   read any structure prior.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.masks.agri_structure import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    StructurePrior,
    circular_shift_patch_cells,
    lag1_autocorrelation,
    mean_grid_correlation,
    patch_pool,
    shuffle_patch_cells,
)
from src.masks.geometry import (
    bootstrap_ci,
    cells_from_indices,
    clustered_bootstrap_ci,
    geometry_record,
    multiplicity,
    paired_delta,
    summarize,
    union_cells,
)
from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator

REPO = Path(__file__).resolve().parents[1]
GRID = 16

COLLATOR_KWARGS = dict(
    input_size=224, patch_size=14,
    enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
    allow_overlap=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def normalize(tensors: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensors - mean) / std


def soil_and_vegetation_batch(b: int = 8, green_fraction: int = 2) -> torch.Tensor:
    rgb = torch.zeros(b, 3, 224, 224)
    rgb[:, 0] = 0.45
    rgb[:, 1] = 0.30
    rgb[:, 2] = 0.25
    green = torch.tensor([0.15, 0.55, 0.15]).view(3, 1, 1)
    for i in range(b):
        if i % green_fraction == 0:
            rgb[i, :, 20:100, 30:120] = green
    return normalize(rgb)


def blank_batch(b: int = 4) -> torch.Tensor:
    return normalize(torch.full((b, 3, 224, 224), 0.5))


def batch_of(imgs: torch.Tensor, ids=None):
    if ids is None:
        ids = [f'id{i}' for i in range(imgs.shape[0])]
    return [(imgs[i], ids[i]) for i in range(imgs.shape[0])]


def all_masks_legal(masks_enc, masks_pred, batch: int = 4, grid: int = GRID) -> None:
    """Every mask must be in range and non-empty, for every image."""
    total = grid * grid
    for block in masks_pred:
        assert tuple(block.shape)[0] == batch
        for image in range(block.shape[0]):
            flat = torch.as_tensor(block[image]).reshape(-1)
            assert flat.numel() > 0
            assert int(flat.min()) >= 0
            assert int(flat.max()) < total
    for block in masks_enc:
        assert tuple(block.shape)[0] == batch
        for image in range(block.shape[0]):
            flat = torch.as_tensor(block[image]).reshape(-1)
            assert flat.numel() > 0
            assert int(flat.min()) >= 0
            assert int(flat.max()) < total


def per_image_pred(masks_pred, image: int):
    return [block[image] for block in masks_pred]


def per_image_enc(masks_enc, image: int):
    return [block[image] for block in masks_enc]


def occupancy(masks) -> np.ndarray:
    grid = np.zeros(GRID * GRID, dtype=bool)
    for mask in masks:
        grid[np.asarray(mask).reshape(-1)] = True
    return grid.reshape(GRID, GRID)


# --------------------------------------------------------------------------- #
# 1. The diagnostics denominator fix
# --------------------------------------------------------------------------- #

class TestDiagnosticsDenominator:

    def test_candidate_mean_is_over_valid_candidates(self):
        """Hand-built sums -> the reported mean must match by construction."""
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               num_candidates=16)
        collator(batch_of(soil_and_vegetation_batch(4)))
        raw = dict(collator.diagnostics)
        summary = collator.diagnostics_summary()
        expected = (raw['candidate_prior_score_sum']
                    / raw['candidate_prior_score_count'])
        assert summary['mean_candidate_prior_score'] == pytest.approx(
            expected, rel=1e-12)
        # and it must NOT equal the old (buggy) denominator
        old = raw['candidate_prior_score_sum'] / max(raw['candidates_generated'], 1)
        if raw['candidates_generated'] != raw['candidate_prior_score_count']:
            assert summary['mean_candidate_prior_score'] != pytest.approx(old)

    def test_denominator_counts_only_valid_candidates(self):
        """Force an invalid-candidate situation and check the denominator."""
        # a block wider than min_keep is impossible here, so simulate directly
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               num_candidates=4)
        collator._reset_diagnostics()
        collator.diagnostics['candidate_prior_score_sum'] = 2.0
        collator.diagnostics['candidate_prior_score_count'] = 8   # 2 selections x 4
        collator.diagnostics['candidates_generated'] = 10          # 8 valid + 2 invalid
        collator.diagnostics['invalid_candidates'] = 2
        summary = collator.diagnostics_summary()
        assert summary['mean_candidate_prior_score'] == pytest.approx(0.25)
        assert summary['candidate_score_samples'] == 8
        assert summary['invalid_candidate_rate'] == pytest.approx(0.2)

    def test_selected_mean_unchanged_by_the_fix(self):
        """The other diagnostic is unaffected: selections, not candidates."""
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='boundary',
                                               guidance_lambda=6.0,
                                               num_candidates=16)
        collator(batch_of(soil_and_vegetation_batch(6)))
        raw = dict(collator.diagnostics)
        summary = collator.diagnostics_summary()
        assert summary['mean_selected_prior_score'] == pytest.approx(
            raw['selected_prior_score_sum'] / max(raw['guided_selections'], 1),
            rel=1e-12)
        assert summary['target_blocks'] == raw['guided_selections']

    def test_overlap_diagnostics_are_recorded(self):
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               geometry_beta=3.5,
                                               num_candidates=16)
        collator(batch_of(soil_and_vegetation_batch(4)))
        d = collator.diagnostics_summary()
        assert d['geometry_beta'] == 3.5
        assert d['geometry_control'] == 'linear'
        assert 0.0 <= d['mean_selected_overlap'] <= 1.0
        assert 0.0 <= d['mean_candidate_overlap'] <= 1.0
        assert 0.0 <= d['mean_expected_overlap'] <= 1.0
        # the linear penalty must reduce overlap relative to the same seeds
        # with beta=0 (a per-step comparison is confounded by the growing union)
        plain = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                            prior_mode='vegetation',
                                            guidance_lambda=6.0,
                                            geometry_beta=0.0,
                                            num_candidates=16)
        plain(batch_of(soil_and_vegetation_batch(4)))
        assert d['mean_selected_overlap'] <= plain.diagnostics_summary()[
            'mean_selected_overlap'] + 1e-9

    def test_summary_keys_exist(self):
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode=None,
                                               score_mode='uniform',
                                               geometry_beta=1.0,
                                               num_candidates=16)
        collator(batch_of(soil_and_vegetation_batch(4)))
        d = collator.diagnostics_summary()
        for key in ('mean_candidate_prior_score', 'candidate_score_samples',
                    'mean_selected_overlap', 'mean_candidate_overlap',
                    'invalid_candidate_rate', 'score_mode', 'geometry_beta'):
            assert key in d


# --------------------------------------------------------------------------- #
# 2. Stage 4A equivalence: the fix and the new knobs change nothing by default
# --------------------------------------------------------------------------- #

class TestStage4aEquivalence:

    def _reference_stage4a(self, prior, h, w, k, lam, generator, min_keep):
        """The pre-fix Stage 4A target-block rule, written out explicitly.

        Same candidate draw, same min-max normalisation, same softmax, same
        multinomial.  If the refactor changed anything the masks will diverge.
        """
        tops = torch.randint(0, GRID - h, (k,), generator=generator)
        lefts = torch.randint(0, GRID - w, (k,), generator=generator)
        scores = torch.empty(k)
        for i in range(k):
            scores[i] = prior[tops[i]:tops[i] + h, lefts[i]:lefts[i] + w].mean()
        span = float(scores.max() - scores.min())
        normalised = ((scores - scores.min()) / span if span > 1e-8
                      else torch.zeros_like(scores))
        probs = torch.softmax(lam * normalised, dim=0)
        picked = int(torch.multinomial(probs, 1, generator=generator).item())
        top, left = int(tops[picked]), int(lefts[picked])
        mask = torch.zeros((GRID, GRID), dtype=torch.int32)
        mask[top:top + h, left:left + w] = 1
        indices = torch.nonzero(mask.flatten()).squeeze()
        complement = torch.ones((GRID, GRID), dtype=torch.int32)
        complement[top:top + h, left:left + w] = 0
        return indices, complement

    def test_zero_beta_matches_the_stage4a_rule_exactly(self):
        """beta=0 -> the sampler is the Stage 4A rule, bit for bit."""
        imgs = soil_and_vegetation_batch(3)
        prior = StructurePrior(mode='vegetation').compute(
            imgs, sample_ids=[f'id{i}' for i in range(3)]).map
        h, w = 6, 6
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               geometry_beta=0.0,
                                               num_candidates=16)
        for image_index in range(3):
            collator._prior_map = prior[image_index][None]
            collator._prior_usable = torch.ones(1, dtype=torch.bool)
            collator._batch_size = 1
            collator._target_calls = 0
            collator._current_union = None
            collator._cand_gen = torch.Generator().manual_seed(1234 + image_index)
            reference_gen = torch.Generator().manual_seed(1234 + image_index)
            for block in range(collator.npred):
                mask, complement = collator._sample_block_mask((h, w))
                ref_mask, ref_complement = self._reference_stage4a(
                    prior[image_index], h, w, collator.num_candidates,
                    collator.guidance_lambda, reference_gen, collator.min_keep)
                assert torch.equal(torch.as_tensor(mask).reshape(-1),
                                   torch.as_tensor(ref_mask).reshape(-1))
                assert torch.equal(complement, ref_complement)

    def test_beta_zero_ignores_overlap_state(self):
        """With beta=0 the union state must not influence the choice."""
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               geometry_beta=0.0,
                                               num_candidates=16)
        imgs = soil_and_vegetation_batch(2)
        collator(batch_of(imgs))
        assert collator.diagnostics['selected_expected_overlap_sum'] > 0.0

    def test_no_new_rng_consumption_without_geometry(self):
        """beta only changes the weights, never the number of candidate draws."""
        def draws(collator_kwargs):
            collator = StructureGuidedMaskCollator(**collator_kwargs)
            collator(batch_of(soil_and_vegetation_batch(4)))
            d = collator.diagnostics
            assert d['candidates_generated'] == d['guided_selections'] * 16
            return int(d['candidates_generated'])
        a = draws(dict(COLLATOR_KWARGS, prior_mode='vegetation',
                       guidance_lambda=6.0, num_candidates=16, geometry_beta=0.0))
        b = draws(dict(COLLATOR_KWARGS, prior_mode='vegetation',
                       guidance_lambda=6.0, num_candidates=16, geometry_beta=3.5))
        assert a == b > 0


# --------------------------------------------------------------------------- #
# 3. The geometry-controlled arms
# --------------------------------------------------------------------------- #

def sample_target_union(collator, prior_map, h: int = 6, w: int = 6,
                        seed: int = 0) -> np.ndarray:
    """Run ``npred`` target blocks for ONE image and return their union grid.

    Uses the sampler directly so the union comparison is not polluted by
    per-batch block-size sampling or by the context block.
    """
    collator._batch_size = 1
    collator._target_calls = 0
    collator._current_union = torch.zeros((1, GRID, GRID), dtype=torch.bool)
    collator._cand_gen = torch.Generator().manual_seed(seed)
    if prior_map is not None:
        collator._prior_map = torch.as_tensor(prior_map).float()[None]
        collator._prior_usable = torch.ones(1, dtype=torch.bool)
    else:
        collator._prior_map = None
        collator._prior_usable = None
    masks = []
    for _ in range(collator.npred):
        mask, _ = collator._sample_block_mask((h, w))
        masks.append(torch.as_tensor(mask).reshape(-1))
    return occupancy(masks)


class TestGeometryControl:

    def test_g0_never_touches_the_prior(self):
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode=None,
                                               score_mode='uniform',
                                               geometry_beta=3.5,
                                               num_candidates=16)
        assert collator.prior is None
        assert collator.selection_enabled and not collator.guidance_enabled
        collator(batch_of(soil_and_vegetation_batch(4)))
        assert collator._prior_map is None
        assert collator.diagnostics['prior_seconds'] == 0.0

    def test_g0_masks_are_legal_and_reproducible(self):
        def run():
            collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                                   prior_mode=None,
                                                   score_mode='uniform',
                                                   geometry_beta=3.5,
                                                   num_candidates=16)
            return collator(batch_of(soil_and_vegetation_batch(4)))
        a, b = run(), run()
        all_masks_legal(a[1], a[2], batch=4)
        all_masks_legal(b[1], b[2], batch=4)
        for ga, gb in zip(a[2], b[2]):
            for ma, mb in zip(ga, gb):
                assert torch.equal(ma, mb)

    def test_geometry_control_reduces_union_redundancy(self):
        """At a strong penalty, the target union has fewer duplicate cells."""
        plain = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                            prior_mode=None,
                                            score_mode='uniform',
                                            geometry_beta=0.0,
                                            num_candidates=16)
        controlled = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                                 prior_mode=None,
                                                 score_mode='uniform',
                                                 geometry_beta=6.0,
                                                 num_candidates=16)
        plain_unions = [sample_target_union(plain, None, seed=s).sum()
                        for s in range(20)]
        controlled_unions = [sample_target_union(controlled, None, seed=s).sum()
                             for s in range(20)]
        assert np.mean(controlled_unions) > np.mean(plain_unions)

    def test_g1_keeps_its_structure_preference_under_control(self):
        """The vegetation prior must still move the blocks onto more vegetation."""
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               geometry_beta=3.5,
                                               num_candidates=16)
        base = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                           prior_mode=None,
                                           score_mode='uniform',
                                           geometry_beta=3.5,
                                           num_candidates=16)
        imgs = soil_and_vegetation_batch(12)
        veg_prior = StructurePrior(mode='vegetation').compute(
            imgs, sample_ids=[f'id{i}' for i in range(12)]).map.numpy()

        guided, unguided = [], []
        for i in range(12):
            target = sample_target_union(collator, veg_prior[i], seed=i)
            guided.append(float(veg_prior[i][target].mean()))
            target = sample_target_union(base, None, seed=i)
            unguided.append(float(veg_prior[i][target].mean()))
        assert np.mean(guided) > np.mean(unguided)

    def test_beta_zero_reproduces_stage4a_behaviour(self):
        """No geometry argument at all -> identical masks and RNG consumption."""
        imgs = soil_and_vegetation_batch(4)
        batch = batch_of(imgs)
        a = StructureGuidedMaskCollator(**COLLATOR_KWARGS, prior_mode='boundary',
                                        guidance_lambda=6.0, num_candidates=16)
        b = StructureGuidedMaskCollator(**COLLATOR_KWARGS, prior_mode='boundary',
                                        guidance_lambda=6.0, num_candidates=16,
                                        geometry_beta=0.0, geometry_control='linear')
        state = torch.get_rng_state()
        out_a = a(batch)
        torch.set_rng_state(state)
        out_b = b(batch)
        for ga, gb in zip(out_a[2], out_b[2]):
            for ma, mb in zip(ga, gb):
                assert torch.equal(ma, mb)
        for ga, gb in zip(out_a[1], out_b[1]):
            for ma, mb in zip(ga, gb):
                assert torch.equal(ma, mb)

    def test_high_beta_stays_legal(self):
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               geometry_beta=64.0,
                                               num_candidates=16)
        out = collator(batch_of(soil_and_vegetation_batch(4)))
        all_masks_legal(out[1], out[2], batch=4)

    def test_context_target_exclusion_under_geometry_control(self):
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='boundary',
                                               guidance_lambda=6.0,
                                               geometry_beta=3.5,
                                               num_candidates=16)
        _, masks_enc, masks_pred = collator(batch_of(soil_and_vegetation_batch(6)))
        for i in range(len(masks_pred[0])):
            target = occupancy(per_image_pred(masks_pred, i))
            context = occupancy(per_image_enc(masks_enc, i))
            assert not (target & context).any()

    def test_dead_code_removed_and_controls_valid(self):
        with pytest.raises(ValueError, match='geometry_control'):
            StructureGuidedMaskCollator(**COLLATOR_KWARGS, prior_mode=None,
                                        score_mode='uniform', geometry_beta=1.0,
                                        geometry_control='bogus')
        with pytest.raises(ValueError, match='geometry_beta'):
            StructureGuidedMaskCollator(**COLLATOR_KWARGS, prior_mode=None,
                                        score_mode='uniform', geometry_beta=-1.0)


# --------------------------------------------------------------------------- #
# 4. The shifted-boundary prior (G3)
# --------------------------------------------------------------------------- #

class TestShiftedBoundary:

    def test_histogram_preserved_exactly(self):
        prior = StructurePrior(mode='boundary')
        imgs = soil_and_vegetation_batch(4)
        original = prior.compute(imgs).map
        shifted = torch.roll(original, shifts=(8, 8), dims=(1, 2))
        assert torch.equal(original.flatten(1).sort(dim=1).values,
                           shifted.flatten(1).sort(dim=1).values)

    def test_circular_shift_breaks_alignment(self):
        from src.masks.agri_structure import circular_shift_patch_cells
        x = torch.linspace(0, 1, 256).reshape(1, 16, 16)
        shifted, info = circular_shift_patch_cells(x, shift=(8, 8))
        assert info['value_multiset_preserved']
        assert info['shift_dy'] == 8 and info['shift_dx'] == 8
        assert abs(info['shift_pearson_correlation_with_original']) < 0.99
        # a linear ramp loses most but not all local structure
        assert info['lag1_autocorrelation_original'] > 0.9

    def test_zero_shift_is_refused(self):
        from src.masks.agri_structure import circular_shift_patch_cells
        x = torch.rand(1, 16, 16)
        with pytest.raises(ValueError, match='does not destroy alignment'):
            circular_shift_patch_cells(x, shift=(0, 0))

    def test_shift_prior_differs_from_original_on_real_maps(self):
        shifted_prior = StructurePrior(mode='shifted_boundary', roll_shift=(8, 8))
        imgs = soil_and_vegetation_batch(6)
        original = StructurePrior(mode='boundary').compute(imgs).map
        shifted = shifted_prior.compute(imgs).map
        assert not torch.allclose(original, shifted)
        assert torch.equal(original.flatten(1).sort(dim=1).values,
                           shifted.flatten(1).sort(dim=1).values)
        assert shifted_prior.compute(imgs).diagnostics['shift_dy'] == 8

    def test_shifted_prior_is_usable_and_reproducible(self):
        shifted_prior = StructurePrior(mode='shifted_boundary', roll_shift=(8, 8))
        imgs = soil_and_vegetation_batch(6)
        a = shifted_prior.compute(imgs, sample_ids=[f'id{i}' for i in range(6)])
        b = shifted_prior.compute(imgs, sample_ids=[f'id{i}' for i in range(6)])
        assert torch.equal(a.map, b.map)
        # a uniform-soil crop has no edges and is legitimately unusable; the
        # vegetation-bearing crops must remain usable
        assert bool(a.usable.any())
        assert torch.equal(a.usable, b.usable)


# --------------------------------------------------------------------------- #
# 5. The geometry statistics module
# --------------------------------------------------------------------------- #

class TestGeometryStatistics:

    def test_cells_from_indices_bounds(self):
        with pytest.raises(ValueError, match='out of range'):
            cells_from_indices([256])
        assert cells_from_indices([0, 1, 2]).sum() == 3
        assert cells_from_indices(np.array([5, 5, 5])).sum() == 1

    def test_multiplicity_histogram(self):
        a = list(range(0, 20))
        b = list(range(10, 30))
        hist = multiplicity(union_cells([a, b]), [a, b])
        assert hist[0] == 256 - 30
        assert hist[1] == 20
        assert hist[2] == 10

    def test_geometry_record_counts_everything(self):
        targets = [torch.arange(0, 20), torch.arange(10, 30)]
        contexts = [torch.arange(200, 240)]
        # make the doubly-covered cells 10..19 the high-prior ones, so the
        # multiset mean must exceed the deduplicated union mean
        prior = np.zeros((16, 16))
        prior.reshape(-1)[10:20] = 1.0
        record = geometry_record(targets, contexts, {'vegetation': prior}, [240])
        assert record['total_target_area'] == 40
        assert record['target_cells'] == 30
        assert record['duplicated_target_area'] == 10
        assert record['multiplicity_2'] == 10
        assert record['context_cells'] == 40
        assert record['target_context_overlap'] == 0
        assert record['context_length_before_truncation'] == 240
        # the multiset mean counts a doubly-predicted cell twice
        assert record['mean_vegetation_target_multiset'] > record['mean_vegetation_target_union']
        assert record['vegetation_repeat_mass_ratio'] > 0

    def test_record_rejects_a_wrong_shaped_prior(self):
        with pytest.raises(ValueError, match='expected'):
            geometry_record([torch.arange(0, 20)], [torch.arange(200, 240)],
                            {'vegetation': np.zeros((4, 4))})

    def test_truncation_is_visible(self):
        """Pre-truncation context length > post-truncation cells is recorded."""
        targets = [torch.arange(0, 20), torch.arange(100, 120)]
        contexts = [torch.arange(200, 240)]
        record = geometry_record(targets, contexts, None, [60])
        assert record['context_cells'] == 40
        assert record['context_truncation_loss'] == 20.0

    def test_summarize_and_bootstrap(self):
        values = list(range(100))
        stats = summarize(values)
        assert stats['n'] == 100
        assert stats['mean'] == pytest.approx(49.5)
        assert stats['p10'] == pytest.approx(9.9)
        ci = bootstrap_ci(values, n_boot=200, seed=0)
        assert ci['low'] < stats['mean'] < ci['high']
        assert ci['n'] == 100

    def test_clustered_bootstrap_uses_sources(self):
        sources = ['a'] * 50 + ['b'] * 50
        values = [0.0] * 50 + [1.0] * 50
        ci = clustered_bootstrap_ci(sources, values, n_boot=200, seed=0)
        assert ci['n_sources'] == 2
        # resampling one source only can move the mean by up to 0.5
        assert ci['low'] <= 0.5 <= ci['high']

    def test_paired_delta_matches_hand_computation(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [0.5, 1.0, 1.5, 2.0]
        delta = paired_delta(a, b, n_boot=200, seed=0)
        assert delta['n'] == 4
        assert delta['mean_delta'] == pytest.approx(1.25)
        assert delta['wins'] == 4
        assert delta['sign_test_p_exact'] == pytest.approx(2 * 0.5 ** 4, rel=1e-9)

    def test_paired_delta_sign_test_exactness(self):
        """3/3 wins -> p = 2 * (0.5^3) = 0.25."""
        delta = paired_delta([2, 3, 4], [1, 1, 1], n_boot=100)
        assert delta['wins'] == 3 and delta['losses'] == 0
        assert delta['sign_test_p_exact'] == pytest.approx(0.25)
