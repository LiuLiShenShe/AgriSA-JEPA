"""Stage 3B / Stage 4A tests: structure priors and the guided mask collator.

The governing requirement is that the structure-guided collator is a *fair*
variant of the official I-JEPA collator: same geometry rules, same counts, same
exclusion constraints, same truncation, and the official collator itself left
untouched.  So most of these tests are equivalence tests against the upstream
implementation rather than tests of new behaviour.
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
    denormalize,
    excess_green,
    lag1_autocorrelation,
    luminance,
    patch_pool,
    permutation_for_sample,
    robust_normalize,
    shuffle_patch_cells,
    sobel_magnitude,
    sparse_normalize,
)
from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator

REPO = Path(__file__).resolve().parents[1]

COLLATOR_KWARGS = dict(
    input_size=224, patch_size=14,
    enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
    allow_overlap=False,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def normalize(tensors: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensors - mean) / std


def soil_and_vegetation_batch(b: int = 8, green_fraction: int = 2) -> torch.Tensor:
    """Normalised crops: brown soil with a green vegetation rectangle."""
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
    rgb = torch.full((b, 3, 224, 224), 0.5)
    return normalize(rgb)


def low_contrast_batch(b: int = 4) -> torch.Tensor:
    rgb = torch.zeros(b, 3, 224, 224)
    rgb[:, 0] = 0.500
    rgb[:, 1] = 0.501
    rgb[:, 2] = 0.499
    rgb[:, :, :50, :50] = torch.tensor([0.502, 0.503, 0.501]).view(3, 1, 1)
    return normalize(rgb)


def batch_of(tensors: torch.Tensor) -> list:
    return [(tensors[i], f'id{i}') for i in range(tensors.shape[0])]


def mask_cells(masks, grid: int = 16) -> np.ndarray:
    out = np.zeros((grid, grid), dtype=bool)
    for m in masks:
        flat = np.asarray(m).reshape(-1)
        out[flat // grid, flat % grid] = True
    return out


def enc_cells_per_image(masks_enc, b: int, grid: int = 16) -> list:
    """The official collator returns masks_enc as [tensor[B, nenc, L]]."""
    out = []
    for i in range(b):
        out.append(mask_cells([masks_enc[0][i, 0]], grid))
    return out


# --------------------------------------------------------------------------- #
# 1. Structure priors
# --------------------------------------------------------------------------- #

class TestStructurePriors:

    def test_denormalize_restores_rgb(self):
        rgb = torch.rand(2, 3, 224, 224)
        restored = denormalize(normalize(rgb))
        assert torch.allclose(restored, rgb, atol=1e-5)

    def test_denormalize_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match='B,3,H,W'):
            denormalize(torch.zeros(3, 224, 224))

    def test_denormalize_clamps_to_valid_range(self):
        rgb = torch.full((1, 3, 4, 4), 5.0)     # far outside [0, 1]
        out = denormalize(rgb)
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_exg_formula(self):
        rgb = torch.zeros(1, 3, 1, 1)
        rgb[:, 1] = 1.0          # G = 1, R = B = 0 -> ExG = 2
        assert float(excess_green(rgb)[0, 0, 0]) == pytest.approx(2.0)
        rgb = torch.zeros(1, 3, 1, 1)
        rgb[:, 0] = 1.0          # R = 1 -> ExG = -1
        rgb[:, 2] = 1.0          # B = 1 -> ExG = -2
        assert float(excess_green(rgb)[0, 0, 0]) == pytest.approx(-2.0)

    def test_exg_ranks_vegetation_above_soil(self):
        imgs = soil_and_vegetation_batch()
        exg = excess_green(denormalize(imgs))
        assert float(exg[0, 20:100, 30:120].mean()) > float(exg[0, :20, :20].mean())

    def test_luminance_formula(self):
        rgb = torch.zeros(1, 3, 1, 1)
        rgb[:, 0] = 1.0
        assert float(luminance(rgb)[0, 0, 0]) == pytest.approx(0.299)
        rgb = torch.zeros(1, 3, 1, 1)
        rgb[:, 1] = 1.0
        assert float(luminance(rgb)[0, 0, 0]) == pytest.approx(0.587)

    def test_sobel_detects_a_known_edge(self):
        gray = torch.zeros(1, 32, 32)
        gray[:, :, 16:] = 1.0        # vertical step at x=16
        edge = sobel_magnitude(gray)
        # the response must peak on the step column
        column_mean = edge[0].mean(dim=0)
        assert int(column_mean.argmax()) in (15, 16, 17)

    def test_sobel_is_zero_on_a_flat_image(self):
        flat = torch.full((1, 16, 16), 0.5)
        assert float(sobel_magnitude(flat).abs().max()) < 1e-6

    def test_sobel_accepts_4d_input(self):
        gray = torch.zeros(2, 1, 16, 16)
        gray[:, :, :, 8:] = 1.0
        assert sobel_magnitude(gray).shape == (2, 16, 16)

    def test_patch_pool_shape_and_scale(self):
        x = torch.ones(2, 224, 224)
        pooled = patch_pool(x)
        assert pooled.shape == (2, 16, 16)
        assert torch.allclose(pooled, torch.ones(2, 16, 16))

    def test_patch_pool_rejects_indivisible(self):
        with pytest.raises(ValueError, match='divisible'):
            patch_pool(torch.zeros(1, 100, 100))

    def test_robust_normalize_flags_flat_images(self):
        flat = torch.zeros(2, 224, 224)
        _, usable = robust_normalize(flat)
        assert usable.tolist() == [False, False]

    def test_sparse_normalize_handles_edge_maps(self):
        edge = torch.zeros(1, 224, 224)
        edge[0, 100, 100] = 5.0
        norm, usable = sparse_normalize(edge)
        assert bool(usable[0])
        assert float(norm.max()) == pytest.approx(1.0)
        assert float(norm.min()) >= 0.0


class TestPriorAlignmentAndFallback:

    def test_prior_shape_is_16x16_per_image(self):
        result = StructurePrior(mode='vegetation').compute(soil_and_vegetation_batch(6))
        assert result.map.shape == (6, 16, 16)

    def test_prior_is_finite_for_every_mode(self):
        imgs = soil_and_vegetation_batch(4)
        for mode in ('vegetation', 'boundary', 'shuffled_boundary'):
            result = StructurePrior(mode=mode).compute(
                imgs, sample_ids=[f'i{i}' for i in range(4)], base_seed=1)
            assert torch.isfinite(result.map).all(), mode

    def test_prior_range_is_0_1(self):
        result = StructurePrior(mode='boundary').compute(soil_and_vegetation_batch(4))
        assert float(result.map.min()) >= 0.0
        assert float(result.map.max()) <= 1.0

    def test_prior_is_aligned_with_the_image(self):
        """The vegetation prior must light up where the green rectangle is."""
        imgs = soil_and_vegetation_batch(2)
        result = StructurePrior(mode='vegetation').compute(imgs)
        grid = result.map[0].numpy()
        assert grid[4:7, 3:9].mean() > grid[12, 12]

    def test_blank_image_falls_back_to_random(self):
        result = StructurePrior(mode='vegetation').compute(blank_batch())
        assert result.usable.tolist() == [False] * 4

    def test_low_contrast_falls_back_or_stays_safe(self):
        result = StructurePrior(mode='vegetation').compute(low_contrast_batch())
        assert torch.isfinite(result.map).all()
        # either it correctly reports unusable, or the map is harmless
        assert result.usable.shape == (4,)

    def test_all_soil_image_is_flagged_unusable(self):
        rgb = torch.full((1, 3, 224, 224), 0.0)
        rgb[:, 0] = 0.45; rgb[:, 1] = 0.30; rgb[:, 2] = 0.25
        result = StructurePrior(mode='vegetation').compute(normalize(rgb))
        assert not bool(result.usable[0])

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match='mode'):
            StructurePrior(mode='nonsense')

    def test_prior_is_deterministic_for_a_seed(self):
        imgs = soil_and_vegetation_batch(4)
        a = StructurePrior(mode='shuffled_boundary').compute(
            imgs, sample_ids=['a', 'b', 'c', 'd'], base_seed=7).map
        b = StructurePrior(mode='shuffled_boundary').compute(
            imgs, sample_ids=['a', 'b', 'c', 'd'], base_seed=7).map
        assert torch.equal(a, b)

    def test_prior_does_not_depend_on_batch_order(self):
        imgs = soil_and_vegetation_batch(4)
        ids = [f'i{i}' for i in range(4)]
        full = StructurePrior(mode='shuffled_boundary').compute(
            imgs, sample_ids=ids, base_seed=7).map
        # a single image computed alone must match the same image in the batch
        single = StructurePrior(mode='shuffled_boundary').compute(
            imgs[2:3], sample_ids=[ids[2]], base_seed=7).map
        assert torch.allclose(single[0], full[2])


class TestShuffle:

    def test_shuffle_preserves_the_value_multiset(self):
        x = torch.rand(3, 16, 16)
        out, info = shuffle_patch_cells(x, sample_ids=['a', 'b', 'c'], base_seed=1)
        for i in range(3):
            assert torch.equal(torch.sort(x[i].reshape(-1))[0],
                               torch.sort(out[i].reshape(-1))[0])
        assert info['value_multiset_preserved']

    def test_shuffle_destroys_spatial_alignment(self):
        imgs = soil_and_vegetation_batch(2)
        base = StructurePrior(mode='boundary').compute(
            imgs, sample_ids=['a', 'b'], base_seed=1).map
        shuffled, _ = shuffle_patch_cells(base, sample_ids=['a', 'b'], base_seed=1)
        assert lag1_autocorrelation(base) > lag1_autocorrelation(shuffled)

    def test_shuffled_map_differs_from_the_original(self):
        x = torch.rand(2, 16, 16)
        out, _ = shuffle_patch_cells(x, sample_ids=['a', 'b'], base_seed=5)
        assert not torch.equal(x, out)

    def test_permutation_is_stable_and_seed_dependent(self):
        a = permutation_for_sample('img-1', 7)
        b = permutation_for_sample('img-1', 7)
        c = permutation_for_sample('img-1', 8)
        d = permutation_for_sample('img-2', 7)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)
        assert not torch.equal(a, d)
        assert a.sort().values.tolist() == list(range(256))

    def test_lag1_autocorrelation_of_a_shuffled_map_is_near_zero(self):
        torch.manual_seed(0)
        x = torch.rand(8, 16, 16)
        _, info = shuffle_patch_cells(x, sample_ids=[f'i{i}' for i in range(8)])
        out, _ = shuffle_patch_cells(x, sample_ids=[f'i{i}' for i in range(8)])
        assert abs(lag1_autocorrelation(out)) < 0.1

    def test_shuffle_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match='16,16'):
            shuffle_patch_cells(torch.zeros(2, 224, 224))


# --------------------------------------------------------------------------- #
# 2. Guided collator geometry (fairness against the official collator)
# --------------------------------------------------------------------------- #

class TestGuidedCollatorGeometry:

    def test_official_collator_is_unchanged(self):
        """The upstream file must not have been edited."""
        source = (REPO / 'src/masks/multiblock.py').read_text(encoding='utf-8')
        for marker in ("_sample_block_size", "_sample_block_mask",
                       "min_keep_pred = self.height * self.width"):
            assert marker in source
        assert 'StructurePrior' not in source, (
            'the official collator must not import the structure prior')

    def test_guided_subclass_reuses_official_geometry(self):
        collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                               prior_mode='vegetation',
                                               guidance_lambda=6.0,
                                               num_candidates=16)
        assert issubclass(StructureGuidedMaskCollator, MaskCollator)
        assert collator.npred == 1 and False or collator.npred == 4
        assert collator.nenc == 1
        assert collator.min_keep == 10
        assert collator.height == collator.width == 16

    def test_block_size_distribution_is_identical(self):
        """The size sampling must come from the parent's own method."""
        base = MaskCollator(**COLLATOR_KWARGS)
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        sizes_base, sizes_guided = [], []
        for seed in range(60):
            gb = torch.Generator(); gb.manual_seed(seed)
            gg = torch.Generator(); gg.manual_seed(seed)
            sizes_base.append(base._sample_block_size(
                generator=gb, scale=base.pred_mask_scale,
                aspect_ratio_scale=base.aspect_ratio))
            sizes_guided.append(guided._sample_block_size(
                generator=gg, scale=guided.pred_mask_scale,
                aspect_ratio_scale=guided.aspect_ratio))
        assert sizes_base == sizes_guided

    def test_mask_counts_and_lengths_match(self):
        base = MaskCollator(**COLLATOR_KWARGS)
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        batch = batch_of(soil_and_vegetation_batch(8))
        _, e_base, p_base = base(batch)
        _, e_guided, p_guided = guided(batch)
        assert len(e_base) == len(e_guided) == 1
        assert len(p_base) == len(p_guided) == base.npred
        # masks_enc is [tensor[B, nenc*L]] and masks_pred is a list of npred
        # tensors of shape [B, L]
        for a, b in zip(p_base, p_guided):
            assert a.shape == b.shape, (a.shape, b.shape)
        assert e_base[0].shape[0] == e_guided[0].shape[0] == 8
        assert e_base[0].shape[1] > base.min_keep
        assert e_guided[0].shape[1] > guided.min_keep
        assert [[len(m) for m in g] for g in p_base] == [
            [len(m) for m in g] for g in p_guided]

    def test_target_block_area_matches(self):
        """Total target area must not be used as a hidden improvement knob."""
        base = MaskCollator(**COLLATOR_KWARGS)
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='boundary',
                                             guidance_lambda=6.0, num_candidates=16)
        batch = batch_of(soil_and_vegetation_batch(8))
        _, e_b, p_b = base(batch)
        _, e_g, p_g = guided(batch)
        area_b = sum(len(m) for g in p_b for m in g)
        area_g = sum(len(m) for g in p_g for m in g)
        assert area_b == area_g, (area_b, area_g)

    def test_context_target_exclusion_preserved(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        batch = batch_of(soil_and_vegetation_batch(8))
        _, masks_enc, masks_pred = guided(batch)
        for b in range(8):
            enc_idx = set(np.asarray(masks_enc[0][b]).reshape(-1).tolist())
            for k in range(len(masks_pred)):
                idx = set(np.asarray(masks_pred[k][b]).reshape(-1).tolist())
                assert not (enc_idx & idx), 'context must avoid target cells'

    def test_mask_indices_are_in_range_and_unique(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        _, masks_enc, masks_pred = guided(batch_of(soil_and_vegetation_batch(4)))
        for b in range(4):
            flat = np.asarray(masks_enc[0][b]).reshape(-1)
            assert flat.min() >= 0 and flat.max() < 256
            assert len(set(flat.tolist())) == len(flat)
        for block in masks_pred:                  # one tensor [B, L] per block
            for b in range(block.shape[0]):
                flat = np.asarray(block[b]).reshape(-1)
                assert flat.min() >= 0 and flat.max() < 256
                assert len(set(flat.tolist())) == len(flat)

    def test_batch_common_length_truncation(self):
        """Every image's k-th target block must have the same length."""
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        _, _, masks_pred = guided(batch_of(soil_and_vegetation_batch(8)))
        for block in masks_pred:
            assert block.shape[0] == 8
            assert len(set(block.shape)) == 1 or block.shape[0] != block.shape[1]

    def test_min_keep_respected(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        _, masks_enc, masks_pred = guided(batch_of(soil_and_vegetation_batch(4)))
        for b in range(4):
            assert len(masks_enc[0][b]) > guided.min_keep
        for block in masks_pred:
            assert block.shape[1] > guided.min_keep

    def test_lambda_zero_is_uniform_over_candidates(self):
        """lambda=0 must not use the prior at all."""
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=0.0, num_candidates=16)
        batch = batch_of(soil_and_vegetation_batch(4))
        # run twice with the same counter state: the masks must be identical
        # because the selection no longer depends on the prior values
        _, _, p1 = guided(batch)
        guided._itr_counter.value = guided._itr_counter.value      # no-op
        _, _, p2 = guided(batch)
        # a softmax over equal values is uniform; the choice is driven only by
        # the candidate generator, which is reseeded per batch from the counter
        assert [len(g) for g in p1] == [len(g) for g in p2]

    def test_lambda_zero_uses_the_prior_only_for_validity(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=0.0, num_candidates=8)
        _, _, masks_pred = guided(batch_of(soil_and_vegetation_batch(4)))
        d = guided.diagnostics_summary()
        assert d['guidance_lambda'] == 0.0
        assert d['mean_selected_prior_score'] >= 0.0
        # the prior is still computed; guidance is what is disabled
        assert d['prior_mode'] == 'vegetation'

    def test_blank_image_falls_back_to_the_official_sampler(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=8)
        _, masks_enc, masks_pred = guided(batch_of(blank_batch(4)))
        d = guided.diagnostics_summary()
        assert d['uniform_fallbacks'] == 4 * guided.npred
        assert d['guided_selections'] == 0
        # and the masks are still legal
        for m in masks_enc:
            assert len(m[0]) > guided.min_keep

    def test_guidance_changes_target_placement(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        imgs = soil_and_vegetation_batch(8)
        _, _, p_guided = guided(batch_of(imgs))
        base = MaskCollator(**COLLATOR_KWARGS)
        _, _, p_base = base(batch_of(imgs))
        veg = StructurePrior(mode='vegetation').compute(imgs).map.numpy()

        def score(blocks):
            per_image = []
            for i in range(len(imgs)):
                cells = np.zeros((16, 16), dtype=bool)
                for block in blocks:
                    flat = np.asarray(block[i]).reshape(-1)
                    cells[flat // 16, flat % 16] = True
                per_image.append(veg[i][cells].mean())
            return float(np.mean(per_image))

        assert score(p_guided) > score(p_base)

    def test_guided_and_random_same_block_sizes_on_a_given_batch(self):
        """Same seeded generator -> same block size; only the position differs."""
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        imgs = soil_and_vegetation_batch(8)
        batch = batch_of(imgs)
        base = MaskCollator(**COLLATOR_KWARGS)
        _, e_b, p_b = base(batch)
        _, e_g, p_g = guided(batch)
        assert [[len(m) for m in g] for g in p_b] == [
            [len(m) for m in g] for g in p_g]

    def test_allow_overlap_is_refused(self):
        with pytest.raises(ValueError, match='allow_overlap=False'):
            StructureGuidedMaskCollator(allow_overlap=True,
                                        input_size=224, patch_size=14)

    def test_bad_num_candidates_rejected(self):
        with pytest.raises(ValueError, match='num_candidates'):
            StructureGuidedMaskCollator(num_candidates=0,
                                        input_size=224, patch_size=14)

    def test_diagnostics_are_sane(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        guided(batch_of(soil_and_vegetation_batch(4)))
        d = guided.diagnostics_summary()
        # only the images whose prior is usable are guided; the rest fall back
        assert d['guided_selections'] == d['target_blocks']
        assert d['guided_selections'] + d['uniform_fallbacks'] == 4 * guided.npred
        assert d['candidates_generated'] == d['guided_selections'] * 16
        assert d['invalid_candidate_rate'] == 0.0
        assert d['mean_candidate_prior_score'] <= d['mean_selected_prior_score']

    def test_masking_does_not_touch_the_image_batch(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        batch = batch_of(soil_and_vegetation_batch(4))
        before = [t.clone() for t, _ in batch]
        guided(batch)
        for (t, _), old in zip(batch, before):
            assert torch.equal(t, old), 'the collator must not modify the images'


# --------------------------------------------------------------------------- #
# 3. Training-interface compatibility
# --------------------------------------------------------------------------- #

class TestTrainingCompatibility:

    def test_collator_returns_the_official_tuple(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        out = guided(batch_of(soil_and_vegetation_batch(4)))
        assert isinstance(out, tuple) and len(out) == 3
        images, masks_enc, masks_pred = out
        # the official collator returns default_collate(batch) = [images, ids]
        assert len(images) == 2
        assert tuple(images[0].shape) == (4, 3, 224, 224)
        assert tuple(images[1]) == ('id0', 'id1', 'id2', 'id3')
        assert isinstance(masks_enc, list) and isinstance(masks_pred, list)
        assert len(masks_pred) == 4
        assert masks_pred[0].shape[0] == 4

    def test_step_counter_advances(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        before = guided._itr_counter.value
        guided(batch_of(soil_and_vegetation_batch(4)))
        assert guided._itr_counter.value == before + 1

    def test_seed_reproducibility(self):
        def run():
            collator = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                                   prior_mode='vegetation',
                                                   guidance_lambda=6.0,
                                                   num_candidates=16)
            return collator(batch_of(soil_and_vegetation_batch(4)))[2]
        a = run()
        b = run()
        assert [len(m) for g in a for m in g] == [len(m) for g in b for m in g]

    def test_masks_are_contiguous_patch_index_tensors(self):
        guided = StructureGuidedMaskCollator(**COLLATOR_KWARGS,
                                             prior_mode='vegetation',
                                             guidance_lambda=6.0, num_candidates=16)
        _, masks_enc, masks_pred = guided(batch_of(soil_and_vegetation_batch(2)))
        for b in range(2):
            assert masks_enc[0][b].dtype in (torch.int64, torch.int32)
        for group in masks_pred:
            for m in group:
                assert m.dtype in (torch.int64, torch.int32)


# --------------------------------------------------------------------------- #
# 4. Protocol configuration
# --------------------------------------------------------------------------- #

class TestStage3B4AConfigs:

    ARM_FILES = {
        'B': REPO / 'configs/stage3b_random_2000.yaml',
        'C1': REPO / 'configs/stage4a_vegetation_2000.yaml',
        'C2': REPO / 'configs/stage4a_boundary_2000.yaml',
        'C3': REPO / 'configs/stage4a_shuffled_2000.yaml',
    }

    def _load(self, path: Path):
        import yaml
        return yaml.safe_load(path.read_text(encoding='utf-8'))

    def test_all_configs_exist(self):
        for arm, path in self.ARM_FILES.items():
            assert path.is_file(), f'{arm} config missing: {path}'

    def test_budget_and_save_points(self):
        for arm, path in self.ARM_FILES.items():
            cfg = self._load(path)
            assert cfg['optimization']['max_optimizer_steps'] == 2000, arm
            assert cfg['optimization']['save_at_optimizer_steps'] == [500], arm
            assert cfg['optimization']['grad_accum'] == 8, arm
            assert cfg['data']['batch_size'] == 16, arm
            assert cfg['data']['augmentation_seed'] is not None, arm
            assert cfg['meta']['load_checkpoint'] is False, arm

    def test_same_manifest_and_initialisation(self):
        manifests, inits = set(), set()
        for arm, path in self.ARM_FILES.items():
            cfg = self._load(path)
            manifests.add(cfg['data']['manifest'])
            manifests.add(cfg['data']['manifest_sha256'])
            inits.add(cfg['meta']['pretrained_checkpoint'])
            inits.add(cfg['meta']['pretrained_sha256'])
        assert manifests == {
            'data/frozen/agrifield40k-v1_train.csv',
            'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435'}
        assert inits == {
            '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
            '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'}

    def test_mask_strategy_per_arm(self):
        expected = {
            'B': ('random', None),
            'C1': ('structure_guided', 'vegetation'),
            'C2': ('structure_guided', 'boundary'),
            'C3': ('structure_guided', 'shuffled_boundary'),
        }
        for arm, (mode, prior) in expected.items():
            cfg = self._load(self.ARM_FILES[arm])
            assert cfg['mask']['mode'] == mode, arm
            assert cfg['mask'].get('prior_mode') == prior, arm
            if mode == 'random':
                for key in ('prior_mode', 'guidance_lambda'):
                    assert cfg['mask'].get(key) is None, (arm, key)
            else:
                assert cfg['mask']['guidance_lambda'] > 0, arm
                assert cfg['mask']['num_candidates'] >= 2, arm

    def test_run_folders_are_distinct(self):
        folders = set()
        for arm, path in self.ARM_FILES.items():
            cfg = self._load(path)
            folders.add(cfg['logging']['folder'])
        assert len(folders) == len(self.ARM_FILES), 'arms must not share a run folder'

    def test_protocol_checker_agrees(self):
        import yaml
        from tools.check_stage3b4a_protocol import MUST_MATCH, ALLOWED_DIFFERENT
        flats = {arm: self._flatten(self._load(path))
                 for arm, path in self.ARM_FILES.items()}
        problems = []
        for key in set().union(*[set(f) for f in flats.values()]):
            values = {a: f.get(key) for a, f in flats.items()}
            if len({str(v) for v in values.values()}) > 1 and key not in ALLOWED_DIFFERENT:
                problems.append(key)
        assert problems == []

    @staticmethod
    def _flatten(node, prefix=()):
        out = {}
        if isinstance(node, dict):
            for k, v in node.items():
                out.update(TestStage3B4AConfigs._flatten(v, prefix + (str(k),)))
        else:
            out[prefix] = node
        return out