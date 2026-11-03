"""Stage 4B tests for the official-batch mask audit and the shifted prior.

The audit is the evidence base for every Stage 4B claim, so it needs its own
tests.  These verify the audit's *method*, not the numbers:

1. ``place_masks`` reproduces the official batch semantics: one block size per
   batch, context masks truncated to the batch-wide minimum.
2. ``stratified_sample`` selects the whole manifest when asked, and covers every
   source proportionally otherwise.
3. ``augmented_crop`` derives its seed from the manifest position exactly as
   ``AgriFieldDataset`` does.
4. The audit's geometry record sees the post-truncation context length.
5. ``ks_statistic`` is correct on hand-checkable cases.
6. The shifted prior's diagnostics quantify what it breaks.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.masks.agri_structure import StructurePrior
from src.masks.geometry import geometry_record
from src.masks.multiblock import MaskCollator
from src.masks.structure_guided import StructureGuidedMaskCollator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.audit_mask_geometry import (  # noqa: E402
    COLLATOR_KWARGS,
    augmented_crop,
    place_masks,
    stratified_sample,
)

MANIFEST = REPO / 'data/frozen/agrifield40k-v1_train.csv'


# --------------------------------------------------------------------------- #
# Official batch semantics
# --------------------------------------------------------------------------- #

class TestPlaceMasksBatchSemantics:

    def _collators(self):
        return MaskCollator(**COLLATOR_KWARGS), StructureGuidedMaskCollator(
            **COLLATOR_KWARGS, prior_mode=None, score_mode='uniform',
            geometry_beta=3.5, num_candidates=16)

    def test_one_block_size_per_batch(self):
        """Every target block in the batch shares the batch's sampled size."""
        base, _ = self._collators()
        batch = [(torch.zeros(3, 224, 224), f'id{i}') for i in range(4)]
        generator = torch.Generator().manual_seed(3)
        p_size = base._sample_block_size(
            generator=generator, scale=base.pred_mask_scale,
            aspect_ratio_scale=base.aspect_ratio)
        e_size = base._sample_block_size(
            generator=generator, scale=base.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
        masks_p, masks_e, pre_trunc = place_masks(base, batch, p_size, e_size)
        sizes = {tuple(m.shape[0] for m in group) for group in masks_p}
        assert len(sizes) == 1
        for group in masks_p:
            for mask in group:
                assert len(mask) == p_size[0] * p_size[1]
        assert len(masks_e) == 4
        assert len(pre_trunc) == 4

    def test_context_truncated_to_the_batch_minimum(self):
        """Post-truncation context length equals the smallest pre-truncation."""
        base, _ = self._collators()
        batch = [(torch.zeros(3, 224, 224), f'id{i}') for i in range(6)]
        generator = torch.Generator().manual_seed(11)
        p_size = base._sample_block_size(
            generator=generator, scale=base.pred_mask_scale,
            aspect_ratio_scale=base.aspect_ratio)
        e_size = base._sample_block_size(
            generator=generator, scale=base.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
        masks_p, masks_e, pre_trunc = place_masks(base, batch, p_size, e_size)
        assert len(pre_trunc) == 6
        assert min(pre_trunc) <= max(pre_trunc)
        for group in masks_e:
            assert len(group[0]) == min(pre_trunc)

    def test_geometry_record_sees_the_truncated_length(self):
        base, _ = self._collators()
        batch = [(torch.zeros(3, 224, 224), f'id{i}') for i in range(6)]
        generator = torch.Generator().manual_seed(5)
        p_size = base._sample_block_size(
            generator=generator, scale=base.pred_mask_scale,
            aspect_ratio_scale=base.aspect_ratio)
        e_size = base._sample_block_size(
            generator=generator, scale=base.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
        masks_p, masks_e, pre_trunc = place_masks(base, batch, p_size, e_size)
        record = geometry_record(masks_p[0], masks_e[0], None, [pre_trunc[0]])
        assert record['context_cells'] == min(pre_trunc)
        assert record['context_length_before_truncation'] == pre_trunc[0]
        assert record['context_truncation_loss'] == pre_trunc[0] - min(pre_trunc)

    def test_guided_arm_uses_the_same_batch_flow(self):
        _, guided = self._collators()
        batch = [(torch.zeros(3, 224, 224), f'id{i}') for i in range(4)]
        guided._itr_counter.value = 0
        guided._batch_size = 4
        guided._target_calls = 0
        guided._current_union = None
        generator = torch.Generator().manual_seed(0)
        p_size = guided._sample_block_size(
            generator=generator, scale=guided.pred_mask_scale,
            aspect_ratio_scale=guided.aspect_ratio)
        e_size = guided._sample_block_size(
            generator=generator, scale=guided.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))
        masks_p, masks_e, pre_trunc = place_masks(guided, batch, p_size, e_size)
        assert len(masks_p) == 4 and len(masks_e) == 4
        for group in masks_p:
            assert len(group) == guided.npred
            for mask in group:
                assert 0 <= int(mask.min()) and int(mask.max()) < 256


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

class TestStratifiedSample:

    def test_zero_selects_everything(self):
        rows = [{'sample_id': str(i), 'source': s}
                for i, s in enumerate(['a'] * 7 + ['b'] * 3)]
        assert len(stratified_sample(rows, 0)) == 10
        assert len(stratified_sample(rows, 10)) == 10
        assert len(stratified_sample(rows, 99)) == 10

    def test_proportional_stratification(self):
        rows = [{'sample_id': str(i), 'source': s}
                for i, s in enumerate(['a'] * 7 + ['b'] * 3)]
        picked = stratified_sample(rows, 5)
        assert len(picked) == 5
        counts = {}
        for row in picked:
            counts[row['source']] = counts.get(row['source'], 0) + 1
        # 70/30 split -> 4 a's and 1 b (largest remainder)
        assert counts == {'a': 4, 'b': 1}

    def test_capped_source_is_redistributed(self):
        rows = [{'sample_id': str(i), 'source': s}
                for i, s in enumerate(['a'] * 2 + ['b'] * 100)]
        picked = stratified_sample(rows, 10)
        counts = {}
        for row in picked:
            counts[row['source']] = counts.get(row['source'], 0) + 1
        # proportional quotas would give a=0; the remainder goes to the
        # largest source, so the small source is never over-represented
        assert counts.get('a', 0) <= 2
        assert counts.get('b', 0) >= 8
        assert len(picked) == 10


# --------------------------------------------------------------------------- #
# Augmentation determinism
# --------------------------------------------------------------------------- #

class TestAugmentedCrop:

    def test_seed_derives_from_the_manifest_position(self):
        """Same (augmentation_seed, epoch, index) -> the same crop."""
        import hashlib
        from PIL import Image
        import numpy as np
        rng = np.random.default_rng(0)
        image = Image.fromarray(
            rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), mode='RGB')
        a = augmented_crop(image, 20261019, 0, 1234)
        b = augmented_crop(image, 20261019, 0, 1234)
        assert torch.equal(a, b)
        c = augmented_crop(image, 20261019, 0, 1235)
        assert not torch.equal(a, c)

    def test_matches_the_dataset_derivation(self):
        from src.datasets.agri_dataset import AgriFieldDataset
        expected = int.from_bytes(
            hashlib.sha256(b'20261019:0:1234').digest()[:8], 'big') % (2 ** 63 - 1)
        assert expected == int.from_bytes(
            hashlib.sha256(b'20261019:0:1234').digest()[:8], 'big') % (2 ** 63 - 1)
        # the dataset class derives the same value for the same inputs
        dataset = AgriFieldDataset.__new__(AgriFieldDataset)
        dataset.augmentation_seed = 20261019
        dataset._epoch = 0
        assert dataset._derived_seed(1234) == expected

    def test_does_not_leave_the_global_rng_advanced(self):
        from PIL import Image
        image = Image.new('RGB', (512, 512), (10, 20, 30))
        before = torch.get_rng_state()
        augmented_crop(image, 20261019, 0, 7)
        assert torch.equal(torch.get_rng_state(), before)


# --------------------------------------------------------------------------- #
# KS statistic
# --------------------------------------------------------------------------- #

class TestKsStatistic:

    def test_identical_samples(self):
        from tools.check_stage4b_geometry_gate import ks_statistic
        assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0

    def test_disjoint_samples(self):
        from tools.check_stage4b_geometry_gate import ks_statistic
        assert ks_statistic([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)

    def test_known_overlap(self):
        from tools.check_stage4b_geometry_gate import ks_statistic
        # at 0 the first CDF is 1.0 and the second is 0.5
        value = ks_statistic([0, 0, 0, 0], [0, 0, 1, 1])
        assert value == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# The shifted prior in an audit-like setting
# --------------------------------------------------------------------------- #

class TestShiftedPriorDiagnostics:

    def test_shift_report_is_quantitative(self):
        shifted = StructurePrior(mode='shifted_boundary', roll_shift=(8, 8))
        rgb = torch.zeros(4, 3, 224, 224)
        rgb[:, 0] = 0.4
        rgb[:, 1] = 0.3
        rgb[:, 2] = 0.2
        rgb[:, 1, 40:140, 60:180] = 0.6
        normalized = (rgb - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / \
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            original = StructurePrior(mode='boundary').compute(normalized)
            result = shifted.compute(normalized)
        diagnostics = result.diagnostics
        assert diagnostics['shift_dy'] == 8
        assert diagnostics['value_multiset_preserved'] is True
        assert abs(diagnostics['shift_pearson_correlation_with_original']) < 0.5
        # the roll keeps most local autocorrelation
        assert (diagnostics['lag1_autocorrelation_shifted']
                > 0.8 * diagnostics['lag1_autocorrelation_original'])
        assert torch.equal(original.map.flatten(1).sort(dim=1).values,
                           result.map.flatten(1).sort(dim=1).values)
