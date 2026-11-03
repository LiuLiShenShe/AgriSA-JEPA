"""Structure-guided variant of the official I-JEPA multi-block mask collator.

The official :class:`src.masks.multiblock.MaskCollator` is **not modified**.  This
module subclasses it and changes exactly one thing: the *position* of each target
block is drawn from a scored candidate set instead of uniformly.

Everything else keeps the official semantics:

* block sizes and aspect ratios come from the parent's ``_sample_block_size``
  with the same per-batch seeded generator, so the size distribution is
  identical to arm B;
* target blocks are ``npred`` per image, context blocks ``nenc`` per image;
* context blocks come from the parent's ``_sample_block_mask`` with the same
  ``acceptable_regions`` rejection logic (context must avoid target cells);
* ``min_keep`` and the batch-wide common-length truncation are the parent's;
* no mask is ever allowed to place a target block where the parent would not.

Candidate sampling
------------------
For each target block the sampler draws ``K`` legally positioned candidates and
chooses one with

.. math::

    p_i \\propto \\exp(\\lambda \\, \\hat{s}_i - \\beta \\, o_i)

where ``s_i`` is the mean prior over the block's patch cells (min-max
normalised over the candidate set), and ``o_i`` is the fraction of the
candidate block's cells already covered by target blocks selected earlier for
the **same image**.  ``K=1`` reproduces the official behaviour exactly.

Three knobs cover the Stage 4B arms:

* ``prior_mode=None`` and ``score_mode='uniform'`` -- candidates carry no
  structure score at all (arm G0; the structure prior is never computed);
* ``geometry_beta > 0`` -- overlap-aware placement (the geometry control);
* ``force_candidates=True`` with ``lambda=beta=0`` -- the U diagnostic: the
  candidate mechanism alone, uniformly over positions.

The prior is computed from the **augmented crop** the encoder sees and is never
used as a training target.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import torch

from src.masks.agri_structure import StructurePrior
from src.masks.multiblock import MaskCollator


class StructureGuidedMaskCollator(MaskCollator):
    """Target-block positions sampled from a scored candidate set.

    Parameters
    ----------
    prior_mode:
        ``'vegetation'`` (C1), ``'boundary'`` (C2), ``'shuffled_boundary'``
        (C3), ``'shifted_boundary'`` (G3) or ``None`` (no structure prior).
        With ``None`` the prior is never computed and ``score_mode`` must be
        ``'uniform'`` unless the geometry control alone is being exercised.
    guidance_lambda:
        Structure weight in the selection logits.  0 -> no structure term.
    geometry_beta:
        Overlap weight in the selection logits.  0 -> no geometry control, which
        reproduces the Stage 4A sampler exactly.
    geometry_control:
        ``'linear'`` penalises overlap itself (``-beta * o_i``, the Stage 4B
        proposal and the form used by the trained G arms).  ``'deviation'``
        penalises the squared deviation from the overlap a *uniformly placed*
        block would have on average, computed exactly over the legal top-left
        placements.  Both were swept offline; the deviation form matched union
        means but produced too much context visibility, so the linear form was
        selected from mask geometry alone.
    score_mode:
        ``'prior'`` (mean prior over the candidate block) or ``'uniform'``
        (i.i.d. uniform scores drawn from the candidate generator).  G0 uses
        ``'uniform'`` so that positions are not driven by structure while the
        geometry control is still active.
    num_candidates:
        ``K``.  ``K=1`` reduces to the official uniform position sampling.
    force_candidates:
        Run the candidate machinery even when ``K, lambda, beta`` would
        otherwise reduce to the parent.  Used only by the U diagnostic arm.
    prior_device:
        Device for the (small) prior computation.  ``None`` -> CPU.
    base_seed:
        Seed mixed into the per-image C3 permutation.
    roll_shift:
        Fixed circular shift for the G3 ``shifted_boundary`` prior.
    """

    def __init__(self, *args, prior_mode: Optional[str] = None,
                 guidance_lambda: float = 0.0, num_candidates: int = 16,
                 geometry_beta: float = 0.0,
                 geometry_control: str = 'linear',
                 score_mode: str = 'prior',
                 force_candidates: bool = False,
                 prior_device=None, base_seed: int = 0,
                 boundary_edge_weight: float = 0.5,
                 roll_shift: Tuple[int, int] = (8, 8), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.allow_overlap:
            # the guided path assumes the parent's context-rejection logic is on
            raise ValueError('StructureGuidedMaskCollator requires allow_overlap=False')
        if num_candidates < 1:
            raise ValueError(f'num_candidates must be >= 1, got {num_candidates}')
        if score_mode not in ('prior', 'uniform'):
            raise ValueError(f"score_mode must be 'prior' or 'uniform', got {score_mode!r}")
        if score_mode == 'prior' and prior_mode is None:
            raise ValueError("score_mode='prior' requires a prior_mode; use "
                             "'uniform' for the no-structure control")
        if geometry_beta < 0:
            raise ValueError(f'geometry_beta must be >= 0, got {geometry_beta}')
        if geometry_control not in ('linear', 'deviation'):
            raise ValueError("geometry_control must be 'linear' or 'deviation', "
                             f"got {geometry_control!r}")
        self.prior_mode = prior_mode
        self.guidance_lambda = float(guidance_lambda)
        self.geometry_beta = float(geometry_beta)
        self.geometry_control = geometry_control
        self.score_mode = score_mode
        self.num_candidates = int(num_candidates)
        self.force_candidates = bool(force_candidates)
        self.prior_device = prior_device
        self.base_seed = int(base_seed)
        self.roll_shift = (int(roll_shift[0]), int(roll_shift[1]))
        self.prior = (StructurePrior(mode=prior_mode,
                                     boundary_edge_weight=boundary_edge_weight,
                                     roll_shift=self.roll_shift)
                      if prior_mode is not None else None)

        self._prior_map: Optional[torch.Tensor] = None      # [B, g, g]
        self._prior_usable: Optional[torch.Tensor] = None   # [B] bool
        self._cand_gen: Optional[torch.Generator] = None
        self._target_calls = 0
        self._batch_size = 0
        self._current_union: Optional[torch.Tensor] = None  # [B, g, g] bool
        self._reset_diagnostics()

    # -- diagnostics -------------------------------------------------------- #

    def _reset_diagnostics(self) -> None:
        self.diagnostics: Dict[str, object] = {
            'target_blocks': 0,
            'guided_selections': 0,
            'uniform_fallbacks': 0,
            'invalid_candidates': 0,
            'candidates_generated': 0,
            'selected_prior_score_sum': 0.0,
            'candidate_prior_score_sum': 0.0,
            'candidate_prior_score_count': 0,
            # kept for the Stage 4A reporting tools: sum over selections of the
            # mean valid-candidate score (divide by guided_selections)
            'uniform_candidate_score_sum': 0.0,
            'selected_overlap_sum': 0.0,
            'candidate_overlap_sum': 0.0,
            'candidate_overlap_count': 0,
            'selected_expected_overlap_sum': 0.0,
            'prior_seconds': 0.0,
            'selection_seconds': 0.0,
        }

    @property
    def guidance_enabled(self) -> bool:
        return self.prior is not None and self.num_candidates > 1

    @property
    def selection_enabled(self) -> bool:
        """True whenever candidate sampling must run."""
        return self.num_candidates > 1 and (
            self.prior is not None
            or self.geometry_beta > 0.0
            or self.force_candidates)

    @property
    def geometry_enabled(self) -> bool:
        return self.geometry_beta > 0.0 and self.num_candidates > 1

    # -- per-batch setup ---------------------------------------------------- #

    def __call__(self, batch):
        """Compute the prior for this batch, then run the official flow."""
        if self.guidance_enabled:
            images = torch.utils.data.default_collate(
                [item[0] for item in batch])
            sample_ids = [item[1] for item in batch]
            device = self.prior_device
            if device is not None:
                images = images.to(device)
            t0 = time.time()
            result = self.prior.compute(images, sample_ids=sample_ids,
                                        base_seed=self.base_seed)
            self.diagnostics['prior_seconds'] += time.time() - t0
            self._prior_map = result.map.detach().to('cpu')
            self._prior_usable = result.usable.detach().to('cpu')
            self._prior_diagnostics = result.diagnostics
        else:
            # G0: no structure prior is read or computed.
            self._prior_map = None
            self._prior_usable = None
            self._prior_diagnostics = {}

        # A dedicated generator keeps candidate sampling out of the global RNG
        # stream, so it cannot shift the augmentation or loader streams.
        self._cand_gen = torch.Generator()
        self._cand_gen.manual_seed(int(self._itr_counter.value) + 1_000_003)
        self._target_calls = 0
        self._batch_size = len(batch)
        self._current_union = None
        try:
            return super().__call__(batch)
        finally:
            self._prior_map = None
            self._prior_usable = None
            self._current_union = None

    # -- guided target sampling --------------------------------------------- #

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        """Parent semantics, with scored positions for target blocks."""
        if acceptable_regions is not None or not self.selection_enabled:
            return super()._sample_block_mask(b_size, acceptable_regions)

        h, w = b_size
        image_index = self._target_calls // max(self.npred, 1)
        self._target_calls += 1
        # `_target_calls` counts only TARGET blocks: context blocks are served by
        # the parent (they take the `acceptable_regions` path above).
        if image_index >= self._batch_size:
            return super()._sample_block_mask(b_size, acceptable_regions)

        if self._current_union is None:
            self._current_union = torch.zeros(
                (self._batch_size, self.height, self.width), dtype=torch.bool)

        prior = None
        if self.prior is not None:
            if self._prior_map is None or not bool(self._prior_usable[image_index]):
                # flat / blank / low-contrast crop: fall back to the official sampler
                self.diagnostics['uniform_fallbacks'] += 1
                return super()._sample_block_mask(b_size, acceptable_regions)
            prior = self._prior_map[image_index]

        max_top = self.height - h
        max_left = self.width - w
        if max_top <= 0 or max_left <= 0:
            return super()._sample_block_mask(b_size, acceptable_regions)

        k = self.num_candidates
        tops = torch.randint(0, max_top, (k,), generator=self._cand_gen)
        lefts = torch.randint(0, max_left, (k,), generator=self._cand_gen)
        self.diagnostics['candidates_generated'] += k

        t0 = time.time()
        scores = torch.empty(k, dtype=torch.float32)
        overlaps = torch.empty(k, dtype=torch.float32)
        union = self._current_union[image_index]
        # one vectorised draw keeps the uniform-score control as cheap as the
        # prior path; it is still fully reproducible from `_cand_gen`
        noise = (torch.rand(k, generator=self._cand_gen)
                 if self.score_mode == 'uniform' else None)
        for i in range(k):
            top, left = int(tops[i]), int(lefts[i])
            if self.score_mode == 'prior':
                block = prior[top:top + h, left:left + w]
                scores[i] = block.mean()
            else:
                scores[i] = noise[i]
            # fraction of the candidate block already predicted by an earlier
            # target block of THIS image
            overlaps[i] = union[top:top + h, left:left + w].float().mean()

        # a block of h*w patches must clear min_keep, exactly as the parent
        valid = torch.full((k,), h * w > self.min_keep, dtype=torch.bool)
        self.diagnostics['invalid_candidates'] += int((~valid).sum())

        if not bool(valid.any()):
            self.diagnostics['uniform_fallbacks'] += 1
            self.diagnostics['selection_seconds'] += time.time() - t0
            return super()._sample_block_mask(b_size, acceptable_regions)

        valid_scores = scores[valid]
        valid_overlaps = overlaps[valid]
        # Expected overlap of a UNIFORMLY placed block given the union already
        # selected for this image.  The candidates are drawn only from the legal
        # top-left positions (0 .. max_top-1) x (0 .. max_left-1), so the naive
        # U / grid^2 is wrong: a union cell near the excluded border belongs to
        # fewer placements.  The exact count is separable:
        #   cover_row[r] = #{t in [0,max_top) : t <= r <= t+h-1}
        # and E[overlap fraction] = sum_{cell in U} cover_row*cover_col / (P*A).
        rows = torch.arange(self.height)
        cols = torch.arange(self.width)
        cover_rows = (torch.minimum(rows, torch.tensor(max_top - 1))
                      - torch.clamp(rows - (h - 1), min=0) + 1).clamp(min=0).float()
        cover_cols = (torch.minimum(cols, torch.tensor(max_left - 1))
                      - torch.clamp(cols - (w - 1), min=0) + 1).clamp(min=0).float()
        placements = max(int(max_top) * int(max_left), 1)
        area = max(int(h) * int(w), 1)
        expected_overlap = float(
            (union.float() * cover_rows[:, None] * cover_cols[None, :]).sum()
            / (placements * area))
        self.diagnostics['candidate_prior_score_sum'] += float(valid_scores.sum())
        self.diagnostics['candidate_prior_score_count'] += int(valid_scores.numel())
        self.diagnostics['uniform_candidate_score_sum'] += float(valid_scores.mean())
        self.diagnostics['candidate_overlap_sum'] += float(valid_overlaps.sum())
        self.diagnostics['candidate_overlap_count'] += int(valid_overlaps.numel())

        if self.guidance_lambda == 0.0 and self.geometry_beta == 0.0:
            probs = torch.full_like(valid_scores, 1.0 / valid_scores.numel())
        else:
            if self.guidance_lambda == 0.0:
                structure_term = torch.zeros_like(valid_scores)
            else:
                span = float(valid_scores.max() - valid_scores.min())
                normalised = ((valid_scores - valid_scores.min()) / span
                              if span > 1e-8 else torch.zeros_like(valid_scores))
                structure_term = self.guidance_lambda * normalised
            if self.geometry_beta == 0.0:
                geometry_term = torch.zeros_like(valid_overlaps)
            elif self.geometry_control == 'linear':
                geometry_term = self.geometry_beta * valid_overlaps
            else:
                geometry_term = self.geometry_beta * (
                    valid_overlaps - expected_overlap) ** 2
            probs = torch.softmax(structure_term - geometry_term, dim=0)
        chosen = int(torch.multinomial(probs, 1, generator=self._cand_gen).item())

        valid_idx = torch.nonzero(valid).squeeze(1)
        picked = int(valid_idx[chosen])
        top, left = int(tops[picked]), int(lefts[picked])

        self.diagnostics['target_blocks'] += 1
        self.diagnostics['guided_selections'] += 1
        self.diagnostics['selected_prior_score_sum'] += float(scores[picked])
        self.diagnostics['selected_overlap_sum'] += float(overlaps[picked])
        self.diagnostics['selected_expected_overlap_sum'] += expected_overlap
        self.diagnostics['selection_seconds'] += time.time() - t0

        # record the selected block in this image's target union
        union[top:top + h, left:left + w] = True

        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask[top:top + h, left:left + w] = 1
        mask_indices = torch.nonzero(mask.flatten()).squeeze()
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top + h, left:left + w] = 0
        return mask_indices, mask_complement

    # -- summary ------------------------------------------------------------ #

    def diagnostics_summary(self) -> Dict[str, object]:
        d = dict(self.diagnostics)
        selections = max(int(d['guided_selections']), 1)
        candidate_count = max(int(d['candidate_prior_score_count']), 1)
        candidate_overlap_count = max(int(d['candidate_overlap_count']), 1)
        d['mean_selected_prior_score'] = (
            d.pop('selected_prior_score_sum') / selections)
        # Stage 4B fix: ``candidate_prior_score_sum`` accumulates one score per
        # *valid candidate*, so its denominator must be the number of valid
        # candidates -- not ``candidates_generated`` (which counts every draw,
        # including invalid ones, and produced a mean that was wrong by roughly
        # the candidate count per selection).  This is a diagnostics-only
        # change: sampling, RNG use, masks and gradients are untouched.
        d['mean_candidate_prior_score'] = (
            d.pop('candidate_prior_score_sum') / candidate_count)
        d['candidate_score_samples'] = candidate_count
        d['mean_selected_overlap'] = (
            d.pop('selected_overlap_sum') / selections)
        d['mean_expected_overlap'] = (
            d.pop('selected_expected_overlap_sum') / selections)
        d['mean_candidate_overlap'] = (
            d.pop('candidate_overlap_sum') / candidate_overlap_count)
        d['invalid_candidate_rate'] = (
            d['invalid_candidates'] / max(int(d['candidates_generated']), 1))
        d['prior_mode'] = self.prior_mode
        d['score_mode'] = self.score_mode
        d['guidance_lambda'] = self.guidance_lambda
        d['geometry_beta'] = self.geometry_beta
        d['geometry_control'] = self.geometry_control
        d['num_candidates'] = self.num_candidates
        d['prior_diagnostics'] = getattr(self, '_prior_diagnostics', {})
        return d
