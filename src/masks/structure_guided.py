"""Structure-guided variant of the official I-JEPA multi-block mask collator.

The official :class:`src.masks.multiblock.MaskCollator` is **not modified**.  This
module subclasses it and changes exactly one thing: the *position* of each target
block is drawn from a structure-weighted candidate set instead of uniformly.

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
For each target block the sampler draws ``K`` legally positioned candidates,
scores each by the mean prior over the block's patch cells, and selects one by
:math:`p_i = \\mathrm{softmax}(\\lambda \\cdot s_i)`.  ``K=1`` reproduces the
official behaviour exactly.  ``\\lambda = 0`` gives uniform selection over the
candidates, which is the diagnostic mode used to verify that the candidate
machinery itself introduces no geometric bias.

The prior is computed from the **augmented crop** the encoder sees and is never
used as a training target.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.masks.agri_structure import StructurePrior
from src.masks.multiblock import MaskCollator


class StructureGuidedMaskCollator(MaskCollator):
    """Target-block positions sampled from a structure-weighted candidate set.

    Parameters
    ----------
    prior_mode:
        ``'vegetation'`` (C1), ``'boundary'`` (C2), ``'shuffled_boundary'`` (C3)
        or ``None`` (no guidance -- identical to the parent, used by arm B).
    guidance_lambda:
        Softmax temperature scale.  0 -> uniform candidate selection.
    num_candidates:
        ``K``.  ``K=1`` reduces to the official uniform position sampling.
    prior_device:
        Device for the (small) prior computation.  ``None`` -> CPU.
    base_seed:
        Seed mixed into the per-image C3 permutation.
    """

    def __init__(self, *args, prior_mode: Optional[str] = None,
                 guidance_lambda: float = 0.0, num_candidates: int = 16,
                 prior_device=None, base_seed: int = 0,
                 boundary_edge_weight: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.allow_overlap:
            # the guided path assumes the parent's context-rejection logic is on
            raise ValueError('StructureGuidedMaskCollator requires allow_overlap=False')
        if num_candidates < 1:
            raise ValueError(f'num_candidates must be >= 1, got {num_candidates}')
        self.prior_mode = prior_mode
        self.guidance_lambda = float(guidance_lambda)
        self.num_candidates = int(num_candidates)
        self.prior_device = prior_device
        self.base_seed = int(base_seed)
        self.prior = (StructurePrior(mode=prior_mode,
                                     boundary_edge_weight=boundary_edge_weight)
                      if prior_mode is not None else None)

        self._prior_map: Optional[torch.Tensor] = None      # [B, g, g]
        self._prior_usable: Optional[torch.Tensor] = None   # [B] bool
        self._cand_gen: Optional[torch.Generator] = None
        self._target_calls = 0
        self._batch_size = 0
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
            'uniform_candidate_score_sum': 0.0,
            'prior_seconds': 0.0,
            'selection_seconds': 0.0,
        }

    @property
    def guidance_enabled(self) -> bool:
        return self.prior is not None and self.num_candidates > 1

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
            self._prior_map = None
            self._prior_usable = None
            self._prior_diagnostics = {}

        # A dedicated generator keeps candidate sampling out of the global RNG
        # stream, so it cannot shift the augmentation or loader streams.
        self._cand_gen = torch.Generator()
        self._cand_gen.manual_seed(int(self._itr_counter.value) + 1_000_003)
        self._target_calls = 0
        self._batch_size = len(batch)
        try:
            return super().__call__(batch)
        finally:
            self._prior_map = None
            self._prior_usable = None

    # -- guided target sampling --------------------------------------------- #

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        """Parent semantics, with guided positions for target blocks."""
        if acceptable_regions is not None or not self.guidance_enabled:
            return super()._sample_block_mask(b_size, acceptable_regions)

        h, w = b_size
        image_index = self._target_calls // max(self.npred, 1)
        self._target_calls += 1
        # `_target_calls` counts only TARGET blocks: context blocks are served by
        # the parent (they take the `acceptable_regions` path above).
        if image_index >= self._batch_size:
            return super()._sample_block_mask(b_size, acceptable_regions)
        if not bool(self._prior_usable[image_index]):
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
        for i in range(k):
            block = prior[tops[i]:tops[i] + h, lefts[i]:lefts[i] + w]
            scores[i] = block.mean()
        # a block of h*w patches must clear min_keep, exactly as the parent
        valid = torch.full((k,), h * w > self.min_keep, dtype=torch.bool)
        self.diagnostics['invalid_candidates'] += int((~valid).sum())

        if not bool(valid.any()):
            self.diagnostics['uniform_fallbacks'] += 1
            self.diagnostics['selection_seconds'] += time.time() - t0
            return super()._sample_block_mask(b_size, acceptable_regions)

        valid_scores = scores[valid]
        if self.guidance_lambda == 0.0:
            probs = torch.full_like(valid_scores, 1.0 / valid_scores.numel())
        else:
            span = float(valid_scores.max() - valid_scores.min())
            normalised = ((valid_scores - valid_scores.min()) / span
                          if span > 1e-8 else torch.zeros_like(valid_scores))
            probs = torch.softmax(self.guidance_lambda * normalised, dim=0)
        chosen = int(torch.multinomial(probs, 1, generator=self._cand_gen).item())

        valid_idx = torch.nonzero(valid).squeeze(1)
        picked = int(valid_idx[chosen])
        top, left = int(tops[picked]), int(lefts[picked])

        self.diagnostics['target_blocks'] += 1
        self.diagnostics['guided_selections'] += 1
        self.diagnostics['selected_prior_score_sum'] += float(scores[picked])
        self.diagnostics['uniform_candidate_score_sum'] += float(valid_scores.mean())
        self.diagnostics['selection_seconds'] += time.time() - t0

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
        generated = max(int(d['candidates_generated']), 1)
        d['mean_selected_prior_score'] = d.pop('selected_prior_score_sum') / selections
        d['mean_candidate_prior_score'] = d.pop('uniform_candidate_score_sum') / generated
        d['invalid_candidate_rate'] = d['invalid_candidates'] / generated
        d['prior_mode'] = self.prior_mode
        d['guidance_lambda'] = self.guidance_lambda
        d['num_candidates'] = self.num_candidates
        d['prior_diagnostics'] = getattr(self, '_prior_diagnostics', {})
        return d
