"""Unsupervised agricultural structure priors for structure-guided JEPA masks.

This module computes *prior maps* from an RGB image.  It never reads a
segmentation, detection or instance label, and it is only ever used to decide
**where** a JEPA target block is placed -- never as a training target and never
as an extra loss term.

Three priors are provided:

``vegetation`` (C1)
    Excess Green (ExG) saliency.  ``ExG = 2G - R - B`` on ``[0, 1]`` RGB, then a
    per-image robust percentile normalisation to ``[0, 1]``.

``boundary`` (C2)
    A structure map that combines the raw luminance edge magnitude with the
    edge magnitude of the vegetation saliency, i.e. it prefers plant outlines
    and plant/soil boundaries over bare soil texture, but does not discard
    generic boundaries entirely.

``shuffled_boundary`` (C3)
    The C2 map with its spatial arrangement destroyed while its value
    distribution is preserved exactly: the 16x16 patch cells are permuted with
    a reproducible per-image permutation.  This is the control that separates
    "the prior carries useful statistics" from "the prior is aligned with the
    image".

Every prior is reduced to the ViT patch grid (16x16 for 224px / patch 14) by
mean pooling, so it is aligned with the patch tokens the encoder sees.

Known limitations (documented, not hidden):
    * ExG fails on yellowing/senes of dried vegetation, non-green crops and
      unusual illumination.
    * Edge strength is also produced by soil texture, stones and shadows, so
      the boundary prior can be dominated by bare ground.
    * Both are computed on the *augmented crop*, so the prior describes the
      crop, not the whole source image.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: ViT-H/14 at 224px -> 16x16 patch grid.
PATCH_GRID = 16
PATCH_SIZE = 14


# --------------------------------------------------------------------------- #
# Basic conversions
# --------------------------------------------------------------------------- #

def denormalize(images: torch.Tensor,
                mean: Sequence[float] = IMAGENET_MEAN,
                std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    """Undo ImageNet normalisation and clamp back into the valid ``[0, 1]`` range.

    ``images`` is ``[B, 3, H, W]`` as the training pipeline hands it over.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f'Expected [B,3,H,W] images, got {tuple(images.shape)}')
    mean_t = torch.tensor(mean, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return (images * std_t + mean_t).clamp(0.0, 1.0)


def excess_green(rgb: torch.Tensor) -> torch.Tensor:
    """``ExG = 2G - R - B`` on ``[B, 3, H, W]`` RGB in ``[0, 1]``.

    Returns ``[B, H, W]``.  The raw ExG range is ``[-2, 1]``.
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return 2.0 * g - r - b


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luminance of ``[B, 3, H, W]`` RGB in ``[0, 1]``."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                          [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]])
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0],
                          [0.0, 0.0, 0.0],
                          [1.0, 2.0, 1.0]])


def sobel_magnitude(gray: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of ``[B, H, W]`` (or ``[B, 1, H, W]``) grayscale.

    Uses reflect padding so the border does not produce artificial edges.
    """
    if gray.ndim == 3:
        gray = gray.unsqueeze(1)
    if gray.ndim != 4 or gray.shape[1] != 1:
        raise ValueError(f'Expected [B,1,H,W] grayscale, got {tuple(gray.shape)}')
    kernel = torch.stack([_SOBEL_X, _SOBEL_Y], dim=0).unsqueeze(1).to(
        device=gray.device, dtype=gray.dtype)          # [2,1,3,3]
    padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    grad = F.conv2d(padded, kernel)
    return torch.sqrt(grad[:, 0] ** 2 + grad[:, 1] ** 2 + 1e-12).squeeze(1)


def robust_normalize(x: torch.Tensor, low_q: float = 10.0, high_q: float = 90.0,
                      eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-image percentile normalisation to ``[0, 1]``.

    Returns ``(normalised, usable)`` where ``usable[b]`` is False when the
    image's dynamic range is below ``eps`` (blank soil, flat colour, extreme
    low contrast).  Callers must fall back to random placement for those.
    """
    flat = x.reshape(x.shape[0], -1)
    lo = torch.quantile(flat, low_q / 100.0, dim=1, keepdim=True)
    hi = torch.quantile(flat, high_q / 100.0, dim=1, keepdim=True)
    span = hi - lo
    usable = (span.squeeze(1) > eps)
    safe_span = torch.where(span > eps, span, torch.ones_like(span))
    norm = ((flat - lo) / safe_span).clamp(0.0, 1.0)
    return norm.reshape(x.shape), usable


def sparse_normalize(x: torch.Tensor, high_q: float = 99.0,
                     eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalise a SPARSE non-negative map (e.g. an edge magnitude) to ``[0, 1]``.

    Percentile-range normalisation fails on sparse maps: an edge map is zero
    over most of the image, so its 90th percentile is 0 and the "range" is
    degenerate.  Dividing by a high percentile (default 99th) keeps the
    normalisation well defined while preserving the relative ordering of the
    edges.  Returns ``(normalised, usable)``; ``usable[b]`` is False when the
    map has no signal at all.
    """
    flat = x.reshape(x.shape[0], -1)
    hi = torch.quantile(flat, high_q / 100.0, dim=1, keepdim=True)
    # A very sparse map (a handful of edge pixels) can have a zero percentile
    # even though it carries signal.  Falling back to a fraction of the maximum
    # keeps the normalisation well defined without flattening the ordering.
    hi = torch.maximum(hi, 0.1 * flat.amax(dim=1, keepdim=True))
    usable = (hi.squeeze(1) > eps)
    norm = (flat / hi.clamp_min(eps)).clamp(0.0, 1.0)
    return norm.reshape(x.shape), usable


def patch_pool(x: torch.Tensor, grid: int = PATCH_GRID,
               patch: int = PATCH_SIZE) -> torch.Tensor:
    """Mean-pool a ``[B, H, W]`` map onto the ``[B, grid, grid]`` patch grid."""
    if x.ndim != 3:
        raise ValueError(f'Expected [B,H,W] map, got {tuple(x.shape)}')
    b, h, w = x.shape
    if h % patch or w % patch:
        raise ValueError(f'{h}x{w} is not divisible by patch size {patch}')
    pooled = F.avg_pool2d(x.unsqueeze(1), kernel_size=patch, stride=patch)
    if pooled.shape[-1] != grid or pooled.shape[-2] != grid:
        raise ValueError(
            f'patch pooling gave {tuple(pooled.shape[-2:])}, expected {grid}x{grid}')
    return pooled.squeeze(1)


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #

@dataclass
class PriorResult:
    """One prior map and the diagnostics needed to audit it."""
    map: torch.Tensor                 # [B, grid, grid] in [0, 1]
    usable: torch.Tensor              # [B] bool, False -> fall back to random
    diagnostics: Dict[str, object] = field(default_factory=dict)


class StructurePrior:
    """Compute a structure / vegetation prior for a batch of augmented crops.

    Parameters
    ----------
    mode:
        ``'vegetation'`` (C1), ``'boundary'`` (C2) or ``'shuffled_boundary'`` (C3).
    boundary_edge_weight:
        Weight of the raw luminance edge magnitude in the C2 combination.
        The remainder goes to the vegetation-boundary magnitude.
    low_q / high_q:
        Percentiles used by the robust normalisation.
    """

    VALID_MODES = ('vegetation', 'boundary', 'shuffled_boundary')

    def __init__(self, mode: str = 'vegetation',
                 boundary_edge_weight: float = 0.5,
                 low_q: float = 10.0, high_q: float = 90.0) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f'mode must be one of {self.VALID_MODES}, got {mode!r}')
        self.mode = mode
        self.boundary_edge_weight = float(boundary_edge_weight)
        self.low_q = float(low_q)
        self.high_q = float(high_q)

    # -- internals ---------------------------------------------------------- #

    def _vegetation(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        exg = excess_green(rgb)
        veg, usable = robust_normalize(exg, self.low_q, self.high_q)
        return veg, usable, exg

    def _boundary(self, rgb: torch.Tensor,
                  veg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        gray = luminance(rgb)
        raw_edge = sobel_magnitude(gray)
        edge_n, edge_usable = sparse_normalize(raw_edge)
        veg_edge = sobel_magnitude(veg)
        veg_edge_n, veg_edge_usable = sparse_normalize(veg_edge)
        w = self.boundary_edge_weight
        combined = w * edge_n + (1.0 - w) * veg_edge_n
        usable = edge_usable & veg_edge_usable
        parts = {'raw_edge': raw_edge, 'edge_normalized': edge_n,
                 'veg_edge': veg_edge, 'veg_edge_normalized': veg_edge_n}
        return combined, usable, parts

    # -- public API --------------------------------------------------------- #

    def compute(self, images: torch.Tensor,
                sample_ids: Optional[Sequence[str]] = None,
                base_seed: int = 0) -> PriorResult:
        """Prior map for a batch of NORMALISED crops.

        ``images``: ``[B, 3, H, W]`` normalised with the ImageNet statistics.
        Returns a :class:`PriorResult` whose ``map`` is ``[B, grid, grid]``.
        """
        rgb = denormalize(images)
        veg, veg_usable, exg = self._vegetation(rgb)
        diagnostics: Dict[str, object] = {
            'exg_mean': float(exg.mean()),
            'exg_std': float(exg.std()),
            'vegetation_mean': float(veg.mean()),
        }
        usable = veg_usable

        if self.mode == 'vegetation':
            final = veg
        else:
            boundary, boundary_usable, parts = self._boundary(rgb, veg)
            diagnostics['raw_edge_mean'] = float(parts['raw_edge'].mean())
            diagnostics['veg_edge_mean'] = float(parts['veg_edge'].mean())
            diagnostics['boundary_edge_weight'] = self.boundary_edge_weight
            if self.mode == 'boundary':
                final = boundary
                usable = usable & boundary_usable
            else:
                final = boundary
                usable = usable & boundary_usable

        grid_map = patch_pool(final)
        if self.mode == 'shuffled_boundary':
            # permute the patch cells AFTER pooling: pooling is a local average,
            # so permuting the pooled grid is exactly permuting the cells of the
            # full-resolution map.
            grid_map, shuffle_info = shuffle_patch_cells(
                grid_map, sample_ids=sample_ids, base_seed=base_seed)
            diagnostics.update(shuffle_info)
        if not torch.isfinite(grid_map).all():
            raise FloatingPointError('structure prior contains NaN/Inf')
        diagnostics['usable_fraction'] = float(usable.float().mean())
        diagnostics['mode'] = self.mode
        return PriorResult(map=grid_map, usable=usable, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# C3: destroying the spatial alignment while keeping the value distribution
# --------------------------------------------------------------------------- #

def permutation_for_sample(sample_id: str, base_seed: int, grid: int = PATCH_GRID) -> torch.Tensor:
    """A reproducible permutation of the ``grid*grid`` cells for one image.

    Derived from a hash of the sample id, so it is stable across epochs and
    independent of batch composition or worker scheduling.
    """
    digest = hashlib.sha256(f'{base_seed}:{sample_id}'.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:8], 'big') % (2 ** 63 - 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(grid * grid, generator=generator)


def shuffle_patch_cells(x: torch.Tensor, sample_ids: Optional[Sequence[str]] = None,
                        base_seed: int = 0,
                        grid: int = PATCH_GRID) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Permute the cells of a ``[B, grid, grid]`` map per image.

    The multiset of values is preserved exactly (it is a permutation), while the
    spatial arrangement -- and therefore the alignment with the true image
    content -- is destroyed.

    Without ``sample_ids`` a single generator seeded by ``base_seed`` is used
    for the whole batch, which is still reproducible but weaker.
    """
    if x.ndim != 3 or x.shape[1] != grid or x.shape[2] != grid:
        raise ValueError(f'Expected [B,{grid},{grid}] map, got {tuple(x.shape)}')
    b = x.shape[0]
    out = torch.empty_like(x)
    ident = 0
    for i in range(b):
        if sample_ids is not None and i < len(sample_ids):
            perm = permutation_for_sample(str(sample_ids[i]), base_seed, grid)
        else:
            generator = torch.Generator()
            generator.manual_seed(base_seed + i)
            perm = torch.randperm(grid * grid, generator=generator)
        out[i] = x[i].reshape(-1)[perm].reshape(grid, grid)
        ident += int((perm == torch.arange(grid * grid)).all())
    info = {
        'shuffle_strategy': 'per-image random permutation of the 16x16 patch cells',
        'shuffle_seed_source': 'sha256(base_seed:sample_id)',
        'images_with_identity_permutation': ident,
        'value_multiset_preserved': True,
    }
    return out, info


def lag1_autocorrelation(x: torch.Tensor) -> float:
    """Mean lag-1 spatial autocorrelation of a ``[B, g, g]`` map.

    1.0 means the map is smooth in space; ~0 means the neighbouring cells carry
    no shared information.  Used to quantify how much structure the C3 shuffle
    destroyed.
    """
    if x.ndim != 3:
        raise ValueError(f'Expected [B,g,g] map, got {tuple(x.shape)}')
    xf = x.reshape(x.shape[0], -1)
    xf = xf - xf.mean(dim=1, keepdim=True)
    num = (xf[:, :-1] * xf[:, 1:]).sum(dim=1)
    den = (xf * xf).sum(dim=1)
    valid = den > 1e-12
    if not bool(valid.any()):
        return 0.0
    return float((num[valid] / den[valid]).mean())
