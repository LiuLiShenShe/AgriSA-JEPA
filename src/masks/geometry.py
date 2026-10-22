"""Mask-geometry statistics for the Stage 4B audit.

Everything here operates on the **final** masks that the collator returns --
i.e. after the batch-wide common-length truncation -- so a recorded value is
what the encoder and predictor actually receive.  The tools additionally record
the pre-truncation context length, which lets the audit separate "the sampling
rule changed the geometry" from "the batch truncation changed it".

The module is deliberately dependency-light (numpy only) so the audit can run
over the full 12,567-image training manifest without a GPU.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from math import lgamma

GRID = 16
NUM_CELLS = GRID * GRID


# --------------------------------------------------------------------------- #
# Cell / block primitives
# --------------------------------------------------------------------------- #

def cells_from_indices(indices) -> np.ndarray:
    """Boolean ``[GRID, GRID]`` occupancy from a flat patch-index array."""
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    if flat.size and (flat.min() < 0 or flat.max() >= NUM_CELLS):
        raise ValueError(f'patch index out of range [0,{NUM_CELLS}): '
                         f'min={flat.min()} max={flat.max()}')
    grid = np.zeros(NUM_CELLS, dtype=bool)
    grid[flat] = True
    return grid.reshape(GRID, GRID)


def union_cells(masks: Sequence) -> np.ndarray:
    """Union occupancy of a sequence of masks."""
    grid = np.zeros(NUM_CELLS, dtype=bool)
    for mask in masks:
        grid |= cells_from_indices(mask).reshape(-1)
    return grid.reshape(GRID, GRID)


def multiplicity(union: np.ndarray, masks: Sequence) -> np.ndarray:
    """How many target blocks cover each cell: counts for 0..len(masks)."""
    counts = np.zeros(GRID * GRID, dtype=np.int64)
    for mask in masks:
        counts[np.asarray(mask, dtype=np.int64).reshape(-1)] += 1
    hist = np.bincount(counts, minlength=len(masks) + 1)
    return hist[:len(masks) + 1]


def block_boxes(masks: Sequence) -> List[Tuple[int, int, int, int, int]]:
    """``(ymin, ymax, xmin, xmax, area)`` in patch cells for each mask.

    Works for a rectangle (the official target blocks) and degrades gracefully
    to the bounding box of an arbitrary cell set.
    """
    boxes: List[Tuple[int, int, int, int, int]] = []
    for mask in masks:
        grid = cells_from_indices(mask)
        rows, cols = np.nonzero(grid)
        if rows.size == 0:
            boxes.append((0, -1, 0, -1, 0))
            continue
        boxes.append((int(rows.min()), int(rows.max()),
                      int(cols.min()), int(cols.max()), int(grid.sum())))
    return boxes


def block_centers(masks: Sequence) -> np.ndarray:
    """``[n, 2]`` centres (row, col) of the target blocks."""
    centers: List[Tuple[float, float]] = []
    for mask in masks:
        grid = cells_from_indices(mask)
        rows, cols = np.nonzero(grid)
        if rows.size == 0:
            centers.append((np.nan, np.nan))
        else:
            centers.append((float(rows.mean()), float(cols.mean())))
    return np.asarray(centers, dtype=np.float64)


def mean_pairwise_center_distance(centers: np.ndarray) -> float:
    """Mean Euclidean distance between all pairs of block centres (cells)."""
    finite = centers[np.isfinite(centers).all(axis=1)]
    if len(finite) < 2:
        return float('nan')
    diffs = finite[:, None, :] - finite[None, :, :]
    distances = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(len(finite), k=1)
    return float(distances[iu].mean())


# --------------------------------------------------------------------------- #
# One-image record
# --------------------------------------------------------------------------- #

def geometry_record(target_masks: Sequence,
                    context_masks: Sequence,
                    prior_maps: Optional[Dict[str, np.ndarray]] = None,
                    pre_truncation_context_lengths: Optional[Sequence[int]] = None
                    ) -> Dict[str, float]:
    """Full geometry + structure record for one image and one mask method.

    ``prior_maps`` maps a name (``vegetation``, ``boundary``, ...) to a
    ``[GRID, GRID]`` array aligned with the augmented crop.  Only the maps the
    caller supplies are scored; none of them is ever a training target.
    """
    target_union = union_cells(target_masks)
    context_union = union_cells(context_masks)
    hist = multiplicity(target_union, target_masks)
    centers = block_centers(target_masks)
    boxes = block_boxes(target_masks)

    total_target_area = int(sum(len(np.asarray(m).reshape(-1)) for m in target_masks))
    target_cells = int(target_union.sum())
    context_cells = int(context_union.sum())
    overlap = int((target_union & context_union).sum())

    rows, cols = np.nonzero(target_union)
    if rows.size:
        row_std = float(rows.std())
        col_std = float(cols.std())
        bbox = float((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
    else:
        row_std = col_std = bbox = 0.0

    record: Dict[str, float] = {
        'n_target_blocks': len(target_masks),
        'total_target_area': total_target_area,
        'target_cells': target_cells,
        'duplicated_target_area': total_target_area - target_cells,
        'target_overlap_fraction': ((total_target_area - target_cells) / total_target_area
                                    if total_target_area else 0.0),
        'multiplicity_0': int(hist[0]),
        'multiplicity_1': int(hist[1]),
        'multiplicity_2': int(hist[2]) if len(hist) > 2 else 0,
        'multiplicity_3': int(hist[3]) if len(hist) > 3 else 0,
        'multiplicity_4': int(hist[4]) if len(hist) > 4 else 0,
        'context_cells': context_cells,
        'context_fraction': context_cells / NUM_CELLS,
        'target_context_overlap': overlap,
        'coverage_fraction': float((target_union | context_union).mean()),
        'target_row_std': row_std,
        'target_col_std': col_std,
        'target_bbox_cells': bbox,
        'target_center_spread': mean_pairwise_center_distance(centers),
        'target_block_area_mean': float(np.mean([b[4] for b in boxes])) if boxes else 0.0,
        'target_block_aspect_mean': float(np.mean(
            [(b[3] - b[2] + 1) / (b[1] - b[0] + 1) for b in boxes if b[4] > 0]))
        if any(b[4] > 0 for b in boxes) else 0.0,
    }

    if pre_truncation_context_lengths is not None and len(pre_truncation_context_lengths):
        record['context_length_before_truncation'] = float(
            np.mean(pre_truncation_context_lengths))
    else:
        record['context_length_before_truncation'] = float(context_cells)
    record['context_truncation_loss'] = (
        record['context_length_before_truncation'] - float(context_cells))

    if prior_maps:
        for name, prior in prior_maps.items():
            prior = np.asarray(prior, dtype=np.float64)
            if prior.shape != (GRID, GRID):
                raise ValueError(f'{name} prior has shape {prior.shape}, expected '
                                 f'{(GRID, GRID)}')
            multiset = 0.0
            for mask in target_masks:
                flat = np.asarray(mask, dtype=np.int64).reshape(-1)
                multiset += float(prior.reshape(-1)[flat].sum())
            unique = float(prior[target_union].sum()) if target_cells else 0.0
            record[f'mean_{name}_target_multiset'] = (
                multiset / total_target_area if total_target_area else 0.0)
            record[f'mean_{name}_target_union'] = (
                unique / target_cells if target_cells else 0.0)
            record[f'mean_{name}_context'] = (
                float(prior[context_union].mean()) if context_cells else 0.0)
            record[f'mean_{name}_overall'] = float(prior.mean())
            record[f'{name}_union_mass'] = unique
            record[f'{name}_repeat_mass'] = multiset - unique
            record[f'{name}_repeat_mass_ratio'] = (
                (multiset - unique) / unique if unique > 1e-12 else 0.0)
            record[f'{name}_union_mass_fraction'] = (
                unique / float(prior.sum()) if prior.sum() > 1e-12 else 0.0)
    return record


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def summarize(values: Iterable[float]) -> Dict[str, float]:
    """mean / std / median / P10 / P90 for one metric."""
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    if array.size == 0:
        return {'n': 0, 'mean': float('nan'), 'std': float('nan'),
                'median': float('nan'), 'p10': float('nan'), 'p90': float('nan')}
    return {
        'n': int(array.size),
        'mean': float(array.mean()),
        'std': float(array.std(ddof=1)) if array.size > 1 else 0.0,
        'median': float(np.median(array)),
        'p10': float(np.percentile(array, 10)),
        'p90': float(np.percentile(array, 90)),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 statistic: str = 'mean') -> Dict[str, float]:
    """Percentile bootstrap CI of the mean, resampling images.

    Assumption, stated explicitly: images are treated as independent
    observations.  That is optimistic when several frames come from the same
    acquisition sequence; ``tools.audit_mask_geometry`` therefore also reports a
    source-clustered bootstrap as a sensitivity check.
    """
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                       dtype=np.float64)
    if array.size < 2:
        return {'low': float('nan'), 'high': float('nan'), 'n': int(array.size)}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = array[rng.integers(0, array.size, array.size)]
        stats[i] = sample.mean() if statistic == 'mean' else np.median(sample)
    return {'low': float(np.percentile(stats, 2.5)),
            'high': float(np.percentile(stats, 97.5)),
            'n': int(array.size)}


def clustered_bootstrap_ci(sources: Sequence[str], values: Sequence[float],
                           n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap that resamples SOURCES, not images.

    This is the conservative companion to :func:`bootstrap_ci`: it answers
    "would the conclusion survive a different draw of agricultural sources?"
    rather than "a different draw of frames from the same sources?".
    """
    values = np.asarray(values, dtype=np.float64)
    sources = np.asarray(sources)
    keep = np.isfinite(values)
    values, sources = values[keep], sources[keep]
    unique = sorted(set(sources.tolist()))
    if len(unique) < 2 or values.size < 2:
        return {'low': float('nan'), 'high': float('nan'), 'n_sources': len(unique)}
    groups = {s: values[sources == s] for s in unique}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picks = rng.integers(0, len(unique), len(unique))
        pooled = np.concatenate([groups[unique[j]] for j in picks])
        stats[i] = pooled.mean()
    return {'low': float(np.percentile(stats, 2.5)),
            'high': float(np.percentile(stats, 97.5)),
            'n_sources': len(unique)}


def paired_delta(values_a: Sequence[float], values_b: Sequence[float],
                 n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired ``a - b`` mean with a bootstrap CI and a sign test."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f'paired arrays must match: {a.shape} vs {b.shape}')
    keep = np.isfinite(a) & np.isfinite(b)
    diff = a[keep] - b[keep]
    if diff.size == 0:
        return {'n': 0, 'mean_delta': float('nan')}
    ci = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int(diff.size - wins - losses)
    # exact two-sided sign test without scipy
    n = wins + losses
    if n:
        k = min(wins, losses)
        # P(X <= k) for X ~ Binomial(n, 0.5), doubled and clipped at 1
        log_terms = [lgamma(n + 1) - lgamma(i + 1)
                     - lgamma(n - i + 1) for i in range(k + 1)]
        tail = sum(np.exp(t - n * np.log(2.0)) for t in log_terms)
        p_value = float(min(1.0, 2.0 * tail))
    else:
        p_value = 1.0
    return {
        'n': int(diff.size),
        'mean_delta': float(diff.mean()),
        'std_delta': float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        'ci_low': ci['low'],
        'ci_high': ci['high'],
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'sign_test_p_exact': p_value,
    }


def source_balanced_mean(records: Sequence[Dict[str, object]], metric: str,
                         source_key: str = 'source') -> float:
    """Mean of per-source means (equal weight per source)."""
    groups: Dict[str, List[float]] = {}
    for record in records:
        value = record.get(metric)
        if value is None or not np.isfinite(value):
            continue
        groups.setdefault(str(record[source_key]), []).append(float(value))
    if not groups:
        return float('nan')
    return float(np.mean([np.mean(v) for v in groups.values()]))


def vegetation_bins(mean_overall: Sequence[float],
                    edges: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
                    ) -> List[str]:
    """Assign each image to a vegetation-coverage bin from its mean ExG prior."""
    labels: List[str] = []
    for value in mean_overall:
        label = f'[{edges[-1]:.2f},{edges[-1]:.2f}]'
        for lo, hi in zip(edges[:-1], edges[1:]):
            if lo <= value < hi:
                label = f'[{lo:.2f},{hi:.2f})'
                break
        labels.append(label)
    return labels
