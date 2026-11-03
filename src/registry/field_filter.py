"""Image-level field filtering: quality features and domain triage.

This module produces *evidence*, not verdicts.  Two things are deliberately
kept apart:

1. ``quality_metrics`` / ``quality_gate`` -- objective measurements (size,
   blur, exposure, colour mode).  A quality gate may reject an image.
2. ``suggest_domain`` -- a cheap heuristic that only orders images for human
   or model-assisted inspection.  It never writes a final domain label, and
   callers must not treat its output as a source-of-truth classification.

The policy that decides which domain labels are admissible lives in
``src/registry/domain_policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityGate:
    min_short_side: int = 224
    exclude_grayscale: bool = True
    #: Optional per-source blur floor on the Laplacian variance.  ``None``
    #: means "report the distribution, do not reject".  A single global floor
    #: would silently delete legitimate low-texture soil images.
    min_laplacian_variance: Optional[float] = None
    max_mean_brightness: float = 250.0
    min_mean_brightness: float = 5.0


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian, computed with numpy only."""
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center)
    return float(lap.var())


def _border_fraction(rgb: np.ndarray, band: int = 4,
                     tolerance: float = 18.0) -> Tuple[float, float]:
    """Fraction of pixels close to the image-border colour.

    A laboratory leaf photo has a large uniform white/black background that
    touches the border; a field photo usually does not.  This is a *hint*.
    """
    h, w = rgb.shape[:2]
    band = max(1, min(band, h // 4, w // 4))
    border = np.concatenate([
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)])
    reference = np.median(border, axis=0)
    distance = np.abs(rgb.astype(np.float32) - reference).mean(axis=2)
    fraction = float((distance <= tolerance).mean())
    return fraction, float(reference.mean())


def _vegetation_fraction(rgb: np.ndarray) -> float:
    """Excess-green index fraction, the standard field-vegetation proxy."""
    arr = rgb.astype(np.float32)
    exg = 2.0 * arr[..., 1] - arr[..., 0] - arr[..., 2]
    return float((exg > 20.0).mean())


def quality_metrics(image: Image.Image) -> Dict[str, float]:
    """Measure one decoded image; values are stable and unitless where possible."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    border_fraction, border_luma = _border_fraction(rgb)
    hsv_sat = np.asarray(image.convert('HSV'), dtype=np.uint8)[..., 1]
    return {
        'width': int(image.size[0]),
        'height': int(image.size[1]),
        'short_side': int(min(image.size)),
        'aspect': round(image.size[0] / max(1, image.size[1]), 4),
        'mode': image.mode,
        'mean_brightness': round(float(gray.mean()), 3),
        'std_brightness': round(float(gray.std()), 3),
        'laplacian_variance': round(_laplacian_variance(gray), 4),
        'saturation_mean': round(float(hsv_sat.mean()), 3),
        'border_fraction': round(border_fraction, 4),
        'border_luma': round(border_luma, 3),
        'vegetation_fraction': round(_vegetation_fraction(rgb), 4),
    }


def quality_gate(metrics: Mapping[str, object],
                 gate: Optional[QualityGate] = None) -> Tuple[bool, str]:
    """Apply the objective gate.  Returns ``(passed, reason)``."""
    gate = gate or QualityGate()
    short = int(metrics.get('short_side') or 0)
    if short < gate.min_short_side:
        return False, (f'short side {short}px < {gate.min_short_side}px before any '
                       'resize (upscaling is not allowed to pass this gate)')
    brightness = float(metrics.get('mean_brightness') or 0.0)
    if brightness > gate.max_mean_brightness:
        return False, f'over-exposed: mean brightness {brightness:.1f}'
    if brightness < gate.min_mean_brightness:
        return False, f'under-exposed: mean brightness {brightness:.1f}'
    if gate.exclude_grayscale and str(metrics.get('mode', '')) in ('L', '1', 'I;16'):
        return False, f'grayscale image (mode {metrics.get("mode")})'
    if gate.min_laplacian_variance is not None:
        variance = float(metrics.get('laplacian_variance') or 0.0)
        if variance < gate.min_laplacian_variance:
            return False, (f'blur: Laplacian variance {variance:.2f} < '
                           f'{gate.min_laplacian_variance:.2f}')
    return True, 'passed objective quality gate'


def suggest_domain(metrics: Mapping[str, object],
                   path_hint: str = '') -> Tuple[str, str]:
    """Cheap triage label used only to *order* the inspection queue.

    Returns ``(suggested_domain, reason)``.  The suggestion is intentionally
    conservative: anything ambiguous becomes ``UNKNOWN``.
    """
    border = float(metrics.get('border_fraction') or 0.0)
    luma = float(metrics.get('border_luma') or 0.0)
    veg = float(metrics.get('vegetation_fraction') or 0.0)
    hint = path_hint.lower()

    if border >= 0.80 and (luma >= 200.0 or luma <= 35.0):
        return 'CONTROLLED_LAB', (
            f'uniform border background (border_fraction={border:.2f}, '
            f'border_luma={luma:.0f}) with vegetation_fraction={veg:.2f}')
    if any(token in hint for token in ('uav', 'drone', 'aerial', 'orthomosaic')):
        return 'UAV_AERIAL', 'path/filename names an aerial platform'
    if any(token in hint for token in ('satellite', 'sentinel', 'landsat')):
        return 'SATELLITE_REMOTE', 'path/filename names a satellite product'
    if any(token in hint for token in ('greenhouse', 'glasshouse', 'lab', 'scanner')):
        return 'PROTECTED_CULTIVATION', 'path/filename names a protected environment'
    if border < 0.55 and veg >= 0.05:
        return 'FIELD_IN_SITU', (
            f'non-uniform border (border_fraction={border:.2f}) with '
            f'vegetation_fraction={veg:.2f}; requires documented confirmation')
    return 'UNKNOWN', (
        f'ambiguous triage (border_fraction={border:.2f}, '
        f'vegetation_fraction={veg:.2f})')


def triage_rows(rows: Sequence[Mapping[str, object]],
                gate: Optional[QualityGate] = None) -> Iterable[Dict[str, object]]:
    """Yield one triage record per row carrying ``width``/``height`` metrics."""
    gate = gate or QualityGate()
    for row in rows:
        metrics = {k: row.get(k) for k in (
            'width', 'height', 'short_side', 'mean_brightness', 'laplacian_variance',
            'border_fraction', 'border_luma', 'vegetation_fraction', 'mode')}
        passed, reason = quality_gate(metrics, gate)
        suggestion, triage_reason = suggest_domain(metrics, str(row.get('rel_path', '')))
        yield {
            'sample_id': row.get('sample_id', ''),
            'source_id': row.get('source_id', ''),
            'quality_passed': passed,
            'quality_reason': reason,
            'suggested_domain': suggestion,
            'suggestion_reason': triage_reason,
            'review_required': True,
        }
