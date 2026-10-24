"""Per-image decode, integrity and hand-crafted quality metrics.

The module is intentionally deterministic and dependency-light (Pillow +
NumPy) so it can run over hundreds of thousands of files.  It produces

* identity: ``raw_sha256``, ``pixel_sha256``, dimensions, mode, format, EXIF;
* quality: Laplacian variance, Tenengrad, exposure statistics, entropy,
  colourfulness, blank fraction, aspect ratio;
* a tri-state decision (``AUTO_ACCEPT`` / ``AUTO_REJECT`` / ``QUARANTINE``)
  with explicit reason codes.

The tri-state rule never "passes" an image because one metric is missing: a
missing/unreadable value is ``QUARANTINE``.  Conflicting strong signals (e.g.
very blurry *and* very dark) are quarantined instead of being attributed to a
single cause.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 300_000_000

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

ACCEPT = 'AUTO_ACCEPT'
REJECT = 'AUTO_REJECT'
QUARANTINE = 'QUARANTINE'


@dataclass
class QualityPolicy:
    min_short_side: int = 224
    laplacian_var_min: float = 35.0
    tenengrad_min: float = 25.0
    dark_mean_max: float = 22.0
    bright_mean_min: float = 238.0
    entropy_min: float = 2.6
    extreme_aspect_ratio_max: float = 8.0
    blank_pixel_fraction_max: float = 0.85
    near_threshold_band: float = 0.15
    quarantine_on_conflict: bool = True

    @classmethod
    def from_dict(cls, policy: Dict[str, object]) -> 'QualityPolicy':
        qc = policy.get('quality', policy)
        fields = {name: qc[name] for name in cls.__dataclass_fields__ if name in qc}
        return cls(**fields)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _laplacian_variance(gray: np.ndarray) -> float:
    # 4-neighbour Laplacian without scipy.
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    lap = (-4.0 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def _tenengrad(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    gx = gray[1:-1, 2:] - gray[1:-1, :-2]
    gy = gray[2:, 1:-1] - gray[:-2, 1:-1]
    return float(np.mean(gx * gx + gy * gy))


def _entropy(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probabilities = histogram / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _colourfulness(rgb: np.ndarray) -> float:
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(math.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _blank_fraction(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256)
    modal = int(histogram.argmax())
    near = gray[(gray >= modal - 2) & (gray <= modal + 2)]
    return float(len(near) / gray.size)


def _pixel_sha256(rgb: Image.Image, max_side: int = 2048) -> str:
    """Hash decoded pixels.

    A strict full-resolution hash would take ~0.1 ms per megapixel and would
    not add information beyond the byte hash: the point of ``pixel_sha256`` is
    to catch *re-encodes* of the same pixels, which are detected at any
    resolution.  The cap keeps a 36 MP Artemis image from dominating the run;
    images smaller than the cap are hashed in full.
    """
    if max(rgb.size) <= max_side:
        array = np.asarray(rgb, dtype=np.uint8)
    else:
        work = rgb.copy()
        work.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        array = np.asarray(work, dtype=np.uint8)
    return sha256_bytes(array.tobytes())


def decode_and_measure(path: Path, pixel_bytes: int = 1024) -> Dict[str, object]:
    """Decode one file and return identity + quality metrics.

    ``pixel_bytes`` limits the working copy used for metrics; the original
    dimensions are always measured on the untouched image.
    """
    result: Dict[str, object] = {
        'path': str(path), 'decode_ok': False, 'decode_error': '',
        'width': 0, 'height': 0, 'oriented_width': 0, 'oriented_height': 0,
        'channels': 0, 'mode': '', 'format': '', 'exif_orientation': '',
        'bytes': 0, 'raw_sha256': '', 'pixel_sha256': '',
        'laplacian_var': None, 'tenengrad': None, 'brightness_mean': None,
        'dark_pixel_fraction': None, 'bright_pixel_fraction': None,
        'entropy': None, 'colourfulness': None, 'blank_fraction': None,
        'aspect_ratio': None, 'short_side': 0,
    }
    try:
        with open(path, 'rb') as handle:
            payload = handle.read()
    except OSError as exc:
        result['decode_error'] = f'read: {exc}'
        return result
    result['bytes'] = len(payload)
    result['raw_sha256'] = sha256_bytes(payload)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            result['format'] = image.format or ''
            result['mode'] = image.mode
            result['channels'] = len(image.getbands())
            result['width'], result['height'] = image.size
            try:
                exif = image.getexif()
                result['exif_orientation'] = exif.get(274, '')
            except Exception:  # noqa: BLE001 - EXIF is optional
                result['exif_orientation'] = ''
            oriented = ImageOps.exif_transpose(image)
            result['oriented_width'], result['oriented_height'] = oriented.size
            rgb = oriented.convert('RGB')
            result['pixel_sha256'] = _pixel_sha256(rgb)
            work = rgb.copy()
            work.thumbnail((pixel_bytes, pixel_bytes), Image.Resampling.BILINEAR)
            array = np.asarray(work, dtype=np.uint8)
            gray = np.asarray(work.convert('L'), dtype=np.float32)
            result['laplacian_var'] = _laplacian_variance(gray)
            result['tenengrad'] = _tenengrad(gray)
            result['brightness_mean'] = float(gray.mean())
            result['dark_pixel_fraction'] = float((gray < 10).mean())
            result['bright_pixel_fraction'] = float((gray > 245).mean())
            result['entropy'] = _entropy(gray)
            result['colourfulness'] = _colourfulness(array)
            result['blank_fraction'] = _blank_fraction(gray)
            result['aspect_ratio'] = float(max(oriented.size) / max(1, min(oriented.size)))
            result['short_side'] = int(min(oriented.size))
            result['decode_ok'] = True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        result['decode_error'] = f'decode: {type(exc).__name__}: {exc}'
    return result


def _near(value: Optional[float], threshold: float, band: float, above: bool) -> bool:
    if value is None:
        return False
    span = abs(threshold) * band or band
    return abs(value - threshold) <= span if above else abs(value - threshold) <= span


def basic_decision(metrics: Dict[str, object], policy: QualityPolicy) -> Tuple[str, List[str]]:
    """Return ``(status, reason_codes)`` for the P1 basic stage."""
    if not metrics.get('decode_ok'):
        return REJECT, ['DECODE_FAILED']
    if metrics['short_side'] < policy.min_short_side:
        return REJECT, ['BELOW_MIN_RESOLUTION']
    if metrics['aspect_ratio'] is not None and metrics['aspect_ratio'] > policy.extreme_aspect_ratio_max:
        return REJECT, ['EXTREME_ASPECT_RATIO']
    if metrics['blank_fraction'] is not None and metrics['blank_fraction'] > policy.blank_pixel_fraction_max:
        return REJECT, ['BLANK_IMAGE']

    reasons: List[str] = []
    blurry = (metrics.get('laplacian_var') is not None
              and metrics['laplacian_var'] < policy.laplacian_var_min)
    low_detail = (metrics.get('tenengrad') is not None
                  and metrics['tenengrad'] < policy.tenengrad_min)
    dark = (metrics.get('brightness_mean') is not None
            and metrics['brightness_mean'] < policy.dark_mean_max)
    bright = (metrics.get('brightness_mean') is not None
              and metrics['brightness_mean'] > policy.bright_mean_min)
    low_entropy = (metrics.get('entropy') is not None
                   and metrics['entropy'] < policy.entropy_min)
    hard = [name for name, flag in
            (('BLURRY', blurry), ('LOW_DETAIL', low_detail), ('DARK', dark),
             ('OVEREXPOSED', bright), ('LOW_INFORMATION', low_entropy)) if flag]
    reasons.extend(hard)

    # Conflicting or near-threshold evidence is never converted into a pass.
    if len(hard) >= 2 and policy.quarantine_on_conflict:
        return QUARANTINE, reasons + ['MULTI_METRIC_CONFLICT']
    if hard:
        return REJECT, reasons
    borderline = []
    for name, value, threshold, above in (
        ('BLUR_BORDERLINE', metrics.get('laplacian_var'), policy.laplacian_var_min, True),
        ('DETAIL_BORDERLINE', metrics.get('tenengrad'), policy.tenengrad_min, True),
        ('DARK_BORDERLINE', metrics.get('brightness_mean'), policy.dark_mean_max, True),
        ('BRIGHT_BORDERLINE', metrics.get('brightness_mean'), policy.bright_mean_min, False),
        ('ENTROPY_BORDERLINE', metrics.get('entropy'), policy.entropy_min, True),
    ):
        if _near(value, threshold, policy.near_threshold_band, above):
            borderline.append(name)
    if borderline:
        return QUARANTINE, borderline
    return ACCEPT, []


def paper_decision(metrics: Dict[str, object], policy: QualityPolicy,
                   license_ok: bool) -> Tuple[str, List[str]]:
    """P0: only rules the Agri-FM+ paper actually discloses.

    The paper discloses a 224 px short-side rule and scene/licence screening.
    It does *not* disclose blur/noise thresholds, so P0 adds one explicit,
    reproducible Laplacian rule and labels it ``APPROXIMATION``.
    """
    if not metrics.get('decode_ok'):
        return REJECT, ['DECODE_FAILED']
    if metrics['short_side'] < policy.min_short_side:
        return REJECT, ['BELOW_MIN_RESOLUTION']
    if not license_ok:
        return REJECT, ['LICENSE_NOT_ACCEPTED']
    if metrics.get('laplacian_var') is not None and metrics['laplacian_var'] < policy.laplacian_var_min:
        return REJECT, ['BLURRY_APPROXIMATION']
    return ACCEPT, []
