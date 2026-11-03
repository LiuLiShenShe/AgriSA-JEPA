"""Synthetic degradation harness for validating the automatic QC pipeline.

The harness answers a narrow question: *given a deliberately damaged copy of a
real field image, does the pipeline notice?*  It does not, and cannot, estimate
accuracy on natural images, because no human quality label exists.

Every degradation is deterministic.  Each clean base image is paired with its
degraded copies, so sensitivity and false-rejection are measured on the same
scene.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .stage5c_qc_basic import (_blank_fraction, _colourfulness, _entropy,
                               _laplacian_variance, _tenengrad)


# --------------------------------------------------------------------------- #
# Pixel metrics (mirrors stage5c_qc_basic.decode_and_measure, but for an
# in-memory image so the harness never has to write derived pixels to disk).
# --------------------------------------------------------------------------- #
def measure_pil(image: Image.Image, pixel_side: int = 1024) -> Dict[str, object]:
    rgb = image.convert('RGB')
    width, height = rgb.size
    work = rgb.copy()
    work.thumbnail((pixel_side, pixel_side), Image.Resampling.BILINEAR)
    array = np.asarray(work, dtype=np.uint8)
    gray = np.asarray(work.convert('L'), dtype=np.float32)
    return {
        'width': width, 'height': height, 'short_side': int(min(width, height)),
        'laplacian_var': _laplacian_variance(gray),
        'tenengrad': _tenengrad(gray),
        'brightness_mean': float(gray.mean()),
        'entropy': _entropy(gray),
        'colourfulness': _colourfulness(array),
        'blank_fraction': _blank_fraction(gray),
        'aspect_ratio': float(max(width, height) / max(1, min(width, height))),
    }


# --------------------------------------------------------------------------- #
# Deterministic degradations.
# --------------------------------------------------------------------------- #
DEGRADATIONS: List[Tuple[str, str, str]] = [
    ('gaussian_blur', 'sigma1.0', 'mild blur'),
    ('gaussian_blur', 'sigma2.0', 'moderate blur'),
    ('gaussian_blur', 'sigma4.0', 'severe blur'),
    ('defocus_blur', 'radius3', 'defocus'),
    ('motion_blur', 'length9', 'camera shake'),
    ('darken', 'factor0.25', 'underexposed'),
    ('brighten', 'factor1.8', 'overexposed'),
    ('brighten', 'factor2.4', 'severely overexposed'),
    ('low_contrast', 'factor0.3', 'flat contrast'),
    ('jpeg', 'quality10', 'heavy compression'),
    ('jpeg', 'quality25', 'moderate compression'),
    ('gaussian_noise', 'sigma15', 'sensor noise'),
    ('gaussian_noise', 'sigma30', 'heavy noise'),
    ('white_balance', 'blue1.5', 'colour cast'),
    ('downscale', 'factor0.15', 'below minimum resolution'),
    ('occlusion', 'fraction0.4', 'blocked view'),
]


def _seed(*parts: str) -> int:
    digest = hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()
    return int(digest[:8], 16)


def degrade(image: Image.Image, kind: str, value: str,
            seed_key: str = '') -> Optional[Image.Image]:
    rgb = image.convert('RGB')
    if kind == 'gaussian_blur':
        sigma = float(value.replace('sigma', ''))
        return rgb.filter(ImageFilter.GaussianBlur(radius=sigma))
    if kind == 'defocus_blur':
        radius = float(value.replace('radius', ''))
        return rgb.filter(ImageFilter.BoxBlur(radius))
    if kind == 'motion_blur':
        # PIL 12 restricts ``Kernel`` to 3x3 / 5x5, so the horizontal smear is
        # applied directly on the array: average ``length`` shifted copies.
        length = int(value.replace('length', ''))
        half = length // 2
        array = np.asarray(rgb, dtype=np.float32)
        acc = np.zeros_like(array)
        for offset in range(-half, half + 1):
            acc += np.roll(array, offset, axis=1)
        return Image.fromarray(np.clip(acc / length, 0, 255).astype(np.uint8))
    if kind == 'darken':
        return ImageEnhance.Brightness(rgb).enhance(float(value.replace('factor', '')))
    if kind == 'brighten':
        return ImageEnhance.Brightness(rgb).enhance(float(value.replace('factor', '')))
    if kind == 'low_contrast':
        return ImageEnhance.Contrast(rgb).enhance(float(value.replace('factor', '')))
    if kind == 'jpeg':
        import io
        quality = int(value.replace('quality', ''))
        buffer = io.BytesIO()
        rgb.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB').copy()
    if kind == 'gaussian_noise':
        sigma = float(value.replace('sigma', ''))
        rng = np.random.default_rng(_seed(seed_key, kind, value))
        array = np.asarray(rgb, dtype=np.float32)
        noise = rng.normal(0.0, sigma, array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
    if kind == 'white_balance':
        gain = float(value.replace('blue', ''))
        array = np.asarray(rgb, dtype=np.float32).copy()
        array[..., 2] = np.clip(array[..., 2] * gain, 0, 255)
        return Image.fromarray(array.astype(np.uint8))
    if kind == 'downscale':
        factor = float(value.replace('factor', ''))
        small = rgb.resize((max(8, int(rgb.width * factor)),
                            max(8, int(rgb.height * factor))),
                           Image.Resampling.BILINEAR)
        return small
    if kind == 'occlusion':
        fraction = float(value.replace('fraction', ''))
        array = np.asarray(rgb).copy()
        height, width = array.shape[:2]
        side = int(math.sqrt(fraction))
        h = int(height * side)
        w = int(width * side)
        array[0:h, 0:w] = 0
        return Image.fromarray(array)
    return None


# --------------------------------------------------------------------------- #
# Pipeline decisions.
# --------------------------------------------------------------------------- #
def basic_decision_from_metrics(metrics: Dict[str, object], thresholds) -> Tuple[str, List[str]]:
    """Deterministic basic gate, identical in spirit to the Stage 5D one."""
    hard: List[str] = []
    if metrics['short_side'] < thresholds.min_short_side:
        return 'REJECT', ['BELOW_MIN_RESOLUTION']
    if metrics['blank_fraction'] > thresholds.blank_pixel_fraction_max:
        return 'REJECT', ['BLANK_IMAGE']
    if metrics['laplacian_var'] < thresholds.laplacian_var_min:
        hard.append('BLURRY')
    if metrics['tenengrad'] < thresholds.tenengrad_min:
        hard.append('LOW_DETAIL')
    if metrics['brightness_mean'] < thresholds.dark_mean_max:
        hard.append('DARK')
    if metrics['brightness_mean'] > thresholds.bright_mean_min:
        hard.append('OVEREXPOSED')
    if metrics['entropy'] < thresholds.entropy_min:
        hard.append('LOW_INFORMATION')
    if len(hard) >= 2:
        return 'QUARANTINE', hard + ['MULTI_METRIC_CONFLICT']
    if hard:
        return 'REJECT', hard
    return 'PASS', []


def iqa_decision(musiq: Optional[float], spec) -> Tuple[str, str]:
    return spec.evaluate(musiq)


@dataclass
class Result:
    source_id: str
    base_id: str
    variant: str
    kind: str
    value: str
    musiq: Optional[float]
    basic_status: str
    iqa_status: str
    iqa_reason: str
    pipeline_status: str
    short_side: int
    laplacian_var: float
    ret: int  # 1 = detected (reject/quarantine), 0 = passed

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()
