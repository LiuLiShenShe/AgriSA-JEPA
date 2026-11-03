"""Modality classification with explicit evidence.

A JPEG that Pillow decodes as three channels is *not* evidence of visible-light
RGB: NDVI false-colour exports, NIR-raw captures and multispectral composites
are all commonly stored as 3-channel JPEGs.

The rules therefore work from documented dataset structure first:

* ``agrifm_ref11`` ships two sibling channel folders, ``rgb/`` (file token
  ``rgbnormal``) and ``nir/`` (token ``nirraw``).  The folder token decides the
  modality; the row's ``PIL.mode`` is never consulted.
* A source whose published/registered modality is ``multispectral`` or
  ``RGB+NDVI`` cannot be claimed as visible RGB without a per-image channel
  product that documents the visible bands.
* Everything else is visible RGB only when the source registry declares RGB
  *and* a sampled pixel check (``tools/stage5d_validate_modality.py``) does not
  contradict it.

``UNKNOWN_MODALITY`` is a first-class outcome and is quarantined.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .stage5d_metadata_schema import (MODALITY_FALSE_COLOR,
                                      MODALITY_MULTISPECTRAL,
                                      MODALITY_NDVI_PSEUDOCOLOR,
                                      MODALITY_NIR_RAW, MODALITY_RGB_VISIBLE,
                                      MODALITY_STATUS_EVIDENCED,
                                      MODALITY_STATUS_NOT_VISIBLE,
                                      MODALITY_STATUS_UNKNOWN,
                                      MODALITY_UNKNOWN)

#: Evidence attached to a source-level declaration.  ``REGISTRY_DECLARED_RGB``
#: is weaker than a published channel layout but is still a documented claim by
#: the dataset publisher; it is recorded, never upgraded.
SOURCE_MODALITY_EVIDENCE: Dict[str, tuple] = {
    'agrifm_ref11': ('RGB+NDVI', 'DATASET_CHANNEL_LAYOUT'),
    'agrifm_ref82': ('multispectral', 'REGISTRY_DECLARED_MULTISPECTRAL'),
}


def _declared(source_id: str, source_modality: Optional[str]) -> tuple:
    if source_id in SOURCE_MODALITY_EVIDENCE:
        declared, evidence = SOURCE_MODALITY_EVIDENCE[source_id]
        return declared, evidence
    declared = (source_modality or '').strip() or 'UNKNOWN'
    if declared.upper() == 'RGB':
        return 'RGB', 'REGISTRY_DECLARED_RGB'
    if 'MULTI' in declared.upper():
        return declared, 'REGISTRY_DECLARED_MULTISPECTRAL'
    if 'NDVI' in declared.upper():
        return declared, 'REGISTRY_DECLARED_NDVI'
    return declared, 'NONE'


def classify_row(source_id: str, relative_path: str,
                 source_modality: Optional[str] = None) -> Dict[str, str]:
    """Classify a single image.  Returns modality/evidence/status."""
    path = str(relative_path).replace('\\', '/').lower()
    declared, evidence = _declared(source_id, source_modality)

    if source_id == 'agrifm_ref11':
        # The sibling channel folders are the dataset's own statement about
        # what each file is.  ``right``/``nir`` never counts as visible RGB.
        is_nir = '/nir/' in f'/{path}' or 'nirraw' in path
        is_rgb = '/rgb/' in f'/{path}' or 'rgbnormal' in path
        if is_nir:
            return {'modality': MODALITY_NIR_RAW,
                    'modality_evidence': 'DATASET_CHANNEL_LAYOUT',
                    'modality_status': MODALITY_STATUS_NOT_VISIBLE}
        if is_rgb:
            return {'modality': MODALITY_RGB_VISIBLE,
                    'modality_evidence': 'DATASET_CHANNEL_LAYOUT',
                    'modality_status': MODALITY_STATUS_EVIDENCED}
        return {'modality': MODALITY_UNKNOWN,
                'modality_evidence': 'NONE',
                'modality_status': MODALITY_STATUS_UNKNOWN}

    if 'MULTI' in declared.upper():
        return {'modality': MODALITY_MULTISPECTRAL,
                'modality_evidence': evidence,
                'modality_status': MODALITY_STATUS_NOT_VISIBLE}

    if declared.upper() == 'RGB':
        return {'modality': MODALITY_RGB_VISIBLE,
                'modality_evidence': evidence,
                'modality_status': MODALITY_STATUS_EVIDENCED}

    return {'modality': MODALITY_UNKNOWN,
            'modality_evidence': evidence,
            'modality_status': MODALITY_STATUS_UNKNOWN}


def apply_modality(frame: pd.DataFrame,
                   source_modalities: Dict[str, str]) -> pd.DataFrame:
    out = frame.copy()
    records = [
        classify_row(row['physical_source_id'], row.get('relative_path', ''),
                     source_modalities.get(row['physical_source_id'], ''))
        for _, row in out.iterrows()
    ]
    out['modality'] = [r['modality'] for r in records]
    out['modality_evidence'] = [r['modality_evidence'] for r in records]
    out['modality_status'] = [r['modality_status'] for r in records]
    return out


# --------------------------------------------------------------------------- #
# Pixel-level screening.  This is deliberately a *contradiction check*, not a
# positive proof of spectral content.
# --------------------------------------------------------------------------- #
def channel_stats(array) -> Dict[str, float]:
    """Return cheap statistics used to flag a file that is not photographic RGB."""
    import numpy as np
    rgb = np.asarray(array, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return {'valid': 0.0}
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Correlations are undefined for a flat channel; guard the denominator.
    def corr(x, y):
        xs, ys = x.ravel(), y.ravel()
        if xs.std() == 0 or ys.std() == 0:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])
    return {
        'valid': 1.0,
        'mean_r': float(r.mean()),
        'mean_g': float(g.mean()),
        'mean_b': float(b.mean()),
        'corr_rg': corr(r, g),
        'corr_rb': corr(r, b),
        'corr_gb': corr(g, b),
    }


def screen_visible_rgb(stats: Dict[str, float]) -> tuple:
    """Return ``(passed, reason)`` for a sampled visible-RGB sanity check.

    Thresholds are intentionally loose: they only reject a file that clearly
    is not a photographic RGB image (a dead green channel, a channel offset
    larger than any plausible white balance).  They do not prove that the
    sensor was a visible-light camera.
    """
    if not stats.get('valid'):
        return False, 'NOT_THREE_CHANNEL'
    means = (stats['mean_r'], stats['mean_g'], stats['mean_b'])
    if min(means) <= 1.0:
        return False, 'DEAD_CHANNEL'
    if max(means) / max(min(means), 1e-6) > 25.0:
        return False, 'EXTREME_CHANNEL_OFFSET'
    if max(stats['corr_rg'], stats['corr_rb'], stats['corr_gb']) < 0.15:
        return False, 'UNCORRELATED_CHANNELS'
    return True, 'VISIBLE_RGB_CONSISTENT'
