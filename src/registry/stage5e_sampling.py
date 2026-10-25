"""Sampling profiles for Stage 5E.

Three profiles are computed over one fixed entity set:

``natural``
    Draws proportional to the real source counts.
``capped``
    No source exceeds ``cap_share`` of the epoch budget.  If capping removes
    draws from an over-represented source, the surplus is **not** redistributed
    to other sources; the profile reports whether the exposure budget is still
    filled, so the caller can decide.
``balanced``
    Equal exposure per source.  Analysis reference only.

All three change exposure frequency only.  The entity list, the split
assignment and the number of independent images are inputs and are returned
unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def hhi(counts: Dict[str, float]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    return float(sum((value / total) ** 2 for value in counts.values()))


def entropy(counts: Dict[str, float]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    proportions = np.array([value / total for value in counts.values()], dtype=float)
    proportions = proportions[proportions > 0]
    if proportions.size <= 1:
        return 0.0
    return float(-(proportions * np.log(proportions)).sum())


def natural_profile(counts: Dict[str, int], epoch_length: int) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {source: 0.0 for source in counts}
    return {source: epoch_length * value / total for source, value in counts.items()}


def capped_profile(counts: Dict[str, int], epoch_length: int, cap_share: float
                   ) -> Tuple[Dict[str, float], Dict[str, object]]:
    natural = natural_profile(counts, epoch_length)
    cap = cap_share * epoch_length
    exposures = {source: min(value, cap) for source, value in natural.items()}
    unfulfilled = epoch_length - sum(exposures.values())
    info = {
        'cap_share': cap_share,
        'cap_exposure': cap,
        'capped_sources': sorted(source for source, value in natural.items() if value > cap),
        'unfulfilled_exposure': float(unfulfilled),
        'budget_filled': bool(unfulfilled <= 1e-6),
        'note': ('Surplus exposure from capped sources is deliberately NOT redistributed '
                 'to other sources; doing so would change their natural share.'),
    }
    return exposures, info


def balanced_profile(counts: Dict[str, int], epoch_length: int) -> Dict[str, float]:
    sources = len(counts)
    if sources == 0:
        return {}
    return {source: epoch_length / sources for source in counts}


def build_profiles(counts: Dict[str, int], epoch_length: int, cap_share: float
                   ) -> Dict[str, object]:
    capped, capped_info = capped_profile(counts, epoch_length, cap_share)
    profiles = {
        'natural': natural_profile(counts, epoch_length),
        'capped': capped,
        'balanced': balanced_profile(counts, epoch_length),
    }
    return {
        'epoch_length': epoch_length,
        'cap_share': cap_share,
        'capped_info': capped_info,
        'counts': counts,
        'share': {source: value / float(sum(counts.values())) if sum(counts.values()) else 0.0
                  for source, value in counts.items()},
        'entity_hhi': hhi({source: float(value) for source, value in counts.items()}),
        'entity_entropy': entropy({source: float(value) for source, value in counts.items()}),
        'exposure': profiles,
        'exposure_hhi': {name: hhi(exposure) for name, exposure in profiles.items()},
        'exposure_entropy': {name: entropy(exposure) for name, exposure in profiles.items()},
        'invariants': {
            'entity_count_unchanged': int(sum(counts.values())),
            'source_count_unchanged': len(counts),
            'profiles_change_exposure_only': True,
        },
    }


def profile_from_frame(frame: pd.DataFrame, key: str, epoch_length: int, cap_share: float
                       ) -> Dict[str, object]:
    counts = frame[key].fillna('').astype(str).value_counts().to_dict()
    return build_profiles(counts, epoch_length, cap_share)


def reproducibility_check(profiles: Dict[str, object], frame: pd.DataFrame, key: str,
                          ) -> Dict[str, object]:
    """Sampling must not alter entity counts or splits."""
    counts = frame[key].fillna('').astype(str).replace('', 'unknown') \
        .value_counts().to_dict()
    exposures = profiles['exposure']
    totals = {name: float(sum(exposure.values())) for name, exposure in exposures.items()}
    entity_count = int(sum(counts.values()))
    return {
        'entity_count': entity_count,
        'counts_identical_to_input': counts == profiles['counts'],
        'exposure_budget': totals,
        'entity_count_preserved': profiles['invariants']['entity_count_unchanged'] == entity_count,
    }
