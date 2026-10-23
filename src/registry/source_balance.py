"""Source-composition accounting and sampling strategies.

The v1 corpus is dominated by LUCASVision (~70% of train).  Removing
PlantVillage does not fix that.  This module measures the composition that
actually exists and offers three transparent sampling plans.  It never deletes
legitimate data and never claims that a balanced plan is better -- that is a
downstream experiment (arms B1/B2 in ``reports/stage5b_pretraining_plan.md``).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

STRATEGIES = ('natural', 'source_balanced', 'source_capped')


def _share(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _value(row: Mapping[str, object], key: str) -> str:
    """Read a composition field, accepting the corpus' ``source_prefix`` alias.

    Frozen v1 rows call the source column ``source``/``source_prefix`` while the
    ingested manifests call it ``source_id``.  Treating the two as the same
    field avoids silently collapsing every row into ``UNKNOWN``.
    """
    value = row.get(key)
    if (value is None or value == '') and key == 'source_id':
        value = row.get('source_prefix')
    return str(value or 'UNKNOWN')


def composition(rows: Sequence[Mapping[str, object]],
                key: str = 'source_id') -> Dict[str, Dict[str, object]]:
    """Count images, share and distinct groups per value of ``key``."""
    counts: Counter = Counter()
    groups: Dict[str, set] = defaultdict(set)
    for row in rows:
        value = _value(row, key)
        counts[value] += 1
        group = str(row.get('group_key') or row.get('component')
                    or row.get('sample_id') or '')
        if group:
            groups[value].add(group)
    total = sum(counts.values())
    return {
        value: {
            'images': count,
            'share': _share(count, total),
            'groups': len(groups[value]),
        }
        for value, count in counts.most_common()
    }


def max_source_share(rows: Sequence[Mapping[str, object]],
                     key: str = 'source_id') -> float:
    table = composition(rows, key)
    return max((entry['share'] for entry in table.values()), default=0.0)


def shannon_entropy(rows: Sequence[Mapping[str, object]],
                    key: str = 'source_id') -> float:
    """Entropy of the image distribution over ``key`` in bits."""
    table = composition(rows, key)
    total = sum(entry['images'] for entry in table.values())
    if not total:
        return 0.0
    entropy = 0.0
    for entry in table.values():
        p = entry['images'] / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 6)


def strategy_weights(rows: Sequence[Mapping[str, object]],
                     strategy: str = 'natural',
                     cap: float = 0.5,
                     min_source_images: int = 1) -> Dict[str, float]:
    """Per-source sampling multipliers for one strategy.

    ``natural``          -- 1.0 for every source (the corpus as it is).
    ``source_balanced``  -- every source contributes the same expected mass.
    ``source_capped``    -- no source exceeds ``cap`` of the sampled mass; the
                            excess is redistributed proportionally over the rest.

    Sources with fewer than ``min_source_images`` images are left at 1.0 so a
    tiny source cannot be repeated hundreds of times (that would show the model
    the same image over and over -- a documented failure mode of naive
    balancing).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f'unknown strategy {strategy!r}; expected {STRATEGIES}')
    if not 0.0 < cap <= 1.0:
        raise ValueError('cap must be in (0, 1]')
    table = composition(rows, 'source_id')
    weights = {source: 1.0 for source in table}
    if strategy == 'natural' or len(table) < 2:
        return weights
    eligible = {s: e['images'] for s, e in table.items()
                if e['images'] >= min_source_images}
    if len(eligible) < 2:
        return weights
    total = sum(eligible.values())
    if strategy == 'source_balanced':
        target = total / len(eligible)
        for source, images in eligible.items():
            weights[source] = round(target / images, 6)
        return weights
    # source_capped: solve w_s such that capped sources are held at cap and the
    # rest share (1 - cap * n_capped) proportionally.
    capped = [s for s, n in eligible.items() if n / total > cap]
    if not capped:
        return weights
    free = [s for s in eligible if s not in capped]
    if not free:
        return weights
    remaining = 1.0 - cap * len(capped)
    free_total = sum(eligible[s] for s in free)
    for source in capped:
        weights[source] = round((cap * total) / eligible[source], 6)
    for source in free:
        weights[source] = round((remaining * total / free_total), 6)
    return weights


def effective_sample_counts(rows: Sequence[Mapping[str, object]],
                            weights: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    """Images, weight and effective images per source."""
    table = composition(rows, 'source_id')
    out: Dict[str, Dict[str, float]] = {}
    for source, entry in table.items():
        weight = float(weights.get(source, 1.0))
        out[source] = {
            'images': entry['images'],
            'weight': weight,
            'effective_images': round(entry['images'] * weight, 2),
            'share_in_corpus': entry['share'],
        }
    total_effective = sum(v['effective_images'] for v in out.values())
    for entry in out.values():
        entry['share_after_sampling'] = (
            round(entry['effective_images'] / total_effective, 6)
            if total_effective else 0.0)
    return out


def describe(rows: Sequence[Mapping[str, object]],
             keys: Sequence[str] = ('source_id', 'acquisition_domain', 'platform')
             ) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Composition over several columns, for the distribution report."""
    return {key: composition(rows, key) for key in keys}
