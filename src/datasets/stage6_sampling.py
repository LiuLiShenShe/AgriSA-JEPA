"""Stage 6 exposure samplers.

Stage 5E.1 publishes two sampling profiles over the frozen R3v3 train split:

``natural``
    sample proportional to the number of rows each ``physical_source_id``
    contributes (the empirical manifest distribution);

``by_source`` (the formal capped profile)
    identical to natural, except the merged ``artemis_imagesafari`` source is
    capped at ``cap_share * epoch_length`` exposures.  Surplus exposure is
    deliberately **not** redistributed to the other sources ("no backfill"),
    so a nominal 150,000-exposure epoch actually emits ~111,171.6 exposures.

This module turns those *fractional exposure budgets* into a deterministic
integer sampling stream and implements the capped sampler used by the B2 arm.

Design decisions (all auditable)

* **Integer allocation.** A source's nominal exposure is fractional because
  ``epoch_length`` (150,000) does not divide the manifest size (147,277).
  Each epoch is realised with the largest-remainder method at
  ``round(sum(capped_exposure))`` emissions.  Artemis keeps exactly the
  integer cap (75,000); only the non-capped sources absorb the rounding,
  and the deficit of the capped source is never moved to them.
* **Interleaving.** Sources are interleaved with smooth weighted round-robin
  (the same family as Bresenham scheduling).  Every prefix of the epoch stream
  tracks the profile proportions to within one sample per source, which makes
  a 2,000-update run (256,000 samples) representative before the epoch ends.
* **Within-source order.** Each source's rows are visited through a seeded
  permutation, cycling to a fresh permutation only when a source is asked for
  more exposures than it has distinct rows (the nominal 150k epoch does this
  for every source by construction).  NumPy's legacy ``RandomState`` is used
  because its stream is guaranteed stable across NumPy versions.
* **Resumption.** The sampler is a pure function of ``(seed, epoch)``; there is
  no mutable RNG state to checkpoint.  Replaying an epoch reproduces the exact
  same index stream, which is what the training loop's mid-epoch resume needs.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Sampler

DEFAULT_EPOCH_LENGTH = 150_000
DEFAULT_CAP_SHARE = 0.5
DEFAULT_CAPPED_SOURCES: Tuple[str, ...] = ('artemis_imagesafari',)


# --------------------------------------------------------------------------- #
# exposure arithmetic
# --------------------------------------------------------------------------- #

def natural_exposure(counts: Mapping[str, int], epoch_length: int) -> Dict[str, float]:
    """Scale manifest counts to a nominal ``epoch_length`` budget."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError('source counts must sum to a positive number')
    return {source: count * epoch_length / total
            for source, count in counts.items()}


def capped_exposure(
    counts: Mapping[str, int],
    epoch_length: int = DEFAULT_EPOCH_LENGTH,
    cap_share: float = DEFAULT_CAP_SHARE,
    capped_sources: Iterable[str] = DEFAULT_CAPPED_SOURCES,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, object]]:
    """Return ``(natural, capped, info)`` for the no-backfill profile.

    ``capped[s] = min(natural[s], cap_share * epoch_length)`` for sources in
    ``capped_sources`` and ``natural[s]`` otherwise.  The removed surplus is
    **not** redistributed; the returned ``info`` states the resulting total and
    the unfulfilled exposure explicitly so callers cannot mistake the capped
    total for the nominal epoch length.
    """
    if not 0 < cap_share <= 1:
        raise ValueError(f'cap_share must be in (0, 1], got {cap_share}')
    natural = natural_exposure(counts, epoch_length)
    capped_set = set(capped_sources)
    unknown = capped_set.difference(counts)
    if unknown:
        raise ValueError(f'capped sources absent from the manifest: {sorted(unknown)}')
    cap = cap_share * epoch_length
    capped = {
        source: (min(exposure, cap) if source in capped_set else exposure)
        for source, exposure in natural.items()
    }
    unfulfilled = sum(natural[s] - capped[s] for s in capped_set)
    total = sum(capped.values())
    info = {
        'cap_share': cap_share,
        'cap_exposure': cap,
        'capped_sources': sorted(capped_set),
        'natural_total_exposure': sum(natural.values()),
        'capped_total_exposure': total,
        'unfulfilled_exposure': unfulfilled,
        'budget_filled': abs(total - epoch_length) < 1e-9,
        'backfill': False,
    }
    return natural, capped, info


def largest_remainder(weights: Mapping[str, float], total: int) -> Dict[str, int]:
    """Integer apportionment of ``weights`` summing exactly to ``total``."""
    if total < 0:
        raise ValueError('total must be non-negative')
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError('weights must sum to a positive number')
    scaled = {key: value * total / weight_sum for key, value in weights.items()}
    allocation = {key: int(math.floor(value)) for key, value in scaled.items()}
    remainder = total - sum(allocation.values())
    # -- stable tie-break: larger fractional part first, then source name
    order = sorted(weights, key=lambda key: (-(scaled[key] - allocation[key]), key))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


def concentration(counts: Mapping[str, int]) -> Dict[str, object]:
    """Shares, HHI, Shannon entropy and the maximum source share."""
    total = sum(counts.values())
    if total <= 0:
        raise ValueError('counts must sum to a positive number')
    shares = {key: value / total for key, value in counts.items()}
    entropy = -sum(share * math.log(share) for share in shares.values() if share > 0)
    max_source = max(shares, key=shares.get)
    return {
        'total': total,
        'counts': dict(sorted(counts.items())),
        'shares': dict(sorted(shares.items())),
        'hhi': sum(share * share for share in shares.values()),
        'entropy': entropy,
        'max_share': shares[max_source],
        'max_source': max_source,
    }


# --------------------------------------------------------------------------- #
# deterministic ordering primitives
# --------------------------------------------------------------------------- #

def _derive_seed(base_seed: int, epoch: int, tag: str) -> int:
    digest = hashlib.sha256(f'{base_seed}:{epoch}:{tag}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 32)


def _seeded_permutation(length: int, seed: int) -> np.ndarray:
    """Deterministic permutation; ``RandomState`` guarantees a stable stream."""
    return np.random.RandomState(seed).permutation(length)


def smooth_weighted_interleave(counts: Mapping[str, int],
                               tie_order: Sequence[str]) -> List[str]:
    """Smooth weighted round-robin: each prefix tracks ``counts`` proportionally."""
    total = sum(counts.values())
    if total <= 0:
        return []
    rank = {source: position for position, source in enumerate(tie_order)}
    missing = set(counts).difference(rank)
    if missing:
        raise ValueError(f'tie_order is missing sources: {sorted(missing)}')
    current = {source: 0 for source in counts}
    sequence: List[str] = []
    sources = list(counts)
    for _ in range(total):
        for source in sources:
            current[source] += counts[source]
        best = max(sources, key=lambda source: (current[source], -rank[source]))
        current[best] -= total
        sequence.append(best)
    return sequence


class ProfileExposureSampler(Sampler[int]):
    """Deterministic no-backfill exposure sampler over dataset indices.

    Parameters
    ----------
    index_sources:
        One ``physical_source_id`` per dataset index, in dataset order.
    epoch_length, cap_share, capped_sources:
        The Stage 5E.1 profile parameters.
    seed:
        Pre-registered sampling seed.  ``set_epoch`` combined with this seed
        fully determines the stream.
    """

    def __init__(
        self,
        index_sources: Sequence[str],
        epoch_length: int = DEFAULT_EPOCH_LENGTH,
        cap_share: float = DEFAULT_CAP_SHARE,
        capped_sources: Iterable[str] = DEFAULT_CAPPED_SOURCES,
        seed: int = 0,
    ) -> None:
        self.index_sources = list(index_sources)
        self.epoch_length = int(epoch_length)
        self.cap_share = float(cap_share)
        self.capped_sources = tuple(capped_sources)
        self.seed = int(seed)
        self.epoch = 0

        self.source_indices: Dict[str, List[int]] = {}
        for index, source in enumerate(self.index_sources):
            self.source_indices.setdefault(source, []).append(index)
        self.counts = {source: len(indices) for source, indices in self.source_indices.items()}

        _, self.capped, self.profile_info = capped_exposure(
            self.counts, self.epoch_length, self.cap_share, self.capped_sources)
        self.epoch_total = int(round(sum(self.capped.values())))
        # -- one epoch = the integer realisation of the capped exposure budget
        self.per_epoch_allocation = largest_remainder(self.capped, self.epoch_total)

        self._schedule: List[int] = []
        self.set_epoch(0)

    # -- deterministic stream -------------------------------------------------
    def _source_sequence(self, source: str, needed: int) -> List[int]:
        pool = self.source_indices[source]
        if not pool:
            raise ValueError(f'source {source!r} has no rows')
        ordered: List[int] = []
        pass_index = 0
        while len(ordered) < needed:
            permutation = _seeded_permutation(
                len(pool), _derive_seed(self.seed, self.epoch, f'{source}:pass{pass_index}'))
            ordered.extend(pool[position] for position in permutation)
            pass_index += 1
        return ordered[:needed]

    def _build_schedule(self) -> List[int]:
        tie_order = sorted(
            self.per_epoch_allocation,
            key=lambda source: _derive_seed(self.seed, self.epoch, f'tie:{source}'))
        interleaved = smooth_weighted_interleave(self.per_epoch_allocation, tie_order)
        cursors = {source: 0 for source in self.per_epoch_allocation}
        orders = {
            source: self._source_sequence(source, self.per_epoch_allocation[source])
            for source in self.per_epoch_allocation
        }
        schedule: List[int] = []
        for source in interleaved:
            schedule.append(orders[source][cursors[source]])
            cursors[source] += 1
        return schedule

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._schedule = self._build_schedule()

    # -- Sampler protocol -----------------------------------------------------
    def __iter__(self) -> Iterable[int]:
        return iter(self._schedule)

    def __len__(self) -> int:
        return len(self._schedule)

    # -- introspection --------------------------------------------------------
    def describe(self) -> Dict[str, object]:
        return {
            'mode': 'profile_capped',
            'seed': self.seed,
            'epoch_length': self.epoch_length,
            'cap_share': self.cap_share,
            'capped_sources': list(self.capped_sources),
            'backfill': False,
            'manifest_counts': dict(sorted(self.counts.items())),
            'natural_exposure': {
                source: value for source, value
                in sorted(natural_exposure(self.counts, self.epoch_length).items())},
            'capped_exposure': dict(sorted(self.capped.items())),
            'capped_total_exposure': self.profile_info['capped_total_exposure'],
            'unfulfilled_exposure': self.profile_info['unfulfilled_exposure'],
            'budget_filled': self.profile_info['budget_filled'],
            'integer_epoch_total': self.epoch_total,
            'integer_allocation': dict(sorted(self.per_epoch_allocation.items())),
        }


# --------------------------------------------------------------------------- #
# manifest helpers
# --------------------------------------------------------------------------- #

def source_counts_from_rows(rows: Iterable[Mapping[str, str]]) -> Dict[str, int]:
    counter: Counter = Counter(str(row.get('physical_source_id', '')) for row in rows)
    if '' in counter:
        raise ValueError('manifest rows are missing physical_source_id')
    return dict(counter)


def index_sources_from_rows(rows: Sequence[Mapping[str, str]]) -> List[str]:
    return [str(row['physical_source_id']) for row in rows]


def emitted_source_counts(
    sampled: Mapping[str, int],
    allocation: Mapping[str, int],
) -> Dict[str, object]:
    """Compare the realised stream with the integer epoch allocation."""
    total = sum(sampled.values())
    summary = concentration(sampled)
    summary['allocation'] = dict(sorted(allocation.items()))
    summary['allocation_total'] = sum(allocation.values())
    summary['matches_allocation'] = dict(sampled) == dict(allocation)
    return summary


__all__ = [
    'DEFAULT_EPOCH_LENGTH',
    'DEFAULT_CAP_SHARE',
    'DEFAULT_CAPPED_SOURCES',
    'natural_exposure',
    'capped_exposure',
    'largest_remainder',
    'concentration',
    'smooth_weighted_interleave',
    'ProfileExposureSampler',
    'source_counts_from_rows',
    'index_sources_from_rows',
    'emitted_source_counts',
]
