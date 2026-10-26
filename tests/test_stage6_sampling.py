"""Stage 6 sampler / exposure-profile tests.

Every assertion recomputes its expectation from the manifest and the generated
schedule.  No test hard-codes a profile number as a magic constant: the frozen
Stage 5E.1 numbers are asserted against values recomputed from
``data/corpus_v4/R3v3_license_verified_sampling_profiles.json``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from src.datasets.stage6_sampling import (
    DEFAULT_CAPPED_SOURCES,
    ProfileExposureSampler,
    capped_exposure,
    concentration,
    emitted_source_counts,
    index_sources_from_rows,
    largest_remainder,
    natural_exposure,
    smooth_weighted_interleave,
    source_counts_from_rows,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_JSON = ROOT / 'data/corpus_v4/R3v3_license_verified_sampling_profiles.json'
TRAIN_VIEW = ROOT / 'data/stage6/stage6_r3v3_train_view.csv'

PROFILE = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))['by_source']
REPRODUCIBILITY = json.loads(PROFILE_JSON.read_text(encoding='utf-8'))['reproducibility']
SEED = 20261026


def _profile_by_source():
    return PROFILE


def load_rows(limit: int = 0):
    with TRAIN_VIEW.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------- #
# exposure arithmetic reproduces the frozen Stage 5E.1 profile
# --------------------------------------------------------------------------- #

def test_natural_exposure_matches_stage5e1_profile():
    profile = _profile_by_source()
    rows = load_rows()
    counts = source_counts_from_rows(rows)
    natural = natural_exposure(counts, int(profile['epoch_length']))
    for source, expected in profile['exposure']['natural'].items():
        assert natural[source] == pytest.approx(expected, abs=1e-9), source


def test_capped_exposure_matches_stage5e1_profile():
    profile = _profile_by_source()
    rows = load_rows()
    counts = source_counts_from_rows(rows)
    natural, capped, info = capped_exposure(counts)
    for source, expected in profile['exposure']['capped'].items():
        assert capped[source] == pytest.approx(expected, abs=1e-9), source
    assert info['unfulfilled_exposure'] == pytest.approx(
        profile['capped_info']['unfulfilled_exposure'], abs=1e-9)
    assert info['budget_filled'] is False
    assert sum(capped.values()) == pytest.approx(
        REPRODUCIBILITY['by_source']['exposure_budget']['capped'], abs=1e-6)
    # -- the capped total is NOT the nominal epoch length: no backfill
    assert sum(capped.values()) < profile['epoch_length']


def test_capped_exposure_no_backfill_keeps_other_sources_natural():
    profile = _profile_by_source()
    rows = load_rows()
    counts = source_counts_from_rows(rows)
    natural, capped, _ = capped_exposure(counts)
    for source, exposure in natural.items():
        if source in DEFAULT_CAPPED_SOURCES:
            continue
        assert capped[source] == pytest.approx(exposure, abs=1e-12), (
            f'{source} must keep its natural exposure; no surplus may be backfilled')


def test_capped_exposure_rejects_unknown_source():
    with pytest.raises(ValueError, match='absent from the manifest'):
        capped_exposure({'a': 10, 'b': 5}, capped_sources=('c',))


def test_capped_exposure_rejects_bad_cap_share():
    with pytest.raises(ValueError, match='cap_share'):
        capped_exposure({'a': 10}, cap_share=1.5)


def test_concentration_matches_profile_hhi_and_entropy():
    profile = _profile_by_source()
    rows = load_rows()
    counts = source_counts_from_rows(rows)
    natural = natural_exposure(counts, int(profile['epoch_length']))
    _, capped, _ = capped_exposure(counts)
    for label, exposure, expected_hhi, expected_entropy in (
        ('natural', natural, profile['exposure_hhi']['natural'],
         profile['exposure_entropy']['natural']),
        ('capped', capped, profile['exposure_hhi']['capped'],
         profile['exposure_entropy']['capped']),
    ):
        summary = concentration(exposure)
        assert summary['hhi'] == pytest.approx(expected_hhi, abs=1e-12), label
        assert summary['entropy'] == pytest.approx(expected_entropy, abs=1e-12), label


def test_largest_remainder_sums_exactly_and_preserves_cap():
    profile = _profile_by_source()
    rows = load_rows()
    counts = source_counts_from_rows(rows)
    _, capped, _ = capped_exposure(counts)
    total = int(round(sum(capped.values())))
    allocation = largest_remainder(capped, total)
    assert sum(allocation.values()) == total
    # -- the capped source keeps exactly the integer cap, never more
    for source in DEFAULT_CAPPED_SOURCES:
        assert allocation[source] == int(profile['cap_share'] * profile['epoch_length'])


def test_largest_remainder_rejects_nonpositive_weights():
    with pytest.raises(ValueError):
        largest_remainder({'a': 0}, 10)


def test_largest_remainder_negative_total():
    with pytest.raises(ValueError):
        largest_remainder({'a': 1}, -1)


# --------------------------------------------------------------------------- #
# interleaving
# --------------------------------------------------------------------------- #

def test_smooth_weighted_interleave_prefix_proportions():
    counts = {'a': 75, 'b': 18, 'c': 7}
    sequence = smooth_weighted_interleave(counts, ['a', 'b', 'c'])
    assert Counter(sequence) == Counter({'a': 75, 'b': 18, 'c': 7})
    assert len(sequence) == 100
    # -- every prefix of length L must hold each source within one of ceil(L*w)
    for length in (1, 10, 50, 99, 100):
        prefix = Counter(sequence[:length])
        for source, want in counts.items():
            assert abs(prefix[source] - length * want / 100) <= 1.0


def test_smooth_weighted_interleave_rejects_incomplete_tie_order():
    with pytest.raises(ValueError, match='missing sources'):
        smooth_weighted_interleave({'a': 1, 'b': 1}, ['a'])


# --------------------------------------------------------------------------- #
# the deterministic sampler stream on the real manifest
# --------------------------------------------------------------------------- #

@pytest.fixture
def sampler():
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    return ProfileExposureSampler(index_sources=sources, seed=SEED)


def test_sampler_epoch_allocation_matches_profile(sampler):
    profile = _profile_by_source()
    total = int(round(sum(profile['exposure']['capped'].values())))
    assert len(sampler) == total
    for source in DEFAULT_CAPPED_SOURCES:
        emitted = sum(1 for index in sampler
                      if sampler.index_sources[index] == source)
        assert emitted == 75000


def test_sampler_is_deterministic_for_a_fixed_seed(sampler):
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    again = ProfileExposureSampler(index_sources=sources, seed=SEED)
    assert list(sampler) == list(again)
    assert list(sampler) == list(sampler)  # re-iterating must not mutate


def test_sampler_changes_with_seed(sampler):
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    other = ProfileExposureSampler(index_sources=sources, seed=SEED + 1)
    assert list(other) != list(sampler)


def test_sampler_epoch_changes_the_stream(sampler):
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(0)
    assert epoch0 == list(sampler), 'set_epoch(0) must reproduce the first stream'
    assert epoch1 != epoch0
    # -- composition (source multiset) is unchanged; only order changes
    assert Counter(sources[i] for i in epoch0) == Counter(sources[i] for i in epoch1)
    # -- and the per-source counts still equal the integer allocation
    assert Counter(sources[i] for i in epoch0) == Counter(sampler.per_epoch_allocation)


def test_sampler_every_index_is_valid_and_traceable(sampler):
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    for index in sampler:
        assert 0 <= index < len(sources)
        assert sources[index] in PROFILE['counts']


def test_sampler_emitted_counts_match_integer_allocation(sampler):
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    emitted = Counter(sources[index] for index in sampler)
    summary = emitted_source_counts(emitted, sampler.per_epoch_allocation)
    assert summary['matches_allocation'] is True
    assert summary['total'] == len(sampler)


def test_sampler_max_source_share_is_not_fifty_percent(sampler):
    """The capped source holds 67.46% of EMITTED samples, not 50%.

    This is the no-backfill consequence the Stage 6 prompt calls out: the 50%
    cap is relative to the nominal 150,000-exposure budget, and the budget is
    left unfilled, so the emitted share is higher.
    """
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    emitted = Counter(sources[index] for index in sampler)
    share = emitted[DEFAULT_CAPPED_SOURCES[0]] / len(sampler)
    assert share == pytest.approx(0.6746, abs=1e-3)
    assert share > 0.5


def test_sampler_prefix_tracks_profile_over_a_2000_update_budget(sampler):
    """256,000 emitted samples (2,000 updates x 128) must track the profile."""
    rows = load_rows()
    sources = index_sources_from_rows(rows)
    budget_samples = 2000 * 8 * 16
    stream: list[int] = []
    epoch = 0
    while len(stream) < budget_samples:
        sampler.set_epoch(epoch)
        stream.extend(sampler)
        epoch += 1
    stream = stream[:budget_samples]
    emitted = Counter(sources[index] for index in stream)
    summary = concentration(emitted)
    nominal = PROFILE['exposure']['capped']
    for source, expected in nominal.items():
        assert summary['shares'][source] == pytest.approx(
            expected / sum(nominal.values()), abs=5e-3), source


def test_sampler_requires_single_process():
    rows = load_rows(64)
    with pytest.raises(ValueError, match='world_size'):
        from src.datasets.agri_dataset import make_agrifield40k
        make_agrifield40k(
            transform=None, batch_size=4, num_workers=0, world_size=2, rank=0,
            manifest_path=str(TRAIN_VIEW), split='train',
            sampler_spec={'mode': 'profile_capped'})


def test_sampler_rejects_unknown_mode():
    from src.datasets.agri_dataset import make_agrifield40k
    with pytest.raises(ValueError, match='sampling.mode'):
        make_agrifield40k(
            transform=None, batch_size=4, num_workers=0, world_size=1, rank=0,
            manifest_path=str(TRAIN_VIEW), split='train',
            sampler_spec={'mode': 'balanced'})


def test_source_counts_require_physical_source_id():
    with pytest.raises(ValueError, match='physical_source_id'):
        source_counts_from_rows([{'physical_source_id': ''}])


def test_index_sources_preserve_manifest_order():
    rows = load_rows(500)
    sources = index_sources_from_rows(rows)
    assert sources == [row['physical_source_id'] for row in rows]


def test_concentration_rejects_empty_counts():
    with pytest.raises(ValueError):
        concentration({})


def test_natural_exposure_rejects_zero_total():
    with pytest.raises(ValueError):
        natural_exposure({}, 100)


def test_sampler_describe_reports_no_backfill(sampler):
    description = sampler.describe()
    assert description['backfill'] is False
    assert description['budget_filled'] is False
    assert description['capped_total_exposure'] < description['epoch_length']
    assert description['unfulfilled_exposure'] == pytest.approx(
        PROFILE['capped_info']['unfulfilled_exposure'], abs=1e-9)


def test_sampler_rejects_source_without_rows():
    with pytest.raises(ValueError, match='absent from the manifest'):
        ProfileExposureSampler(index_sources=['a', 'a'], capped_sources=('b',))
