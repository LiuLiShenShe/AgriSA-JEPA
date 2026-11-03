"""Sampling profiles over one fixed image set.

A sampling profile decides how often each source is seen in an epoch.  It never
adds or removes entity images, and it never changes the split.  Three profiles
are exported:

* ``natural``    - proportional to the accepted count;
* ``capped``     - the dominant source is capped at a fixed share of the epoch;
* ``balanced``   - every source is seen equally often.

Each profile reports the expected exposure per source per epoch and the implied
source HHI, so the bias of a training run can be inspected before training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd


def hhi(counts: Dict[str, float]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(sum((v / total) ** 2 for v in counts.values()))


def exposure(counts: Dict[str, int], epoch_length: int,
             cap_share: float) -> Dict[str, Dict[str, float]]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    natural = {s: epoch_length * c / total for s, c in counts.items()}
    ceiling = cap_share * epoch_length
    capped = {s: min(v, ceiling) for s, v in natural.items()}
    balanced = {s: epoch_length / len(counts) for s in counts}
    profiles = {'natural': natural, 'capped': capped, 'balanced': balanced}
    for name, values in profiles.items():
        excess = max(0.0, sum(values.values()) - epoch_length)
        if excess:
            # Redistribute the unused budget proportionally so the profile is a
            # real probability distribution over an epoch of exactly this length.
            slack = {s: max(0.0, ceiling - v) for s, v in values.items()}
            total_slack = sum(slack.values())
            if total_slack > 0:
                for source in values:
                    values[source] += excess * slack[source] / total_slack
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/corpus_v3/R3v2_strict_rgb_train.csv')
    parser.add_argument('--epoch-length', type=int, default=150_000)
    parser.add_argument('--cap-share', type=float, default=0.5)
    parser.add_argument('--out', default='data/corpus_v3/R3v2_sampling_profiles.json')
    args = parser.parse_args()

    train = pd.read_csv(ROOT / args.train, low_memory=False)
    if not len(train):
        raise SystemExit('train manifest missing; build the corpus first')
    counts = train['physical_source_id'].value_counts().to_dict()
    if 'upstream_dataset' in train.columns:
        upstream_series = train['upstream_dataset'].fillna('').astype(str)
    else:
        # The published train CSV keeps the canonical columns only; the
        # per-source ledger carries ``upstream_dataset`` for the same ids.
        mapping: Dict[str, str] = {}
        for path in sorted((ROOT / 'data/corpus_v3/images').glob('*.csv')):
            frame = pd.read_csv(path, low_memory=False,
                                usecols=['image_id', 'upstream_dataset'])
            mapping.update(dict(zip(
                frame['image_id'].astype(str),
                frame['upstream_dataset'].fillna('').astype(str))))
        upstream_series = (train['image_id'].astype(str).map(mapping)
                           .fillna('').astype(str))
    upstream_counts = upstream_series.value_counts().to_dict()
    profiles = exposure(counts, args.epoch_length, args.cap_share)
    upstream_profiles = exposure(upstream_counts, args.epoch_length, args.cap_share)

    payload = {
        'note': ('profiles decide exposure frequency only; the entity image set '
                 'is fixed and the split is unchanged'),
        'epoch_length': args.epoch_length,
        'cap_share': args.cap_share,
        'accepted_total': int(sum(counts.values())),
        'train_total': int(sum(counts.values())),
        'by_source': {
            'counts': counts,
            'share': {s: c / sum(counts.values()) for s, c in counts.items()},
            'hhi': hhi(counts),
            'profiles': profiles,
            'profile_hhi': {name: hhi(values) for name, values in profiles.items()},
        },
        'by_upstream_dataset': {
            'counts': upstream_counts,
            'hhi': hhi(upstream_counts),
            'profiles': upstream_profiles,
            'profile_hhi': {name: hhi(values)
                            for name, values in upstream_profiles.items()},
        },
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'sources': len(counts),
                      'hhi': round(payload['by_source']['hhi'], 4),
                      'profile_hhi': {k: round(v, 4) for k, v in
                                      payload['by_source']['profile_hhi'].items()},
                      'top_share': max(payload['by_source']['share'].values())},
                     indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
