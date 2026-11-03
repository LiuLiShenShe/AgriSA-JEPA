"""Threshold-sensitivity and source-balance analysis for Stage 5D.

Two questions are answered offline, with no retraining:

1. How does the accepted count and the per-source retention rate move as the
   MUSIQ accept/reject thresholds move?  A decision that only holds at one
   arbitrary point is not a decision.
2. What would a capped or balanced *sampling* profile look like?  This changes
   only how often each source is exposed, never the number of images.

Inputs are the Stage 5C enriched ledgers (they carry the measured MUSIQ) and the
Stage 5D ledger for the group/split decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def load_enriched_ledgers() -> pd.DataFrame:
    frames = []
    for path in sorted((ROOT / 'data/corpus_v2/images').glob('*_qc.csv')):
        if 'cleanvision_issues' in path.name:
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame['physical_source_id'] = path.stem.replace('_qc', '')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def musiq_curve(frame: pd.DataFrame, accepts: List[float],
                rejects: List[float]) -> pd.DataFrame:
    scores = pd.to_numeric(frame['iqa_musiq'], errors='coerce')
    rows = []
    for accept in accepts:
        for reject in rejects:
            if reject > accept:
                continue
            kept = (scores >= accept)
            quarantine = (scores >= reject) & (scores < accept)
            rejected = scores.notna() & (scores < reject)
            missing = scores.isna()
            rows.append({
                'accept_threshold': accept,
                'reject_threshold': reject,
                'pass': int(kept.sum()),
                'quarantine': int(quarantine.sum()),
                'reject': int(rejected.sum()),
                'missing': int(missing.sum()),
                'retention_among_scored': float(kept.sum() / max(scores.notna().sum(), 1)),
            })
    return pd.DataFrame(rows)


def per_source_retention(frame: pd.DataFrame, accept: float, reject: float) -> pd.DataFrame:
    rows = []
    for source, group in frame.groupby('physical_source_id'):
        scores = pd.to_numeric(group['iqa_musiq'], errors='coerce')
        rows.append({
            'source_id': source,
            'rows': len(group),
            'scored': int(scores.notna().sum()),
            'pass': int((scores >= accept).sum()),
            'quarantine': int(((scores >= reject) & (scores < accept)).sum()),
            'reject': int((scores.notna() & (scores < reject)).sum()),
            'missing': int(scores.isna().sum()),
            'retention_among_scored': float(
                (scores >= accept).sum() / max(scores.notna().sum(), 1)),
        })
    return pd.DataFrame(rows).sort_values('rows', ascending=False)


def hhi(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(sum((v / total) ** 2 for v in counts.values()))


def entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probabilities = np.array([v / total for v in counts.values() if v > 0])
    return float(-(probabilities * np.log(probabilities)).sum())


def sampling_profiles(counts: Dict[str, int], epoch_length: int) -> Dict[str, Dict[str, float]]:
    """Exposure per source for three sampling profiles."""
    total = sum(counts.values())
    natural = {s: epoch_length * c / total for s, c in counts.items()}
    cap = 0.6 * epoch_length  # no single source may exceed 60% of an epoch
    capped = {s: min(natural[s], cap) for s in counts}
    balanced = {s: epoch_length / len(counts) for s in counts}
    return {'natural': natural, 'capped_60pct': capped, 'balanced': balanced}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept', type=float, default=45.0)
    parser.add_argument('--reject', type=float, default=25.0)
    args = parser.parse_args()

    frame = load_enriched_ledgers()
    accepts = [40.0, 45.0, 50.0, 55.0, 60.0]
    rejects = [15.0, 20.0, 25.0, 30.0, 35.0]
    curve = musiq_curve(frame, accepts, rejects)
    per_source = per_source_retention(frame, args.accept, args.reject)

    # Composition of the accepted set, using the Stage 5D ledger if it exists.
    composition_path = ROOT / 'data/corpus_v3/R3v2_strict_rgb_train.csv'
    composition: Dict[str, Dict[str, int]] = {}
    if composition_path.exists():
        accepted = pd.read_csv(composition_path, low_memory=False)
        for column in ('physical_source_id', 'crop', 'country', 'upstream_dataset',
                       'farm_id', 'group_tier'):
            if column in accepted.columns:
                composition[column] = accepted[column].value_counts().to_dict()

    counts = composition.get('physical_source_id', {})
    payload = {
        'musiq_sensitivity': curve.to_dict('records'),
        'per_source_retention_at_policy_thresholds': per_source.to_dict('records'),
        'accepted_composition': composition,
        'diversity': {
            'source_hhi': hhi(counts) if counts else None,
            'source_entropy_bits': entropy(counts) if counts else None,
            'crop_hhi': hhi(composition['crop']) if 'crop' in composition else None,
            'country_hhi': hhi(composition['country']) if 'country' in composition else None,
        },
        'sampling_profiles': sampling_profiles(counts, 150_000) if counts else {},
    }
    REPORTS = ROOT / 'reports'
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'stage5d_threshold_sensitivity.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')
    (REPORTS / 'stage5d_source_balance.csv').write_text(
        per_source.to_csv(index=False), encoding='utf-8')
    print(curve.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
