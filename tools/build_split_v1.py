#!/usr/bin/env python
"""Build the ``agrifield40k-v1`` frozen split by repairing two structural leaks.

Stage 2C measured two split leaks that pixel hashing cannot see:

* **ACRECropWeed** — one robot capture campaign straddled train and val (nearest
  cross-split frames 2 s apart, 112/120 val frames within 60 s of a train frame).
* **iNatWeeds** — 31 of 1,227 iNaturalist observations (one photographed plant)
  had images on both sides of the split.

This tool writes a NEW data version and never touches ``agrifield40k-v0``.  It
only re-partitions samples already present in the v0 frozen manifests; it does
not admit a single ``pending_audit`` row, so the source/licence gate stays unmet
and the new version is still PROVISIONAL.

Split rules
-----------
ACRECropWeed
    The whole source is treated as ONE capture campaign and moved to train.
    The 791 images span 2022-10-06 17:16 to 2022-10-10 17:30 with gaps of up to
    87 min.  Those gaps do NOT prove independent sessions — same robot, same
    task, same field — and the task forbids inventing independent sequences from
    a time threshold alone.  So the source is not split at all: all 120 val
    images move to train.  The consequence (no ACRECropWeed in val) is recorded
    rather than papered over.

iNatWeeds
    Grouping unit is the observation id parsed from ``observation_url``.  All
    images of an observation go to one split.  The assignment is a pure function
    of (seed, sorted observation ids): observations are shuffled with the fixed
    seed and greedily placed into val until the source's v0 val fraction is
    reached, so the source-level train/val ratio is preserved.  Observations
    with a missing/unparseable id are conservatively forced to train (never val).

Everything else is copied unchanged from v0.

Usage
-----
    ./.venv/bin/python -m tools.build_split_v1
    ./.venv/bin/python -m tools.build_split_v1 --dry-run
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DATA_VERSION = 'agrifield40k-v1'
SEED = 20261019
OBSERVATION_RE = re.compile(r'/observations?/(\d+)')
ACW_SOURCE = 'ACRECropWeed'
INAT_SOURCE = 'iNatWeeds'

V0_TRAIN = Path('data/frozen/agrifield40k-v0_train.csv')
V0_VAL = Path('data/frozen/agrifield40k-v0_val.csv')
OUT_TRAIN = Path('data/frozen/agrifield40k-v1_train.csv')
OUT_VAL = Path('data/frozen/agrifield40k-v1_val.csv')
OUT_MANIFEST = Path('data/frozen/agrifield40k-v1_freeze_manifest.json')

FIELDNAMES = [
    'sample_id', 'path', 'source_prefix', 'source', 'candidate_source',
    'source_confirmed', 'original_id', 'split', 'original_split', 'width',
    'height', 'mode', 'format', 'bytes', 'sha256', 'pixel_sha256', 'phash64',
    'dhash64', 'decode_ok', 'license', 'compilation_license', 'source_url',
    'source_revision', 'attribution', 'observation_url', 'status', 'reason',
    'component',
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def observation_id(row: Dict[str, str]) -> Optional[str]:
    match = OBSERVATION_RE.search(row.get('observation_url', '') or '')
    return match.group(1) if match else None


def acw_session_key(original_id: str) -> str:
    """Calendar date of an ACRECropWeed frame.

    Reported for diagnosis only.  The v1 rule does NOT split on this key: the
    gaps between days are not evidence of independent capture sessions.
    """
    match = re.search(r'(\d{4}-\d{2}-\d{2})', original_id or '')
    return match.group(1) if match else 'unknown'


def assign_inat_observations(
    rows: Sequence[Dict[str, str]],
    target_val_fraction: float,
    seed: int,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Deterministically assign each iNatWeeds observation to train or val.

    Pure function of (rows, target_val_fraction, seed).  Sorting the observation
    ids before shuffling makes the result independent of manifest row order.
    """
    by_obs: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    missing: List[Dict[str, str]] = []
    for row in rows:
        obs = observation_id(row)
        if obs is None:
            missing.append(row)
        else:
            by_obs[obs].append(row)

    obs_ids = sorted(by_obs)
    rng = random.Random(seed)
    rng.shuffle(obs_ids)

    total_images = sum(len(v) for v in by_obs.values())
    target_val = int(round(total_images * target_val_fraction))
    assignment: Dict[str, str] = {}
    val_images = 0
    for obs in obs_ids:
        size = len(by_obs[obs])
        if val_images + size <= target_val:
            assignment[obs] = 'val'
            val_images += size
        else:
            assignment[obs] = 'train'

    stats = {
        'observations': len(by_obs),
        'images_with_observation_id': total_images,
        'target_val_fraction': target_val_fraction,
        'target_val_images': target_val,
        'assigned_val_images': val_images,
        'assigned_train_images': total_images - val_images,
        'missing_observation_id_images': len(missing),
        'missing_observation_id_policy': 'forced to train',
        'seed': seed,
    }
    return assignment, stats


def build_v1(
    v0_train: Sequence[Dict[str, str]],
    v0_val: Sequence[Dict[str, str]],
    seed: int = SEED,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
    """Re-partition v0 into v1.  Pure function; no file IO."""
    combined = list(v0_train) + list(v0_val)

    v0_inat = [r for r in combined if r['source'] == INAT_SOURCE]
    v0_inat_val = sum(1 for r in v0_inat if r['split'] == 'val')
    inat_val_fraction = v0_inat_val / len(v0_inat) if v0_inat else 0.0

    inat_assignment, inat_stats = assign_inat_observations(
        v0_inat, inat_val_fraction, seed)

    train: List[Dict[str, str]] = []
    val: List[Dict[str, str]] = []
    moved = collections.Counter()
    moved_examples: Dict[str, List[str]] = collections.defaultdict(list)

    for row in combined:
        row = dict(row)
        old_split = row['split']
        source = row['source']

        if source == ACW_SOURCE:
            new_split = 'train'
            reason = 'acw_whole_capture_campaign_to_train'
        elif source == INAT_SOURCE:
            obs = observation_id(row)
            if obs is None:
                new_split = 'train'
                reason = 'inat_missing_observation_id_forced_train'
            else:
                new_split = inat_assignment[obs]
                reason = f'inat_observation_level_split_obs={obs}'
        else:
            new_split = old_split
            reason = 'unchanged_from_v0'

        row['split'] = new_split
        row['original_split'] = old_split
        if new_split != old_split:
            moved[(source, old_split, new_split)] += 1
            if len(moved_examples[f'{source}:{old_split}->{new_split}']) < 5:
                moved_examples[f'{source}:{old_split}->{new_split}'].append(
                    row['sample_id'])
        (val if new_split == 'val' else train).append(row)

    # -- invariants asserted before anything is written
    train_ids = {r['sample_id'] for r in train}
    val_ids = {r['sample_id'] for r in val}
    if train_ids & val_ids:
        raise RuntimeError(f'{len(train_ids & val_ids)} sample ids appear in both splits')
    if train_ids | val_ids != {r['sample_id'] for r in combined}:
        raise RuntimeError('v1 does not cover exactly the v0 sample set')

    # -- grouping invariants the repair exists to guarantee.  Check the NEW rows
    #    (train + val), not `combined`, which still holds the unmutated v0 rows.
    result = train + val
    acw_splits = {r['split'] for r in result if r['source'] == ACW_SOURCE}
    if acw_splits != {'train'}:
        raise RuntimeError(f'ACRECropWeed still spans {sorted(acw_splits)}')
    obs_splits: Dict[str, set] = collections.defaultdict(set)
    for row in result:
        if row['source'] != INAT_SOURCE:
            continue
        obs = observation_id(row)
        if obs is not None:
            obs_splits[obs].add(row['split'])
    crossing = {k: v for k, v in obs_splits.items() if len(v) > 1}
    if crossing:
        raise RuntimeError(f'{len(crossing)} iNatWeeds observations still cross splits')

    train.sort(key=lambda r: r['sample_id'])
    val.sort(key=lambda r: r['sample_id'])

    def breakdown(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
        return dict(sorted(collections.Counter(r['source'] for r in rows).items()))

    # -- count the moves from the ACW source only, using the new rows' recorded
    #    `original_split` so this works even when the input rows carry no such
    #    column (they are set above from the observed split).
    acw_moved = sum(
        1 for r in result
        if r['source'] == ACW_SOURCE and r.get('original_split') == 'val')

    summary = {
        'seed': seed,
        'counts': {'train': len(train), 'val': len(val), 'total': len(train) + len(val)},
        'v0_counts': {'train': len(v0_train), 'val': len(v0_val)},
        'per_source': {'train': breakdown(train), 'val': breakdown(val)},
        'moved': {f'{s}:{a}->{b}': n for (s, a, b), n in sorted(moved.items())},
        'moved_examples': dict(moved_examples),
        'inat': inat_stats,
        'acw': {
            'policy': 'whole capture campaign -> train',
            'images': sum(1 for r in result if r['source'] == ACW_SOURCE),
            'val_images_moved_to_train': acw_moved,
            'session_dates': dict(sorted(collections.Counter(
                acw_session_key(r['original_id'])
                for r in result if r['source'] == ACW_SOURCE).items())),
            'note': ('Gaps between frames (up to 87 min) do not prove independent '
                     'sessions; no session split is invented.'),
        },
    }
    return train, val, summary


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})


def build_manifest(summary: Dict[str, object], train_path: Path, val_path: Path,
                   v0_manifest_path: Path) -> Dict[str, object]:
    v0 = json.loads(v0_manifest_path.read_text(encoding='utf-8')) \
        if v0_manifest_path.is_file() else {}
    manifest = {
        'data_version': DATA_VERSION,
        'freeze_status': 'provisional',
        'freeze_revision': v0.get('freeze_revision'),
        'card_url': v0.get('card_url'),
        'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'stage': 'Stage 2D',
        'supersedes': 'agrifield40k-v0',
        'included_sample_count': summary['counts'],
        'v0_included_sample_count': summary['v0_counts'],
        'per_source_breakdown': summary['per_source'],
        'split_rule': {
            'origin': 'v0 publisher split, repaired for two measured structural leaks',
            'acw': summary['acw'],
            'inat': summary['inat'],
            'other_sources': 'copied unchanged from v0',
            'train_val_disjoint': True,
            'seed': summary['seed'],
        },
        'changes_from_v0': {
            'moved_counts': summary['moved'],
            'moved_examples': summary['moved_examples'],
        },
        'leakage_status': {
            'acw_capture_sequence_cross_split': 'resolved (whole source in train)',
            'inat_observation_cross_split': 'resolved (observation-level split)',
            'unreviewed_perceptual_duplicate_candidates': 784,
            'unresolved_source_prefixes_excluded': 14743,
        },
        'gate': {
            'source_license_gate_met': False,
            'reason': ('14,743 pending_audit images are still unadopted and the 784 '
                       'duplicate candidate edges are still unreviewed; the v1 '
                       'repair does not change either, so the version stays '
                       'PROVISIONAL.'),
        },
        'sha256': {
            train_path.name: sha256_file(train_path),
            val_path.name: sha256_file(val_path),
        },
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    v0_train = read_manifest(V0_TRAIN)
    v0_val = read_manifest(V0_VAL)
    train, val, summary = build_v1(v0_train, v0_val, seed=args.seed)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print('\n[dry-run] no files written')
        return 0

    write_csv(OUT_TRAIN, train)
    write_csv(OUT_VAL, val)
    manifest = build_manifest(summary, OUT_TRAIN, OUT_VAL,
                              Path('data/frozen/freeze_manifest.json'))
    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'\nwrote {OUT_TRAIN} ({len(train)}), {OUT_VAL} ({len(val)}), {OUT_MANIFEST}')
    print(f"sha256 train={manifest['sha256'][OUT_TRAIN.name]}")
    print(f"sha256 val  ={manifest['sha256'][OUT_VAL.name]}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())