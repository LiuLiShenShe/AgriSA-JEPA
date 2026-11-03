#!/usr/bin/env python
"""Downstream-holdout cross-check for Stage 5D.

Three evidence levels are reported separately, never merged into one
"NO_LEAKAGE" claim:

* **source level** - a whole registered holdout dataset must never appear as a
  pretraining source;
* **prefix/identity level** - a mixed pool may embed a holdout dataset under a
  filename prefix (the Stage 5C ``agrifield40k_pool`` case);
* **file/pixel/near-duplicate level** - hashes computed from the holdout files
  that exist on this machine (PhenoBench, GWHD 2021).

The remaining levels (metadata/site overlap with sources that are not on disk)
are reported as *not verifiable here*.

Outputs
-------
data/corpus_v3/holdout_hash_index.csv        (cached hash index)
reports/stage5d_holdout_evidence.json
reports/stage5d_holdout_evidence.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.registry.ingest import phash64

HOLDOUT_ROOTS = {
    'phenobench': Path('/data/datasets/PhenoBench/PhenoBench-v110'),
    'gwhd_2021': Path('/data/datasets/GWHD2021/extracted'),
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_hash_index(cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache, dtype={'phash': str, 'raw_sha256': str})
    rows: List[Dict[str, str]] = []
    for dataset, root in HOLDOUT_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    phash = phash64(image)
                raw = sha256_file(path)
            except Exception:  # noqa: BLE001
                continue
            rows.append({'dataset': dataset, 'raw_sha256': raw,
                         'phash': str(phash), 'path': str(path)})
    frame = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def near_matches(candidate: pd.DataFrame, holdout: pd.DataFrame,
                 max_distance: int = 6) -> pd.DataFrame:
    """Return candidate/holdout pairs sharing a pHash neighbourhood."""
    if candidate.empty or holdout.empty:
        return pd.DataFrame()
    candidates = candidate[candidate['phash'].notna()].reset_index(drop=True)
    holdout = holdout[holdout['phash'].notna()].reset_index(drop=True)
    cand_hashes = [int(str(v), 16) for v in candidates['phash']]
    hold_hashes = [int(str(v), 16) for v in holdout['phash']]
    if not cand_hashes or not hold_hashes:
        return pd.DataFrame()
    import faiss
    index = faiss.IndexBinaryFlat(64)
    hold_array = np.array(hold_hashes, dtype=np.uint64)
    index.add(hold_array.view(np.uint8).reshape(len(hold_array), 8))
    query = np.array(cand_hashes, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    distances, neighbours = index.search(query, min(5, len(hold_hashes)))
    matches = []
    for row_index in range(len(candidates)):
        for distance, other in zip(distances[row_index], neighbours[row_index]):
            if other < 0 or distance > max_distance:
                continue
            matches.append({
                'candidate_image_id': candidates.iloc[row_index]['image_id'],
                'candidate_source': candidates.iloc[row_index]['physical_source_id'],
                'phash_distance': int(distance),
                'holdout_dataset': holdout.iloc[int(other)]['dataset'],
                'holdout_path': holdout.iloc[int(other)]['path'],
            })
    return pd.DataFrame(matches)


def load_master() -> pd.DataFrame:
    """Accepted Stage 5D rows read from the per-source ledgers."""
    frames = []
    for path in sorted((ROOT / 'data/corpus_v3/images').glob('*.csv')):
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame['selected'].astype(str).str.lower().isin(('true', '1'))]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data/datasets/Stage5C')
    args = parser.parse_args()

    cache = ROOT / 'data/corpus_v3/holdout_hash_index.csv'
    holdout = build_hash_index(cache)

    master = load_master()
    old_r3 = pd.read_csv(ROOT / 'data/corpus_v2/R3_field_extended_manifest.csv',
                         low_memory=False)

    # Prefix/identity audit of the mixed AgriField pool.
    old_pool = pd.read_csv(ROOT / 'data/corpus_v2/images/agrifield40k_pool.csv',
                           low_memory=False)
    old_pool['filename'] = old_pool['rel_path'].astype(str).str.split('/').str[-1]
    pool = pd.read_csv(ROOT / 'data/corpus_v3/images/agrifield40k_pool.csv',
                       low_memory=False)
    if 'filename' not in pool.columns:
        pool['filename'] = pool['relative_path'].astype(str).str.split('/').str[-1]
    samples = pd.read_csv(ROOT / 'data/all_samples.csv', low_memory=False)
    identity = dict(zip(samples['sample_id'].astype(str), samples['source'].astype(str)))
    if 'upstream_dataset' in pool.columns:
        pool['upstream'] = pool['upstream_dataset'].fillna('').astype(str)
    else:
        pool['upstream'] = pool['filename'].map(identity).fillna('')
    # ``pnb_`` filenames are the PhenoBench prefix regardless of the ledger's
    # own holdout flag; count them across the whole pool.
    prefix_holdout = pool[pool['filename'].astype(str).str.startswith('pnb_')]
    old_prefix = old_pool[old_pool['filename'].astype(str).str.startswith('pnb_')]

    exact_holdout_hashes = set(holdout['raw_sha256'])
    master_exact = int(master['raw_sha256'].isin(exact_holdout_hashes).sum()) if len(master) else 0
    old_exact = int(old_r3['raw_sha256'].isin(exact_holdout_hashes).sum())
    master_near = near_matches(master, holdout) if len(master) else pd.DataFrame()
    old_near = near_matches(old_r3, holdout)

    # Stage 5C's own holdout gate passed while 1,864 PhenoBench files were in R3.
    old_pnb_inside = int(old_r3['raw_path'].astype(str).str.split('/').str[-1]
                         .str.startswith('pnb_').sum())

    payload = {
        'holdout_hash_index': {
            'datasets': holdout['dataset'].value_counts().to_dict() if len(holdout) else {},
            'rows': int(len(holdout)),
            'cache': str(cache),
        },
        'source_level': {
            'registered_holdout_sources': sorted(set(HOLDOUT_ROOTS)),
            'present_in_master_sources': sorted(
                set(master['physical_source_id']) & set(HOLDOUT_ROOTS)) if len(master) else [],
        },
        'prefix_level': {
            'agrifield_pool_upstream_counts': pool['upstream'].value_counts().to_dict(),
            'phenobench_prefixed_images_in_pool': int(len(prefix_holdout)),
            'phenobench_prefixed_by_split_dir': prefix_holdout['relative_path'].astype(str)
            .str.split('/').str[0].value_counts().to_dict(),
            'phenobench_flagged_by_old_ledger': int(
                (old_prefix['holdout_flag'].astype(str) == 'yes').sum()),
            'phenobench_actually_blocked_by_stage5d': int(
                (prefix_holdout['primary_reason'].astype(str)
                 == 'HOLDOUT_EXCLUDED').sum()),
        },
        'file_level': {
            'old_r3_phenobench_prefixed_images': old_pnb_inside,
            'old_r3_exact_hash_matches': old_exact,
            'old_r3_near_hash_matches': int(len(old_near)),
            'new_master_exact_hash_matches': master_exact,
            'new_master_near_hash_matches': int(len(master_near)),
            'new_master_near_matches_detail': (
                master_near.sort_values('phash_distance').head(50).to_dict('records')
                if len(master_near) else []),
            'near_match_note': (
                'pHash distance 0 is a perceptual-hash collision, not pixel '
                'identity; verify any distance-0 pair before calling it leakage'),
        },
        'not_verifiable_here': [
            'plantdoc', 'minneapple', 'plantseg', 'multiclass_weeds',
            'growliflower', 'cropandweed', 'deepweeds', 'roboflow_*',
        ],
    }
    REPORTS = ROOT / 'reports'
    (REPORTS / 'stage5d_holdout_evidence.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')

    lines = [
        '# Stage 5D - downstream holdout evidence',
        '',
        '## What was checked',
        '',
        f"- Hash index built from {payload['holdout_hash_index']['rows']:,} holdout files "
        f"({payload['holdout_hash_index']['datasets']}).",
        f"- Registered holdout sources present in the new corpus: "
        f"**{payload['source_level']['present_in_master_sources'] or 'none'}**.",
        '',
        '## The Stage 5C leak',
        '',
        f"- `agrifield40k_pool` contains **{len(prefix_holdout):,}** PhenoBench-prefixed",
        f"  images: {payload['prefix_level']['phenobench_prefixed_by_split_dir']}.",
        f"- The Stage 5C ledger flagged only "
        f"{payload['prefix_level']['phenobench_flagged_by_old_ledger']:,} of them as holdout",
        '  (the `train/` prefix), so the `val/` half escaped.',
        f"- Old R3 contained **{old_pnb_inside:,}** PhenoBench-prefixed images.",
        f"- New corpus contains **{master_exact:,}** exact hash matches and "
        f"**{len(master_near):,}** near matches.",
        '',
        '## Limits',
        '',
        'The following holdouts are not on this machine, so only source/identity',
        'level evidence exists for them: '
        + ', '.join(payload['not_verifiable_here']) + '.',
        'No claim of "PROVEN_NO_LEAKAGE" is made for those.',
    ]
    (REPORTS / 'stage5d_holdout_evidence.md').write_text('\n'.join(lines) + '\n',
                                                         encoding='utf-8')
    print(json.dumps(payload, indent=2, default=str)[:3000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
