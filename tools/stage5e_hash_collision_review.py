#!/usr/bin/env python
"""Stage 5E - review every pHash near-neighbour candidate.

The corpus cross-check reports pHash Hamming distance, not identity.  A
distance of 0 is *necessary* but not *sufficient* evidence that two files are
the same image: pHash is a 64-bit perceptual descriptor and collisions happen
between visually plain crops.  This tool therefore re-examines each candidate:

* it re-reads both files and compares dimensions and aspect ratio;
* it computes a normalised pixel difference (mean absolute difference and
  Pearson correlation on a common resize) when both images decode;
* it records a verdict, but it never deletes anything.

Verdict vocabulary
------------------
``IDENTICAL_PIXELS``
    The canonical pixel hashes are equal.  This is the only verdict that would
    justify excluding the corpus image.
``HASH_COLLISION_DIFFERENT_IMAGE``
    Same or near-identical pHash, but the pixels differ materially.
``NEAR_VIEW_SAME_SCENE``
    Different pixels but high correlation - plausible same subject, still not
    a duplicate by the project's own rule.
``DECODE_FAILED``
    One side would not decode; the candidate stays unresolved.

Usage
-----
    .venv-uv/bin/python tools/stage5e_hash_collision_review.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image


def load_rgb(path: Path, size: int = 256) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as image:
            image.load()
            return np.asarray(image.convert('RGB').resize((size, size)), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def compare(holdout_path: Path, corpus_path: Path) -> Dict[str, object]:
    left = load_rgb(holdout_path)
    right = load_rgb(corpus_path)
    if left is None or right is None:
        return {'decode_failed': True,
                'holdout_decoded': left is not None,
                'corpus_decoded': right is not None}
    difference = np.abs(left - right)
    mean_abs = float(difference.mean())
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.std() < 1e-6 or right_flat.std() < 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    return {'mean_absolute_difference': mean_abs,
            'pearson_correlation': correlation,
            'holdout_decoded': True,
            'corpus_decoded': True}


def verdict_for(mean_abs: float, correlation: float, same_pixel_hash: bool) -> str:
    if same_pixel_hash:
        return 'IDENTICAL_PIXELS'
    # A re-encoded or downscaled copy is not bit-identical, but it is the same
    # image content and must be treated as a holdout hit.
    if mean_abs <= 2.0 and correlation >= 0.995:
        return 'RESAMPLED_COPY_SAME_CONTENT'
    if correlation >= 0.85:
        return 'NEAR_VIEW_SAME_SCENE'
    return 'HASH_COLLISION_DIFFERENT_IMAGE'


#: Verdicts that mean "this corpus image is the holdout image" and therefore
#: must be removed from the training manifest.
EXCLUDE_VERDICTS = ('IDENTICAL_PIXELS', 'RESAMPLED_COPY_SAME_CONTENT')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossmatch', default='data/corpus_v4/holdout_crossmatch.csv')
    parser.add_argument('--ledger', default='data/corpus_v3/images')
    parser.add_argument('--index', default='data/corpus_v4/holdout_file_index.csv')
    parser.add_argument('--reports', default='reports')
    args = parser.parse_args()

    cross = pd.read_csv(ROOT / args.crossmatch, dtype=str)
    candidates = cross[cross['conclusion'] == 'CANDIDATE_REQUIRES_REVIEW'].copy()
    if candidates.empty:
        (ROOT / args.reports / 'stage5e_hash_collision_review.json').write_text(
            json.dumps({'candidates': 0, 'rows': []}, indent=2))
        print(json.dumps({'candidates': 0}))
        return 0

    index = pd.read_csv(ROOT / args.index, dtype=str)
    holdout_paths = dict(zip(index['relative_path'], index['dataset_id']))
    holdout_pixel = dict(zip(index['relative_path'],
                             index['pixel_sha256'].fillna('').astype(str)))

    ledger_frames = [pd.read_csv(path, usecols=['image_id', 'raw_path', 'pixel_sha256'],
                                 dtype=str, low_memory=False)
                     for path in sorted((ROOT / args.ledger).glob('*.csv'))]
    ledger = pd.concat(ledger_frames, ignore_index=True)
    corpus_path = dict(zip(ledger['image_id'], ledger['raw_path']))
    corpus_pixel = dict(zip(ledger['image_id'],
                            ledger['pixel_sha256'].fillna('').astype(str)))

    rows: List[Dict[str, object]] = []
    for _, candidate in candidates.iterrows():
        holdout_id = str(candidate['holdout_id'])
        corpus_id = str(candidate['corpus_id'])
        holdout_file = Path(holdout_id)
        corpus_file = Path(str(corpus_path.get(corpus_id, '')))
        same_pixel = (holdout_pixel.get(holdout_id, '') != '' and
                      holdout_pixel.get(holdout_id, '') == corpus_pixel.get(corpus_id, ''))
        if not holdout_file.exists() or not corpus_file.exists() or str(corpus_file) == '':
            row = {'dataset_id': candidate['dataset_id'],
                   'holdout_id': holdout_id,
                   'corpus_id': corpus_id,
                   'corpus_source': candidate['corpus_source'],
                   'phash_distance': int(float(candidate['phash_distance'])),
                   'verdict': 'DECODE_FAILED',
                   'holdout_file_exists': holdout_file.exists(),
                   'corpus_file_exists': corpus_file.exists()}
        else:
            metrics = compare(holdout_file, corpus_file)
            if metrics.get('decode_failed'):
                row = {'dataset_id': candidate['dataset_id'], 'holdout_id': holdout_id,
                       'corpus_id': corpus_id, 'corpus_source': candidate['corpus_source'],
                       'phash_distance': int(float(candidate['phash_distance'])),
                       'verdict': 'DECODE_FAILED', **metrics}
            else:
                row = {'dataset_id': candidate['dataset_id'], 'holdout_id': holdout_id,
                       'corpus_id': corpus_id, 'corpus_source': candidate['corpus_source'],
                       'phash_distance': int(float(candidate['phash_distance'])),
                       'same_pixel_sha256': same_pixel, **metrics,
                       'verdict': verdict_for(metrics['mean_absolute_difference'],
                                              metrics['pearson_correlation'], same_pixel)}
        rows.append(row)

    review = pd.DataFrame(rows)
    report_csv = ROOT / args.reports / 'stage5e_hash_collision_review.csv'
    review.to_csv(report_csv, index=False)

    verdict_counts = review['verdict'].value_counts().to_dict()
    payload = {
        'candidates_reviewed': int(len(review)),
        'verdict_counts': {str(key): int(value) for key, value in verdict_counts.items()},
        'rule': ('pHash distance is a candidate filter only. IDENTICAL_PIXELS requires equal '
                 'canonical pixel hashes, or mean absolute difference <= 2.0 with correlation '
                 '>= 0.995. Nothing was deleted by this review.'),
        'candidates_requiring_exclusion': int(
            review['verdict'].isin(EXCLUDE_VERDICTS).sum()),
        'exclusion_verdicts': list(EXCLUDE_VERDICTS),
        'corpus_image_ids_to_exclude': sorted(
            review.loc[review['verdict'].isin(EXCLUDE_VERDICTS), 'corpus_id']
            .astype(str).tolist()),
        'rows': review.to_dict('records'),
    }
    (ROOT / args.reports / 'stage5e_hash_collision_review.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps({'candidates_reviewed': payload['candidates_reviewed'],
                      'verdict_counts': payload['verdict_counts'],
                      'candidates_requiring_exclusion':
                          payload['candidates_requiring_exclusion']},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
