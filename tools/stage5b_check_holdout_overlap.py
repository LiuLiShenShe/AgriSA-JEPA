#!/usr/bin/env python
"""Stage 5B Gate 5: screen a new source against holdout and existing corpora.

Two comparisons run:

1. exact hashes (``sha256`` and ``pixel_sha256``) against every holdout corpus
   and the frozen v1 manifests;
2. perceptual hashes (pHash/dHash) as *candidates* for human review.

A candidate never blocks a freeze by itself; an exact match always does.

    .venv-uv/bin/python tools/stage5b_check_holdout_overlap.py \
        --manifest data/registry/ingested/sesame_weed_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.registry.ingest import _bits


def load_csv(path: Path):
    if not path.is_file():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a[:, None] ^ b[None, :])


def compare(new_rows, reference_rows, phash_threshold=6):
    """Return exact matches and perceptual candidates vs a reference set."""
    new_by_sha = {}
    new_by_pixel = {}
    for row in new_rows:
        sha = row.get('sha256', '')
        pixel = row.get('pixel_sha256', '')
        if sha:
            new_by_sha.setdefault(sha, row)
        if pixel:
            new_by_pixel.setdefault(pixel, row)

    exact = []
    for ref in reference_rows:
        hit = new_by_sha.get(ref.get('sha256', '')) or new_by_pixel.get(
            ref.get('pixel_sha256', ''))
        if hit is not None:
            exact.append({
                'new_sample_id': hit.get('sample_id'),
                'reference_sample_id': ref.get('sample_id'),
                'reference_source': ref.get('source') or ref.get('source_id'),
                'reference_split': ref.get('split', ''),
                'kind': 'exact_sha256_or_pixel_sha256',
            })

    ref_ph = [(r.get('phash64'), r.get('sample_id'), r.get('source'))
              for r in reference_rows if r.get('phash64')]
    ref_dh = [(r.get('dhash64'), r.get('sample_id'), r.get('source'))
              for r in reference_rows if r.get('dhash64')]
    candidates = []
    new_decoded = [r for r in new_rows if r.get('phash64')]
    if new_decoded and ref_ph:
        new_hashes = _bits([r['phash64'] for r in new_decoded])
        ref_hashes = _bits([h for h, _, _ in ref_ph])
        ref_ids = [i for _, i, _ in ref_ph]
        ref_src = [s for _, _, s in ref_ph]
        for start in range(0, len(new_hashes), 128):
            block = new_hashes[start:start + 128]
            dist = hamming(block, ref_hashes)
            hits = np.argwhere(dist <= phash_threshold)
            for r, c in hits:
                candidates.append({
                    'new_sample_id': new_decoded[start + int(r)]['sample_id'],
                    'reference_sample_id': ref_ids[int(c)],
                    'reference_source': ref_src[int(c)],
                    'kind': 'phash_candidate',
                    'distance': int(dist[r, c]),
                    'review_status': 'pending_manual_review',
                })
                if len(candidates) >= 200000:
                    break
            if len(candidates) >= 200000:
                break
    return exact, candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--threshold', type=int, default=6)
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    new_rows = load_csv(manifest_path)
    label = args.label or manifest_path.stem

    references = {
        'frozen_v1_train': (ROOT / 'data/frozen/agrifield40k-v1_train.csv'),
        'frozen_v1_val': (ROOT / 'data/frozen/agrifield40k-v1_val.csv'),
        'frozen_d1_train': (ROOT / 'data/frozen/agri-corpus-d1-candidate_train.csv'),
        'frozen_d1_val': (ROOT / 'data/frozen/agri-corpus-d1-candidate_val.csv'),
        'all_samples': (ROOT / 'data/all_samples.csv'),
    }

    exact_all = []
    candidates_all = []
    per_reference = {}
    for name, path in references.items():
        ref = load_csv(path)
        if not ref:
            per_reference[name] = {'rows': 0, 'exact': 0, 'phash_candidates': 0}
            continue
        exact, cand = compare(new_rows, ref, args.threshold)
        per_reference[name] = {
            'rows': len(ref),
            'exact': len(exact),
            'phash_candidates': len(cand),
        }
        for item in exact:
            item['reference_corpus'] = name
        for item in cand:
            item['reference_corpus'] = name
        exact_all.extend(exact)
        candidates_all.extend(cand)

    result = {
        'source_manifest': str(manifest_path),
        'label': label,
        'images_screened': len(new_rows),
        'threshold': args.threshold,
        'per_reference': per_reference,
        'exact_matches': exact_all,
        'exact_match_count': len(exact_all),
        'perceptual_candidates': candidates_all[:5000],
        'perceptual_candidate_count': len(candidates_all),
        'note': ('Perceptual candidates are screening evidence for human review, '
                 'not a verdict; exact hash matches are blocking.'),
    }
    out = Path(args.out) if args.out else (
        ROOT / f'reports/stage5b_holdout_overlap_{label}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('exact_matches', 'perceptual_candidates')},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
