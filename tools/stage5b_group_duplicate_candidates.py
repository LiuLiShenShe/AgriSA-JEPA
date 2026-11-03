#!/usr/bin/env python
"""Stage 5B Gate 5: group perceptual duplicate candidates into components.

Perceptual hashes produce pairs, and a pair is not a verdict.  This tool:

* builds a graph over the candidate edges (pHash and dHash, separately and
  combined);
* reports connected components, so a 3-way augmentation chain becomes one
  cluster instead of three unrelated pairs;
* attaches the source's own capture-group key, so "same photo, augmented" is
  separated from "same tree, two views";
* assigns a risk tier: exact duplicate > same-group perceptual > cross-group
  perceptual > cross-source.

Nothing is deleted.  A cluster is a review unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
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


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def perceptual_pairs(rows, threshold_phash=4, threshold_dhash=4,
                     cluster_kind='phash'):
    """Return (clustering pairs, supporting pairs).

    Only ``cluster_kind`` edges are used to build components.  dHash is a weaker
    descriptor on repetitive top-down imagery and chains badly, so by default it
    supports a review decision but does not merge clusters.
    """
    pairs = []
    decoded = [r for r in rows if r.get('phash64') and r.get('dhash64')]
    if len(decoded) < 2:
        return pairs
    ph = _bits([r['phash64'] for r in decoded])
    dh = _bits([r['dhash64'] for r in decoded])
    ids = [r['sample_id'] for r in decoded]
    for kind, hashes, threshold in (('phash', ph, threshold_phash),
                                    ('dhash', dh, threshold_dhash)):
        if kind != cluster_kind:
            # still record dhash pairs as support, but never cluster on them
            pass
        for start in range(0, len(hashes), 256):
            block = hashes[start:start + 256]
            dist = np.bitwise_count(block[:, None] ^ hashes[None, :])
            # only the triangle to the right of the diagonal is new
            hits = np.argwhere(np.triu(dist, k=start + 1) <= threshold)
            for r, c in hits:
                i, j = start + int(r), int(c)
                if i >= j:
                    continue
                pairs.append({
                    'a': ids[i], 'b': ids[j], 'kind': kind,
                    'distance': int(dist[r, c]),
                    'threshold': threshold,
                    'clustering': kind == cluster_kind,
                })
    return pairs


def build_clusters(rows, threshold_phash=4, threshold_dhash=4):
    by_id = {r['sample_id']: r for r in rows}
    pairs = perceptual_pairs(rows, threshold_phash, threshold_dhash)
    uf = UnionFind()
    for row in rows:
        uf.find(row['sample_id'])
    for pair in pairs:
        if pair['clustering']:
            uf.union(pair['a'], pair['b'])
    components = defaultdict(list)
    for sample_id in by_id:
        components[uf.find(sample_id)].append(sample_id)
    return by_id, pairs, components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', action='append', required=True)
    parser.add_argument('--out', default='data/registry/field_duplicate_candidates.csv')
    parser.add_argument('--json', default='reports/stage5b_duplicate_audit.json')
    parser.add_argument('--threshold-phash', type=int, default=4)
    parser.add_argument('--threshold-dhash', type=int, default=4)
    args = parser.parse_args()

    rows = []
    for manifest in args.manifest:
        rows.extend(load_csv(Path(manifest)))

    by_id, pairs, components = build_clusters(rows, args.threshold_phash,
                                              args.threshold_dhash)
    members = {}
    for root, ids in components.items():
        for sid in ids:
            members[sid] = root

    out_rows = []
    cluster_sizes = Counter()
    for sid, row in by_id.items():
        root = members[sid]
        cluster = components[root]
        cluster_sizes[len(cluster)] += 1
        same_group = [s for s in cluster if s != sid and
                      by_id[s].get('group_key') == row.get('group_key')]
        cross_group = len(cluster) - 1 - len(same_group)
        if len(cluster) > 50:
            risk = 'chaining_suspected'
            action = 'manual_review_before_use'
        elif row.get('duplicate_status', '').startswith('exact_'):
            risk = 'exact_duplicate'
            action = 'drop_duplicate'
        elif same_group:
            risk = 'same_group_perceptual'
            action = 'keep_one_per_group'
        elif cross_group:
            risk = 'cross_group_perceptual'
            action = 'manual_review'
        else:
            risk = 'singleton'
            action = 'keep'
        out_rows.append({
            'sample_id': sid,
            'source_id': row.get('source_id', ''),
            'rel_path': row.get('rel_path', ''),
            'group_key': row.get('group_key', ''),
            'cluster_id': f'c{abs(hash(root)) % 10**10:010d}',
            'cluster_size': len(cluster),
            'same_group_neighbours': len(same_group),
            'cross_group_neighbours': cross_group,
            'risk_tier': risk,
            'recommended_action': action,
            'review_status': 'pending_manual_review' if risk != 'singleton' else 'not_needed',
            'duplicate_status': row.get('duplicate_status', ''),
        })

    out_rows.sort(key=lambda r: (r['cluster_size'], r['sample_id']), reverse=True)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    multi = [c for c in components.values() if len(c) > 1]
    audit = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifests': args.manifest,
        'images': len(rows),
        'perceptual_pairs': len(pairs),
        'pairs_by_kind': dict(Counter(p['kind'] for p in pairs)),
        'clusters_with_more_than_one_image': len(multi),
        'largest_cluster': max((len(c) for c in components.values()), default=1),
        'cluster_size_histogram': {str(k): v for k, v in
                                   sorted(Counter(len(c) for c in multi).items())},
        'risk_tiers': dict(Counter(r['risk_tier'] for r in out_rows)),
        'note': ('A cluster is a review unit, not a duplication verdict. The '
                 'recommended action is what a reviewer should do next.'),
    }
    json_path = ROOT / args.json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
