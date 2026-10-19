#!/usr/bin/env python
"""Stage 2C leakage checks that pixel-hash screening cannot see.

The Stage 1/2B duplicate screener compares pixel content (SHA-256, pHash,
dHash). Two structural leaks sit outside its reach:

  1. A source whose images form one capture SEQUENCE (consecutive timestamps)
     that the split assigned to both train and val. The frames are different
     pixels, so no hash flags them, but they are seconds apart in time and
     depict a near-static scene.
  2. A source whose images share a higher-level grouping (e.g. an iNaturalist
     observation = one plant photographed several times) that straddles the
     split.

Both checks are metadata joins against two manifests and are cheap enough to
re-run on any future manifest. They are READ-ONLY: this tool never modifies a
manifest.

Usage:
    ./.venv/bin/python tools/leakage_check.py \
        --train data/frozen/agrifield40k-v0_train.csv \
        --val data/frozen/agrifield40k-v0_val.csv \
        [--window 60] [--json out.json] [--md out.md]

Exit code is 0 even when leaks are found: this tool reports, it does not gate.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TIMESTAMP_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})[-T](\d{2})[-:](\d{2})[-:](\d{2})')
OBSERVATION_RE = re.compile(r'/observations?/(\d+)')


def fmt_int(n) -> str:
    return f'{n:,}'


def parse_timestamp(text: str) -> Optional[datetime]:
    """Parse the `YYYY-MM-DD-HH-MM-SS` form used by sequence-derived sources."""
    if not text:
        return None
    m = TIMESTAMP_RE.search(text)
    if not m:
        return None
    year, month, day, hour, minute, second = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def parse_observation_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = OBSERVATION_RE.search(text)
    return m.group(1) if m else None


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sequence_check(train: Sequence[Dict[str, str]], val: Sequence[Dict[str, str]],
                   window_seconds: float) -> Dict[str, Dict[str, object]]:
    """Nearest cross-split gap in capture time, per source.

    Only sources whose rows carry a parseable timestamp on BOTH sides are
    reported; a source with no timestamps is absent from this table rather
    than reported as clean.
    """
    train_ts: Dict[str, List[tuple]] = collections.defaultdict(list)
    val_ts: Dict[str, List[tuple]] = collections.defaultdict(list)
    for row in train:
        ts = parse_timestamp(row.get('original_id', ''))
        if ts is not None:
            train_ts[row['source']].append((ts, row.get('original_id', '')))
    for row in val:
        ts = parse_timestamp(row.get('original_id', ''))
        if ts is not None:
            val_ts[row['source']].append((ts, row.get('original_id', '')))

    findings: Dict[str, Dict[str, object]] = {}
    for source in sorted(set(train_ts) & set(val_ts)):
        tr = sorted(train_ts[source], key=lambda p: p[0])
        va = sorted(val_ts[source], key=lambda p: p[0])
        tr_times = [t for t, _ in tr]
        within = 0
        nearest_seconds: Optional[float] = None
        nearest_val_id: Optional[str] = None
        nearest_train_id: Optional[str] = None
        for t, rid in va:
            i = bisect.bisect_left(tr_times, t)
            best: Optional[float] = None
            best_train_id: Optional[str] = None
            for j in (i - 1, i):
                if 0 <= j < len(tr):
                    gap = abs((tr[j][0] - t).total_seconds())
                    if best is None or gap < best:
                        best = gap
                        best_train_id = tr[j][1]
            if best is None:
                continue
            if best <= window_seconds:
                within += 1
            if nearest_seconds is None or best < nearest_seconds:
                nearest_seconds = best
                nearest_val_id = rid
                nearest_train_id = best_train_id
        if nearest_seconds is not None:
            findings[source] = {
                'train_timestamped': len(tr),
                'val_timestamped': len(va),
                'val_within_window': within,
                'window_seconds': window_seconds,
                'nearest_seconds': nearest_seconds,
                'nearest_val_original_id': nearest_val_id,
                'nearest_train_original_id': nearest_train_id,
            }
    return findings


def grouping_check(train: Sequence[Dict[str, str]], val: Sequence[Dict[str, str]]) -> Dict[str, object]:
    """Grouping keys (e.g. iNaturalist observation) appearing in both splits."""
    groups: Dict[str, Dict[str, set]] = collections.defaultdict(
        lambda: {'train': set(), 'val': set()})
    for split_name, rows in (('train', train), ('val', val)):
        for row in rows:
            key = parse_observation_id(row.get('observation_url', ''))
            if key:
                groups[key][split_name].add(row['sample_id'])
    straddling = {k: {s: sorted(v) for s, v in g.items() if v}
                  for k, g in groups.items() if g['train'] and g['val']}
    return {
        'distinct_groups': len(groups),
        'straddling_groups': len(straddling),
        'straddling_samples_train': sum(len(v['train']) for v in straddling.values()),
        'straddling_samples_val': sum(len(v['val']) for v in straddling.values()),
        'examples': dict(list(straddling.items())[:10]),
    }


def render_md(result: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append('# Structural leakage check')
    lines.append('')
    lines.append('Metadata-only checks that pixel hashing cannot perform. This tool '
                 'reports; it does not gate and it never modifies a manifest.')
    lines.append('')
    lines.append(f'- Train manifest: `{result["train"]}` ({fmt_int(result["train_rows"])} rows)')
    lines.append(f'- Val manifest: `{result["val"]}` ({fmt_int(result["val_rows"])} rows)')
    lines.append('')
    lines.append('## 1. Capture-sequence separation')
    lines.append('')
    lines.append('| Source | Train (timestamped) | Val (timestamped) | Val within window | Nearest gap (s) |')
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    for source, info in result['sequence'].items():
        lines.append(f'| {source} | {fmt_int(info["train_timestamped"])} | '
                     f'{fmt_int(info["val_timestamped"])} | '
                     f'{fmt_int(info["val_within_window"])} | '
                     f'{info["nearest_seconds"]:g} |')
    if not result['sequence']:
        lines.append('| _no source carried timestamps on both sides_ | | | | |')
    lines.append('')
    g = result['grouping']
    lines.append('## 2. Grouping-key separation')
    lines.append('')
    lines.append(f'- Distinct grouping keys: **{fmt_int(g["distinct_groups"])}**')
    lines.append(f'- Keys present in BOTH train and val: **{fmt_int(g["straddling_groups"])}** '
                 f'({fmt_int(g["straddling_samples_train"])} train / '
                 f'{fmt_int(g["straddling_samples_val"])} val samples)')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='data/frozen/agrifield40k-v0_train.csv')
    parser.add_argument('--val', default='data/frozen/agrifield40k-v0_val.csv')
    parser.add_argument('--window', type=float, default=60.0,
                        help='seconds; how close counts as the same capture window')
    parser.add_argument('--json', dest='json_out', default=None)
    parser.add_argument('--md', dest='md_out', default=None)
    args = parser.parse_args()

    train = read_manifest(Path(args.train))
    val = read_manifest(Path(args.val))
    result = {
        'train': str(args.train),
        'val': str(args.val),
        'train_rows': len(train),
        'val_rows': len(val),
        'window_seconds': args.window,
        'sequence': sequence_check(train, val, args.window),
        'grouping': grouping_check(train, val),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.md_out:
        Path(args.md_out).write_text(render_md(result), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
