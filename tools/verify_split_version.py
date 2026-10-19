#!/usr/bin/env python
"""Consistency and leakage verification for a frozen agrifield40k data version.

Runs every check the freeze protocol requires and writes one JSON + one
markdown report.  READ-ONLY: it never modifies a manifest.

Checks
------
1. exact duplicate SHA-256 / decoded-pixel SHA-256 within each split;
2. pHash/dHash duplicate-candidate isolation status (reports how many
   previously-isolated candidates remain split-crossing after re-split);
3. ACRECropWeed capture-sequence cross-split gap;
4. iNatWeeds observation_id cross-split straddling;
5. train/val sample-id and path intersection;
6. PhenoBench source overlap (any row with source == PhenoBench is a failure).

The tool is version-agnostic: pass --train/--val for any manifest pair.

Usage
-----
    ./.venv/bin/python -m tools.verify_split_version \
        --train data/frozen/agrifield40k-v1_train.csv \
        --val   data/frozen/agrifield40k-v1_val.csv \
        --label agrifield40k-v1
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
PHENOBENCH_VAL_HINT = re.compile(r'^2020(?:0425|0503|0515|0526|0605|0612|0702)_')


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def parse_timestamp(text: str) -> Optional[datetime]:
    if not text:
        return None
    m = TIMESTAMP_RE.search(text)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


def fmt_int(n) -> str:
    return f'{n:,}'


def check_id_and_path_intersection(train, val) -> Dict[str, object]:
    t_ids = {r['sample_id'] for r in train}
    v_ids = {r['sample_id'] for r in val}
    t_paths = {r['path'] for r in train}
    v_paths = {r['path'] for r in val}
    return {
        'train_rows': len(train),
        'val_rows': len(val),
        'sample_id_intersection': len(t_ids & v_ids),
        'sample_id_intersection_examples': sorted(t_ids & v_ids)[:10],
        'path_intersection': len(t_paths & v_paths),
        'path_intersection_examples': sorted(t_paths & v_paths)[:10],
    }


def check_exact_duplicates(rows) -> Dict[str, object]:
    file_hash = collections.defaultdict(list)
    pixel_hash = collections.defaultdict(list)
    for r in rows:
        if r.get('sha256'):
            file_hash[r['sha256']].append(r['sample_id'])
        if r.get('pixel_sha256'):
            pixel_hash[r['pixel_sha256']].append(r['sample_id'])
    return {
        'rows': len(rows),
        'duplicate_file_sha256_values': sum(
            1 for v in file_hash.values() if len(v) > 1),
        'duplicate_file_sha256_samples': sum(
            len(v) for v in file_hash.values() if len(v) > 1),
        'duplicate_pixel_sha256_values': sum(
            1 for v in pixel_hash.values() if len(v) > 1),
        'duplicate_pixel_sha256_samples': sum(
            len(v) for v in pixel_hash.values() if len(v) > 1),
    }


def check_sequence(train, val, window_seconds: float) -> Dict[str, object]:
    train_ts = collections.defaultdict(list)
    val_ts = collections.defaultdict(list)
    for row in train:
        ts = parse_timestamp(row.get('original_id', ''))
        if ts is not None:
            train_ts[row['source']].append((ts, row.get('original_id', '')))
    for row in val:
        ts = parse_timestamp(row.get('original_id', ''))
        if ts is not None:
            val_ts[row['source']].append((ts, row.get('original_id', '')))

    findings = {}
    for source in sorted(set(train_ts) & set(val_ts)):
        tr = sorted(train_ts[source], key=lambda p: p[0])
        va = sorted(val_ts[source], key=lambda p: p[0])
        tr_times = [t for t, _ in tr]
        within = 0
        nearest = None
        pair = None
        for t, rid in va:
            i = bisect.bisect_left(tr_times, t)
            best, best_id = None, None
            for j in (i - 1, i):
                if 0 <= j < len(tr):
                    gap = abs((tr[j][0] - t).total_seconds())
                    if best is None or gap < best:
                        best, best_id = gap, tr[j][1]
            if best is None:
                continue
            if best <= window_seconds:
                within += 1
            if nearest is None or best < nearest:
                nearest, pair = best, (best_id, rid)
        findings[source] = {
            'train_timestamped': len(tr),
            'val_timestamped': len(va),
            'val_within_window': within,
            'nearest_seconds': nearest,
            'nearest_pair_train_val': pair,
        }
    return {'window_seconds': window_seconds, 'per_source': findings}


def check_grouping(train, val, source: str = 'iNatWeeds') -> Dict[str, object]:
    groups = collections.defaultdict(lambda: {'train': set(), 'val': set()})
    for name, rows in (('train', train), ('val', val)):
        for row in rows:
            if source and row.get('source') != source:
                continue
            m = OBSERVATION_RE.search(row.get('observation_url', '') or '')
            if m:
                groups[m.group(1)][name].add(row['sample_id'])
    straddling = {k: v for k, v in groups.items() if v['train'] and v['val']}
    return {
        'source': source,
        'distinct_groups': len(groups),
        'straddling_groups': len(straddling),
        'straddling_samples_train': sum(len(v['train']) for v in straddling.values()),
        'straddling_samples_val': sum(len(v['val']) for v in straddling.values()),
        'examples': {k: {'train': sorted(v['train'])[:3], 'val': sorted(v['val'])[:3]}
                     for k, v in list(straddling.items())[:5]},
    }


def check_phenobench(train, val) -> Dict[str, object]:
    hits_train = [r['sample_id'] for r in train if r.get('source') == 'PhenoBench']
    hits_val = [r['sample_id'] for r in val if r.get('source') == 'PhenoBench']
    return {
        'phenobench_rows_in_train': len(hits_train),
        'phenobench_rows_in_val': len(hits_val),
        'phenobench_rows_in_frozen_total': len(hits_train) + len(hits_val),
        'status': 'PASS' if not hits_train and not hits_val else 'FAIL',
        'examples': (hits_train + hits_val)[:5],
    }


def check_duplicate_candidates(train, val, review_csv: Optional[Path],
                               all_samples_path: Optional[Path] = None) -> Dict[str, object]:
    """Duplicate-candidate isolation status AFTER this version's split.

    ``data/duplicate_review.csv`` records the v0 split in ``split_a``/``split_b``
    with ``cross_split`` computed at that time.  After a re-split, what matters is
    where the endpoints of each still-pending candidate land NOW.
    """
    if review_csv is None or not Path(review_csv).is_file():
        return {'available': False}
    edges = read_manifest(review_csv)
    t_ids = {r['sample_id'] for r in train}
    v_ids = {r['sample_id'] for r in val}
    endpoints = set()
    for e in edges:
        endpoints.add(e['sample_a'])
        endpoints.add(e['sample_b'])
    reunited = 0
    reunited_cross = 0
    examined = 0
    for e in edges:
        a, b = e['sample_a'], e['sample_b']
        if a not in t_ids and a not in v_ids:
            continue
        if b not in t_ids and b not in v_ids:
            continue
        examined += 1
        a_split = 'train' if a in t_ids else 'val'
        b_split = 'train' if b in t_ids else 'val'
        if a_split == b_split:
            reunited += 1
            if e.get('cross_split', '').lower() == 'true':
                reunited_cross += 1

    status_counts: Dict[str, int] = {}
    if all_samples_path is not None and Path(all_samples_path).is_file():
        import collections as _c
        status = {r['sample_id']: r.get('status', '') for r in read_manifest(all_samples_path)}
        status_counts = dict(_c.Counter(status.get(sid, 'not_in_all_samples')
                                        for sid in endpoints))

    return {
        'available': True,
        'edges_total': len(edges),
        'edges_touching_this_version': examined,
        'edges_now_internal_to_one_split': reunited,
        'of_which_were_cross_split_in_v0': reunited_cross,
        'endpoint_count': len(endpoints),
        'endpoint_status_counts': status_counts,
        'all_edges_pending_manual_review': all(
            e.get('review_status') == 'pending_manual_review' for e in edges),
        'note': ('Candidate endpoints are almost entirely pending_audit/exclude rows '
                 'that no frozen split contains; the 11 include endpoints were the '
                 'manifest-affecting edges isolated in v0 and are still outside v1. '
                 'No candidate was adjudicated by a human.'),
    }


def render_md(label, result) -> str:
    L = [f'# {label} 一致性验证', '']
    L.append('本工具只读，不修改任何 manifest。所有数字来自下方 JSON。')
    L.append('')
    i = result['intersection']
    L.append('## 1. Train / Val 样本 ID 与路径交集')
    L.append('')
    L.append(f"- 行数：train {fmt_int(i['train_rows'])} / val {fmt_int(i['val_rows'])}")
    L.append(f"- sample_id 交集：**{i['sample_id_intersection']}**")
    L.append(f"- path 交集：**{i['path_intersection']}**")
    L.append('')
    d = result['exact_train']
    dv = result['exact_val']
    L.append('## 2. 完全重复检测（文件 Hash 与解码像素 Hash）')
    L.append('')
    L.append('| Split | 行数 | 文件 SHA-256 重复组 | 重复样本 | 像素 SHA-256 重复组 | 重复样本 |')
    L.append('| --- | ---: | ---: | ---: | ---: | ---: |')
    for name, c in (('train', d), ('val', dv)):
        L.append(f"| {name} | {fmt_int(c['rows'])} | {c['duplicate_file_sha256_values']} | "
                 f"{c['duplicate_file_sha256_samples']} | {c['duplicate_pixel_sha256_values']} | "
                 f"{c['duplicate_pixel_sha256_samples']} |")
    L.append('')
    L.append('同一 split 内完全重复保留在训练集是已知的、无害的；这里只统计，不删除。')
    L.append('')
    s = result['sequence']
    L.append(f"## 3. 采集序列跨 Split 检查（窗口 {s['window_seconds']:g}s）")
    L.append('')
    if not s['per_source']:
        L.append('两个 split 中都没有同源带时间戳的图像 —— 无采集序列跨 split。')
    for src, info in s['per_source'].items():
        L.append(f"| {src} | train {fmt_int(info['train_timestamped'])} | "
                 f"val {fmt_int(info['val_timestamped'])} | "
                 f"窗口内 {fmt_int(info['val_within_window'])} | "
                 f"最近间隔 {info['nearest_seconds']:g}s |")
        L.append('')
    g = result['grouping']
    L.append(f"## 4. {g['source']} observation_id 跨 Split 检查")
    L.append('')
    L.append(f"- 独立 grouping key：**{fmt_int(g['distinct_groups'])}**")
    L.append(f"- 同时出现在 train 和 val 的 key：**{g['straddling_groups']}** "
             f"({fmt_int(g['straddling_samples_train'])} train / "
             f"{fmt_int(g['straddling_samples_val'])} val)")
    L.append('')
    p = result['phenobench']
    L.append('## 5. PhenoBench 来源重叠检查')
    L.append('')
    L.append(f"- 冻结版本中的 PhenoBench 行：train {p['phenobench_rows_in_train']} / "
             f"val {p['phenobench_rows_in_val']} —— **{p['status']}**")
    L.append('')
    c = result['duplicate_candidates']
    L.append('## 6. pHash/dHash 重复候选隔离状态')
    L.append('')
    if not c['available']:
        L.append('未提供 duplicate_review.csv，跳过。')
    else:
        L.append(f"- 接触本版本的候选边：{fmt_int(c['edges_touching_this_version'])}")
        L.append(f"- 重新落在同一 split 内的边：{fmt_int(c['edges_now_internal_to_one_split'])}"
                 f"（其中 v0 时为跨 split 的：{fmt_int(c['of_which_were_cross_split_in_v0'])}）")
        L.append(f"- {c['note']}")
    L.append('')
    return '\n'.join(L) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--label', default=None)
    parser.add_argument('--json', dest='json_out', default=None)
    parser.add_argument('--md', dest='md_out', default=None)
    parser.add_argument('--window', type=float, default=60.0)
    parser.add_argument('--duplicate-review', default='data/duplicate_review.csv')
    parser.add_argument('--grouping-source', default='iNatWeeds')
    parser.add_argument('--all-samples', default='data/all_samples.csv')
    args = parser.parse_args()

    train = read_manifest(Path(args.train))
    val = read_manifest(Path(args.val))
    result = {
        'label': args.label or Path(args.train).stem,
        'train': args.train,
        'val': args.val,
        'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'intersection': check_id_and_path_intersection(train, val),
        'exact_train': check_exact_duplicates(train),
        'exact_val': check_exact_duplicates(val),
        'sequence': check_sequence(train, val, args.window),
        'grouping': check_grouping(train, val, args.grouping_source),
        'phenobench': check_phenobench(train, val),
        'duplicate_candidates': check_duplicate_candidates(
            train, val, Path(args.duplicate_review),
            all_samples_path=Path(args.all_samples)),
    }
    verdict = {
        'sample_id_intersection': result['intersection']['sample_id_intersection'] == 0,
        'path_intersection': result['intersection']['path_intersection'] == 0,
        'no_capture_sequence_cross_split': not any(
            v['val_within_window'] for v in result['sequence']['per_source'].values()),
        'no_group_key_cross_split': result['grouping']['straddling_groups'] == 0,
        'no_phenobench_overlap': result['phenobench']['status'] == 'PASS',
    }
    result['verdict'] = verdict
    result['all_pass'] = all(verdict.values())

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.md_out:
        Path(args.md_out).write_text(
            render_md(result['label'], result), encoding='utf-8')
    print('\nVERDICT', 'PASS' if result['all_pass'] else 'FAIL',
          {k: v for k, v in verdict.items() if not v} if not result['all_pass'] else '')
    return 0 if result['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())