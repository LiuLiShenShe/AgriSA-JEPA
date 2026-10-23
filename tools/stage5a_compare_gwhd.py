#!/usr/bin/env python
"""Compare GWHD A/B runs and refuse non-identical protocols.

Usage
-----
    .venv-uv/bin/python tools/stage5a_compare_gwhd.py \
        --a runs/stage5a_gwhd/A_seed0/results.json \
        --b runs/stage5a_gwhd/B_seed0/results.json \
        --out-prefix reports/stage5a_gwhd

For several seeds pass ``--seed-results seed=pathA,pathB`` repeatedly; the tool
reports per-seed and mean ± std.  Only the backbone may differ.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

FIELDS = ['input_size', 'iterations', 'batch_size', 'lr', 'momentum',
          'weight_decay', 'warmup_iterations', 'eval_every', 'min_box_size',
          'hflip_prob', 'score_thresh', 'seed', 'architecture', 'anchors',
          'optimizer', 'scheduler']

METRICS = ['map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large',
           'mar_100', 'precision@0.5', 'recall@0.5', 'ms_per_image']


def load(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol_diff(a: Dict, b: Dict) -> List[str]:
    pa, pb = a['protocol'], b['protocol']
    diffs = [f for f in FIELDS if pa.get(f) != pb.get(f)]
    if pa.get('backbone_sha256') == pb.get('backbone_sha256'):
        diffs.append('backbone_sha256 (identical weights)')
    return diffs


def row(label: str, metrics: Dict) -> Dict[str, object]:
    out = {'run': label}
    for m in METRICS:
        value = metrics.get(m)
        if isinstance(value, (int, float)):
            out[m] = round(float(value), 4)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    import csv
    keys = ['run'] + METRICS
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--seed-results', action='append', default=[],
                        help='seed=N:pathA,pathB')
    parser.add_argument('--out-prefix', default='reports/stage5a_gwhd')
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    diffs = protocol_diff(a, b)
    if diffs:
        raise SystemExit(f'REFUSING to compare: protocols differ in {diffs}')

    seeds = [(0, args.a, args.b)]
    for item in args.seed_results:
        head, _, rest = item.partition(':')
        pa, _, pb = rest.partition(',')
        seeds.append((int(head.split('=')[-1]), pa, pb))

    rows = [row('A_val', a['final_val']), row('B_val', b['final_val'])]
    if a.get('final_test') and b.get('final_test'):
        rows += [row('A_test', a['final_test']), row('B_test', b['final_test'])]

    deltas = {}
    for split in ('final_val', 'final_test'):
        if not (a.get(split) and b.get(split)):
            continue
        deltas[split] = {m: round(float(b[split].get(m, 0)) - float(a[split].get(m, 0)), 4)
                         for m in METRICS if isinstance(a[split].get(m), (int, float))}

    per_seed = []
    for seed, pa, pb in seeds:
        ra, rb = load(pa), load(pb)
        d = protocol_diff(ra, rb)
        if d:
            raise SystemExit(f'seed {seed}: protocols differ in {d}')
        per_seed.append({'seed': seed,
                         'A': ra['final_test'], 'B': rb['final_test']})

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(str(prefix) + '_comparison.csv'), rows)
    payload = {
        'protocol_identical_except_backbone': True,
        'protocol_fields': {f: a['protocol'].get(f) for f in FIELDS},
        'backbones': {'A': a['protocol']['backbone_sha256'],
                      'B': b['protocol']['backbone_sha256']},
        'runs': rows,
        'deltas_B_minus_A': deltas,
        'per_seed_test': per_seed,
    }
    Path(str(prefix) + '_comparison.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# Stage 5A — GWHD 2021 检测 A/B 结果',
        '',
        '> 唯一差异为 Backbone 权重；输入、划分、Head、预算、种子、评价完全一致。',
        '> 指标为 COCO AP（IoU 0.50:0.95），单类别（麦穗）。',
        '',
        '## 1. 最终结果',
        '',
        '| run | AP50:95 | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | P@0.5 | R@0.5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append('| ' + ' | '.join(str(r.get(m, '')) for m in
                     ['run', 'map', 'map_50', 'map_75', 'map_small',
                      'map_medium', 'map_large', 'mar_100', 'precision@0.5',
                      'recall@0.5']) + ' |')
    lines += ['', '## 2. Δ (B − A)', '']
    for split, d in deltas.items():
        lines.append(f"- {split}: " + ', '.join(f'{k} {v:+.3f}' for k, v in d.items()))
    lines += ['', '## 3. 多 seed', '']
    if len(per_seed) > 1:
        for metric in ('map', 'map_50'):
            a_vals = [s['A'].get(metric, 0) for s in per_seed]
            b_vals = [s['B'].get(metric, 0) for s in per_seed]
            if len(a_vals) > 1:
                lines.append(f"- {metric}: A {statistics.mean(a_vals):.4f} ± "
                             f"{statistics.pstdev(a_vals):.4f}; B "
                             f"{statistics.mean(b_vals):.4f} ± "
                             f"{statistics.pstdev(b_vals):.4f}; "
                             f"Δ {statistics.mean(b_vals)-statistics.mean(a_vals):+.4f}")
    else:
        lines.append('- 仅 1 个训练 seed，差值属于单次运行，不能当作显著性结论。')
    lines += ['', '## 4. 解释边界', '',
              '- 这里的绝对 mAP 不能与 Agri-FM+ 的 Detectron2 数字直接比较（协议不同）。',
              '- 冻结 ViT-H + Faster R-CNN Head 的预算固定；不做测试集调参。',
              '- 若 B ≤ A，应优先归因于任务/分辨率/冻结特征，而不是修改协议找正数。', '']
    Path(str(prefix) + '_results.md').write_text('\n'.join(lines) + '\n',
                                                 encoding='utf-8')
    print(json.dumps({'deltas': deltas, 'rows': rows}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
