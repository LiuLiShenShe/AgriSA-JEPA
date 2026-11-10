#!/usr/bin/env python
"""Stage 7B: paired schedule-ablation comparison and pre-registered decision.

Reads the six PhenoBench probe results (A/B/C x P1/P2), aggregates per probe
seed, and reports the paired contrasts **A vs B** (horizon at equal warmup) and
**B vs C** (warmup at equal horizon).  Direction consistency across the two
pretraining seeds is the primary evidence; a single mIoU number never hides a
class-level conflict.

Pre-registered interpretation (Stage 7B section 6):

* both seeds agree AND |mean paired delta| >= 0.2 pp (mIoU) -> a clear schedule
  effect in that direction;
* |delta| < 0.2 pp OR the two seeds disagree -> INCONCLUSIVE;
* class metrics conflict (crop/weed move >= 0.5 pp opposite to mIoU) -> reported
  separately as a conflict, not folded into mIoU.

Outputs:
* ``reports/stage7b_phenobench_per_checkpoint.json``
* ``reports/stage7b_schedule_comparison.csv``
* ``reports/stage7b_schedule_comparison.md``
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from tools.stage7a_build_configs import ROOT

R = ROOT / 'reports'
RUN_ROOT = Path('/data/agrisa_jepa_runs/stage7b')
PHENOBENCH = RUN_ROOT / 'phenobench'
MANIFEST = R / 'stage7b_phenobench_manifest.json'
METRICS = ('soil', 'crop', 'weed', 'mIoU')
ARMS = ('A', 'B', 'C')
SEEDS = {'p1': 20261104, 'p2': 20261105}
MIOU_THRESHOLD_PP = 0.2
CLASS_THRESHOLD_PP = 0.5


def load_result(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def checkpoint_metrics(result: Mapping[str, object]) -> Dict[str, object]:
    seeds = result.get('seeds', [])
    if [s.get('seed') for s in seeds] != [0, 1, 2]:
        raise ValueError(f'{result.get("arm_label")}: unexpected probe seeds')
    official = {'crop': 'IoU (crop)', 'weed': 'IoU (weed)', 'mIoU': 'mIoU'}
    raw: Dict[str, List[float]] = {}
    for metric, key in official.items():
        raw[metric] = [float(s['official_metrics'][key]) for s in seeds]
    raw['soil'] = [float(s['self_metrics']['per_class_iou_percent']['soil'])
                   for s in seeds]
    out = {'per_probe_seed': raw}
    for metric in METRICS:
        out[metric] = float(np.mean(raw[metric]))
        out[f'{metric}_std'] = float(np.std(raw[metric]))
    return out


def read_all(only_manifest: Optional[Path] = None) -> Dict[str, Dict[str, object]]:
    manifest = json.loads((only_manifest or MANIFEST).read_text(encoding='utf-8'))
    out = {}
    for entry in manifest['entries']:
        key = entry['run_id']
        result = load_result(ROOT / entry['result'] if not Path(entry['result']).is_absolute()
                             else Path(entry['result']))
        metrics = checkpoint_metrics(result)
        metrics.update({
            'run_id': key, 'arm': entry['arm'], 'tag': entry['pair_id'].lower(),
            'pretraining_seed': entry['seed'],
            'checkpoint_sha256': entry['checkpoint_sha256'],
            'result': entry['result'],
        })
        out[key] = metrics
    return out


def pair_contrast(arm_x: str, arm_y: str, data: Mapping[str, Mapping[str, object]],
                  label: str) -> Dict[str, object]:
    """Paired contrast between two arms, delta = arm_x - arm_y per seed.

    Both arms are taken from the *same* pretraining seed (paired design); a
    previous version pinned the first seed and compared it twice, collapsing the
    two seed deltas into one.
    """
    rows = []
    for tag in SEEDS:
        kx, ky = f'{arm_x}_{tag}', f'{arm_y}_{tag}'
        row = {'tag': tag}
        for metric in METRICS:
            row[metric] = round(data[kx][metric] - data[ky][metric], 4)
        rows.append(row)
    deltas = {metric: [row[metric] for row in rows] for metric in METRICS}
    mean = {metric: round(float(np.mean(deltas[metric])), 4) for metric in METRICS}
    directions = {metric: [float(np.sign(v)) for v in deltas[metric]]
                  for metric in METRICS}
    consistent = {metric: len(set(directions[metric])) == 1 for metric in METRICS}
    detail = {
        'contrast': f'{arm_x}_vs_{arm_y}', 'delta_of': f'{arm_x} - {arm_y}',
        'per_seed_delta': rows,
        'mean_delta': mean,
        'seed_direction_consistent': consistent,
    }
    # decision on mIoU
    mean_miou = mean['mIoU']
    miou_consistent = consistent['mIoU']
    class_conflict = []
    for metric in ('crop', 'weed'):
        if (abs(mean[metric]) >= CLASS_THRESHOLD_PP
                and np.sign(mean[metric]) != np.sign(mean_miou)
                and mean_miou != 0):
            class_conflict.append(metric)
    if not miou_consistent:
        decision = 'INCONCLUSIVE'
        reason = 'two pretraining seeds move in opposite directions on mIoU'
    elif abs(mean_miou) < MIOU_THRESHOLD_PP:
        decision = 'INCONCLUSIVE'
        reason = f'|mean mIoU delta| {abs(mean_miou):.3f} pp < {MIOU_THRESHOLD_PP} pp'
    elif class_conflict:
        decision = 'CLASS_CONFLICT'
        reason = ('mIoU favors one side but ' + ', '.join(class_conflict)
                  + ' move(s) the other way; explain separately')
    else:
        decision = 'CLEAR_EFFECT'
        winner = arm_x if mean_miou > 0 else arm_y
        reason = (f'both seeds agree and mean mIoU delta = {mean_miou:+.3f} pp '
                  f'(>= {MIOU_THRESHOLD_PP} pp, {arm_x}-{arm_y}); {winner} is higher')
    detail.update({'decision': decision, 'reason': reason,
                   'class_conflict': class_conflict})
    return detail


def build_payload(data: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    checkpoints = []
    for key in [f'{arm}_{tag}' for arm in ARMS for tag in SEEDS]:
        if key not in data:
            continue
        row = data[key]
        checkpoints.append({
            'run_id': key, 'arm': row['arm'], 'tag': row['tag'],
            'pretraining_seed': row['pretraining_seed'],
            'checkpoint_sha256': row['checkpoint_sha256'],
            'per_probe_seed': row['per_probe_seed'],
            **{m: round(row[m], 4) for m in METRICS},
            **{f'{m}_std': round(row[f'{m}_std'], 4) for m in METRICS},
        })
    ab = pair_contrast('A', 'B', data, 'A_vs_B') if all(
        f'A_{t}' in data and f'B_{t}' in data for t in SEEDS) else None
    bc = pair_contrast('B', 'C', data, 'B_vs_C') if all(
        f'B_{t}' in data and f'C_{t}' in data for t in SEEDS) else None
    return {
        'schema_version': 'stage7b_schedule_comparison/v1',
        'thresholds_pp': {'miou': MIOU_THRESHOLD_PP, 'class': CLASS_THRESHOLD_PP},
        'checkpoints': checkpoints,
        'A_vs_B': ab,
        'B_vs_C': bc,
    }


def write_csv(payload: Mapping[str, object], path: Path) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'run_id', 'arm', 'tag', 'pretraining_seed',
                         'probe_seed', 'soil', 'crop', 'weed', 'mIoU'])
        for row in payload['checkpoints']:
            for i, probe_seed in enumerate((0, 1, 2)):
                writer.writerow(['checkpoint', row['run_id'], row['arm'], row['tag'],
                                 row['pretraining_seed'], probe_seed,
                                 row['per_probe_seed']['soil'][i],
                                 row['per_probe_seed']['crop'][i],
                                 row['per_probe_seed']['weed'][i],
                                 row['per_probe_seed']['mIoU'][i]])
            writer.writerow(['mean', row['run_id'], row['arm'], row['tag'],
                             row['pretraining_seed'], 'mean',
                             row['soil'], row['crop'], row['weed'], row['mIoU']])
        for label in ('A_vs_B', 'B_vs_C'):
            contrast = payload.get(label)
            if not contrast:
                continue
            for row in contrast['per_seed_delta']:
                writer.writerow([label, contrast['delta_of'], '', row['tag'], '',
                                 'delta', row['soil'], row['crop'],
                                 row['weed'], row['mIoU']])


def render_md(payload: Mapping[str, object]) -> str:
    lines = ['# Stage 7B — 配对调度消融比较', '']
    lines.append('- 6 个 encoder checkpoint，每个 2,000 optimizer updates；'
                 '3 臂 × 2 预训练 seed。同一 seed 的三臂只差调度。')
    lines.append(f"- 阈值：mIoU {payload['thresholds_pp']['miou']} pp；"
                 f"类别 {payload['thresholds_pp']['class']} pp。")
    lines.append('')
    lines.append('## 逐 checkpoint（probe seeds 0/1/2 均值）')
    lines.append('')
    lines.append('| run | arm | seed | soil | crop | weed | mIoU |')
    lines.append('|---|---|---|---|---|---|---|')
    for row in payload['checkpoints']:
        lines.append(f"| {row['run_id']} | {row['arm']} | {row['pretraining_seed']} "
                     f"| {row['soil']:.2f} | {row['crop']:.2f} | {row['weed']:.2f} "
                     f"| {row['mIoU']:.2f} |")
    lines.append('')
    lines.append('## 逐 probe seed 原始值（mIoU / crop / weed）')
    lines.append('')
    lines.append('| run | probe0 mIoU | probe1 mIoU | probe2 mIoU | crop(0/1/2) | weed(0/1/2) |')
    lines.append('|---|---|---|---|---|---|')
    for row in payload['checkpoints']:
        m = row['per_probe_seed']['mIoU']
        c = row['per_probe_seed']['crop']
        w = row['per_probe_seed']['weed']
        lines.append(f"| {row['run_id']} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} "
                     f"| {'/'.join(f'{v:.2f}' for v in c)} "
                     f"| {'/'.join(f'{v:.2f}' for v in w)} |")
    lines.append('')
    for label, title in (('A_vs_B', 'A vs B（同 warmup，horizon 2000 vs 50000）'),
                         ('B_vs_C', 'B vs C（同 horizon 50000，warmup 100 vs 2500）')):
        lines.append(f'## {title}')
        lines.append('')
        contrast = payload.get(label)
        if not contrast:
            lines.append('（缺少数据）')
            lines.append('')
            continue
        lines.append(f"（Δ = {contrast['delta_of']}，逐预训练 seed 配对）")
        lines.append('')
        lines.append('| seed | Δsoil | Δcrop | Δweed | ΔmIoU |')
        lines.append('|---|---|---|---|---|')
        for row in contrast['per_seed_delta']:
            lines.append(f"| {row['tag']} | {row['soil']:+.3f} | {row['crop']:+.3f} "
                         f"| {row['weed']:+.3f} | {row['mIoU']:+.3f} |")
        mean = contrast['mean_delta']
        lines.append(f"| **均值** | {mean['soil']:+.3f} | {mean['crop']:+.3f} "
                     f"| {mean['weed']:+.3f} | {mean['mIoU']:+.3f} |")
        lines.append('')
        cons = contrast['seed_direction_consistent']
        lines.append(f"- 两个 seed 方向一致（mIoU / crop / weed）："
                     f"{cons['mIoU']} / {cons['crop']} / {cons['weed']}")
        lines.append(f"- 决策：**{contrast['decision']}** — {contrast['reason']}")
        if contrast['class_conflict']:
            lines.append(f"- 类别冲突：{contrast['class_conflict']}")
        lines.append('')
    lines.append('## 说明')
    lines.append('')
    lines.append('- 两个预训练 seed 只支持初步配对证据，不做统计显著性声明。')
    lines.append('- Stage 7A 的 10k/20k/50k 为历史参考，不计入本阶段新重复。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    data = read_all(args.manifest)
    payload = build_payload(data)
    R.mkdir(exist_ok=True)
    (R / 'stage7b_phenobench_per_checkpoint.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_csv(payload, R / 'stage7b_schedule_comparison.csv')
    (R / 'stage7b_schedule_comparison.md').write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({
        'n_checkpoints': len(payload['checkpoints']),
        'A_vs_B': payload['A_vs_B']['decision'] if payload['A_vs_B'] else None,
        'B_vs_C': payload['B_vs_C']['decision'] if payload['B_vs_C'] else None,
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
