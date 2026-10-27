#!/usr/bin/env python
"""Aggregate Stage 6D probe results the way the protocol requires.

The protocol's primary repetition unit is the **pretraining checkpoint**, not
the probe seed.  A checkpoint's three probe seeds (0/1/2) estimate only the
downstream head-training variance, and must never be counted as independent
pretraining replicates.

Inputs
------
``result.json`` files produced by ``src/downstream/phenobench_probe.py`` for
each (arm, run, checkpoint).  The script takes an explicit manifest of which
checkpoints belong to which arm, so it cannot silently mix arms or seeds.

Outputs
-------
* per-checkpoint table (3 probe seeds + mean/stdev + delta vs A),
* per-arm across-checkpoint summary (mean/stdev over the checkpoint-level means),
* paired B1/B2 per-pair deltas,
* a sensitivity summary over a chosen subset (e.g. only the new traceable runs),
* machine-readable JSON + Markdown.

The tool refuses to aggregate a checkpoint whose probe seeds are not exactly
[0,1,2] or whose val-image count is not 772.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_VAL_IMAGES = 772
METRICS = ('soil', 'crop', 'weed', 'mIoU')


def display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise (tests use temp dirs)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_result(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def checkpoint_mean_miou(result: Mapping[str, object]) -> Dict[str, object]:
    """Per-checkpoint mean over the 3 probe seeds (one pretraining replicate).

    Official evaluator keys are ``IoU (crop)``, ``IoU (weed)`` and ``mIoU``.
    The official evaluator does not emit soil, so soil is taken from the
    self-implemented accumulator -- exactly the Stage 6C convention.
    """
    seeds = [s.get('seed') for s in result.get('seeds', [])]
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f'{result.get("arm_label")}: probe seeds {seeds} != {EXPECTED_SEEDS}')
    proto = result.get('protocol', {})
    if proto.get('val_images') != EXPECTED_VAL_IMAGES:
        raise ValueError(
            f'{result.get("arm_label")}: val_images {proto.get("val_images")} '
            f'!= {EXPECTED_VAL_IMAGES}')

    official_keys = {'crop': 'IoU (crop)', 'weed': 'IoU (weed)', 'mIoU': 'mIoU'}
    out: Dict[str, object] = {'_seed_values': {}}
    for metric, key in official_keys.items():
        values = [float(s['official_metrics'][key]) for s in result['seeds']]
        out['_seed_values'][metric] = values
        out[metric] = float(np.mean(values))
    soil_values = [float(s['self_metrics']['per_class_iou_percent']['soil'])
                   for s in result['seeds']]
    out['_seed_values']['soil'] = soil_values
    out['soil'] = float(np.mean(soil_values))
    return out


@dataclass
class CheckpointRecord:
    arm: str
    run_id: str
    pair_id: str
    seed: int
    traceable: bool
    checkpoint_sha256: str
    result_path: Path
    mean: Dict[str, float]


def read_checkpoints(specs: Sequence[Mapping[str, object]]) -> List[CheckpointRecord]:
    records: List[CheckpointRecord] = []
    for spec in specs:
        result = load_result(ROOT / spec['result'])
        mean = checkpoint_mean_miou(result)
        records.append(CheckpointRecord(
            arm=str(spec['arm']), run_id=str(spec['run_id']),
            pair_id=str(spec.get('pair_id', '')), seed=int(spec['seed']),
            traceable=bool(spec.get('traceable', True)),
            checkpoint_sha256=str(result.get('backbone_checkpoint_sha256')),
            result_path=ROOT / spec['result'], mean=mean))
    return records


def arm_summary(records: Sequence[CheckpointRecord]) -> Dict[str, object]:
    """Mean/std of a metric across checkpoints (the pretraining repetition)."""
    means: Dict[str, List[float]] = {m: [] for m in METRICS}
    for rec in records:
        for m in METRICS:
            means[m].append(rec.mean[m])
    summary: Dict[str, object] = {'n_checkpoints': len(records), 'checkpoints': []}
    for m in METRICS:
        vals = np.array(means[m], dtype=float)
        summary[m] = {
            'mean': round(float(np.mean(vals)), 4) if vals.size else None,
            'std': round(float(np.std(vals)), 4) if vals.size else None,
            'values': [round(float(v), 4) for v in vals],
        }
    for rec in records:
        summary['checkpoints'].append({
            'run_id': rec.run_id, 'pair_id': rec.pair_id, 'seed': rec.seed,
            'traceable': rec.traceable, 'checkpoint_sha256': rec.checkpoint_sha256,
            'result': display_path(rec.result_path),
            **{m: round(rec.mean[m], 4) for m in METRICS},
        })
    return summary


def delta_vs_A(arm_summary_map: Mapping[str, Dict[str, object]],
               arm: str, baseline: str = 'A') -> Dict[str, object]:
    out: Dict[str, object] = {}
    for m in METRICS:
        a = arm_summary_map[baseline][m]['mean']
        b = arm_summary_map[arm][m]['mean']
        out[m] = {
            'A_mean': a, f'{arm}_mean': b,
            f'{arm}_minus_A': round(b - a, 4) if (a is not None and b is not None) else None,
        }
    return out


def paired_b1_b2(records: Sequence[CheckpointRecord]) -> Dict[str, object]:
    """For each pair, compare the B1 and B2 checkpoint of the same seed."""
    by_pair: Dict[str, Dict[str, CheckpointRecord]] = {}
    for rec in records:
        if rec.pair_id and rec.pair_id.startswith('P0'):
            continue
        by_pair.setdefault(rec.pair_id, {})[rec.arm] = rec
    per_pair = []
    for pair_id in sorted(by_pair):
        arms = by_pair[pair_id]
        if 'B1' not in arms or 'B2' not in arms:
            continue
        if arms['B1'].seed != arms['B2'].seed:
            raise ValueError(
                f'pair {pair_id}: B1 seed {arms["B1"].seed} != B2 seed '
                f'{arms["B2"].seed}; B1/B2 must be paired by seed')
        row = {'pair_id': pair_id, 'seed': arms['B1'].seed}
        for m in METRICS:
            row[m] = {
                'B1': round(arms['B1'].mean[m], 4),
                'B2': round(arms['B2'].mean[m], 4),
                'B2_minus_B1': round(arms['B2'].mean[m] - arms['B1'].mean[m], 4),
            }
        per_pair.append(row)
    # cross-pair B2-B1 mean
    cross: Dict[str, object] = {}
    if per_pair:
        for m in METRICS:
            vals = np.array([row[m]['B2_minus_B1'] for row in per_pair], dtype=float)
            cross[m] = {
                'mean': round(float(np.mean(vals)), 4),
                'values': [round(float(v), 4) for v in vals],
            }
    return {'per_pair': per_pair, 'cross_pair': cross}


def build_payload(specs: Sequence[Mapping[str, object]],
                  subset_traceable: bool = False) -> Dict[str, object]:
    records = read_checkpoints(specs)
    if subset_traceable:
        records = [r for r in records if r.traceable]
    A = [r for r in records if r.arm == 'A']
    B1 = [r for r in records if r.arm == 'B1']
    B2 = [r for r in records if r.arm == 'B2']
    if not A:
        raise ValueError('no Arm A record; A baseline must be supplied')
    arm_sum = {a: arm_summary(rs) for a, rs in (('A', A), ('B1', B1), ('B2', B2))}
    payload = {
        'schema_version': 'stage6d_comparison/v1',
        'subset_traceable_only': subset_traceable,
        'n_checkpoints': {'A': len(A), 'B1': len(B1), 'B2': len(B2)},
        'arm_summaries': arm_sum,
        'delta': {
            'B1_minus_A': delta_vs_A(arm_sum, 'B1', 'A'),
            'B2_minus_A': delta_vs_A(arm_sum, 'B2', 'A'),
        },
        'paired_b1_b2': paired_b1_b2(records),
    }
    return payload


def write_markdown(payload: Dict[str, object], path: Path) -> None:
    lines = ['# Stage 6D — PhenoBench 跨预训练运行比较', '']
    if payload['subset_traceable_only']:
        lines.append('> 敏感性子集：仅包含新增且完全可追溯的 run（排除历史 P0）。')
        lines.append('')
    lines.append(f"检查点数：A={payload['n_checkpoints']['A']} "
                 f"B1={payload['n_checkpoints']['B1']} "
                 f"B2={payload['n_checkpoints']['B2']}")
    lines.append('')
    lines.append('## 每臂跨 checkpoint 汇总（预训练为重复单位）')
    lines.append('')
    lines.append('| arm | n | soil mean±std | crop mean±std | weed mean±std | mIoU mean±std |')
    lines.append('|---|---:|---|---|---|---|')
    for arm in ('A', 'B1', 'B2'):
        s = payload['arm_summaries'][arm]
        cells = [f'{arm}', str(s['n_checkpoints'])]
        for m in METRICS:
            mean = s[m]['mean']
            std = s[m]['std']
            cells.append(f'{mean}±{std}' if mean is not None else '—')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('## 相对 A 差值（每臂跨 checkpoint 均值）')
    lines.append('')
    for key in ('B1_minus_A', 'B2_minus_A'):
        lines.append(f'### {key}')
        lines.append('')
        lines.append('| 指标 | A mean | arm mean | 差值 (pp) |')
        lines.append('|---|---:|---:|---:|')
        for m in METRICS:
            d = payload['delta'][key][m]
            dv = d.get(f'{key.split("_")[0]}_minus_A')
            lines.append(f"| {m} | {d['A_mean']} | {d[key.split('_')[0] + '_mean']} | "
                         f"{dv if dv is not None else '—'} |")
        lines.append('')
    lines.append('## 配对 B1/B2（同 seed，仅新增 pair）')
    lines.append('')
    pb = payload['paired_b1_b2']
    if pb['per_pair']:
        lines.append('| pair | seed | crop B2−B1 | weed B2−B1 | mIoU B2−B1 |')
        lines.append('|---|---:|---:|---:|---:|')
        for row in pb['per_pair']:
            lines.append(f"| {row['pair_id']} | {row['seed']} | "
                         f"{row['crop']['B2_minus_B1']} | "
                         f"{row['weed']['B2_minus_B1']} | "
                         f"{row['mIoU']['B2_minus_B1']} |")
        lines.append('')
        lines.append('跨 pair 均值 B2−B1：' +
                     '，'.join(f"{m} {pb['cross_pair'][m]['mean']}" for m in METRICS))
    else:
        lines.append('（无配对数据）')
    lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path,
                        help='JSON: list of {arm, run_id, pair_id, seed, '
                             'traceable, result} entries')
    parser.add_argument('--out-prefix', required=True, type=Path)
    parser.add_argument('--traceable-only', action='store_true',
                        help='aggregate only the new fully-traceable runs')
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    specs = manifest.get('entries', []) if isinstance(manifest, dict) else manifest
    payload = build_payload(specs, subset_traceable=args.traceable_only)
    json_path = args.out_prefix.with_suffix('.json')
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_markdown(payload, args.out_prefix.with_suffix('.md'))
    print(json.dumps({
        'n_checkpoints': payload['n_checkpoints'],
        'B1_minus_A_mIoU': payload['delta']['B1_minus_A']['mIoU']['B1_minus_A'],
        'B2_minus_A_mIoU': payload['delta']['B2_minus_A']['mIoU']['B2_minus_A'],
        'outputs': [str(json_path), str(args.out_prefix.with_suffix('.md'))],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
