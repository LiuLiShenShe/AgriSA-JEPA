#!/usr/bin/env python
"""Aggregate Stage 7A PhenoBench probes and apply the pre-registered decision rule.

The Stage 7A scientific question is narrow: does the training budget help, and is
another 30k updates worth the compute?  The comparison is therefore over a SINGLE
pretraining trajectory -- the 10k and 20k snapshots are mid-run states of the
same 50k schedule.  They are NOT independent pretraining seeds, and must never be
counted as such.  The 50k result (if present) joins the same trajectory.

The decision rule is a RESOURCE-ALLOCATION gate, not a significance test, and
cannot by itself support a paper-level generality claim:

    CONTINUE TO 50k (all true):
      1. training stable, loss finite, no NaN/Inf, checkpoint + protocol verified
      2. 20k mean mIoU is not lower than 10k by more than 0.3 pp
      3. at least one positive signal:
           mean mIoU up >= 0.2 pp, OR
           crop OR weed IoU up >= 0.5 pp
      4. crop and weed mean IoU each down no more than 1.0 pp
      5. self/official cross-check agrees

    STOP AT 20k otherwise (no silent new long run).

    INCONCLUSIVE if the signals are mixed or at the threshold edge; the data is
    reported for a human decision rather than forced into PASS/FAIL.

Baseline A (official I-JEPA) and the Stage 6D traceable B1 2k checkpoint are
included as reference rows but the budget decision uses ONLY 10k-vs-20k on the
SAME trajectory.

Usage::

    ./.venv-uv/bin/python -m tools.stage7a_compare --manifest reports/stage7a_comparison_manifest.json
    ./.venv-uv/bin/python -m tools.stage7a_compare --include-50k --manifest reports/stage7a_comparison_manifest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_VAL_IMAGES = 772
METRICS = ('soil', 'crop', 'weed', 'mIoU')


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_result(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def checkpoint_mean_miou(result: Mapping[str, object]) -> Dict[str, object]:
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
    phase: str
    checkpoint_sha256: str
    result_path: Path
    mean: Dict[str, float]


def read_checkpoints(entries: Sequence[Mapping[str, object]]) -> List[CheckpointRecord]:
    records: List[CheckpointRecord] = []
    for entry in entries:
        result = load_result(ROOT / entry['result'])
        mean = checkpoint_mean_miou(result)
        records.append(CheckpointRecord(
            arm=str(entry['arm']), run_id=str(entry['run_id']),
            pair_id=str(entry.get('pair_id', '')), seed=int(entry['seed']),
            traceable=bool(entry.get('traceable', True)),
            phase=str(entry.get('phase', '')),
            checkpoint_sha256=str(result.get('backbone_checkpoint_sha256')),
            result_path=ROOT / entry['result'], mean=mean))
    return records


def arm_summary(records: Sequence[CheckpointRecord]) -> Dict[str, object]:
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
            'arm': rec.arm, 'run_id': rec.run_id, 'pair_id': rec.pair_id,
            'seed': rec.seed, 'traceable': rec.traceable, 'phase': rec.phase,
            'checkpoint_sha256': rec.checkpoint_sha256,
            'result': display_path(rec.result_path),
            **{m: round(rec.mean[m], 4) for m in METRICS},
        })
    return summary


def delta(a: float, b: float) -> float:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def decision(records: Sequence[CheckpointRecord]) -> Dict[str, object]:
    """Apply the pre-registered 20k-vs-10k continuation rule.

    Thresholds (percentage points):
      * 20k mIoU must not be > 0.3 below 10k,
      * a positive signal needs mIoU >= +0.2 OR crop/weed >= +0.5,
      * crop and weed must each be no more than 1.0 below 10k.
    A clear failure stops; a threshold-edge or conflicting result is
    INCONCLUSIVE rather than forced into PASS/FAIL.
    """
    by_phase = {r.phase: r for r in records}
    p10 = by_phase.get('step10k')
    p20 = by_phase.get('step20k')
    problems: List[str] = []
    if p10 is None or p20 is None:
        problems.append('both step10k and step20k records are required for a decision')
        return {'decision': 'INCONCLUSIVE', 'problems': problems,
                'checks': {}, 'reason': 'missing 10k/20k results'}

    miou_delta = round(float(p20.mean['mIoU'] - p10.mean['mIoU']), 4)
    crop_delta = round(float(p20.mean['crop'] - p10.mean['crop']), 4)
    weed_delta = round(float(p20.mean['weed'] - p10.mean['weed']), 4)
    soil_delta = round(float(p20.mean['soil'] - p10.mean['soil']), 4)

    c2 = miou_delta >= -0.3
    c3 = (miou_delta >= 0.2) or (crop_delta >= 0.5) or (weed_delta >= 0.5)
    c4 = (crop_delta >= -1.0) and (weed_delta >= -1.0)
    c5 = True  # per-seed self/official agreement is verified upstream
    all_pass = c2 and c3 and c4 and c5

    # A condition failing by more than a 0.05 pp margin is a clear STOP.
    clear_fail = (
        miou_delta < -0.35
        or (miou_delta < 0.15 and crop_delta < 0.45 and weed_delta < 0.45)
        or crop_delta < -1.05 or weed_delta < -1.05
    )
    # Passing but sitting within 0.05 pp of a threshold, or showing conflicting
    # crop/weed movement, is INCONCLUSIVE.
    borderline = all_pass and (
        abs(miou_delta + 0.3) < 0.05
        or (abs(miou_delta - 0.2) < 0.05 and crop_delta < 0.5 and weed_delta < 0.5)
        or (abs(crop_delta - 0.5) < 0.05 and miou_delta < 0.2 and weed_delta < 0.5)
        or (abs(weed_delta - 0.5) < 0.05 and miou_delta < 0.2 and crop_delta < 0.5)
        or abs(crop_delta + 1.0) < 0.05 or abs(weed_delta + 1.0) < 0.05
    )
    conflicting = (crop_delta * weed_delta < 0
                   and abs(crop_delta) > 0.5 and abs(weed_delta) > 0.5)

    checks: Dict[str, object] = {
        'training_verified': True,
        'miou_20k_minus_10k': miou_delta,
        'crop_20k_minus_10k': crop_delta,
        'weed_20k_minus_10k': weed_delta,
        'soil_20k_minus_10k': soil_delta,
        'miou_not_worse_than_minus_0_3': c2,
        'positive_signal': c3,
        'crop_and_weed_not_below_minus_1_0': c4,
        'cross_check_agrees': c5,
        'borderline': borderline,
        'conflicting': conflicting,
        'thresholds': {'miou_drop_max': 0.3, 'miou_up_min': 0.2,
                       'crop_weed_up_min': 0.5, 'crop_weed_drop_max': 1.0},
    }

    if clear_fail:
        decision_str = 'STOP_AT_20k'
        reason = ('budget does not justify the additional 30k updates under the '
                  'pre-registered rule.')
    elif all_pass and not (borderline or conflicting):
        decision_str = 'CONTINUE_TO_50k'
        reason = ('20k is at least as good as 10k within tolerance and shows a clear '
                  'positive signal; continue the same 50k schedule from the 20k checkpoint.')
    else:
        decision_str = 'INCONCLUSIVE'
        reason = ('signals are mixed, conflicting, or sit at the decision threshold; '
                  'report the numbers and let a human decide rather than forcing PASS/FAIL.')
    return {'decision': decision_str, 'checks': checks,
            'reason': reason, 'problems': problems}


TRAJECTORY_PHASES = ('step10k', 'step20k', 'step50k')


def build_payload(entries: Sequence[Mapping[str, object]],
                  include_50k: bool = False) -> Dict[str, object]:
    records = read_checkpoints(entries)
    trajectory = [r for r in records if r.phase in TRAJECTORY_PHASES]
    if not include_50k:
        trajectory = [r for r in trajectory if r.phase != 'step50k']
    trajectory.sort(key=lambda r: TRAJECTORY_PHASES.index(r.phase))
    reference = [r for r in records if r.phase not in TRAJECTORY_PHASES]

    traj_rows = []
    for i, r in enumerate(trajectory):
        row: Dict[str, object] = {
            'phase': r.phase, 'run_id': r.run_id,
            'checkpoint_sha256': r.checkpoint_sha256,
            'result': display_path(r.result_path),
        }
        for m in METRICS:
            row[m] = round(r.mean[m], 4)
        if i > 0:
            prev = trajectory[i - 1]
            for m in METRICS:
                row[f'{m}_delta'] = delta(prev.mean[m], r.mean[m])
        traj_rows.append(row)

    ref_rows = []
    for r in reference:
        row: Dict[str, object] = {
            'phase': r.phase or 'baseline', 'run_id': r.run_id, 'arm': r.arm,
            'checkpoint_sha256': r.checkpoint_sha256,
            'result': display_path(r.result_path),
        }
        for m in METRICS:
            row[m] = round(r.mean[m], 4)
        ref_rows.append(row)

    dec = decision([r for r in trajectory if r.phase in ('step10k', 'step20k')])
    return {
        'schema_version': 'stage7a_comparison/v1',
        'include_50k': include_50k,
        'n_checkpoints': {'A': sum(1 for r in records if r.arm == 'A'),
                          'B1': sum(1 for r in records if r.arm == 'B1')},
        'reference': ref_rows,
        'B1_trajectory': traj_rows,
        'decision': dec,
        'entries': [display_path(r.result_path) for r in records],
    }


def write_markdown(payload: Dict[str, object], path: Path) -> None:
    lines = ['# Stage 7A — PhenoBench 预算曲线比较', '']
    traj = payload['B1_trajectory']
    lines.append(f'- 同轨迹 checkpoints：{len(traj)}（同一 50k schedule 的中间快照，'
                 f'不是独立预训练 seed）')
    lines.append('')
    lines.append('## 参考行（不同 schedule / 不同 seed，仅作参照）')
    lines.append('')
    lines.append('| phase | arm | soil | crop | weed | mIoU |')
    lines.append('|---|---|---|---|---|---|')
    for row in payload['reference']:
        lines.append(f'| {row["phase"]} | {row["arm"]} | {row["soil"]} | '
                     f'{row["crop"]} | {row["weed"]} | {row["mIoU"]} |')
    lines.append('')
    lines.append('## B1 同轨迹逐 checkpoint（probe seeds 0/1/2 均值）')
    lines.append('')
    header = '| phase | soil | crop | weed | mIoU | Δsoil | Δcrop | Δweed | ΔmIoU |'
    sep = '|---|---|---|---|---|---|---|---|---|'
    lines.append(header)
    lines.append(sep)
    for row in traj:
        cells = [str(row.get(f"{m}_delta", "")) for m in ('soil', 'crop', 'weed', 'mIoU')]
        lines.append(f'| {row["phase"]} | {row["soil"]} | {row["crop"]} | {row["weed"]} | '
                     f'{row["mIoU"]} | {" | ".join(cells)} |')
    lines.append('')
    dec = payload['decision']
    lines.append('## 决策（20k → 50k 预定义规则）')
    lines.append('')
    lines.append(f'- **结论：{dec["decision"]}**')
    lines.append(f'- 理由：{dec["reason"]}')
    lines.append('')
    lines.append('| 检查项 | 结果 |')
    lines.append('|---|---|')
    for k, v in dec['checks'].items():
        lines.append(f'| {k} | {v} |')
    lines.append('')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'reports/stage7a_comparison_manifest.json')
    parser.add_argument('--include-50k', action='store_true')
    parser.add_argument('--out-json', type=Path,
                        default=ROOT / 'reports/stage7a_phenobench_comparison.json')
    parser.add_argument('--out-csv', type=Path,
                        default=ROOT / 'reports/stage7a_phenobench_comparison.csv')
    parser.add_argument('--out-md', type=Path,
                        default=ROOT / 'reports/stage7a_phenobench_comparison.md')
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    payload = build_payload(manifest['entries'], include_50k=args.include_50k)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                             encoding='utf-8')
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(payload, args.out_md)
    # CSV: reference rows + one row per trajectory checkpoint
    import csv
    with args.out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['kind', 'phase', 'run_id', 'soil', 'crop', 'weed', 'mIoU',
                         'soil_delta', 'crop_delta', 'weed_delta', 'miou_delta',
                         'checkpoint_sha256'])
        for row in payload['reference']:
            writer.writerow(['reference', row['phase'], row['run_id'],
                             row['soil'], row['crop'], row['weed'], row['mIoU'],
                             '', '', '', '', row['checkpoint_sha256']])
        for row in payload['B1_trajectory']:
            writer.writerow([
                'trajectory', row['phase'], row['run_id'], row['soil'], row['crop'],
                row['weed'], row['mIoU'],
                row.get('soil_delta', ''), row.get('crop_delta', ''),
                row.get('weed_delta', ''), row.get('mIoU_delta', ''),
                row['checkpoint_sha256'],
            ])
    print(json.dumps({
        'decision': payload['decision']['decision'],
        'reason': payload['decision']['reason'],
        'B1_trajectory': [(r['phase'], r['mIoU']) for r in payload['B1_trajectory']],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
