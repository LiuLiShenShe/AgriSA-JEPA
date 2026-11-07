#!/usr/bin/env python
"""Stage 8 Phase 5: paired contrasts across tasks and backbones.

For every completed task, computes seed-level paired contrasts:

* ``Official - A`` and ``Official - B`` per arm (does agricultural continued
  pretraining transfer?  RQ1);
* ``A - B`` per pretraining seed (does the Stage 7B schedule difference transfer
  to this task?  RQ3);

using the same seed-consistency rule as Stage 7B: a contrast counts as a clear
effect only when both pretraining seeds move in the same direction AND the mean
delta exceeds a task-specific threshold.  Classification macro-F1 uses 0.005
absolute; segmentation/detection use their native scale (1 pp mIoU / 0.002 AP).

Also emits ``reports/stage8_results_table.{md,csv}`` with per-backbone mean and
per-seed detail so every published number is traceable to a ``result.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'reports'
RUN_ROOT = Path('/data/agrisa_jepa_runs/stage8')

TASKS = {
    'plantdoc_cls': 'macro-F1 (28-class PlantDoc leaf disease)',
    'deepweeds_cls': 'macro-F1 (9-class DeepWeeds)',
    'cropandweed_sem': 'mIoU (CropsOrWeed9 CropAndWeed)',
    'gwhd_det': 'test AP (GWHD 2021 COCO)',
    'phenobench_sem': 'mIoU (PhenoBench official)',
}

# thresholds for "clear effect" per task family (absolute, in native units)
THRESHOLDS = {
    'plantdoc_cls': 0.005,
    'deepweeds_cls': 0.005,
    'cropandweed_sem': 0.5,
    'gwhd_det': 0.002,
    'phenobench_sem': 0.2,
}

PRIMARY = ('OFFICIAL', 'A_p1', 'A_p2', 'B_p1', 'B_p2')


def load_result(backbone: str, task: str) -> Optional[Dict]:
    p = RUN_ROOT / task / backbone / 'result.json'
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def extract_per_seed(backbone: str, task: str) -> Optional[List[Dict]]:
    """Return ``[{seed, primary}]`` sorted by head seed."""
    result = load_result(backbone, task)
    if result is None:
        return None
    if task == 'gwhd_det':
        seed = int(backbone.rsplit('seed', 1)[1])
        v = (result.get('final_test') or {}).get('map')
        return [{'seed': seed, 'primary': v}]
    rows = []
    for s in result.get('seeds') or []:
        if task in ('plantdoc_cls', 'deepweeds_cls'):
            primary = (s.get('test_metrics') or {}).get('macro_f1')
        elif task == 'cropandweed_sem':
            primary = ((s.get('self_metrics') or {}).get('mIoU_percent'))
        else:  # phenobench
            primary = (s.get('official_metrics') or {}).get('mIoU')
        rows.append({'seed': s.get('seed'), 'primary': primary})
    rows.sort(key=lambda r: r['seed'] if r['seed'] is not None else 0)
    return rows


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float('nan')


def render_tables(per_task: Dict[str, Dict[str, List[Dict]]]) -> None:
    csv_rows = []
    md = ['# Stage 8 — 结果表（Phase 5）', '']
    for task, title in TASKS.items():
        per_backbone = per_task.get(task, {})
        if not per_backbone:
            continue
        md += [f'## {task} — {title}', '']
        md.append('| backbone | n | mean | per-seed |')
        md.append('|---|---|---|---|')
        for backbone in PRIMARY:
            rows = per_backbone.get(backbone)
            if not rows:
                md.append(f'| {backbone} | 0 | — | (missing) |')
                csv_rows.append({'task': task, 'backbone': backbone,
                                 'n_seeds': 0, 'mean': '', 'per_seed': ''})
                continue
            values = [r['primary'] for r in rows if r['primary'] is not None]
            m = round(mean(values), 4)
            per_seed = '/'.join(f"{r['primary']:.4f}" for r in rows)
            md.append(f'| {backbone} | {len(values)} | {m} | {per_seed} |')
            csv_rows.append({'task': task, 'backbone': backbone,
                             'n_seeds': len(values), 'mean': m,
                             'per_seed': per_seed})
        md.append('')
    (R / 'stage8_results_table.md').write_text('\n'.join(md) + '\n',
                                               encoding='utf-8')
    with (R / 'stage8_results_table.csv').open('w', newline='',
                                               encoding='utf-8') as handle:
        w = csv.DictWriter(handle, fieldnames=['task', 'backbone', 'n_seeds',
                                               'mean', 'per_seed'])
        w.writeheader()
        w.writerows(csv_rows)


def paired_contrast(per_backbone, x, y, task) -> Dict[str, object]:
    """x - y paired by head seed; both must share the same seed set."""
    dx, dy = per_backbone.get(x), per_backbone.get(y)
    if not dx or not dy:
        return {'contrast': f'{x}-{y}', 'status': 'INCOMPLETE'}
    seeds_x = {r['seed']: r['primary'] for r in dx}
    seeds_y = {r['seed']: r['primary'] for r in dy}
    common = sorted(set(seeds_x) & set(seeds_y))
    if not common:
        return {'contrast': f'{x}-{y}', 'status': 'NO_SHARED_SEEDS'}
    deltas = {s: seeds_x[s] - seeds_y[s] for s in common
              if seeds_x[s] is not None and seeds_y[s] is not None}
    if not deltas:
        return {'contrast': f'{x}-{y}', 'status': 'NO_VALID_SEEDS'}
    values = list(deltas.values())
    consistent = len({round(v, 8) > 0 for v in values}) == 1
    threshold = THRESHOLDS[task]
    decision = ('CLEAR_EFFECT' if consistent and abs(mean(values)) >= threshold
                else 'INCONCLUSIVE')
    return {'contrast': f'{x}-{y}',
            'per_seed_delta': {str(s): round(v, 4) for s, v in deltas.items()},
            'mean_delta': round(mean(values), 4),
            'direction_consistent': consistent,
            'threshold': threshold,
            'decision': decision}


def main() -> int:
    per_task: Dict[str, Dict[str, List[Dict]]] = {}
    for task in TASKS:
        for backbone in PRIMARY:
            rows = extract_per_seed(backbone, task)
            if rows:
                per_task.setdefault(task, {})[backbone] = rows

    render_tables(per_task)

    contrasts = {}
    for task, per_backbone in per_task.items():
        if 'OFFICIAL' not in per_backbone:
            continue
        contrasts[task] = {
            'Official-A_p1': paired_contrast(per_backbone, 'OFFICIAL', 'A_p1', task),
            'Official-A_p2': paired_contrast(per_backbone, 'OFFICIAL', 'A_p2', task),
            'Official-B_p1': paired_contrast(per_backbone, 'OFFICIAL', 'B_p1', task),
            'Official-B_p2': paired_contrast(per_backbone, 'OFFICIAL', 'B_p2', task),
            'A-B (per arm)': {
                arm: paired_contrast(per_backbone, f'A_{arm}', f'B_{arm}', task)
                for arm in ('p1', 'p2')},
        }
    (R / 'stage8_paired_contrasts.json').write_text(
        json.dumps(contrasts, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    for task, c in contrasts.items():
        print(f'\n[{task}]')
        for k, v in c.items():
            if 'mean_delta' in v:
                print(f'  {k:14s} mean={v["mean_delta"]:+.4f} '
                      f'consistent={v["direction_consistent"]} '
                      f'-> {v["decision"]}')
            else:
                print(f'  {k:14s} {v["status"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())