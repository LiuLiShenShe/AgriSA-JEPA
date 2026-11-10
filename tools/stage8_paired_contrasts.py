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

STAGE7B_MANIFEST = R / 'stage7b_phenobench_per_checkpoint.json'


def stage7b_seeds(backbone: str) -> Optional[List[Dict]]:
    """PhenoBench A/B probe results live in the Stage 7B manifest (REUSE rows:
    their stage8 run row never re-ran; the registry points here)."""
    if not STAGE7B_MANIFEST.is_file():
        return None
    manifest = json.loads(STAGE7B_MANIFEST.read_text(encoding='utf-8'))
    seeds = manifest.get('probe_seeds', [0, 1, 2])
    for entry in manifest.get('checkpoints', []):
        if entry.get('run_id') == backbone:
            vals = entry.get('per_probe_seed', {}).get('mIoU')
            if not vals:
                return None
            return [{'seed': seeds[i], 'primary': v}
                    for i, v in enumerate(vals)]
    return None


def load_result(backbone: str, task: str) -> Optional[Dict]:
    p = RUN_ROOT / task / backbone / 'result.json'
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def _registry() -> List[Dict]:
    return json.loads((R / 'stage8_experiment_registry.json')
                      .read_text(encoding='utf-8')).get('rows', [])


def _registry_rows(backbone: str, task: str) -> List[Dict]:
    return [r for r in _registry() if r['task_id'] == task
            and r['backbone'] == backbone]


def _run_result(row: Dict) -> Optional[Dict]:
    """Load the run's final result: gwhd writes results.json, others result.json."""
    for marker in ('result.json', 'results.json'):
        p = Path(row['output_dir']) / marker
        if p.is_file():
            return json.loads(p.read_text(encoding='utf-8'))
    return None


def _primary(task: str, seed_entry: Dict, result: Dict) -> Optional[float]:
    """Seed-level primary metric for one completed run."""
    if task in ('plantdoc_cls', 'deepweeds_cls'):
        return (seed_entry.get('test_metrics') or {}).get('macro_f1')
    if task == 'cropandweed_sem':
        return (seed_entry.get('self_metrics') or {}).get('mIoU_percent')
    if task == 'gwhd_det':
        v = seed_entry.get('final_test') or result.get('final_test') or {}
        return v.get('map')
    return None


def extract_per_seed(backbone: str, task: str) -> Optional[List[Dict]]:
    """Return ``[{seed, primary}]`` for every completed registry row.

    Runs are per-seed (one result.json per row with a single seed entry)
    EXCEPT PhenoBench, whose command trains all three seeds in one run.
    """
    if task == 'phenobench_sem':
        result = load_result(backbone, task)
        if result is not None:
            rows = []
            for s in result.get('seeds') or []:
                rows.append({'seed': s.get('seed'),
                             'primary': (s.get('official_metrics') or {})
                             .get('mIoU')})
            rows.sort(key=lambda r: r['seed'] if r['seed'] is not None else 0)
            return rows or None
        return stage7b_seeds(backbone)

    out = []
    for row in _registry_rows(backbone, task):
        result = _run_result(row)
        if result is None:
            continue
        seed = row['downstream_seed']
        # per-seed runs contain exactly one seed entry, but match by seed so a
        # mislabeled run cannot silently pair the wrong numbers.
        entry = next((s for s in result.get('seeds') or []
                      if str(s.get('seed')) == str(seed)), None)
        if entry is None:
            entry = (result.get('seeds') or [None])[0]
        primary = _primary(task, entry or {}, result)
        if primary is not None:
            out.append({'seed': int(seed), 'primary': primary})
    out.sort(key=lambda r: r['seed'])
    return out or None


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
            if k == 'A-B (per arm)':
                for arm, res in v.items():
                    if 'mean_delta' in res:
                        print(f'  A-B/{arm}     mean={res["mean_delta"]:+.4f} '
                              f'consistent={res["direction_consistent"]} '
                              f'-> {res["decision"]}')
                    else:
                        print(f'  A-B/{arm}     {res["status"]}')
                continue
            if 'mean_delta' in v:
                print(f'  {k:14s} mean={v["mean_delta"]:+.4f} '
                      f'consistent={v["direction_consistent"]} '
                      f'-> {v["decision"]}')
            else:
                print(f'  {k:14s} {v["status"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())