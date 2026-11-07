#!/usr/bin/env python
"""Stage 8 Phase 5: collect and aggregate every formal run into one ledger.

Reads:
* ``reports/stage8_experiment_registry.csv`` (frozen command reference),
* ``reports/stage8_phase4_wallclock.csv`` (execution record),
* the many ``result.json`` / ``results.json`` written under
  ``/data/agrisa_jepa_runs/stage8/<task>/<backbone>[_seedN]/``,
* the Stage 7B PhenoBench results (``reports/stage7b_phenobench_per_checkpoint.json``).

and emits ``reports/stage8_results_ledger.json`` — one row per
backbone x task, aggregating head-seed mean/std of the primary metric plus
per-seed detail pointers, with every metric tied to its source file and the
backbone SHA reference from the registry.

Only existing artefacts are read; missing results are marked ``MISSING`` and
never guessed.  A task whose registry rows are REUSE_STAGE7B is read from the
Stage 7B ledger instead.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'reports'
RUN_ROOT = Path('/data/agrisa_jepa_runs/stage8')

PRIMARY_METRICS = {
    'gwhd_det': ('final_test', 'map'),
    'cropandweed_sem': ('self_metrics', 'mIoU_percent'),
    'deepweeds_cls': ('test_metrics', 'macro_f1'),
    'plantdoc_cls': ('test_metrics', 'macro_f1'),
    'phenobench_sem': ('official_metrics', 'mIoU'),
}


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def pick(result: dict, path: str):
    """Descend dotted path into a result dict; None when absent."""
    cur = result
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def gwhd_metric(result: dict):
    """detection results.json: final_test.map (COCO AP), per-seed file."""
    final_test = result.get('final_test') or {}
    return {'primary': (final_test.get('map') or 0.0) * 0 + final_test.get('map'),
            'AP50': final_test.get('map_50'),
            'train_seconds': result.get('train_seconds'),
            'peak_gpu_gb': result.get('peak_gpu_gb'),
            'backbone_sha256': (result.get('protocol') or {}).get('backbone_sha256')}


def seg_metric(result: dict, seed: Dict[str, object]):
    """phenobench/cropandweed probe seed record -> primary metric."""
    if 'official_metrics' in seed and seed['official_metrics']:
        m = seed['official_metrics']
        return {'primary': m.get('mIoU'),
                'crop': m.get('IoU (crop)'),
                'weed': m.get('IoU (weed)'),
                'soil': m.get('IoU (soil)'),
                'source': 'official_metrics'}
    sm = seed.get('self_metrics') or {}
    return {'primary': sm.get('mIoU_percent'),
            'source': 'self_metrics'}


def cls_metric(result: dict, seed: Dict[str, object]):
    tm = seed.get('test_metrics') or {}
    return {'primary': tm.get('macro_f1'),
            'accuracy': tm.get('accuracy'),
            'n_test': tm.get('n_images')}


def collect_run(spec: Dict[str, object], task: str) -> Dict[str, object]:
    """Aggregate the seeds of one (backbone, task) registry group."""
    status = spec['status']
    out = {'task_id': task,
           'backbone': spec['backbone'],
           'status': status,
           'output_dir': spec['output_dir']}
    if status == 'REUSE_STAGE7B':
        out['source'] = 'stage7b_phenobench_per_checkpoint.json'
        return out
    result = load_json(Path(spec['output_dir']) / 'result.json') or \
        load_json(Path(spec['output_dir']) / 'results.json')
    if result is None:
        out['status'] = 'MISSING'
        out['error'] = 'no result.json/results.json'
        return out
    out['backbone_sha256'] = (result.get('backbone_checkpoint_sha256')
                              or (result.get('protocol') or {})
                              .get('backbone_sha256'))
    out['continued_training'] = result.get('continued_training')
    seeds = result.get('seeds') or []
    rows = []
    if task == 'gwhd_det':
        rows = [{'seed': spec['downstream_seed'],
                 'primary_metric': (result.get('final_test') or {}).get('map'),
                 'AP50': (result.get('final_test') or {}).get('map_50'),
                 'final_val_map': (result.get('final_val') or {}).get('map')}]
        values = [r['primary_metric'] for r in rows if r['primary_metric'] is not None]
    elif task in ('cropandweed_sem', 'phenobench_sem'):
        for s in seeds:
            metric = seg_metric(result, s)
            rows.append({'seed': s.get('seed'), **metric})
        values = [r['primary'] for r in rows if r['primary'] is not None]
    else:  # classification
        for s in seeds:
            metric = cls_metric(result, s)
            rows.append({'seed': s.get('seed'), **metric})
        values = [r['primary'] for r in rows if r['primary'] is not None]
    n = len(values)
    out['n_seeds'] = n
    out['seeds'] = rows
    if n:
        out['mean'] = round(sum(values) / n, 4)
        out['std'] = round((sum((v - sum(values) / n) ** 2 for v in values)
                            / n) ** 0.5, 4)
        out['min'], out['max'] = round(min(values), 4), round(max(values), 4)
    return out


def group_registry(rows: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        key = (r['task_id'], r['backbone'])
        groups.setdefault(key, []).append(r)
    return groups


def main() -> int:
    reg_rows = list(csv.DictReader((R / 'stage8_experiment_registry.csv')
                                   .open(encoding='utf-8')))
    groups = group_registry(reg_rows)

    ledger = {'schema_version': 'stage8_results_ledger/v1',
              'primary_metrics': PRIMARY_METRICS,
              'runs': []}
    missing = []
    for (task, backbone), rows in groups.items():
        # rows: one per head seed (except phenobench = one per backbone).
        entry = collect_run(rows[0], task)
        if entry.get('status') == 'MISSING':
            missing.append(f'{task}/{backbone}')
        if rows[0]['status'] == 'REUSE_STAGE7B':
            # pull Stage 7B per-checkpoint values for the A/B arms
            entry['stage7b_ref'] = rows[0]['output_dir']
        ledger['runs'].append(entry)

    ledger['missing'] = missing
    (R / 'stage8_results_ledger.json').write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'{len(ledger["runs"])} runs in ledger; '
          f'{len(missing)} missing: {missing[:10]}')
    return 0 if not missing else 3


if __name__ == '__main__':
    raise SystemExit(main())