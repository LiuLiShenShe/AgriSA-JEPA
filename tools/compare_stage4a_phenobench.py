#!/usr/bin/env python
"""Build the Stage 4A PhenoBench comparison and ablation tables.

Reads every ``runs/stage4a_phenobench/results/<label>/result.json`` and writes

* ``reports/stage4a_phenobench_comparison.csv`` -- one row per evaluation point
* ``reports/stage4a_phenobench_comparison.json`` -- the same plus the
  paired arm-vs-arm differences in percentage points and relative percent,
  the per-class breakdown, the metric cross-check, and the arm summary.

Differences are always reported relative to the same-budget baseline B at the
same optimizer step (C1-B, C2-B, C3-B), plus B_2000 - A_0 for the training
effect.  Percentages are ``delta_pp / baseline * 100`` computed on the same
metric.

Usage:
    ./.venv/bin/python -m tools.compare_stage4a_phenobench \
        --results runs/stage4a_phenobench/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

#: label -> (arm, optimizer steps).  Imported from the evaluation driver so the
#: driver, the queue and this table cannot disagree about the evaluation points.
from tools.eval_stage4a_checkpoints import LABELS  # noqa: E402  (re-export)

METRICS = ('crop', 'weed', 'soil', 'mIoU')


def load_results(results_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for label in LABELS:
        path = results_dir / label / 'result.json'
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        out[label] = payload
    return out


def aggregate(payload: dict, which: str = 'official') -> Dict[str, Dict[str, float]]:
    """metric -> {mean, std, n, values} from one result payload."""
    block = payload.get('aggregate', {}).get(which, {})
    out: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        if metric not in block:
            continue
        entry = block[metric]
        out[metric] = {
            'mean': float(entry['mean']),
            'std': float(entry['std']),
            'n': int(payload['aggregate'].get('n_seeds', len(entry.get('values', [])))),
            'values': [float(v) for v in entry['values']],
        }
    return out


def paired_delta(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]],
                 metric: str) -> Dict[str, float]:
    """a - b for one metric, in percentage points, with a paired sign test."""
    if metric not in a or metric not in b:
        return {}
    va, vb = a[metric]['values'], b[metric]['values']
    n = min(len(va), len(vb))
    diffs = [va[i] - vb[i] for i in range(n)]
    wins = sum(1 for d in diffs if d > 0)
    mean = sum(diffs) / max(n, 1)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    # exact two-sided sign test via the normal approximation with continuity
    # correction (no scipy dependency, and honest about being an approximation)
    p = 1.0
    if n > 0:
        centred = abs(wins - n / 2.0) - 0.5
        z = centred / (0.5 * math.sqrt(n)) if n > 1 else 0.0
        p = 2.0 * 0.5 * math.erfc(z / math.sqrt(2.0)) if n > 1 else 1.0
        p = min(1.0, max(p, 1.0 / (2 ** max(n, 1))))
    base = b[metric]['mean']
    return {
        'baseline_mean': base,
        'candidate_mean': a[metric]['mean'],
        'delta_pp': mean,
        'relative_percent': (100.0 * mean / base) if base else float('nan'),
        'n_seeds_paired': n,
        'wins': wins,
        'losses': n - wins,
        'sign_test_p_approx': p,
        'per_seed_delta_pp': [round(d, 4) for d in diffs],
        'baseline_std': b[metric]['std'],
        'candidate_std': a[metric]['std'],
        'se_of_delta_pp': se,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='runs/stage4a_phenobench/results')
    parser.add_argument('--out-csv', default='reports/stage4a_phenobench_comparison.csv')
    parser.add_argument('--out-json', default='reports/stage4a_phenobench_comparison.json')
    args = parser.parse_args()

    results_dir = Path(args.results)
    payloads = load_results(results_dir)
    if not payloads:
        print(f'no results found in {results_dir}')
        return 1

    official = {label: aggregate(p, 'official') for label, p in payloads.items()}
    self_metrics = {label: aggregate(p, 'self') for label, p in payloads.items()}

    # ---- per evaluation point rows -------------------------------------- #
    rows: List[Dict[str, object]] = []
    for label in sorted(official):
        arm, steps = LABELS[label]
        payload = payloads[label]
        row: Dict[str, object] = {
            'label': label, 'arm': arm, 'optimizer_steps': steps,
            'checkpoint': payload.get('backbone_checkpoint', ''),
            'checkpoint_sha256': payload.get('backbone_checkpoint_sha256', ''),
            'continued_training': payload.get('continued_training', ''),
        }
        for metric in METRICS:
            if metric in official[label]:
                row[f'official_{metric}_mean'] = round(official[label][metric]['mean'], 4)
                row[f'official_{metric}_std'] = round(official[label][metric]['std'], 4)
            if metric in self_metrics[label]:
                row[f'self_{metric}_mean'] = round(self_metrics[label][metric]['mean'], 4)
        seeds = payload.get('seeds', [])
        row['n_seeds'] = len(seeds)
        row['cross_check_max_abs_diff_pp'] = max(
            (s.get('metric_cross_check', {}).get('max_abs_difference_pp', 0.0)
             for s in seeds), default=0.0)
        row['cross_check_agrees'] = all(
            s.get('metric_cross_check', {}).get('agrees', False) for s in seeds)
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # ---- paired differences --------------------------------------------- #
    comparisons: Dict[str, object] = {}
    for steps in (500, 2000):
        base_label = f'B_{steps}'
        if base_label not in official:
            continue
        for arm in ('C1', 'C2', 'C3'):
            label = f'{arm}_{steps}'
            if label not in official:
                continue
            block: Dict[str, object] = {}
            for metric in METRICS:
                delta = paired_delta(official[label], official[base_label], metric)
                if delta:
                    block[metric] = delta
            comparisons[f'{arm}-B @ {steps} updates'] = block
    if 'B_2000' in official and 'A_0' in official:
        block = {}
        for metric in METRICS:
            delta = paired_delta(official['B_2000'], official['A_0'], metric)
            if delta:
                block[metric] = delta
        comparisons['B-A (2,000 updates vs official weights)'] = block

    payload = {
        'results_dir': str(results_dir),
        'labels_present': sorted(official),
        'labels_missing': sorted(set(LABELS) - set(official)),
        'metrics': list(METRICS),
        'protocol': (payloads[sorted(payloads)[0]].get('protocol')
                     if payloads else None),
        'per_evaluation_point': rows,
        'comparisons': comparisons,
        'note': ('deltas are paired across probe seeds (same seeds 0/1/2 for '
                 'every arm) and reported in percentage points (pp); the '
                 'sign test is a normal approximation, not an exact test.'),
    }
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                        encoding='utf-8')

    # ---- console table --------------------------------------------------- #
    print('official metrics (mean +/- std over 3 probe seeds)')
    header = f'{"label":<10}{"crop":>18}{"weed":>18}{"mIoU":>18}'
    print(header)
    for label in sorted(official):
        def fmt(metric: str) -> str:
            if metric not in official[label]:
                return '-'
            m = official[label][metric]
            return f'{m["mean"]:.2f}+/-{m["std"]:.2f}'
        print(f'{label:<10}{fmt("crop"):>18}{fmt("weed"):>18}{fmt("mIoU"):>18}')
    print()
    for name, block in comparisons.items():
        if 'mIoU' in block:
            d = block['mIoU']
            print(f'{name}: mIoU {d["delta_pp"]:+.2f} pp '
                  f'({d["relative_percent"]:+.2f}% rel), '
                  f'{d["wins"]}/{d["n_seeds_paired"]} seeds win')
    print(f'\nwrote {out_csv} and {out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
