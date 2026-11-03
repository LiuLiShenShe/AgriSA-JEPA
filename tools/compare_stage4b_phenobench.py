#!/usr/bin/env python
"""Build the Stage 4B PhenoBench comparison and the mechanism ablation.

Reads every ``runs/stage4b_phenobench/results/<label>/result.json`` (the G arms
plus the symlinked Stage 4A references) and writes

* ``reports/stage4b_phenobench_comparison.{csv,json}`` -- per evaluation point
  metrics and the paired arm-vs-arm differences in percentage points;
* ``reports/stage4b_mechanism_ablation.{csv,json}`` -- the three mechanisms the
  protocol asks to keep separate:
    * structure information    -> G1 - G0, G2 - G0
    * mask geometry            -> G0 - B
    * prior spatial alignment  -> G2 - G3, and G1/G2 vs C1/C2

All differences are paired over the three probe seeds (0/1/2 for every arm) and
reported with a paired sign test.  With three seeds the smallest attainable
two-sided p is 0.25, so the report states explicitly that these are
exploratory screenings and not significance tests.

Usage:
    ./.venv/bin/python -m tools.compare_stage4b_phenobench \
        --results runs/stage4b_phenobench/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

METRICS = ('crop', 'weed', 'soil', 'mIoU')

LABELS = {
    'A_0': ('A', 0),
    'B_500': ('B', 500),
    'B_2000': ('B', 2000),
    'C1_500': ('C1', 500),
    'C2_500': ('C2', 500),
    'C3_500': ('C3', 500),
    'G0_500': ('G0', 500),
    'G1_500': ('G1', 500),
    'G2_500': ('G2', 500),
    'G3_500': ('G3', 500),
}

#: name -> (candidate, baseline, question answered)
COMPARISONS = {
    'G0-B': ('G0_500', 'B_500',
             'does the geometry-control mechanism itself help?'),
    'G1-G0': ('G1_500', 'G0_500',
              'does vegetation structure help beyond geometry control?'),
    'G2-G0': ('G2_500', 'G0_500',
              'does boundary structure help beyond geometry control?'),
    'G3-G0': ('G3_500', 'G0_500',
              'does a misaligned prior behave like no structure at all?'),
    'G2-G3': ('G2_500', 'G3_500',
              'is the correct spatial position of the boundary prior useful?'),
    'G1-C1': ('G1_500', 'C1_500',
              'did controlling geometry change the vegetation result?'),
    'G2-C2': ('G2_500', 'C2_500',
              'did controlling geometry change the boundary result?'),
    'G1-B': ('G1_500', 'B_500',
             'geometry-controlled vegetation against the random baseline'),
    'G2-B': ('G2_500', 'B_500',
             'geometry-controlled boundary against the random baseline'),
    'G3-B': ('G3_500', 'B_500',
             'the shifted (misaligned) control against the random baseline'),
    'B-A': ('B_2000', 'A_0',
            'agricultural continued pretraining against official weights'),
}

EXPLORATORY_THRESHOLD_PP = 0.5


def load_results(results_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for label in LABELS:
        path = results_dir / label / 'result.json'
        if path.is_file():
            out[label] = json.loads(path.read_text(encoding='utf-8'))
    return out


def aggregate(payload: dict, which: str = 'official') -> Dict[str, Dict[str, float]]:
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
    if metric not in a or metric not in b:
        return {}
    va, vb = a[metric]['values'], b[metric]['values']
    n = min(len(va), len(vb))
    diffs = [va[i] - vb[i] for i in range(n)]
    wins = sum(1 for d in diffs if d > 0)
    mean = sum(diffs) / max(n, 1)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)) if n > 1 else 0.0
    # exact two-sided sign test (n is at most 3 here, so the normal
    # approximation would be dishonest)
    p = 1.0
    if n > 0:
        k = min(wins, n - wins)
        from math import lgamma
        tail = sum(math.exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
                            - n * math.log(2.0)) for i in range(k + 1))
        p = min(1.0, 2.0 * tail)
    base = b[metric]['mean']
    return {
        'baseline_mean': base,
        'candidate_mean': a[metric]['mean'],
        'delta_pp': mean,
        'relative_percent': (100.0 * mean / base) if base else float('nan'),
        'n_seeds_paired': n,
        'wins': wins,
        'losses': n - wins,
        'sign_test_p_exact': p,
        'per_seed_delta_pp': [round(d, 4) for d in diffs],
        'se_pp': sd / math.sqrt(n) if n > 1 else 0.0,
    }


def signal_label(deltas: Dict[str, Dict[str, float]]) -> str:
    """Exploratory screening label, not a significance verdict."""
    miou = deltas.get('mIoU')
    weed = deltas.get('weed')
    if not miou:
        return 'incomplete'
    big = abs(miou['delta_pp']) >= EXPLORATORY_THRESHOLD_PP
    weed_big = weed is not None and abs(weed['delta_pp']) >= EXPLORATORY_THRESHOLD_PP
    if miou['delta_pp'] > 0 and (big or weed_big):
        return 'positive'
    if miou['delta_pp'] < 0 and big:
        return 'negative'
    return 'mixed'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='runs/stage4b_phenobench/results')
    parser.add_argument('--out-csv',
                        default='reports/stage4b_phenobench_comparison.csv')
    parser.add_argument('--out-json',
                        default='reports/stage4b_phenobench_comparison.json')
    parser.add_argument('--ablation-csv',
                        default='reports/stage4b_mechanism_ablation.csv')
    parser.add_argument('--ablation-json',
                        default='reports/stage4b_mechanism_ablation.json')
    args = parser.parse_args()

    results_dir = Path(args.results)
    payloads = load_results(results_dir)
    if not payloads:
        print(f'no results found in {results_dir}')
        return 1

    official = {label: aggregate(p, 'official') for label, p in payloads.items()}

    # ---- per evaluation point ------------------------------------------ #
    rows: List[Dict[str, object]] = []
    for label in sorted(official):
        arm, steps = LABELS[label]
        payload = payloads[label]
        row: Dict[str, object] = {
            'label': label, 'arm': arm, 'optimizer_steps': steps,
            'checkpoint': payload.get('backbone_checkpoint', ''),
            'checkpoint_sha256': payload.get('backbone_checkpoint_sha256', ''),
            'continued_training': payload.get('continued_training', ''),
            'n_seeds': len(payload.get('seeds', [])),
            'cross_check_agrees': all(
                s.get('metric_cross_check', {}).get('agrees', False)
                for s in payload.get('seeds', [])),
        }
        for metric in METRICS:
            if metric in official[label]:
                entry = official[label][metric]
                row[f'official_{metric}_mean'] = round(entry['mean'], 4)
                row[f'official_{metric}_std'] = round(entry['std'], 4)
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

    # ---- paired comparisons --------------------------------------------- #
    comparisons: Dict[str, object] = {}
    ablation_rows: List[Dict[str, object]] = []
    for name, (candidate_label, baseline_label, question) in COMPARISONS.items():
        if candidate_label not in official or baseline_label not in official:
            comparisons[name] = {'missing': [candidate_label, baseline_label]}
            continue
        block: Dict[str, object] = {}
        for metric in METRICS:
            delta = paired_delta(official[candidate_label],
                                 official[baseline_label], metric)
            if delta:
                block[metric] = delta
        label = signal_label(block)
        block['question'] = question
        block['exploratory_signal'] = label
        block['seeds_direction_consistent'] = bool(
            block.get('mIoU', {}).get('per_seed_delta_pp')
            and all((d > 0) == (block['mIoU']['delta_pp'] > 0)
                    for d in block['mIoU']['per_seed_delta_pp']))
        comparisons[name] = block
        ablation_rows.append({
            'comparison': name,
            'candidate': candidate_label,
            'baseline': baseline_label,
            'question': question,
            'mIoU_delta_pp': round(block.get('mIoU', {}).get('delta_pp', float('nan')), 4),
            'crop_delta_pp': round(block.get('crop', {}).get('delta_pp', float('nan')), 4),
            'weed_delta_pp': round(block.get('weed', {}).get('delta_pp', float('nan')), 4),
            'soil_delta_pp': round(block.get('soil', {}).get('delta_pp', float('nan')), 4),
            'seeds_winning': block.get('mIoU', {}).get('wins'),
            'seeds_total': block.get('mIoU', {}).get('n_seeds_paired'),
            'sign_test_p_exact': round(
                block.get('mIoU', {}).get('sign_test_p_exact', float('nan')), 4),
            'exploratory_signal': label,
            'direction_consistent': block['seeds_direction_consistent'],
        })

    ablation_csv = Path(args.ablation_csv)
    ablation_csv.parent.mkdir(parents=True, exist_ok=True)
    with ablation_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        for row in ablation_rows:
            writer.writerow(row)

    payload = {
        'results_dir': str(results_dir),
        'labels_present': sorted(official),
        'labels_missing': sorted(set(LABELS) - set(official)),
        'metrics': list(METRICS),
        'per_evaluation_point': rows,
        'comparisons': comparisons,
        'thresholds': {
            'exploratory_pp': EXPLORATORY_THRESHOLD_PP,
            'note': ('exploratory screening threshold, not a significance '
                     'level; with 3 probe seeds the smallest attainable '
                     'two-sided sign-test p is 0.25'),
        },
    }
    Path(args.out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    Path(args.ablation_json).write_text(
        json.dumps({'rows': ablation_rows,
                    'note': ('three mechanisms kept separate: structure '
                             'information (G*-G0), mask geometry (G0-B) and '
                             'prior spatial alignment (G2-G3)')},
                   indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # ---- console ------------------------------------------------------- #
    print('official metrics (mean +/- std over 3 probe seeds)')
    print(f'{"label":<10}{"crop":>18}{"weed":>18}{"soil":>18}{"mIoU":>18}')
    for label in sorted(official):
        def fmt(metric: str) -> str:
            if metric not in official[label]:
                return '-'
            entry = official[label][metric]
            return f'{entry["mean"]:.2f}+/-{entry["std"]:.2f}'
        print(f'{label:<10}{fmt("crop"):>18}{fmt("weed"):>18}'
              f'{fmt("soil"):>18}{fmt("mIoU"):>18}')
    print()
    for name, block in comparisons.items():
        if 'mIoU' not in block:
            continue
        d = block['mIoU']
        print(f'{name:<8} mIoU {d["delta_pp"]:+.2f} pp '
              f'({d["relative_percent"]:+.2f}% rel), '
              f'weed {block["weed"]["delta_pp"]:+.2f} pp, '
              f'crop {block["crop"]["delta_pp"]:+.2f} pp, '
              f'{d["wins"]}/{d["n_seeds_paired"]} seeds, '
              f'{block["exploratory_signal"]}')
    print(f'\nwrote {out_csv}, {args.out_json}, {ablation_csv}, '
          f'{args.ablation_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
