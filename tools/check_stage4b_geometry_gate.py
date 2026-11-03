#!/usr/bin/env python
"""Stage 4B offline geometry gate.

Reads the full-corpus audit payload and decides, BEFORE any GPU training, whether

* G1/G2 kept a real structure preference,
* G1/G2/G3 reached the geometry of the random baseline B (target union, context
  visibility, duplicate coverage and their distributions),
* the U candidate-sampling control is distributionally equivalent to B,
* the G3 shifted prior kept the boundary map's local statistics.

Nothing here looks at PhenoBench or at a trained checkpoint.  The tolerance is
the protocol's exploratory engineering target: means within 5 patch cells of B,
with the tighter 3-cell band reported separately.  A pass is necessary but not
sufficient for calling an arm "geometry-matched": the distribution distances are
reported alongside so a mean-only match cannot be presented as a full match.

Usage:
    ./.venv/bin/python -m tools.check_stage4b_geometry_gate \
        --audit runs/stage4b_mask_audit/full_audit.json \
        --out reports/stage4b_geometry_gate.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

#: metrics whose means must be close to B for the structure arms
GEOMETRY_METRICS = ('target_cells', 'context_cells', 'duplicated_target_area',
                    'target_overlap_fraction')
#: metrics that must move AWAY from B for a structure arm to be meaningful
STRUCTURE_EXPECTATIONS = {
    'G1': ('mean_vegetation_target_union', 'greater'),
    'G2': ('mean_boundary_target_union', 'greater'),
    'G3': ('mean_boundary_target_union', 'not_greater'),
}
MEAN_TOLERANCE = 5.0        # exploratory engineering target (patch cells)
TIGHT_TOLERANCE = 3.0
DISTRIBUTION_TOLERANCE = 0.08   # KS statistic on the per-image distribution


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float('nan')
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side='right') / a.size
    cdf_b = np.searchsorted(b, grid, side='right') / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def load_per_image(path: Path, arms: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the gzipped long-format CSV written by the audit."""
    import csv
    import gzip
    data: Dict[str, Dict[str, List[float]]] = {arm: {} for arm in arms}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        columns = [c for c in reader.fieldnames if ':' in c]
        for column in columns:
            arm, metric = column.split(':', 1)
            if arm in data:
                data[arm][metric] = []
        for row in reader:
            for column in columns:
                arm, metric = column.split(':', 1)
                if arm in data:
                    value = row[column]
                    data[arm][metric].append(float(value) if value != '' else float('nan'))
    return {arm: {metric: np.asarray(values, dtype=np.float64)
                  for metric, values in metrics.items()}
            for arm, metrics in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='runs/stage4b_mask_audit/full_audit_batch16.json')
    parser.add_argument('--per-image',
                        default='runs/stage4b_mask_audit/full_audit_batch16_per_image.csv.gz')
    parser.add_argument('--out', default='reports/stage4b_geometry_gate.json')
    args = parser.parse_args()

    payload = json.loads(Path(args.audit).read_text(encoding='utf-8'))
    arms = payload['arms']
    aggregate = payload['aggregate']
    per_image = (load_per_image(Path(args.per_image), arms)
                 if Path(args.per_image).is_file() else None)

    checks: List[Dict[str, object]] = []

    def add(name: str, ok: bool, detail: object,
            severity: str = 'required') -> None:
        checks.append({'check': name, 'pass': bool(ok),
                       'severity': severity, 'detail': detail})

    def delta(arm: str, metric: str) -> float:
        return (aggregate[arm]['overall'][metric]['mean']
                - aggregate['B']['overall'][metric]['mean'])

    # ---- 1. structure preference survives the geometry control ---------- #
    for arm, (metric, direction) in STRUCTURE_EXPECTATIONS.items():
        if arm not in aggregate:
            continue
        d = delta(arm, metric)
        if direction == 'greater':
            ok = d > 0.005
        else:
            ok = d <= 0.002
        add(f'{arm}:structure_preference', ok,
            {'metric': metric, 'delta_vs_B': d, 'expected': direction},
            severity='control' if arm == 'G3' else 'required')

    # ---- 2. geometry means close to B ----------------------------------- #
    for arm in ('G1', 'G2'):
        if arm not in aggregate:
            continue
        details = {}
        ok_all = True
        for metric in GEOMETRY_METRICS:
            d = delta(arm, metric)
            scale = 1.0 if metric == 'target_overlap_fraction' else 1.0
            ok = abs(d) <= MEAN_TOLERANCE * scale
            details[metric] = {'delta_vs_B': d, 'within_5': ok,
                               'within_3': abs(d) <= TIGHT_TOLERANCE * scale}
            ok_all = ok_all and ok
        add(f'{arm}:geometry_mean_match', ok_all, details)

    # G3 and G0 are controls, not structure arms.  Their geometry is reported
    # with the same tolerance but a control failure does not block training:
    # G3 must be interpreted with its residual distribution mismatch, and G0's
    # over-dispersion is itself a result (the penalty with no structure term
    # cannot be geometry-preserving).
    for arm in ('G3', 'G0'):
        if arm not in aggregate:
            continue
        details = {}
        ok_all = True
        for metric in GEOMETRY_METRICS:
            d = delta(arm, metric)
            ok = abs(d) <= MEAN_TOLERANCE
            details[metric] = {'delta_vs_B': d, 'within_5': ok,
                               'within_3': abs(d) <= TIGHT_TOLERANCE}
            ok_all = ok_all and ok
        add(f'{arm}:geometry_mean_match', ok_all, details,
            severity='control')

    # ---- 3. distribution distances -------------------------------------- #
    if per_image is not None:
        for arm in ('G1', 'G2'):
            if arm not in per_image or 'target_cells' not in per_image[arm]:
                continue
            details = {}
            ok_all = True
            for metric in ('target_cells', 'context_cells',
                           'duplicated_target_area'):
                ks = ks_statistic(per_image[arm][metric],
                                  per_image['B'][metric])
                ok = ks <= DISTRIBUTION_TOLERANCE
                details[metric] = {'ks': ks, 'within_tolerance': ok}
                ok_all = ok_all and ok
            add(f'{arm}:geometry_distribution_match', ok_all, details)

        for arm in ('G3', 'U', 'G0'):
            if arm not in per_image or 'target_cells' not in per_image[arm]:
                continue
            details = {}
            ok_all = True
            for metric in ('target_cells', 'context_cells',
                           'duplicated_target_area'):
                ks = ks_statistic(per_image[arm][metric],
                                  per_image['B'][metric])
                ok = ks <= DISTRIBUTION_TOLERANCE
                details[metric] = {'ks': ks, 'within_tolerance': ok}
                ok_all = ok_all and ok
            add(f'{arm}:geometry_distribution_match', ok_all, details,
                severity='control' if arm != 'U' else 'required')

        # multiplicity distribution: compare the fraction of cells repeated
        repeat_fraction = per_image['B']['target_overlap_fraction']
        for arm in ('G1', 'G2'):
            ks = ks_statistic(per_image[arm]['target_overlap_fraction'],
                              repeat_fraction)
            add(f'{arm}:overlap_fraction_distribution',
                ks <= DISTRIBUTION_TOLERANCE,
                {'ks': ks, 'tolerance': DISTRIBUTION_TOLERANCE})
        for arm in ('G3', 'G0'):
            ks = ks_statistic(per_image[arm]['target_overlap_fraction'],
                              repeat_fraction)
            add(f'{arm}:overlap_fraction_distribution',
                ks <= DISTRIBUTION_TOLERANCE,
                {'ks': ks, 'tolerance': DISTRIBUTION_TOLERANCE},
                severity='control')

    # ---- 4. U is distributionally equivalent to B ----------------------- #
    if per_image is not None and 'U' in per_image:
        details = {}
        ok_all = True
        for metric in ('target_cells', 'context_cells',
                       'duplicated_target_area',
                       'mean_vegetation_target_union',
                       'mean_boundary_target_union'):
            ks = ks_statistic(per_image['U'][metric], per_image['B'][metric])
            d = (aggregate['U']['overall'][metric]['mean']
                 - aggregate['B']['overall'][metric]['mean'])
            ok = ks <= DISTRIBUTION_TOLERANCE and abs(d) <= 1.0
            details[metric] = {'ks': ks, 'delta_vs_B': d, 'ok': ok}
            ok_all = ok_all and ok
        add('U:equivalent_to_B', ok_all, details)

    # ---- 5. G3 keeps the boundary map's local statistics ---------------- #
    shift_path = Path('reports/stage4b_shift_prior_stats.json')
    if shift_path.is_file():
        shift = json.loads(shift_path.read_text(encoding='utf-8'))
        corr = abs(float(shift['mean_pearson_correlation_with_original']))
        add('G3:shift_keeps_local_structure',
            (shift['value_multiset_preserved_fraction'] == 1.0
             and corr < 0.3
             and shift['lag1_retention_ratio'] > 0.8
             and shift['fraction_abs_correlation_above_0.5'] < 0.05),
            {'value_multiset_preserved_fraction':
                 shift['value_multiset_preserved_fraction'],
             'mean_abs_correlation_with_original': corr,
             'lag1_retention_ratio': shift['lag1_retention_ratio'],
             'fraction_abs_correlation_above_0.5':
                 shift['fraction_abs_correlation_above_0.5'],
             'seam_to_interior_jump': shift['mean_seam_to_interior_ratio']},
            severity='control')

    # ---- 6. no degenerate samplers -------------------------------------- #
    if per_image is not None:
        bad = {arm: int(np.isnan(per_image[arm]['target_cells']).sum())
               for arm in arms if arm in per_image}
        add('no_nan_masks', all(v == 0 for v in bad.values()), bad)

    required = [c for c in checks if c['severity'] == 'required']
    controls = [c for c in checks if c['severity'] == 'control']
    passed = all(c['pass'] for c in required)
    report = {
        'audit': str(args.audit),
        'n_images': payload['n_images'],
        'geometry_beta': payload.get('geometry_beta'),
        'geometry_control': payload.get('geometry_control'),
        'tolerances': {'mean_patch_cells': MEAN_TOLERANCE,
                       'tight_patch_cells': TIGHT_TOLERANCE,
                       'ks_statistic': DISTRIBUTION_TOLERANCE},
        'required_checks': required,
        'control_checks': controls,
        'gate_passed': passed,
        'interpretation': (
            'required checks are the precondition for training the structure '
            'arms G1/G2; control checks describe G3/G0 and are reported rather '
            'than enforced because G3 is an alignment control (its residual '
            'distribution mismatch must be quoted with G2-G3) and G0 is the '
            'no-structure mechanism control (its over-dispersion is a result).'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')

    print(f'STAGE 4B GEOMETRY GATE: {"PASS" if passed else "FAIL"} '
          f'(required checks only)')
    for check in required + controls:
        tag = 'required' if check['severity'] == 'required' else 'control '
        print(f'  [{tag}] [{"ok" if check["pass"] else "NOTE"}] {check["check"]}')
    print(f'wrote {out}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
