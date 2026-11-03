#!/usr/bin/env python
"""Select the Stage 4B geometry-control strength from OFFLINE mask geometry.

Pre-registered rule, applied mechanically so the choice cannot be justified by
downstream PhenoBench results after the fact:

1. Candidate grid: the protocol's ``beta in {0, 1, 2, 4, 8}`` plus a refinement
   grid ``{2.5, 3, 3.5, 4.5, 5}`` for the linear form and
   ``{1, 2, 4, 8, 16, 32}`` for the squared-deviation form.
2. Keep only candidates where the structure preference survives:
   ``G1 vegetation-union gain > 0.02`` and ``G2 boundary-union gain > 0.005``.
3. Among those, minimise the worst geometry mismatch over G1/G2/G3:
   ``max(|union delta|, |context delta|)`` against B.
4. Tie-break: smaller beta.

The context column used here is the one recorded by the audit methodology that
matches the official batch semantics (the batch-wide minimum truncation).

The primary criterion is evaluated over G1/G2 only.  They are the two arms the
protocol requires to be geometry-matched before the structure question can be
asked.  G3 is an alignment CONTROL: its geometry is reported and its residual
mismatch is quoted with every G2-G3 comparison, but it is not allowed to choose
the control strength for the structure arms.  The all-arm variant is computed
and reported alongside, so the trade-off is visible rather than hidden.

IMPORTANT: this rule is applied to mask geometry only; no downstream metric is
read.  It was fixed before any G-arm PhenoBench evaluation existed.

Usage:
    ./.venv/bin/python -m tools.select_stage4b_beta \
        --glob 'runs/stage4b_mask_audit/batch_sweep_*.json' \
        --out reports/stage4b_beta_calibration.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glob', default='runs/stage4b_mask_audit/*.json')
    parser.add_argument('--out', default='reports/stage4b_beta_calibration.json')
    args = parser.parse_args()

    candidates: List[Dict[str, object]] = []
    for path in sorted(glob.glob(args.glob)):
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not {'aggregate', 'geometry_beta', 'geometry_control'} <= set(data):
            continue
        if not {'B', 'G1', 'G2'} <= set(data['aggregate']):
            continue
        agg = data['aggregate']
        def mean(arm, metric):
            return float(agg[arm]['overall'][metric]['mean'])
        union_deltas = {arm: mean(arm, 'target_cells') - mean('B', 'target_cells')
                        for arm in ('G1', 'G2', 'G3') if arm in agg}
        context_deltas = {arm: mean(arm, 'context_cells') - mean('B', 'context_cells')
                          for arm in ('G1', 'G2', 'G3') if arm in agg}
        veg_gain = (mean('G1', 'mean_vegetation_target_union')
                    - mean('B', 'mean_vegetation_target_union'))
        bnd_gain = (mean('G2', 'mean_boundary_target_union')
                    - mean('B', 'mean_boundary_target_union'))
        structure_ok = veg_gain > 0.02 and bnd_gain > 0.005
        primary = [abs(union_deltas[a]) for a in ('G1', 'G2') if a in union_deltas] + \
                  [abs(context_deltas[a]) for a in ('G1', 'G2') if a in context_deltas]
        all_arms = [abs(v) for v in union_deltas.values()] + \
                   [abs(v) for v in context_deltas.values()]
        worst_primary = max(primary) if primary else float('inf')
        worst_all = max(all_arms) if all_arms else float('inf')
        candidates.append({
            'file': path,
            'geometry_control': data['geometry_control'],
            'geometry_beta': float(data['geometry_beta']),
            'n_images': data.get('n_images'),
            'batch_size': data.get('batch_size'),
            'union_delta': union_deltas,
            'context_delta': context_deltas,
            'vegetation_union_gain': veg_gain,
            'boundary_union_gain': bnd_gain,
            'structure_preference_ok': structure_ok,
            'worst_geometry_mismatch_primary_G1G2': worst_primary,
            'worst_geometry_mismatch_all_G1G2G3': worst_all,
        })

    eligible = [c for c in candidates if c['structure_preference_ok']]
    eligible.sort(key=lambda c: (c['worst_geometry_mismatch_primary_G1G2'],
                                 c['geometry_beta']))
    chosen = eligible[0] if eligible else None

    payload = {
        'rule': ('structure preference must survive (G1 vegetation-union gain '
                 '> 0.02, G2 boundary-union gain > 0.005); then minimise the '
                 'worst |union delta| / |context delta| over the PRIMARY '
                 'geometry-matched arms G1/G2; tie-break by smaller beta. '
                 'G3 is a control and its mismatch is reported separately. '
                 'No downstream metric is used.'),
        'candidates': candidates,
        'eligible': [{'beta': c['geometry_beta'],
                      'control': c['geometry_control'],
                      'worst_primary': c['worst_geometry_mismatch_primary_G1G2'],
                      'worst_all_arms': c['worst_geometry_mismatch_all_G1G2G3'],
                      'file': c['file']} for c in eligible],
        'alternative_all_arm_choice': (
            min(eligible, key=lambda c: (c['worst_geometry_mismatch_all_G1G2G3'],
                                         c['geometry_beta']))['geometry_beta']
            if eligible else None),
        'chosen': chosen,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    print(f'candidates: {len(candidates)}, eligible: {len(eligible)}')
    if chosen:
        print(f"chosen: {chosen['geometry_control']} beta={chosen['geometry_beta']} "
              f"(worst primary mismatch "
              f"{chosen['worst_geometry_mismatch_primary_G1G2']:.2f})")
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
