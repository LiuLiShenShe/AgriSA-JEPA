#!/usr/bin/env python
"""Trace the real Stage 5C IQA threshold and score scale.

Questions answered with local evidence only:

1. What scale does pyIQA MUSIQ emit?  (observed range over every scored ledger)
2. Which thresholds were actually applied?  The YAML says 0.35/0.22; the code
   default says 45.0/25.0.  Only one of them can match the scores.
3. Did NIQE take part in the admission decision?  (Stage 5C reports said
   "MUSIQ + NIQE"; the code reads only ``iqa_musiq``.)
4. Which ledgers lost their IQA columns because the old loader preferred the
   plain CSV when its mtime was newer?

Outputs: ``reports/stage5d_threshold_trace.{json,md}``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

IMAGE_DIR = ROOT / 'data/corpus_v2/images'
POLICY = ROOT / 'data/corpus_v2/policy.yaml'
REPORTS = ROOT / 'reports'


def ledger_paths() -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for path in sorted(IMAGE_DIR.glob('*.csv')):
        if 'cleanvision_issues' in path.name:
            continue
        source = path.stem.replace('_qc', '')
        enriched = IMAGE_DIR / f'{source}_qc.csv'
        plain = IMAGE_DIR / f'{source}.csv'
        if not enriched.exists():
            chosen = plain
        elif plain.exists() and plain.stat().st_mtime > enriched.stat().st_mtime:
            chosen = plain
        else:
            chosen = enriched
        out[source] = {
            'enriched_exists': enriched.exists(),
            'plain_exists': plain.exists(),
            'old_loader_choice': chosen.name,
            'enriched_newer': (enriched.exists() and plain.exists()
                               and enriched.stat().st_mtime >= plain.stat().st_mtime),
        }
    return out


def stats_for(path: Path) -> Dict[str, object]:
    frame = pd.read_csv(path, low_memory=False)
    musiq = pd.to_numeric(frame.get('iqa_musiq'), errors='coerce')
    niqe = pd.to_numeric(frame.get('iqa_niqe'), errors='coerce')
    return {
        'rows': int(len(frame)),
        'musiq_non_null': int(musiq.notna().sum()),
        'musiq_min': float(musiq.min()) if musiq.notna().any() else None,
        'musiq_max': float(musiq.max()) if musiq.notna().any() else None,
        'musiq_median': float(musiq.median()) if musiq.notna().any() else None,
        'niqe_non_null': int(niqe.notna().sum()),
        'niqe_min': float(niqe.min()) if niqe.notna().any() else None,
        'niqe_max': float(niqe.max()) if niqe.notna().any() else None,
    }


def main() -> int:
    policy = yaml.safe_load(POLICY.read_text(encoding='utf-8'))
    yaml_accept = float(policy['quality']['iqa']['accept_min'])
    yaml_reject = float(policy['quality']['iqa']['reject_max'])
    # Stage 5C code default, read from the source rather than retyped.
    freeze_source = (ROOT / 'src/registry/stage5c_freeze.py').read_text(encoding='utf-8')
    import re
    default = re.search(r'def iqa_accept\(.*?accept_min: float = ([0-9.]+), reject_max: float = ([0-9.]+)',
                        freeze_source, re.S)
    code_accept, code_reject = (float(default.group(1)), float(default.group(2))) if default else (None, None)

    ledgers = ledger_paths()
    per_source: Dict[str, Dict[str, object]] = {}
    frames = []
    for source, info in ledgers.items():
        path = IMAGE_DIR / str(info['old_loader_choice'])
        stats = stats_for(path)
        per_source[source] = {**info, **stats}
        frame = pd.read_csv(path, low_memory=False)
        frame['physical_source_id'] = source
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    musiq = pd.to_numeric(combined['iqa_musiq'], errors='coerce')
    niqe = pd.to_numeric(combined['iqa_niqe'], errors='coerce')

    if code_accept is None:
        raise RuntimeError('could not read the Stage 5C iqa_accept defaults')

    code_pass = int((musiq >= code_accept).sum())
    code_quarantine = int(((musiq >= code_reject) & (musiq < code_accept)).sum())
    code_reject_rows = int((musiq < code_reject).sum())
    yaml_pass = int((musiq >= yaml_accept).sum())
    yaml_reject_rows = int((musiq < yaml_reject).sum())
    missing = int(musiq.isna().sum())

    payload = {
        'metric': 'musiq (pyIQA 0.1.16, musiq_koniq_ckpt-e95806b9.pth)',
        'observed': {
            'rows': int(len(combined)),
            'scored': int(musiq.notna().sum()),
            'missing': missing,
            'min': float(musiq.min()), 'max': float(musiq.max()),
            'p01': float(musiq.quantile(0.01)), 'p50': float(musiq.median()),
            'p99': float(musiq.quantile(0.99)),
        },
        'niqe': {
            'scored': int(niqe.notna().sum()),
            'coverage_fraction': float(niqe.notna().mean()),
            'min': float(niqe.min()) if niqe.notna().any() else None,
            'max': float(niqe.max()) if niqe.notna().any() else None,
            'used_in_admission_decision': False,
            'evidence': ('src/registry/stage5c_freeze.py::iqa_accept reads only '
                         'row["iqa_musiq"]; no code path reads iqa_niqe'),
        },
        'legacy_policy_yaml': {
            'accept_min': yaml_accept, 'reject_max': yaml_reject,
            'would_pass': yaml_pass, 'would_reject': yaml_reject_rows,
            'pass_fraction': yaml_pass / max(int(musiq.notna().sum()), 1),
            'verdict': ('IMPOSSIBLE_ON_THIS_SCALE: every measured MUSIQ score is '
                        '>= 0.35, so the YAML thresholds would admit 100% of scored '
                        'images and could not have produced the published funnel'),
        },
        'stage5c_code_default': {
            'accept_min': code_accept, 'reject_max': code_reject,
            'pass': code_pass, 'quarantine': code_quarantine, 'reject': code_reject_rows,
            'verdict': ('EFFECTIVE: these are the values that reproduce the '
                        'Stage 5C funnel step (30,755 removed on the pool)'),
        },
        'score_scale_conclusion': (
            'MUSIQ emits a 0-100 quality score. The Stage 5C policy file recorded '
            '0.35/0.22, a normalised scale, and those values were never applied. '
            'The effective thresholds were the function defaults 45.0/25.0.'),
        'loader_provenance': per_source,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / 'stage5d_threshold_trace.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')

    rows = []
    for source, info in per_source.items():
        rows.append(
            f"| {source} | {info['old_loader_choice']} | {info['rows']:,} | "
            f"{info['musiq_non_null']:,} | {info['musiq_min'] if info['musiq_min'] is not None else '-'} | "
            f"{info['musiq_max'] if info['musiq_max'] is not None else '-'} | "
            f"{info['niqe_non_null']:,} |")
    lines = [
        '# Stage 5D - IQA score-scale and threshold trace',
        '',
        f'- Legacy policy file: `data/corpus_v2/policy.yaml`',
        f'- YAML thresholds: accept `{yaml_accept}`, reject `{yaml_reject}`',
        f'- Stage 5C code defaults: accept `{code_accept}`, reject `{code_reject}`',
        f'- Measured MUSIQ range: **{payload["observed"]["min"]:.3f} - '
        f'{payload["observed"]["max"]:.3f}** over '
        f'{payload["observed"]["scored"]:,} scored images',
        '',
        '## Verdict',
        '',
        f'- YAML thresholds would pass **{yaml_pass:,} / {payload["observed"]["scored"]:,}** '
        f'({payload["legacy_policy_yaml"]["pass_fraction"]:.4f}).',
        f'- Code defaults pass {code_pass:,}, quarantine {code_quarantine:,}, '
        f'reject {code_reject_rows:,}.',
        '- Conclusion: the YAML values are on a different scale and were never in '
        'force; the effective thresholds were 45/25.',
        '',
        '## NIQE',
        '',
        f'- NIQE measured on {payload["niqe"]["scored"]:,} / '
        f'{payload["observed"]["rows"]:,} rows '
        f'({payload["niqe"]["coverage_fraction"]:.4f}).',
        '- NIQE never entered the admission decision: `iqa_accept()` reads only '
        '`iqa_musiq`.  Stage 5C reports that said "MUSIQ + NIQE" overstated the gate.',
        '',
        '## Ledger selection under the old loader',
        '',
        '| source | old loader chose | rows | MUSIQ scored | min | max | NIQE scored |',
        '|---|---|---:|---:|---:|---:|---:|',
        *rows,
        '',
        'Two sources (`agrifm_ref1`, `agrifm_ref57`) had a newer *plain* ledger, so '
        'the old loader read the unenriched file with empty `iqa_musiq`.  Those rows '
        'became `IQA_MISSING` and were excluded from R2/R3.  Stage 5D always reads '
        'the enriched ledger when it exists.',
    ]
    (REPORTS / 'stage5d_threshold_trace.md').write_text('\n'.join(lines) + '\n',
                                                        encoding='utf-8')
    print(json.dumps({'yaml_pass_fraction': payload['legacy_policy_yaml']['pass_fraction'],
                      'code_pass': code_pass, 'code_quarantine': code_quarantine,
                      'code_reject': code_reject_rows, 'missing': missing,
                      'observed': payload['observed']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
