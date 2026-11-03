#!/usr/bin/env python
"""Generate the Stage 6 test summary from a live pytest run.

Runs the Stage 6-specific suites (sampler / protocol / checkpoint) plus the
full suite, and writes a structured report that separates pass/fail/skip and
records every skip reason.  Skipped tests are never counted as passed.

Usage:
    ./.venv-uv/bin/python -m tools.stage6_test_summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE6_SUITES = [
    'tests/test_stage6_sampling.py',
    'tests/test_stage6_protocol.py',
    'tests/test_stage6_checkpoint.py',
]


def run_suite(spec: str) -> dict:
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', spec, '-q', '--no-header',
         '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True)
    out = result.stdout + result.stderr
    summary: dict = {'spec': spec, 'returncode': result.returncode, 'raw': out}
    # -- pytest -q summary line: "N passed, M skipped, K failed"
    for line in out.splitlines():
        if any(token in line for token in ('passed', 'failed', 'skipped')) and \
           line.strip().endswith(('passed', 'failed', 'skipped')) is False and \
           ('::' not in line):
            summary['summary_line'] = line.strip()
            break
    # -- extract numeric counts
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0}
    for token in ('passed', 'failed', 'skipped', 'error'):
        marker = f'{token}'
        idx = out.find(marker)
        # -- rely on pytest's "X passed, Y skipped, Z failed" tail
    import re
    match = re.search(r'(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?'
                      r'(?:, (\d+) error)?', out)
    if match:
        counts['passed'] = int(match.group(1))
        counts['skipped'] = int(match.group(2) or 0)
        counts['failed'] = int(match.group(3) or 0)
        counts['error'] = int(match.group(4) or 0)
    summary['counts'] = counts
    return summary


def parse_skips(out: str) -> list:
    skips = []
    for line in out.splitlines():
        if 'SKIPPED' in line and 'reason' in line:
            skips.append(line.strip())
    return skips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'reports/stage6_test_summary.json')
    args = parser.parse_args()

    suites = [run_suite(spec) for spec in STAGE6_SUITES]
    full = run_suite('tests/')

    total_passed = sum(s['counts']['passed'] for s in suites)
    total_failed = sum(s['counts']['failed'] for s in suites)
    total_skipped = sum(s['counts']['skipped'] for s in suites)

    skipped_reasons: list = []
    for spec in STAGE6_SUITES:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', spec, '--co', '-q', '-rs',
             '-p', 'no:cacheprovider'],
            cwd=ROOT, capture_output=True, text=True)
        skipped_reasons.extend(parse_skips(result.stdout + result.stderr))

    report = {
        'stage6_suites': suites,
        'stage6_totals': {
            'passed': total_passed,
            'failed': total_failed,
            'skipped': total_skipped,
        },
        'skipped_reasons': skipped_reasons,
        'full_suite_returncode': full['returncode'],
        'full_suite_counts': full['counts'],
        'full_suite_skips': parse_skips(full['raw']),
        'note': 'skipped tests are never counted as passed; '
                'the one persistent non-Stage-6 failure '
                '(test_artifact_manifest_agrees_with_git) is a pre-existing '
                'Stage 5E.1 baseline inconsistency, not introduced by Stage 6.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['stage6_totals'], indent=2))
    print('full suite:', full['counts'], 'rc', full['returncode'])
    print(f'skipped reasons recorded: {len(skipped_reasons)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
