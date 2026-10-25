#!/usr/bin/env python
"""Record the pytest result for Stage 5E.1 in ``reports/stage5e1_pytest.json``.

Reads the pytest short-summary output (default ``/tmp/stage5e1_pytest.txt``) and
writes the passed / failed / skipped counts plus every skip reason.  A skip is
never counted as a pass.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='/tmp/stage5e1_pytest.txt')
    args = parser.parse_args()

    text = Path(args.input).read_text() if Path(args.input).exists() else ''
    counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
    match = re.search(r'(\d+) passed', text)
    if match:
        counts['passed'] = int(match.group(1))
    match = re.search(r'(\d+) failed', text)
    if match:
        counts['failed'] = int(match.group(1))
    match = re.search(r'(\d+) skipped', text)
    if match:
        counts['skipped'] = int(match.group(1))
    match = re.search(r'(\d+) error', text)
    if match:
        counts['errors'] = int(match.group(1))

    skip_reasons: dict = {}
    for line in text.splitlines():
        hit = re.match(r'SKIPPED \[\d+\] \S+:\d+: (.*)', line.strip())
        if hit:
            reason = hit.group(1).strip()
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    if not text:
        payload = {'note': 'pytest output not found; run .venv-uv/bin/python -m pytest tests/ -q'}
    else:
        payload = {**counts, 'skip_reasons': skip_reasons,
                   'rule': 'skipped is never counted as passed',
                   'command': '.venv-uv/bin/python -m pytest tests/ -q'}
    (REPORTS / 'stage5e1_pytest.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
