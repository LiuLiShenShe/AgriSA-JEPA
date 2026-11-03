#!/usr/bin/env python
"""Run the CleanVision + pyIQA enrichment for every finished ledger.

A ledger is considered finished when its ``<source>.summary.json`` exists and
the ``.csv`` is older than the summary.  Each source is processed in a fresh
subprocess so GPU memory is released between runs.  Existing ``_qc.csv`` files
newer than the ledger are left alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / 'data/corpus_v2/images'
PYTHON = ROOT / '.venv-uv/bin/python'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', default='ALL')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--sample-large', type=int, default=400)
    parser.add_argument('--sample-small', type=int, default=1000)
    parser.add_argument('--large-threshold', type=int, default=10000)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.sources.strip().upper() == 'ALL':
        sources = sorted(p.stem for p in IMAGES.glob('*.csv')
                         if not p.name.endswith('_qc.csv')
                         and p.name != 'cleanvision_issues.csv')
    else:
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]

    done, skipped, failed = [], [], []
    for source in sources:
        ledger = IMAGES / f'{source}.csv'
        enriched = IMAGES / f'{source}_qc.csv'
        summary = IMAGES / f'{source}.summary.json'
        if not ledger.exists():
            skipped.append((source, 'no ledger'))
            continue
        if not summary.exists():
            skipped.append((source, 'summary missing (ledger may be running)'))
            continue
        if ledger.stat().st_mtime > summary.stat().st_mtime:
            skipped.append((source, 'ledger newer than summary'))
            continue
        if (enriched.exists() and not args.force
                and enriched.stat().st_mtime >= ledger.stat().st_mtime):
            skipped.append((source, 'already enriched'))
            continue
        import csv as _csv
        with open(ledger, newline='', encoding='utf-8') as handle:
            rows = sum(1 for _ in _csv.DictReader(handle))
        sample = args.sample_large if rows > args.large_threshold else args.sample_small
        started = time.time()
        command = [str(PYTHON), str(ROOT / 'tools/stage5c_run_qc.py'),
                   '--source', source, '--paths-csv', str(ledger),
                   '--out', str(IMAGES / f'{source}_qc.csv'),
                   '--sample', str(sample), '--device', args.device,
                   '--iqa-metrics', 'musiq', '--max-side', '512']
        print(f'== QC {source} ({rows} rows, cleanvision sample {sample})', flush=True)
        result = subprocess.run(command, cwd=str(ROOT))
        if result.returncode == 0:
            done.append((source, rows, round(time.time() - started, 1)))
        else:
            failed.append((source, result.returncode))
            print(f'!! QC failed for {source}', flush=True)

    print(json.dumps({'done': done, 'skipped': skipped, 'failed': failed}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
