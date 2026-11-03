#!/usr/bin/env python
"""Sharded CleanVision + pyIQA enrichment for very large sources.

One process decoding 130k images is I/O-bound and wastes the GPU.  This driver
splits a ledger into contiguous chunks, runs one subprocess per chunk (all on
the same GPU), then merges the per-chunk scores back into the full ledger.  No
row is dropped: unscored rows keep empty IQA fields and are therefore excluded
from R2/R3 by the IQA gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / 'data/corpus_v2/images'
PYTHON = ROOT / '.venv-uv/bin/python'
TMP = ROOT / 'data/corpus_v2/work/qc_shards'


def read_csv(path: Path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--iqa-metrics', nargs='*', default=['musiq', 'niqe'])
    parser.add_argument('--cleanvision-sample', type=int, default=400)
    parser.add_argument('--max-side', type=int, default=512)
    args = parser.parse_args()

    ledger = IMAGES / f'{args.source}.csv'
    rows = read_csv(ledger)
    fields = list(rows[0].keys()) if rows else []
    path_key = 'raw_path' if rows and 'raw_path' in rows[0] else 'path'
    decoded = [r for r in rows if str(r.get('decode_ok')).lower() in ('1', 'true')]
    print(f'{args.source}: {len(rows)} rows, {len(decoded)} decoded', flush=True)

    TMP.mkdir(parents=True, exist_ok=True)
    started = time.time()
    chunk_paths, procs = [], []
    per = (len(decoded) + args.shards - 1) // args.shards
    for index in range(args.shards):
        chunk = decoded[index * per:(index + 1) * per]
        if not chunk:
            continue
        chunk_csv = TMP / f'{args.source}.part{index}.csv'
        write_csv(chunk_csv, chunk, fields)
        out_csv = TMP / f'{args.source}.part{index}.qc.csv'
        command = [str(PYTHON), str(ROOT / 'tools/stage5c_run_qc.py'),
                   '--source', args.source, '--paths-csv', str(chunk_csv),
                   '--out', str(out_csv), '--sample', '0',
                   '--device', args.device,
                   '--iqa-metrics', *args.iqa_metrics,
                   '--max-side', str(args.max_side)]
        log = open(TMP / f'{args.source}.part{index}.log', 'w')
        procs.append((subprocess.Popen(command, cwd=str(ROOT), stdout=log,
                                       stderr=subprocess.STDOUT), out_csv, log))
        chunk_paths.append(chunk_csv)

    failed = []
    for index, (proc, out_csv, log) in enumerate(procs):
        code = proc.wait()
        log.close()
        if code != 0 or not out_csv.exists():
            failed.append((index, code))
            print(f'!! shard {index} failed ({code})', flush=True)
    print(f'shards finished in {time.time() - started:.0f}s, failed={failed}', flush=True)

    # Merge scores by raw path.
    scored = {}
    for index in range(args.shards):
        out_csv = TMP / f'{args.source}.part{index}.qc.csv'
        if not out_csv.exists():
            continue
        for row in read_csv(out_csv):
            scored[str(row.get(path_key))] = row
    for row in rows:
        found = scored.get(str(row.get(path_key)))
        if found:
            for key in ('iqa_musiq', 'iqa_niqe'):
                if found.get(key) not in (None, ''):
                    row[key] = found[key]
        row.setdefault('iqa_musiq', '')
        row.setdefault('iqa_niqe', '')
        row.setdefault('cleanvision_flags', '')

    # One cleanvision sample over the whole source (diagnostic only).
    if args.cleanvision_sample > 0:
        sample_csv = TMP / f'{args.source}.cleanvision.csv'
        stride = max(1, len(decoded) // args.cleanvision_sample)
        write_csv(sample_csv, decoded[::stride][:args.cleanvision_sample], fields)
        out_csv = TMP / f'{args.source}.cleanvision.qc.csv'
        command = [str(PYTHON), str(ROOT / 'tools/stage5c_run_qc.py'),
                   '--source', args.source, '--paths-csv', str(sample_csv),
                   '--out', str(out_csv), '--sample', str(args.cleanvision_sample),
                   '--device', args.device, '--skip-iqa']
        subprocess.run(command, cwd=str(ROOT))
        if out_csv.exists():
            flags = {str(r.get(path_key)): r.get('cleanvision_flags', '')
                     for r in read_csv(out_csv)}
            for row in rows:
                if str(row.get(path_key)) in flags:
                    row['cleanvision_flags'] = flags[str(row.get(path_key))]

    write_csv(IMAGES / f'{args.source}_qc.csv', rows, fields)
    provenance = {
        'source_id': args.source, 'rows': len(rows), 'decoded': len(decoded),
        'shards': args.shards, 'failed_shards': failed,
        'iqa_metrics': args.iqa_metrics,
        'scored': sum(1 for r in rows if str(r.get('iqa_musiq', '')).strip()),
        'seconds': round(time.time() - started, 1),
    }
    (TMP / f'{args.source}.sharded.json').write_text(
        json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
