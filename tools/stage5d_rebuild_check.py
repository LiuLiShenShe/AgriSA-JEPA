#!/usr/bin/env python
"""Second independent build, byte-comparison and gate-10 evidence.

Runs ``tools/stage5d_build_corpus.py`` into a separate output directory with the
same inputs, then compares the resulting train/val CSV SHA-256 with the first
build's manifest.  Writes ``reports/stage5d_rebuild_check.json`` which
``tools/stage5d_final_report.py`` consumes for gate 10.

Usage
-----
    .venv-uv/bin/python tools/stage5d_rebuild_check.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
REPORTS = ROOT / 'reports'
REBUILD = ROOT / 'data/corpus_v3_rebuild'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    first = json.loads((ROOT / 'data/corpus_v3/R3v2_strict_rgb_manifest.json').read_text())
    if REBUILD.exists():
        shutil.rmtree(REBUILD)
    REBUILD.mkdir(parents=True)
    (REBUILD / 'policy.yaml').write_bytes((ROOT / 'data/corpus_v3/policy.yaml').read_bytes())
    # The build writes into <out>/images and <out>/R3v2_*.csv.  Derived IQA is
    # read from data/corpus_v3/derived_iqa, so both builds see identical input.
    env = dict(os.environ)
    env.setdefault('AGRI_DATA_ROOT', '/data/datasets/Stage5C')
    result = subprocess.run(
        [PY, '-I', 'tools/stage5d_build_corpus.py', '--out', 'data/corpus_v3_rebuild',
         '--reports', 'data/corpus_v3_rebuild/reports'],
        cwd=ROOT, capture_output=True, text=True, env=env)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit(f'rebuild failed with {result.returncode}')

    second = json.loads((REBUILD / 'R3v2_strict_rgb_manifest.json').read_text())
    payload = {
        'first': {'train_csv_sha256': first['train_csv_sha256'],
                  'val_csv_sha256': first['val_csv_sha256'],
                  'accepted': first['counts']['accepted']},
        'second': {'train_csv_sha256': second['train_csv_sha256'],
                   'val_csv_sha256': second['val_csv_sha256'],
                   'accepted': second['counts']['accepted']},
        'train_csv_sha256': second['train_csv_sha256'],
        'val_csv_sha256': second['val_csv_sha256'],
        'identical': (first['train_csv_sha256'] == second['train_csv_sha256']
                      and first['val_csv_sha256'] == second['val_csv_sha256']
                      and first['counts']['accepted'] == second['counts']['accepted']),
    }
    (REPORTS / 'stage5d_rebuild_check.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
