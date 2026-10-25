#!/usr/bin/env python
"""Stage 5E rebuild determinism check.

Rebuilds R3v3 from the same ledger into a scratch directory and compares the
two manifests and CSVs byte for byte.  Emits a single JSON object so
``tools/stage5e_gates.py`` can consume it.

    {"byte_identical": true, ...}
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    'R3v3_license_verified_train.csv',
    'R3v3_license_verified_val.csv',
    'R3v3_license_verified_manifest.json',
    'R3v3_license_verified_sampling_profiles.json',
    'R3v3_license_verified_composition.json',
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_dir = ROOT / 'data/corpus_v4'
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch) / 'corpus_v4'
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'tools/stage5e_build_corpus.py'),
             '--ledger', 'data/corpus_v3/images', '--out', str(scratch_dir)],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(json.dumps({'byte_identical': False,
                              'error': f'rebuild failed: {proc.stderr[-500:]}'}, indent=2))
            return 0
        comparison = {}
        identical = True
        for name in TARGETS:
            original = source_dir / name
            rebuilt = scratch_dir / name
            if not original.exists() or not rebuilt.exists():
                comparison[name] = {'identical': False, 'reason': 'missing file'}
                identical = False
                continue
            a, b = sha256_file(original), sha256_file(rebuilt)
            same = a == b
            identical = identical and same
            comparison[name] = {'identical': same, 'original_sha256': a, 'rebuilt_sha256': b}
    result = {
        'byte_identical': identical,
        'files_compared': len(TARGETS),
        'comparison': comparison,
        'note': 'rebuild runs the same tool on the same ledger into a scratch directory',
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
