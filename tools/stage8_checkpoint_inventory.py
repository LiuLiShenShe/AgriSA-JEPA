#!/usr/bin/env python
"""Stage 8 Phase 1: emit the frozen checkpoint inventory.

For every registered Stage 8 backbone (Official + A/B primary + C exploratory),
record: key, arm, pretraining seed, checkpoint path, file size, the SHA-256
recorded in the Stage 7B probe manifest (for the six continued-pretraining
encoders) or the known official SHA, and a live SHA computation of the file.

The inventory is the single reference Phase 2 protocol and every downstream
run points at: a run that pins checkpoint key ``A_p1`` is pinned to this exact
file hash.

Output ``reports/stage8_checkpoint_inventory.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.stage8_backbones import (build_registry, recorded_checkpoint_sha256,
                                    OFFICIAL_SHA)

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'reports'


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    reg = build_registry(include_exploratory=True)
    recorded = recorded_checkpoint_sha256()

    rows = []
    for key in reg:
        spec = reg[key]
        path = Path(spec.checkpoint)
        if not path.is_file():
            rows.append({'key': key, 'arm': spec.arm, 'exists': False,
                         'error': f'missing:{path}'})
            continue
        actual = sha256_file(path)
        expected = recorded.get(key) if key != 'OFFICIAL' else OFFICIAL_SHA
        rows.append({
            'key': key,
            'arm': spec.arm,
            'pretraining_seed': spec.pretraining_seed,
            'role': spec.role,
            'checkpoint': str(path),
            'size_bytes': path.stat().st_size,
            'recorded_sha256': expected,
            'live_sha256': actual,
            'match': actual == expected,
        })

    n_ok = sum(1 for r in rows if r.get('exists') and r.get('match'))
    payload = {
        'schema_version': 'stage8_checkpoint_inventory/v1',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'sha256_algorithm': 'sha256 (full file)',
        'official_sha_source': 'official I-JEPA ViT-H/14 300e release',
        'stage7b_manifest': 'reports/stage7b_phenobench_manifest.json',
        'checkpoints': rows,
        'summary': {'n_checkpoints': len(rows),
                    'n_present_and_matching': n_ok},
    }
    R.mkdir(exist_ok=True)
    (R / 'stage8_checkpoint_inventory.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(json.dumps(payload['summary'], indent=2))
    return 0 if n_ok == len(rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())