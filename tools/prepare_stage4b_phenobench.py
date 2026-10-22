#!/usr/bin/env python
"""Link the already-verified Stage 4A reference evaluations into Stage 4B.

B_500, C1_500 and C2_500 are the same checkpoints with the same evaluation
protocol; recomputing them would add ~40 minutes of GPU time and, worse, would
create a second copy of the same numbers that could drift.  This tool creates
symlinks (never copies, never edits the Stage 4A tree) so the Stage 4B
comparison reads exactly one source of truth.

Usage:
    ./.venv/bin/python -m tools.prepare_stage4b_phenobench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE_LABELS = ('B_500', 'C1_500', 'C2_500', 'C3_500', 'B_2000', 'A_0')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/stage4a_phenobench/results')
    parser.add_argument('--target', default='runs/stage4b_phenobench/results')
    args = parser.parse_args()
    source = Path(args.source)
    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    linked = []
    for label in REFERENCE_LABELS:
        src = (source / label).resolve()
        if not (src / 'result.json').is_file():
            print(f'[{label}] missing in {source}, not linked')
            continue
        dst = target / label
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and dst.resolve() == src:
                linked.append(label)
                continue
            raise SystemExit(f'{dst} exists and is not the expected symlink; '
                             f'refusing to overwrite')
        dst.symlink_to(src, target_is_directory=True)
        linked.append(label)
        print(f'[{label}] -> {src}')

    payload = {
        'source': str(source),
        'target': str(target),
        'linked': linked,
        'note': ('symlinks to the Stage 4A evaluations; the numbers are not '
                 'copied, so there is a single source of truth'),
    }
    (target / 'reference_links.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
