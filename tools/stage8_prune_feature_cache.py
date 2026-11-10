#!/usr/bin/env python
"""Stage 8 Phase 4 support: prune fully-consumed CropAndWeed feature caches.

The CropAndWeed train+eval feature cache is ~102 GB per backbone and lives on
the NVMe holding ``runs/``, which has limited free space; keeping all seven
backbones' caches would need ~720 GB.  Once every registry row (primary and
exploratory) for a backbone has produced its ``result.json``, the cache for
that backbone can no longer influence any pending result, so it is deleted to
make room for the next backbone's extraction.

Safety rules (hard errors, never warnings):

* a backbone with any pending row keeps its cache;
* a cache is only deleted after its ``cache_manifest.json`` label matches the
  registry label and (for Stage 7B backbones) its recorded checkpoint SHA-256
  matches the Stage 7B manifest — the same binding the probe itself enforces
  via ``verify_cache_manifest``;
* only directories directly under ``runs/stage8_cropandweed/cache/`` matching
  ``<label>__<sha16>`` are ever considered.

Called automatically by ``tools/stage8_phase4_runner.py`` after every
successful cropandweed row, and runnable standalone for inspection with
``--dry-run`` (default: actually delete).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / 'reports'
CACHE_ROOT = ROOT / 'runs' / 'stage8_cropandweed' / 'cache'

from src.downstream.features import cache_key
from tools.stage8_backbones import (EXPLORATORY, PRIMARY, build_registry,
                                    recorded_checkpoint_sha256)


def _registry_rows() -> list:
    reg = json.loads((R / 'stage8_experiment_registry.json').read_text('utf-8'))
    return [r for r in reg['rows'] if r['task_id'] == 'cropandweed_sem']


def _has_result(out_dir: str) -> bool:
    out = Path(out_dir)
    return any((out / marker).is_file() for marker in ('result.json',
                                                       'results.json'))


def _find_cache_dir(spec) -> Path:
    """Locate this backbone's cache directory by its unique label prefix."""
    matches = sorted(CACHE_ROOT.glob(f'{spec.label}__*'))
    if len(matches) > 1:
        raise SystemExit(f'ambiguous caches for {spec.label}: {matches}')
    return matches[0] if matches else None


def prune_for_backbone(backbone: str, dry_run: bool = False) -> bool:
    """Delete the cache for ``backbone`` iff all its rows are complete.

    Returns True if a cache existed and was (or would be) deleted.
    """
    reg = build_registry(include_exploratory=True)
    if backbone not in reg:
        raise SystemExit(f'unknown backbone {backbone}')
    spec = reg[backbone]
    rows = [r for r in _registry_rows() if r['backbone'] == backbone]
    if not rows:
        return False
    pending = [r['output_dir'] for r in rows if not _has_result(r['output_dir'])]
    cache_dir = _find_cache_dir(spec)
    if cache_dir is None:
        return False
    if pending:
        print(f'  prune: {backbone} has {len(pending)} pending row(s), '
              f'keeping {cache_dir}', flush=True)
        return False
    manifest = json.loads((cache_dir / 'cache_manifest.json').read_text('utf-8'))
    if manifest.get('label') != spec.label:
        raise SystemExit(f'refusing: {cache_dir} manifest label '
                         f'{manifest.get("label")} != {spec.label}')
    recorded = recorded_checkpoint_sha256()
    if backbone in recorded:
        expected = recorded[backbone]
        if manifest.get('checkpoint_sha256') != expected:
            raise SystemExit(f'refusing: {cache_dir} manifest sha '
                             f'{manifest.get("checkpoint_sha256")} != '
                             f'stage7b manifest {expected}')
        if cache_dir.name != cache_key(spec.label, expected):
            raise SystemExit(f'refusing: {cache_dir.name} != expected '
                             f'{cache_key(spec.label, expected)}')
    if dry_run:
        print(f'  prune (dry-run): would delete {cache_dir}', flush=True)
    else:
        shutil.rmtree(cache_dir)
        print(f'  prune: deleted {cache_dir} (all {len(rows)} rows complete)',
              flush=True)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbones', default=None,
                        help='comma-separated subset; default = all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    keys = (args.backbones.split(',') if args.backbones
            else list(PRIMARY) + list(EXPLORATORY))
    for backbone in keys:
        prune_for_backbone(backbone, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
