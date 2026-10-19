#!/usr/bin/env python
"""Load an arm config YAML and resolve it into CLI arguments.

Keeps ``configs/phenobench_arm{A,B}_linear.yaml`` as the single source of truth
for the downstream protocol instead of hard-coding values in two places.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import yaml


def load_arm_config(path: str | Path) -> Dict[str, object]:
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = ('meta', 'data', 'tiling', 'probe', 'evaluation', 'compute', 'outputs')
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f'Config {path} missing sections {missing}')
    if cfg['meta'].get('arm') not in ('A', 'B'):
        raise ValueError(f"Config {path} must set meta.arm to 'A' or 'B'")
    return cfg


def config_to_argv(cfg: Dict[str, object]) -> Sequence[str]:
    """Translate an arm config into ``phenobench_probe`` CLI arguments."""
    argv = [
        '--arm', str(cfg['meta']['arm']),
        '--root', str(cfg['data']['root']),
        '--cache-dir', str(cfg['outputs']['cache_dir']),
        '--out-dir', str(cfg['outputs']['out_dir']),
        '--device', str(cfg['compute']['device']),
        '--feature-batch-size', str(cfg['compute']['feature_batch_size']),
        '--epochs', str(cfg['probe']['epochs']),
        '--lr', str(cfg['probe']['lr']),
        '--batch-size', str(cfg['probe']['batch_size']),
        '--weight-decay', str(cfg['probe']['weight_decay']),
        '--seed', str(cfg['probe']['seeds'][0]),
    ]
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['features', 'train_eval', 'all'],
                        default='all')
    args = parser.parse_args()
    cfg = load_arm_config(args.config)
    argv = list(config_to_argv(cfg)) + ['--stage', args.stage]
    print(' '.join(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())