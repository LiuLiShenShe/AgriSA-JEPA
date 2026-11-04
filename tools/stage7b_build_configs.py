#!/usr/bin/env python
"""Stage 7B: build the 3-arm x 2-seed paired schedule-ablation configs.

Base: ``configs/stage6_b1_natural_2000.yaml`` (the frozen B1 configuration).
Only schedule fields change; data membership, sampling, augmentation, mask,
model, optimizer and all frozen inputs are inherited unchanged.

Arms (all run exactly 2,000 optimizer updates from the official I-JEPA init):

==========  =================  ==========================  ============
Arm         max_optimizer_steps  interrupt_after            warmup_steps
==========  =================  ==========================  ============
A           2,000              null                        100
B           50,000             2,000                       100
C           50,000             2,000                       2,500
==========  =================  ==========================  ============

A's schedule completes at 2,000; B/C keep a 50,000 horizon and stop at 2,000,
so LR/WD/EMA sit at the 2,000 position of a 50k schedule.  B vs C isolates
warmup length; A vs B isolates horizon length at equal warmup.

Seeds: P1=20261104, P2=20261105.  Within a seed all three arms share the same
data.seed/augmentation_seed (hence identical data order, masks and model init);
only the schedule differs.

Usage::

    ./.venv-uv/bin/python -m tools.stage7b_build_configs --write
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / 'configs/stage6_b1_natural_2000.yaml'
RUN_ROOT = Path('/data/agrisa_jepa_runs/stage7b')

ARMS = {
    'A': {'max_optimizer_steps': 2000, 'interrupt_after_optimizer_steps': None,
          'warmup_steps': 100, 'save_at_optimizer_steps': [2000]},
    'B': {'max_optimizer_steps': 50000, 'interrupt_after_optimizer_steps': 2000,
          'warmup_steps': 100, 'save_at_optimizer_steps': [2000]},
    'C': {'max_optimizer_steps': 50000, 'interrupt_after_optimizer_steps': 2000,
          'warmup_steps': 2500, 'save_at_optimizer_steps': [2000]},
}
SEEDS = {'p1': 20261104, 'p2': 20261105}
EXPECTED_OPTIMIZER_STEPS_PER_EPOCH = 1151   # ceil(9204 / 8)
EPOCHS = 50                                  # covers the 50k horizon
PRETRAINED_SHA = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'
MANIFEST_SHA = '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2'

#: fields the builder is allowed to touch relative to the frozen base config
ALLOWED_CHANGES = {
    ('meta', 'stage'), ('meta', 'run_id'), ('meta', 'pair_id'), ('meta', 'phase'),
    ('meta', 'load_checkpoint'), ('meta', 'read_checkpoint'),
    ('data', 'seed'), ('data', 'augmentation_seed'),
    ('optimization', 'max_optimizer_steps'),
    ('optimization', 'interrupt_after_optimizer_steps'),
    ('optimization', 'warmup_steps'),
    ('optimization', 'save_at_optimizer_steps'),
    ('optimization', 'epochs'),
    ('logging', 'folder'),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def flatten(cfg: Mapping[str, object], prefix: Sequence[str] = ()) -> Dict[tuple, object]:
    out: Dict[tuple, object] = {}
    for key, value in cfg.items():
        path = tuple(prefix) + (key,)
        if isinstance(value, Mapping):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def run_id(arm: str, tag: str) -> str:
    return f'stage7b_{arm}_{tag}'


def build_one(arm: str, tag: str, base: Mapping[str, object]) -> Dict[str, object]:
    cfg = copy.deepcopy(base)
    spec = ARMS[arm]
    seed = SEEDS[tag]
    cfg['meta']['stage'] = 'stage7b_schedule_ablation'
    cfg['meta']['run_id'] = run_id(arm, tag)
    cfg['meta']['pair_id'] = tag.upper()
    cfg['meta']['phase'] = f'arm_{arm}'
    cfg['meta']['load_checkpoint'] = False
    cfg['meta']['read_checkpoint'] = None
    cfg['data']['seed'] = seed
    cfg['data']['augmentation_seed'] = seed
    cfg['optimization']['max_optimizer_steps'] = spec['max_optimizer_steps']
    cfg['optimization']['interrupt_after_optimizer_steps'] = spec['interrupt_after_optimizer_steps']
    cfg['optimization']['warmup_steps'] = spec['warmup_steps']
    cfg['optimization']['save_at_optimizer_steps'] = list(spec['save_at_optimizer_steps'])
    cfg['optimization']['epochs'] = EPOCHS
    cfg['logging']['folder'] = str(RUN_ROOT / run_id(arm, tag))
    return cfg


def assert_only_allowed_changes(base: Mapping[str, object],
                                cfg: Mapping[str, object]) -> None:
    flat_base, flat_cfg = flatten(base), flatten(cfg)
    for path in set(flat_base) | set(flat_cfg):
        if flat_base.get(path) != flat_cfg.get(path):
            if path not in ALLOWED_CHANGES:
                raise AssertionError(f'illegal change at {path}: '
                                     f'{flat_base.get(path)!r} -> {flat_cfg.get(path)!r}')


def validate(cfg: Mapping[str, object], arm: str, tag: str) -> List[str]:
    problems: List[str] = []
    opt = cfg['optimization']
    spec = ARMS[arm]
    if opt['max_optimizer_steps'] != spec['max_optimizer_steps']:
        problems.append(f'{arm}/{tag}: max_optimizer_steps')
    if opt['interrupt_after_optimizer_steps'] != spec['interrupt_after_optimizer_steps']:
        problems.append(f'{arm}/{tag}: interrupt')
    if int(opt['warmup_steps']) != spec['warmup_steps']:
        problems.append(f'{arm}/{tag}: warmup_steps')
    if cfg['data']['seed'] != SEEDS[tag] or cfg['data']['augmentation_seed'] != SEEDS[tag]:
        problems.append(f'{arm}/{tag}: seed mismatch')
    if cfg['meta'].get('pretrained_sha256') != PRETRAINED_SHA:
        problems.append(f'{arm}/{tag}: pretrained SHA')
    if cfg['data'].get('manifest_sha256') != MANIFEST_SHA:
        problems.append(f'{arm}/{tag}: manifest SHA')
    if cfg['data'].get('batch_size') != 16:
        problems.append(f'{arm}/{tag}: batch_size')
    if int(opt.get('grad_accum', 0)) != 8:
        problems.append(f'{arm}/{tag}: grad_accum')
    if cfg['meta'].get('load_checkpoint'):
        problems.append(f'{arm}/{tag}: must not load a checkpoint')
    return problems


def build_plan() -> Dict[str, object]:
    base = load(BASE_CONFIG)
    configs: Dict[str, Dict[str, object]] = {}
    problems: List[str] = []
    for arm in ARMS:
        for tag, seed in SEEDS.items():
            key = f'{arm}_{tag}'
            cfg = build_one(arm, tag, base)
            assert_only_allowed_changes(base, cfg)
            problems.extend(validate(cfg, arm, tag))
            configs[key] = cfg
    epochs_min = -(-max(arm['max_optimizer_steps'] for arm in ARMS.values())
                   // EXPECTED_OPTIMIZER_STEPS_PER_EPOCH)
    return {
        'schema_version': 'stage7b_config_plan/v1',
        'base_config': str(BASE_CONFIG.relative_to(ROOT)),
        'base_config_sha256': sha256_file(BASE_CONFIG),
        'run_root': str(RUN_ROOT),
        'arms': ARMS,
        'seeds': SEEDS,
        'epochs': EPOCHS,
        'epochs_minimum': epochs_min,
        'expected_optimizer_steps_per_epoch': EXPECTED_OPTIMIZER_STEPS_PER_EPOCH,
        'runs': list(configs),
        'configs': configs,
        'problems': problems,
    }


def write_configs(plan: Mapping[str, object]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for key, cfg in plan['configs'].items():
        path = ROOT / 'configs' / f'stage7b_{key}.yaml'
        path.write_text(
            '# Stage 7B paired schedule ablation -- generated by\n'
            '# tools/stage7b_build_configs.py.  Do not hand-edit; regenerate.\n'
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True,
                             default_flow_style=False), encoding='utf-8')
        paths[key] = path
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    plan = build_plan()
    if plan['problems']:
        print(json.dumps({'problems': plan['problems']}, indent=2))
        return 1
    payload = {k: v for k, v in plan.items() if k != 'configs'}
    if args.write:
        paths = write_configs(plan)
        payload['config_paths'] = {k: str(v.relative_to(ROOT)) for k, v in paths.items()}
        (ROOT / 'reports/stage7b_config_plan.json').write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
