#!/usr/bin/env python
"""Generate the Stage 6D pretraining configs from the frozen Stage 6 configs.

Stage 6D asks one question: does the Stage 6C positive-transfer signal repeat
under independent pretraining seeds?  The training protocol must therefore stay
byte-for-byte identical to Stage 6 except for

* the paired pretraining seed (``data.seed`` / ``data.augmentation_seed`` /
  ``data.sampling.seed`` for the capped arm),
* run identity (``meta.run_id`` / ``meta.pair_id`` / ``meta.pretraining_seed``),
* the output folder (new runs are written to ``/data`` because the root
  filesystem cannot hold them).

This tool derives the four new configs mechanically and refuses to emit a config
that differs from its Stage 6 base anywhere else.  It is deliberately a pure
function of the seed table below -- no "pick a seed that looks good" step.

Usage::

    ./.venv/bin/python -m tools.stage6d_build_configs            # write configs
    ./.venv/bin/python -m tools.stage6d_build_configs --check    # verify only
    ./.venv/bin/python -m tools.stage6d_build_configs --print-plan
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_SEED = 20261026
#: Paired pretraining seeds for the new fully traceable runs.  Chosen only to
#: be distinct from the historical 20261026 and from each other; they are not
#: selected after seeing any result.
PAIRS: Tuple[Dict[str, object], ...] = (
    {'pair_id': 'P1', 'seed': 20261027},
    {'pair_id': 'P2', 'seed': 20261028},
)

ARMS = ('B1', 'B2')
BASE_CONFIGS: Dict[str, Path] = {
    'B1': ROOT / 'configs/stage6_b1_natural_2000.yaml',
    'B2': ROOT / 'configs/stage6_b2_capped_no_backfill_2000.yaml',
}
RUN_ROOT = '/data/agrisa_jepa_runs/stage6d'
STAGE_NAME = 'stage6d_pretrain_seed_replication'

#: Paths allowed to differ from the Stage 6 base config.  Anything else is a
#: protocol change and this tool fails.
ALLOWED_CHANGES: Dict[str, set] = {
    'B1': {
        ('meta', 'stage'),
        ('meta', 'run_id'),
        ('meta', 'pair_id'),
        ('meta', 'pretraining_seed'),
        ('data', 'seed'),
        ('data', 'augmentation_seed'),
        ('logging', 'folder'),
    },
    'B2': {
        ('meta', 'stage'),
        ('meta', 'run_id'),
        ('meta', 'pair_id'),
        ('meta', 'pretraining_seed'),
        ('data', 'seed'),
        ('data', 'augmentation_seed'),
        ('data', 'sampling', 'seed'),
        ('logging', 'folder'),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(node, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], object]:
    out: Dict[Tuple[str, ...], object] = {}
    if isinstance(node, Mapping):
        for key, value in node.items():
            out.update(flatten(value, prefix + (str(key),)))
    else:
        out[prefix] = node
    return out


def run_id_for(arm: str, seed: int) -> str:
    if arm == 'B1':
        return f'stage6d_b1_natural_seed{seed}'
    if arm == 'B2':
        return f'stage6d_b2_capped_no_backfill_seed{seed}'
    raise ValueError(f'unknown arm {arm!r}')


def config_path_for(arm: str, seed: int) -> Path:
    return ROOT / 'configs' / f'{run_id_for(arm, seed)}.yaml'


def build_one(arm: str, seed: int, pair_id: str) -> Dict[str, object]:
    """Return a deep-copied config with only the documented fields changed."""
    with BASE_CONFIGS[arm].open('r', encoding='utf-8') as handle:
        base = yaml.safe_load(handle)
    cfg = copy.deepcopy(base)

    run_id = run_id_for(arm, seed)
    cfg['meta']['stage'] = STAGE_NAME
    cfg['meta']['run_id'] = run_id
    cfg['meta']['pair_id'] = pair_id
    cfg['meta']['pretraining_seed'] = int(seed)
    cfg['data']['seed'] = int(seed)
    cfg['data']['augmentation_seed'] = int(seed)
    cfg['logging']['folder'] = f'{RUN_ROOT}/{run_id}'
    if arm == 'B2':
        cfg['data']['sampling']['seed'] = int(seed)

    return cfg


def assert_only_allowed_changes(arm: str, base: Mapping[str, object],
                                cfg: Mapping[str, object]) -> List[str]:
    """Return violations of the frozen-protocol rule (empty means clean)."""
    flat_base = flatten(base)
    flat_cfg = flatten(cfg)
    problems: List[str] = []
    for key in sorted(set(flat_base) | set(flat_cfg)):
        before = flat_base.get(key, '<missing>')
        after = flat_cfg.get(key, '<missing>')
        if before == after:
            continue
        if key not in ALLOWED_CHANGES[arm]:
            problems.append(f'{arm}: {"/".join(key)} changed {before!r} -> {after!r}')
    return problems


def build_plan() -> Dict[str, object]:
    runs: List[Dict[str, object]] = []
    problems: List[str] = []
    seen_run_ids: set = set()
    seen_seeds: set = set()
    for pair in PAIRS:
        seed = int(pair['seed'])
        pair_id = str(pair['pair_id'])
        if seed in seen_seeds:
            problems.append(f'duplicate seed {seed}')
        if seed == HISTORICAL_SEED:
            problems.append(f'seed {seed} collides with the historical seed')
        seen_seeds.add(seed)
        for arm in ARMS:
            run_id = run_id_for(arm, seed)
            if run_id in seen_run_ids:
                problems.append(f'duplicate run_id {run_id}')
            seen_run_ids.add(run_id)
            cfg = build_one(arm, seed, pair_id)
            with BASE_CONFIGS[arm].open('r', encoding='utf-8') as handle:
                base = yaml.safe_load(handle)
            problems.extend(assert_only_allowed_changes(arm, base, cfg))
            runs.append({
                'pair_id': pair_id,
                'seed': seed,
                'arm': arm,
                'run_id': run_id,
                'config': str(config_path_for(arm, seed).relative_to(ROOT)),
                'base_config': str(BASE_CONFIGS[arm].relative_to(ROOT)),
                'base_config_sha256': sha256_file(BASE_CONFIGS[arm]),
                'output_dir': cfg['logging']['folder'],
                'config_payload': cfg,
            })

    # paired seeds must match across arms
    by_pair: Dict[str, Dict[str, int]] = {}
    for run in runs:
        by_pair.setdefault(str(run['pair_id']), {})[str(run['arm'])] = int(run['seed'])
    for pair_id, seeds in by_pair.items():
        if len(set(seeds.values())) != 1:
            problems.append(f'pair {pair_id}: seeds differ across arms: {seeds}')

    return {'runs': runs, 'problems': problems, 'ok': not problems}


def _header(arm: str, seed: int, pair_id: str, run_id: str) -> str:
    base = BASE_CONFIGS[arm].relative_to(ROOT)
    return (
        '# Stage 6D -- generated by tools/stage6d_build_configs.py\n'
        f'# arm={arm} pair={pair_id} pretraining_seed={seed} run_id={run_id}\n'
        f'# Derived mechanically from {base}; only seed, run identity and\n'
        '# logging.folder differ.  Do not hand-edit: regenerate instead.\n'
        '# B1/B2 still differ ONLY in sampling policy and arm identity.\n'
    )


def write_configs(plan: Dict[str, object]) -> List[Path]:
    written: List[Path] = []
    for run in plan['runs']:
        arm = str(run['arm'])
        seed = int(run['seed'])
        pair_id = str(run['pair_id'])
        run_id = str(run['run_id'])
        path = config_path_for(arm, seed)
        header = _header(arm, seed, pair_id, run_id)
        body = yaml.safe_dump(run['config_payload'], sort_keys=False,
                              allow_unicode=True, default_flow_style=False)
        path.write_text(header + body, encoding='utf-8')
        written.append(path)
    return written


def plan_payload(plan: Dict[str, object]) -> Dict[str, object]:
    """Serializable plan (drops the full config payload, keeps SHAs)."""
    runs = []
    for run in plan['runs']:
        cfg = run['config_payload']
        path = config_path_for(str(run['arm']), int(run['seed']))
        runs.append({
            'pair_id': run['pair_id'],
            'seed': run['seed'],
            'arm': run['arm'],
            'run_id': run['run_id'],
            'config': run['config'],
            'base_config': run['base_config'],
            'base_config_sha256': run['base_config_sha256'],
            'output_dir': run['output_dir'],
            'config_sha256': (sha256_file(path) if path.is_file() else None),
            'meta_pretraining_seed': cfg['meta']['pretraining_seed'],
            'data_seed': cfg['data']['seed'],
            'augmentation_seed': cfg['data']['augmentation_seed'],
            'sampling_seed': ((cfg['data'].get('sampling') or {}).get('seed')),
        })
    return {
        'schema_version': 'stage6d_config_plan/v1',
        'historical_seed': HISTORICAL_SEED,
        'pairs': [dict(pair) for pair in PAIRS],
        'runs': runs,
        'problems': plan['problems'],
        'ok': plan['ok'],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='verify the plan and existing configs without writing')
    parser.add_argument('--print-plan', action='store_true',
                        help='print the plan JSON')
    parser.add_argument('--plan-out', type=Path,
                        default=ROOT / 'reports/stage6d_config_plan.json')
    args = parser.parse_args(argv)

    plan = build_plan()
    if not plan['ok']:
        print('REFUSING TO GENERATE:')
        for problem in plan['problems']:
            print(f'  - {problem}')
        return 1

    if not args.check:
        written = write_configs(plan)
        for path in written:
            print(f'wrote {path.relative_to(ROOT)}')

    if args.check:
        stale = []
        for run in plan['runs']:
            path = config_path_for(str(run['arm']), int(run['seed']))
            if not path.is_file():
                stale.append(f'missing {path.relative_to(ROOT)}')
                continue
            expected = _header(str(run['arm']), int(run['seed']),
                               str(run['pair_id']), str(run['run_id'])) + \
                yaml.safe_dump(run['config_payload'], sort_keys=False,
                               allow_unicode=True, default_flow_style=False)
            if path.read_text(encoding='utf-8') != expected:
                stale.append(f'stale {path.relative_to(ROOT)}')
        if stale:
            print('CONFIGS OUT OF DATE:')
            for problem in stale:
                print(f'  - {problem}')
            return 1
        print('configs match the plan')
        return 0

    payload = plan_payload(plan)
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'wrote {args.plan_out.relative_to(ROOT)}')
    if args.print_plan:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
