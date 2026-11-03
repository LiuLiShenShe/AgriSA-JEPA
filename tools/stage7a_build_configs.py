#!/usr/bin/env python
"""Generate the Stage 7A training-budget configs from the frozen Stage 6 B1 config.

Stage 7A answers one question: how does agricultural continued-pretraining value
change with the training budget, and is another 30k updates worth the compute?
It keeps the Stage 6 B1 protocol byte-for-byte identical and changes ONLY the
fields that define a budget experiment:

* training budget          max_optimizer_steps : 2,000 -> 50,000
* schedule length          epochs              : 24    -> 50
* warmup length            warmup_steps        : 100   -> 2,500 (5% of 50k)
* mid-run snapshots        save_at_optimizer_steps : [500,1000,1500,2000] -> [10000,20000,50000]
* phase interruption       interrupt_after_optimizer_steps : (absent) -> 20,000
* paired pretraining seed  data.seed / data.augmentation_seed -> 20261029
* run identity + output folder

Everything else -- model, predictor, official initialisation checkpoint, data
manifest, natural sampling, official random multi-block masks, batch size,
gradient accumulation, LR/WD/EMA values and augmentation -- must stay equal to
the frozen Stage 6 B1 base.  ``assert_only_allowed_changes`` fails otherwise.

Two configs are emitted, both on the SAME 50,000-update schedule:

* ``stage7a_b1_natural_50k.yaml``          -- phase 1: init from the official
  checkpoint, train to 20,000 updates, save 10k/20k, interrupt (schedule
  horizon stays 50k).
* ``stage7a_b1_natural_50k_resume.yaml``   -- phase 2: resume from the 20k full
  checkpoint, continue to 50,000.  It differs from phase 1 only in
  load_checkpoint / read_checkpoint / removal of the interruption point.

The two checkpoints are therefore intermediate states of one 50k schedule, not
two models trained on separate 10k/20k schedules.

Usage::

    ./.venv-uv/bin/python -m tools.stage7a_build_configs            # write
    ./.venv-uv/bin/python -m tools.stage7a_build_configs --check    # verify
    ./.venv-uv/bin/python -m tools.stage7a_build_configs --print-plan
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

#: The single frozen base every Stage 7A field is derived from.
BASE_CONFIG: Path = ROOT / 'configs/stage6_b1_natural_2000.yaml'

STAGE_NAME = 'stage7a_20k_checkpoint_curve'
RUN_ID = 'stage7a_b1_natural_50k'
PAIR_ID = 'S7A'
PRETRAINING_SEED = 20261029
RUN_ROOT = '/data/agrisa_jepa_runs/stage7a_b1_natural_50k'

MAX_OPTIMIZER_STEPS = 50000
INTERRUPT_AFTER = 20000
SAVE_AT_STEPS = [10000, 20000, 50000]
WARMUP_STEPS = 2500

#: Measured Stage 6 B1 value (147,277 rows, batch 16, drop_last, grad_accum 8).
#: 147277 // 16 = 9204 micro-batches/epoch; ceil(9204 / 8) = 1151 updates/epoch.
EXPECTED_OPTIMIZER_STEPS_PER_EPOCH = 1151
#: epochs >= ceil(50000 / updates_per_epoch) = ceil(43.44) = 44.  50 leaves
#: margin so a small change in emitted batch count can never end the run early.
EPOCHS = 50

#: Seeds already used by Stages 6 / 6D -- the new trajectory must be distinct.
HISTORICAL_SEEDS = (20261026, 20261027, 20261028)

STAGE1_CONFIG = ROOT / 'configs' / f'{RUN_ID}.yaml'
STAGE2_CONFIG = ROOT / 'configs' / f'{RUN_ID}_resume.yaml'

#: Paths allowed to differ from the frozen Stage 6 B1 base (phase 1).
ALLOWED_CHANGES: set = {
    ('meta', 'stage'),
    ('meta', 'run_id'),
    ('meta', 'pair_id'),
    ('meta', 'phase'),
    ('data', 'seed'),
    ('data', 'augmentation_seed'),
    ('optimization', 'max_optimizer_steps'),
    ('optimization', 'epochs'),
    ('optimization', 'warmup_steps'),
    ('optimization', 'save_at_optimizer_steps'),
    ('optimization', 'interrupt_after_optimizer_steps'),
    ('logging', 'folder'),
}

#: Phase 2 (resume) relative to phase 1: only these may differ.
ALLOWED_RESUME_CHANGES: set = {
    ('meta', 'phase'),
    ('meta', 'load_checkpoint'),
    ('meta', 'read_checkpoint'),
    ('optimization', 'interrupt_after_optimizer_steps'),
}

STAGE2_READ_CHECKPOINT = 'jepa-step20000.pth.tar'


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


def epochs_needed(max_steps: int = MAX_OPTIMIZER_STEPS,
                  updates_per_epoch: int = EXPECTED_OPTIMIZER_STEPS_PER_EPOCH) -> int:
    """Minimum epoch count that covers an exact update budget (ceiling)."""
    if updates_per_epoch < 1:
        raise ValueError('updates_per_epoch must be >= 1')
    return -(-int(max_steps) // int(updates_per_epoch))


def _base() -> Dict[str, object]:
    with BASE_CONFIG.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def build_stage1() -> Dict[str, object]:
    """Phase-1 config: 50k schedule, snapshots at 10k/20k, stop at 20k."""
    cfg = copy.deepcopy(_base())
    cfg['meta']['stage'] = STAGE_NAME
    cfg['meta']['run_id'] = RUN_ID
    cfg['meta']['pair_id'] = PAIR_ID
    cfg['meta']['phase'] = 'stage1_to_20k'
    cfg['data']['seed'] = PRETRAINING_SEED
    cfg['data']['augmentation_seed'] = PRETRAINING_SEED
    cfg['optimization']['max_optimizer_steps'] = MAX_OPTIMIZER_STEPS
    cfg['optimization']['epochs'] = EPOCHS
    cfg['optimization']['warmup_steps'] = WARMUP_STEPS
    cfg['optimization']['save_at_optimizer_steps'] = list(SAVE_AT_STEPS)
    cfg['optimization']['interrupt_after_optimizer_steps'] = INTERRUPT_AFTER
    cfg['logging']['folder'] = RUN_ROOT
    return cfg


def build_stage2() -> Dict[str, object]:
    """Phase-2 config: resume the 20k full checkpoint, continue to 50k."""
    cfg = copy.deepcopy(build_stage1())
    cfg['meta']['phase'] = 'stage2_to_50k'
    cfg['meta']['load_checkpoint'] = True
    cfg['meta']['read_checkpoint'] = STAGE2_READ_CHECKPOINT
    cfg['optimization']['interrupt_after_optimizer_steps'] = None
    return cfg


def assert_only_allowed_changes(base: Mapping[str, object],
                                cfg: Mapping[str, object],
                                allowed: set,
                                label: str) -> List[str]:
    """Return violations of the frozen-protocol rule (empty means clean)."""
    flat_base = flatten(base)
    flat_cfg = flatten(cfg)
    problems: List[str] = []
    for key in sorted(set(flat_base) | set(flat_cfg)):
        before = flat_base.get(key, '<missing>')
        after = flat_cfg.get(key, '<missing>')
        if before == after:
            continue
        if key not in allowed:
            problems.append(f'{label}: {"/".join(key)} changed {before!r} -> {after!r}')
    return problems


def validate_config(cfg: Mapping[str, object], label: str) -> List[str]:
    """Semantic checks the flat diff cannot express."""
    problems: List[str] = []
    opt = cfg['optimization']
    if opt['max_optimizer_steps'] != MAX_OPTIMIZER_STEPS:
        problems.append(f'{label}: max_optimizer_steps {opt["max_optimizer_steps"]} '
                        f'!= {MAX_OPTIMIZER_STEPS}')
    if opt['warmup_steps'] != WARMUP_STEPS:
        problems.append(f'{label}: warmup_steps {opt["warmup_steps"]} != {WARMUP_STEPS}')
    if opt['warmup'] != 0.05:
        problems.append(f'{label}: warmup {opt["warmup"]} != 0.05')
    if list(opt['save_at_optimizer_steps']) != SAVE_AT_STEPS:
        problems.append(f'{label}: save_at_optimizer_steps '
                        f'{opt["save_at_optimizer_steps"]} != {SAVE_AT_STEPS}')
    minimum = epochs_needed()
    if opt['epochs'] < minimum:
        problems.append(f'{label}: epochs {opt["epochs"]} < minimum {minimum} '
                        f'needed to cover {MAX_OPTIMIZER_STEPS} updates')
    if cfg['data']['seed'] != PRETRAINING_SEED:
        problems.append(f'{label}: data.seed {cfg["data"]["seed"]} != {PRETRAINING_SEED}')
    if cfg['data']['augmentation_seed'] != PRETRAINING_SEED:
        problems.append(f'{label}: data.augmentation_seed '
                        f'{cfg["data"]["augmentation_seed"]} != {PRETRAINING_SEED}')
    if PRETRAINING_SEED in HISTORICAL_SEEDS:
        problems.append(f'{label}: seed {PRETRAINING_SEED} collides with a historical seed')
    if (cfg['data'].get('sampling') or {}).get('mode') != 'natural':
        problems.append(f'{label}: data.sampling.mode is not natural')
    if cfg['mask'].get('mode') != 'random':
        problems.append(f'{label}: mask.mode is not the official random collator')
    if int(opt['grad_accum']) != 8 or int(cfg['data']['batch_size']) != 16:
        problems.append(f'{label}: effective batch is not 16 x 8 = 128')
    return problems


def stage1_resume_diff(stage1: Mapping[str, object],
                       stage2: Mapping[str, object]) -> List[str]:
    return assert_only_allowed_changes(stage1, stage2,
                                       ALLOWED_RESUME_CHANGES, 'stage1->stage2')


def build_plan() -> Dict[str, object]:
    base = _base()
    stage1 = build_stage1()
    stage2 = build_stage2()

    problems: List[str] = []
    problems.extend(assert_only_allowed_changes(base, stage1, ALLOWED_CHANGES, 'B1'))
    problems.extend(validate_config(stage1, 'B1'))
    problems.extend(validate_config(stage2, 'B1-resume'))
    problems.extend(stage1_resume_diff(stage1, stage2))

    # phase 2 must keep the schedule horizon and NOT reintroduce an interruption
    if stage2['optimization'].get('interrupt_after_optimizer_steps') is not None:
        problems.append('stage2 must not interrupt before the 50k budget')
    if stage2['optimization']['max_optimizer_steps'] != MAX_OPTIMIZER_STEPS:
        problems.append('stage2 changed the 50k schedule horizon')
    if stage2['meta']['read_checkpoint'] != STAGE2_READ_CHECKPOINT:
        problems.append('stage2 read_checkpoint is not the 20k full checkpoint')

    return {
        'base_config': str(BASE_CONFIG.relative_to(ROOT)),
        'base_config_sha256': sha256_file(BASE_CONFIG),
        'stage1_config': str(STAGE1_CONFIG.relative_to(ROOT)),
        'stage2_config': str(STAGE2_CONFIG.relative_to(ROOT)),
        'run_id': RUN_ID,
        'pair_id': PAIR_ID,
        'pretraining_seed': PRETRAINING_SEED,
        'max_optimizer_steps': MAX_OPTIMIZER_STEPS,
        'interrupt_after_optimizer_steps': INTERRUPT_AFTER,
        'save_at_optimizer_steps': list(SAVE_AT_STEPS),
        'warmup_steps': WARMUP_STEPS,
        'epochs': EPOCHS,
        'epochs_minimum': epochs_needed(),
        'optimizer_steps_per_epoch': EXPECTED_OPTIMIZER_STEPS_PER_EPOCH,
        'output_dir': RUN_ROOT,
        'stage1': stage1,
        'stage2': stage2,
        'problems': problems,
        'ok': not problems,
    }


def _header(phase: str) -> str:
    base = BASE_CONFIG.relative_to(ROOT)
    return (
        '# Stage 7A -- generated by tools/stage7a_build_configs.py\n'
        f'# phase={phase} run_id={RUN_ID} seed={PRETRAINING_SEED}\n'
        f'# Derived mechanically from {base}.  The 50,000-update schedule is the\n'
        '# common horizon; 10k and 20k are mid-run snapshots of THIS trajectory,\n'
        '# not separately scheduled models.  Do not hand-edit: regenerate.\n'
    )


def _dump(cfg: Mapping[str, object]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


def write_configs(plan: Mapping[str, object]) -> List[Path]:
    STAGE1_CONFIG.write_text(_header('stage1_to_20k') + _dump(plan['stage1']),
                             encoding='utf-8')
    STAGE2_CONFIG.write_text(_header('stage2_to_50k') + _dump(plan['stage2']),
                             encoding='utf-8')
    return [STAGE1_CONFIG, STAGE2_CONFIG]


def plan_payload(plan: Mapping[str, object]) -> Dict[str, object]:
    return {
        'schema_version': 'stage7a_config_plan/v1',
        'base_config': plan['base_config'],
        'base_config_sha256': plan['base_config_sha256'],
        'stage1_config': plan['stage1_config'],
        'stage1_config_sha256': (sha256_file(STAGE1_CONFIG)
                                 if STAGE1_CONFIG.is_file() else None),
        'stage2_config': plan['stage2_config'],
        'stage2_config_sha256': (sha256_file(STAGE2_CONFIG)
                                 if STAGE2_CONFIG.is_file() else None),
        'run_id': plan['run_id'],
        'pair_id': plan['pair_id'],
        'pretraining_seed': plan['pretraining_seed'],
        'max_optimizer_steps': plan['max_optimizer_steps'],
        'interrupt_after_optimizer_steps': plan['interrupt_after_optimizer_steps'],
        'save_at_optimizer_steps': plan['save_at_optimizer_steps'],
        'warmup_steps': plan['warmup_steps'],
        'epochs': plan['epochs'],
        'epochs_minimum': plan['epochs_minimum'],
        'optimizer_steps_per_epoch': plan['optimizer_steps_per_epoch'],
        'output_dir': plan['output_dir'],
        'problems': plan['problems'],
        'ok': plan['ok'],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='verify the plan and existing configs without writing')
    parser.add_argument('--print-plan', action='store_true')
    parser.add_argument('--plan-out', type=Path,
                        default=ROOT / 'reports/stage7a_config_plan.json')
    args = parser.parse_args(argv)

    plan = build_plan()
    if not plan['ok']:
        print('REFUSING TO GENERATE:')
        for problem in plan['problems']:
            print(f'  - {problem}')
        return 1

    if args.check:
        stale: List[str] = []
        expected = {
            STAGE1_CONFIG: _header('stage1_to_20k') + _dump(plan['stage1']),
            STAGE2_CONFIG: _header('stage2_to_50k') + _dump(plan['stage2']),
        }
        for path, text in expected.items():
            if not path.is_file():
                stale.append(f'missing {path.relative_to(ROOT)}')
            elif path.read_text(encoding='utf-8') != text:
                stale.append(f'stale {path.relative_to(ROOT)}')
        if stale:
            print('CONFIGS OUT OF DATE:')
            for problem in stale:
                print(f'  - {problem}')
            return 1
        print('configs match the plan')
        return 0

    written = write_configs(plan)
    payload = plan_payload(plan)
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for path in written:
        print(f'wrote {path.relative_to(ROOT)}')
    print(f'wrote {args.plan_out.relative_to(ROOT)}')
    if args.print_plan:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
