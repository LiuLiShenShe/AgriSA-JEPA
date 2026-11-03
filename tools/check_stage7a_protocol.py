#!/usr/bin/env python
"""Verify the Stage 7A configs against the frozen Stage 6 B1 protocol.

The Stage 7A experiment may change ONLY the budget-defining fields:

* ``max_optimizer_steps`` / ``epochs`` / ``warmup_steps`` / ``save_at_optimizer_steps``
  / ``interrupt_after_optimizer_steps`` (the schedule and its snapshots),
* ``data.seed`` / ``data.augmentation_seed`` (a fresh, non-historical seed),
* run identity and ``logging.folder``.

Everything else -- model, predictor, official initialisation checkpoint, data
manifest, natural sampling, official random multi-block masks, batch size,
gradient accumulation, LR/WD/EMA values and augmentation -- must be unchanged.

This checker also verifies the SCHEDULER HORIZON: because ``init_opt`` derives
LR, WD and EMA from ``total_steps``, the 50k budget must be the common horizon
of all three.  It mirrors the arithmetic in ``src/helper.py`` and
``src/utils/schedulers.py`` so a config drift that silently reschedules the run
is caught before it costs a day of GPU time.

Usage::

    ./.venv-uv/bin/python -m tools.check_stage7a_protocol
    ./.venv-uv/bin/python -m tools.check_stage7a_protocol --json reports/stage7a_protocol.json
    ./.venv-uv/bin/python -m tools.check_stage7a_protocol --md reports/stage7a_protocol.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import yaml

from tools.stage7a_build_configs import (
    ALLOWED_CHANGES,
    ALLOWED_RESUME_CHANGES,
    BASE_CONFIG,
    EPOCHS,
    EXPECTED_OPTIMIZER_STEPS_PER_EPOCH,
    HISTORICAL_SEEDS,
    INTERRUPT_AFTER,
    MAX_OPTIMIZER_STEPS,
    PAIR_ID,
    PRETRAINING_SEED,
    ROOT,
    RUN_ID,
    RUN_ROOT,
    SAVE_AT_STEPS,
    STAGE1_CONFIG,
    STAGE2_CONFIG,
    STAGE2_READ_CHECKPOINT,
    STAGE_NAME,
    WARMUP_STEPS,
    assert_only_allowed_changes,
    build_plan,
    epochs_needed,
    flatten,
    sha256_file,
    validate_config,
)

#: Fields that must equal the frozen Stage 6 B1 values in both configs.
FROZEN: Dict[Tuple[str, ...], object] = {
    ('data', 'dataset'): 'agrifield40k',
    ('data', 'manifest'): 'data/stage6/stage6_r3v3_train_view_nvme.csv',
    ('data', 'manifest_sha256'): '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2',
    ('data', 'split'): 'train',
    ('data', 'batch_size'): 16,
    ('data', 'num_workers'): 0,
    ('data', 'image_cache_dir'): None,
    ('data', 'max_samples'): None,
    ('data', 'crop_size'): 224,
    ('data', 'crop_scale'): [0.3, 1.0],
    ('data', 'use_color_distortion'): False,
    ('data', 'use_gaussian_blur'): False,
    ('data', 'use_horizontal_flip'): False,
    ('data', 'color_jitter_strength'): 0.0,
    ('data', 'pin_mem'): True,
    ('data', 'record_source_exposure'): True,
    ('data', 'sampling', 'mode'): 'natural',
    ('meta', 'arm'): 'B1',
    ('meta', 'arm_label'): 'natural',
    ('meta', 'pretrained_checkpoint'): '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
    ('meta', 'pretrained_sha256'): '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22',
    ('meta', 'load_checkpoint'): False,
    ('meta', 'read_checkpoint'): None,
    ('meta', 'model_name'): 'vit_huge',
    ('meta', 'patch_size'): 14,
    ('meta', 'pred_depth'): 12,
    ('meta', 'pred_emb_dim'): 384,
    ('meta', 'use_bfloat16'): True,
    ('meta', 'use_sdpa'): True,
    ('meta', 'activation_checkpointing'): False,
    ('meta', 'copy_data'): False,
    ('optimization', 'grad_accum'): 8,
    ('optimization', 'lr'): 0.0002,
    ('optimization', 'start_lr'): 1.0e-05,
    ('optimization', 'final_lr'): 1.0e-06,
    ('optimization', 'weight_decay'): 0.04,
    ('optimization', 'final_weight_decay'): 0.4,
    ('optimization', 'ema'): [0.996, 1.0],
    ('optimization', 'ipe_scale'): 1.0,
    ('optimization', 'stop_after_epoch'): None,
    ('mask', 'mode'): 'random',
    ('mask', 'allow_overlap'): False,
    ('mask', 'aspect_ratio'): [0.75, 1.5],
    ('mask', 'enc_mask_scale'): [0.85, 1.0],
    ('mask', 'pred_mask_scale'): [0.15, 0.2],
    ('mask', 'num_enc_masks'): 1,
    ('mask', 'num_pred_masks'): 4,
    ('mask', 'min_keep'): 10,
    ('mask', 'patch_size'): 14,
}

FROZEN_FILES = {
    'train_csv': ('data/corpus_v4/R3v3_license_verified_train.csv',
                  '9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08'),
    'val_csv': ('data/corpus_v4/R3v3_license_verified_val.csv',
                'f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686'),
    'sampling_profile': ('data/corpus_v4/R3v3_license_verified_sampling_profiles.json',
                         '91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505'),
    'train_view': ('data/stage6/stage6_r3v3_train_view_nvme.csv',
                   '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2'),
}


def check_frozen(flat: Mapping[Tuple[str, ...], object], label: str) -> List[str]:
    problems: List[str] = []
    for key, expected in FROZEN.items():
        actual = flat.get(key, '<missing>')
        if actual != expected:
            problems.append(f'{label}: frozen field {".".join(key)} = {actual!r}, '
                            f'expected {expected!r}')
    return problems


def check_budget(flat: Mapping[Tuple[str, ...], object], label: str) -> List[str]:
    problems: List[str] = []
    if flat.get(('optimization', 'max_optimizer_steps')) != MAX_OPTIMIZER_STEPS:
        problems.append(f'{label}: max_optimizer_steps != {MAX_OPTIMIZER_STEPS}')
    if flat.get(('optimization', 'interrupt_after_optimizer_steps')) != INTERRUPT_AFTER:
        problems.append(f'{label}: interrupt_after_optimizer_steps != {INTERRUPT_AFTER}')
    if flat.get(('optimization', 'warmup_steps')) != WARMUP_STEPS:
        problems.append(f'{label}: warmup_steps != {WARMUP_STEPS}')
    if flat.get(('optimization', 'epochs')) < epochs_needed():
        problems.append(f'{label}: epochs {flat.get(("optimization", "epochs"))} '
                        f'< minimum {epochs_needed()}')
    return problems


def scheduler_horizon(cfg: Mapping[str, object]) -> Dict[str, object]:
    """Mirror init_opt: LR/WD/EMA all decay over the exact update budget."""
    opt = cfg['optimization']
    horizon = int(opt.get('max_optimizer_steps')
                  or int(opt['ipe_scale'] * opt['epochs']
                         * EXPECTED_OPTIMIZER_STEPS_PER_EPOCH))
    warmup_steps = int(opt.get('warmup_steps')
                       or int(opt['warmup'] * EXPECTED_OPTIMIZER_STEPS_PER_EPOCH))
    return {
        'horizon': horizon,
        'warmup_steps': warmup_steps,
        'lr_T_max': max(0, horizon - warmup_steps),   # WarmupCosineSchedule
        'wd_T_max': horizon,                          # CosineWDSchedule
        'ema_steps': horizon,                         # ema[0]->ema[1] over horizon
        'lr_reaches_final_at': horizon,
        'wd_reaches_final_at': horizon,
        'ema_reaches_1_at': horizon,
        'consistent': (horizon == MAX_OPTIMIZER_STEPS
                       and warmup_steps == WARMUP_STEPS),
    }


def check_frozen_files(hash_large_pretrained: bool = False) -> Dict[str, object]:
    report: Dict[str, object] = {'files': {}, 'problems': []}
    for name, (rel, expected) in FROZEN_FILES.items():
        path = ROOT / rel
        if not path.is_file():
            report['problems'].append(f'{name}: missing {rel}')
            continue
        actual = sha256_file(path)
        ok = actual == expected
        report['files'][name] = {'path': rel, 'expected_sha256': expected,
                                 'actual_sha256': actual, 'ok': ok}
        if not ok:
            report['problems'].append(f'{name}: SHA {actual} != {expected}')
    if hash_large_pretrained:
        ckpt = Path('/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar')
        expected = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'
        actual = sha256_file(ckpt) if ckpt.is_file() else 'missing'
        report['files']['pretrained'] = {
            'path': str(ckpt), 'expected_sha256': expected,
            'actual_sha256': actual, 'ok': actual == expected}
        if actual != expected:
            report['problems'].append(f'pretrained: SHA {actual} != {expected}')
    return report


def run_checks(hash_large_pretrained: bool = False) -> Dict[str, object]:
    plan = build_plan()
    problems: List[str] = list(plan['problems'])

    stage1_flat = flatten(plan['stage1'])
    stage2_flat = flatten(plan['stage2'])
    problems.extend(check_frozen(stage1_flat, 'B1'))
    problems.extend(check_budget(stage1_flat, 'B1'))
    problems.extend(validate_config(plan['stage1'], 'B1'))

    # stage2: frozen fields identical, budget identical, only resume fields differ.
    # meta.load_checkpoint / meta.read_checkpoint are exactly the resume fields,
    # validated separately through ALLOWED_RESUME_CHANGES, so they are exempt.
    RESUME_EXEMPT = {('meta', 'load_checkpoint'), ('meta', 'read_checkpoint')}
    for key, expected in FROZEN.items():
        if key in RESUME_EXEMPT:
            continue
        if stage2_flat.get(key, '<missing>') != expected:
            problems.append(f'B1-resume: frozen field {".".join(key)} = '
                            f'{stage2_flat.get(key, "<missing>")!r}, expected {expected!r}')
    for key in (('optimization', 'max_optimizer_steps'),
                ('optimization', 'warmup_steps'),
                ('optimization', 'save_at_optimizer_steps'),
                ('optimization', 'ema'),
                ('optimization', 'lr'),
                ('optimization', 'start_lr'),
                ('optimization', 'final_lr'),
                ('optimization', 'weight_decay'),
                ('optimization', 'final_weight_decay'),
                ('optimization', 'grad_accum')):
        if stage2_flat.get(key) != stage1_flat.get(key):
            problems.append(f'B1-resume: {".".join(key)} changed across phases')
    resume_diff = assert_only_allowed_changes(plan['stage1'], plan['stage2'],
                                              ALLOWED_RESUME_CHANGES, 'stage1->stage2')
    problems.extend(resume_diff)

    # run identity
    if plan['stage1']['meta'].get('run_id') != RUN_ID:
        problems.append('stage1 run_id mismatch')
    if plan['stage1']['meta'].get('pair_id') != PAIR_ID:
        problems.append('stage1 pair_id mismatch')
    if plan['stage1']['data'].get('seed') != PRETRAINING_SEED:
        problems.append('stage1 data.seed mismatch')
    if plan['stage1']['logging'].get('folder') != RUN_ROOT:
        problems.append('stage1 logging.folder mismatch')
    if PRETRAINING_SEED in HISTORICAL_SEEDS:
        problems.append(f'seed {PRETRAINING_SEED} collides with a historical seed')

    # scheduler horizons
    s1 = scheduler_horizon(plan['stage1'])
    s2 = scheduler_horizon(plan['stage2'])
    if not s1['consistent'] or not s2['consistent']:
        problems.append('scheduler horizon is not the 50k budget in one/both phases')
    if s1 != s2:
        problems.append('scheduler horizon differs between phase 1 and phase 2')

    frozen = check_frozen_files(hash_large_pretrained=hash_large_pretrained)
    problems.extend(frozen['problems'])

    return {
        'schema_version': 'stage7a_protocol_check/v1',
        'ok': not problems,
        'problems': problems,
        'stage_name': STAGE_NAME,
        'run_id': RUN_ID,
        'pair_id': PAIR_ID,
        'pretraining_seed': PRETRAINING_SEED,
        'base_config': str(BASE_CONFIG.relative_to(ROOT)),
        'base_config_sha256': sha256_file(BASE_CONFIG),
        'stage1_config': str(STAGE1_CONFIG.relative_to(ROOT)),
        'stage1_config_sha256': (sha256_file(STAGE1_CONFIG)
                                 if STAGE1_CONFIG.is_file() else None),
        'stage2_config': str(STAGE2_CONFIG.relative_to(ROOT)),
        'stage2_config_sha256': (sha256_file(STAGE2_CONFIG)
                                 if STAGE2_CONFIG.is_file() else None),
        'budget': {
            'max_optimizer_steps': MAX_OPTIMIZER_STEPS,
            'interrupt_after_optimizer_steps': INTERRUPT_AFTER,
            'save_at_optimizer_steps': list(SAVE_AT_STEPS),
            'warmup_steps': WARMUP_STEPS,
            'epochs': EPOCHS,
            'epochs_minimum': epochs_needed(),
            'optimizer_steps_per_epoch': EXPECTED_OPTIMIZER_STEPS_PER_EPOCH,
        },
        'scheduler': {'stage1': s1, 'stage2': s2},
        'frozen_inputs': frozen,
        'resume_read_checkpoint': STAGE2_READ_CHECKPOINT,
        'allowed_changes_from_stage6': sorted('.'.join(key) for key in ALLOWED_CHANGES),
        'allowed_resume_changes': sorted('.'.join(key) for key in ALLOWED_RESUME_CHANGES),
        'hash_large_pretrained': hash_large_pretrained,
    }


def render_md(report: Mapping[str, object]) -> str:
    lines: List[str] = []
    lines.append('# Stage 7A — 协议检查')
    lines.append('')
    lines.append(f'- 状态：{"**通过**" if report["ok"] else "**失败**"}')
    lines.append(f'- 运行标识：{report["run_id"]}（pair={report["pair_id"]}，seed={report["pretraining_seed"]}）')
    lines.append(f'- 基础配置：{report["base_config"]}（SHA-256 `{report["base_config_sha256"][:16]}…`）')
    lines.append(f'- 阶段1配置：{report["stage1_config"]}')
    lines.append(f'- 阶段2配置：{report["stage2_config"]}')
    lines.append('')
    b = report['budget']
    lines.append('## 预算与调度')
    lines.append('')
    lines.append('| 字段 | 值 |')
    lines.append('|---|---|')
    lines.append(f'| max_optimizer_steps | {b["max_optimizer_steps"]} |')
    lines.append(f'| interrupt_after_optimizer_steps | {b["interrupt_after_optimizer_steps"]} |')
    lines.append(f'| save_at_optimizer_steps | {b["save_at_optimizer_steps"]} |')
    lines.append(f'| warmup_steps | {b["warmup_steps"]} |')
    lines.append(f'| epochs | {b["epochs"]}（最低 {b["epochs_minimum"]}） |')
    lines.append(f'| optimizer_steps_per_epoch | {b["optimizer_steps_per_epoch"]} |')
    lines.append('')
    s = report['scheduler']['stage1']
    lines.append('## 调度器 horizon 核验（LR / WD / EMA 均在 50k 处结束）')
    lines.append('')
    lines.append(f'- LR cosine：`warmup={s["warmup_steps"]}` → `T_max={s["lr_T_max"]}`，'
                 f'在 step {s["lr_reaches_final_at"]} 到达 final_lr')
    lines.append(f'- WD cosine：`T_max={s["wd_T_max"]}`，在 step {s["wd_reaches_final_at"]} 到达 final_wd')
    lines.append(f'- EMA：`0.996 → 1.0` 在 step {s["ema_reaches_1_at"]}')
    lines.append(f'- 一致性：{"通过" if s["consistent"] else "失败"}')
    lines.append('')
    lines.append('## 冻结输入文件')
    lines.append('')
    lines.append('| 文件 | 状态 | SHA-256 |')
    lines.append('|---|---|---|')
    for name, info in report['frozen_inputs']['files'].items():
        ok = '通过' if info['ok'] else '失败'
        lines.append(f'| {name} | {ok} | `{info["actual_sha256"]}` |')
    lines.append('')
    if report['problems']:
        lines.append('## 问题')
        lines.append('')
        for problem in report['problems']:
            lines.append(f'- {problem}')
    else:
        lines.append('未发现协议偏离。')
    return '\n'.join(lines) + '\n'


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, default=ROOT / 'reports/stage7a_protocol.json')
    parser.add_argument('--md', type=Path, default=ROOT / 'reports/stage7a_protocol.md')
    parser.add_argument('--hash-large-pretrained', action='store_true')
    args = parser.parse_args(argv)

    report = run_checks(hash_large_pretrained=args.hash_large_pretrained)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(render_md(report), encoding='utf-8')
    print(json.dumps({'ok': report['ok'], 'problems': report['problems'],
                      'scheduler': report['scheduler']['stage1']}, indent=2,
                     ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
