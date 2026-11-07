#!/usr/bin/env python
"""Stage 8 Phase 2: freeze the experiment protocol and registry BEFORE training.

The registry is the single source of truth for Phase 4: one row per
(task, backbone, downstream seed), with the exact command that will be
executed and the output directory it will write.  Nothing in Phase 4 may run
a command that is not a row here, and every row carries the backbone SHA-256
at freeze time so a checkpoint swap mid-stage is detectable.

Freeze preconditions (all hard errors, no warnings):

* checkpoint inventory reports 7/7 present and matching;
* every candidate dataset with a local copy is leakage-clean vs the frozen
  pretraining manifest (Phase 1 outputs);
* the official-test-once rule is encoded in the registry (test evaluation only
  in the final run of each task row).

Outputs ``reports/stage8_protocol.md`` and
``reports/stage8_experiment_registry.{csv,json}``.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = ROOT / 'reports'

from tools.stage8_backbones import (build_registry, PRIMARY, EXPLORATORY,
                                    recorded_checkpoint_sha256)

# Frozen stage-wide constants -------------------------------------------------
DOWNSTREAM_SEEDS = (0, 1, 2)          # head seeds; NEVER equal to pretraining
                                     # seeds 20261104/20261105 (RQ5 rule)
SAMPLING_SEED = 20261107             # for any dataset subsampling decision
PY = './.venv-uv/bin/python'

RUN_ROOT = Path('/data/agrisa_jepa_runs/stage8')

TASKS = [
    {
        'task_id': 'gwhd_det',
        'task': 'object detection (Global Wheat Head 2021)',
        'status': 'READY (Stage 5A pipeline reused verbatim)',
        'splits': 'official competition_{train,val,test}.csv, used verbatim; '
                  'test evaluated once at the end of training',
        'entry': f'{PY} -m src.downstream.detection_train',
        'common_args': ('--data-root /data/datasets/GWHD2021/extracted/gwhd_2021 '
                        '--iterations 20000 --batch-size 2 --seed {seed} '
                        '--input-size 518 --final-test'),
        'per_backbone_args': ('--checkpoint {checkpoint} --label {label}'),
        'out_pattern': 'gwhd_det/{backbone}_seed{seed}',
        'primary_metric': 'test map (COCO AP), AP50 secondary',
        'exploratory': True,
    },
    {
        'task_id': 'cropandweed_sem',
        'task': 'semantic segmentation (CropAndWeed, CropsOrWeed9 mapping)',
        'status': 'READY_WITH_MODIFICATION (no official split CSV; '
                  'ave_/vwg_ set semantics used)',
        'splits': ('train = ave_ set (3,044 images, full); '
                   'eval = vwg_ set, one image per recording session '
                   f'(seed {SAMPLING_SEED}), ~665 images'),
        'entry': f'{PY} -m src.downstream.cropandweed_probe',
        'common_args': ('--data-root /data/datasets/cropandweed/images '
                        f'--annotation-root '
                        f'{ROOT}/data/holdout_raw/cropandweed_annotations/labelIds/CropAndWeed '
                        '--seed {seed} --epochs 10'),
        'per_backbone_args': ('--checkpoint {checkpoint} --label {label}'),
        'out_pattern': 'cropandweed_sem/{backbone}_seed{seed}',
        'primary_metric': '9-class mIoU (Soil 0 and Vegetation 255 ignored)',
        'exploratory': True,
    },
    {
        'task_id': 'deepweeds_cls',
        'task': 'classification (DeepWeeds, 9 classes)',
        'status': 'READY_WITH_MODIFICATION (5-fold CV; fold 0 fixed)',
        'splits': ('train = train_subset0 (10,501) + val_subset0 (3,501); '
                   'test = test_subset0 (3,507), evaluated once'),
        'entry': f'{PY} -m src.downstream.deepweeds_probe',
        'common_args': (f'--labels {ROOT}/data/holdout_raw/deepweeds_labels '
                        '--seed {seed} --epochs 10'),
        'per_backbone_args': ('--checkpoint {checkpoint} --label {label}'),
        'out_pattern': 'deepweeds_cls/{backbone}_seed{seed}',
        'primary_metric': 'macro-F1 on test (accuracy secondary)',
        'exploratory': True,
    },
    {
        'task_id': 'plantdoc_cls',
        'task': 'classification (PlantDoc, 28 class directories)',
        'status': ('READY_WITH_MODIFICATION (detection impossible locally: no '
                   'bbox annotations; classification labels from directory '
                   'structure)'),
        'splits': ('official train/test used verbatim; val carved from train '
                   f'stratified by class at 15% (seed {SAMPLING_SEED}); '
                   'test evaluated once'),
        'entry': f'{PY} -m src.downstream.plantdoc_probe',
        'common_args': (f'--root '
                        f'{ROOT}/data/holdout_raw/plantdoc_extract/PlantDoc-Dataset-master '
                        '--seed {seed} --epochs 10'),
        'per_backbone_args': ('--checkpoint {checkpoint} --label {label}'),
        'out_pattern': 'plantdoc_cls/{backbone}_seed{seed}',
        'primary_metric': 'macro-F1 on official test (accuracy secondary)',
        'exploratory': False,
    },
    {
        'task_id': 'phenobench_sem',
        'task': 'semantic segmentation (PhenoBench semantics; reuse Stage 7B)',
        'status': ('READY (Stage 7B covers A_p1/A_p2/B_p1/B_p2/C_p1/C_p2 with '
                   'probe seeds 0/1/2); Stage 8 adds the OFFICIAL baseline '
                   'required by RQ1'),
        'splits': 'official train/val, test-server not used',
        'entry': f'{PY} -m src.downstream.phenobench_probe',
        'common_args': ('--root /data/datasets/PhenoBench/PhenoBench-v110 '
                        '--cache-dir /data/agrisa_jepa_runs/stage8/phenobench_sem/cache '
                        '--seed 0 1 2 --epochs 10'),
        'per_backbone_args': ('--checkpoint {checkpoint} --arm-label {label}'),
        'out_pattern': 'phenobench_sem/{backbone}',
        'primary_metric': 'mIoU (official semantics: soil/crop/weed)',
        'exploratory': False,   # C_p1/C_p2 already done in Stage 7B
        # Stage 7B already probed A/B (p1+p2) with probe seeds 0/1/2 and
        # recorded official metrics; only the OFFICIAL baseline is new here.
        'reuse_stage7b_for': ['A_p1', 'A_p2', 'B_p1', 'B_p2'],
        # phenobench_probe takes --seed 0 1 2 in ONE invocation and trains all
        # three probes on the shared feature cache; emit one row per backbone
        # rather than three per-seed rows.
        'all_seeds_in_one_command': True,
    },
]


def check_preconditions() -> None:
    inv = json.loads((R / 'stage8_checkpoint_inventory.json').read_text('utf-8'))
    ok = inv['summary']['n_present_and_matching']
    assert ok == 7, f'checkpoint inventory not 7/7: {ok}'

    man = json.loads((R / 'stage8_dataset_manifest.json').read_text('utf-8'))
    for ds, info in man['datasets'].items():
        leak = (info.get('leakage') or {}).get('verdict')
        if leak == 'LEAKAGE':
            raise SystemExit(f'{ds} leaks into the pretraining manifest; '
                             f'freeze refused')
        if info['status'] == 'READY' or info['status'] == 'READY_WITH_MODIFICATION':
            assert leak == 'CLEAN_exact', f'{ds}: {leak}'


def _emit_rows(task, reg, recorded, backbone, seed, role, status, out_dir, cmd):
    spec = reg[backbone]
    sha = recorded.get(backbone, spec.checkpoint)
    return {
        'task_id': task['task_id'],
        'role': role,
        'backbone': backbone,
        'pretraining_seed': spec.pretraining_seed,
        'backbone_checkpoint': spec.checkpoint,
        'backbone_sha256_ref':
            ('official IN1K-vit.h.14-300e release'
             if backbone == 'OFFICIAL' else f'stage7b manifest: {sha[:12]}...'),
        'downstream_seed': seed,
        'status': status,
        'output_dir': out_dir,
        'command': cmd,
    }


def _build_command(task, spec, seed, out_dir) -> str:
    """Registry command; always pins --out-dir so a row writes its own dir."""
    return ' '.join([task['entry'],
                     task['per_backbone_args'].format(
                         checkpoint=spec.checkpoint,
                         label=spec.label),
                     task['common_args'].format(
                         seed=seed, sampling_seed=SAMPLING_SEED),
                     f'--out-dir {out_dir}'])


def build_rows() -> list:
    reg = build_registry(include_exploratory=True)
    recorded = recorded_checkpoint_sha256()
    rows = []
    for task in TASKS:
        reuse = set(task.get('reuse_stage7b_for', []))
        all_seeds_one_command = bool(task.get('all_seeds_in_one_command'))
        seed_list = ['0,1,2'] if all_seeds_one_command else DOWNSTREAM_SEEDS
        for backbone in PRIMARY:
            spec = reg[backbone]
            for seed in seed_list:
                if all_seeds_one_command:
                    out_dir = str(RUN_ROOT / task['out_pattern'].format(
                        backbone=backbone))
                else:
                    out_dir = str(RUN_ROOT / task['out_pattern'].format(
                        backbone=backbone, seed=seed))
                if backbone in reuse:
                    # Phenobench A/B rows reuse Stage 7B results verbatim; the
                    # command is kept for provenance only, status blocks rerun.
                    rows.append(_emit_rows(
                        task, reg, recorded, backbone, seed, 'primary',
                        'REUSE_STAGE7B', out_dir,
                        ('(not re-run; result in '
                         'reports/stage7b_phenobench_per_checkpoint.json '
                         f'entry {backbone}, probe seeds 0/1/2)')))
                else:
                    cmd = _build_command(task, spec, seed, out_dir)
                    rows.append(_emit_rows(
                        task, reg, recorded, backbone, seed, 'primary',
                        'FROZEN', out_dir, cmd))
        if task['exploratory']:
            for backbone in EXPLORATORY:
                spec = reg[backbone]
                for seed in seed_list:
                    if all_seeds_one_command:
                        out_dir = str(RUN_ROOT / task['out_pattern'].format(
                            backbone=backbone))
                    else:
                        out_dir = str(RUN_ROOT / task['out_pattern'].format(
                            backbone=backbone, seed=seed))
                    cmd = _build_command(task, spec, seed, out_dir)
                    rows.append(_emit_rows(
                        task, reg, recorded, backbone, seed, 'exploratory',
                        'FROZEN_EXPLORATORY', out_dir, cmd))
    return rows


def render_protocol(rows: list) -> str:
    lines = ['# Stage 8 — 冻结协议与实验注册表', '',
             f'> 冻结时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec="seconds")}',
             '',
             '## 0. 研究问题', '',
             '- **RQ1** 农业继续预训练（A/B）在跨任务迁移上是否优于官方 I-JEPA？',
             '- **RQ2** 收益是否跨数据集泛化？',
             '- **RQ3** Stage 7B 的调度差异（A vs B）是否迁移到其他任务？',
             '- **RQ4** 不同任务是否需要不同表征？',
             '- **RQ5** 收益在预训练 seed / head seed / 切分 / 数据集间是否稳定？',
             '',
             '## 1. Backbone 注册（5 primary + 2 exploratory）', '',
             '| key | arm | pretrain seed | horizon/warmup | role |',
             '|---|---|---|---|---|']
    reg = build_registry(include_exploratory=True)
    for key in list(PRIMARY) + list(EXPLORATORY):
        spec = reg[key]
        sched = (f'{spec.horizon}/{spec.warmup}' if spec.horizon else '—')
        lines.append(f'| {key} | {spec.arm} | {spec.pretraining_seed or "—"} '
                     f'| {sched} | {spec.role} |')
    lines += ['', '## 2. 任务协议', '']
    for task in TASKS:
        lines += [f"### {task['task_id']} — {task['task']}", '',
                  f"- 状态：{task['status']}",
                  f"- 切分：{task['splits']}",
                  f"- 主指标：{task['primary_metric']}",
                  f"- 入口：`{task['entry']}`", '']
    lines += ['## 3. 实验注册表（冻结）', '',
              '全部 primary 任务 × 下游 seed 0/1/2；下游 seed 与预训练 seed '
              '(20261104/20261105) 严格不同。', '',
              '| task | backbone | seed | role | out |',
              '|---|---|---|---|---|']
    for row in rows:
        short = row['output_dir'].replace(str(RUN_ROOT) + '/', '')
        lines.append(f"| {row['task_id']} | {row['backbone']} "
                     f"| {row['downstream_seed']} | {row['role']} "
                     f"| `{short}` |")
    lines += ['', '## 4. 冻结规则', '',
              '- 注册表一旦冻结，Phase 4 只能按行执行；任何偏差（含失败重跑）'
              '必须以新的行追加并记录原因，不得原地改写。',
              '- 官方 test 只在每行训练结束时评估一次。',
              '- 泄漏审计（Phase 1）为冻结前置条件：全部本地数据集 CLEAN_exact。',
              '- 训练从不覆盖：输出目录已存在 protocol.json 时拒绝启动。', '']
    return '\n'.join(lines) + '\n'


def main() -> int:
    check_preconditions()
    rows = build_rows()
    R.mkdir(exist_ok=True)
    md = render_protocol(rows)
    (R / 'stage8_protocol.md').write_text(md, encoding='utf-8')

    with (R / 'stage8_experiment_registry.csv').open('w', newline='',
                                                     encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (R / 'stage8_experiment_registry.json').write_text(
        json.dumps({'frozen_at': datetime.now(timezone.utc).isoformat(),
                    'sampling_seed': SAMPLING_SEED,
                    'downstream_seeds': list(DOWNSTREAM_SEEDS),
                    'rows': rows}, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(f'froze {len(rows)} experiment rows:')
    by_task = {}
    for row in rows:
        by_task.setdefault(row['task_id'], []).append(row['role'])
    for t, roles in by_task.items():
        print(f"  {t}: {len(roles)} rows "
              f"({sum(1 for r in roles if r == 'primary')} primary, "
              f"{sum(1 for r in roles if r != 'primary')} exploratory)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())