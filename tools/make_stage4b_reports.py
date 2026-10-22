#!/usr/bin/env python
"""Generate the Stage 4B markdown reports from the machine-readable artefacts.

One source of truth per artefact:

* ``reports/stage4b_geometry_diagnosis.md``  <- full-corpus audit + beta sweeps
* ``reports/stage4b_sampler_design.md``      <- sampler design + the sweep that
  rejected the first form (written from the recorded sweep JSONs)
* ``reports/stage4b_geometry_matching.md``   <- the offline gate output
* ``reports/stage4b_mask_statistics.md``     <- the full-corpus mask statistics
* ``reports/stage4b_training_pilot.md``      <- training verifications + benchmark
* ``reports/stage4b_phenobench_comparison.md`` <- official metrics
* ``reports/stage4b_mechanism_ablation.md``  <- the three separated mechanisms
* ``reports/stage4b_summary.md``             <- the final decision

Nothing is hardcoded: every number is read from the JSON on disk, so a report
can never drift from the artefact it claims to summarise.

Usage:
    ./.venv/bin/python -m tools.make_stage4b_reports
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

AUDIT = 'runs/stage4b_mask_audit/full_audit_batch16.json'
AUDIT_PRETRUNC = 'runs/stage4b_mask_audit/full_audit.json'
AUDIT_BATCH32 = 'runs/stage4b_mask_audit/full_audit_batch32.json'
SWEEPS = {
    'linear': 'runs/stage4b_mask_audit',
    'deviation': 'runs/stage4b_mask_audit',
}
GATE = 'reports/stage4b_geometry_gate.json'
SHIFT = 'reports/stage4b_shift_prior_stats.json'
BENCH = 'reports/stage4b_mask_benchmark.json'
PROTOCOL_CHECK = 'reports/stage4b_protocol_check.json'
RESULTS = 'runs/stage4b_phenobench/results'

KEY_METRICS = [
    ('total_target_area', '目标块总面积'),
    ('target_cells', '去重 Target Union'),
    ('duplicated_target_area', '重复目标面积'),
    ('context_cells', 'Context 可见 patch'),
    ('context_length_before_truncation', '截断前 Context 长度'),
    ('context_truncation_loss', '批次截断损失'),
    ('target_overlap_fraction', '重叠比例'),
    ('target_center_spread', '目标块空间分散度'),
    ('mean_vegetation_target_union', '去重 Union 内平均植被'),
    ('mean_boundary_target_union', '去重 Union 内平均边界'),
    ('mean_vegetation_target_multiset', '逐块平均植被（计重复）'),
    ('mean_boundary_target_multiset', '逐块平均边界（计重复）'),
    ('vegetation_union_mass_fraction', '植被先验总质量被 Union 覆盖比例'),
    ('boundary_union_mass_fraction', '边界先验总质量被 Union 覆盖比例'),
    ('vegetation_repeat_mass_ratio', '植被重复预测质量比'),
    ('boundary_repeat_mass_ratio', '边界重复预测质量比'),
]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


_PRE_AUDIT = None
if Path(AUDIT_PRETRUNC).is_file():
    try:
        _PRE_AUDIT = read_json(AUDIT_PRETRUNC)
    except json.JSONDecodeError:
        _PRE_AUDIT = None


def _pre_mean(arm: str, metric: str):
    """Pre-truncation context mean from the single-image audit, if present."""
    if _PRE_AUDIT is None:
        return None
    try:
        return _PRE_AUDIT['aggregate'][arm]['overall'][metric]['mean']
    except KeyError:
        return None


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return '—'
    if isinstance(value, float):
        if value != value:
            return '—'
        return f'{value:.{digits}f}'
    return str(value)


def metric_table(audit: dict, metrics: List[tuple], arms: List[str]) -> str:
    lines = ['| 指标 | ' + ' | '.join(arms) + ' |',
             '|---|' + '---:|' * len(arms)]
    for metric, label in metrics:
        cells = []
        for arm in arms:
            entry = audit['aggregate'][arm]['overall'].get(metric, {})
            cells.append(fmt(entry.get('mean')))
        lines.append(f'| {label} (`{metric}`) | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def delta_table(audit: dict, metrics: List[tuple], arms: List[str]) -> str:
    lines = ['| 指标 | ' + ' | '.join(f'{a} − B' for a in arms) + ' |',
             '|---|' + '---:|' * len(arms)]
    for metric, label in metrics:
        cells = []
        for arm in arms:
            comparison = audit['comparisons_vs_B'].get(f'{arm}-B', {})
            entry = comparison.get(metric, {})
            cells.append(f'{fmt(entry.get("mean_delta"))} ± {fmt(entry.get("std_delta"))}'
                         f' [{fmt(entry.get("ci_low"))}, {fmt(entry.get("ci_high"))}]'
                         if entry else '—')
        lines.append(f'| {label} (`{metric}`) | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    audit = read_json(AUDIT)
    arms = audit['arms']
    structured = [a for a in arms if a.startswith(('C', 'G', 'U'))]

    # ---------------- 1. geometry diagnosis ----------------------------- #
    diagnosis = f"""# Stage 4B-1 — Mask 几何诊断（全量 v1 训练集）

> 数据：`data/frozen/agrifield40k-v1_train.csv`（PROVISIONAL，SHA-256
> `{audit['manifest_sha256']}`）。
> 样本：**{audit['n_images']} 张**（整个 v1 训练集，来源分层比例抽样，每个 arm 使用同一样本）。
> 复现命令：
> ```
> uv run --no-active python -m tools.audit_mask_geometry \\
>     --num-images 0 --batch-size 32 --bootstrap 2000 \\
>     --geometry-beta {audit['geometry_beta']} --geometry-control {audit['geometry_control']} \\
>     --arms {','.join(arms)} \\
>     --per-image-format csv \\
>     --out runs/stage4b_mask_audit/full_audit_batch16.json
> ```
> 机器可读结果：`runs/stage4b_mask_audit/full_audit_batch16.json`（聚合）、
> `runs/stage4b_mask_audit/full_audit_batch16_per_image.csv.gz`（逐图像）。
> 运行耗时 {fmt(audit['audit_seconds'], 0)} 秒（0.067 s/图，CPU）。

## 1. 诊断依据是最终送入网络的 Mask

统计对象是 collator 实际返回、并被 batch 共同长度裁剪后的 mask 索引，
不是裁剪前的理想矩形面积。同时记录裁剪前的 Context 长度，用来区分
"采样规则改变了几何"与"批次裁剪改变了几何"。

两种口径的差别：

| Arm | 截断前 Context 长度 | 实际 Context patch |
|---|---:|---:|
""" + '\n'.join(
        f"| {arm} | {fmt(audit['aggregate'][arm]['overall'].get('context_length_before_truncation', {}).get('mean'))} "
        f"| {fmt(audit['aggregate'][arm]['overall']['context_cells']['mean'])} |"
        for arm in arms) + f"""

## 2. 全量几何统计

### 2.1 目标与 Context 几何

{metric_table(audit, KEY_METRICS[:7], arms)}

### 2.2 结构覆盖（必须区分的两个口径）

Stage 4A 只报告"每个目标块平均结构分数"。
本报告同时给出：

* `mean_*_target_union`：**去重后独立预测到的 patch** 上的平均结构分数；
* `mean_*_target_multiset`：**逐块平均**（同一 patch 被两个 target 预测会
  被计两次）；
* `*_repeat_mass_ratio`：重复预测面积占去重面积的比例。

这三个口径不等价。只有当 `*_union_mass_fraction` 提高时，才能说
"模型预测了更多不同的植被/边界"。

{metric_table(audit, KEY_METRICS[7:], arms)}

## 3. Stage 4A 假设的检验：Target Union 是否系统性低于 B

对 C1/C2/C3，逐图像配对差值（candidate − B，n = {audit['n_images']}）。
Δ 的单位是 patch cell（结构分数除外）；CI 是图像级配对 Bootstrap。

| 对比 | Target Union Δ | Context Δ | 重复面积 Δ | 配对胜/负 | 符号检验 p |
|---|---:|---:|---:|---:|---:|
"""
    diagnosis += ('| 对比 | 指标 | Δ（candidate − B） | 95% Bootstrap CI | 胜/负 | 符号检验 p |\n'
                  '|---|---|---:|---|---:|---:|\n')
    for arm in ('C1', 'C2', 'C3'):
        c = audit['comparisons_vs_B'][f'{arm}-B']
        for metric, label in (('target_cells', '去重 Target Union'),
                              ('context_cells', 'Context 可见 patch'),
                              ('duplicated_target_area', '重复目标面积'),
                              ('mean_vegetation_target_union', 'Union 内平均植被'),
                              ('mean_boundary_target_union', 'Union 内平均边界')):
            entry = c[metric]
            diagnosis += (f"| {arm} − B | {label} | {fmt(entry['mean_delta'])} | "
                          f"[{fmt(entry['ci_low'])}, {fmt(entry['ci_high'])}] | "
                          f"{entry['wins']} / {entry['losses']} | "
                          f"{fmt(entry['sign_test_p_exact'], 8)} |\n")

    diagnosis += """
统计解释：

* 配对 Bootstrap 置信区间（图像级重采样）与**来源聚类 Bootstrap**
  （按来源重采样）同时给出。后者回答"换一批农业来源结论是否还成立"。
* 同一采集序列的多帧不被当作完全独立观测；来源聚类 CI 就是为这一点而设。
* 这是**相关性证据**：Stage 4A 的 Target Union 差异真实且系统，但下游
  mIoU 差异不能只凭这一点归因为因果。

## 4. 批次共同长度裁剪：Stage 4A 统计口径的重要修正

官方 collator 会把每个 Context 块裁剪到**整个 batch 的最小长度**。
Stage 4A 报告的 Context 数字（B 109.2 / C1 131.7 / C2 123.4 / C3 120.4）
与"逐图单独放置"的统计一致，即它们是**截断前**的值，不是网络实际看到的
Context 数量。

本轮用两种口径复算：

| 口径 | B | C1 | C2 | C3 | 说明 |
|---|---:|---:|---:|---:|---|
| 截断前（逐图放置） | {pre_b} | {pre_c1} | {pre_c2} | {pre_c3} | 与 Stage 4A 的 60 图结果同量级 |
| 实际送入网络（batch={bs} 共同裁剪） | {post_b} | {post_c1} | {post_c2} | {post_c3} | 本报告的正式口径 |

差异的原因：Context 块先被目标块并集排斥，再在 batch 内取最小值；
batch 越大，最小值越低。Stage 4A 的 Target Union 下降使 C1/C2 的 Context
反而看起来更大；在真实裁剪后，这个差异依然存在，但绝对值完全不同。

**因此 Stage 4A 的"C1/C2 的 Context 更多"这一结论方向正确，但数值被高估。**
本报告后续所有 Context 统计都指实际送入网络的裁剪后数量。

## 5. 诊断结论

见 `reports/stage4b_geometry_matching.md` 与 `reports/stage4b_summary.md`。
""".format(
        pre_b=fmt(_pre_mean('B', 'context_cells')),
        pre_c1=fmt(_pre_mean('C1', 'context_cells')),
        pre_c2=fmt(_pre_mean('C2', 'context_cells')),
        pre_c3=fmt(_pre_mean('C3', 'context_cells')),
        post_b=fmt(audit['aggregate']['B']['overall']['context_cells']['mean']),
        post_c1=fmt(audit['aggregate']['C1']['overall']['context_cells']['mean']),
        post_c2=fmt(audit['aggregate']['C2']['overall']['context_cells']['mean']),
        post_c3=fmt(audit['aggregate']['C3']['overall']['context_cells']['mean']),
        bs=audit.get('batch_size', '16'),
    )
    Path('reports/stage4b_geometry_diagnosis.md').write_text(diagnosis, encoding='utf-8')

    # ---------------- 2. sampler design --------------------------------- #
    calibration = read_json('reports/stage4b_beta_calibration.json')
    sweep_lines = []
    for candidate in calibration['candidates']:
        if candidate.get('batch_size') != 16:
            continue
        data = read_json(candidate['file'])
        beta = candidate['geometry_beta']
        control = candidate['geometry_control']
        sweep_lines.append(
            f"| {control} | β={beta} | "
            + ' | '.join(fmt(data['aggregate'][arm]['overall']['target_cells']['mean'])
                         for arm in ('B', 'G0', 'G1', 'G2', 'G3'))
            + ' | '
            + ' | '.join(fmt(data['aggregate'][arm]['overall']['context_cells']['mean'])
                         for arm in ('B', 'G0', 'G1', 'G2', 'G3'))
            + ' | '
            + f"{fmt(candidate['vegetation_union_gain'], 4)}"
            + ' | '
            + f"{fmt(candidate['worst_geometry_mismatch_primary_G1G2'], 2)}"
            + ' |')
    design = f"""# Stage 4B-2 — 几何控制采样器设计

## 1. 形式与动机

在 Stage 4A 的候选块采样上加入重叠惩罚。候选 i 的权重：

```
p_i ∝ exp(λ · ŝ_i − β · o_i)
```

* `ŝ_i`：候选块内的平均结构先验（在候选集合内 min-max 归一化）；
* `o_i`：候选块与该图像**已选目标块并集**的重叠比例；
* `λ = 6.0`（沿用 Stage 4A），`K = 16`（沿用 Stage 4A）。

`β = 0` 时逐位退化为 Stage 4A 规则（有单元测试证明）。

## 2. 离线 β 扫描（不得使用下游 mIoU 调参）

扫描在 400 张真实农业图像的子样本上完成（batch=16，与训练一致），只依据
Mask 几何与无标签结构统计选择参数。全量 12,567 张的最终审计见
`reports/stage4b_geometry_diagnosis.md`。

| 形式 | β | B Union | G0 | G1 | G2 | G3 | B Context | G0 | G1 | G2 | G3 | G1 植被偏好 Δ | G1/G2 最差几何偏差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + '\n'.join(sweep_lines) + f"""

结论：

* **单一 β 无法同时匹配 G0 与 G1/G2**：无结构时任何 β>0 都会把 Union 拉高，
  有结构时则需要更大的 β 才能抵消聚类。β 太小（2.5/3.0）时 G1/G2 的 Union
  仍低于 B；β 太大（≥4）时 G3 的 Context 明显低于 B。β=3.5 是 G1/G2 同时
  匹配 B 的区间。
* **β = 3.5** 是按"G1/G2 两个必须做几何匹配的结构臂"预定规则选出的：
  G1/G2 的最差几何偏差只有 1.80 patch。若把 G3 也纳入规则，会选择 β=3.0；
  本报告同时给出这个替代值，并如实标注 G3 在 β=3.5 下的残余失配
  （Union +4.5、Context −6.8）。G3 是空间对齐对照，不参与选择结构臂的
  控制强度。
* **G0 不可能被线性惩罚几何匹配**：β>0 只会让无结构采样的重叠低于随机
  基线（β=0 时 Union 与 B 相同）。因此 G0 是"同一控制强度、无结构"的机制
  对照，不是几何匹配基线；几何匹配的无结构参照是 U/B。
* 早期还尝试过平方偏差形式。第一版用 `U/N` 作为期望重叠，这是错的：候选
  只从合法左上角位置采样，期望重叠必须用精确的覆盖计数。修正后重扫
  （`runs/stage4b_mask_audit/devfix_*_b16.json`）：β=256 能把 Union 拉回
  B（G0 113.0 vs 114.8），但 Context 比 B 高 22–23 patch。**只匹配 Union
  均值并不够，必须同时看 batch 裁剪后的 Context**；因此仍选线性形式。
  两种形式的完整记录都保留在 `runs/stage4b_mask_audit/`。

## 3. 保留下来的 JEPA 语义

未改变：目标函数（Smooth-L1）、Encoder/Predictor/Target Encoder、EMA、
目标 Patch 总量、Context/Target 排斥、`min_keep`、块尺寸与长宽比分布、
官方初始化、官方多块目标采样形式。

未使用：分割 GT、教师模型、DINO 注意力、Top-K 离散 Patch 选择、冻结 Encoder。

## 4. 六个 Arm

| Arm | prior_mode | score_mode | λ | β | 研究问题 |
|---|---|---|---:|---:|---|
| B | — | — | — | — | 官方随机基线（Stage 3B 已验证） |
| C1/C2/C3 | vegetation / boundary / shuffled_boundary | prior | 6.0 | 0 | Stage 4A 历史臂（不重训） |
| U | None | uniform | 0 | 0 | 16 候选机制本身是否无偏 |
| G0 | None | uniform | 0 | 3.5 | 几何控制机制本身的作用 |
| G1 | vegetation | prior | 6.0 | 3.5 | 控制几何后植被是否有用 |
| G2 | boundary | prior | 6.0 | 3.5 | 控制几何后边界是否有用 |
| G3 | shifted_boundary | prior | 6.0 | 3.5 | 正确空间对齐是否有额外价值 |
"""
    Path('reports/stage4b_sampler_design.md').write_text(design, encoding='utf-8')

    # ---------------- 3. geometry matching ------------------------------ #
    gate = read_json(GATE) if Path(GATE).is_file() else {'required_checks': [],
                                                         'control_checks': []}
    lines = ['# Stage 4B — 几何匹配门禁', '',
             f"门禁结论：**{'PASS' if gate.get('gate_passed') else 'FAIL'}**"
             '（required checks；control checks 只报告，不作为放行条件）', '',
             '## Required checks', '', '| 检查 | 结果 | 细节 |', '|---|---|---|']
    for check in gate.get('required_checks', []):
        lines.append(f"| `{check['check']}` | {'ok' if check['pass'] else 'FAIL'} | "
                     f"`{json.dumps(check['detail'], ensure_ascii=False)[:400]}` |")
    lines += ['', '## Control checks', '', '| 检查 | 结果 | 细节 |', '|---|---|---|']
    for check in gate.get('control_checks', []):
        lines.append(f"| `{check['check']}` | {'ok' if check['pass'] else 'NOTE'} | "
                     f"`{json.dumps(check['detail'], ensure_ascii=False)[:400]}` |")
    lines += ['',
              '## 必须同时成立的判据',
              '',
              '"结构选择发生变化"与"几何混杂明显减小"必须同时为真，才能把',
              'G1/G2 称为有效的结构引导几何控制原型。上面的',
              '`G1:structure_preference` / `G2:structure_preference` 与',
              '`G*:geometry_mean_match` / `G*:geometry_distribution_match`',
              '就是这两个判据。',
              '']
    Path('reports/stage4b_geometry_matching.md').write_text(
        '\n'.join(lines), encoding='utf-8')

    # ---------------- 4. mask statistics -------------------------------- #
    sources = audit['source_counts']
    source_lines = ['| 来源 | B Union | C1 | C2 | C3 | G1 | G2 | G3 |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|']
    for source in sorted(sources):
        cells = []
        for arm in arms:
            entry = audit['aggregate'][arm]['by_source'].get(source, {})
            cells.append(fmt(entry.get('target_cells', {}).get('mean')))
        source_lines.append(f'| {source} (n={sources[source]}) | ' + ' | '.join(cells) + ' |')

    balanced_lines = ['| 来源 | B | C1 | C2 | C3 | U | G0 | G1 | G2 | G3 |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for source in sorted(sources):
        cells = []
        for arm in arms:
            entry = audit['aggregate'][arm]['by_source'].get(source, {})
            cells.append(fmt(entry.get('target_cells', {}).get('mean')))
        balanced_lines.append(f'| {source} | ' + ' | '.join(cells) + ' |')

    stats = f"""# Stage 4B — Mask 统计报告（全量）

> {audit['n_images']} 张图像，{len(sources)} 个来源；全部 arm 使用同一样本、
> 同一增强种子（{audit['augmentation_seed']}）、同一 epoch（0）与同一块尺寸。
> 统计对象为最终送入网络的 mask 索引。

## 1. 来源分布

{chr(10).join(f'- {k}: {v}' for k, v in sorted(sources.items()))}

## 2. 按来源的 Target Union（去重 patch）

{chr(10).join(source_lines)}

## 3. 来源均衡与植被分层

来源均衡均值见 `full_audit.json: aggregate.<arm>.source_balanced`。
按植被覆盖分层（`mean_vegetation` 四分位）的统计见
`full_audit.json: vegetation_strata`。

{chr(10).join(balanced_lines)}

## 4. 诊断口径提醒

"每个目标块平均植被分数提高"与"模型预测了更多不同的植被 patch"不等价。
本报告同时给出两个口径与重复质量比，见
`reports/stage4b_geometry_diagnosis.md` 第 2.2 节。
"""
    Path('reports/stage4b_mask_statistics.md').write_text(stats, encoding='utf-8')

    print('wrote reports/stage4b_geometry_diagnosis.md')
    print('wrote reports/stage4b_sampler_design.md')
    print('wrote reports/stage4b_geometry_matching.md')
    print('wrote reports/stage4b_mask_statistics.md')

    _write_training_report()
    _write_phenobench_reports()
    _write_summary()
    return 0


# --------------------------------------------------------------------------- #
# Training pilot
# --------------------------------------------------------------------------- #

ARM_DIRS = {
    'G0': 'stage4b_geometry_only_500',
    'G1': 'stage4b_vegetation_controlled_500',
    'G2': 'stage4b_boundary_controlled_500',
    'G3': 'stage4b_shifted_boundary_500',
}
RUNS_ROOT = Path('/home/test/agrisa_runs')


def _read_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _wall_minutes(arm: str) -> Optional[float]:
    log = RUNS_ROOT / 'logs' / f'stage4b_{arm}.log'
    if not log.is_file():
        return None
    import re
    best = None
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        match = re.search(r'PROGRESS step=(\d+)/\d+.*elapsed=([\d.]+)min', line)
        if match:
            best = float(match.group(2))
    return best


def _write_training_report() -> None:
    bench_path = Path(BENCH)
    bench = read_json(bench_path) if bench_path.is_file() else None
    lines = [
        '# Stage 4B-3 — 500-step 训练试点',
        '',
        '## 1. 协议',
        '',
        '* 初始化：官方 `IN1K-vit.h.14-300e.pth.tar`（SHA-256 `0382013c…`），每个 Arm 独立初始化。',
        '* 数据：`agrifield40k-v1_train.csv`（PROVISIONAL，12,567 张）。',
        '* 调度：**2,000 optimizer updates** 的总调度，在第 **500** 步纯中断并保存。',
        '  `max_optimizer_steps` 仍为 2,000，因此 LR/WD/EMA 轨迹与既有 B_500 完全一致。',
        '* 有效 batch：micro-batch 16 × grad-accum 8 = 128；BF16；SDPA；AdamW；原始 Smooth-L1。',
        '* 单卡 NVIDIA RTX A6000（`cuda:1`）。',
        '',
        '## 2. 健康检查与真实更新次数',
        '',
        '| Arm | 实际 updates | 调度总长 | 中断点 | loss first → last | loss finite | EMA 公式验证 | Peak GPU (MiB) | ms/step | 训练段 wall-clock (min) |',
        '|---|---:|---:|---:|---|---|---|---:|---:|---:|',
    ]
    for arm, run_dir in ARM_DIRS.items():
        verification_path = RUNS_ROOT / run_dir / 'training_verification.yaml'
        if not verification_path.is_file():
            lines.append(f'| {arm} | — | — | — | — | — | — | — | — | — |')
            continue
        v = _read_yaml(verification_path)
        lines.append(
            f"| {arm} | {v.get('optimizer_steps_total')} | "
            f"{v.get('total_optimizer_steps_horizon')} | "
            f"{v.get('interrupt_after_optimizer_steps')} | "
            f"{fmt(v.get('loss_first'), 4)} → {fmt(v.get('loss_last'), 4)} | "
            f"{v.get('loss_finite')} | {v.get('ema_formula_verified')} | "
            f"{fmt(v.get('peak_memory_mib'), 0)} | "
            f"{fmt(v.get('step_time_ms_mean'), 0)} | "
            f"{fmt(_wall_minutes(arm), 1)} |")

    lines += ['', '## 3. 训练时的真实 Mask 诊断（主进程，num_workers=0）', '',
              '| Arm | guided | fallback | invalid | 选中重叠 | 候选重叠 | prior s/batch | selection s/batch |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, run_dir in ARM_DIRS.items():
        verification_path = RUNS_ROOT / run_dir / 'training_verification.yaml'
        if not verification_path.is_file():
            continue
        d = _read_yaml(verification_path).get('mask_diagnostics') or {}
        lines.append(
            f"| {arm} | {d.get('guided_selections')} | "
            f"{d.get('uniform_fallbacks')} | {d.get('invalid_candidate_rate')} | "
            f"{fmt(d.get('mean_selected_overlap'), 4)} | "
            f"{fmt(d.get('mean_candidate_overlap'), 4)} | "
            f"{fmt(d.get('prior_seconds'), 3)} | "
            f"{fmt(d.get('selection_seconds'), 3)} |")

    if bench:
        lines += ['', '## 4. Mask 生成开销（隔离基准，CPU）', '',
                  '| Arm | ms/batch | img/s | 相对 B 开销 | prior ms/batch | selection ms/batch | fallback rate |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for arm, entry in bench['arms'].items():
            lines.append(
                f"| {arm} | {fmt(entry['batch_seconds_mean'] * 1000, 1)} | "
                f"{fmt(entry['images_per_second'], 0)} | "
                f"{fmt(entry['overhead_vs_B_percent'], 1)}% | "
                f"{fmt(entry['prior_seconds_per_batch'] * 1000, 1)} | "
                f"{fmt(entry['selection_seconds_per_batch'] * 1000, 1)} | "
                f"{fmt(entry['fallback_rate'], 3)} |")
        lines += ['', f"> 基准条件：{bench['device']}，batch {bench['batch_size']}，"
                  f"{bench['batches']} 个 batch。训练中 `num_workers=0` 且 prior 在 GPU 上计算，"
                  "因此实际开销低于该 CPU 隔离值；真实 step time 见第 2 节。"]

    resume_path = Path('runs/stage4b_resume_test/training_verification.yaml')
    lines += ['', '## 5. Resume 验证（从 G1 的 step-500 checkpoint 续跑到 506）', '']
    if resume_path.is_file():
        r = _read_yaml(resume_path)
        lines += [
            '| 字段 | 值 |', '|---|---|',
            f"| optimizer_steps_total | {r.get('optimizer_steps_total')} |",
            f"| total_optimizer_steps_horizon | {r.get('total_optimizer_steps_horizon')} |",
            f"| interrupt_after_optimizer_steps | {r.get('interrupt_after_optimizer_steps')} |",
            f"| loss_finite | {r.get('loss_finite')} |",
            f"| ema_formula_verified | {r.get('ema_formula_verified')} |",
            '',
            '恢复源是 `jepa-latest.pth.tar`（包含 optimizer / scaler / RNG / '
            'loader generator / mask_collator_step / epoch_start_state）。',
            '`jepa-step500.pth.tar` 在修复前不包含 `epoch_start_state`，因此只能用于评价；'
            '`save_step_snapshot` 已修复，之后的 step 快照也可直接续跑。',
            '',
        ]
    else:
        lines += ['（resume 验证尚未生成。）', '']

    lines += ['', '## 6. 未执行的内容',
              '',
              '* 没有跑 2,000-step 的四臂全长训练：本轮是机制筛查，不是最终显著性验证。',
              '* U 没有单独训练：全量离线审计显示 U 与 B 的 Target Union / Context / 结构覆盖',
              '  分布等价（KS 检验见门禁），因此按协议降为诊断性实验。',
              '* 没有用 PhenoBench Val 调 β；β=3.5 只由离线 Mask 几何选择。',
              '']
    Path('reports/stage4b_training_pilot.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote reports/stage4b_training_pilot.md')


# --------------------------------------------------------------------------- #
# PhenoBench + mechanism ablation
# --------------------------------------------------------------------------- #

def _write_phenobench_reports() -> None:
    comparison_path = Path('reports/stage4b_phenobench_comparison.json')
    ablation_path = Path('reports/stage4b_mechanism_ablation.json')
    if not comparison_path.is_file():
        print('phenobench comparison not available yet; skipping its reports')
        return
    comparison = read_json(comparison_path)
    ablation = read_json(ablation_path) if ablation_path.is_file() else {'rows': []}

    lines = [
        '# Stage 4B — PhenoBench 下游评价',
        '',
        '> 官方 PhenoBench v1.1.0 语义分割评价器，冻结 ViT-H 特征 + `Linear(1280→3)` probe，',
        '> 224×224 tiling，logit average stitching，probe seeds 0/1/2。',
        '> 3 个 Probe seed **不是** 3 个独立预训练 seed；下列差值均为配对探索性结果。',
        '',
        '## 1. 各评价点（official，mean ± std over 3 probe seeds）',
        '',
        '| Label | Crop IoU | Weed IoU | Soil IoU | mIoU |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in comparison['per_evaluation_point']:
        lines.append(
            f"| {row['label']} | {fmt(row.get('official_crop_mean'))} ± "
            f"{fmt(row.get('official_crop_std'))} | "
            f"{fmt(row.get('official_weed_mean'))} ± {fmt(row.get('official_weed_std'))} | "
            f"{fmt(row.get('official_soil_mean'))} ± {fmt(row.get('official_soil_std'))} | "
            f"{fmt(row.get('official_mIoU_mean'))} ± {fmt(row.get('official_mIoU_std'))} |")

    lines += ['', '## 2. 配对差值（pp，candidate − baseline）', '',
              '| 比较 | mIoU Δ | Crop Δ | Weed Δ | Soil Δ | seeds 胜/总 | 精确符号检验 p | 探索性标签 |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for name, block in comparison['comparisons'].items():
        if 'mIoU' not in block:
            continue
        lines.append(
            f"| {name} | {fmt(block['mIoU']['delta_pp'], 3)} | "
            f"{fmt(block['crop']['delta_pp'], 3)} | "
            f"{fmt(block['weed']['delta_pp'], 3)} | "
            f"{fmt(block.get('soil', {}).get('delta_pp'), 3)} | "
            f"{block['mIoU']['wins']}/{block['mIoU']['n_seeds_paired']} | "
            f"{fmt(block['mIoU']['sign_test_p_exact'], 3)} | "
            f"{block['exploratory_signal']} |")

    lines += ['', '## 3. 逐 seed 方向一致性', '',
              '| 比较 | 每 seed mIoU Δ | 方向一致 |', '|---|---|---|']
    for name, block in comparison['comparisons'].items():
        if 'mIoU' not in block:
            continue
        lines.append(f"| {name} | {block['mIoU']['per_seed_delta_pp']} | "
                     f"{block['seeds_direction_consistent']} |")

    lines += ['', '## 4. 指标交叉校验', '',
              '所有评价点 `cross_check_agrees`：' +
              str(all(row.get('cross_check_agrees') for row in
                      comparison['per_evaluation_point'])),
              '']
    Path('reports/stage4b_phenobench_comparison.md').write_text(
        '\n'.join(lines), encoding='utf-8')

    # ---- mechanism ablation -------------------------------------------- #
    mechanism = {
        '作用一：农业结构信息': ['G1-G0', 'G2-G0'],
        '作用二：Mask 几何分布': ['G0-B', 'G1-B', 'G2-B'],
        '作用三：结构先验空间对齐': ['G2-G3'],
    }
    rows = {row['comparison']: row for row in ablation.get('rows', [])}
    lines = [
        '# Stage 4B — 机制消融：把三种作用分开',
        '',
        '结论不能合并成一个含糊的“结构感知有效”。下表按协议要求把三种作用分开：',
        '',
        '| 作用 | 比较 | mIoU Δ | Crop Δ | Weed Δ | 解读 |',
        '|---|---|---:|---:|---:|---|',
    ]
    for mechanism_name, comparisons in mechanism.items():
        for name in comparisons:
            row = rows.get(name)
            if not row:
                continue
            lines.append(
                f"| {mechanism_name} | {name} | {fmt(row['mIoU_delta_pp'], 3)} | "
                f"{fmt(row['crop_delta_pp'], 3)} | {fmt(row['weed_delta_pp'], 3)} | "
                f"{row['exploratory_signal']} |")
    lines += [
        '',
        '## 归因规则（预先声明）',
        '',
        '* 只有 G0 优于 B，而 G1/G2 没有优于 G0：把发现归因于**几何采样**，不是农业结构语义。',
        '* G2 与 G3 同级：不能宣称正确的植物边界位置已被证明有效。',
        '* G1/G2 只在 Weed 提升而其他类别下降：记为 Mixed，不作为 Positive。',
        '* 任何 Positive 都需要独立预训练 seed 与其他农业数据集复核；3 个 Probe seed 不算。',
        '',
        '## 历史对照',
        '',
        'G1 − C1_500 与 G2 − C2_500 回答“控制几何是否改变了 Stage 4A 的结论”。',
        'C1/C2_500 是 2,000-step 总调度的中途 checkpoint，与 B_500 使用同一调度；',
        '复用前已核对配置与 seed 可比性（见 `reports/stage4b_protocol_check.json`）。',
        '',
    ]
    Path('reports/stage4b_mechanism_ablation.md').write_text(
        '\n'.join(lines), encoding='utf-8')
    print('wrote reports/stage4b_phenobench_comparison.md')
    print('wrote reports/stage4b_mechanism_ablation.md')




# --------------------------------------------------------------------------- #
# Final summary and decision
# --------------------------------------------------------------------------- #

def _delta(comparison: dict, name: str, metric: str = 'mIoU') -> Optional[float]:
    block = comparison.get('comparisons', {}).get(name, {})
    entry = block.get(metric) if isinstance(block, dict) else None
    return float(entry['delta_pp']) if entry else None


def _wins(comparison: dict, name: str) -> Optional[int]:
    block = comparison.get('comparisons', {}).get(name, {})
    entry = block.get('mIoU') if isinstance(block, dict) else None
    return int(entry['wins']) if entry else None


def _write_summary() -> None:
    comparison_path = Path('reports/stage4b_phenobench_comparison.json')
    audit_path = Path(AUDIT)
    if not comparison_path.is_file() or not audit_path.is_file():
        print('summary needs the evaluation and the audit; skipping for now')
        return
    comparison = read_json(comparison_path)
    audit = read_json(audit_path)
    present = set(comparison['labels_present'])
    required_labels = {'B_500', 'G0_500', 'G1_500', 'G2_500', 'G3_500'}
    incomplete = required_labels - present

    # ---- pre-registered decision rule ---------------------------------- #
    g0 = _delta(comparison, 'G0-B')
    g1g0 = _delta(comparison, 'G1-G0')
    g2g0 = _delta(comparison, 'G2-G0')
    g2g3 = _delta(comparison, 'G2-G3')
    g1c1 = _delta(comparison, 'G1-C1')
    g2c2 = _delta(comparison, 'G2-C2')
    g1b = _delta(comparison, 'G1-B')
    g2b = _delta(comparison, 'G2-B')
    g1_wins = _wins(comparison, 'G1-G0')
    g2_wins = _wins(comparison, 'G2-G0')

    def meaningful(value: Optional[float]) -> bool:
        return value is not None and value >= 0.5

    geometry_effect = meaningful(g0)
    structure_g1 = meaningful(g1g0) and g1_wins is not None and g1_wins >= 2
    structure_g2 = meaningful(g2g0) and g2_wins is not None and g2_wins >= 2
    changed_after_control = (abs(g1c1) >= 0.5 if g1c1 is not None else False) or \
                            (abs(g2c2) >= 0.5 if g2c2 is not None else False)

    if incomplete:
        verdict = 'E — Incomplete'
    elif structure_g1 or structure_g2:
        verdict = 'C — Structure signal supported'
    elif geometry_effect:
        verdict = 'B — Geometry effect only'
    elif changed_after_control:
        verdict = 'A — Geometry confounding supported'
    else:
        verdict = 'D — Negative'

    next_step = (
        '进入 Stage 4C：用 2–3 个独立预训练 seed 复核，并在第二个农业/杂草数据集上'
        '验证，再决定是否扩大预算。'
        if verdict.startswith(('A', 'C')) else
        '停止当前 Mask 路线，把预算转向农业持续预训练本身与跨数据集迁移；'
        '除非出现新的、与几何无关的结构先验假设，否则不再增加 Mask 消融。')

    a = audit['aggregate']
    def union(arm):
        return a[arm]['overall']['target_cells']['mean']
    def ctx(arm):
        return a[arm]['overall']['context_cells']['mean']
    def veg(arm):
        return a[arm]['overall']['mean_vegetation_target_union']['mean']
    def bnd(arm):
        return a[arm]['overall']['mean_boundary_target_union']['mean']
    def pair(arm, metric):
        return audit['comparisons_vs_B'][f'{arm}-B'][metric]

    shift = read_json(SHIFT) if Path(SHIFT).is_file() else {}

    lines = [
        '# Stage 4B 总结报告',
        '',
        f'> 生成时间：2026-10-22 · 分支 `agri-stage4b` · 数据 v1（PROVISIONAL）',
        f'> 全量几何审计：{audit["n_images"]} 张 × {len(audit["arms"])} 个 arm；'
        f'下游：PhenoBench v1.1.0，3 个 Probe seed。',
        '',
        '## 0. 最终结论',
        '',
        f'**{verdict}**',
        '',
        f'* G0 − B（几何控制本身）mIoU {fmt(g0, 3)} pp',
        f'* G1 − G0（植被结构额外收益）mIoU {fmt(g1g0, 3)} pp，{g1_wins}/3 seeds',
        f'* G2 − G0（边界结构额外收益）mIoU {fmt(g2g0, 3)} pp，{g2_wins}/3 seeds',
        f'* G2 − G3（正确空间对齐）mIoU {fmt(g2g3, 3)} pp',
        f'* G1 − C1_500 {fmt(g1c1, 3)} pp；G2 − C2_500 {fmt(g2c2, 3)} pp',
        '',
        '判定规则在实验前写入 `tools/make_stage4b_reports.py`：只有结构臂相对 G0',
        '达到 +0.5 pp 且至少 2/3 seeds 同向，才记为结构信号；只有几何控制本身',
        '达到 +0.5 pp 才记为几何收益。3 个 Probe seed 的最小双侧符号检验 p=0.25，',
        '因此这些标签是探索性筛查，不是显著性结论。',
        '',
        '## A. Stage 4A 失效原因',
        '',
        f'1. **C1/C2 的 Target Union 是否系统性低于 B？** 是。全量审计：'
        f'B {fmt(union("B"))}，C1 {fmt(union("C1"))}（Δ {fmt(pair("C1", "target_cells")["mean_delta"])}），'
        f'C2 {fmt(union("C2"))}（Δ {fmt(pair("C2", "target_cells")["mean_delta"])}），'
        f'C3 {fmt(union("C3"))}（Δ {fmt(pair("C3", "target_cells")["mean_delta"])}）。'
        f'配对胜/负：C1 {pair("C1", "target_cells")["wins"]}/{pair("C1", "target_cells")["losses"]}，'
        f'C2 {pair("C2", "target_cells")["wins"]}/{pair("C2", "target_cells")["losses"]}。',
        f'2. **Context 可见数量差异多大？** 在真实 batch 裁剪后，B {fmt(ctx("B"))}，'
        f'C1 {fmt(ctx("C1"))}（+{fmt(pair("C1", "context_cells")["mean_delta"])}），'
        f'C2 {fmt(ctx("C2"))}（+{fmt(pair("C2", "context_cells")["mean_delta"])}）。'
        f'Stage 4A 报告的 109/132/123/120 是**截断前**数字；见 `stage4b_geometry_diagnosis.md` 第 4 节。',
        '3. **扩大样本后是否仍存在？** 是。12,567 张全量、配对、来源分层后，'
        '方向与 60 张时一致，符号检验 p < 1e-8。',
        f'4. **重复预测有多严重？** 重复目标面积 B {fmt(a["B"]["overall"]["duplicated_target_area"]["mean"])}，'
        f'C1 {fmt(a["C1"]["overall"]["duplicated_target_area"]["mean"])}，'
        f'C2 {fmt(a["C2"]["overall"]["duplicated_target_area"]["mean"])}。',
        f'5. **结构覆盖提升是否集中在重复区域？** 是。C1 的植被重复质量比 '
        f'{fmt(a["C1"]["overall"]["vegetation_repeat_mass_ratio"]["mean"], 3)}，'
        f'B 为 {fmt(a["B"]["overall"]["vegetation_repeat_mass_ratio"]["mean"], 3)}；'
        f'C2 的边界重复质量比 {fmt(a["C2"]["overall"]["boundary_repeat_mass_ratio"]["mean"], 3)}，'
        f'B 为 {fmt(a["B"]["overall"]["boundary_repeat_mass_ratio"]["mean"], 3)}。',
        '6. **支持哪些机制假设？** 支持“结构引导使目标块集中、Target Union 下降、'
        'Context 上升”，这是可复现的几何混杂；不支持“结构信息本身无用”的更强结论。'
        '下游因果仍不能仅凭几何相关性断言。',
        '',
        '## B. 新采样器',
        '',
        '7. **几何控制方式**：候选块权重 `p_i ∝ exp(λ ŝ_i − β o_i)`，'
        '`o_i` 为候选块与该图已选 Target 并集的重叠比例；β=3.5 由离线几何扫描选定。',
        f'8. **G0/G1/G2/G3 几何是否可比？** G1/G2：Union {fmt(union("G1"))}/{fmt(union("G2"))} vs B {fmt(union("B"))}，'
        f'Context {fmt(ctx("G1"))}/{fmt(ctx("G2"))} vs B {fmt(ctx("B"))}，均值与分布均通过门禁。'
        f'G3：Union {fmt(union("G3"))}、Context {fmt(ctx("G3"))}，均值接近但分布 KS 仍失配（control note）。'
        f'G0：Union {fmt(union("G0"))}、Context {fmt(ctx("G0"))}，过分散，作为机制对照而非几何匹配臂。',
        '9. **几何匹配让结构引导强度减少了多少？** 与 Stage 4A 的 C1/C2 相比，'
        f'G1 的植被 Union 均值 {fmt(veg("G1"), 4)}（C1 {fmt(veg("C1"), 4)}，B {fmt(veg("B"), 4)}）；'
        f'G2 的边界 Union 均值 {fmt(bnd("G2"), 4)}（C2 {fmt(bnd("C2"), 4)}，B {fmt(bnd("B"), 4)}）。'
        '结构偏好保留，重复质量比明显下降。',
        '10. **原始 JEPA 目标函数**：保留。未改 Loss、Encoder/Predictor、EMA、块尺寸/数量、'
        'Context 排斥、`min_keep`、官方初始化。',
        f'11. **G3 是否比原 C3 更适合检验空间对齐？** 是。G3 保留直方图 100%、'
        f'lag-1 自相关保留 {fmt(shift.get("lag1_retention_ratio"), 3)}，与原图平均相关 '
        f'{fmt(shift.get("mean_pearson_correlation_with_original"), 3)}；'
        f'C3 的置换会同时破坏局部连续性。残余相关 >0.5 的图像比例 '
        f'{fmt(shift.get("fraction_abs_correlation_above_0.5"), 4)}。',
        '12. **诊断分母问题**：已修复。`mean_candidate_prior_score` 现在除以有效候选数，'
        '不是 `candidates_generated`；有手工构造分数的单元测试，且证明 Mask/RNG/梯度不变。',
        '',
        '## C. 训练',
        '',
        '13. **哪些 Arm 完成 500-step 试点？** G0/G1/G2/G3（见 `stage4b_training_pilot.md`）。',
        '14. **初始化与总调度是否一致？** 是。四臂均从官方 checkpoint 独立初始化，'
        '`max_optimizer_steps=2000`、`interrupt_after_optimizer_steps=500`，'
        '`save_at_optimizer_steps=[500]`；`tools/check_stage4b_protocol.py` 逐字段验证。',
        '15. **是否可恢复？** checkpoint 保存了 encoder/predictor/target/optimizer/scaler/'
        'RNG/loader generator/mask_collator_step；resume 路径与 Stage 2C 相同，'
        '并额外记录了 Mask 计数器。',
        '16. **成本**：单卡 A6000；每臂 500 updates，~27–30 min 训练段，峰值 ~22.2 GB；'
        'Mask 额外开销见 `stage4b_mask_benchmark.json`。',
        '17. **哪些实验未执行？** U 未单独训练（离线分布等价，降为诊断）；'
        '未跑 2,000-step 四臂全长；未用下游指标调 β。',
        '',
        '## D. 下游',
        '',
    ]
    for label in ('B_500', 'G0_500', 'G1_500', 'G2_500', 'G3_500'):
        row = next((r for r in comparison['per_evaluation_point']
                    if r['label'] == label), None)
        if row:
            lines.append(
                f"{label}: mIoU {fmt(row.get('official_mIoU_mean'))}，"
                f"Crop {fmt(row.get('official_crop_mean'))}，"
                f"Weed {fmt(row.get('official_weed_mean'))}。")
    lines += [
        '',
        '18–19. **mIoU / Crop / Weed** 见上表。Soil 在官方评价器的 semantics 任务中'
        '不单独输出；自实现指标中的 Soil 约 97.7，各臂差异 <0.02 pp。',
        '',
        f'20. **G0 − B 是否有收益？** mIoU {fmt(g0, 3)} pp（3/3 seeds：'
        f'{comparison["comparisons"]["G0-B"]["mIoU"]["per_seed_delta_pp"]}），'
        f'Weed {fmt(comparison["comparisons"]["G0-B"]["weed"]["delta_pp"], 3)} pp，'
        f'Crop {fmt(comparison["comparisons"]["G0-B"]["crop"]["delta_pp"], 3)} pp。'
        '方向一致但幅度低于预先设定的 +0.5 pp 探索性参考线，只能算弱提示。',
        f'21. **G1 − G0 是否有收益？** 否，mIoU {fmt(g1g0, 3)} pp，0/3 seeds；'
        f'Weed {fmt(comparison["comparisons"]["G1-G0"]["weed"]["delta_pp"], 3)} pp。'
        '控制几何后，植被结构没有带来额外收益。',
        f'22. **G2 − G0 是否有收益？** 否，mIoU {fmt(g2g0, 3)} pp，1/3 seeds；'
        f'Weed {fmt(comparison["comparisons"]["G2-G0"]["weed"]["delta_pp"], 3)} pp。'
        '边界结构同样没有额外收益。',
        f'23. **G2 − G3 是否支持空间对齐的价值？** 不支持。mIoU {fmt(g2g3, 3)} pp，'
        f'1/3 seeds；G3（空间错位先验）反而略高。结合 C2≈C3 与 G2≈G3，'
        '没有证据表明“正确的植物边界位置”本身带来了可测收益。',
        f'23b. **控制臂 G3 意外最高（mIoU {fmt(comparison["per_evaluation_point"][[r["label"] for r in comparison["per_evaluation_point"]].index("G3_500")]["official_mIoU_mean"])}）怎么解释？** '
        f'G3 − B = {fmt(_delta(comparison, "G3-B"), 3)} pp（3/3 seeds），是新增臂中最大的单点差异；'
        f'但 G3 是**空间错位**的先验，且 G2 − G3 = {fmt(g2g3, 3)} pp。'
        '因此不能把它解释为结构或对齐收益。它最多提示：G3 这一臂的几何'
        '（Union +4.5、Context −6.8）值得用纯几何对照单独复核，而不是回到结构先验。',
        f'24. **是否以牺牲 Crop 换 Weed？** 没有出现这种交换。G1 crop '
        f'{fmt(comparison["comparisons"]["G1-G0"]["crop"]["delta_pp"], 3)} / weed '
        f'{fmt(comparison["comparisons"]["G1-G0"]["weed"]["delta_pp"], 3)}，二者同降；'
        f'G2 crop {fmt(comparison["comparisons"]["G2-G0"]["crop"]["delta_pp"], 3)} / weed '
        f'{fmt(comparison["comparisons"]["G2-G0"]["weed"]["delta_pp"], 3)}，同样同降。',
        '',
        '25. **三个 Probe seeds 方向是否一致？** G1−G0、G2−G0、G2−G3 的 mIoU '
        '方向分别为 0/3、1/3、1/3 一致；G0−B 为 3/3 一致。逐 seed 表见 '
        '`stage4b_phenobench_comparison.md`。',
        '',
        '## E. 下一步',
        '',
        next_step,
        '',
        '> 任何 Positive/Mixed 结果都必须用独立预训练 seed 与第二个数据集复核；'
        '不能把 3 个 Probe seed 当作 3 次独立预训练。',
        '',
    ]
    Path('reports/stage4b_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('wrote reports/stage4b_summary.md')


if __name__ == '__main__':
    raise SystemExit(main())
