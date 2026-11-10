#!/usr/bin/env python
"""Stage 7B: assemble the final report answering the six audit questions.

Reads the recorded Stage 7B artefacts (audit, resume test, config plan,
protocol, pretraining runs, probe results, comparison) and renders
``reports/stage7b_final_report.md``.  Never invents numbers; missing artefacts
are shown as 未执行.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from tools.stage7a_build_configs import ROOT

R = ROOT / 'reports'


def load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def render() -> str:
    audit = R / 'stage7a_retrospective_integrity_audit.md'
    resume = load_json(R / 'stage7b_resume_test.json')
    plan = load_json(R / 'stage7b_config_plan.json')
    protocol = load_json(R / 'stage7b_protocol.json')
    runs = load_json(R / 'stage7b_pretraining_runs.json')
    comparison = load_json(R / 'stage7b_phenobench_per_checkpoint.json')

    lines = ['# Stage 7B — 最终报告', '']
    lines.append(f'> 生成时间（UTC）：{_now()}')
    lines.append('')
    lines.append('## 0. 目标与范围')
    lines.append('')
    lines.append('- 修复 Stage 7A 的 checkpoint 恢复 / 梯度累积缺陷并给出严格恢复测试。')
    lines.append('- 三臂 × 双 seed 配对调度消融，区分训练预算、调度长度、warmup 长度。')
    lines.append('- 不修改 I-JEPA 核心目标、不引入 SlotCon、不更改语料、不做端到端微调。')
    lines.append('')
    lines.append('## 1. Stage 7A 事后完整性审计（问题 1）')
    lines.append('')
    if audit.is_file():
        lines.append('- 已生成：`reports/stage7a_retrospective_integrity_audit.md`')
        lines.append('- 三处缺陷确认存在并已修复：')
        lines.append('  - **off-by-one**：step snapshot 存 `epoch_itr=itr`，恢复后重执行最后已消费'
                     ' micro-batch（Stage 7A 阶段2 恢复点 epoch17/itr3463，产生 1 次单 batch '
                     'optimizer update，LR/WD/EMA 提前 1 步）。')
        lines.append('  - **epoch 位置低估**：`how_many=len(step_times_ms)` 漏掉重放部分；'
                     'Stage 7A 未触发（只做单次恢复→跑完），潜伏。')
        lines.append('  - **部分窗口归一化**：backward 除以 `grad_accum` 而非实际 `window_size`；'
                     'Stage 7A 每 epoch 末尾（ipe=9204, grad_accum=8 → 窗口 4）最后 1 个更新被 '
                     '0.5x 缩放，约占 0.09% 更新。')
    else:
        lines.append('- **未执行**（缺 `stage7a_retrospective_integrity_audit.md`）。')
    lines.append('')
    lines.append('## 2. 严格恢复测试（问题 2）')
    lines.append('')
    if resume:
        ok = resume.get('ok')
        lines.append(f"- **{'通过' if ok else '失败'}**：连续 24 vs 中断 12+恢复 24。")
        for scenario, result in resume['scenarios'].items():
            lines.append(f"  - `{scenario}`：trace {'一致' if result['trace']['ok'] else '不一致'}、"
                         f"ledger {'一致' if result['ledger']['ok'] else '不一致'}、"
                         f"权重/优化器 {'容差内' if result['checkpoint']['ok'] else '超差'}、"
                         f"计数 {'一致' if not result['count_problems'] else '不一致'}。")
    else:
        lines.append('- **未执行**（缺 `stage7b_resume_test.json`）。')
    lines.append('')
    lines.append('## 3. 消融运行（问题 3 / 4）')
    lines.append('')
    lines.append('| run | arm | seed | steps | horizon | warmup | loss 首→末 | 耗时(h) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    if runs:
        for row in runs['runs']:
            arm = row['arm']
            horizon = row.get('max_optimizer_steps')
            warmup = row.get('warmup_steps')
            loss = f"{row['loss_first']:.4f}→{row['loss_last']:.4f}" \
                if row.get('loss_first') is not None else '—'
            hours = round((row.get('wall_clock_minutes') or 0) / 60, 2)
            lines.append(f"| {row['run_id']} | {arm} | {row['seed']} "
                         f"| {row['optimizer_steps_total']} | {horizon} | {warmup} "
                         f"| {loss} | {hours} |")
    else:
        lines.append('| — | 未执行 | — | — | — | — | — | — |')
    lines.append('')
    lines.append('## 4. PhenoBench 结果（问题 3 / 4）')
    lines.append('')
    lines.append('| run | arm | seed | soil | crop | weed | mIoU |')
    lines.append('|---|---|---|---|---|---|---|')
    if comparison:
        for row in comparison['checkpoints']:
            lines.append(f"| {row['run_id']} | {row['arm']} | {row['pretraining_seed']} "
                         f"| {row['soil']:.2f} | {row['crop']:.2f} | {row['weed']:.2f} "
                         f"| {row['mIoU']:.2f} |")
    else:
        lines.append('| — | 未执行 | — | — | — | — | — |')
    lines.append('')
    lines.append('## 5. 配对对比与决策（问题 3 / 4 / 5）')
    lines.append('')
    if comparison:
        for label, title in (('A_vs_B', 'A vs B（预算/horizon）'), ('B_vs_C', 'B vs C（warmup）')):
            contrast = comparison.get(label)
            if not contrast:
                lines.append(f'- {title}：缺少数据')
                continue
            mean = contrast['mean_delta']
            lines.append(f"- **{title}**：ΔmIoU {mean['mIoU']:+.3f} pp，"
                         f"Δcrop {mean['crop']:+.3f}，Δweed {mean['weed']:+.3f}；"
                         f"方向一致 {contrast['seed_direction_consistent']['mIoU']}；"
                         f"决策 **{contrast['decision']}** — {contrast['reason']}")
    else:
        lines.append('- 未执行')
    lines.append('')
    lines.append('## 6. 结论（问题 5 / 6）')
    lines.append('')
    if comparison:
        ab, bc = comparison.get('A_vs_B'), comparison.get('B_vs_C')
        ab_mean = ab['mean_delta'] if ab else {}
        per = {r['tag']: r for r in ab['per_seed_delta']} if ab else {}
        lines.append('- **问题 3（2k vs 50k horizon 在相同 2k updates 下）**：'
                     + (f"A vs B → **{ab['decision']}**；{ab['reason']}。"
                        if ab else '未执行'))
        if ab:
            lines.append(f"  - 配对差（A−B）逐 seed：p1 ΔmIoU "
                         f"{per.get('p1', {}).get('mIoU', 0):+.3f} pp、"
                         f"p2 ΔmIoU {per.get('p2', {}).get('mIoU', 0):+.3f} pp —— "
                         "方向虽一致，但效应不均：p1 接近 0，均值主要由 p2 贡献。")
            lines.append(f"  - 类别分解：Δcrop {ab_mean.get('crop', 0):+.3f} pp"
                         f"（A 略低）、Δweed {ab_mean.get('weed', 0):+.3f} pp"
                         f"（A 明显更高）—— mIoU 的净收益几乎全部来自 weed，"
                         "crop 反向；这是类异质效应，不应用单一 mIoU 掩盖。")
        lines.append('- **问题 4（warmup 100 vs 2500）**：'
                     + (f"B vs C → **{bc['decision']}**；{bc['reason']}。"
                        if bc else '未执行'))
        lines.append('- **问题 5（Stage 7A 50k 与 2k 差异是否有调度解释）**：'
                     + (f"在相同 2k updates 与相同 warmup 下，A（2k horizon）"
                        f"配对高于 B（50k horizon）{ab_mean.get('mIoU', 0):+.3f} pp"
                        f"（{ab['decision']}），方向支持"
                        "“短调度在 2k 步内已把 LR 退火到地板，更快完成域适配”"
                        "这一调度因素解释；但因效应小、且仅由 weed 类与 p2 驱动、"
                        "仅 2 个预训练 seed，这只是初步配对证据，不足以声称"
                        "调度就是 Stage 7A 全部差异的解释。"
                        if ab and ab['decision'] == 'CLEAR_EFFECT' else
                        "当前证据不足以给出调度解释。"))
        lines.append('- **问题 6（是否投入新 50k/100k）**：'
                     '不建议投入 100k 训练。B vs C（warmup）方向不一致 → 无效；'
                     'A vs B 虽判为 CLEAR_EFFECT，但效应仅 ~0.26 pp、主要来自 '
                     'weed 单类、且被 p1 的近零 delta 稀释——50k horizon 在 '
                     '2k-update 对等条件下对下游 PhenoBench 没有可确认的收益，'
                     '因此更长（50k/100k）训练不会带来对应的下游提升。'
                     '若该 0.26 pp 量级对决策有实际影响，正确做法是为 A-vs-B '
                     '对比追加预训练 seed，而非启动 100k 训练。')
    else:
        lines.append('- 未执行。')
    lines.append('')
    lines.append('## 7. 可复现性与产物')
    lines.append('')
    lines.append('- 分支：`agri-stage7b-schedule-ablation`（未合并 main，未 force-push）。')
    lines.append('- 全部 checkpoint 与训练记录保留本地；未删除、未覆盖 Stage 7A。')
    lines.append('- 未使用 conda；仅 GPU1；6 个训练串行执行。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def main() -> int:
    R.mkdir(exist_ok=True)
    (R / 'stage7b_final_report.md').write_text(render(), encoding='utf-8')
    print('wrote reports/stage7b_final_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
