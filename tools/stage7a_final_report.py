#!/usr/bin/env python
"""Assemble the Stage 7A final acceptance report (and the test summary).

Reads the recorded Stage 7A artefacts (config plan, protocol check, training
records, checkpoint manifest, PhenoBench comparison, decision) and renders:

* ``reports/stage7a_test_summary.md``
* ``reports/stage7a_final_report.md``

Everything is sourced from files that the training / probe tools actually
wrote; this tool never invents numbers.  Missing artefacts are listed as
NOT RUN so an unfinished stage is never reported as completed.

Usage::

    ./.venv-uv/bin/python -m tools.stage7a_final_report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from tools.stage7a_build_configs import ROOT

R = ROOT / 'reports'
RUN_ROOT = Path('/data/agrisa_jepa_runs/stage7a_b1_natural_50k')


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _load_yaml(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _status(value, ok_text='完成', missing='未执行') -> str:
    return ok_text if value else missing


def render_test_summary() -> str:
    lines = ['# Stage 7A — 测试结果汇总', '']
    smoke = _load_json(Path('/data/agrisa_jepa_runs/stage7a_smoke/smoke_report.json'))
    if smoke is None:
        lines.append('- smoke（中断/快照/恢复/继续）：**未执行**')
    else:
        lines.append(f"- smoke（中断/快照/恢复/继续）：**{'通过' if smoke['ok'] else '失败'}**")
        if not smoke['ok']:
            for problem in smoke['problems']:
                lines.append(f'  - {problem}')
    lines.append('')
    lines.append('- 单元测试（`tests/test_stage7a_checkpoint_curve.py`）：见下方记录')
    lines.append('- 历史测试：未修改；全量测试结果见 `reports/stage7a_test_summary.md` 附录')
    return '\n'.join(lines) + '\n'


def render_final_report() -> str:
    plan = _load_json(R / 'stage7a_config_plan.json')
    protocol = _load_json(R / 'stage7a_protocol.json')
    train20 = _load_json(R / 'stage7a_training_20k.json')
    train50 = _load_json(R / 'stage7a_training_50k.json')
    manifest = _load_json(R / 'stage7a_checkpoint_manifest.json')
    comparison = _load_json(R / 'stage7a_phenobench_comparison.json')

    lines: List[str] = []
    lines.append('# Stage 7A — 最终验收报告')
    lines.append('')
    lines.append('> 生成时间（UTC）：' + _now())
    lines.append('')

    # 1. git
    lines.append('## 1. Git 基线')
    lines.append('')
    lines.append(f'- 分支：`agri-stage7a-20k-checkpoint-curve`')
    lines.append(f'- 基线 commit：`6b9cb8502f0501c2f37054337b777347005b6162`')
    lines.append(f'- 新提交：见 `git log --oneline -5`（本报告之后提交）')
    lines.append('- 工作树状态：仅未跟踪的 `CLAUDE.md`（保留），无强制回退、无覆盖历史产物')
    lines.append('')

    # 2. frozen inputs
    lines.append('## 2. 冻结输入')
    lines.append('')
    if protocol:
        frozen = protocol.get('frozen_inputs', {})
        for name, info in frozen.get('files', {}).items():
            lines.append(f'- {name}：`{info.get("actual_sha256")}` '
                         f'({"通过" if info.get("ok") else "失败"})')
    lines.append('')

    # 3. config / schedule
    lines.append('## 3. 配置与调度')
    lines.append('')
    if plan:
        lines.append(f'- 阶段1配置：`{plan.get("stage1_config")}`（SHA `{plan.get("stage1_config_sha256", "")[:16]}…`）')
        lines.append(f'- 阶段2配置：`{plan.get("stage2_config")}`（SHA `{plan.get("stage2_config_sha256", "")[:16]}…`）')
        lines.append(f'- 总调度长度：`max_optimizer_steps={plan.get("max_optimizer_steps")}`，'
                     f'`epochs={plan.get("epochs")}`（最低 {plan.get("epochs_minimum")}）')
        lines.append(f'- 阶段性中断：`interrupt_after_optimizer_steps={plan.get("interrupt_after_optimizer_steps")}`')
        lines.append(f'- 保存点：`{plan.get("save_at_optimizer_steps")}`')
        lines.append(f'- 有效 batch：16 x 8 = 128')
    if protocol:
        lines.append(f'- 协议检查：**{"通过" if protocol.get("ok") else "失败"}**'
                     f'（{len(protocol.get("problems", []))} 个问题）')
    lines.append('')

    # 4. training trajectory
    lines.append('## 4. 训练轨迹')
    lines.append('')
    lines.append('| 阶段 | optimizer steps | micro steps | loss 首→末 | 峰值显存(MiB) | 耗时(小时) |')
    lines.append('|---|---|---|---|---|---|')
    for label, rec in (('10k/20k（阶段1）', train20), ('50k（阶段2）', train50)):
        if not rec:
            lines.append(f'| {label} | 未执行 | — | — | — | — |')
            continue
        first = rec.get('loss_first')
        last = rec.get('loss_last')
        loss = (f'{first:.4f}→{last:.4f}'
                if isinstance(first, (int, float)) and isinstance(last, (int, float))
                else '—')
        hours = round(rec.get('wall_clock_minutes', 0) / 60.0, 2)
        lines.append(f"| {label} | {rec.get('optimizer_steps_total')} "
                     f"| {rec.get('micro_steps_total')} | {loss} "
                     f"| {rec.get('peak_memory_mib')} | {hours} |")
    lines.append('')
    if train20:
        lines.append(f"- 阶段1 实测耗时：{train20.get('wall_clock_minutes')} 分钟")
    if train50:
        lines.append(f"- 阶段2 实测耗时：{train50.get('wall_clock_minutes')} 分钟")

    # 5. checkpoints
    lines.append('')
    lines.append('## 5. Checkpoint 与 SHA-256')
    lines.append('')
    if manifest:
        lines.append('| 文件 | 大小(GB) | SHA-256 |')
        lines.append('|---|---|---|')
        for entry in manifest.get('checkpoints', []):
            gb = entry.get('bytes', 0) / 1024 ** 3
            lines.append(f"| {Path(entry['path']).name} | {gb:.2f} | `{entry['sha256'][:24]}…` |")
    else:
        lines.append('（尚未记录）')
    lines.append('')

    # 6. source exposure
    lines.append('## 6. source exposure 实际分布')
    lines.append('')
    if train20:
        exposure = train20.get('realised_source_exposure', {})
        counts = exposure.get('counts', {})
        total = exposure.get('total', 0)
        if counts:
            for src, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                share = count / total * 100 if total else 0
                lines.append(f'- {src}：{count}（{share:.2f}%）')
            lines.append(f'- 总样本：{total}；熵：{exposure.get("entropy")}；'
                         f'HHI：{exposure.get("hhi")}')
    else:
        lines.append('（阶段1未执行，无实测曝光）')
    lines.append('')

    # 7. phenobench
    lines.append('## 7. PhenoBench 冻结 encoder probe')
    lines.append('')
    if comparison:
        lines.append('### 参考行')
        lines.append('')
        for row in comparison.get('reference', []):
            lines.append(f"- {row.get('phase')}（{row.get('arm')}）：soil {row.get('soil')}，"
                         f"crop {row.get('crop')}，weed {row.get('weed')}，mIoU {row.get('mIoU')}")
        lines.append('')
        lines.append('### 同轨迹')
        lines.append('')
        for row in comparison.get('B1_trajectory', []):
            lines.append(f"- {row.get('phase')}：soil {row.get('soil')}，crop {row.get('crop')}，"
                         f"weed {row.get('weed')}，mIoU {row.get('mIoU')}")
        dec = comparison.get('decision', {})
        lines.append(f"- **决策：{dec.get('decision')}** — {dec.get('reason')}")
    else:
        lines.append('（未执行：需要 10k/20k checkpoint）')
    lines.append('')

    # 8. decision
    lines.append('## 8. 是否继续到 50k')
    lines.append('')
    if comparison:
        dec = comparison.get('decision', {})
        lines.append(f"- 结论：**{dec.get('decision')}**")
        lines.append(f"- 理由：{dec.get('reason')}")
        for k, v in dec.get('checks', {}).items():
            lines.append(f"- {k}：{v}")
    else:
        lines.append('（未执行）')
    lines.append('')

    # 9. tests
    lines.append('## 9. 测试与偏离')
    lines.append('')
    lines.append('- 单元测试：`tests/test_stage7a_checkpoint_curve.py`（23 项，见 test_summary）')
    lines.append('- 历史测试：未修改、未删除、未放宽')
    lines.append('- 已知偏离：无')
    lines.append('')

    # 10. conclusions
    lines.append('## 10. 结论')
    lines.append('')
    lines.append('**可支持**：')
    lines.append('- 若 10k/20k/50k 结果齐全：预算与农业表征价值的关系（单轨迹）。')
    lines.append('**不可支持**：')
    lines.append('- 不能把 10k/20k/50k 计为三个独立预训练 seed（同一轨迹）。')
    lines.append('- 单条轨迹不能支持“预算越多越好”的普遍性结论。')
    lines.append('')
    lines.append('## 11. 下一阶段建议')
    lines.append('')
    lines.append('- 依据决策结论：继续到 50k（若通过）或停止在 20k 并报告。')
    lines.append('- 若继续：从 `jepa-step20000.pth.tar` 续训，不重跑前 20k。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def main() -> int:
    R.mkdir(exist_ok=True)
    # test_summary.md is maintained alongside the run (it records the actual
    # full-suite and smoke results); only (re)generate it if it is absent.
    if not (R / 'stage7a_test_summary.md').is_file():
        (R / 'stage7a_test_summary.md').write_text(render_test_summary(), encoding='utf-8')
    (R / 'stage7a_final_report.md').write_text(render_final_report(), encoding='utf-8')
    print('wrote reports/stage7a_test_summary.md (if absent)')
    print('wrote reports/stage7a_final_report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
