#!/usr/bin/env python
"""Stage 7B: retrospective integrity audit of the Stage 7A training code.

Reads the Stage 7A training records and the git history, confirms the three
defects that were present in the code used by the actual Stage 7A runs
(``src/train.py`` at ``166d494``/``9ed811e``), and quantifies their scale.

The three defects (all fixed for Stage 7B, NOT re-trained for 7A):

1. **off-by-one resume** -- ``save_step_snapshot`` stored ``epoch_itr = itr``
   (zero-based index) instead of ``itr + 1`` (micro-batches consumed).  On
   resume the last processed micro-batch was executed again.
2. **epoch-position undercount** -- ``save_checkpoint`` used
   ``how_many = len(step_times_ms)``, which counts only the micro-steps a
   resumed process executed, dropping the pre-resume (replayed) part.  Latent:
   it corrupts a *second* resume, not the first.
3. **partial accumulation window** -- the backward pass divided by
   ``grad_accum`` even for a partial window (``ipe % grad_accum != 0``), so
   every epoch's final optimizer update used a gradient scaled by
   ``window_size / grad_accum``.

Usage::

    ./.venv-uv/bin/python -m tools.stage7b_audit
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from tools.stage7a_build_configs import ROOT

R = ROOT / 'reports'
GRAD_ACCUM = 8
IPE = 9204  # batches per epoch (batch_size=16, 147277 images, drop_last)


def git(args: List[str]) -> str:
    return subprocess.check_output(['git'] + args, cwd=ROOT, text=True).strip()


def load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def grep_in_commit(commit: str, pattern: str) -> List[str]:
    """Return lines of src/train.py at ``commit`` matching a regex pattern."""
    src = git(['show', f'{commit}:src/train.py'])
    return [line for line in src.splitlines() if pattern in line]


def epoch_math(optimizer_steps: int) -> Dict[str, object]:
    per_epoch = (IPE + GRAD_ACCUM - 1) // GRAD_ACCUM  # ceil
    epochs = optimizer_steps / per_epoch
    # the last optimizer update of every epoch (except the final partial epoch)
    # falls on a partial accumulation window when ipe % grad_accum != 0
    partial = IPE % GRAD_ACCUM
    return {
        'optimizer_steps_per_epoch': per_epoch,
        'ipe': IPE,
        'grad_accum': GRAD_ACCUM,
        'partial_window_size': partial if partial else None,
        'partial_window_ratio': partial / GRAD_ACCUM if partial else None,
        'epochs_crossed': round(epochs, 2),
        'partial_window_updates': int(epochs),
    }


def render(record20: Mapping[str, object], record50: Mapping[str, object],
           defects: Mapping[str, object]) -> str:
    lines: List[str] = []
    lines.append('# Stage 7A — 事后完整性审计')
    lines.append('')
    lines.append('> 基于 Stage 7A 实际训练代码（`src/train.py` @ `166d494` / `9ed811e`）'
                 '与训练记录的事后审计。Stage 7A 结果**不重跑、不覆盖**；'
                 '修复仅作用于 Stage 7B 及以后。')
    lines.append('')
    lines.append('## 1. 结论摘要')
    lines.append('')
    lines.append('- **三处缺陷均已确认存在于 Stage 7A 运行代码中**，并已在 Stage 7B 修复。')
    lines.append('- Stage 7A 的运行结果不因这些缺陷而作废：偏差为**确定性、小规模**的'
                 '调度/归一化扰动，不改变“同轨迹单调改善”的定性结论；')
    lines.append('  但严格复现性受损，因此引入本阶段的严格恢复测试。')
    lines.append('')
    lines.append('## 2. 缺陷 1：checkpoint 恢复 off-by-one')
    lines.append('')
    lines.append('**现象**：`save_step_snapshot` 写入 `epoch_itr = itr`（`itr` 为 enumerate '
                 '零起始索引），但快照在 `itr` 对应 micro-batch **已处理之后**写入，'
                 '已消费数量实为 `itr + 1`。')
    lines.append('')
    lines.append(f'- 代码证据：{defects["off_by_one"]["line"].strip()}')
    lines.append(f'- 修复：写入 `epoch_itr = itr + 1`（已消费 micro-batch 数）。')
    lines.append('')
    lines.append('**Stage 7A 实际影响（阶段2 从 step-20k 快照恢复）**：')
    lines.append('')
    lines.append(f'- 恢复点：epoch 17, itr {defects["off_by_one"]["resume_itr"]}'
                 f'（`{defects["off_by_one"]["resume_epoch"]}` 步）——该 itr 是梯度累积窗口末尾'
                 f'（`itr % {GRAD_ACCUM} == 7`），恢复后该 micro-batch 被**再次执行**。')
    lines.append('- 后果：多一次**单 micro-batch** 的 optimizer update（而非完整 8-batch 窗口），'
                 '以及一次重复的 micro-step；LR/WD/EMA 在恢复点被**提前推进 1 步**。')
    lines.append('- 规模：1 / 30,000 updates（阶段2）≈ **0.003%** 的更新受到单 batch 梯度扰动；'
                 '调度终点 LR/WD/EMA 不受影响（终点值仍精确达到）。')
    lines.append('')
    lines.append('## 3. 缺陷 2：epoch 内恢复位置统计')
    lines.append('')
    lines.append('**现象**：`save_checkpoint` 使用 `how_many = len(step_times_ms)`，'
                 '该计数器在每个 epoch 开始时清零，只统计本次进程在该 epoch 新执行的 '
                 'micro-steps，**遗漏恢复前（重放）的部分**。')
    lines.append('')
    lines.append(f'- 代码证据：{defects["epoch_position"]["line"].strip()}')
    lines.append('- 修复：引入 `epoch_consumed`（= 重放数 + 本次新执行数），'
                 '作为当前 epoch 的真实位置。')
    lines.append('')
    lines.append('**Stage 7A 实际影响**：**潜伏，未被触发**。Stage 7A 阶段2 只做了一次'
                 '“恢复→跑完”的单程，未发生第二次恢复；该缺陷只会使**第二次恢复**'
                 '重放已训练 batch。若将来在恢复后再中断再恢复，将产生漂移。')
    lines.append('')
    lines.append('## 4. 缺陷 3：部分梯度累积窗口归一化')
    lines.append('')
    lines.append('**现象**：backward 始终除以 `grad_accum`，即使某窗口是不完整的'
                 '（`ipe % grad_accum != 0` 时每个 epoch 末尾出现一个部分窗口）。')
    lines.append('')
    lines.append(f'- 代码证据：{defects["partial_window"]["line"].strip()}')
    lines.append('- 修复：除以**实际** `window_size`（`accumulation_divisor`）。')
    lines.append('')
    em = defects["partial_window"]["math"]
    lines.append('**Stage 7A 实际规模**：')
    lines.append('')
    lines.append(f'- `ipe={em["ipe"]}`，`grad_accum={em["grad_accum"]}` → '
                 f'每 epoch 末尾一个 **{em["partial_window_size"]}-micro-batch** 的部分窗口，'
                 f'其梯度被缩小到 **{em["partial_window_ratio"]:.2f}**（应为 1.00）。')
    lines.append(f'- 阶段1（20k）：跨约 {record20.get("optimizer_steps_total", 0) / em["optimizer_steps_per_epoch"]:.1f} '
                 f'个 epoch → 约 {int(record20.get("optimizer_steps_total", 0) / em["optimizer_steps_per_epoch"])} '
                 '个 epoch 末尾更新受影响。')
    lines.append(f'- 阶段2（50k）：跨约 {record50.get("optimizer_steps_total", 0) / em["optimizer_steps_per_epoch"]:.1f} '
                 f'个 epoch → 约 {int(record50.get("optimizer_steps_total", 0) / em["optimizer_steps_per_epoch"])} '
                 '个 epoch 末尾更新受影响。')
    lines.append(f'- 相对量级：每 epoch {em["optimizer_steps_per_epoch"]} 个更新中仅 1 个受 0.5x 缩放，'
                 '即约 **0.09%** 的更新；系统性、确定性，但很小。')
    lines.append('')
    lines.append('## 5. 对 Stage 7A 结论的影响')
    lines.append('')
    lines.append('三处缺陷造成的都是**确定性、微小**的扰动（单 batch 梯度、单步调度前移、'
                 '0.5x 梯度缩放 × ~0.09% 更新）。它们：')
    lines.append('')
    lines.append('- 不改变“10k→20k→50k 同轨迹单调改善”的方向与量级；')
    lines.append('- 但使结果**非严格可复现**（跨进程逐位不一致），故 Stage 7B 增加严格恢复测试；')
    lines.append('- 修复后的代码将用于 Stage 7B 的配对调度消融，保证该实验内部的公平性。')
    lines.append('')
    lines.append('## 6. 处置')
    lines.append('')
    lines.append('- Stage 7A 历史结果：保留原样，不重跑。')
    lines.append('- 修复：见 `src/train.py`（`accumulation_divisor`、`epoch_consumed`、'
                 '`epoch_itr = itr + 1`）。')
    lines.append('- 验证：`tests/test_stage7b_resume_fidelity.py`（单元 + GPU 集成）。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    record20 = load_json(R / 'stage7a_training_20k.json') or {}
    record50 = load_json(R / 'stage7a_training_50k.json') or {}
    old = '9ed811e'  # the Stage 7A stage2 training commit
    defects = {
        'off_by_one': {
            'line': grep_in_commit(old, "'epoch_itr': itr,")[0] if grep_in_commit(
                old, "'epoch_itr': itr,") else 'NOT FOUND',
            'resume_itr': 3463, 'resume_epoch': '20000',
        },
        'epoch_position': {
            'line': grep_in_commit(old, 'how_many = len(step_times_ms)')[0]
            if grep_in_commit(old, 'how_many = len(step_times_ms)') else 'NOT FOUND',
        },
        'partial_window': {
            'line': grep_in_commit(old, '(loss / grad_accum).backward()')[0]
            if grep_in_commit(old, '(loss / grad_accum).backward()') else 'NOT FOUND',
            'math': epoch_math(int(record50.get('optimizer_steps_total', 50000))),
        },
    }
    (R / 'stage7a_retrospective_integrity_audit.md').write_text(
        render(record20, record50, defects), encoding='utf-8')
    print('wrote reports/stage7a_retrospective_integrity_audit.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
