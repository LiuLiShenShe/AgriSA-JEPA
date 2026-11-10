# Stage 7A — 事后完整性审计

> 基于 Stage 7A 实际训练代码（`src/train.py` @ `166d494` / `9ed811e`）与训练记录的事后审计。Stage 7A 结果**不重跑、不覆盖**；修复仅作用于 Stage 7B 及以后。

## 1. 结论摘要

- **三处缺陷均已确认存在于 Stage 7A 运行代码中**，并已在 Stage 7B 修复。
- Stage 7A 的运行结果不因这些缺陷而作废：偏差为**确定性、小规模**的调度/归一化扰动，不改变“同轨迹单调改善”的定性结论；
  但严格复现性受损，因此引入本阶段的严格恢复测试。

## 2. 缺陷 1：checkpoint 恢复 off-by-one

**现象**：`save_step_snapshot` 写入 `epoch_itr = itr`（`itr` 为 enumerate 零起始索引），但快照在 `itr` 对应 micro-batch **已处理之后**写入，已消费数量实为 `itr + 1`。

- 代码证据：'epoch_itr': itr,
- 修复：写入 `epoch_itr = itr + 1`（已消费 micro-batch 数）。

**Stage 7A 实际影响（阶段2 从 step-20k 快照恢复）**：

- 恢复点：epoch 17, itr 3463（`20000` 步）——该 itr 是梯度累积窗口末尾（`itr % 8 == 7`），恢复后该 micro-batch 被**再次执行**。
- 后果：多一次**单 micro-batch** 的 optimizer update（而非完整 8-batch 窗口），以及一次重复的 micro-step；LR/WD/EMA 在恢复点被**提前推进 1 步**。
- 规模：1 / 30,000 updates（阶段2）≈ **0.003%** 的更新受到单 batch 梯度扰动；调度终点 LR/WD/EMA 不受影响（终点值仍精确达到）。

## 3. 缺陷 2：epoch 内恢复位置统计

**现象**：`save_checkpoint` 使用 `how_many = len(step_times_ms)`，该计数器在每个 epoch 开始时清零，只统计本次进程在该 epoch 新执行的 micro-steps，**遗漏恢复前（重放）的部分**。

- 代码证据：how_many = len(step_times_ms)  # micro-steps actually trained this epoch
- 修复：引入 `epoch_consumed`（= 重放数 + 本次新执行数），作为当前 epoch 的真实位置。

**Stage 7A 实际影响**：**潜伏，未被触发**。Stage 7A 阶段2 只做了一次“恢复→跑完”的单程，未发生第二次恢复；该缺陷只会使**第二次恢复**重放已训练 batch。若将来在恢复后再中断再恢复，将产生漂移。

## 4. 缺陷 3：部分梯度累积窗口归一化

**现象**：backward 始终除以 `grad_accum`，即使某窗口是不完整的（`ipe % grad_accum != 0` 时每个 epoch 末尾出现一个部分窗口）。

- 代码证据：scaler.scale(loss / grad_accum).backward()
- 修复：除以**实际** `window_size`（`accumulation_divisor`）。

**Stage 7A 实际规模**：

- `ipe=9204`，`grad_accum=8` → 每 epoch 末尾一个 **4-micro-batch** 的部分窗口，其梯度被缩小到 **0.50**（应为 1.00）。
- 阶段1（20k）：跨约 17.4 个 epoch → 约 17 个 epoch 末尾更新受影响。
- 阶段2（50k）：跨约 43.4 个 epoch → 约 43 个 epoch 末尾更新受影响。
- 相对量级：每 epoch 1151 个更新中仅 1 个受 0.5x 缩放，即约 **0.09%** 的更新；系统性、确定性，但很小。

## 5. 对 Stage 7A 结论的影响

三处缺陷造成的都是**确定性、微小**的扰动（单 batch 梯度、单步调度前移、0.5x 梯度缩放 × ~0.09% 更新）。它们：

- 不改变“10k→20k→50k 同轨迹单调改善”的方向与量级；
- 但使结果**非严格可复现**（跨进程逐位不一致），故 Stage 7B 增加严格恢复测试；
- 修复后的代码将用于 Stage 7B 的配对调度消融，保证该实验内部的公平性。

## 6. 处置

- Stage 7A 历史结果：保留原样，不重跑。
- 修复：见 `src/train.py`（`accumulation_divisor`、`epoch_consumed`、`epoch_itr = itr + 1`）。
- 验证：`tests/test_stage7b_resume_fidelity.py`（单元 + GPU 集成）。

