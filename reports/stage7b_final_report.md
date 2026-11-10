# Stage 7B — 最终报告

> 生成时间（UTC）：2026-11-07T00:29:52+00:00

## 0. 目标与范围

- 修复 Stage 7A 的 checkpoint 恢复 / 梯度累积缺陷并给出严格恢复测试。
- 三臂 × 双 seed 配对调度消融，区分训练预算、调度长度、warmup 长度。
- 不修改 I-JEPA 核心目标、不引入 SlotCon、不更改语料、不做端到端微调。

## 1. Stage 7A 事后完整性审计（问题 1）

- 已生成：`reports/stage7a_retrospective_integrity_audit.md`
- 三处缺陷确认存在并已修复：
  - **off-by-one**：step snapshot 存 `epoch_itr=itr`，恢复后重执行最后已消费 micro-batch（Stage 7A 阶段2 恢复点 epoch17/itr3463，产生 1 次单 batch optimizer update，LR/WD/EMA 提前 1 步）。
  - **epoch 位置低估**：`how_many=len(step_times_ms)` 漏掉重放部分；Stage 7A 未触发（只做单次恢复→跑完），潜伏。
  - **部分窗口归一化**：backward 除以 `grad_accum` 而非实际 `window_size`；Stage 7A 每 epoch 末尾（ipe=9204, grad_accum=8 → 窗口 4）最后 1 个更新被 0.5x 缩放，约占 0.09% 更新。

## 2. 严格恢复测试（问题 2）

- **通过**：连续 24 vs 中断 12+恢复 24。
  - `clean`：trace 一致、ledger 一致、权重/优化器 容差内、计数 一致。
  - `partial`：trace 一致、ledger 一致、权重/优化器 容差内、计数 一致。

## 3. 消融运行（问题 3 / 4）

| run | arm | seed | steps | horizon | warmup | loss 首→末 | 耗时(h) |
|---|---|---|---|---|---|---|---|
| A_p1 | A | 20261104 | 2000 | 2000 | 100 | 0.1875→0.1349 | 3.51 |
| A_p2 | A | 20261105 | 2000 | 2000 | 100 | 0.1849→0.1370 | 3.69 |
| B_p1 | B | 20261104 | 2000 | 50000 | 100 | 0.1875→0.1460 | 3.66 |
| B_p2 | B | 20261105 | 2000 | 50000 | 100 | 0.1849→0.1450 | 3.61 |
| C_p1 | C | 20261104 | 2000 | 50000 | 2500 | 0.1875→0.1420 | 3.8 |
| C_p2 | C | 20261105 | 2000 | 50000 | 2500 | 0.1849→0.1474 | 3.99 |

## 4. PhenoBench 结果（问题 3 / 4）

| run | arm | seed | soil | crop | weed | mIoU |
|---|---|---|---|---|---|---|
| A_p1 | A | 20261104 | 97.68 | 82.12 | 38.10 | 72.64 |
| A_p2 | A | 20261105 | 97.68 | 82.14 | 38.51 | 72.78 |
| B_p1 | B | 20261104 | 97.72 | 82.30 | 37.82 | 72.62 |
| B_p2 | B | 20261105 | 97.72 | 82.25 | 36.91 | 72.29 |
| C_p1 | C | 20261104 | 97.71 | 82.34 | 37.47 | 72.51 |
| C_p2 | C | 20261105 | 97.71 | 82.21 | 37.47 | 72.46 |

## 5. 配对对比与决策（问题 3 / 4 / 5）

- **A vs B（预算/horizon）**：ΔmIoU +0.255 pp，Δcrop -0.140，Δweed +0.947；方向一致 True；决策 **CLEAR_EFFECT** — both seeds agree and mean mIoU delta = +0.255 pp (>= 0.2 pp, A-B); A is higher
- **B vs C（warmup）**：ΔmIoU -0.030 pp，Δcrop -0.005，Δweed -0.107；方向一致 False；决策 **INCONCLUSIVE** — two pretraining seeds move in opposite directions on mIoU

## 6. 结论（问题 5 / 6）

- **问题 3（2k vs 50k horizon 在相同 2k updates 下）**：A vs B → **CLEAR_EFFECT**；both seeds agree and mean mIoU delta = +0.255 pp (>= 0.2 pp, A-B); A is higher。
  - 配对差（A−B）逐 seed：p1 ΔmIoU +0.023 pp、p2 ΔmIoU +0.487 pp —— 方向虽一致，但效应不均：p1 接近 0，均值主要由 p2 贡献。
  - 类别分解：Δcrop -0.140 pp（A 略低）、Δweed +0.947 pp（A 明显更高）—— mIoU 的净收益几乎全部来自 weed，crop 反向；这是类异质效应，不应用单一 mIoU 掩盖。
- **问题 4（warmup 100 vs 2500）**：B vs C → **INCONCLUSIVE**；two pretraining seeds move in opposite directions on mIoU。
- **问题 5（Stage 7A 50k 与 2k 差异是否有调度解释）**：在相同 2k updates 与相同 warmup 下，A（2k horizon）配对高于 B（50k horizon）+0.255 pp（CLEAR_EFFECT），方向支持“短调度在 2k 步内已把 LR 退火到地板，更快完成域适配”这一调度因素解释；但因效应小、且仅由 weed 类与 p2 驱动、仅 2 个预训练 seed，这只是初步配对证据，不足以声称调度就是 Stage 7A 全部差异的解释。
- **问题 6（是否投入新 50k/100k）**：不建议投入 100k 训练。B vs C（warmup）方向不一致 → 无效；A vs B 虽判为 CLEAR_EFFECT，但效应仅 ~0.26 pp、主要来自 weed 单类、且被 p1 的近零 delta 稀释——50k horizon 在 2k-update 对等条件下对下游 PhenoBench 没有可确认的收益，因此更长（50k/100k）训练不会带来对应的下游提升。若该 0.26 pp 量级对决策有实际影响，正确做法是为 A-vs-B 对比追加预训练 seed，而非启动 100k 训练。

## 7. 可复现性与产物

- 分支：`agri-stage7b-schedule-ablation`（未合并 main，未 force-push）。
- 全部 checkpoint 与训练记录保留本地；未删除、未覆盖 Stage 7A。
- 未使用 conda；仅 GPU1；6 个训练串行执行。

