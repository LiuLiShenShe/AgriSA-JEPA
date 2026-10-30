# Stage 7A — 最终验收报告

> 生成时间（UTC）：2026-10-30T10:46:03+00:00

## 1. Git 基线

- 分支：`agri-stage7a-20k-checkpoint-curve`
- 基线 commit：`6b9cb8502f0501c2f37054337b777347005b6162`
- 新提交：见 `git log --oneline -5`（本报告之后提交）
- 工作树状态：仅未跟踪的 `CLAUDE.md`（保留），无强制回退、无覆盖历史产物

## 2. 冻结输入

- train_csv：`9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` (通过)
- val_csv：`f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` (通过)
- sampling_profile：`91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505` (通过)
- train_view：`26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` (通过)

## 3. 配置与调度

- 阶段1配置：`configs/stage7a_b1_natural_50k.yaml`（SHA `18f713d9c5ab6d90…`）
- 阶段2配置：`configs/stage7a_b1_natural_50k_resume.yaml`（SHA `d63fcf4223e9bc2e…`）
- 总调度长度：`max_optimizer_steps=50000`，`epochs=50`（最低 44）
- 阶段性中断：`interrupt_after_optimizer_steps=20000`
- 保存点：`[10000, 20000, 50000]`
- 有效 batch：16 x 8 = 128
- 协议检查：**通过**（0 个问题）

## 4. 训练轨迹

| 阶段 | optimizer steps | micro steps | loss 首→末 | 峰值显存(MiB) | 耗时(小时) |
|---|---|---|---|---|---|
| 10k/20k（阶段1） | 未执行 | — | — | — | — |
| 50k（阶段2） | 未执行 | — | — | — | — |


## 5. Checkpoint 与 SHA-256

（尚未记录）

## 6. source exposure 实际分布

（阶段1未执行，无实测曝光）

## 7. PhenoBench 冻结 encoder probe

（未执行：需要 10k/20k checkpoint）

## 8. 是否继续到 50k

（未执行）

## 9. 测试与偏离

- 单元测试：`tests/test_stage7a_checkpoint_curve.py`（23 项，见 test_summary）
- 历史测试：未修改、未删除、未放宽
- 已知偏离：无

## 10. 结论

**可支持**：
- 若 10k/20k/50k 结果齐全：预算与农业表征价值的关系（单轨迹）。
**不可支持**：
- 不能把 10k/20k/50k 计为三个独立预训练 seed（同一轨迹）。
- 单条轨迹不能支持“预算越多越好”的普遍性结论。

## 11. 下一阶段建议

- 依据决策结论：继续到 50k（若通过）或停止在 20k 并报告。
- 若继续：从 `jepa-step20000.pth.tar` 续训，不重跑前 20k。

