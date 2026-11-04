# Stage 7B — 恢复保真测试报告

- 生成时间（UTC）：2026-11-04T08:29:23+00:00
- 预算：24 optimizer updates；中断/恢复点：12
- 权重容差：rtol=0.001, atol=1e-05（CUDA 非确定性）
- 数据流 / mask / 计数 / LR / WD / ledger 要求逐 micro-batch 完全一致

## 结论

- 总体：**通过**

## 场景 `clean`（ipe=192）

- 状态：通过
- 数据流 trace：A 192 / B 192 条，一致
- ledger：A 3072 / B 3072 行，一致
- 权重/优化器：一致
  - encoder：ok，max|diff|=0.000e+00
  - predictor：ok，max|diff|=0.000e+00
  - target_encoder：ok，max|diff|=0.000e+00
- 计数：{"optimizer_steps_total": {"a": 24, "b": 24}, "micro_steps_total": {"a": 192, "b": 192}, "partial_grad_accum_windows": {"a": 0, "b": 0}, "partial_window_micro_steps": {"a": 0, "b": 0}}

## 场景 `partial`（ipe=190）

- 状态：通过
- 数据流 trace：A 190 / B 190 条，一致
- ledger：A 3040 / B 3040 行，一致
- 权重/优化器：一致
  - encoder：ok，max|diff|=0.000e+00
  - predictor：ok，max|diff|=0.000e+00
  - target_encoder：ok，max|diff|=0.000e+00
- 计数：{"optimizer_steps_total": {"a": 24, "b": 24}, "micro_steps_total": {"a": 190, "b": 190}, "partial_grad_accum_windows": {"a": 1, "b": 1}, "partial_window_micro_steps": {"a": 6, "b": 6}}

