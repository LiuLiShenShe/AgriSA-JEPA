# Stage 6D — PhenoBench 跨预训练运行比较

> 状态：**未执行**（not_executed）。
> 原因：物理 GPU1 被外部进程（PID 2594918，`/fj/Agro-JEPA`）占用（14,912 MiB，100%），项目约束只允许 GPU1；且根盘仅余 187 G，4 个新 run 约需 232 G。本文件不含任何 Stage 6D 实测指标。

## 主重复单位

- **预训练 checkpoint**（不是 probe seed）。
- 同一 checkpoint 的 3 个 probe seed（0/1/2）只衡量下游 head 训练方差，绝不冒充独立预训练重复。

## 计划（预注册）

| arm | checkpoint 数（主分析） | 其中新增可追溯 | 历史带 caveat |
|---|---:|---:|---:|
| A | 1（复用 Stage 6C） | 0 | 0 |
| B1 | 3 | 2 | 1 |
| B2 | 3 | 2 | 1 |

- 配对 seed：P0(历史) = 20261026、P1 = 20261027、P2 = 20261028。
- 每 checkpoint × probe seeds 0/1/2；val 772 张；test 不参与。

## 历史参考（仅工具回归，不是 Stage 6D 结果）

`reports/stage6d_historical_reference.json` 用 Stage 6D 聚合工具重算了 Stage 6C 的三臂结果作为数值回归基线：

| arm | mIoU mean | B1/B2 − A mIoU |
|---|---:|---:|
| A | 71.68 | — |
| B1 | 72.85 | +1.17 |
| B2 | 72.64 | +0.96 |

## 待执行步骤

1. GPU1 空闲且工作树干净后：`.venv/bin/python -m tools.run_stage6d_pretraining --execute --device 1`
2. probe：`.venv/bin/python -m tools.run_stage6d_probe --execute --device 1`
3. 统计：`tools/stage6d_compare --manifest reports/stage6d_comparison_manifest.json --out-prefix reports/stage6d_phenobench_comparison`
4. 敏感性（仅新增）：`tools/stage6d_compare ... --traceable-only --out-prefix reports/stage6d_phenobench_sensitivity`

## 判定逻辑（预注册，不依赖事后阈值）

- 支持进一步投入：多个独立 B1 checkpoint 相对 A 的 mIoU 改善方向基本一致，非单个异常 run 驱动。
- 结果不确定：方向不一致、跨 run 波动 ≥ 平均收益，或改善主要来自单个 run。
- 不支持短预算稳定收益：多数独立 run 未优于 A。
- B1 vs B2：仅用配对比较；差异小于运行间波动或类别方向不一致 → 结论“无法区分”。
