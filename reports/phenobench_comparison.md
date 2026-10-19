# Arm A / Arm B PhenoBench 语义分割对比

| 指标 | 官方 ViT-H (Arm A) | 农业 JEPA 500-step (Arm B) | 差值 (B-A) |
| --- | --- | --- | --- |
| mIoU | 71.68 ± 1.35 | 73.13 ± 1.09 | +1.45 |
| Crop IoU | 81.46 ± 0.17 | 82.38 ± 0.16 | +0.92 |
| Weed IoU | 35.98 ± 3.89 | 39.32 ± 3.12 | +3.34 |

### 按 seed 配对的差值 (B-A)

| 指标 | 配对均值 | 配对标准差 | df | t |
| --- | --- | --- | ---: | ---: |
| mIoU | +1.450 | 0.627 | 2 | 4.00 |
| Crop IoU | +0.923 | 0.076 | 2 | 20.94 |
| Weed IoU | +3.340 | 1.826 | 2 | 3.17 |

每个 arm 3 个 seed (0,1,2)，官方 `phenobench-eval --task semantics` 输出。自实现指标与官方指标在每个 seed 上的最大偏差 < 0.005 个百分点。

平凡基线（全部预测 soil）：mIoU 29.97。

**数据版本声明**：Arm B 在 `agrifield40k-v0`（PROVISIONAL）上完成 500 次 optimizer update；`agrifield40k-v1` 在其之后才建立。这是探索性诊断，不是最终同数据版本实验。
