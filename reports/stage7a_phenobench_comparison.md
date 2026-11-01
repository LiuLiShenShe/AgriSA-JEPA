# Stage 7A — PhenoBench 预算曲线比较

- 同轨迹 checkpoints：2（同一 50k schedule 的中间快照，不是独立预训练 seed）

## 参考行（不同 schedule / 不同 seed，仅作参照）

| phase | arm | soil | crop | weed | mIoU |
|---|---|---|---|---|---|
| baseline | A | 97.6085 | 81.4567 | 35.98 | 71.6833 |
| ref_2k_stage6d | B1 | 97.6902 | 82.2333 | 38.52 | 72.81 |

## B1 同轨迹逐 checkpoint（probe seeds 0/1/2 均值）

| phase | soil | crop | weed | mIoU | Δsoil | Δcrop | Δweed | ΔmIoU |
|---|---|---|---|---|---|---|---|---|
| step10k | 97.7342 | 82.3333 | 35.9867 | 72.0167 |  |  |  |  |
| step20k | 97.7424 | 82.44 | 37.37 | 72.5167 | 0.0082 | 0.1067 | 1.3833 | 0.5 |

## 决策（20k → 50k 预定义规则）

- **结论：CONTINUE_TO_50k**
- 理由：20k is at least as good as 10k within tolerance and shows a clear positive signal; continue the same 50k schedule from the 20k checkpoint.

| 检查项 | 结果 |
|---|---|
| training_verified | True |
| miou_20k_minus_10k | 0.5 |
| crop_20k_minus_10k | 0.1067 |
| weed_20k_minus_10k | 1.3833 |
| soil_20k_minus_10k | 0.0082 |
| miou_not_worse_than_minus_0_3 | True |
| positive_signal | True |
| crop_and_weed_not_below_minus_1_0 | True |
| cross_check_agrees | True |
| borderline | False |
| conflicting | False |
| thresholds | {'miou_drop_max': 0.3, 'miou_up_min': 0.2, 'crop_weed_up_min': 0.5, 'crop_weed_drop_max': 1.0} |

