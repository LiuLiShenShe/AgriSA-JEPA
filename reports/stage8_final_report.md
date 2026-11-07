# Stage 8 — 最终报告（草稿骨架，待 Phase 5 汇总填充）

> 生成时间（UTC）：TBD

## 0. 目标与范围

- 验证 AgriSA-JEPA（农业继续预训练 I-JEPA ViT-H/14）是否真正具有跨农业任务迁移价值。
- 5 个冻结 backbone（Official / A_p1 / A_p2 / B_p1 / B_p2）× 3 个下游 head seed；
  3 个新任务（GWHD 检测、CropAndWeed 分割、DeepWeeds 分类）+ PlantDoc 分类 +
  PhenoBench 分割（Stage 7B 复用）；C 臂探索（条件允许）。
- 冻结 encoder + 线性/感知头，无端到端微调，无 adapter（Stage 9 范围外）。

## 1. 数据集与泄漏审计（Phase 1 结论）

| dataset | 任务 | 状态 | 图数 | raw/pixel 泄漏命中 |
|---|---|---|---|---|
| GWHD 2021 | detection | READY | 6,515 | 0 / 0 |
| CropAndWeed | segmentation | READY_WITH_MODIFICATION | 8,034 | 0 / 0 |
| DeepWeeds | classification | READY_WITH_MODIFICATION | 17,509 | 0 / 0 |
| PlantDoc | classification | READY_WITH_MODIFICATION | 2,579 | 0 / 0 |
| PhenoBench | segmentation | READY | 13,767(含mask) | 0 / 0 |
| MinneApple | detection | BLOCKED | 0 | not_downloaded |
| GrowliFlower | segmentation | BLOCKED | 0 | not_downloaded |
| PlantSeg | segmentation | BLOCKED | 0 | not_downloaded |

- 精确哈希（raw_sha256 + pixel_sha256）对 147,277 张预训练图零命中；历史结论
  （GWHD 2 张重采样同图样本已在 R3v3 排除）复核无误。
- CropAndWeed：官方 CropsOrWeed9 映射（库内逐字对照），Soil/Vegetation 为 ignore。

## 2. 冻结协议（Phase 2）

- 83 行注册表：5 任务 × 5 primary backbone × 3 head seed（phenobench 单命令 3 seed）。
- 下游 head seed 0/1/2 与预训练 seed 20261104/20261105 严格不同。
- 每行含完整命令、backbone checkpoint SHA 引用、output dir；官方 test 只评一次。

## 3. 结果（待 Phase 5 填表）

### PhenoBench 分割（Stage 7B + OFFICIAL 补测）

| run | arm | crop | weed | mIoU |
|---|---|---|---|---|
| OFFICIAL | — | 81.46 | 35.98 | **71.68** |
| A_avg | A | 82.13 | 38.31 | **72.71** |
| B_avg | B | 82.28 | 37.37 | **72.46** |

### PlantDoc 分类（28 类叶病）

| backbone | macroF1 mean | per-seed |
|---|---|---|
| OFFICIAL | 0.432 | ... |
| A_avg | 0.400 | ... |
| B_avg | 0.356 | ... |

### DeepWeeds 分类（9 类）

| backbone | macroF1 mean | per-seed |
|---|---|---|
| OFFICIAL | 0.836 | ... |
| A_avg | 0.824 | ... |
| B_avg | 0.779 | ... |

### CropAndWeed / GWHD（待运行）

TBD

## 4. 问题回答（待填）

- RQ1 尺度分化：...（PhenoBench 正、分类负）
- RQ2 跨数据集：...
- RQ3 调度迁移：...
- RQ4 任务×表征：...
- RQ5 稳定性：...

## 5. 可复现性与产物

- 分支 `agri-stage8-multitask-transfer`，未合并 main，未 force-push。
- 全部 checkpoint、特征缓存、结果 JSON、wall clock 记录保留本地。
- 未使用 conda；仅 GPU1；任务串行。