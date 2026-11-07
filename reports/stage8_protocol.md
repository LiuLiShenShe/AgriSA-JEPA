# Stage 8 — 冻结协议与实验注册表

> 冻结时间（UTC）：2026-11-07T09:36:30+00:00

## 0. 研究问题

- **RQ1** 农业继续预训练（A/B）在跨任务迁移上是否优于官方 I-JEPA？
- **RQ2** 收益是否跨数据集泛化？
- **RQ3** Stage 7B 的调度差异（A vs B）是否迁移到其他任务？
- **RQ4** 不同任务是否需要不同表征？
- **RQ5** 收益在预训练 seed / head seed / 切分 / 数据集间是否稳定？

## 1. Backbone 注册（5 primary + 2 exploratory）

| key | arm | pretrain seed | horizon/warmup | role |
|---|---|---|---|---|
| OFFICIAL | OFFICIAL | — | — | baseline |
| A_p1 | A | 20261104 | 2000/100 | primary |
| A_p2 | A | 20261105 | 2000/100 | primary |
| B_p1 | B | 20261104 | 50000/100 | primary |
| B_p2 | B | 20261105 | 50000/100 | primary |
| C_p1 | C | 20261104 | 50000/2500 | exploratory |
| C_p2 | C | 20261105 | 50000/2500 | exploratory |

## 2. 任务协议

### gwhd_det — object detection (Global Wheat Head 2021)

- 状态：READY (Stage 5A pipeline reused verbatim)
- 切分：official competition_{train,val,test}.csv, used verbatim; test evaluated once at the end of training
- 主指标：test map (COCO AP), AP50 secondary
- 入口：`./.venv-uv/bin/python -m src.downstream.detection_train`

### cropandweed_sem — semantic segmentation (CropAndWeed, CropsOrWeed9 mapping)

- 状态：READY_WITH_MODIFICATION (no official split CSV; ave_/vwg_ set semantics used)
- 切分：train = ave_ set (3,044 images, full); eval = vwg_ set, one image per recording session (seed 20261107), ~665 images
- 主指标：9-class mIoU (Soil 0 and Vegetation 255 ignored)
- 入口：`./.venv-uv/bin/python -m src.downstream.cropandweed_probe`

### deepweeds_cls — classification (DeepWeeds, 9 classes)

- 状态：READY_WITH_MODIFICATION (5-fold CV; fold 0 fixed)
- 切分：train = train_subset0 (10,501) + val_subset0 (3,501); test = test_subset0 (3,507), evaluated once
- 主指标：macro-F1 on test (accuracy secondary)
- 入口：`./.venv-uv/bin/python -m src.downstream.deepweeds_probe`

### plantdoc_cls — classification (PlantDoc, 28 class directories)

- 状态：READY_WITH_MODIFICATION (detection impossible locally: no bbox annotations; classification labels from directory structure)
- 切分：official train/test used verbatim; val carved from train stratified by class at 15% (seed 20261107); test evaluated once
- 主指标：macro-F1 on official test (accuracy secondary)
- 入口：`./.venv-uv/bin/python -m src.downstream.plantdoc_probe`

### phenobench_sem — semantic segmentation (PhenoBench semantics; reuse Stage 7B)

- 状态：READY (Stage 7B covers A_p1/A_p2/B_p1/B_p2/C_p1/C_p2 with probe seeds 0/1/2); Stage 8 adds the OFFICIAL baseline required by RQ1
- 切分：official train/val, test-server not used
- 主指标：mIoU (official semantics: soil/crop/weed)
- 入口：`./.venv-uv/bin/python -m src.downstream.phenobench_probe`

## 3. 实验注册表（冻结）

全部 primary 任务 × 下游 seed 0/1/2；下游 seed 与预训练 seed (20261104/20261105) 严格不同。

| task | backbone | seed | role | out |
|---|---|---|---|---|
| gwhd_det | OFFICIAL | 0 | primary | `gwhd_det/OFFICIAL_seed0` |
| gwhd_det | OFFICIAL | 1 | primary | `gwhd_det/OFFICIAL_seed1` |
| gwhd_det | OFFICIAL | 2 | primary | `gwhd_det/OFFICIAL_seed2` |
| gwhd_det | A_p1 | 0 | primary | `gwhd_det/A_p1_seed0` |
| gwhd_det | A_p1 | 1 | primary | `gwhd_det/A_p1_seed1` |
| gwhd_det | A_p1 | 2 | primary | `gwhd_det/A_p1_seed2` |
| gwhd_det | A_p2 | 0 | primary | `gwhd_det/A_p2_seed0` |
| gwhd_det | A_p2 | 1 | primary | `gwhd_det/A_p2_seed1` |
| gwhd_det | A_p2 | 2 | primary | `gwhd_det/A_p2_seed2` |
| gwhd_det | B_p1 | 0 | primary | `gwhd_det/B_p1_seed0` |
| gwhd_det | B_p1 | 1 | primary | `gwhd_det/B_p1_seed1` |
| gwhd_det | B_p1 | 2 | primary | `gwhd_det/B_p1_seed2` |
| gwhd_det | B_p2 | 0 | primary | `gwhd_det/B_p2_seed0` |
| gwhd_det | B_p2 | 1 | primary | `gwhd_det/B_p2_seed1` |
| gwhd_det | B_p2 | 2 | primary | `gwhd_det/B_p2_seed2` |
| gwhd_det | C_p1 | 0 | exploratory | `gwhd_det/C_p1_seed0` |
| gwhd_det | C_p1 | 1 | exploratory | `gwhd_det/C_p1_seed1` |
| gwhd_det | C_p1 | 2 | exploratory | `gwhd_det/C_p1_seed2` |
| gwhd_det | C_p2 | 0 | exploratory | `gwhd_det/C_p2_seed0` |
| gwhd_det | C_p2 | 1 | exploratory | `gwhd_det/C_p2_seed1` |
| gwhd_det | C_p2 | 2 | exploratory | `gwhd_det/C_p2_seed2` |
| cropandweed_sem | OFFICIAL | 0 | primary | `cropandweed_sem/OFFICIAL_seed0` |
| cropandweed_sem | OFFICIAL | 1 | primary | `cropandweed_sem/OFFICIAL_seed1` |
| cropandweed_sem | OFFICIAL | 2 | primary | `cropandweed_sem/OFFICIAL_seed2` |
| cropandweed_sem | A_p1 | 0 | primary | `cropandweed_sem/A_p1_seed0` |
| cropandweed_sem | A_p1 | 1 | primary | `cropandweed_sem/A_p1_seed1` |
| cropandweed_sem | A_p1 | 2 | primary | `cropandweed_sem/A_p1_seed2` |
| cropandweed_sem | A_p2 | 0 | primary | `cropandweed_sem/A_p2_seed0` |
| cropandweed_sem | A_p2 | 1 | primary | `cropandweed_sem/A_p2_seed1` |
| cropandweed_sem | A_p2 | 2 | primary | `cropandweed_sem/A_p2_seed2` |
| cropandweed_sem | B_p1 | 0 | primary | `cropandweed_sem/B_p1_seed0` |
| cropandweed_sem | B_p1 | 1 | primary | `cropandweed_sem/B_p1_seed1` |
| cropandweed_sem | B_p1 | 2 | primary | `cropandweed_sem/B_p1_seed2` |
| cropandweed_sem | B_p2 | 0 | primary | `cropandweed_sem/B_p2_seed0` |
| cropandweed_sem | B_p2 | 1 | primary | `cropandweed_sem/B_p2_seed1` |
| cropandweed_sem | B_p2 | 2 | primary | `cropandweed_sem/B_p2_seed2` |
| cropandweed_sem | C_p1 | 0 | exploratory | `cropandweed_sem/C_p1_seed0` |
| cropandweed_sem | C_p1 | 1 | exploratory | `cropandweed_sem/C_p1_seed1` |
| cropandweed_sem | C_p1 | 2 | exploratory | `cropandweed_sem/C_p1_seed2` |
| cropandweed_sem | C_p2 | 0 | exploratory | `cropandweed_sem/C_p2_seed0` |
| cropandweed_sem | C_p2 | 1 | exploratory | `cropandweed_sem/C_p2_seed1` |
| cropandweed_sem | C_p2 | 2 | exploratory | `cropandweed_sem/C_p2_seed2` |
| deepweeds_cls | OFFICIAL | 0 | primary | `deepweeds_cls/OFFICIAL_seed0` |
| deepweeds_cls | OFFICIAL | 1 | primary | `deepweeds_cls/OFFICIAL_seed1` |
| deepweeds_cls | OFFICIAL | 2 | primary | `deepweeds_cls/OFFICIAL_seed2` |
| deepweeds_cls | A_p1 | 0 | primary | `deepweeds_cls/A_p1_seed0` |
| deepweeds_cls | A_p1 | 1 | primary | `deepweeds_cls/A_p1_seed1` |
| deepweeds_cls | A_p1 | 2 | primary | `deepweeds_cls/A_p1_seed2` |
| deepweeds_cls | A_p2 | 0 | primary | `deepweeds_cls/A_p2_seed0` |
| deepweeds_cls | A_p2 | 1 | primary | `deepweeds_cls/A_p2_seed1` |
| deepweeds_cls | A_p2 | 2 | primary | `deepweeds_cls/A_p2_seed2` |
| deepweeds_cls | B_p1 | 0 | primary | `deepweeds_cls/B_p1_seed0` |
| deepweeds_cls | B_p1 | 1 | primary | `deepweeds_cls/B_p1_seed1` |
| deepweeds_cls | B_p1 | 2 | primary | `deepweeds_cls/B_p1_seed2` |
| deepweeds_cls | B_p2 | 0 | primary | `deepweeds_cls/B_p2_seed0` |
| deepweeds_cls | B_p2 | 1 | primary | `deepweeds_cls/B_p2_seed1` |
| deepweeds_cls | B_p2 | 2 | primary | `deepweeds_cls/B_p2_seed2` |
| deepweeds_cls | C_p1 | 0 | exploratory | `deepweeds_cls/C_p1_seed0` |
| deepweeds_cls | C_p1 | 1 | exploratory | `deepweeds_cls/C_p1_seed1` |
| deepweeds_cls | C_p1 | 2 | exploratory | `deepweeds_cls/C_p1_seed2` |
| deepweeds_cls | C_p2 | 0 | exploratory | `deepweeds_cls/C_p2_seed0` |
| deepweeds_cls | C_p2 | 1 | exploratory | `deepweeds_cls/C_p2_seed1` |
| deepweeds_cls | C_p2 | 2 | exploratory | `deepweeds_cls/C_p2_seed2` |
| plantdoc_cls | OFFICIAL | 0 | primary | `plantdoc_cls/OFFICIAL_seed0` |
| plantdoc_cls | OFFICIAL | 1 | primary | `plantdoc_cls/OFFICIAL_seed1` |
| plantdoc_cls | OFFICIAL | 2 | primary | `plantdoc_cls/OFFICIAL_seed2` |
| plantdoc_cls | A_p1 | 0 | primary | `plantdoc_cls/A_p1_seed0` |
| plantdoc_cls | A_p1 | 1 | primary | `plantdoc_cls/A_p1_seed1` |
| plantdoc_cls | A_p1 | 2 | primary | `plantdoc_cls/A_p1_seed2` |
| plantdoc_cls | A_p2 | 0 | primary | `plantdoc_cls/A_p2_seed0` |
| plantdoc_cls | A_p2 | 1 | primary | `plantdoc_cls/A_p2_seed1` |
| plantdoc_cls | A_p2 | 2 | primary | `plantdoc_cls/A_p2_seed2` |
| plantdoc_cls | B_p1 | 0 | primary | `plantdoc_cls/B_p1_seed0` |
| plantdoc_cls | B_p1 | 1 | primary | `plantdoc_cls/B_p1_seed1` |
| plantdoc_cls | B_p1 | 2 | primary | `plantdoc_cls/B_p1_seed2` |
| plantdoc_cls | B_p2 | 0 | primary | `plantdoc_cls/B_p2_seed0` |
| plantdoc_cls | B_p2 | 1 | primary | `plantdoc_cls/B_p2_seed1` |
| plantdoc_cls | B_p2 | 2 | primary | `plantdoc_cls/B_p2_seed2` |
| phenobench_sem | OFFICIAL | 0 | primary | `phenobench_sem/OFFICIAL` |
| phenobench_sem | OFFICIAL | 1 | primary | `phenobench_sem/OFFICIAL` |
| phenobench_sem | OFFICIAL | 2 | primary | `phenobench_sem/OFFICIAL` |
| phenobench_sem | A_p1 | 0 | primary | `phenobench_sem/A_p1` |
| phenobench_sem | A_p1 | 1 | primary | `phenobench_sem/A_p1` |
| phenobench_sem | A_p1 | 2 | primary | `phenobench_sem/A_p1` |
| phenobench_sem | A_p2 | 0 | primary | `phenobench_sem/A_p2` |
| phenobench_sem | A_p2 | 1 | primary | `phenobench_sem/A_p2` |
| phenobench_sem | A_p2 | 2 | primary | `phenobench_sem/A_p2` |
| phenobench_sem | B_p1 | 0 | primary | `phenobench_sem/B_p1` |
| phenobench_sem | B_p1 | 1 | primary | `phenobench_sem/B_p1` |
| phenobench_sem | B_p1 | 2 | primary | `phenobench_sem/B_p1` |
| phenobench_sem | B_p2 | 0 | primary | `phenobench_sem/B_p2` |
| phenobench_sem | B_p2 | 1 | primary | `phenobench_sem/B_p2` |
| phenobench_sem | B_p2 | 2 | primary | `phenobench_sem/B_p2` |

## 4. 冻结规则

- 注册表一旦冻结，Phase 4 只能按行执行；任何偏差（含失败重跑）必须以新的行追加并记录原因，不得原地改写。
- 官方 test 只在每行训练结束时评估一次。
- 泄漏审计（Phase 1）为冻结前置条件：全部本地数据集 CLEAN_exact。
- 训练从不覆盖：输出目录已存在 protocol.json 时拒绝启动。

