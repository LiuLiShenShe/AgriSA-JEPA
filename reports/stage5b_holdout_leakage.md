# Stage 5B — 下游保留集泄漏审计

> 机器可读：`reports/stage5b_holdout_overlap_{sesame_weed,applebbch81,weedcrop}.json`
> 工具：`tools/stage5b_check_holdout_overlap.py`
> 保留集定义：`data/registry/downstream_holdout_sources.csv`（12 个）

## 1. 规则

主实验定义为 **Source-Disjoint Pretraining / Downstream Evaluation**：

- 12 个保留集来源（PhenoBench、GWHD 2021、GWHD Instance Segmentation、PlantDoc、
  MinneApple、Roboflow Disease Detection、Roboflow Crop Detection、PlantSeg、
  Multiclass Weeds、DeepWeeds、GrowliFlower、CropAndWeed）不得出现在任何预训练
  manifest 中；
- 该规则在域政策之前执行，白名单参数无法覆盖（见
  `tests/test_stage5b_field_corpus.py::test_whitelist_cannot_override_holdout`）；
- 精确哈希命中是阻断条件；感知候选只是复核线索。

## 2. 新来源与保留集的比对

| 新来源 | 比对参考 | 精确命中 | 感知候选 |
|---|---|---:|---:|
| `sesame_weed` | v1 train / v1 val / D1 candidate train / D1 candidate val | 0 / 0 / 0 / 0 | 0 |
| `sesame_weed` | AgriField-40K 全量索引（含未采纳行） | 1,300 | 2,716 |
| `applebbch81` | 上述全部参考 | 0 | 0 |
| `weedcrop` | 上述全部参考 | 0 | 0 |

`sesame_weed` 的 1,300 个精确命中全部指向 `pending_audit` 的 `saw_*` 行，
**不是** PhenoBench 或其他保留集。逐条核对结果：命中参考的 `source` 字段全部为
`pending_audit`，`reference_corpus` 全部为 `all_samples`。

## 3. 与 PhenoBench 的直接哈希比对

用本机 AgriField-40K 索引中的 9,606 条 PhenoBench 行做精确比对：

| 语料 | 行数 | 与 PhenoBench 精确重复 |
|---|---:|---:|
| agrifield40k-v1 train+val | 15,592 | 0 |
| agri-field-d1-candidate train+val | 18,730 | 0 |

## 4. 来源层面的隔离

Field-D1 的 8 个来源与 12 个保留集来源的交集为空：

```
lucas_vision, applebbch81, vegann, inat_weeds,
sesame_weed, acre_crop_weed, radish_wheat, sorghum_weed
```

## 5. GWHD 与 PhenoBench 的下游评价未被破坏

- Stage 5A 的 GWHD 检测 A/B 结果与 `reports/stage5a_gwhd_*` 未改动；
- PhenoBench 的 probe 配置与缓存未改动；
- 全部既有测试仍通过（见 `reports/stage5b_summary.md` 工程部分）。

## 6. 已知风险（未解决）

1. **来源级隔离 ≠ 场景级隔离**。例如 `sorghum_weed` 与 `minneapple` 都是田间/果园，
   但来自不同国家与作物；本轮没有跨来源的场景重叠证据，也没有做作物层面的
   leave-one-crop-out 分析。后续若下游任务换作物，需要在协议里写明。
2. **Agri-FM+ 原始来源表中的保留集**：`agrifm_ref31`（GWHD）与 `agrifm_ref88`
   （PhenoBench）在 Stage 5A 台账里存在。本轮在构建阶段按物理数据集归并，未把
   它们带入 Field-D1；`test_field_d1_manifest_contains_no_holdout_source` 固化该行为。
3. **第三方转载**：本轮三个新来源都是原始发布页（Kaggle 数据集页），已做精确与
   感知筛查；但如果将来引入 Roboflow 转载版，必须重新跑本工具，因为转载常见重采样。
