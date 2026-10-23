# Stage 5B — 来源核验与采集域分类

> 产物：`data/registry/agricultural_sources_v2.csv`（64 行）、
> `data/registry/field_source_review.csv`（63 行）、
> `reports/stage5b_source_verification.json`、
> `reports/stage5b_domain_audit.json`
> 生成工具：`tools/stage5b_verify_sources.py`、`tools/stage5b_audit_domains.py`

## 1. 覆盖情况

Stage 5A 的 63 条登记行全部完成采集域分类，未跳过任何一行：

| 指标 | 数量 |
|---|---:|
| 登记行 | 63 |
| 独立物理数据集 | 59（4 组同源映射合并） |
| 有官方文档证据 | 54 行 |
| 无可用证据（`none`） | 9 行 |
| 下游保留集 | 12 行 |

9 个 `none` 行全部记为 `UNKNOWN`，没有一个被"按名字猜成田间"。它们分别是：
`agrifm_ref1`、`agrifm_ref10`、`agrifm_ref58`、`agrifm_ref7`、`agrifm_ref75`、
`agrifm_ref76`、`agrifm_ref8`、`agrifm_ref81`、`roboflow_disease_detection`。

## 2. 独立数据集的采集域分布

| 采集域 | 独立数据集 | 含重复行的登记行 |
|---|---:|---:|
| `FIELD_IN_SITU` | 33 | 36 |
| `ORCHARD_IN_SITU` | 3 | 3 |
| `CONTROLLED_LAB` | 5 | 5 |
| `PROTECTED_CULTIVATION` | 1 | 1 |
| `UAV_AERIAL` | 5 | 5 |
| `SATELLITE_REMOTE` | 0 | 0 |
| `UNKNOWN` | 12 | 13 |
| **合计** | **59** | **63** |

近地田间/果园合计 **36 个独立数据集**（33 + 3），占 59 个的 61%。

## 3. Agri-FM+ Table 1 的 35 个来源

| 采集域 | 35 个来源 | 剔除同源重复后的独立来源 |
|---|---:|---:|
| `FIELD_IN_SITU` | 12 | 9 |
| `ORCHARD_IN_SITU` | 2 | 2 |
| `CONTROLLED_LAB` | 5 | 5 |
| `PROTECTED_CULTIVATION` | 1 | 1 |
| `UAV_AERIAL` | 5 | 5 |
| `UNKNOWN` | 10 | 9 |
| **合计** | **35** | **31** |

近地田间/果园 14 个，占 35 个的 40%。其中：

- `agrifm_ref31`（GWHD 2021）与 `agrifm_ref88`（PhenoBench）是**下游保留集**，
  即使域标签是田间，也不能进入预训练；
- `agrifm_ref79` 与 `ronin` 是同一物理数据集，只能计一次；
- `agrifm_ref11`（NDVI 相机）、`agrifm_ref82`（多光谱马铃薯）虽然是田间，但
  **不是纯 RGB**，本轮 2D RGB 语料不使用。

因此"Agri-FM+ 35 个来源"中，真正可用于本轮预训练语料、且不重复、且为近地 RGB 的
只有 9 个独立数据集；再剔除以实验室/受控为主的部分，实际可接入的候选更少。

## 4. 混合来源的处理

| 来源 | 判定 | 处理 |
|---|---|---|
| `must_c` | 地面机器人 + UAV 混合 | `UNKNOWN`，需图像级分离后才能用 |
| `agrifm_ref57`（Crop Pest and Disease Detection） | 官方描述明确写有 white / dark / illuminated / real 四种背景 | `UNKNOWN`，混合来源，不能给统一标签 |
| `perennial_plants` / `agrifm_ref4` | 描述为小批量栽培 + 每周人工除草，未说明是否温室 | `UNKNOWN` |
| `weedcrop` | 派生自 Ronin，官方只说明做了增强，未说明原始采集 | `UNKNOWN` |

关于 `agrifm_ref57`：Stage 5A 的台账里有该来源，论文报告 24,881 张。本轮核验到的
Mendeley 描述原文（DOI `10.17632/bwh3zbpkpv.1`）：

> "The images were captured under various conditions and with different backgrounds
> such as white, dark, illuminated, and real backgrounds."

这直接否定了"名称含 Detection 就是田间"的假设。该来源**不能**作为单一域来源接入；
若要使用，必须逐图分离并保留"real background"子集。本轮未能下载（见下载审计），
因此只记录判定，不产生图像。

## 5. 三个新增来源

| source_id | 官方名称 | 域 | 证据 | 置信度 | 复核状态 |
|---|---|---|---|---|---|
| `sesame_weed` | Crop and Weed Detection Data with Bounding Boxes（芝麻） | `FIELD_IN_SITU` | 发布者描述 + 18/18 视觉抽样 | high | `model_assisted_pending_human` |
| `applebbch81` | AppleBBCH81（果园苹果，BBCH 81-85） | `ORCHARD_IN_SITU` | 论文描述 + 18/18 视觉抽样 | high | `model_assisted_pending_human` |
| `weedcrop` | WeedCrop Image Dataset（Ronin 派生） | `UNKNOWN` | 仅说明增强，未说明采集 | low | `pending_human_review` |

`weedcrop` 因此**被拒绝进入 Field-D1**（`UNKNOWN` 不允许进入主语料），其 2,822 张图像
仍保留在 `data/registry/ingested/` 中，作为后续人工审核或 Mixed/Controlled 研究的
候选，没有被删除。

## 6. 同源去重

`src/registry/source_domains.py:SAME_PHYSICAL_DATASET` 记录 5 组同源映射：

| 重复行 | 归并到 |
|---|---|
| `agrifm_ref4` | `perennial_plants` |
| `agrifm_ref79` | `ronin` |
| `agrifm_ref88` | `phenobench` |
| `agrifm_ref31` | `gwhd_2021` |
| `weedcrop` | `ronin`（派生关系，不计为新来源） |

`field_source_review.csv` 中的 `counted_as_independent` 列把这些行标为 `no`，
所以任何按行数加总的统计都会明确排除它们。

## 7. 本轮新增的字节级发现

`sesame_weed` 的 1,300 张图像与 AgriField-40K 中 `saw_*` 前缀的 1,300 行
**逐字节完全相同**（SHA-256 与解码像素哈希全部命中）。这带来两个结论：

1. `saw` 前缀的真实来源被确认，Stage 5A 的 14,743 张 `pending_audit` 中有一组映射落地；
2. 这 1,300 张**没有**进入 v1 冻结语料，所以把它们加入 Field-D1 是净新增，不是重复计数。
   工具 `tools/stage5b_ingest_field_sources.py` 为此把"命中未采纳的上游行"从
   `cross_source_duplicate` 重新标记为 `upstream_pending_match`，并在审计中单列。
