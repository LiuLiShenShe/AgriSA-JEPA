# Stage 5B — 采集域审计汇总

> 机器可读：`reports/stage5b_domain_audit.json`
> 产物：`data/registry/field_source_review.csv`、`data/registry/field_image_review.csv`

## 1. 来源级（63 行 / 59 个独立数据集）

| 域 | 独立数据集 | 说明 |
|---|---:|---|
| `FIELD_IN_SITU` | 33 | 开放田、农田手持/机器人、农田自然光照近景 |
| `ORCHARD_IN_SITU` | 3 | `minneapple`、`agrifm_ref18`、`agrifm_ref9` |
| `CONTROLLED_LAB` | 5 | 含 PlantVillage、离体叶片、种子/幼苗台拍 |
| `PROTECTED_CULTIVATION` | 1 | `agrifm_ref90` 植物工厂 |
| `UAV_AERIAL` | 5 | `agrifm_ref2/3/46/48/74` |
| `SATELLITE_REMOTE` | 0 | 登记表中无纯卫星源 |
| `UNKNOWN` | 12 | 无证据或混合采集 |
| **合计** | **59** | |

其中 12 个是下游保留集（域标签不影响其用途——它们只用于评价）。

## 2. 图像级（400 张抽样，10 个本地来源）

分诊标签仅用于排序；真正的域标签来自来源级文档 + 视觉抽样。统计见
`reports/stage5b_quality_audit.md` 第 2、4 节。

## 3. 关键判定举例

| 来源 | 标签 | 证据类型 | 理由摘要 |
|---|---|---|---|
| `plantvillage` | `CONTROLLED_LAB` | official | 纯色背景离体叶片；8/9 视觉抽样识别出受控痕迹 |
| `rumex_leaves` | `FIELD_IN_SITU` | official | 叶片近景但采集于牧场机器人作业环境；田间特写保留 |
| `agrifm_ref57` | `UNKNOWN` | official | 官方原文写有 white/dark/illuminated/real 四种背景 |
| `must_c` | `UNKNOWN` | official | 多平台（地面机器人 + UAV），无法给统一标签 |
| `agrifm_ref90` | `PROTECTED_CULTIVATION` | official | 室内植物工厂 |
| `agrifm_ref37` | `CONTROLLED_LAB` | official | 标准化背景幼苗分类数据集 |
| `weedcrop` | `UNKNOWN` | filename/metadata | 只说明增强，未说明原始采集环境 |
| `lucas_vision` | `FIELD_IN_SITU` | official | 相机架对地面拍摄农田地块，有地块与日期记录 |
| `inat_weeds` | `FIELD_IN_SITU` | official | iNaturalist 就地观测；部分为花园/栽培，故 medium 而非 high |
| `sesame_weed` | `FIELD_IN_SITU` | official | 发布者描述 + 18/18 视觉抽样 |
| `applebbch81` | `ORCHARD_IN_SITU` | official | 论文描述 LatHort 果园 + 18/18 视觉抽样 |

## 4. 域标签的不可推翻约束

`apply_gate()` 的执行顺序是：保留集 → 精确重复 → 域政策 → 哈希完备性。也就是说，
即使某行被写进 `allow_sources` 白名单，只要它属于 12 个保留集来源，就会被剔除。
单元测试 `test_whitelist_cannot_override_holdout` 固化该行为。
