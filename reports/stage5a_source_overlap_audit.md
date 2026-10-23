# Stage 5A — 跨目录来源重叠审计

> 同一物理数据集可能同时出现在 AgriField-40K、Agri-FM+ 与下游基准中。本文件只登记重叠关系与处置规则；**不合并计数**。

| 重叠组 | source_id | 角色 / 数量证据 | 处置 |
|---|---|---|---|
| phenobench | phenobench | AgriField-40K retains 9,606 PhenoBench images AND Stage 5A downstream benchmark (official v1.1.0); all splits held out | single copy only; never sum retained counts |
| phenobench | agrifm_ref88 | Agri-FM+ Table 1 ref [88]: 129,000 PhenoBench images | single copy only; never sum retained counts |
| perennial_plants | perennial_plants | AgriField-40K: 392 -> 240 retained | single copy only; never sum retained counts |
| perennial_plants | agrifm_ref4 | Agri-FM+ Table 1 ref [4]: 392 images | single copy only; never sum retained counts |
| ronin | ronin | AgriField-40K: 1,176 -> 135 retained | single copy only; never sum retained counts |
| ronin | agrifm_ref79 | Agri-FM+ Table 1 ref [79]: 2,822 images | single copy only; never sum retained counts |
| gwhd_2021 | agrifm_ref31 | Agri-FM+ pretraining pool (6,514 images) | holdout |
| gwhd_2021 | gwhd_2021 | Stage 5A object-detection benchmark (6,514 images) | holdout |
| deepweeds | agrifm_ref60 | Agri-FM+ pretraining pool (17,509 images) | holdout |
| deepweeds | deepweeds | AgriMAE / Stage 6 benchmark (17,509 images) | holdout |

## 处置原则

1. 同一数据集的多份“保留数量”不可相加；v1 只使用本地冻结清单。
2. PhenoBench 与 GWHD 是下游评价保留集，任何预训练 manifest 必须整来源排除。
3. DeepWeeds 若用于 Stage 6 下游，同样整来源排除。
4. 图像级 SHA-256 / pHash 去重只能发现像素重复，不能替代 session/plot 级隔离；跨来源重复候选进入人工复核队列。

## 已知同源对

- `perennial_plants` ↔ `agrifm_ref4`（Kaggle Perennial Plants）
- `ronin` ↔ `agrifm_ref79`（Sudars et al. Data in Brief 2020）
- `phenobench` ↔ `agrifm_ref88` ↔ `phenobench_sem`
- `agrifm_ref31` ↔ `gwhd_2021`
- `agrifm_ref60` ↔ `deepweeds`

