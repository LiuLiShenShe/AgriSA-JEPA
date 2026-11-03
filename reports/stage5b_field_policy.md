# Stage 5B — 采集域分类政策（可执行规则）

> 代码：`src/registry/domain_policy.py`、`src/registry/field_filter.py`
> 机器可读版本：`data/registry/field_domain_policy.json`
> SHA-256：`87e9ff45d125f4c02001603e3f83ba20ea81b710007f840a6827be43752ad3c0`

Stage 5A 记录了 63 行来源登记，但没有记录"相机在哪"。本文件把这一步写成可执行规则，
并把七个采集域、三种证据强度、四档置信度、四种复核状态组合成一个准入函数。

## 1. 采集域定义

| 域 | 定义 | Field-D1 |
|---|---|---|
| `FIELD_IN_SITU` | 真实农田、农田手持设备、地面机器人拍摄、田间自然光照下的作物/杂草/病害/虫害 | 允许 |
| `ORCHARD_IN_SITU` | 真实果园、葡萄园、茶园等生产环境 | 允许 |
| `PROTECTED_CULTIVATION` | 真实温室、设施农业、植物工厂 | 单独登记，默认排除 |
| `CONTROLLED_LAB` | 纯色背景离体叶片、实验室拍摄台、扫描件 | 排除 |
| `UAV_AERIAL` | 无人机、航拍、正射 | 排除 |
| `SATELLITE_REMOTE` | 卫星遥感 | 排除 |
| `UNKNOWN` | 采集环境无法可靠确认 | 排除 |

田间特写仍然可以采纳：只要采集上下文是农田而不是工作台（见 `rumex_leaves`）。
反过来，数据集名称含 crop/weed/plant 不构成任何证据。

## 2. 两层准入

```
candidate（进入 agri-field-d1-candidate）：
    domain ∈ {FIELD_IN_SITU, ORCHARD_IN_SITU}
    evidence ∈ {official_documentation, manual_inspection, model_assisted_inspection}
    confidence ∈ {high, medium}
    review_status ∈ {reviewed, model_assisted_pending_human}

freeze（进入 FROZEN 版本）：
    domain ∈ {FIELD_IN_SITU, ORCHARD_IN_SITU}
    evidence ∈ {official_documentation, manual_inspection}   ← 不含模型证据
    confidence ∈ {high, medium}
    review_status = reviewed                                  ← 必须人工记录
```

模型输出只用于排序待审队列，永远不能单独成为冻结依据。`src/registry/field_filter.py`
中的 `suggest_domain()` 返回的标签一律写入 `domain_evidence_type =
model_assisted_inspection`，且 `domain_confidence = unknown`。

## 3. 质量门禁

`QualityGate` 默认参数：

- `min_short_side = 224`：Agri-FM+ 使用的下限。缩放前测量，**不通过放大低分辨率图像来
  通过本门禁**；未通过者在 manifest 中记为排除并保留原文件。
- `exclude_grayscale = True`：本阶段只做 RGB。
- `min_laplacian_variance = None`：默认只报告模糊分布，不设全局阈值。低纹理土壤图像
  的 Laplacian 方差天然低，全局阈值会误删合法样本。若要启用，按来源单独设定并记录依据。

各来源实测短边分布见 `reports/stage5b_quality_audit.md`；本轮三个新来源全部 ≥ 360 px，
无一低于 224 px。

## 4. 保留集规则

`data/registry/downstream_holdout_sources.csv` 中的 12 个来源，无论域标签如何、无论是否
出现在 Agri-FM+ 原始来源表中，一律不进入任何预训练 manifest。`apply_gate()` 在域判定
**之前**执行该规则，白名单参数无法覆盖。

## 5. 来源级与图像级判断的分工

- 来源级：`data/registry/field_source_review.csv`，63 行 + 1 行新增，逐行记录
  `acquisition_domain` / `domain_evidence_type` / `domain_confidence` /
  `domain_review_status` / `domain_review_reason`。
- 图像级：`data/registry/field_image_review.csv`，仅对本地有数据的 10 个来源各抽
  40 张做分层抽样画像，作为人工复核的排队依据。

判定依据全部来自官方文档原文（论文摘要、数据集卡、Kaggle/Mendeley 描述），不存在从
文件名或数据集体量推断出来的域标签；没有文档证据的行一律记为 `UNKNOWN` + `none`。

## 6. 设计取舍

| 决定 | 理由 |
|---|---|
| 保留 `UNKNOWN` 而不是猜一个域 | 猜测的域标签会污染后续所有统计，且无法追溯 |
| `model_assisted_inspection` 可进 candidate、不可进 freeze | 冻结是承诺；模型不是证据链的一环 |
| `PROTECTED_CULTIVATION` 单独登记而非并入 FIELD | 温室光照、冠层结构与开放田差异大，混入后无法解释下游结果 |
| 模糊门禁默认关闭 | 全局阈值会误删低纹理土壤/远景田块；本轮改为按来源报告分布 |
