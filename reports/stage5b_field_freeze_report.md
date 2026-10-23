# Stage 5B — Field-D1 候选版本冻结报告

> 机器可读：`reports/stage5b_field_freeze_report.json`、
> `data/frozen/agri-field-d1-candidate_manifest.json`
> 产物：`data/frozen/agri-field-d1-candidate_{train,val}.csv`
> 工具：`tools/stage5b_build_field_corpus.py`（自动摘要另存
> `reports/stage5b_field_freeze_auto.md`）

## 1. 版本状态

**`agri-field-d1-candidate` — CANDIDATE（未冻结）**

| 指标 | 数值 |
|---|---:|
| train 图像 | 15,118 |
| val 图像 | 3,612 |
| 合计 | 18,730 |
| 来源数 | 8 |
| 采集组总数 | 17,032 |
| 跨 train/val 的采集组 | 0 |
| 待人工确认域标签的图像 | 3,138 |

`train.csv` SHA-256：`6ba4ba172f09c7d6d38ac7d2ae771400af2f4ae6bb74b57e8ca401023c6ae437`
`val.csv` SHA-256：`db4054c5e68f119771a6facdd7cc15ad1b7f1842a5c6d118a94f8506457a23d6`

两次独立构建的 SHA-256 相同（已验证）。

## 2. 冻结门禁

| 条件 | 结果 | 说明 |
|---|---|---|
| 采集组不跨 train/val | 通过 | 交集为 0 |
| 全部图像域标签经人工确认 | **未通过** | 3,138 张新图像仍是 `model_assisted_pending_human` |
| 无来源占比超过 90% | 通过 | 最大 59.1%（train） |

两项通过、一项未通过，因此保持 CANDIDATE。`--freeze` 参数不会强制提升状态：
`freeze_ok` 为 False 时即使传了也写 CANDIDATE。

## 3. 继承与新增

### 3.1 继承自 agrifield40k-v1

- v1 train/val 的 15,592 张全部读取，全部进入 Field-D1（`status == include`）。
- v1 的 6 个来源全部通过域政策（FIELD_IN_SITU）。
- v1 的两个 SHA-256 与 Stage 2D 记录完全一致，v1 文件未被修改。
- v1 的**未决风险被继承而不是清除**：784 条感知候选未复核、14,743 张
  `pending_audit` 行仍未采纳。这两件事不阻塞 Field-D1 候选，但会阻塞
  "Field-D1 已完全冻结"的结论。

### 3.2 本轮新增

| 来源 | 域 | 图像 | train | val | 判定依据 |
|---|---|---:|---:|---:|---|
| `sesame_weed` | FIELD_IN_SITU | 1,300 | 1,040 | 260 | 发布者描述 + 18/18 视觉抽样 |
| `applebbch81` | ORCHARD_IN_SITU | 1,838 | 1,511 | 327 | 论文描述 LatHort 果园 + 18/18 视觉抽样 |

净新增合格田间图像 **3,138 张**，全部为地面相机拍摄。

### 3.3 被排除

| 来源 | 图像 | 原因 |
|---|---:|---|
| `weedcrop` | 2,822 | 采集域 `UNKNOWN`，未审核不进主语料 |

排除的图像**没有被删除**，其 manifest 与哈希保留在 `data/registry/ingested/`。

## 4. manifest 结构

`agri-field-d1-candidate_manifest.json` 包含：状态、图像数、来源数、按来源与按域组成、
划分完整性（组数、跨划分组数、是否 disjoint）、两个 CSV 的 SHA-256、loader 契约列、
政策版本、v1 继承记录、新来源与排除记录、门禁拒绝统计、分布摘要。

manifest 的 SHA-256 取自 `canonical_csv()` 的序列化结果，字段顺序固定
（`FIELD_FIELDS`），换行符固定为 `\n`，所以可跨机器复现。

## 5. 与既有版本的关系

| 版本 | 状态 | 本轮是否改动 |
|---|---|---|
| `agrifield40k-v0` | 历史冻结 | 未改动（SHA-256 不变） |
| `agrifield40k-v1` | PROVISIONAL | 未改动（SHA-256 不变） |
| `agri-corpus-d1-candidate` | 历史候选（PlantVillage 主导） | 未改动（SHA-256 不变） |
| `agri-field-d1-candidate` | CANDIDATE（新） | 本轮建立 |
| `agri-controlled-d1-candidate` | DEFINED_NOT_BUILT | 见第 6 节 |
| `agri-mixed-d1-candidate` | DEFINED_NOT_BUILT | 见第 6 节 |

## 6. Controlled-D1 与 Mixed-D1 的定义

**Controlled-D1**：只包含受控/实验室来源。本轮写出
`data/frozen/agri-controlled-d1-candidate_manifest.json`，声明：

- 来源为 PlantVillage（`agrifm_ref93`），54,305 张文件、54,284 张去重候选；
- 图像清单指向既有 `data/registry/ingested/plantvillage_manifest.csv`，**不复制行**，
  避免仓库里出现两份互相漂移的 PlantVillage 清单；
- 状态 `DEFINED_NOT_BUILT`。原因是 CC-BY-SA-3.0-US 的传染性与署名未定案、
  12,436 条感知候选未复核、来源许可存在元数据冲突。

**Mixed-D1**：写出 `data/frozen/agri-mixed-d1-candidate_definition.json`，明确
组成规则（Field-D1 + Controlled-D1）、用途（检验受控叶片是否帮助或损害田间迁移）
与前置条件。本轮不建立文件、不训练。

## 7. 为什么不是 FROZEN

冻结意味着"这批图像可以写入论文的数据声明，且不再变动"。当前 3,138 张新图像的域
标签建立在视觉模型抽样上，抽样量是 18/1,838 与 18/1,300。要冻结，需要：

1. 对 `sesame_weed`、`applebbch81` 的逐图域确认，或至少按来源签字确认并记录理由；
2. 解决 `weedcrop` 的处置；
3. 复核 v1 的 784 条感知候选（或明确记录"v1 遗留问题不影响 Field-D1 冻结"的决策）。

这三件事都不是本轮能伪造的，因此保持 CANDIDATE。
