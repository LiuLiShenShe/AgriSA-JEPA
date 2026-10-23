# Stage 5B 总结 — 田间农业语料建设、审计与冻结

> 分支：`agri-stage5b-field-corpus`
> 执行日期：2026-10-23
> 环境：`.venv-uv`，Python 3.13.12，PyTorch 2.9.1+cu128；本轮以 CPU/磁盘/下载为主，
> 未运行 GPU 长训练。
> 所有数字来自本仓库与 `/data/datasets/` 的实际产物，未使用论文报告的数值替代实测值。

## 0. 一句话结论

**B — Field-D1 Candidate。** 新增两个来源、3,138 张可验证的真实田间/果园 RGB 图像，
建立了完整的域政策、去重、保留集隔离与可复现构建链；但新图像的域标签仍依赖
视觉模型抽样（3,138 张待人工签字），因此版本保持 CANDIDATE，不冻结。

---

## 数据来源（1–7）

**1. Stage 5A 的 59 个独立数据集是否都完成采集域分类？**

是。63 条登记行全部有 `acquisition_domain`、证据类型、置信度、复核状态与理由。
其中 4 组同源映射被合并计数，得到 59 个独立数据集。
54 行有官方文档证据，9 行无可用证据并记为 `UNKNOWN`。

**2. Agri-FM+ 的 35 个来源中多少个属于真实近地田间图像？**

14 行（12 个 `FIELD_IN_SITU` + 2 个 `ORCHARD_IN_SITU`）。剔除同源重复后为
11 个独立数据集；其中 GWHD（`ref31`）与 PhenoBench（`ref88`）是下游保留集，
NDVI 玉米（`ref11`）与多光谱马铃薯（`ref82`）不是纯 RGB。
真正可用于本阶段近地 RGB 预训练且非保留集的独立来源约 7 个。

**3. 多少个是实验室、温室、航拍或混合来源？**

21/35：实验室 5、植物工厂 1、航拍 5、无法确认 10（其中 `ref57` 明确是
"white/dark/illuminated/real"四种背景的混合来源）。

**4. 本轮实际下载了哪些新数据集？**

| 来源 | 入口 | 图像 | 归档 SHA-256 |
|---|---|---:|---|
| `sesame_weed` | Kaggle `ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes` | 1,300 | `01de6825…5741c4` |
| `applebbch81` | Kaggle `projectlzp201910094/applebbch81` | 1,838 | `d5f47554…e519a` |
| `weedcrop` | Kaggle `vinayakshanawad/weedcrop-image-dataset` | 2,822 | `db7a8ccb…dfbb` |

**5. 实际下载、解码、去重、审核通过的图片分别是多少？**

下载 5,960；解码成功 5,960（0 失败）；字节/像素重复 0；通过采集域政策 **3,138** 张。

**6. 哪些来源因许可证或网络问题被阻塞？**

- 网络/权限：`agrifm_ref57`、`vcd`、`we3ds`、`grass_clover`、`palmer_amaranth`、
  `rumex_leaves`、`agrifm_ref20`、`agrifm_ref90`。证据见
  `reports/stage5b_download_audit.md` 第 3 节（S3 403、Zenodo 不可达、
  ScienceDirect 403、Cloudflare、WP 下载处理器失效）。
- 许可证/域未确认：`perennial_plants`（MIT vs "Other" 冲突，且采集环境未说明）。

**7. 有多少新图像满足 Field-D1 的采集域要求？**

3,138 张（`sesame_weed` 1,300 + `applebbch81` 1,838）。

---

## PlantVillage（8–10）

**8. PlantVillage 是否已从 Field-D1 完全排除？**

是。来源级域为 `CONTROLLED_LAB`，`admission()` 拒绝；构建后的 manifest 中不存在
`plantvillage` 或 `agrifm_ref93`。测试固化该行为。

**9. 54,284 张历史候选是否完整保留？**

是。54,305 个文件全部保留，54,284 张去重候选的 manifest、哈希、许可与审计
（`reports/stage5a_plantvillage_audit.json`）未删除、未改写。

**10. 其授权、感知候选和未来 Mixed-D1 用途是否明确？**

- 授权：Zenodo 元数据写 `cc-by-4.0`，描述写 "CC BY-SA 3.0 US"，按更严格记录，
  未批准；
- 感知候选：12,436 条仍待复核，状态未变；
- 未来用途：`data/frozen/agri-controlled-d1-candidate_manifest.json`
  定义 Controlled-D1（`DEFINED_NOT_BUILT`），Mixed-D1 定义见
  `agri-mixed-d1-candidate_definition.json`，均未建立、未训练。

---

## 数据质量与泄漏（11–16）

**11. 新增来源中发现多少完全重复图像？**

来源内字节/解码重复 **0**。另发现 1,300 张 `sesame_weed` 图像与 AgriField-40K
`pending_audit` 的 `saw_*` 行逐字节相同——这是**来源映射确认**，不是与已采纳语料
的重复（那些行不在 v1 中）。

**12. 有多少近重复候选？**

pHash≤4：1,049 对，428 个多图连通分量，最大分量 55（标记为链式疑似）。
风险层：单例 4,816、跨组候选 1,013、同组候选 76、链式疑似 55。
dHash 只作支持证据，不参与聚类（用 dHash 聚类会产生 1,337 张的巨型簇，已实测）。

**13. 有多少真实采集序列或场景分组？**

`applebbch81` 140 棵树 / 150 个采集会话（每树多视图）；`weedcrop` 1,544 张原始图
（2,822 个增强版本共享 1,544 个组）；`sesame_weed` 无会话元数据，按单图成组（保守）。
v1 来源的 `component` 是逐图 id，这是继承局限，已在报告中写明。

**14. 是否发现与下游 Holdout 的重叠？**

没有。三个新来源与 12 个保留集的精确哈希命中为 0；v1 与 Field-D1 对 PhenoBench
的精确重复均为 0。感知候选也为 0。

**15. 有没有发生跨预训练 train/val 的采集组泄漏？**

没有。Field-D1 的 `cross_split_groups = 0`；`applebbch81` 的 140 棵树按组划分
（112 train / 28 val）；`sesame_weed` 单图成组。

**16. 哪些候选仍需人工审核？**

- `sesame_weed` 823 张跨组感知候选；
- `weedcrop` 190 + 76 + 55 = 321 张候选/链式疑似；
- v1 遗留 784 条感知候选；
- PlantVillage 12,436 条（不进入 Field-D1，不阻塞本轮）；
- **3,138 张新图像的域标签**需要人工签字。

---

## Field-D1（17–24）

**17. Field-D1 当前实际 train/val 各有多少张？**

train **15,118**；val **3,612**；合计 18,730。

**18. 相比 v1 真正新增多少合格图像？**

**3,138 张**（v1 train 12,567 + 2,551 = 15,118；v1 val 3,025 + 587 = 3,612）。

**19. 数据来源数量和种类是什么？**

8 个来源：继承 6 个（LUCASVision、VegAnn、iNatWeeds、ACRECropWeed、RadishWheat、
SorghumWeed）+ 新增 2 个（Sesame 作物/杂草田间、AppleBBCH81 果园）。

**20. 最大来源占比是多少？**

train 59.1%（`lucas_vision`）。v1 为 71.7%。仍高于 50% 参考目标。

**21. 各来源、设备、场景的分布是什么？**

来源分布见第 20 条；采集域 FIELD_IN_SITU 90.0% / ORCHARD_IN_SITU 10.0%；
设备 100% 为地面手持/车载相机；场景覆盖大田作物、作物-杂草混生、果园果实。

**22. 质量和多样性是否改善？**

- 来源熵 1.438 → 2.036 bit；
- 最大来源占比 71.7% → 59.1%；
- 新增果园域（此前为 0）；
- 新图像短边 ≥ 360 px，0 解码失败、0 灰度；
- 代价：`sesame_weed` 的 823 张视觉相似候选需要人工复核。

**23. 新数据版本的 SHA-256 是什么？**

- `agri-field-d1-candidate_train.csv`：
  `6ba4ba172f09c7d6d38ac7d2ae771400af2f4ae6bb74b57e8ca401023c6ae437`
- `agri-field-d1-candidate_val.csv`：
  `db4054c5e68f119771a6facdd7cc15ad1b7f1842a5c6d118a94f8506457a23d6`

两次独立构建 SHA-256 相同。

**24. 数据是 CANDIDATE、READY_FOR_REVIEW 还是 FROZEN？为什么？**

**CANDIDATE。** 三个冻结门禁中，"采集组不跨划分"与"无来源占比超过 90%"通过，
"全部图像域标签经人工确认"未通过（3,138 张仍是 `model_assisted_pending_human`）。
`--freeze` 参数在门禁未通过时不会提升状态。

---

## 科研计划（25–30）

**25. 当前是否具备正式扩大 JEPA 预训练的条件？**

具备**小规模、可比的 Stage 5C 实验**条件（数据可读、manifest 可复现、DataLoader
兼容），但**不具备**大规模长训练条件：域标签未人工确认、单一来源仍占 59.1%、
采样器（capped）尚未实现。

**26. 新增数据应该按自然分布还是来源均衡方式采样？**

默认 natural；同时把 source_capped(cap=0.5) 作为对照臂。source_balanced 会让
`sorghum_weed` 的 172 张重复约 13.6 倍，只作为第三臂，不作为默认。

**27. 哪些新增来源最可能补足既有田间场景缺口？**

已接入：果园（`applebbch81`）、作物-杂草田间（`sesame_weed`）。
若网络恢复，优先 VCD（早期蔬菜幼苗）、RumexLeaves（牧场机器人近景）、
PalmerAmaranth（杂草生长阶段）、WE3DS 的 RGB 通道；以及 `agrifm_ref57` 的
"real background" 子集（需先做逐图分离）。

**28. 是否建议保留 Controlled-D1 用于后续消融？**

建议保留，但先解决 CC-BY-SA-3.0-US 的署名与传染性，再建版本。Mixed-D1 只在
Controlled-D1 全部门禁通过后才允许建立。

**29. 下一阶段推荐多大的真实训练预算？**

先跑与 B0 完全一致的 **2,000 updates / 有效 batch 128 / 256k 曝光**，
以便与 Stage 3B/4A 的既有结果直接比较。若 Field-D2 达到 25k–30k 张，
再考虑 6,000–10,000 updates；每次扩展都必须同时报告"相同 updates"与
"相同 epoch"两种口径。

**30. 后续如何验证新增 Field-D1 对 PhenoBench、GWHD、PlantDoc 等任务的作用？**

固定模型、初始化、更新预算与下游协议，比较 A / B0 / B1 / B2：
PhenoBench 语义分割 mIoU、GWHD 检测 mAP50-95、PlantDoc 检测 mAP（数据待接入）。
所有比较必须使用同一评价脚本与同一 seed 集合；不得只报告预训练 loss。

---

## 工程（31–35）

**31. 运行的 pytest 结果。**

`499 passed, 7 skipped`（Stage 5A 基线为 466 passed / 7 skipped；新增 33 条
Stage 5B 测试，无既有测试回归）。

**32. 修改的代码和配置。**

新增：
- `src/registry/domain_policy.py` — 七域政策与两级准入；
- `src/registry/field_filter.py` — 质量指标、质量门禁、启发式分诊；
- `src/registry/source_balance.py` — 组成统计与三种采样权重；
- `src/registry/field_freeze.py` — Field-D1 manifest 构建、组级划分、门禁；
- `src/registry/source_domains.py` — 63 行来源域台账（带理由）；
- `tools/stage5b_verify_sources.py`、`stage5b_audit_domains.py`、
  `stage5b_ingest_field_sources.py`、`stage5b_group_duplicate_candidates.py`、
  `stage5b_check_holdout_overlap.py`、`stage5b_build_field_corpus.py`、
  `stage5b_summarize_corpus.py`；
- `tests/test_stage5b_field_corpus.py`（33 条）；
- `configs/stage5c_field_jepa_candidate.yaml`。

未修改：`src/registry/{freeze,ingest,schema,catalog,benchmarks}.py` 的既有逻辑
（仅新增模块调用它们）、Stage 5A 的 checkpoints/结果、v0/v1 冻结文件。

**33. 新增数据路径。**

- 原始归档与解压：`/data/datasets/Stage5B/{sesame_weed,applebbch81,weedcrop}/`
  （Git 之外）；
- 逐图 manifest：`data/registry/ingested/*_stage5b_manifest.csv`；
- 语料版本：`data/frozen/agri-field-d1-candidate_{train,val}.csv` 与 manifest；
- 注册表：`data/registry/agricultural_sources_v2.csv`、
  `field_source_review.csv`、`field_image_review.csv`、
  `field_duplicate_candidates.csv`、`field_source_download.csv`、
  `field_domain_policy.json`。

**34. 最终 Git branch、commit SHA。**

分支 `agri-stage5b-field-corpus`。

- Stage 5B 内容提交（代码、配置、数据清单、报告）：
  `3cd21cad120b60407d01ec5f574766ab8c3cc570`
- 本总结文件写入 commit SHA 的文档更新紧随其后，最终 HEAD 以 `git log` 为准。

**35. 尚未完成的事项与真实阻塞原因。**

1. 3,138 张新图像的域标签未人工签字（阻塞冻结）。
2. `weedcrop` 的 2,822 张处置未定（`UNKNOWN`，需人工判定）。
3. `sesame_weed` 823 张感知候选未复核；v1 784 条遗留候选未复核。
4. `source_capped` / `source_balanced` 采样器未在 DataLoader 中实现。
5. Field-D1 的图像缓存未构建（Stage 5C 启动前必须做）。
6. 8 个来源因网络/权限受阻，非本项目可控。
7. `perennial_plants` 的许可证冲突未解决。
8. PlantVillage 的 CC-BY-SA 条款未定案，故 Controlled-D1 只定义不建立。

---

## 研究决策

**B — Field-D1 Candidate。**

已取得真实田间图像与可复现的候选版本，来源、质量、许可证、重复与泄漏审计均有
可追溯证据；但人工域确认、`weedcrop` 处置与部分候选复核尚未完成。

Stage 5C **可以启动**，但只能以"候选语料 + 明确标注 CANDIDATE"的方式跑
B1/B2 的小预算对照，不能把它写成已冻结的最终语料，也不能据此宣称数据规模收益。
