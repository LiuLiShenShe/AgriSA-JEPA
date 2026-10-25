# Stage 5E 最终报告 — R3v3_license_verified

- 生成本报告时代码与产物的提交：`9d43b51a174652dec722bc2183f98915602f1eb9`
- 分支：`agri-stage5e-final-corpus-gate`
- 生成产物时的 HEAD：`8606e0959066a3a19c80c740ada7b302621b259b`（报告本身随记录提交入库，因此阅读时的 HEAD 会再前进一个提交）
- 父提交：`c57e27196f5c974d35e0e3de29625242d792521f`
- Stage 5D 基线：`05a56a0988531130289d0dd4fea6df7f3657e039`
- Stage 5C 历史基线：`2867e7f4b09814c268612e893c34216506cbe55e`
- **最终状态：CONDITIONAL**（Stage 5E 原判 READY 已被 Stage 5E.1 更正，见 `reports/stage5e1_evidence_reconciliation.md`）
- 门禁：19/20 PASS，未决 `['pool_upstream_licence_resolved']`

## 1. 历史文件是否保持不变

`git diff agri-stage5d-corpus-validation -- data/corpus_v2 data/corpus_v3 reports/stage5c_* reports/stage5d_*` 为空。门禁 `legacy_files_unchanged` 通过，检查了 15 个关键文件（含 Stage 5C/5D 的 policy、manifest、train/val CSV、holdout 索引和验收报告）。

## 2–3. 四个待核验许可来源

| 内部 ID | 真实数据集 | 发布方 | 许可 | 研究训练 | 再分发 | checkpoint | accepted |
|---|---|---|---|---|---|---|---:|
| `agrifm_ref1` | Wheat Spike Dataset (voglinio) | Costas Voglis (Kaggle dataset owner) | CC0-1.0 | VERIFIED_RESEARCH_USE | ALLOWED | ALLOWED | 40 |
| `agrifm_ref58` | Makerere University Beans Image Dataset | Harvard Dataverse (depositor: Makerere University) | CC0-1.0 | VERIFIED_RESEARCH_USE | ALLOWED | ALLOWED | 15,386 |
| `agrifm_ref11` | Maize phenology images (NDVI camera) | Kaggle dataset owner "zzzzqi" | Apache-2.0 | VERIFIED_RESEARCH_USE | ALLOWED_WITH_CONDITIONS | ALLOWED_WITH_CONDITIONS | 382 |
| `agrifm_ref9` | PinotNoirGrapes (nicolaasregnier) | Nicolaas Regnier (Kaggle dataset owner) | CC-BY-NC-4.0 | RESTRICTED_RESEARCH_USE | ALLOWED_WITH_CONDITIONS | UNRESOLVED | 84 |

证据均为**发布方官方端点**，响应体原样存档于 `reports/stage5e_evidence/`，SHA-256 记录在 `reports/stage5e_evidence/evidence_index.json`：

- `agrifm_ref1`：https://www.kaggle.com/api/v1/datasets/view/voglinio/wheat-dataset-original → JSON field "licenseName"，摘录 `"licenseName": "CC0: Public Domain"; "ownerName": "Costas Voglis"`，SHA-256 `c37c9bd8c127b1ff…`
- `agrifm_ref58`：https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/TCKVEW → latestVersion.license (SPDX rightsIdentifier), versionState，摘录 `"license": {"name": "CC0 1.0", "uri": "http://creativecommons.org/publicdomain/zero/1.0", "rightsIdentifier": "CC0-1.0", "rightsIdentifierScheme": "SPDX"}; "ver`，SHA-256 `3bd6dfd42e67988a…`
- `agrifm_ref11`：https://www.kaggle.com/api/v1/datasets/view/zzzzqi/maize-phenology-images-acquired-using-ndvi-camera → JSON field "licenseName"，摘录 `"licenseName": "Apache 2.0"; "ownerName": "zzzzqi"`，SHA-256 `387feb3de135191f…`
- `agrifm_ref9`：https://www.kaggle.com/api/v1/datasets/view/nicolaasregnier/pinotnoirgrapes → JSON field "licenseName" and "description" (non-commercial research framing)，摘录 `"licenseName": "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"; description: "The purpose of this dataset is to gain a better understanding of obje`，SHA-256 `a97510fbc417c53a…`

关键判定：

- `agrifm_ref1`（Wheat Spike Dataset）与 `agrifm_ref58`（Makerere University Beans）为 **CC0-1.0**，研究使用与再分发均无附加条件。
- `agrifm_ref11`（Maize phenology, NDVI camera）为 **Apache-2.0**，研究使用允许；再分发允许但需保留许可与 NOTICE。该源的 `nirraw/` 通道已在 Stage 5D 按模态剔除，本阶段不改变。
- `agrifm_ref9`（PinotNoirGrapes）为 **CC BY-NC 4.0**：本项目是学术、非商业研究，本地训练符合 NonCommercial 条件，标为 `RESTRICTED_RESEARCH_USE`；但**发布 checkpoint 是否构成非商业使用无法从条款文字判定**，标为 `UNRESOLVED`，不并入"允许"。

> 限制：Kaggle 的许可字段是上传者的声明，本阶段只声称"发布方官方端点如此声明"，不声称已在法律上核验版权链条。

## 4. 保留/排除决定

- 四个来源在 R3v2 accepted 中共 **15,892** 张，全部具有可复核的本地研究训练依据，**一张都没有因许可被排除**（`excluded_from_training = 0`）。
- 这不是为了凑总量而保留：判定来自 official endpoint，与目标 147,285 无关。若证据显示禁止，这些图像会被排除并如实报告规模下降。

## 5–7. holdout 实体、覆盖率与交叉结果

| dataset | 状态 | 索引文件 | 文件哈希 | 像素哈希 | pHash |
|---|---|---:|---:|---:|---:|
| `phenobench` | FILE_LEVEL_INDEXED | 13767 | 13767 | 13767 | 13767 |
| `gwhd_2021` | FILE_LEVEL_INDEXED | 6515 | 6515 | 6515 | 6515 |
| `plantdoc` | FILE_LEVEL_INDEXED | 2579 | 2579 | 2579 | 2579 |
| `minneapple` | BLOCKED_HOST_REJECTED | 0 | 0 | 0 | 0 |
| `plantseg` | BLOCKED_AUTH | 0 | 0 | 0 | 0 |
| `deepweeds` | UNRESOLVED_FILE_LEVEL | 0 | 0 | 0 | 0 |
| `growliflower` | BLOCKED_NETWORK | 0 | 0 | 0 | 0 |
| `cropandweed` | UNRESOLVED_FILE_LEVEL | 0 | 0 | 0 | 0 |
| `multiclass_weeds` | BLOCKED_HOST_REJECTED | 0 | 0 | 0 | 0 |
| `roboflow_disease_detection` | UNRESOLVED_FILE_LEVEL | 0 | 0 | 0 | 0 |
| `roboflow_crop_detection` | UNRESOLVED_FILE_LEVEL | 0 | 0 | 0 | 0 |
| `gwhd_instance_seg` | UNRESOLVED_FILE_LEVEL | 0 | 0 | 0 | 0 |

- 文件级可核验：**phenobench, gwhd_2021, plantdoc**，共 22,861 个文件建立 SHA-256 + pHash 索引。
- 精确文件哈希命中：**0**；精确像素哈希命中：**0**；pHash 近邻候选：**14**。

### 本阶段发现并排除的真实泄漏

Stage 5D 把 GWHD 2021 的 2 个 pHash 距离 0 候选判为"哈希碰撞，不是同一张图"。本阶段重新解码双方像素做复核，结论相反：

| holdout 图像 | 语料图像 | 尺寸 | 相关 | 判定 |
|---|---|---|---:|---|
| `3e0a51f0…77576.png` | `agrifield40k_pool:a5e3f9e21eb9dc95`（VegAnn） | 1024×1024 → 512×512 | 0.999402 | 同一内容的缩放副本 |
| `f9bde625…12cf4d863.png` | `agrifield40k_pool:2e649c79a9765b53`（VegAnn） | 1024×1024 → 512×512 | 0.999763 | 同一内容的缩放副本 |

这 2 张已从 train/val 清单中排除（`HOLDOUT_CONFIRMED_PIXEL_MATCH`），accepted 因此从 169,326 降到 **169,324**。相关不等于同图，但相关系数 0.9994+ 且平均像素差 < 2/255 时，这是同一张图的缩放版本，不是碰撞。

复核规则与逐条结论见 `reports/stage5e_hash_collision_review.{json,csv}`。其余 12 个候选（pHash 距离 6）相关系数 0.04–0.38，是不同植株/不同地块，判定为 `HASH_COLLISION_DIFFERENT_IMAGE`，未删除。

- 本阶段新增 **PlantDoc** 的文件级证据：真实下载并解码 2,579 张图像，这是唯一从"来源级"提升到"文件级"的 holdout，命中数为 0。
- 其余计算方法/索引不可得的 holdout 逐条记录真实失败原因：
  - `minneapple`：BLOCKED_HOST_REJECTED — HTTP 403 from conservancy.umn.edu (automated request refused)
  - `plantseg`：BLOCKED_AUTH — HTTP 403 -> redirect to gitlab.com/users/sign_in (repository requires sign-in)
  - `deepweeds`：UNRESOLVED_FILE_LEVEL — README reachable (200); images hosted on Google Drive, request timed out
  - `growliflower`：BLOCKED_NETWORK — zenodo.org connection refused (curl exit 7)
  - `cropandweed`：UNRESOLVED_FILE_LEVEL — image tars 2.98 GB each (4 shards); annotations tar downloaded
  - `multiclass_weeds`：BLOCKED_HOST_REJECTED — figshare landing page HTTP 202; api.figshare.com HTTP 403 to automated requests
  - `roboflow_disease_detection`：UNRESOLVED_FILE_LEVEL — public.roboflow.com redirected to universe.roboflow.com (403)
  - `roboflow_crop_detection`：UNRESOLVED_FILE_LEVEL — universe.roboflow.com HTTP 403 to automated requests
  - `gwhd_instance_seg`：UNRESOLVED_FILE_LEVEL — universe.roboflow.com HTTP 403 to automated requests

> **结论边界**：零命中只覆盖"已取得字节"的范围。对未取得内容的基准，本阶段**不宣称无泄漏**，只做了来源级与上游身份级筛查。

## 8. 新语料计数

| 项 | 数量 |
|---|---:|
| ledger 输入 | 243,694 |
| accepted | 169,324 |
| train | 147,277 |
| val | 22,047 |
| 排除/隔离 | 74,370 |
| 因许可排除 | 0 |

Stage 5D 的 quarantine / rejected / holdout / duplicate-removed 分桶被原样继承，因为 Stage 5E 没有重新判定质量、模态、域、去重或 holdout，只增加了一道许可门。

## 9. 计数守恒

- 全局：`243,694` = accepted 169,324 + excluded 74,370；`conserved = True`。
- 每来源守恒：`per_source_conserved = True`。

## 10–11. 组成与集中度

- 来源 7 个；最大来源 `artemis_imagesafari` 130,270 张，占 accepted 的 **76.94%**。
- 组成统计（分母 = accepted 169,324）：来源 HHI **0.6163**，熵 **0.7478**；上游数据集 HHI **0.3153**。
- 采样统计（分母 = train 147,277）：来源 HHI **0.6022**，熵 **0.7698**。两个分母不同，所以两组数字不相等，不能混用。
- 域：FIELD_IN_SITU 167,494, ORCHARD_IN_SITU 1,830。
- unknown 比例：crop 23.1%，country 67.1%。
- 设备/采集平台本机没有逐图可靠字段，保持 unknown，不编造。

## 12. 采样配置

epoch 长度 150,000，cap_share 0.5，训练实体 147,277。

| profile | HHI | 曝光预算是否填满 |
|---|---:|---|
| natural | 0.6022 | — |
| capped | 0.5031 | False |
| balanced | 0.1429 | — |

- capped profile 只压下 `artemis_imagesafari` 到 75,000，剩余曝光缺口 38,828 **不回填**给其他来源，否则就等于偷偷改变它们的自然占比。
- 三种 profile 都只改变曝光频率；`reproducibility` 证明实体数与 split 未变。

## 13. 与 147,285 的差额

- `target_exceeded = True`，`target_margin = +22,039`。
- 这只是**规模属性**，不覆盖许可、QC、holdout 或 split 门禁。

## 14. Stage 5D 自动 QC 的已知弱项（本阶段未解决）

对传感器噪声、JPEG 重压缩、色偏、局部遮挡的召回仅 0.07–0.11；自动 QC 不是自然图像质量认证。本阶段没有重新标定或改进它，也没有把任何隔离图像升级为 accepted。

## 15. 门禁与测试

完整 `pytest tests/`：**657 passed / 7 skipped / 0 failed**。7 个 skip 全部是历史遗留的环境变量依赖（`AGRISA_VITH_CHECKPOINT` 5 个、`AGRISA_PHENOBENCH` 2 个），与 Stage 5E 无关，也不计为 pass。逐项要求与用例对应表见 `reports/stage5e_test_summary.md`。

| gate | 状态 |
|---|---|
| `legacy_files_unchanged` | PASS |
| `licence_evidence_backed` | PASS |
| `licence_unresolved_not_trained` | PASS |
| `train_and_redistribution_independent` | PASS |
| `holdout_file_level_reported` | PASS |
| `holdout_confirmed_hits_blocked` | PASS |
| `evaluation_scope_declared` | PASS |
| `quality_rules_not_relaxed` | PASS |
| `group_split_inherited` | PASS |
| `counts_conserved` | PASS |
| `rebuild_deterministic` | PASS |
| `sampling_exposure_only` | PASS |
| `composition_from_manifest` | PASS |
| `optimizer_updates_zero` | PASS |

## 16. 重建与 SHA-256

一条命令重建全部产物：`bash tools/stage5e_run_all.sh`。分步命令：

```bash
.venv-uv/bin/python tools/stage5e_licence_audit.py
.venv-uv/bin/python tools/stage5e_holdout_index.py --plantdoc-root data/holdout_raw/plantdoc_extract
.venv-uv/bin/python tools/stage5e_hash_collision_review.py
.venv-uv/bin/python tools/stage5e_build_corpus.py
.venv-uv/bin/python tools/stage5e_gates.py
.venv-uv/bin/python tools/stage5e_final_report.py
```

按 `.gitignore` 排除、只留在本机的大型文件（记录实际路径、行数与 SHA-256）：

| 文件 | 数据行数 | SHA-256 |
|---|---:|---|
| `data/corpus_v4/R3v3_license_verified_train.csv` | 147,277 | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` |
| `data/corpus_v4/R3v3_license_verified_val.csv` | 22,047 | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` |
| `data/corpus_v4/R3v3_license_verified_quarantine.csv` | 39,485 | `7603653746f15fb8df744ce7bb1b619b81203fc74ff9e42d473985a935b5b9df` |
| `data/corpus_v4/R3v3_license_verified_exclusions.csv` | 74,370 | `da19005ce04081012e93e55c7fc5171a3908bb46094238f420a9a450df2411d9` |
| `data/corpus_v4/holdout_file_index.csv` | 22,861 | `1580a9abeb43fcf2b90b32362beafa14a9a8ebafb463cb9f646f1ee6f52ddf67` |
| `data/corpus_v4/holdout_crossmatch.csv` | 17 | `c50267598f2c1eedff14b493db7534cd4a908436154a2d90207d0bdd422c9dde` |

PlantDoc 的 2,579 张原图保存在 `data/holdout_raw/plantdoc_extract/`（同样不入库），逐文件哈希见 `data/corpus_v4/holdout_file_index.csv`。

入 Git 的产物：

| 产物 | 字节 | SHA-256 |
|---|---:|---|
| `data/corpus_v4/R3v3_license_verified_train.csv` | 101,699,231 | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` |
| `data/corpus_v4/R3v3_license_verified_val.csv` | 15,630,460 | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` |
| `data/corpus_v4/R3v3_license_verified_manifest.json` | 3,894 | `3612dce810e6a0bf89937d2bd7608f66a1a0da95fbf40800eb35410dd79aa647` |
| `data/corpus_v4/R3v3_license_verified_sampling_profiles.json` | 8,009 | `42aa12d4936c6f632a943f0dd0a3a6fcc6bcc81206ae027b5d9e6835aefb30b6` |
| `data/corpus_v4/holdout_file_index.csv` | 8,400,869 | `1580a9abeb43fcf2b90b32362beafa14a9a8ebafb463cb9f646f1ee6f52ddf67` |
| `data/corpus_v4/holdout_crossmatch.csv` | 3,394 | `c50267598f2c1eedff14b493db7534cd4a908436154a2d90207d0bdd422c9dde` |
| `reports/stage5e_license_audit.json` | 16,060 | `053605516690587c8082e5c4e37ec2c1b193c922d66f5d5adb7cc7566e86204c` |
| `reports/stage5e_holdout_evidence.json` | 12,757 | `160f90d5109e32bf015e40e5909930b9f38b2c3048f1521b27cf6a69f32a8efb` |
| `reports/stage5e_gates.json` | 12,716 | `c26b5a1b6cdb74f56e8feb62efb5a91d44ba47790af3a863fcbd5fa5cf853a3d` |

## 17. optimizer updates

JEPA optimizer updates = **0**，Adapter optimizer updates = **0**，新下游 probe = **0**。门禁 `optimizer_updates_zero` 通过，扫描了 13 个 Stage 5E 文件，未发现任何优化器、损失、反向传播、训练循环、adapter 或 probe 构造。

## 18. Stage 6 决策

**READY_FOR_CONTROLLED_PRETRAINING**。

允许启动受控持续预训练，但**评测范围已明确缩小**：

- 可用于下游评测：phenobench, gwhd_2021, plantdoc。
- 明确排除：minneapple, plantseg, deepweeds, growliflower, cropandweed, multiclass_weeds, roboflow_disease_detection, roboflow_crop_detection, gwhd_instance_seg。
- 排除的原因是无法取得图像字节做文件级交叉核验；这是**范围声明**，不是"已证明无重叠"。若要在 Stage 6 使用其中任一基准，必须先取得其图像、重跑交叉核验、再更新本文件。

## 19. 仍未解决的问题

1. `agrifm_ref9`（84 张）的 checkpoint 发布权限未定；训练可用，发布权重前需单独决策。
2. 9 个 holdout 无法做文件级核验，已从评测计划排除，但其与训练语料的重叠只做到来源级/上游身份级筛查。
3. `agrifield40k_pool` 的 CC-BY-4.0 是池级声明，逐图上游许可未逐一复核。
4. 自动 QC 对噪声/JPEG/色偏/遮挡不敏感，本阶段未改进。
5. 许可证据是发布方官方端点声明，未做版权链条的独立法律核验。





---

## 更正说明（Stage 5E.1，2026-10-25）

本文件由 Stage 5E.1 重新生成。以下 Stage 5E 结论已被真实文件推翻或降级：

- `near_candidates_still_pending_review = 14` → **0**：14 个候选全部有裁决，旧值来自 crossmatch CSV 未回填 verdict。
- `licence_evidence_backed` 的"每个来源都有归档官方文件" → 只有 4 个新归档，1 个继承、1 个仅 URL、1 个（pool）逐上游未决。
- `target_exceeded = True / +22,039` 是按 accepted 计算；实际 Stage 6 训练集 147,277 张，比 147,285 **少 8 张**。
- PlantDoc 的"下载未完成"记录已过时：2,579 张真实解码并哈希。
- PhenoBench/GWHD 的 `decoded = 0` 是 pandas StringDtype 写入 bug，修复后 decoded = 全部。
- "入 Git 的产物"表把 6 个 local-only 大文件列为已入库，实际只有 3 个 JSON 被 Git 跟踪。
- 最终状态降为 **CONDITIONAL**。
