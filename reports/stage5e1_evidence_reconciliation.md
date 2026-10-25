# Stage 5E.1 证据一致性修复与 Stage 6 开训前复核

- 分支：`agri-stage5e1-evidence-reconciliation`
- 生成产物时的 HEAD：`5e9bc985541d379b233bb818dcdfe755c5e81ec8`（所有产物的数据基准；不随报告提交变化）
- Stage 5E.1 代码与产物提交：`见包含本报告的那个提交`
- Stage 5E 基线 HEAD：`5e9bc985541d379b233bb818dcdfe755c5e81ec8`
- Stage 5D 基线：`05a56a0988531130289d0dd4fea6df7f3657e039`
- **最终状态：CONDITIONAL**
- 门禁：18/20 PASS，未决：`['licence_evidence_backed', 'pool_upstream_licence_resolved']`，失败：`[]`
- 测试：**709 passed / 7 skipped / 0 failed**
- 本阶段 optimizer updates：JEPA **0**，Adapter **0**，下游 probe **0**
- 三个 HEAD 口径分开记录，不混写：`head_at_artifact_generation`（工具运行时的 HEAD，产物数据基准）、`stage5e1_code_commit`（包含 Stage 5E.1 工具/测试/重建产物的提交）、`final_branch_head`（该提交本身）。

## 1. 结论先行

Stage 5E 的 14/14 PASS 与 `READY_FOR_CONTROLLED_PRETRAINING` **不能保留**。门禁代码本身没有说谎，但三类口径不一致使"证据充分"的主张站不住：

1. **许可证据等级被夸大**。`licence_evidence_backed` 的描述声称每个 accepted 来源都有"已归档且 SHA 已核对的官方文件"，实际只有 4 个来源在 Stage 5E 新归档；另 4 个只是沿用 Stage 5D registry 行。
2. **agrifield40k_pool 的逐上游许可未落实**。池级 CC-BY 标签不能推出第三方图像都可由池维护者再许可。11 个有 accepted 图像的上游中，5 个有 pin 住的编译卡/逐图元数据依据，6 个仍是 `pending_audit`；受影响 accepted **6,120** 张（train 5,174 张）。
3. **"比 Agri-147K 更大" 的基准被混用**。169,324 是 train+val 总量；真正参与 Stage 6 optimizer updates 的是 train **147,277**，比 147,285 **少 8 张**。

因此状态降为 **CONDITIONAL**，阻塞项逐条列在第 8 节。R3v3 **成员未变**，train/val 字节未变。

## 2. PlantDoc：真实文件数与 hash 覆盖

- 解压目录 `data/holdout_raw/plantdoc_extract/` 共 2,581 个文件：2,579 张图像 + `LICENSE.txt` + `README.md`。
- 索引 `holdout_file_index.csv` 中 plantdoc 行数 **2,579**，成功解码 **2,579**，raw SHA-256 **2,579**，规范化像素 SHA-256 **2,579**，pHash **2,579**。
- 抽样 5 张逐字节重算 raw SHA-256 与像素 SHA-256，全部一致。
- Stage 5E evidence 里"下载未完成、只有 Tree API blob 清单、未取得真实图像字节"的记录**已过时**：本轮磁盘上确有完整图像并已逐张解码。该历史观察保留在 registry notes 中，但不再作为当前状态。

**结论：PlantDoc 2,579/2,579 张完成文件级 hash，content_verified 成立。**

## 3. PhenoBench / GWHD：decode 与 pixel hash 矛盾的原因

Stage 5E 的索引里两者同时出现两个互斥事实：`pixel_sha256` 非空，但 `decode_ok=false`、宽高/字节为空，`decode_error` 是 `TypeError: Invalid value '1024' for dtype 'str'`。

根因不是解码失败，而是 **pandas 3.1 的 `StringDtype` 拒绝了把 int 赋给字符串列**：

```python
frame.at[row_index, 'pixel_sha256'] = canonical_pixel_sha256(image)  # 先成功
frame.at[row_index, 'width'] = width   # 抛 TypeError -> 被 except 记成 decode_ok=false
```

像素哈希在异常前已经写入，所以出现"有 hash 但 decode_ok=false"。修复是把 width/height/bytes 显式转成 `str`，并在异常分支清空 `decode_error`。重建后：

| dataset | files | decoded | raw hash | pixel hash | pHash |
|---|---:|---:|---:|---:|---:|
| phenobench | 13,767 | 13,767 | 13,767 | 13,767 | 13,767 |
| gwhd_2021 | 6,515 | 6,515 | 6,515 | 6,515 | 6,515 |
| plantdoc | 2,579 | 2,579 | 2,579 | 2,579 | 2,579 |

抽样重算 PhenoBench/GWHD 的 raw 与像素哈希，全部与索引一致。此前"pixel hash 为空、未回填"的说明与真实索引矛盾，已删除。

## 4. 14 个近重复候选：pending 数 = 0

- `stage5e_hash_collision_review.json` 记录 14 个候选全部复核：2 个 `RESAMPLED_COPY_SAME_CONTENT`，12 个 `HASH_COLLISION_DIFFERENT_IMAGE`。
- 旧 `holdout_crossmatch.csv` 的 14 行仍写着 `CANDIDATE_REQUIRES_REVIEW`，gate 直接数这个字段，于是报 `near_candidates_still_pending_review = 14`。
- 本轮按 `(dataset_id, holdout_id, corpus_id)` 一对一合并裁决：**14 个候选 → 14 条裁决，0 漏配、0 多配、0 重复 ID、0 仍 pending**。
- 2 个确认副本 `agrifield40k_pool:a5e3f9e21eb9dc95`、`agrifield40k_pool:2e649c79a9765b53` 已不在 train，也不在 val，只出现在 exclusions（`HOLDOUT_CONFIRMED_PIXEL_MATCH`）；原图未删除。

## 5. 许可证据：哪些新归档、哪些继承

| 来源 | 等级 | accepted | train | 依据 |
|---|---|---:|---:|---|
| `agrifield40k_pool` | 未决 | 21,416 | 18,240 | `data/source_registry.json` |
| `agrifm_ref1` | 新归档并核对 SHA-256 | 40 | 40 | `reports/stage5e_evidence/agrifm_ref1_kaggle_api_voglinio_wheat-dataset-original.json` |
| `agrifm_ref11` | 新归档并核对 SHA-256 | 382 | 382 | `reports/stage5e_evidence/agrifm_ref11_kaggle_api_zzzzqi_maize-phenology-images-acquired-using-ndvi-camera.json` |
| `agrifm_ref58` | 新归档并核对 SHA-256 | 15,386 | 15,386 | `reports/stage5e_evidence/agrifm_ref58_harvard_dataverse_TCKVEW_api.json` |
| `agrifm_ref9` | 新归档并核对 SHA-256 | 84 | 84 | `reports/stage5e_evidence/agrifm_ref9_kaggle_api_nicolaasregnier_pinotnoirgrapes.json` |
| `applebbch81` | 仅 registry URL | 1,746 | 1,383 | `data/corpus_v3/sources.csv` |
| `artemis_imagesafari` | 继承既有 pin 住证据 | 130,270 | 111,762 | `data/corpus_v3/sources.csv` |

- **新归档（4）**：agrifm_ref1 / ref58 / ref11 / ref9，证据体在 `reports/stage5e_evidence/`，SHA-256 与归档一致。
- **继承（1）**：artemis_imagesafari，引用 `data/corpus_v3/sources.csv`（Stage 5D registry，本轮记录其 SHA-256），本阶段**没有**重新抓取 AWS registry。
- **仅 registry URL（1）**：applebbch81，`official_url` 仍是 `https://github.com/` 占位符，没有可核对的官方正文；受影响 accepted **1,746** 张（train 1,383 张）。
- **未决（1）**：agrifield40k_pool，见第 6 节。

### 5.1 `agrifm_ref9`：三项权利分开记录

| 权利 | 状态 | 依据 |
|---|---|---|
| 本地训练使用 | 允许（`RESTRICTED_RESEARCH_USE`，非商业学术） | CC BY-NC-4.0 官方端点 |
| 图像再分发 | 允许但需署名 + 非商业；项目策略是不分发 | CC BY-NC-4.0 |
| checkpoint 发布 | **未决**（`UNRESOLVED`） | 权重是否构成 NonCommercial 使用无法仅凭条款判定 |

84 张 accepted 全部来自 `agrifm_ref9`；训练可继续，发布权重前必须单独决策。

gate 描述、代码与报告现在使用同一套四级等级；`licence_evidence_backed` 不再声称"每个来源都有归档官方文件"。

## 6. agrifield40k_pool 上游许可覆盖

池级声明本身自相矛盾，已如实记录：Stage 5D registry 写 `CC-BY-4.0`，AgriField-40K 论文 Table 1 与正文写 `CC-BY-SA-4.0`。无论哪一个，都只是**编译池**的标签。

| 上游 | 池声明许可 | 等级 | accepted | train |
|---|---|---|---:|---:|
| `ACRECropWeed` | CC-BY-4.0 | 继承既有 pin 住证据 | 790 | 790 |
| `GrassClover` | CC-BY-SA | 未决 | 434 | 375 |
| `LUCASVision` | CC-BY-4.0 | 继承既有 pin 住证据 | 11,177 | 9,459 |
| `Maize-Weed` | CC-BY-4.0 | 未决 | 255 | 220 |
| `RadishWheat` | CC-BY-4.0 | 继承既有 pin 住证据 | 534 | 448 |
| `RumexLeaves` | CC-BY-4.0 | 未决 | 809 | 683 |
| `SesameWeed` | CC0 | 未决 | 943 | 782 |
| `VCD` | CC-BY-4.0 | 未决 | 2,159 | 1,802 |
| `VegAnn` | CC-BY-1.0 | 继承既有 pin 住证据 | 1,483 | 1,264 |
| `WE3DS` | CC-BY-4.0 | 未决 | 1,520 | 1,312 |
| `iNatWeeds` | per-image CC BY / CC0 | 继承既有 pin 住证据 | 1,312 | 1,105 |

- **有既有依据的上游（5）**：LUCASVision、VegAnn、ACRECropWeed、RadishWheat（pin 住的编译卡 + 计数吻合）、iNatWeeds（逐图 CC BY/CC0 元数据：`{'cc-by': 979, 'cc0': 336}`）。合计 accepted **15,296**，train **13,066**。
- **未决上游（6）**：`GrassClover`, `Maize-Weed`, `RumexLeaves`, `SesameWeed`, `VCD`, `WE3DS`。`data/source_registry.json` 明确把它们标为 `pending_audit`；本轮没有取得任何上游官方 LICENSE/数据卡正文（huggingface.co 超时、zenodo 拒绝、搜索引擎返回无关结果）。受影响 accepted **6,120**，train **5,174**。
- 未把池级 CC-BY 标签当作第三方许可证据，也没有据此宣布这些上游已核实。

## 7. 训练数量口径与 capped 采样口径

### 7.1 三个计数必须分开

| 口径 | 数值 |
|---|---:|
| corpus_accepted_count（train+val） | 169,324 |
| pretraining_train_count（实际参与 optimizer updates） | 147,277 |
| pretraining_validation_count（只验证） | 22,047 |
| Agri-147K 参照 | 147,285 |
| accepted − 参照 | +22,039 |
| **train − 参照** | **-8** |

结论：**若论文声称"实际预训练使用的图像数超过 Agri-147K"，当前不成立**（147,277 < 147,285，少 8 张）。accepted 超出的 +22,039 不能替代 train 口径。val 没有被偷偷并入训练。

证据等级可支撑的计数更小：accepted **161,458**，train **140,720**（扣除 pool 6 个未决上游与 applebbch81 的 REGISTRY_URL_ONLY）。

### 7.2 正式 cap 单位

- 正式口径：**`physical_source_id`**（`cap_unit = physical_source_id`，`official_profile = by_source`）。
- 理由：许可以及采集来源挂在 physical source 上；`artemis_imagesafari` 是 Artemis 与 Image Safari 合并后的**一个**物理来源，占 train 75.9%。若按 upstream 分开 cap，这个合并来源的合计仍会超过 50% 上限。
- capped 曝光预算：**38828.36** 未填满（epoch 150,000），**不回填**。
- upstream 层 capped 视图保留为 `analysis_only_not_the_formal_cap`，不得用来支持"预算已填满"的主张。

## 8. 状态与 Stage 6 允许的训练集合

**最终状态：CONDITIONAL**。

阻塞项（blocking gates 未决）：

1. `pool_upstream_licence_resolved`：6 个上游、accepted 6,120 / train 5,174 张只有池级声明，无逐上游官方文件。
2. `licence_evidence_backed`：四级等级已落地，但 applebbch81（accepted 1,746 / train 1,383）只有 registry URL，未达到"归档官方文件"等级，因此该 gate 也未决。

**Stage 6 允许的训练集合**：R3v3_license_verified **train split，147,277 张**（SHA-256 `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08`）。
- 这是"成员允许"的集合，不等于"证据完全可公开辩护"的集合。
- 证据-backed 的保守子集为 140,720 张。
- **不允许**把 val 并入训练；**不允许**声称训练集规模超过 147,285。
- 解除 CONDITIONAL 的最小路径：为 VCD、WE3DS、SesameWeed、RumexLeaves、Maize-Weed、GrassClover 取得并归档上游官方许可文件；或新建语料版本排除这 5,174 张 train 图像。后者会改变成员资格，必须新建版本，不能原地改 R3v3。

## 9. tracked / ignored / local-only 产物

判定来自 `git ls-files` 与 `git check-ignore -v`，逐文件记录于 `reports/stage5e1_artifacts.json`。

**Git tracked（3 个 json）**：
- `data/corpus_v4/R3v3_license_verified_manifest.json`
- `data/corpus_v4/R3v3_license_verified_sampling_profiles.json`
- `data/corpus_v4/R3v3_license_verified_composition.json`

**local-only（.gitignore 排除，不在 Git）**：
- `data/corpus_v4/R3v3_license_verified_train.csv` — 147,277 数据行，101,699,231 字节，SHA-256 `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08`
- `data/corpus_v4/R3v3_license_verified_val.csv` — 22,047 数据行，15,630,460 字节，SHA-256 `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686`
- `data/corpus_v4/R3v3_license_verified_quarantine.csv` — 39,485 数据行，5,687,800 字节，SHA-256 `7603653746f15fb8df744ce7bb1b619b81203fc74ff9e42d473985a935b5b9df`
- `data/corpus_v4/R3v3_license_verified_exclusions.csv` — 74,370 数据行，17,396,219 字节，SHA-256 `da19005ce04081012e93e55c7fc5171a3908bb46094238f420a9a450df2411d9`
- `data/corpus_v4/holdout_file_index.csv` — 22,861 数据行，6,507,078 字节，SHA-256 `abe5748eb6e918ab8e5a9bbe24e025bf5163b8099ee18891256ecdec7845767f`
- `data/corpus_v4/holdout_crossmatch.csv` — 17 数据行，5,446 字节，SHA-256 `6a1fc06c52a184d8d972020a35a3d727808f0a8e057298abe11ba87532e083b4`

其中 `holdout_file_index.csv` 与 `holdout_crossmatch.csv` **是 local-only**。Stage 5E 报告把它们列进"入 Git 的产物"是错的，本轮已更正。

## 10. 测试

`**709 passed / 7 skipped / 0 failed**`。

```bash
.venv-uv/bin/python -m pytest tests/ -q
```

skipped 不计为 pass；每个 skip 的真实原因见测试输出（历史环境变量依赖，与 Stage 5E.1 无关）。

## 11. 本阶段未做的事

- 没有训练 JEPA/Adapter，没有 optimizer update，没有新下游精度实验。
- 没有重做 Stage 5E 的语料构建逻辑，没有覆盖 Stage 5C/5D 文件。
- 没有删除任何 holdout 原图；确认副本只从 train/val 清单排除。
- 没有为通过门禁编造下载、hash、许可或门禁结果。
- 没有改变 R3v3 成员：train/val SHA-256 与 Stage 5E 记录一致。

## 12. 待办（进入 Stage 6 前）

1. 取得并归档 6 个未决上游的官方许可正文，或决定新建语料版本排除其 5,174 张 train 图像。
2. 把 applebbch81 的 `official_url` 固定到具体 Kaggle 记录并归档正文。
3. 若论文需要"超过 Agri-147K"的表述，先增加 train split 实体，而不是把 val 并入。
4. `agrifm_ref9` 的 checkpoint 发布权仍未决（84 张，CC BY-NC）。
