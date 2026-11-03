# Stage 5D 最终验收报告 — AgriField-Extended (R3v2_strict_rgb)

交付分支：`agri-stage5d-corpus-validation`
代码提交：`9e9bcfa`（已推送到 `origin`）
报告口径：所有数字由本机文件与程序输出产生；每条结论后附可复核路径。

---

## A. 复现与计数

### 1. Git 分支和真实 HEAD？旧 SHA 是否保持？

- 新分支 `agri-stage5d-corpus-validation`，HEAD = `9e9bcfa0749e6256cba0af67cf3d07675d1c1250`
- 上一阶段 Stage 5C HEAD `2867e7f` 仍是本分支的父提交，`git diff 2867e7f HEAD --name-only -- data/corpus_v2 src/registry/stage5c_* tools/stage5c_* reports/stage5c_*` 为空，即 **R0/R1/R2/R3 的旧 manifest、policy、ledger、报告一个字节都没动**。
- `data/corpus_v2/policy.yaml` SHA-256 `45379ed1a669edb64778b97c1e9dd59ba9f3d8d5983418f642a2542b437a565c`，与两个 Stage 5C manifest 内记录的 SHA 完全一致。
- 复核方式：`reports/stage5d_final_report.md` 的 `legacy_unchanged` 门禁 = True；`reports/stage5d_gates.json`。

### 2. R0/R1/R2/R3 的历史统计是否保持？

保持，按 `data/corpus_v2/R{0,1,2,3}_*_manifest.json` 原样继承（本轮不重新计算、不重新解释）：

| 版本 | 图像 | 来源 | 状态 |
|---|---:|---:|---|
| R0 | 243,694 | 14 | AUDIT_ONLY |
| R1 | 60,603 | 9 | PROTOCOL_INFORMED_RECONSTRUCTION |
| R2 | 20,944 | 3 | AUTO_FROZEN |
| R3 | 177,396 | 6 | AUTO_FROZEN |

R3 train/val 150,613 / 26,783；Artemis/Image Safari 130,961（73.8%）；R3 全 unknown 作物；100,387 张无可信组；R2 全部 20,944 张无真实组。Stage 5C 测试 547 passed / 7 skipped / 0 failed（本轮全量跑得 596 passed / 7 skipped，含 Stage 5D 新增用例，见问题 27）。

### 3. 旧 R2 的 8 张、旧 R3 的 899 张到底因什么条件漏算？是否逐图对账？

逐图对账完成，身份与原因如下（证据：`reports/stage5d_legacy_funnel_reconciliation.json/md`，逐图清单 `reports/stage5d_funnel_mismatches.csv`）：

- 用 Stage 5C 的原始 loader（含 mtime 选文件的规则）重跑 Stage 5C 的 `funnel()`，得到 R2 存活 20,952 / R3 存活 178,295，而 manifest 是 20,944 / 177,396。
- 差值全部是 `qc_status = QUARANTINE` 的图像：漏斗的 `basic_qc` 步只排 `AUTO_REJECT`，而最终筛选要求 `selected_for_R{2,3}`（即 `AUTO_ACCEPT`）。
- R2 的 8 张全部来自 `agrifm_ref11`；R3 的 899 张来源分布：artemis_imagesafari 806、applebbch81 52、agrifield40k_pool 33、agrifm_ref11 8。
- 其中 30 张同时违反论文公开的 P0 模糊规则（`BLURRY_APPROXIMATION`），这一步漏斗从未评估。
- 结论：**不是集合运算算错，而是漏斗与最终筛选的谓词不一致**；Stage 5D 用单一 `decide_frame()`（18 个互斥主原因 + 可重叠副原因）派生全部输出，`funnel_table.csv` 的 `remaining_images` 逐行相减等于 manifest 计数，守恒式 243,694 = 169,326 + 42,085 + 9,301 + 9,606 + 2,500 + 0 + 10,876。

### 4. 新 R3v2 总数、train、val、隔离数？是否仍超过 147,285？

`data/corpus_v3/R3v2_strict_rgb_manifest.json`：

| 项 | 数量 |
|---|---:|
| ledger 输入 | 243,694 |
| **accepted** | **169,326** |
| train（含 TRAIN_ONLY_GROUP_UNKNOWN） | 147,279 |
| val | 22,047 |
| quarantined | 42,085 |
| rejected | 9,301 |
| holdout | 9,606 |
| licence blocked | 2,500 |
| duplicate removed | 10,876 |

- **超过 147,285**：超出 22,041 张（169,326 − 147,285），`target_exceeded = true`。这是纯数量属性，与质量门禁分开记录。
- 数量比旧 R3 少 8,070 张，净减少来源：ref11 的 2,949 张 NIR 按模态剔除、SesameWeed 1,300 张与 AgriField pool 重复、ref57 24,075 张域不合格、PhenoBench 前缀 9,606 张被 holdout，同时把 Stage 5C 因 loader mtime bug 丢掉的 ref1 670 张与 ref57 31,500 张补测并重新参与筛选。

### 5. 新阈值有无实际改变？新旧准入名单差异及原因？

- 阈值数值没变，**执行路径变了**：Stage 5D 的 `policy.yaml`（`stage5d-policy-v2`，SHA `f89e1a45…`）是唯一 source-of-truth，`load_policy()` 会校验每个 metric 的方向、区间、accept≥reject（lower_is_better 时反向）以及缺失分数动作；代码里没有任何会覆盖 YAML 的 Python 默认值。
- 旧 YAML 的 `accept_min 0.35 / reject_max 0.22` 是 0–1 尺度，MUSIQ 实际 0–100，所以那两个字段从未生效（证据见问题 6）。
- 新政策新增 `near_threshold_band: 0.15`（冲突带内不静默放行）、`missing_score_action: QUARANTINE`、NIQE 明确 `DIAGNOSTIC_ONLY`。
- 名单差异的实际原因不是阈值，而是：① 补齐 ref1/ref57 的 IQA（旧版这两源 32,170 张被静默判 `IQA_MISSING`）；② 模态隔离 ref11 的 NIR；③ 修复 `phash`/`dhash` 列名不匹配后去重真正生效（旧版近重复阶段是空转）；④ holdout 按来源/前缀/上游数据集三重阻断。

---

## B. 自动 QC

### 6. MUSIQ 的原始数值尺度？实际用的阈值？Policy 和代码是否一致？

- 尺度：pyIQA 的 MUSIQ 输出 **0–100，越大越好**。官方实现见 `https://github.com/chaofengc/IQA-PyTorch`（README 的 "check if lower better or higher better" 一节说明每个 metric 自带方向；本机实测 243,694 张范围 **12.58 – 79.13**，中位数 69.38）。
- 实际生效阈值：**accept 45.0 / reject 25.0**。证据链：`reports/stage5d_threshold_trace.md` — 若按 YAML 的 0.35/0.22 执行，211,524/211,524 全通过（1.0000）；按 45/25 执行才是 209,288 通过 / 2,128 隔离 / 108 拒绝，与旧 manifest 数量对得上。
- Policy 与代码一致性：新 `policy.yaml` 的 `score_range: [0,100]`、`direction: higher_is_better`、`accept_threshold: 45.0`、`reject_threshold: 25.0`、`role: GATE`、`model_sha256: e95806b9…` 全部写进 manifest 的 `iqa_trace`，门禁 `policy_and_iqa_scale_verified` 会重新计算 policy 文件 SHA 并核对每一项，结果 True。

### 7. 旧报告的 NIQE 是否实际参与过滤？新版本采用什么规则？

- **旧版从未参与**。`src/registry/stage5c_freeze.py::iqa_accept()` 只读 `iqa_musiq`；NIQE 在 243,694 行里只有 **2,095 行非空（0.86%）**，且这 2,095 行全部来自 Stage 5C 最后根本没采用的来源。旧报告写 "MUSIQ + NIQE 双指标过滤" 是不实表述。
- 新规则：NIQE 在新 policy 里保留定义（`lower_is_better`，区间 [0,100]，accept 8 / reject 12），但 `role: DIAGNOSTIC_ONLY`，不进入准入判定；如果将来覆盖率足够，需要改 policy 并重新构建版本，而不是改代码。

### 8. 新合成退化实验的实际样本数、完整流水线指标、最明显缺陷？

- 规模：**210 张干净基图 × 16 种退化 = 3,360 个退化样本**，基图来自 7 个来源/域分层（Artemis、ImageSafari 合池、agrifield40k_pool、ref58、ref11、ref9、applebbch81），分层内按 `numpy.random.default_rng(20261025)` 确定性抽样。退化全部由 `src/registry/stage5d_qc_validation.py` 确定性生成（同 seed 同结果，有单测覆盖）。
- 完整流水线指标（`reports/stage5d_synthetic_qc.json`）：
  - clean 保留率 **1.000**，clean 误删率 **0.000**（210/210 通过）；
  - 完整流水线对构造退化的召回 **0.391**；
  - 仅基础规则 0.227；仅 IQA 0.332（说明两者互补，不能互相替代）。
- 最明显缺陷：**对传感器噪声、JPEG 重压缩、色偏、局部遮挡几乎不敏感** —— gaussian_noise σ15/σ30 召回 0.076/0.114，JPEG Q10/Q25 召回 0.095/0.067，white_balance 0.090，occlusion 0.067。强项是严重模糊（0.981）、欠曝（0.867）、低于最小分辨率（0.738）。
- 边界说明：这些召回只衡量对**构造退化**的灵敏度，不是自然图像的质量准确率，也不给出 precision/FPR（无人工真值）。本报告不宣称任何自然图像精度。

### 9. 不同来源误拒或漏筛风险如何体现？

分层召回（`per_stratum`）：

| 分层 | 召回 |
|---|---:|
| applebbch81（果园） | 0.583 |
| agrifield40k_pool（田间） | 0.525 |
| agrifm_ref9（果园） | 0.465 |
| agrifm_ref58（田间） | 0.427 |
| artemis_imagesafari（田间） | 0.419 |
| agrifm_ref11（田间） | 0.204 |
| agrifm_ref1（田间） | 0.115 |

- 果园小图（applebbch81、ref9）召回最高；ref1 是麦穗特写，纹理单一，Laplacian 方差高、MUSIQ 高，构造退化很难触发阈值 → 漏筛风险集中在**特写、低多样性纹理**来源。
- 每来源保留率与筛除率见 `reports/stage5d_source_balance.csv`、阈值邻域曲线见 `reports/stage5d_threshold_sensitivity.json`（accept 40→60 时通过量 211,061→190,254，没有断崖，说明结论不依赖单个阈值点）。

### 10. 哪些图像被隔离而非删除？原因和数量？

隔离（quarantine）42,085 张，全部保留逐图原因在 `data/corpus_v3/quarantine_reasons.csv`：

| 主原因 | 数量 |
|---|---:|
| DOMAIN_QUARANTINE（域判定不确定） | 32,362 |
| IQA_QUARANTINE（MUSIQ 落在 25–45 隔离带） | 5,060 |
| BASIC_QC_QUARANTINE（基础指标互相冲突） | 2,063 |
| DUPLICATE_UNCERTAIN（哈希候选但未确认） | 2,600 |

另有 rejected 9,301（硬性不达标：BASIC_QC_REJECT 5,783、MODALITY_NOT_VISIBLE_RGB 2,949、IQA_REJECT 485、DOMAIN_REJECT 44）、holdout 9,606、licence blocked 2,500、duplicate removed 10,876。所有图像都留在本机原始目录，隔离只意味着不进入本版训练集，没有删文件。

---

## C. 模态与元数据

### 11. agrifm_ref11 是否确认为可见光 RGB？证据？保留/隔离多少？

`agrifm_ref11` 是 **混合来源，必须拆开处理**（证据：`reports/stage5d_modality_audit.json/md` + 像素统计）：

- 该源 5,899 行里，`rgbnormal` 目录 2,950 张、`nirraw` 目录 2,949 张。目录名来自官方 Kaggle 数据集的采集通道命名（`https://www.kaggle.com/datasets/zzzzqi/maize-phenology-images-acquired-using-ndvi-camera`，注册表原文写有 "NDVI camera; not pure RGB"）。
- 对每源抽 40 张做像素级筛查：rgbnormal 组三通道相关、无死通道、无极端通道偏移；nirraw 组 20/20 判 `EXTREME_CHANNEL_OFFSET`（均值约 149.5 / 1.7 / 72.6，R 与 G 相差约 88 倍），是伪彩/近红外编码，不是可见 RGB。
- 处理：`nirraw` 2,949 张全部 `MODALITY_NOT_VISIBLE_RGB` 剔除；`rgbnormal` 走正常 QC 后 **保留 382 张**（其余因重复/质量/域被筛掉）。
- 净变化：旧 R3 的 5,458 → 新版 382，即 **减少 5,076 张**；JPEG 三通道绝不等于可见光 RGB 这条规则由 `src/registry/stage5d_modality.py` 强制，且有单测（`test_multispectral_ndvi_is_not_visible_rgb` 一类）。

### 12. Artemis/Image Safari 的作物标签恢复多少张、国家多少、真实组字段多少？

`data/corpus_v3/R3v2_strict_rgb_manifest.json → artemis_join`：

- 参考索引 `remote_index.csv` 6,920,332 行；候选 135,623 行；**匹配 135,623，未匹配 0，主键冲突 0，一名多行 0**。
- 其中 Artemis 键 56,333、Image Safari 键 79,288（两者子集在统计里可分开，没有被混成一个不可追溯的物理来源）。
- 恢复字段：crop/farm_id/session_id/capture_timestamp/original_dataset_id/group_key 对 130,268 张有效（99.999%）；country 与 plot_id 只对 Artemis 子集 55,774 张有效，Image Safari 的 key 不含国家/地块编码，保持 unknown。
- 全库覆盖率（分母 = 169,326 accepted）：crop/farm/session/timestamp 76.9%、country 32.9%、plot 32.9%、group_key 100%（含推断等级，见问题 14）。
- 作物分布（分母含 unknown 39,058）：common-bean 41,809（已知子集内 32.1%）、sorghum 27,031（20.8%）、potato 13,930、cowpea 8,271、pearl-millet 7,158、banana 4,367、rice 2,371、maize 750……。国家：UGA 18,598、COL 15,415、USA 15,393、TZA 6,368，unknown 113,552（66.6%）。

### 13. R3v2 的作物、来源、国家、农场、设备组成？UNKNOWN 比例？

- 来源（7 个，HHI 0.602，熵 0.770）：artemis_imagesafari 130,270（76.9%）、agrifield40k_pool 21,418（12.6%）、agrifm_ref58 15,386（9.1%）、applebbch81 1,746、agrifm_ref11 382、agrifm_ref9 84、agrifm_ref1 40。
- Artemis 内部拆到上游数据集（19 个，HHI 0.309）：ImageSafari 74,494、Artemis 55,774。
- 农场 19 个，最大 ciat-arusha 18,160、ciat-kawanda 15,518、cimmyt-kiboko 14,989，最小 csir-ghana 6。
- 设备/采集平台：本机没有可靠的逐图设备字段，**不编造**；域构成 FIELD_IN_SITU 167,496（98.9%）/ ORCHARD_IN_SITU 1,830（1.1%），传感器类型只在 ref11 上有证据（RGB 相机 + NDVI 相机两种通道，已拆分）。
- UNKNOWN 比例：crop 23.1%（39,058/169,326）、country 67.1%（113,552）、group_tier 仅 2 张（0.001%）。所有 `unknown` 都是显式字符串，不是空值，`reports/stage5d_metadata_coverage.md` 每行同时给 known_fraction（分母=全库）与 share_among_known（分母=已知子集）。

### 14. 100,387 个原先未知组图像解决多少、仍未知多少？

- 旧 R3 的 100,387 张无可信组图像，本轮通过官方 key join 解决 **130,266 张**（artemis_imagesafari 130,268 进入 OFFICIAL_GROUP）。
- agrifield40k_pool 的 21,418 张靠官方目录布局进入 `DOCUMENTED_PATH_GROUP`（23,164 张合计，含 applebbch81 1,746）。
- 剩下 15,892 张只有弱代理（`acw` 单次campaign、`inat` 无 observation id 等），按 Stage 2D 已修复的规则保留语义、只进 train；**仅 2 张仍为 UNKNOWN_GROUP**，标记 `TRAIN_ONLY_GROUP_UNKNOWN`。
- 结果：`unknown_group_images = 2`，`unknown_group_in_val = 0`。

### 15. 新 train/val 对已知组是否完全隔离？未知组是否继续被随机跨分配？

- 已知组完全隔离：强组 47,674 个，`cross_split_groups = 0`；val 22,047 张全部来自 OFFICIAL_GROUP / DOCUMENTED_PATH_GROUP。`group_disjoint_validation_available = true`。
- 未知组**不再随机跨分配**：UNKNOWN_GROUP 一律 `TRAIN_ONLY_GROUP_UNKNOWN`，WEAK_PROXY 也一律 train。图像级随机拆分只作为 `DIAGNOSTIC_RANDOM_VAL` 存在（23,603 张），manifest 与数据卡都明确标注它**不是** group-disjoint 验证集。
- 近重复交叉检查：15,806 个确认的近重复对里，跨 split 的有 375 对，但**两端都进入发布的 train/val 的是 0 对**（其余都是一端已被剔除），所以没有 split 泄漏。证据：`reports/stage5d_duplicate_audit.json` 的 `cross_split_pairs_both_accepted`。

### 16. 与下游保留集的精确、近重复、采集组交叉检验结果？

三层证据，绝不合并成一句 "NO_LEAKAGE"（`reports/stage5d_holdout_evidence.json/md`）：

- **来源级**：本机两个已注册 holdout 源（phenobench、gwhd_2021）在 accepted 来源列表里出现 0 次。
- **前缀/身份级**：混合 pool 里带 `pnb_` 前缀的 **9,606 张全部阻断**（train 7,703 + val 1,903）。Stage 5C 只看了 `train/pnb_`，val/ 的 1,903 张漏过，旧 R3 因此含 1,864 张 PhenoBench。
- **文件/像素级**：对本机 20,282 个 holdout 文件建了 SHA-256 + pHash 索引；新 master 精确哈希命中 **0**；pHash 邻域（距离 ≤ 6）候选 **11** 个，逐个查证：距离 0 的 2 个是 GWHD 与 VegAnn 的哈希碰撞（平均像素差 59.5、相关系数 0.11，**不是同一张图**），距离 6 的 9 个是不同植株/不同地块的近景叶片，均未自动删除，只记录在案。
- **采集组级**：`HOLDOUT_EXCLUDED` 的 9,606 张在组派生之前就先被阻断，所以不存在"同场景不同文件"绕过。
- **不可在本机核实的**（PlantDoc、MinneApple、PlantSeg、MulticlassWeeds、GrowliFlower、CropAndWeed、DeepWeeds、Roboflow 系列）只做了来源级筛查，报告里如实列出，没有冒充文件级结论。

---

## D. 数据资源与论文价值

### 17. 来源 Artemis 的新占比和内部子域分布？

- accepted 169,326 中 artemis_imagesafari 130,270 = **76.9%**（旧 R3 是 73.8%，因为其他源被剔得更多）。
- 内部子域：ImageSafari 74,494（43.98% of accepted）、Artemis 55,774（32.94%）。
- Artemis 侧国家/农场/作物分布见问题 12/13；季节/平台字段在本机没有可信证据，标 unknown，不猜。

### 18. 有无过量来源或特定国家/作物偏置？

有，必须披露：

- 来源 HHI 0.602（7 源），单源最高 76.9%；上游数据集 HHI 0.309（19 源）。
- 作物偏置：已知作物里 common-bean 32.1%、sorghum 20.8%，前两名占已知子集 52.9%；另有 23.1% 的图没有任何作物证据。
- 国家偏置：已知国家里 UGA 33.4%、COL 27.6%、USA 27.6%，全部来自 Artemis/Image Safari 的 CGIAR 站点；67.1% 的图没有国家证据。
- 小来源重复曝光风险：agrifm_ref1 只有 40 张、agrifm_ref9 84 张，自然采样下每个 epoch 只见 41 / 86 次，容易被大源淹没（`data/corpus_v3/R3v2_sampling_profiles.json` 有期望曝光数）。
- 本轮**不训练、不声称"熵更高=下游更好"**；偏置只作为采样配置的输入记录。

### 19. 训练采样自然/capped/balanced 配置与每来源期望曝光？

epoch 长度 150,000，cap_share 0.5（`data/corpus_v3/R3v2_sampling_profiles.json`）：

| 来源 | 实体数 | 占比 | natural | capped(≤50%) | balanced |
|---|---:|---:|---:|---:|---:|
| artemis_imagesafari | 111,762 | 75.88% | 113,827 | 75,000 | 21,429 |
| agrifield40k_pool | 18,242 | 12.39% | 18,579 | 18,579 | 21,429 |
| agrifm_ref58 | 15,386 | 10.45% | 15,670 | 15,670 | 21,429 |
| applebbch81 | 1,383 | 0.94% | 1,409 | 1,409 | 21,429 |
| agrifm_ref11 | 382 | 0.26% | 389 | 389 | 21,429 |
| agrifm_ref9 | 84 | 0.06% | 86 | 86 | 21,429 |
| agrifm_ref1 | 40 | 0.03% | 41 | 41 | 21,429 |

- 采样 HHI：natural 0.602 → capped 0.503 → balanced 0.143；上游数据集层 natural 0.309 → balanced 0.053。
- 三种 profile 都**只改曝光频率**，不改实体集合、不改 split、不把重复采样算成新增图像。
- 同样阈值敏感性曲线见 `reports/stage5d_figures/threshold_sensitivity.png`。

### 20. 与 Agri-147K 的区别、与 AgriField-40K 的复用关系？

- Agri-FM+（CVPR Workshops 2025）只公开了构建**原则**，没有公开逐图清单；本语料是"参照公开规则的协议式重建 + 有文档的扩展"，**不是 Agri-147K 的复现**，数据卡里明确禁止这样引用（`reports/stage5d_dataset_card.md` 第 1、2 节）。
- AgriField-40K：本机保存的是其策展子池，共 39,963 行，按 `data/all_samples.csv` 的 `sample_id → source` 映射恢复出 **17 个上游数据集身份**（LUCASVision 11,177、VCD 2,159、WE3DS 1,520、VegAnn 1,485、iNatWeeds 1,312、SesameWeed 943、RumexLeaves 809、ACRECropWeed 790、RadishWheat 534、GrassClover 434、Maize-Weed 255、PhenoBench 9,606 等）。其中 PhenoBench 9,606 张整批作为下游 holdout 阻断；独立来源 `sesame_weed` 的 1,300 张全部按精确重复剔除（同一批图已通过 AgriField pool 的 SesameWeed 子集进入，pool 内该上游身份入选 943 张）。
- 同一物理来源的多个论文镜像没有重复计数：漏斗按 `physical_source_id` 分组后再汇总，`per_source_conservation` 的合计等于全局合计（门禁 `counts_conserved` = True）。

### 21. 哪些算法来自前人、真正的工程/数据贡献？

- 前人算法（引用即可，不称原创）：CleanVision（基础规则）、pyIQA 的 MUSIQ/NIQE（`https://github.com/chaofengc/IQA-PyTorch`）、DINOv2（嵌入确认近重复）、pHash/dHash、FAISS、I-JEPA ViT-H/14（下一阶段用）。
- 本轮可主张的贡献：
  1. 把"筛选漏斗 ↔ 最终 manifest"做成**单一位图派生**，18 个互斥主原因 + 可重叠副原因，逐图可追溯，历史 8/899 差异被逐张解释；
  2. 证明并修正 IQA 尺度/阈值脱钩：YAML 0.35/0.22 是死配置，policy 变成唯一 source-of-truth 且带模型 SHA 与尺度证据；
  3. 用官方 key 做 135,623/135,623 零冲突的元数据 join，并把 Artemis/Image Safari 保持为可区分的子集；
  4. 混合池的下游 holdout 阻断从"只看 train/pnb_"扩展到来源 + 前缀 + 上游数据集三重，并给出文件级/pHash 级证据与不可核实项清单；
  5. 组感知拆分与 UNKNOWN_GROUP 的 train-only 约束，以及 `DIAGNOSTIC_RANDOM_VAL` 与 `GROUP_DISJOINT_VAL` 的显式区分；
  6. 210 × 16 的确定性合成退化压力测试，把 QC 的有效性边界（强在模糊/欠曝，弱在噪声/JPEG/色偏/遮挡）写成可复核数字。
- "筛选提升 JEPA 迁移性能"这一条**本轮不主张**，需下一阶段同模型同预算的下游实证。

### 22. 哪些数据可合规公开重分发，哪些只能发布 manifest + 重建脚本？

按 `data/corpus_v2/sources.csv` 的 `redistribution_allowed` 字段（注册表原值，未做法律解释）：

- 记录为可重分发：artemis_imagesafari（CC-BY-SA-4.0，需 share-alike）、agrifield40k_pool（CC-BY-4.0）、applebbch81（CC-BY-4.0）、agrifm_ref20（CC-BY-4.0）、agrifm_ref57（CC-BY-4.0）。
- 记录为 `unknown`：agrifm_ref1、agrifm_ref58、agrifm_ref11、agrifm_ref9（CC-BY-NC-4.0，非商业）。对这几类**只发布来源 id、哈希与重建脚本，不发布像素**。
- 记录为 `Unknown` 许可证：agrifm_ref6、ref7、ref8 共 1,980 张本轮直接 `LICENCE_BLOCKED` 隔离；agrifm_ref90 记录为 archive-only，520 张同样阻断。
- Git 里只有 manifest/policy/sources.csv/小报告；大型 ledger 与 CSV 按 `.gitignore` 排除，实际位置与重建命令见 `reports/stage5d_dataset_card.md` 第 9 节。

### 23. 目前的数据卡还保留哪些限制？

`reports/stage5d_dataset_card.md` 第 10 节原样保留：自动验证而非人工验证；country 只在 Artemis key 编码处恢复；crop 对小来源与混合 pool 为 unknown；不在本机的 holdout 只做了来源级筛查；QC 对噪声/JPEG/色偏/遮挡弱；来源集中度 76.9%。另外显式声明：`AUTO_FROZEN ≠ HUMAN_VERIFIED ≠ PROVEN_NO_LEAKAGE`。

---

## E. 交付、门禁和研究决策

### 24. 新 Manifest、Policy、CSV 的 SHA-256？

- `data/corpus_v3/policy.yaml`：`f89e1a4515c4f7353618c73da77bebf77d1b78e570266a4349177f7fd2bfbc97`
- `data/corpus_v3/R3v2_strict_rgb_manifest.json`：随代码提交入 Git（`git hash-object` 可得，内容见 manifest 内 `train_csv_sha256` / `val_csv_sha256`）
- train CSV：`f33b7cc24eee2d04e4f8b53d6b1026cfebe8fac5854971377216f3edb37abc30`
- val CSV：`a64a1222b27ddbdaa835a5c836a5330ceb1a71d1e18f0392f3f0b10a4d2e22d2`
- 旧 policy SHA 未变：`45379ed1a669edb64778b97c1e9dd59ba9f3d8d5983418f642a2542b437a565c`
- MUSIQ 权重 SHA-256：`e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa`；NIQE 权重 `fa2aa4d12dd9595c3fb6bf8b4e1a15c79d5d1469238d0901c88dead195be578d`

### 25. 新版本状态？哪条门禁不通过？

`reports/stage5d_final_report.md` + `reports/stage5d_gates.json`：

- **READY_AUTO_VALIDATED**，12/12 门禁通过，failed_gates 与 unknown_gates 均为空。
- 12 条：files_readable / policy_and_iqa_scale_verified / visible_rgb_evidenced / field_domain_evidenced / licence_clear / dedup_resolved / holdout_screened / groups_disjoint_and_unknown_flagged / counts_conserved / rebuild_byte_identical / composition_from_manifest / legacy_unchanged。
- 语义边界：只代表"程序化验证通过"，不代表人工核验，也不代表对本机之外的来源证明了无泄漏。

### 26. 是否 TARGET_EXCEEDED？与质量状态是否分开？

- `target_exceeded = true`（169,326 > 147,285，富余 22,041）。
- 在 `status()` 里 TARGET_EXCEEDED 是**正交属性**：`tools/stage5d_final_report.py` 单独输出 `target_exceeded_is_scale_property_only: true`，并且 `src/registry/stage5d_gates.py` 的实现保证它永远不会覆盖一个失败门禁。若本轮 QC 使数量掉到阈值以下，报告会写差额与阻塞来源，而不是降低要求凑数。

### 27. 新增工具与测试？完整 pytest 结果？

新增源码模块（`src/registry/`）：`stage5d_metadata_schema.py`、`stage5d_metadata.py`、`stage5d_modality.py`、`stage5d_qc_policy.py`、`stage5d_qc_validation.py`、`stage5d_funnel.py`、`stage5d_dedup.py`、`stage5d_group_policy.py`、`stage5d_gates.py`。

新增工具（`tools/`）：`stage5d_preflight.py`、`stage5d_audit_score_scales.py`、`stage5d_reconcile_funnel.py`、`stage5d_compute_missing_iqa.py`、`stage5d_build_corpus.py`、`stage5d_validate_modality.py`、`stage5d_synthetic_qc.py`、`stage5d_sensitivity.py`、`stage5d_holdout_audit.py`、`stage5d_publish_materials.py`、`stage5d_sampling_profiles.py`、`stage5d_generate_tables.py`、`stage5d_final_report.py`、`stage5d_rebuild_check.py`、`stage5d_loader_smoke.py`、`stage5d_run_all.sh`（16 步无交互复现）。

测试：新增 `tests/test_stage5d_corpus.py`（49 个用例，覆盖规格 25 项要求 + 合成退化确定性 + 下采样必拒 + 噪声可复现）。

- **全量 `pytest tests/`：596 passed / 7 skipped / 0 failed**（732 s）。
- 7 个 skipped 全部是历史遗留、原因明确：`tests/test_stage2_vith.py` 4 个（未设 `AGRISA_VITH_CHECKPOINT`）、`tests/test_stage3a_downstream.py` 3 个（未设 `AGRISA_VITH_CHECKPOINT` 2 个、未设 `AGRISA_PHENOBENCH` 1 个）。
- Stage 5C 既有用例无回归；PhenoBench/GWHD 数据读取接口无回归（有专门用例覆盖）。
- Loader smoke test：2 个 batch（16×3×224×224），`optimizer_updates: 0`，见 `reports/stage5d_loader_smoke.json`。

### 28. 旧版实验/权重/数据是否完整保留？

完整保留：

- `data/corpus_v2/`（policy、R0–R3 manifest、`images/*.csv` ledger、下载文件）一个字节未改，`git diff 2867e7f HEAD` 对这些路径为空。
- Stage 1–5C 的代码、报告、检查点、缓存全部在原位；本轮没有 `git reset`、没有 force push、没有合并 main。
- 本轮新产物全部落在 `data/corpus_v3/` 与 `reports/stage5d_*` 新命名空间；大型文件不入 Git（`.gitignore` 已更新），本机位置：`data/corpus_v3/images/*.csv`（332 MB，逐图 ledger）、`R3v2_strict_rgb_train.csv`（92 MB）、`R3v2_strict_rgb_val.csv`（15 MB）、`quarantine_reasons.csv`（11 MB）、`near_duplicate_decisions.csv`（7.9 MB）、`derived_iqa/`（21 MB）、`holdout_hash_index.csv`（3.7 MB）。
- 重建命令：`export AGRI_DATA_ROOT=/data/datasets/Stage5C && bash tools/stage5d_run_all.sh`（GPU 版约 60 分钟；`STAGE5D_SKIP_EMBEDDINGS=1` 可跳过 DINOv2 确认，但发布 manifest 是带嵌入的版本）。

### 29. 本轮有没有启动 JEPA optimizer updates？

**零。** 没有任何 JEPA 前向/反向、没有 optimizer step、没有 adapter 训练、没有下游刷分、没有新 Mask 算法、没有新的大规模下载。唯一的 GPU 使用是：MUSIQ 补测（`stage5d_compute_missing_iqa.py`）、DINOv2 嵌入确认近重复、合成退化实验里的 MUSIQ 打分。Loader smoke test 记录 `optimizer_updates: 0`。

### 30. 下一阶段是否允许正式开始 JEPA 的数据规模、来源组成和多任务迁移实验？

按本轮门禁结果给出决定：**允许进入下一阶段的数据准备与小规模读取实验，但正式长训练必须先补两件本机无法自动完成的事**：

1. 对 `redistribution_allowed = unknown` 的 4 个小来源（agrifm_ref1、ref58、ref11、ref9，合计 15,892 张，占 9.4%）逐个确认许可，否则从训练实体中剔除后再统计规模——本报告已按"只发 manifest + 重建脚本"处理，但训练用途本身仍需确认；
2. 下游 holdout 的文件级/pHash 级交叉只覆盖了本机有的 PhenoBench 与 GWHD 2021；PlantDoc、MinneApple、PlantSeg、DeepWeeds、GrowliFlower、CropAndWeed、MulticlassWeeds 与 Roboflow 系列只做了来源级筛查，需在真正用它们评测前补文件级索引。

在以上两条落地前，可以做：采样 profile 对比（natural / capped / balanced）、小步数 I-JEPA 持续预训练探针、PhenoBench/GWHD 的读取与评测链路验证。以上都不构成本轮宣称的"筛选提升迁移性能"结论。

---

## 结论

**B — Candidate with limited issues（主要管线可运行，但仍有可界定的许可与 QC 边界风险）。**

> 口径说明：manifest 的 `READY_AUTO_VALIDATED` 是 **12 条程序化门禁**的状态；本节 A/B/C/D 是更严格的科研验收结论，额外计入门禁没有编码成硬阈值的许可、QC 敏感性和元数据边界。两者不矛盾：门禁全过是事实，残留风险也真实存在。

依据：

- 统计、模态、元数据、组划分、holdout、重建确定性这 7 类核心问题都已用本机证据闭环，12 条程序化门禁全部通过，数量也超过 147,285（`READY_AUTO_VALIDATED`，`TARGET_EXCEEDED` 独立成立）；
- 但仍存在**可界定**的残留风险：① 4 个小来源许可状态 unknown（9.4% 的图像）；② 8 个下游 holdout 数据集无法在本机做文件级核验；③ 自动 QC 对噪声/JPEG/色偏/遮挡召回仅 0.07–0.11，这部分图像未被认证干净；④ 67.1% 的图无国家证据、23.1% 无作物证据；
- 这些风险都能定位到具体来源与具体图像，不需要重建版本，因此不是 C；数据、工具、算力齐备且门禁可测，因此不是 D；风险尚未清零，因此不选 A。
- 因此下一阶段只允许准备配置与小规模读取测试，不进入正式 JEPA 长训练，直到问题 30 列出的两项补齐。

---

### 附：主要证据文件索引

| 主题 | 文件 |
|---|---|
| 门禁与状态 | `reports/stage5d_final_report.md`、`reports/stage5d_gates.json` |
| 计数与 manifest | `data/corpus_v3/R3v2_strict_rgb_manifest.json` |
| 历史 8/899 对账 | `reports/stage5d_legacy_funnel_reconciliation.{json,md}`、`reports/stage5d_funnel_mismatches.csv` |
| 新漏斗逐行守恒 | `reports/stage5d_funnel_table.csv` |
| IQA 尺度 | `reports/stage5d_threshold_trace.{json,md}`、`reports/stage5d_qc_policy_diff.md` |
| 模态 | `reports/stage5d_modality_audit.{json,md}` |
| 合成 QC | `reports/stage5d_synthetic_qc.{csv,json}`、`reports/stage5d_qc_calibration.csv` |
| 阈值敏感性 | `reports/stage5d_threshold_sensitivity.json`、`reports/stage5d_source_balance.csv` |
| 去重 | `reports/stage5d_duplicate_audit.{json,md}`、`data/corpus_v3/near_duplicate_decisions.csv` |
| 组与拆分 | `reports/stage5d_groups_and_splits.{csv,json,md}` |
| holdout | `reports/stage5d_holdout_evidence.{json,md}`、`data/corpus_v3/holdout_exclusions.csv` |
| 组成与覆盖 | `reports/stage5d_dataset_composition.{csv,md}`、`reports/stage5d_metadata_coverage.{csv,md}` |
| 采样 profile | `data/corpus_v3/R3v2_sampling_profiles.json` |
| 数据卡 | `reports/stage5d_dataset_card.md` |
| 重建一致性 | `reports/stage5d_rebuild_check.json` |
| loader smoke | `reports/stage5d_loader_smoke.json` |
| 图 | `reports/stage5d_figures/*.png`（8 张） |

外部来源：pyIQA 官方仓库 `https://github.com/chaofengc/IQA-PyTorch`（MUSIQ/NIQE 实现与方向说明，本轮通过 open-websearch 抓取 README 核实）。

【任务已结束】
