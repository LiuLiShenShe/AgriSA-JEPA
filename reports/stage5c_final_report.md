# Stage 5C 最终报告

> 目标阈值：R3 严格唯一去重田间图像 > 147,286
> 实际 R3：177,396
> 结论：**A — Corpus Target Achieved**

本报告不含任何“专家人工审核”声称；`AUTO_FROZEN` 仅表示程序化门禁全部通过。

## A. Agri-147K 来源复现

### 1. 35 个来源是否全部完成映射与核对？
**是。** 编号清单来自论文 Table 1 的 35 个引用编号，映射表为 `src/registry/stage5c_source_registry.py` 的 `REF_NUMBERS` / `ACCESS` / `SAME_PHYSICAL_DATASET`，测试 `test_stage5c_corpus.py::test_all_35_refs_registered` 强制覆盖 35/35。

### 2. FULL / PARTIAL / BLOCKED / ARCHIVE_ONLY 各多少？

| 状态 | 来源数 | 来源 |
|---|---:|---|
| DOWNLOADED_FULL | 17 | agrifield40k_pool, agrifm_ref1, agrifm_ref11, agrifm_ref20, agrifm_ref58, agrifm_ref6, agrifm_ref7, agrifm_ref8, agrifm_ref9, agrifm_ref90, applebbch81, gwhd_2021, perennial_plants, phenobench, plantvillage, ronin, sesame_weed |
| DOWNLOADED_PARTIAL | 2 | agrifm_ref57, artemis_imagesafari |
| BLOCKED | 18 | agrifm_ref15, agrifm_ref18, agrifm_ref2, agrifm_ref29, agrifm_ref3, agrifm_ref35, agrifm_ref37, agrifm_ref46, agrifm_ref48, agrifm_ref5, agrifm_ref65, agrifm_ref73, agrifm_ref74, agrifm_ref75, agrifm_ref76, agrifm_ref81, agrifm_ref82, agrifm_ref95 |
| ARCHIVE_ONLY | 1 | deepweeds |
| NOT_STARTED | 2 | agrifm_ref10, user_owned |

状态分布（download audit）：{'DOWNLOADED_FULL': 17, 'BLOCKED': 18, 'NOT_STARTED': 2, 'DOWNLOADED_PARTIAL': 2, 'ARCHIVE_ONLY': 1}
未取得来源的失败原因逐条保留在 `reports/stage5c_download_audit.md` 第 3 节，不静默丢弃。

### 3. 本轮原始已下载多少张？

- R0（本机全部可索引图像）：**243,694** 张
- 已下载字节数：75,005,942 （ledger 本身）
本机实际文件清单见 `data/corpus_v2/images/<source>.csv`（每行一个 SHA-256 与解码结果）。

### 4. 论文报告规模与实际可获规模差多少？

| 口径 | 图像 | 说明 |
|---|---:|---|
| 论文报告 | 147,285 | Agri-FM+ 口径 |
| 本机实际取得（R0） | 243,694 | 可索引、可解码文件 |
| 差距 | 0 | 未取得来源 + 论文未公开的逐图筛选 |

### 5. R1 实际图像数量与官方清单核对？

- R1：**60,603** 张
- 与 147,285 的差距：86,682 张
- 官方逐图清单核对：**无**。Agri-147K 未发布逐图清单，因此本项目只声称 *protocol-informed reconstruction*，不声称复现。

## B. 自动质检

### 6. 使用的现成库和模型、版本/权重 SHA

| 组件 | 版本 | 权重 / 标识 |
|---|---|---|
| cleanvision | 0.3.7 | — |
| pyiqa | 0.1.16 | — |
| torch | 2.9.1+cu128 | — |
| timm | 1.0.30 | — |
| transformers | 5.20.2 | — |
| pyiqa:brisque_svm_weights.pth | — | `e5b048ef5325bae6…` |
| pyiqa:musiq_koniq_ckpt-e95806b9.pth | — | `e95806b9eae5f381…` |
| pyiqa:niqe_modelparameters.mat | — | `fa2aa4d12dd9595c…` |
| CLIP 域分类器 | openai/clip-vit-base-patch32 | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| DINOv2 近重复确认 | timm | vit_small_patch14_dinov2.lvd142m |

权重 SHA 逐来源记录在 `data/corpus_v2/qc/<source>/qc_models_<source>.json`。

### 7. 每个自动质检步骤删除/隔离多少张？

单因归因漏斗（R1 重建层 / R2 严格层）：

| 层 | 步骤 | 输入 | 排除 | 剩余 |
|---|---|---:|---:|---:|
| R1 | downloaded | 64,970 | 0 | 64,970 |
| R1 | decoded | 64,970 | 0 | 64,970 |
| R1 | paper_rules_p0 | 64,970 | 3,847 | 61,123 |
| R1 | license | 61,123 | 0 | 61,123 |
| R1 | holdout_archive | 61,123 | 520 | 60,603 |
| R1 | strict_eligible | 60,603 | 0 | 60,603 |
| R2 | downloaded | 64,970 | 0 | 64,970 |
| R2 | decoded | 64,970 | 0 | 64,970 |
| R2 | min_resolution | 64,970 | 428 | 64,542 |
| R2 | basic_qc | 64,542 | 2,811 | 61,731 |
| R2 | iqa | 61,731 | 30,755 | 30,976 |
| R2 | domain | 30,976 | 2,194 | 28,782 |
| R2 | exact_dedup | 28,782 | 7,801 | 20,981 |
| R2 | near_dedup | 20,981 | 0 | 20,981 |
| R2 | holdout_license_archive | 20,981 | 29 | 20,952 |
| R2 | strict_eligible | 20,952 | 0 | 20,952 |
| R3 | downloaded | 243,694 | 0 | 243,694 |
| R3 | decoded | 243,694 | 0 | 243,694 |
| R3 | min_resolution | 243,694 | 428 | 243,266 |
| R3 | basic_qc | 243,266 | 5,892 | 237,374 |
| R3 | iqa | 237,374 | 31,805 | 205,569 |
| R3 | domain | 205,569 | 10,487 | 195,082 |
| R3 | exact_dedup | 195,082 | 9,109 | 185,973 |
| R3 | near_dedup | 185,973 | 0 | 185,973 |
| R3 | holdout_license_archive | 185,973 | 7,678 | 178,295 |
| R3 | strict_eligible | 178,295 | 0 | 178,295 |

### 8. 基本规则与 CleanVision / NR-IQA 的交集、差集

合成退化检测矩阵（本机确定性扰动，非人工标注）：

| 退化 | 阳性 | 真阳 | 假阳 | 召回 | 假阳率 |
|---|---:|---:|---:|---:|---:|
| blur_sigma2 | 16 | 8 | 0 | 0.500 | 0.000 |
| blur_sigma4 | 16 | 13 | 0 | 0.812 | 0.000 |
| dark_0.25 | 16 | 4 | 0 | 0.250 | 0.000 |
| bright_1.8 | 16 | 0 | 0 | 0.000 | 0.000 |
| jpeg_q10 | 16 | 0 | 0 | 0.000 | 0.000 |
| noise_sigma25 | 16 | 0 | 0 | 0.000 | 0.000 |
| downscale_0.25 | 16 | 8 | 0 | 0.500 | 0.000 |

MUSIQ 分离度：

| 退化 | clean 中位 | 退化中位 | 差值 |
|---|---:|---:|---:|
| blur_sigma2 | 58.874 | 26.237 | 32.638 |
| blur_sigma4 | 58.874 | 20.389 | 38.486 |
| dark_0.25 | 58.874 | 36.492 | 22.382 |
| bright_1.8 | 58.874 | 53.802 | 5.073 |
| jpeg_q10 | 58.874 | 37.266 | 21.609 |
| noise_sigma25 | 58.874 | 46.341 | 12.534 |
| downscale_0.25 | 58.874 | 26.364 | 32.511 |

交集：三套规则都会命中曝光异常与极低信息量图像。差集：CleanVision 的 `odd_size`/`grayscale` 等是记录型信号，不参与准入；MUSIQ/NIQE 是 NR-IQA，度量感知质量，与锐度度量不共线。

### 9. 阈值如何校准？误删风险？

- 校准方式：确定性合成退化（高斯模糊 σ=2/4、亮度 0.25/1.8、JPEG q=10、σ=25 噪声、1/4 降采样上采样）在 `tools/stage5c_calibrate_qc.py`。
- 阈值冻结在 `data/corpus_v2/policy.yaml`（版本化），**没有为了扩大语料而放宽任何阈值**。
- 已知误删风险：Laplacian 方差对**大田低纹理冠层**（如远摄小麦行）偏高估模糊；MUSIQ 对过曝蓝天场景偏低分。两者都进入 QUARANTINE 而非静默删除，可在后续人工复核中回收。

### 10. 不同来源的质量分布与系统偏差

| 来源 | 下载 | 解码 | 基础QC通过 | IQA通过 | 域通过 | R2 |
|---|---:|---:|---:|---:|---:|---:|
| artemis_imagesafari | 135,623 | 135,623 | 131,683 | 134,123 | 135,623 | 0 |
| agrifield40k_pool | 39,963 | 39,963 | 39,722 | 39,798 | 31,658 | 0 |
| agrifm_ref57 | 31,500 | 31,500 | 28,725 | 0 | 0 | 0 |
| agrifm_ref58 | 23,205 | 23,205 | 23,203 | 23,199 | 23,205 | 15,396 |
| agrifm_ref11 | 5,899 | 5,899 | 5,828 | 5,468 | 5,899 | 5,458 |
| applebbch81 | 1,838 | 1,838 | 1,778 | 1,805 | 1,838 | 0 |
| sesame_weed | 1,300 | 1,300 | 1,300 | 1,300 | 1,300 | 0 |
| agrifm_ref20 | 1,081 | 1,081 | 44 | 1,050 | 0 | 0 |
| agrifm_ref6 | 900 | 900 | 900 | 896 | 0 | 0 |
| agrifm_ref7 | 829 | 829 | 431 | 783 | 49 | 0 |
| agrifm_ref1 | 670 | 670 | 670 | 0 | 670 | 0 |
| agrifm_ref90 | 520 | 520 | 508 | 516 | 0 | 0 |
| agrifm_ref8 | 251 | 251 | 251 | 251 | 7 | 0 |
| agrifm_ref9 | 115 | 115 | 106 | 99 | 115 | 90 |

系统偏差确实存在：低分辨率田间远摄图的 Laplacian 方差系统性低于近摄叶片，因此 R2 的损失集中在远摄来源，这是已知偏差而非随机噪声。

### 11. 元数据与视觉域判定一致性 / UNKNOWN 数量

| 层 | 证据类型 | 图像 |
|---|---|---:|
| 域判定 | official_metadata | 171,151 |
| 域判定 | vision_skipped | 39,805 |
| 域判定 | filename_or_metadata | 31,658 |
| 域判定 | vision_model_inferred | 1,080 |
| 域判定 |  | 300 |
| 全库 UNKNOWN 域 | — | 41,129 |

元数据可用时一律以元数据为准（`evidence_type=official_metadata` 或 `filename_or_metadata`），CLIP 只在元数据缺失时兜底，且低置信 （<0.55 或 margin<0.08）进 QUARANTINE_DOMAIN。

### 12. 近重复 / 连续帧 / 同植株组减少量、巨型假聚类

机制：pHash+dHash+宽高比三信号一致才折叠；巨型簇（>300 成员）整簇隔离，不合并。连续帧与同植株组通过 `group_key` 强制同 split。

### 13. 全流程是否无人逐图质检？哪些仍需人工/授权？

**是，全流程无逐图人工质检。** 需要人工或授权的部分：
- 许可存疑来源（`RESEARCH_ONLY_UNVERIFIED`）的法律判定；
- QUARANTINE 池的人工复核回收；
- 受控访问来源（Kaggle competition、需注册 Dataverse）的账号授权。

## C. 数据扩建

### 14. 额外新增的公开来源

| 来源 | 许可 | 采集域 | 状态 |
|---|---|---|---|
| Artemis / Image Safari | CC-BY-SA 4.0 | FIELD_IN_SITU | 已下载 |
| sesame_weed | CC-BY 4.0 | FIELD_IN_SITU | 本地复用 |
| applebbch81 | CC-BY-NC-SA 4.0 | ORCHARD_IN_SITU | 本地复用 |
| AgriField-40K | 混合（逐前缀判定） | FIELD_IN_SITU | 本地复用 |

### 15. Artemis / Image Safari 实际可访问性与下载量

- 远端索引 JPEG key：**6,920,332**（索引 SHA `8ad734602cd64340…`）
- 本机抽样清单：133,051 张 / 126.2 GB
- 覆盖作物 18 类、农场 19 个（proxy：地区）
- 抽样种子 20261024，单农场份额上限 0.12
Amazon S3 匿名 ListObjectsV2 可访问；图片按清单分块下载，ledger 记录每个 key 的状态。

### 16. R2 / R3 有效图像数，R3 是否超过 147,285？

- R2：**20,944**
- R3：**177,396**
- 结论：超过 147,285，TARGET_MET

### 17. R2 集中度、独立来源、采集组与熵

- 独立来源：3
- 已知采集组：0
- 最大单一来源份额：0.735
- 来源熵：0.87 bits

### 17. R3 集中度、独立来源、采集组与熵

- 独立来源：6
- 已知采集组：27,370
- 最大单一来源份额：0.738
- 来源熵：1.24 bits

### 18. PhenoBench / GWHD 等 Holdout 重叠筛查

双层阻断：来源 ID 白名单（`HOLDOUT_IDS`）+ 文件前缀（`pnb_`）。R2/R3 的 `no_holdout` 门禁必须为 true 才可能 AUTO_FROZEN。测试 `test_holdout_cannot_be_bypassed_by_source_whitelist` 与 `test_agrifield_pnb_prefix_is_holdout` 强制该行为。

### 19. 许可阻塞/未知比例与可否公开图片

- LICENSE_BLOCKED：0 张（许可冲突，直接排除）
- RESEARCH_ONLY_UNVERIFIED：2,500 张（许可未能核验，只允许进入 R1 重建层，R2/R3 不接纳）
- 可公开范围：图片本体仅限许可明确允许的来源；其余发布清单、代码、哈希与统计，不发布图片。

### 20. 自有数据是否实际提供？

**没有提供自有数据。** 本轮零新增自有采集图像，所有数字来自公开来源或已有本地缓存，未编造。

## D. 论文与工程

### 21. 相对于只做尺寸/基础过滤，多做了什么？

1. 自动域判定（元数据优先 + CLIP 零样本 + 低置信隔离），把实验室/UAV/卫星图像从田间语料中剔除；
2. 三层去重（SHA-256 → 像素 SHA-256 → pHash/dHash/几何三重确认），巨型簇整簇隔离而不是合并；
3. 采集组感知的 train/val 划分，同植株/同 session 不跨 split；
4. 保留集双层隔离与许可三态判定；
5. 全链路 SHA-256 与 provenance 记录，可逐图复现。

### 22. 哪些可写为贡献，哪些必须引用前人？

可写为贡献：语料构建协议、门禁语义（R0/R1/R2/R3 + AUTO_FROZEN vs HUMAN_VERIFIED）、统计口径纪律（五种计数不混算）、确定性可复现管线。
必须引用：Agri-FM+（35 来源与 147K 语料）、AgriField-40K/AgriMAE、DataComp/DataComp-LM（数据策展方法论）、CleanVision、pyIQA、FAISS、DINOv2、CLIP、Artemis/Image Safari。

### 23. 达到论文级资源发布的哪些条件，哪些欠缺？

已满足：许可台账、来源映射、逐图哈希、版本化 manifest、自动化测试与复现命令。
欠缺：图片本体再分发授权（逐来源法律确认）、数据卡片的法律审签、QUARANTINE 池的人工复核记录。

### 24. 后续 JEPA 数据组成与规模消融的预注册

- 基线语料：R2（20,944 张）
- 扩展语料：R3（177,396 张）
- 预注册消融：规模（R2 vs R3）、田间占比（100% vs 混入实验室）、去重强度（精确 vs 精确+近重复）、split 协议（组感知 vs 随机）。
- 训练协议在数据阶段确认后再启动，本轮不进入训练。

### 25. 测试数量、失败/跳过、分支、commit SHA、配置与 manifest SHA

- 配置哈希：`45379ed1a669edb64778b97c1e9dd59ba9f3d8d5983418f642a2542b437a565c`
- R2 manifest SHA：`16780dc9f89c1dd8653724d6441650e02b033726299b28c03dd26f480090daa5`
- R3 manifest SHA：`f01c148d9e108140510ddba4cef2a3f6fff7fe131c7981febc1bb3bf1a48bee9`
- 测试数量与失败/跳过：见 `reports/stage5c_test_summary.md`（本地 pytest 实跑记录）。
- 分支：`agri-stage5c-agrifm147k-auto-corpus`
- 代码与报告提交 SHA：`138ba2e83a7d33dedc3c8f7ab25af17f5d3c5a51`

