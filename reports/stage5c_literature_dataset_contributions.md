# Stage 5C — 文献依据：数据构建是否构成科研贡献

> 核验日期：2026-10-24（本机联网实测抓取）
> 目的：把“已发表、可核实的事实”与“本项目提出的、尚待验证的主张”分开写。
> 本文件不参与任何数字统计；所有语料规模数字来自 `data/corpus_v2/` 的机器可读账本。

## 1. 已核实的文献事实

### 1.1 Agri-FM+（CVPR Workshops 2025）

- 论文：Md Jaber Al Nahian, Tapotosh Ghosh, Farnaz Sheikhi, Farhad Maleki,
  “Agri-FM+: A Self-Supervised Foundation Model for Agricultural Vision”,
  CVPRW 2025, pp. 5550–5562。
- 官方页面：https://openaccess.thecvf.com/content/CVPR2025W/V4A/html/Al_Nahian_Agri-FM_A_Self-Supervised_Foundation_Model_for_Agricultural_Vision_CVPRW_2025_paper.html
- 已核实表述（摘要原文）：Agri-FM+ 从 SlotCon 的 ImageNet 无监督权重出发，
  在 **a curated 147K-image agricultural dataset** 上持续预训练；
  在 **eight diverse benchmarks**（检测、语义分割、实例分割）上评估。
- 与本项目的关系：数据整理是该方法明确的组成部分；我们可以复现“来源台账 +
  场景/质量/许可筛选”的流程，并以其 147,285 作为外部参照规模。
- **不能照搬的表述**：不能说“本项目首次提出农业多来源数据整合”。
  Agri-FM+ 已经把多来源收集与筛选写进方法。

### 1.2 AgriField-40K / AgriMAE（ECCV Workshops 2026）

- 论文页面：https://dtu-pas.github.io/agrifield40k/
- 已核实表述（项目页原文）：**field-centric dataset curated from 17 public
  resources**，约 **40K field images**；AgriMAE 是参数高效的持续预训练基线，
  最多减少 **9×** 可训练参数。
- 与本项目的关系：证明“农业领域 field-centric 语料本身”可以作为论文贡献，
  并且 17 个公开来源的整合是可复现的先例。
- **不能照搬的表述**：不能说“农业领域此前没有 field-centric 语料”。

### 1.3 DataComp（NeurIPS 2023 Datasets & Benchmarks）

- 论文：Gadre et al., “DataComp: In search of the next generation of multimodal
  datasets”, NeurIPS 36 (2023), Datasets and Benchmarks Track。
- 官方页面：https://papers.neurips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html
- 已核实表述（摘要原文）：以 12.8B image-text pair 为候选池，参与者在**统一
  CLIP 训练代码 + 38 个下游测试集**下比较不同的过滤/数据源策略；不同计算预算
  下研究 scaling。
- 与本项目的关系：给“把数据来源与过滤当作受控变量、固定训练设置做消融”的
  方法学先例。我们的 P0/P1、DATA_SCALE、DOMAIN_MIX 臂直接借用这个设计。
- **不能照搬的表述**：不能声称“使用现成筛选器本身就是新算法”。

### 1.4 DataComp-LM（NeurIPS 2024 Datasets & Benchmarks）

- 论文：Li et al., “DataComp-LM: In search of the next generation of training
  sets for language models”, NeurIPS 37 (2024), Datasets and Benchmarks Track。
- 官方页面：https://proceedings.neurips.cc/paper_files/paper/2024/hash/19e4ea30dded58259665db375885e412-Abstract-Datasets_and_Benchmarks_Track.html
- 已核实表述（摘要原文）：240T token 的标准语料、OpenLM 训练配方、53 项下游
  评估；系统研究 **deduplication, filtering, data mixing**。
- 与本项目的关系：说明“数据治理”可以成为系统性研究对象；但这是**语言模型**
  证据，不能直接外推为农业图像的因果证据。
- **不能照搬的表述**：不能把跨模态的方法学类比写成农业图像领域的实验结论。

### 1.5 CleanVision（工具，不是论文贡献）

- 仓库：https://github.com/cleanlab/cleanvision
- 文档：https://cleanvision.readthedocs.io/en/latest/
- 本轮实测版本：`cleanvision 0.3.7`（见 `reports/stage5c_qc_method_and_calibration.md`）。
- 用途：模糊、过暗、过曝、低信息、异常尺寸/纵横比、灰度、精确/近重复等
  自动化风险信号。
- **不能照搬的表述**：调用 CleanVision 不是新算法贡献；必须引用其原始工作。

### 1.6 其他被复用的成熟工具

| 工具 | 版本（本轮实测） | 用途 | 引用边界 |
|---|---|---|---|
| pyIQA | 0.1.16 | MUSIQ / NIQE / BRISQUE 无参考质量评分 | 指标来自原论文，不是本项目发明 |
| imagededup | 0.3.3 | pHash/dHash 近重复初筛 | 仅作候选，需多证据确认 |
| FAISS | 1.15.1 | Hamming / 向量近邻 | 基础设施 |
| timm (DINOv2) | 1.0.30 | 视觉嵌入，用于近重复确认与分布分析 | 表征模型，非本项目训练 |
| CLIP (ViT-B/32) | transformers 5.20.2 | 零样本采集域判定 | 仅作证据之一，不等于采集地点事实证明 |

## 2. 本项目可以主张什么（候选贡献）

### 2.1 新数据资源（条件性）

一个**比 Agri-147K 更大、严格近地田间/果园优先、跨作物/地区/平台、可追溯
授权与采集组**的农业 RGB 预训练语料。该主张只有在：

1. `R3_FIELD_EXTENDED` 的 `strict_unique_eligible_images > 147,285`；
2. 去重、Holdout 隔离、许可与采集组统计完整；
3. 来源集中度与作物/地区覆盖被同时报告

时才成立。数字必须由逐图账本聚合，不能手写。

### 2.2 可复现的自动化策展流程（条件性）

多来源元数据 → 成熟图像质量评价 → 田间域证据 → 重复/采集组约束 → 下游隔离
的集成、校准与风险控制。其可发表性取决于：

- 过程透明（策略 YAML + 模型哈希 + 逐图原因码）；
- 同一输入两次构建结果一致（manifest SHA 相同）；
- 与朴素筛选（P0）做可量化比较；
- 对不确定样本的隔离而不是“强行通过”。

底层算法（CleanVision、pyIQA、imagededup、FAISS、CLIP、DINOv2）全部来自前人
工作，必须正确引用。

### 2.3 后续实证贡献（本阶段未发生）

在固定 I-JEPA ViT-H/14 初始化、相同 mask/更新数/随机种子协议下，比较
`DATA_RAW_BASELINE / DATA_P0 / DATA_P1 / DATA_P1_CAP / DATA_SCALE / DOMAIN_MIX`
对八个农业下游任务迁移的影响。**本轮不训练**，因此不能提前声称
“自动质检已证明优于人工”或“超过 147K 就一定更好”。

## 3. 明确禁止的表述

| 禁止表述 | 原因 |
|---|---|
| “首次提出农业多来源数据整合” | Agri-FM+ 与 AgriField-40K 已做 |
| “农业领域此前没有 field-centric 语料” | AgriField-40K 已发布 |
| “我们提出了新的图像质量算法” | 复用的是 CleanVision/pyIQA 等 |
| “自动质检已被证明优于人工” | 本轮没有该对照实验 |
| “超过 147K 就一定带来更好迁移” | 未训练，无因果证据 |
| “逐图复刻了官方 Agri-147K” | 官方逐图清单/阈值/人工决定未公开 |
| “全部下载完 35 个来源” | 只能报告真实 FULL/PARTIAL/BLOCKED |

## 4. 结论

数据构建在农业视觉中**已经是**被认可的科研贡献形态（Agri-FM+、AgriField-40K），
但“贡献成立”依赖可核实的规模、覆盖、审计与后续任务效果，而不是“把多个数据集
拼在一起”。本项目的差异化只能来自：更严格的田间优先策略、可复现的自动门禁、
明确的 Holdout/许可治理，以及后续固定协议的迁移实验。
