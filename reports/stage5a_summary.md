# Stage 5A 总结报告 — 大规模农业数据源整合与多任务下游评价框架

> 分支：`agri-stage5a`（自 Stage 4B 验证提交 `45a00b5fb62c25b74e0ae547fe91061cf1101c19`）
> 执行日期：2026-10-23
> GPU：仅 GPU 1（RTX A6000 48 GB），`CUDA_VISIBLE_DEVICES=1`
> 环境：`.venv-uv`，Python 3.13.12，PyTorch 2.9.1+cu128

本文件按第 15 节要求的 30 个验收问题逐条回答，数字全部来自本仓库中的可复现产物
（CSV / JSON / 日志 / 运行目录），不引用记忆中的数值。

| 类别 | 产物 |
|---|---|
| 数据源目录 | `data/registry/agricultural_sources.csv`、`reports/stage5a_data_source_catalog.md` |
| 来源重叠 | `data/registry/source_overlap_candidates.csv`、`reports/stage5a_source_overlap_audit.md` |
| 许可证 | `reports/stage5a_dataset_licence_audit.md` |
| 基准注册 | `data/registry/agricultural_benchmarks.csv`、`reports/stage5a_benchmark_registry.md` |
| 保留集 | `data/registry/downstream_holdout_sources.csv` |
| D1 候选语料 | `data/frozen/agri-corpus-d1-candidate_{train,val}.csv` + manifest |
| GWHD 数据审计 | `reports/stage5a_gwhd_data_audit.{md,json}` |
| GWHD 检测协议 | `reports/stage5a_gwhd_protocol.md` |
| GWHD 检测结果 | `reports/stage5a_gwhd_results.md`、`reports/stage5a_gwhd_comparison.{csv,json}`、`reports/stage5a_gwhd_ab_summary.md` |
| Adapter 设计 | `reports/stage5a_adapter_feasibility.md` |

---

## 数据（1–8）

**1. 实际核实了多少个独立原始农业数据来源？**

63 条登记行，覆盖 **59 个独立数据集**（4 组已知同源映射被合并，不重复计数）：

- AgriField-40K 的 17 个来源（`catalog_groups` 含 `agrifield40k`）；
- Agri-FM+ Table 1 的 35 个参考文献条目（`agrifm_ref*`），逐条按参考文献列表还原为真实数据集名；
- 12 个下游基准资源（含 PhenoBench、GWHD、PlantDoc、MinneApple、PlantSeg 等）；
- 1 个本项目新增来源（PlantVillage，归入 `project_d1`）。

同源合并关系：`perennial_plants ↔ agrifm_ref4`、`ronin ↔ agrifm_ref79`、
`phenobench ↔ agrifm_ref88`、`gwhd_2021 ↔ agrifm_ref31`。
63 行 − 4 组重复 = 59 个独立数据集；**行数不可直接相加**，这一点写在
`reports/stage5a_source_overlap_audit.md` 里。

**2. Agri-FM+ 的 35 个来源中多少个已经找到官方入口？**

35/35 全部完成“参考文献编号 → 真实数据集名 → 官方入口 URL”的映射，每行都记录了
论文、年份、论文 URL、下载 URL。URL 探测（`reports/stage5a_url_check.json`，
2026-10-23 实测，共 58 个 URL）：

| 结果 | 全部 58 个 | 35 个 Agri-FM+ 来源 |
|---|---:|---:|
| 可访问（200/206/202） | 40 | 27 |
| 被拦截（403） | 9 | 4 |
| 网络不可达（DNS/连接） | 6 | 3 |
| 404（无公开归档） | 3 | 1 |

被拦截与不可达的行**没有被升级状态**，仍停在 `URL_VERIFIED`，并在 catalog 中写明原因。

**3. 哪些数据实际上已经下载？**

`downloaded=yes` 共 9 行（全部在本机磁盘上，有解码证据）：

| source_id | 数据集 | 已解码图像 |
|---|---|---:|
| `agrifield40k-v1` 六个确认来源 | ACRE CropWeed / VegAnn / LUCASVision / SorghumWeed / RadishWheat / iNatWeeds | 791 / 1,607 / 11,195 / 172 / 534 / 1,315 |
| `phenobench` | PhenoBench v1.1.0 | 9,606 |
| `agrifm_ref93` | PlantVillage（Zenodo 1204914） | 54,305 |
| `gwhd_2021` | GWHD 2021（Zenodo 5092309） | 6,515 |

**4. 哪些数据由于许可证、重叠或质量问题不能纳入？**

- **整来源排除（下游保留集，12 个）**：PhenoBench、GWHD 2021、GWHD Instance Seg、
  PlantDoc、MinneApple、Roboflow Disease、Roboflow Crop、PlantSeg、
  Multiclass Weeds、DeepWeeds、GrowliFlower、CropAndWeed。这些可以用于**评价**，
  但绝不能进入任何预训练 manifest（`src/registry/freeze.py:HOLDOUT_SOURCE_IDS`）。
- **许可证冲突**：PlantVillage 的 Zenodo 记录同时声明 `cc-by-4.0`（metadata）与
  “CC BY-SA 3.0 US”（描述）。按**更严格**的 CC-BY-SA-3.0-US 记录，并保持
  `images_approved` 为空 —— 该来源尚未批准进语料。
- **入口不可达**：`agrifm_ref46` 无公开归档（404）；`agrifm_ref48`、`must_c`、
  `vegann` 等在本机网络下不可达；9 行被 403 拦截。这些行不升级状态。
- **质量未过门**：PlantVillage 仍有 12,436 个感知相似候选对未人工复核；
  v1 语料另有 784 个待复核重复候选与 14,743 张待审计图像。在复核完成前，
  这些都**不计入 approved**。

**5. 有没有真正增加经过审核的训练图像？**

有，但要说清楚“审核到哪一步”：新增 **54,284 张去重后的 PlantVillage 图像**
（54,305 个文件 − 21 个重复文件），已完成 **解码审计 + 精确/像素级去重 +
与 D1 其余部分的跨来源筛查**，状态 `DUPLICATE_SCREENED`。
**尚未**达到 `APPROVED`：12,436 个感知相似候选待人工复核，且许可证冲突未决。
所以准确表述是“新增 54,284 张已解码去重、待感知复核的训练候选图像”，
不是“新增了 5.4 万张审核通过的训练数据”。

**6. 新增多少张？来自哪些数据集？**

- 新增：**54,284**（PlantVillage，Zenodo record 1204914，824,169,630 bytes，
  SHA-256 `d77991cc…db3739`）。
- 精确重复：42 对（42 个重复文件，21 个唯一多余副本），已从候选清单移除。
- 感知候选：12,436 对，进入人工复核队列，**没有自动删除**。
- 与 D1 其余部分的跨来源匹配：0 精确、0 pHash(≤6)。

**7. 是否发生与下游测试数据的交叉？**

没有发现：

- GWHD 全部 6,515 张图与 D1 候选清单（66,851 行）交叉筛查：**0 精确匹配、
  0 个 pHash 距离 ≤6 的候选**（`reports/stage5a_gwhd_data_audit.json`）。
- PhenoBench 在语料侧被**整来源排除**（不是逐图筛查），因此不存在“漏检一张”的风险。
- 其余保留集（PlantDoc、MinneApple、DeepWeeds 等）同样整来源排除。

需要诚实说明的边界：图像级哈希只能发现**像素重复**，不能发现“同一地块/同一植株
的另一次拍摄”。因此 `split_evidence` 与 `grouping_fields` 记录的是采集域/序列，
而不是仅靠哈希。

**8. D1 Candidate 当前实际可用图片数是多少？**

`data/frozen/agri-corpus-d1-candidate_*`：

- train：**66,851**（v1 train 12,567 + PlantVillage 54,284）
- val：**3,025**（v1 val，未改动）
- 合计：**69,876**

来源构成：ACRECropWeed 791、iNatWeeds 1,031、LUCASVision 8,938、RadishWheat 427、
SorghumWeed 137、VegAnn 1,243、PlantVillage 54,284。

状态是 **CANDIDATE（approved=false）**，未冻结。未决项：14,743 张 v1 待审计图像、
784 个待复核重复候选、12,436 个 PlantVillage 感知候选。
规范化 CSV 的 SHA-256：train `a90794de…e36fd3`，val `b4715e14…b6c03f`。

---

## 检测（9–17）

**9. GWHD 2021 是否取得完整数据？**

是。官方 Zenodo 单文件 `gwhd_2021.zip`，10,215,190,132 bytes，
MD5 `22b4b542c9ae7e056d7fcdeae9ecaed5` 与官方公布值**逐字符一致**；
本项目另记 SHA-256 `87649215…b01f68`。解压后 6,515 张图，
全部 1024×1024，**0 解码失败**；3 个官方 CSV 原样读取。

**10. 采用的 Backbone 接口和 Detection Head 是什么？**

Frozen ViT-H/14（224 预训练，patch 14，embed 1280）→ patch tokens 重排为
`[B, 1280, H/14, W/14]` → **Simple Feature Pyramid**（1×1 lateral + 4 个
3×3 smooth，256 通道，P2–P5）→ **torchvision Faster R-CNN**
（RPN anchors 4 层 × (16,32,64,128) × (0.5,1,2)，ROI 7×7），num_classes=2。
可训练 17,188,121 参数；冻结 632,186,880 参数，有断言保证
`backbone.encoder.*` 不出现可训练参数。

**11. 1024×1024 输入如何处理？**

**提高分辨率 + 位置编码插值**，正式协议 **518×518（37×37 tokens）**。

- 位置编码 16×16 → 37×37 用双三次二维插值（I-JEPA 的 `interpolate_pos_encoding`
  假设存在 cls token，本实现单独写了 `resize_pos_embed`，不修改原模型代码）。
- 方形 resize 由自定义 `SquareResizeTransform` 完成（torchvision 默认
  `GeneralizedRCNNTransform` 会把 518 取整到 544，并 padding 到 32 的倍数，
  这会破坏 token 网格，已显式覆写）。
- 框坐标随 resize 精确缩放（有单元测试）。
- 未采用 tiling：36 tiles/图、约 154 GB 特征缓存、36× 计算。

实测取舍（训练前测得）：224 → 204 img/s、**518 → 37 img/s**、700 → 18 img/s、
1008 → 7.4 img/s。518 是“细节 × 吞吐”的折中，并写进预注册协议。

**12. 检测训练实际运行了吗？**

运行了，共 **4 次 20K 训练**（A/B × seed 0/1）。官方 train 训练
20,000 iterations × batch 2，官方 val 每 2,000 步评估，官方 test 最后评估一次。
seed 0 两臂各约 1 GPU·小时（独占）；seed 1 为一致性复核（两臂并行，约 2.4 小时）。
运行目录：`runs/stage5a_gwhd/{A,B}_seed0`、`{A,B}_seed1`。
每个目录含 `protocol.json`（含数据与权重 SHA-256）、`train_log.csv`、
`eval.jsonl`、`results.json`、`head.pth`（该 checkpoint 有已记录的缺陷，见问题 30）。

**13. 官方 ViT-H 的 AP50、AP50:95 是多少？**

| split | seed | AP50:95 | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0 | 0.0759 | 0.2652 | 0.0192 | 0.0005 | 0.0593 | 0.1062 | 0.1977 |
| val | 1 | 0.0748 | 0.2608 | 0.0180 | 0.0003 | 0.0593 | 0.1038 | 0.1959 |
| test | 0 | 0.0155 | 0.0564 | 0.0101 | 0.0030 | 0.0184 | 0.0163 | 0.0604 |
| test | 1 | 0.0137 | 0.0600 | 0.0017 | 0.0031 | 0.0157 | 0.0181 | 0.0617 |

两 seed 平均：val 0.0753 AP / 0.2630 AP50；test 0.0146 AP / 0.0582 AP50。

**14. 农业 JEPA 2k 的 AP50、AP50:95 是多少？**

| split | seed | AP50:95 | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0 | 0.0828 | 0.2737 | 0.0282 | 0.0009 | 0.0645 | 0.1157 | 0.2047 |
| val | 1 | 0.0846 | 0.2822 | 0.0257 | 0.0027 | 0.0655 | 0.1193 | 0.2074 |
| test | 0 | 0.0187 | 0.0785 | 0.0040 | 0.0024 | 0.0230 | 0.0250 | 0.0752 |
| test | 1 | 0.0190 | 0.0787 | 0.0039 | 0.0024 | 0.0238 | 0.0226 | 0.0760 |

两 seed 平均：val 0.0837 AP / 0.2780 AP50；test 0.0189 AP / 0.0786 AP50。

**15. 两者差多少？**

B − A，两 seed 平均（括号内为 seed 0 / seed 1 的逐 seed 差值）：

| split | ΔAP | ΔAP50 | ΔAP_small | ΔAP_medium | ΔAP_large | ΔAR100 | ΔP@0.5 | ΔR@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | **+0.0084**<br>(+0.0070 / +0.0098) | **+0.0150**<br>(+0.0085 / +0.0215) | +0.0014 | +0.0057 | +0.0125 | +0.0093 | −0.0076 | **+0.0178** |
| test | **+0.0043**<br>(+0.0032 / +0.0053) | **+0.0204**<br>(+0.0220 / +0.0187) | −0.0006 | +0.0064 | +0.0066 | **+0.0146** | **+0.0415** | **+0.0191** |

- **AP50 / AR100 / R@0.5 / P@0.5 的差值在两 seed 上都很稳**，且量级约为两臂
  seed 间标准差的 5–10 倍；
- **AP50:95 的差值 +0.0043 仍然小**（约为 A 臂 seed 间波动的 3–5 倍），
  属于“方向一致、幅度未定”；
- 因此不能写成“检测显著提升”，可以写成“AP50 上 2 seed 一致的 +2.0 pp 提升，
  AP50:95 上 +0.4 pp 的同向小差异”。

**16. 不同训练 seed 是否表现一致？**

**是（2/2 同向），但只有 2 个 seed。** 每 seed 的 test 差值：

| seed | A AP | B AP | ΔAP | A AP50 | B AP50 | ΔAP50 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0155 | 0.0187 | +0.0032 | 0.0564 | 0.0785 | +0.0220 |
| 1 | 0.0137 | 0.0190 | +0.0053 | 0.0600 | 0.0787 | +0.0187 |

两臂的 test AP 在 seed 间几乎不动（A 0.0155/0.0137，B 0.0187/0.0190），
**差值没有随 seed 翻转**，这是“不是头部初始化随机性造成”的证据。
val 同样是 2/2 同向（ΔAP +0.0070 / +0.0098）。

必须说清的边界：这里的 seed 只改变**检测头初始化与数据顺序**，两个 seed 共享
**同一份**农业 JEPA 权重（Stage 3B 只有一个预训练 seed）。因此这证明的是
“差值不是检测头 seed 噪声”，**不是**“换一个预训练 seed 也成立”。

**17. 训练速度、显存和参数规模如何？**

| 项 | 数值 |
|---|---:|
| 可训练参数（Neck+RPN+ROI） | 17,188,121 |
| 冻结参数（ViT-H encoder） | 632,186,880 |
| 训练吞吐（518，batch 2，末段） | 11.6 img/s（A）／9.7 img/s（B，受共享 GPU 干扰） |
| 训练墙钟 | 3,567 s（A）／4,301 s（B） |
| 峰值显存 | 2.9 GB |
| 测试集推理延迟 | 45.50 ms/图（A）／45.49 ms/图（B） |

两臂架构完全相同，模型属性（参数、显存、延迟）一致；墙钟差异来自运行期间的
机器负载，不是模型差异。

---

## 科研（18–25）

**18. 农业 JEPA 的正向趋势是否从 PhenoBench 延伸到了 GWHD？**

**延伸到了，但强度不同。**

- PhenoBench（Stage 3B/4B，冻结编码器 + 线性 probe）：B − A = **+1.42 pp mIoU**，
  Weed IoU **+3.29 pp**，**3/3 seeds 一致** —— 这是已经成立的结论。
- GWHD（本阶段，冻结编码器 + Faster R-CNN）：B − A = **AP50 +2.04 pp**、
  AR100 +1.46 pp、R@0.5 +1.91 pp，**2/2 seeds 一致**；但
  **AP50:95 只有 +0.43 pp**（方向一致，幅度弱）。

正确表述：同一个方向（农业持续预训练后冻结特征更适合农业视觉任务）在两个
不同类型的任务上都出现了，且 GWHD 的 AP50 提升在 2 seed 上可复现；
但 GWHD 的强指标（AP50:95）仍弱，且只有 1 个预训练 seed、1 个检测数据集，
因此**不足以**写成“农业预训练提升目标检测”的论文主结论。

**19. 当前扩建的数据语料是否有足够来源多样性？**

**不够。** D1 候选 69,876 张里，PlantVillage 占 54,284（**77.7%**），而它是
实验室/受控条件下拍摄的叶片特写，与田间、遥感、机器人平台的分布差距最大。
其余 6 个来源合计只占 22.3%，且以田间作物/杂草为主，缺少：

- 果园/果实（MinneApple、GrowliFlower 都是保留集）；
- 航空/无人机视角（Agri-FM+ 明确过滤掉了航拍源）；
- 病害场景的**田间**图像（PlantVillage 是离体叶片）。

结论：**数量够、多样性不足**。继续堆 PlantVillage 只会让分布更偏。

**20. 最值得先扩充哪些公开来源？**

按“许可清晰 + 非保留集 + 能补多样性”排序（均已在 registry 中登记）：

| 优先 | source_id | 图像（论文值） | 许可 | 补什么 |
|---:|---|---:|---|---|
| 1 | `agrifm_ref57` Crop Pest and Disease Detection | 24,881 | CC-BY-4.0 | 田间病害，补 PlantVillage 的分布缺口 |
| 2 | `agrifm_ref37` V2 Plant Seedlings | 5,539 | CC-BY-SA-4.0 | 幼苗、受控但不同作物 |
| 3 | `we3ds` WE3DS RGB-D | 2,568 | CC-BY-4.0 | RGB-D、多平台 |
| 4 | `vcd` VCD | 2,258 | CC-BY-4.0 | 早期蔬菜、田间 |
| 5 | `sesame_weed` | 1,300 | **CC0-1.0** | 检测框、许可最干净 |
| 6 | `maize_weed`、`rumex_leaves`、`palmer_amaranth` | 843 / 809 / 614 | CC-BY-4.0 | 杂草检测多样性 |

注意：MuST-C（7,242）本机不可达；DeepWeeds / PlantSeg / Multiclass Weeds /
CropAndWeed / GrowliFlower 是保留集或待定基准，**不得**用于预训练。

**21. 哪些农业下游数据集最容易下一步接入？**

按“已有官方入口 + 许可清楚 + 与现有评价接口兼容”排序：

1. **PlantDoc**（2,569，检测）—— 与 GWHD 共用同一条 Faster R-CNN 管线，
   只需换数据集与类数，是最低成本的第二检测任务。
2. **MinneApple**（1,001，检测 + 实例分割）—— 同一检测管线，可用 COCO bbox；
   实例 mask 需要额外的 mask 评价路径。
3. **DeepWeeds**（17,509，分类）—— 冻结特征 + 线性 probe，与 PhenoBench 的
   probe 代码同构，接入成本最低；但它是 AgriMAE 的基准，必须整来源保留。
4. **Multiclass Weeds**（7,872，语义分割）—— 与 PhenoBench 语义分割接口同构。
5. **PlantSeg**（11,458，语义分割）—— 许可为“see repository”，需先确认条款。

**22. 当前是否具备开展十几万张农业持续预训练的条件？**

工程上**具备**，数据上**还不够**。

- 工程：Stage 3B 的 2,000-step 训练在 224、effective batch 128 下吞吐 38.7 img/s，
  267.3 ms/step。按此推算，10 万张图跑 10 个 epoch（100 万样本）约需
  **7.2 GPU·小时**；100 万样本 × 50 epoch 约 36 GPU·小时。单卡 A6000 完全可行。
- 数据：D1 候选只有 69,876 张，且 77.7% 来自单一来源，质量门也未走完
  （12,436 个感知候选待复核）。**先补多样性、再上规模**比直接堆到 15 万更合理。

**23. Adapter 是否值得进入正式实验？**

值得，但排在数据与第二个下游任务之后。已完成的是**非侵入式设计 + 真实权重烟雾测试**
（`reports/stage5a_adapter_feasibility.md`）：

- 零初始化 bottleneck Adapter 在真实 ViT-H/14 上初始输出与原始 Encoder 逐元素一致；
- 8/16/32 bottleneck × 32 blocks 分别 0.70M / 1.35M / 2.66M 参数（主干的 0.11%–0.42%）；
- 224、fp32、bs=4 下前向开销 **2.7%**（48.5 → 47.2 img/s）；
- 反向传播后主干权重逐元素不变，主干 `grad is None`。

但**没有跑任何 Adapter 下游训练**，所以目前没有精度数字。Stage 6 应先做
A0/B0 vs A1/B1 的 4 臂小对照，再决定是否做 Adapter-JEPA 预训练。

**24. AgriMAE 的公开代码与权重是否已经实际可获取？**

**没有验证。** 本阶段的检索只确认了 AgriMAE 论文与 AgriField-40K 的公开入口，
没有拿到可复现的代码仓库或权重下载地址，因此 registry 中没有任何一行把它记为
“已可获取”。这是一个**未完成项**，不应在论文中引用为对比基线，除非下一阶段
真正下载并跑通。

**25. 哪些结果已经足以作为论文实验材料，哪些仍属于探索性结果？**

足以作为材料：

- **数据侧**：59 个来源的登记表、许可证审计、重叠处置规则、GWHD 完整数据审计
  （官方校验和、6,515 图 0 解码失败、官方划分零交叉）、保留集隔离机制。
- **协议侧**：GWHD 检测的预注册协议（分辨率取舍有实测依据）、A/B 公平性断言。
- **PhenoBench**：+1.42 pp mIoU / +3.29 pp Weed IoU，3/3 seeds（Stage 3B/4B）。
- **GWHD（有保留地）**：AP50 / AR100 / R@0.5 的 2/2 seed 同向提升
  （+2.04 / +1.46 / +1.91 pp），可以作为“第二个任务出现同方向证据”的
  探索性结果写入讨论，但必须同时写出 AP50:95 幅度弱、单数据集、单预训练 seed
  这三条限制。

仍属探索性：

- GWHD 的 A/B 差值中 **AP50:95 那一项**（+0.0043，幅度仅为 A 臂 seed 间
  波动的 3–5 倍，方向一致但不够稳）；
- GWHD 的 A/B 在**其它检测数据集**上是否成立（只有 GWHD 一个检测任务，
  且农业 JEPA 只有一个预训练 seed）；
- D1 候选语料的规模收益（尚未用于训练）；
- Adapter 的任何精度结论（只有吞吐与参数量）。

**不得**写成论文结论的：

- “农业持续预训练（在一般情况下）提升了目标检测” —— 只有 1 个检测数据集、
  1 个预训练 seed，不能从 GWHD 外推；
- “GWHD AP50:95 显著提升” —— 差值 +0.0043，幅度弱。
- “农业语料已有 14.7 万张”（我们是 69,876 张候选，来源与口径都不同）；
- 把 0.0582 / 0.0786（AP50）或 0.0146 / 0.0189（AP50:95）与 Agri-FM+
  的 Detectron2 数字直接比较。

---

## 工程（26–30）

**26. Pytest 结果。**

```
466 passed, 7 skipped in 326.60s (0:05:26)
```

Stage 4B 基线是 389 passed；本阶段新增 4 个测试文件，收集到的用例数为
31 / 27 / 13 / 6 = **77**（466 − 389 = 77），无回归。

**27. 新增及修改的文件。**

见本文件末尾「变更清单」。

**28. 实际数据与 checkpoint 路径。**

| 内容 | 路径 |
|---|---|
| 官方 I-JEPA ViT-H | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| 农业 JEPA v1 2,000-step | `/home/test/agrisa_runs/stage3b_random_2000/jepa-encoder-step2000.pth` |
| GWHD 2021 归档/解压 | `/data/datasets/GWHD2021/gwhd_2021.zip`、`/data/datasets/GWHD2021/extracted` |
| PlantVillage 归档/解压 | 见 `reports/stage5a_plantvillage_audit.json` 中的 `archive` 字段 |
| D1 候选语料 | `data/frozen/agri-corpus-d1-candidate_{train,val}.csv` |
| 检测运行（完整） | `runs/stage5a_gwhd/{A,B}_seed{0,1}/`（被 gitignore） |
| 检测运行（机器可读副本） | `reports/stage5a_gwhd_runs/{A,B}_seed{0,1}/` |

**29. 当前分支、Git commit SHA。**

分支 `agri-stage5a`（自 Stage 4B 验证提交 `45a00b5fb62c25b74e0ae547fe91061cf1101c19`
新建，未修改 Stage 4A/4B 历史与报告）。

本阶段的最终提交为 `git log -1` 显示的那一条，提交信息以
`Stage 5A:` 开头；具体 SHA 在最终答复与本文件所在的提交中给出
（提交无法包含自身的 SHA，因此不在文件内硬编码）。

**30. 未完成工作及原因。**

1. **PlantVillage 的 12,436 个感知候选与 v1 的 784 个待复核重复**：需要人工判断，
   不能自动删除（协议要求候选进复核队列）。→ D1 因此保持 CANDIDATE。
2. **14,743 张 v1 待审计图像**：需要逐图核对来源与分割字段，本阶段未完成。
3. **GWHD 只跑了 2 个 seed**：第 3 个 seed 的 GPU 时间排在 Stage 5B。
4. **第二个检测数据集未接入**：PlantDoc / MinneApple 的接入代码路径已具备，
   但本阶段把预算放在了 GWHD 的完整审计与 A/B 上。
5. **AgriMAE 代码/权重未验证**：未找到公开入口，见问题 24。
6. **Adapter 未做下游训练**：按任务书要求，本阶段只做独立设计 + 烟雾测试。
7. **9 个 URL 被 403 拦截、6 个网络不可达、3 个 404**：需要登录态或换网络环境重试，
   这些行保持 `URL_VERIFIED`，没有伪造“已核实”。
8. **seed 0/1 的 `head.pth` 缺少 Neck 权重**：保存过滤写成了
   `not key.startswith('backbone.')`，把 `backbone.lateral/smooth`（2,688,256 参数）
   一起过滤掉了，只剩 RPN+ROI 的 14,499,865 参数。指标是进程内用完整模型算的，
   **不受影响**；但仅凭这些 checkpoint 不能重建被评价的模型。修复已提交
   （`trainable_state_dict()` + 单元测试），四次运行不返工，问题在
   `reports/stage5a_gwhd_results.md` §8.1 中显式记录。

---

## Stage 5B 建议

1. **D1 是否冻结？** 不冻结。先完成 12,436 + 784 个候选的人工复核、
   14,743 张待审计图，并解决 PlantVillage 的许可证冲突（CC-BY-SA-3.0-US vs
   CC-BY-4.0）。在此之前任何“冻结”都只是把未决问题固化。
2. **先补数据还是先训全参数 JEPA？** 先补数据。当前 77.7% 的图像来自单一来源，
   现在做全参数持续预训练会把分布进一步推向实验室叶片。建议先引入
   Crop Pest & Disease（24,881）、V2 Plant Seedlings（5,539）、WE3DS（2,568）、
   VCD（2,258）四个非保留集来源，把最大单一来源占比降到 50% 以下。
3. **是否需要第二个下游任务？** 需要，而且优先于更多 seed。GWHD 的小目标
   问题让绝对 AP 很低，单个任务的结论太脆弱。PlantDoc（检测）与
   Multiclass Weeds（语义分割）的接口成本最低，建议在 Stage 5B 各接一个。
4. **Adapter + Head 小规模对照？** 可以做，但要等第二个下游任务接入后一起做，
   否则又变成“单任务 + 单 seed”的结论。建议 4 臂（A0/B0/A1/B1）× 2 seed。
5. **GPU 预算投向**：
   - 40% → 新来源的下载/解码/去重审计（数据是当前瓶颈）；
   - 30% → 第二、第三个下游任务的 A/B（含 seed 复核）；
   - 20% → 全参数持续预训练（在新语料冻结之后，按 10 万–30 万张、10–20 epoch 规划）；
   - 10% → Adapter 对照。

---

## 测试

```
$ .venv-uv/bin/python -m pytest -q
466 passed, 7 skipped in 326.60s (0:05:26)
```

新增 4 个 Stage 5A 测试文件，覆盖：门禁状态机与登记表一致性、
官方划分隔离与重复行处理、框校验/裁剪/翻转一致性、位置编码插值、
冻结编码器反向传播不更新主干、坐标缩放、可训练 checkpoint 只含
Neck+Head 且不含 Encoder、pHash/精确去重、Adapter 零初始化恒等性。

## 变更清单

相对 Stage 4B 验证提交 `45a00b5`。

**新增代码（12 个模块）**

| 文件 | 作用 |
|---|---|
| `src/registry/schema.py` | 9 个顺序门禁 + `BLOCKED`、逐门推进断言、CSV/JSON 规范化与哈希 |
| `src/registry/catalog.py` | 63 条登记行（AgriField-40K 17、Agri-FM+ 35、下游 12，含 PlantVillage），逐条证据 |
| `src/registry/benchmarks.py` | 12 个下游基准登记 |
| `src/registry/ingest.py` | 解码、SHA-256/像素哈希、pHash/dHash、精确/感知去重 |
| `src/registry/freeze.py` | 语料冻结（保留集整来源排除、规范化 CSV + SHA-256） |
| `src/downstream/gwhd_data.py` | GWHD 官方 CSV 读取、框校验、数据集、划分隔离 |
| `src/downstream/vit_detection.py` | ViT patch token → Simple Feature Pyramid → Faster R-CNN |
| `src/downstream/detection_train.py` | 固定预算训练/评价 CLI，写协议与结果 |
| `src/downstream/registry.py` | 统一实验 spec 与可比性断言 |
| `src/downstream/backbone_factory.py` | A/B 权重解析与冻结编码器构建 |
| `src/downstream/evaluation_registry.py` | 指标定义注册 |
| `src/downstream/adapters.py` | 零初始化 bottleneck Adapter + 钩子 |

**新增工具（6 个）**：`tools/stage5a_check_source_urls.py`、
`stage5a_build_registries.py`、`stage5a_ingest_source.py`、
`stage5a_build_d1_candidate.py`、`stage5a_audit_gwhd.py`、`stage5a_compare_gwhd.py`。

**新增测试（4 个文件，77 个用例）**：`tests/test_stage5a_{registry,detection,ingest,adapters}.py`。

**新增数据台账**：`data/registry/{agricultural_sources,agricultural_benchmarks,
source_overlap_candidates,downstream_holdout_sources}.csv` + manifest、
`data/registry/ingested/plantvillage_manifest.csv`、
`data/frozen/agri-corpus-d1-candidate_{train,val}.csv` + manifest。

**新增报告**：本文件 + `stage5a_data_source_catalog.md`、
`source_overlap_audit.md`、`dataset_licence_audit.md`、`benchmark_registry.md`、
`gwhd_data_audit.{md,json}`、`gwhd_protocol.md`、`gwhd_results.md`、
`gwhd_comparison.{csv,json}`、`gwhd_ab_summary.md`、
`gwhd_runs/{A,B}_seed{0,1}/{protocol.json,results.json,eval.jsonl,train_log.csv}`、
`plantvillage_audit.json`、`url_check.json`、`adapter_feasibility.md`。

**修改**：`pyproject.toml` / `uv.lock`（加入 `py7zr`、`pycocotools`）。

**未提交**：`runs/`（被 `.gitignore` 忽略）、GWHD 数据、PlantVillage 数据、
模型权重。仓库中不含大模型或完整数据集。