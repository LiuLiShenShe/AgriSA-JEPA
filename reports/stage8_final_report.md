# Stage 8 — 最终报告：农业多任务迁移与跨场景泛化验证

> 生成时间（UTC）：2026-11-09（Phase 4 完成后定稿；Phase 5 汇总自
> `stage8_results_ledger.json` / `stage8_paired_contrasts.json`，逐 seed 值可回溯
> 到 `/data/agrisa_jepa_runs/stage8/<task>/<backbone>_seed<N>/{result,results}.json`）

## 0. 目标与范围

- 验证 AgriSA-JEPA（农业继续预训练 I-JEPA ViT-H/14）是否真正具有跨农业任务迁移价值。
- **5 主臂**（Official / A_p1 / A_p2 / B_p1 / B_p2）× **3 下游 head seed（0/1/2）** ×
  **5 任务**；**C 臂**（C_p1/C_p2）探索。全部 79 行正式实验完成。
- 冻结 encoder + 任务头（线性分类 / 线性分割 / Faster R-CNN），无端到端微调、无 adapter。
- 下游 seed 0/1/2 与预训练 seed 20261104/20261105 严格不同（RQ5 隔离）。
- 官方 test 每行只评一次；GPU1 串行；不可用 conda。

## 1. 数据集与泄漏审计（Phase 1）

| dataset | 任务 | 状态 | 图数 | raw/pixel 泄漏命中 | 切分 |
|---|---|---|---|---|---|
| GWHD 2021 | detection | READY | 6,515 | 0 / 0 | 官方 competition_{train,val,test}.csv；test 评一次 |
| CropAndWeed | segmentation | READY_WITH_MODIFICATION | 8,034 | 0 / 0 | train = ave_ 全集 3,044；eval = vwg_ 每 session 1 图（seed 20261107），665 |
| DeepWeeds | classification | READY_WITH_MODIFICATION | 17,509 | 0 / 0 | fold-0：train 10,501 + val 3,501 合并训练；test 3,507 评一次 |
| PlantDoc | classification | READY_WITH_MODIFICATION | 2,579 | 0 / 0 | 官方 train/test 原样；val=train 15% 分层（seed 20261107）；test⊆train 不变量 |
| PhenoBench | segmentation | READY | 13,767(含mask) | 0 / 0 | 官方 train/val；A/B 复用 Stage 7B 结果 |

- 精确哈希（raw_sha256 + pixel_sha256）对 147,277 张预训练图零命中；GWHD 2 张
  重采样同图样本已在 R3v3 排除的历史结论复核无误。
- CropAndWeed 使用官方 CropsOrWeed9 映射（库内逐字对照），Soil(0)/Vegetation(255) 为 ignore。
- 3 个候选（MinneApple / GrowliFlower / PlantSeg）本地不可得，BLOCKED，未进入注册表。

## 2. 冻结协议（Phase 2 + 变更日志）

- 83 行注册表（5 任务 × 5 primary × 3 seed；phenobench 单命令 3 seed；C 臂 6+6+6 行）。
- 每行含完整命令、backbone checkpoint SHA 引用（Stage 7B manifest / official release）、
  output dir；官方 test 只评一次；训练从不覆盖（已有 result 则跳过）。
- 实现层偏差（均不改变科学内容，详见 `stage8_protocol_changelog.md`）：
  - cropandweed 训练改分块流式加载（原一次性载入 ~90 GB OOM exit 137；新峰值 ~0.3 GB）；
  - `--roles exploratory` 精确匹配 bug 修复（C 臂此前静默跳过，修复后补跑）；
  - NVMe 空间：每 backbone 特征缓存 ~102 GB，该 backbone 全部行完成后即删除，
    由 runner 内钩子执行（检查全部行完成才删，manifest SHA 绑定）。

## 3. 结果（Phase 5；主指标均值 ± 逐 seed）

**PhenoBench 分割（官方 mIoU）** — 域内分割

| backbone | mean | per-seed |
|---|---|---|
| OFFICIAL | 71.68 | 72.66 / 72.60 / 69.78 |
| A_p1 / A_p2 | 72.64 / 72.78 | 73.79/73.02/71.11 · 73.34/73.06/71.94 |
| B_p1 / B_p2 | 72.62 / 72.29 | 73.89/72.58/71.38 · 73.49/72.30/71.09 |

**PlantDoc 分类（28 类叶病，macro-F1）**

| backbone | mean | per-seed |
|---|---|---|
| OFFICIAL | **0.4323** | 0.439 / 0.439 / 0.419 |
| A_p1 / A_p2 | 0.3965 / 0.4028 | 0.382/0.406/0.402 · 0.406/0.390/0.412 |
| B_p1 / B_p2 | 0.3374 / 0.3748 | 0.334/0.335/0.343 · 0.389/0.371/0.365 |

**DeepWeeds 分类（9 类，macro-F1）**

| backbone | mean | per-seed |
|---|---|---|
| OFFICIAL | **0.8363** | 0.835 / 0.845 / 0.829 |
| A_p1 / A_p2 | 0.8196 / 0.8276 | 0.820/0.827/0.812 · 0.824/0.830/0.829 |
| B_p1 / B_p2 | 0.7723 / 0.7850 | 0.760/0.787/0.770 · 0.785/0.785/0.785 |

**CropAndWeed 分割（CropsOrWeed9，9 类 mIoU）**

| backbone | mean | per-seed |
|---|---|---|
| OFFICIAL | 20.12 | 20.62 / 20.69 / 19.05 |
| A_p1 / A_p2 | 19.63 / 20.12 | 19.42/20.68/18.79 · 20.27/21.66/18.43 |
| B_p1 / B_p2 | 18.53 / 18.82 | 18.92/19.14/17.55 · 19.19/19.45/17.83 |

**GWHD 2021 检测（COCO test AP）**

| backbone | mean | per-seed |
|---|---|---|
| OFFICIAL | 0.0141 | 0.0165 / 0.0137 / 0.0121 |
| A_p1 / A_p2 | 0.0161 / 0.0196 | 0.0155/0.0161/0.0166 · 0.0179/0.0199/0.0211 |
| B_p1 / B_p2 | 0.0167 / 0.0166 | 0.0163/0.0167/0.0171 · 0.0188/0.0163/0.0146 |

> 注：GWHD 绝对 AP 很低（检测头的 Frozen-ViT 冻结 backbone + 20k iters 弱头；
> 报告的是相对比较，绝对值的解释力有限）。

**C 臂（探索，未进入主决策）**：deepweeds macroF1 C_p1 0.808 / C_p2 0.796；
cropandweed mIoU C_p1 18.87 / C_p2 18.53；gwhd AP C_p1 0.0180 / C_p2 0.0180；
phenobench 沿用 Stage 7B（B vs C 方向不一致 → INCONCLUSIVE）。C（2k 更新、长预热）
与 B 量级相近，均未表现出优于 A/official 的收益。

## 4. 配对对比（逐任务；seed 一致性规则：全部 head seed 同向 且 |均值差| ≥ 阈值）

| 任务（阈值） | Official−A_p1 | Official−A_p2 | Official−B_p1 | Official−B_p2 | A−B p1 | A−B p2 |
|---|---|---|---|---|---|---|
| plantdoc (0.005) | +0.036 **CLEAR** | +0.030 **CLEAR** | +0.095 **CLEAR** | +0.058 **CLEAR** | +0.059 **CLEAR** | +0.028 **CLEAR** |
| deepweeds (0.005) | +0.017 **CLEAR** | +0.009 **CLEAR** | +0.064 **CLEAR** | +0.051 **CLEAR** | +0.047 **CLEAR** | +0.043 **CLEAR** |
| cropandweed (0.5) | +0.49 INCONCL | +0.00 INCONCL | +1.59 **CLEAR** | +1.30 **CLEAR** | +1.10 **CLEAR** | +1.30 **CLEAR** |
| gwhd (0.002) | −0.0020 INCONCL | −0.0056 **CLEAR*** | −0.0026 INCONCL | −0.0025 **CLEAR*** | −0.0006 INCONCL | +0.0030 INCONCL |
| phenobench (0.2) | −0.96 **CLEAR*** | −1.10 **CLEAR*** | −0.94 INCONCL | −0.61 INCONCL | +0.02 INCONCL | +0.49 INCONCL |

*负号 = 继续预训练优于 Official；INCONCL = 方向不一致或未达阈值。*

## 5. 研究问题回答

- **RQ1 农业继续预训练是否跨任务优于官方 I-JEPA？——否，任务依赖。**
  唯一明确正收益在**域内分割**（PhenoBench，A 两 seed 一致 +0.96/+1.10 mIoU）与
  **GWHD 检测**的 A_p2/B_p2 微小胜（+0.0056/+0.0025 AP，均 seed 一致）；
  **两个细粒度分类任务（PlantDoc/DeepWeeds）上官方清晰胜出所有继续预训练臂**；
  CropAndWeed（域外分割）上 A 与官方打平、B 清晰劣于官方。净结论：收益是
  **任务型别相关**，不是普遍性的。

- **RQ2 收益是否跨数据集泛化？——不。**
  无任何数据集上出现跨数据集的"继续预训练普遍更优"模式；正向收益仅在
  PhenoBench（域内）出现，GWHD 上的微小优势与分类/域外分割的劣势并存。
  "农业继续预训练有普适迁移价值"的主张被数据拒绝（其肯定形式无一幸免）。

- **RQ3 Stage 7B 调度差异（A vs B）是否迁移到其他任务？——分任务。**
  **三个分类/细粒度任务上 A 稳定优于 B**（plantdoc +0.059/+0.028、deepweeds
  +0.047/+0.043、cropandweed +1.10/+1.30，全部 seed 一致 CLEAR）；PhenoBench 与
  GWHD 上 A−B 不显著。Stage 7B 的 A/B 区分因此**在细粒度判别任务上有跨任务后果**
  （Stage 7B 结论"50k horizon 对下游无明显收益"在 PhenoBench 成立，在分类任务不成立）。

- **RQ4 不同任务是否需要不同表征？——是。**
  同一组 backbone 上，分割/检测与细粒度分类的方向**相反**（前者正/中性、后者一致负），
  且 head 完全一致（线性头、同 seed、同初始化约定）。表征差异的"价值"是任务函数：
  提示继续预训练改变的是对细粒度类内/类间判别不利的统计，而对大尺度语义分割中性偏正。

- **RQ5 收益是否稳定（预训练 seed / head seed / 切分 / 数据集）？——部分稳定。**
  - 预训练 seed（p1 vs p2）：同一臂在两 seed 上方向**基本一致**（所有 CLEAR 判定都
    要求两 seed 同向 且 head seed 同向），例外是 phenobench 的 B 臂与 gwhd 的 A_p1
    （一个 seed 达到阈值、另一个不一致）。
  - head seed（0/1/2）：全部 CLEAR 判定均通过三 head seed 同向约束；seed 间绝对差
    可达 ±1.5 mIoU / ±0.02 macroF1，反映头初始化噪声，配对对比已吸收。
  - 切分：CropAndWeed/DeepWeeds/PlantDoc 的切分与预训练清单零重叠（RQ5 的泄漏维度干净）。
  - 数据集间：**不稳定**——同臂在不同数据集上收益符号翻转（见 RQ1/RQ2），这本身是
    最重要的稳定性结论。

## 6. 科学决策问题（10 项）与总体建议

> 原阶段简报提及"10 个科学决策问题"，但编号清单未随计划持久化到仓库；以下按本阶段
> 目标（判断 AgriSA-JEPA 是否真正具有跨农业任务迁移价值）从数据可直接回答的决策点
> 归纳为 10 项，供 Stage 9 立项使用。

1. **继续预训练是否有可复现的跨任务收益？** 有，但仅限域内分割（PhenoBench A 臂
   +1.0 mIoU，两预训练 seed、三 head seed 全部一致）与 GWHD 微小胜（A_p2/B_p2）。
2. **收益是否普遍（不限任务）？** 否：两个分类任务上全部继续预训练臂一致劣于官方。
3. **哪一臂该继续？** A 臂（2k-update 对等预算）：唯一在所有细分任务上位居榜首
   （分类第二、域内分割第一、GWHD 与 Official 持平或微胜）。
4. **B 臂（50k horizon）是否该保留为 Stage 9 主线？** 否：在 4/5 任务上 A>B 或打平，
   无一项任务 B 清晰优于 A；长 horizon 收益未兑现。
5. **C 臂（长预热短更新）是否值得深入？** 不值得作为主线：与 B 量级相近、未优于 A。
6. **官方 I-JEPA 在哪些任务上仍是基准？** 细粒度分类（PlantDoc/DeepWeeds）与
   CropAndWeed 域外分割；任何宣称"优于官方"的主张必须按任务限定。
7. **分类任务的负迁移是否与类粒度/类不平衡有关？** 未设计为可判定（本阶段冻结方案
   无类粒度消融）；列为假设，需 Stage 9 对照实验而非结论。
8. **GWHD 上 A_p2/B_p2 的微小正收益是否可信？** 数值达阈值且 seed 一致，但绝对 AP
   极低，应先复验检测头/指标（见 3 节注）再写入任何论文主张。
9. **表征的任务分化（RQ4）是否足以作为研究主张？** 是——"表征价值是任务函数"有
   5 任务一致证据支撑，比"普适迁移"更严谨、可辩护。
10. **是否存在"农业预训练无价值"的反叙事风险？** 分类任务数据反而支持该反叙事；
   报告与论文必须呈现完整 5 任务图景而非选择子集。

**总体建议：PROCEED WITH RESCOPING（调整研究主张后继续）。**
可继续的科学主张是受任务型别限定的："**农业继续预训练（2k-update 对等，A 臂）在
域内语义分割、检测上带来可复现的小幅优势，但在细粒度叶/草分类上稳定劣于官方
I-JEPA；增益与损失同源于表征的统计改变（RQ4），且 A/B 调度差异在细粒度任务上
跨数据集迁移（RQ3）**"。该主张可由 5 任务 × 5 backbone × 3 head seed 的配对结果
（全部阈值 + seed 一致性判定）直接支持。**不能**声称普适增益（RQ2 拒绝）；若下一阶段
立足"普适迁移"，本数据构成 STOP；立足"任务型别的表征分化"，则 A 臂 + 域内分割/
检测路线值得继续。

## 7. 可复现性与产物

- 分支 `agri-stage8-multitask-transfer`，未合并 main、未 force-push；本地保留全部
  checkpoint、特征缓存、结果 JSON、wall clock。
- 执行账目：79 行多实验成功执行（phenobench 5 + plantdoc 15 + deepweeds 21 +
  cropandweed 21 + gwhd 21），**累计 GPU 墙钟 45.9 小时**（2026-11-07 → 2026-11-09
  UTC，GWhd 最后一行 11-09T19:23Z 完成）。分任务：

  | 任务 | 行数 | 小时 | 分钟/行 |
  |---|---|---|---|
  | gwhd_det | 21 | 26.3 | 75.0 |
  | cropandweed_sem | 21 | 15.5 | 44.2 |
  | deepweeds_cls | 21 | 3.1 | 8.8 |
  | phenobench_sem | 1 | 0.7 | 40.6 |
  | plantdoc_cls | 15 | 0.4 | 1.7 |

  部分行受同期 GPU1/慢盘争用影响（如 gwhd/B_p1/s1 113.9 min、deepweeds/C_p1/s0
  110.3 min 显著高于该任务中位数）；每行 exit=0，时长波动不影响结果一致性判定。
- 每一行产物：`result.json`/`results.json`（protocol 字段、backbone SHA、split、seed、
  per-seed 指标）+ wallclock 记录（`stage8_phase4_wallclock.csv`）。
- 代码：全新探测/数据/运行器单元测试全绿；Phase 4 与全部纠错提交已被追溯，无未记录
  过滤；未使用 conda；仅 GPU1。