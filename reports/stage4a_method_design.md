# Stage 4A — 农业结构感知 Mask 的方法设计

> 状态：**探索性（exploratory）**。`agrifield40k-v1` 仍为 PROVISIONAL，本轮结论不构成论文定稿证据。
> 本文件只解释方法设计与文献定位，不报告实验结果（见 `stage4a_mask_statistics.md`、`stage4a_phenobench_comparison.md`、`stage4a_ablation.md`）。

## 0. 研究问题与边界

核心问题：

> 在相同的官方 I-JEPA 初始化、农业数据、训练预算和下游评价条件下，植被区域或结构边界引导的 Mask，是否比标准随机多块 Mask 学到更适合农业语义分割的表征？

明确**不做**的声明：

- 本文不声称提出了新的世界模型、三维结构建模或时空动力学预测方法。
- 本文不声称首次提出"显著性/注意力引导 Mask"。已有工作（见 §5）在先。
- 本文的结构先验完全无监督，**不使用** PhenoBench 标签、语义分割 GT 或目标检测标注。
- Mask 只改变**目标块的位置**，不改变块尺寸分布、块数量、上下文/目标排斥关系或损失函数。

---

## 1. 为什么农业图像中植物、土壤和边界区域可能需要不同的预测采样策略？

I-JEPA 的预训练目标是从一个（或少数）上下文块预测若干目标块的**表征**，而不是重建像素。I-JEPA 原论文明确指出，掩码策略是引导表征语义性的核心设计：目标块需要足够大的尺度（语义性），上下文需要足够分散（信息量）([Assran et al., 2023, arXiv:2301.08243](https://arxiv.org/abs/2301.08243))。

这个要求在农业图像里会与自然图像产生系统性偏差：

1. **语义密度高度不均匀。** 农田图像常常大部分是土壤、地膜、天空或阴影，只有少数区域包含作物/杂草。均匀随机采样目标块，会让大量预测目标落在低语义密度的裸土上——这些位置的"可预测性"很高（纹理平稳），但对下游语义分割的区分性贡献低。
2. **类别边界集中在窄区域。** 语义分割真正困难的像素是 crop/soil、weed/soil、crop/weed 的边界。这些边界在图像中占的面积比例很小；随机块命中它们的概率与面积成正比。
3. **纹理是强干扰项。** 土壤结块、石块、残茬、阴影都有很强的局部梯度，但没有类别含义。纯边缘先验会把这些区域当作"结构"，这正是 C2 需要与植被响应结合、并需要 C3 打乱对照的原因。
4. **JEPA 预测的是表征而非像素。** 目标块落在"耕过的土"上，模型只需预测出"这是土"的平稳表征即可，任务过于容易；落在"叶片—土壤交界"上，模型必须编码更细的空间关系才有用。

因此假设是：**把目标块的位置向植被区域或植被—土壤边界偏移，可能提高预测任务对下游分割所需信息的覆盖度**。这是假设，不是结论；本轮实验的作用是检验它。

---

## 2. 当前颜色与边界先验能够表达什么，不能表达什么？

### C1：Excess Green 植被显著图

实现（`src/masks/agri_structure.py`）：

1. 对已按 ImageNet 均值方差标准化的输入反归一化并 clamp 到 `[0,1]`；
2. `ExG = 2G - R - B`（原始范围约 `[-2, 1]`）；
3. 逐图**稳健分位数归一化**：用 10% / 90% 分位数做线性拉伸并 clamp 到 `[0,1]`，抑制异常值与曝光差异；
4. 均值池化到 `16×16` patch grid；
5. 若某张图的动态范围低于阈值（空图、纯土、极低对比度），该图 `usable=False`，回退到官方随机采样。

**能表达**：绿色活体植被的*相对*空间分布；在光照均匀、绿色作物为主的图像中，这是一种廉价、可复现、无需训练的先验。

**不能表达**：
- **黄化、枯死、成熟期作物**：ExG 依赖 G 通道占优，黄叶（R≈G）和枯叶（R>G）会被判为低植被；
- **非绿色作物**（紫叶、红茎品种）与覆盖物；
- **光照与白平衡**：强逆光、色偏、夜间补光会整体平移 ExG；
- **物种身份**：ExG 不区分 crop 与 weed——它对两类都高，因此**不可能**直接提供作物/杂草判别；
- **边界**：ExG 给出的是区域响应，不含轮廓信息。

### C2：结构边界图

实现：

1. 亮度 `Y = 0.299R + 0.587G + 0.114B`（BT.601）的 Sobel 梯度幅值 `E_Y`；
2. ExG 响应图的 Sobel 梯度幅值 `E_ExG`（捕捉"绿—非绿"响应突变，即植被边界）；
3. 稀疏归一化（99 分位数，并对极稀疏图退化到 `0.1×max`）；
4. 加权组合 `0.5·E_Y + 0.5·E_ExG`；
5. 池化到 `16×16`。

权重 `0.5` 是**实测选择**，不是先验最优：在扫描 `w ∈ {0, 0.25, 0.5, 0.75, 1.0}` 后，`w=0.5` 的 C2 图与 C1 植被图的相关系数接近 0，说明二者携带**不同**的空间信息，而不是 C1 的近似复制。

**能表达**：植物轮廓、叶片边缘、植被—土壤交界；显式引入了颜色响应梯度，因此比纯亮度边缘更偏向"有意义"的边界。

**不能表达 / 已知失效**：
- 土壤结块、石块、残茬、阴影产生的高梯度与真实植被边界**不可区分**（这正是纯边缘先验的根本限制）；
- 无法区分 crop/weed 边界与 weed/soil 边界；
- 边界强度是局部量，不含"哪一侧是目标"的方向信息；
- 在密集冠层内部（叶片互相遮挡）几乎没有强边界，C2 会近似均匀。

> 上述失效模式在 `stage4a_mask_statistics.md` 中被**定量**记录（raw edge、veg edge、最终结构图、以及目标块内边缘是否由裸土主导），而不是仅作定性声明。

---

## 3. 为什么沿用原始 JEPA 目标？

因为本轮的自变量被限定为**Mask 采样策略**。若同时改变损失（例如加入重建项、对比项或结构辅助损失），则无法把任何性能差异归因于掩码。

具体保持不变的包括：

- 预测目标：context encoder 输出经 predictor 预测 target encoder（EMA 副本）在目标块上的表征；
- 损失：smooth-L1（β=1），在目标块 token 上取平均；
- backbone / predictor / target encoder 结构与初始化；
- EMA（0.996 → 1.0 线性）、优化器、LR/WD 调度、batch、更新步数。

这样，B/C1/C2/C3 之间唯一的差异是"目标块放在哪里"。同时也满足一个方法论要求：**不同 Mask 下的 JEPA loss 数值不可直接比较**——掩码改变了预测目标的难度，loss 更低不代表表征更好。因此本项目的判据始终是一致下游协议下的 mIoU / Crop IoU / Weed IoU。

---

## 4. 为什么选择"引导目标块位置"，而不是直接使用 GT 语义区域？

这是本轮最关键的公平性约束。

若直接用 GT 语义区域选目标：

- 需要分割标注，而农业自监督预训练的价值恰恰在于**没有**标注；用 GT 选 mask 会让预训练与下游评价共享监督信息，产生信息泄漏；
- 会改变目标块的**形状与面积**（GT 区域是不规则连通域），于是"目标 patch 数 / 总掩码面积"也变了，性能差异无法与几何差异分离。

因此采用 **Candidate Block Sampling + 结构引导选择**：

1. 目标块尺寸、长宽比、数量 `npred=4`、上下文块数量 `nenc=1` 全部由**官方 `MaskCollator` 的同一套采样器**产生；
2. 对每个目标块，在**合法位置**（不越界、满足 `min_keep`）中随机取 `K=16` 个候选；
3. 候选分数 = 结构先验在该块覆盖的 patch 单元上的均值；
4. 选择概率 `p_i = softmax(λ · ŝ_i)`，其中 `ŝ` 是候选分数按 `[min, max]` 归一化后的值（消除不同图之间先验尺度差异）；`λ=6.0` 为本轮固定强度；
5. `λ=0` 时退化为候选间的均匀随机选择——这是验证"候选机制本身没有引入几何偏差"的诊断模式；
6. 上下文块仍由父类 `_sample_block_mask` 以 `acceptable_regions` 排斥逻辑生成，`min_keep` 与批次共同长度截断均沿用父类。

这样，B 与 C1/C2/C3 的**目标块数量、尺寸分布、长宽比分布、总覆盖面积、上下文—目标排斥关系、批次截断语义**完全一致，差异只在"落在哪个合法位置"。

---

## 5. 与 AttMask 等显著性/注意力引导 Mask 工作的异同

**相同的核心思想**：随机掩码不是最优的，掩码位置应当携带"信息量"，因此用某种先验决定 mask 位置。这一思想在 AttMask 中表述为"from random masking to informed masking"：

- **AttMask**（Kakogeorgiou et al., "What to Hide from Your Students: Attention-Guided Masked Image Modeling", ECCV 2022, [arXiv:2203.12719](https://arxiv.org/abs/2203.12719)）在**蒸馏式** MIM 中，用教师 ViT 的注意力图引导学生的掩码，让被遮挡的恰是教师关注的高注意力区域，从而构造更难的借口任务。
- **SemMAE**（Li et al., "SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders", NeurIPS 2022, [arXiv:2206.10207](https://arxiv.org/abs/2206.10207)）用语义部件引导掩码，属于"借助某种语义分组信号选择遮挡区域"的路线。

**不同点**：

| 维度 | AttMask | SemMAE | 本工作（C1/C2/C3） |
|---|---|---|---|
| 先验来源 | 教师网络注意力（需额外前向） | 语义部件（依赖聚类/语义信号） | 手工、无参数的颜色/梯度响应（无需额外网络） |
| 预训练框架 | 蒸馏式 MIM | MAE 重建 | **I-JEPA 表征预测** |
| 掩码单元 | token 级 | 语义区域 | **官方矩形多块，仅移动目标块位置** |
| 是否改变掩码几何 | 可改变 | 可改变 | **否**（几何由官方采样器决定） |
| 目标领域 | 自然图像 | 自然图像 | 农田图像（植被/土壤/边界） |
| 结构控制 | 无 | 无 | **C3 打乱先验** |

本工作的定位是：把"informed masking"思路迁移到 **I-JEPA + 农业图像**，并把"先验的空间对齐是否真的有用"用 C3 显式检验。方法上的新意有限，价值在于**受控的实证比较**，而非新算法。

---

## 6. C3 的打乱先验控制了哪个混杂因素？

C3 使用**与 C2 完全相同的结构先验图**，但把 `16×16` 的 patch 单元按**逐图、由 `sha256(base_seed:sample_id)` 派生**的排列做置乱：

- 数值**多重集完全不变**（是排列），因此候选分数的分布、softmax 温度、`λ=6.0` 的引导强度、被选中块分数的统计特性都与 C2 一致；
- 空间排列被破坏，先验与图像内容的对应关系消失。用 lag-1 空间自相关量化：C2 的结构图保持正自相关，C3 被打乱后自相关降到接近 0。

于是 C3 是一个**强对照**：它保留"有偏选择"这一行为（不是均匀随机），只移除"偏置与图像空间对齐"这一成分。

因此：

- `C2 − B` 若为正、`C3 − B` 接近 0 → 证据支持"**空间对齐的结构信息**有用"；
- `C2 − B ≈ C3 − B > 0` → 收益可能来自"任何有偏的、聚集式的目标块选择"（例如集中在少数区域导致上下文更分散），而非结构语义；
- `C3 − B < 0` → 说明随机化的先验甚至有害，提示候选采样机制本身有代价。

C3 因此不是"另一个随机 mask"，它控制了 **prior value distribution vs. prior spatial alignment** 这一混杂因素。

---

## 7. 如果 C1/C2 没有超越 B，可能意味着什么？

需要在报告里区分以下几种解释，而不能笼统地说"方法没用"：

1. **先验本身太弱**。ExG 在黄化/枯死/裸土占优的图像上失效；Sobel 边界被土壤纹理淹没。可以先检查"目标块内植被分数是否真的提高了"（本轮已用离线统计验证 C1 确实提高、C2/C3 未显著提高），若先验没有改变目标分布，则任何下游差异都不应归因于方法。
2. **预算太小**。2,000 optimizer updates × effective batch 128 ≈ 0.26M 样本，相当于 I-JEPA 官方 300 epoch 的极小比例。结构先验带来的收益可能需要更长训练才显现；2,000 步只能检验"方向性信号"。
3. **任务不匹配**。PhenoBench 是 1024×1024 UAV 图像，而预训练是 512×512 近景图像，域间差异可能主导结果；此时 mask 策略的差异被域偏移掩盖。
4. **信号被下游探针摊平**。冻结 backbone + 线性探针只读取最后层 patch 特征；若结构引导改善的是中间层或局部细节，线性探针可能无法体现。
5. **"更难的任务"≠"更好的表征"**。让目标块落在边界上，确实提高了预测难度，但 I-JEPA 的表征质量由下游判别性决定，而非由预测难度决定。这是需要认真对待的负结果解释。
6. **实现层面的公平性问题**：C1/C2/C3 的上下文长度与 B 不完全相同（因为上下文必须避开目标块，目标块位置不同 → 可用上下文区域不同）。本轮报告会给出上下文可见 patch 数的实际分布，并说明其量级。

若出现 C3 ≈ C2 > B，则最可能的结论是"有偏采样"本身有益，而结构语义不是原因——这会直接改变下一阶段的优化方向。

---

## 8. 下一阶段如何根据实验证据改进方法？

按证据分支给出（而不是预先假定成功）：

**若 C1 或 C2 相对 B 呈稳定正向（且 C3 不支持单纯偏置解释）**
- 扩大预训练预算（例如 10k–20k updates），检验趋势是否持续；
- 用 ≥2 个预训练 seed 复现，排除单 seed 偶然性；
- 对 λ ∈ {2, 6, 12} 与 K ∈ {8, 16, 32} 做灵敏度分析；
- 把先验扩展到多任务（实例分割、作物/杂草分类）以检验泛化。

**若结果混合（部分指标升、部分降）**
- 做逐类别归因：Weed IoU 升而 Crop/Soil 降，提示目标块过度集中在小尺度杂草；
- 分离"块位置"与"上下文几何"的贡献（固定上下文分布，仅改目标位置）；
- 引入类别平衡的候选打分（例如按先验分位数分层采样，而非纯 softmax）。

**若 C2 ≈ C3 或无收益**
- 说明空间对齐的结构信息未被利用，优先改进**先验**而非采样器：例如学习式（自监督）显著性、或多尺度边界；
- 或转向"掩码尺度/长宽比"而非"掩码位置"作为自变量；
- 或直接检验"是否应该扩大训练预算/数据质量"而不是继续调 mask。

**若结果为负**
- 保留为负结果并报告：结构感知 mask 在同等预算下不优于随机多块；
- 修正假设：I-JEPA 的随机多块可能已经足够好，因为其目标块尺度（0.15–0.2 面积）本身就不小，"语义性"假设在农田图像上未必成立。

---

## 参考文献（本轮实际核对过条目）

1. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., Ballas, N. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
2. Kakogeorgiou, I., Gidaris, S., Psomas, B., Avrithis, Y., Bursuc, A., Karantzalos, K., Komodakis, N. *What to Hide from Your Students: Attention-Guided Masked Image Modeling.* ECCV 2022. [arXiv:2203.12719](https://arxiv.org/abs/2203.12719)
3. Li, G., Zheng, H., Liu, D., Wang, C., Su, B., Zheng, C., Heseltine, T., et al. *SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders.* NeurIPS 2022. [arXiv:2206.10207](https://arxiv.org/abs/2206.10207)
4. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
5. Bao, H., Dong, L., Piao, S., Wei, F. *BEiT: BERT Pre-Training of Image Transformers.* ICLR 2022. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
6. Grill, J.-B., Strub, F., Altché, F., et al. *Bootstrap your own latent: A new approach to self-supervised Learning.* NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
7. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
8. Weyler, J., Magistri, F., Marks, E., Chong, Y. L., et al. *PhenoBench — A Large Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain.* NeurIPS 2023 Datasets & Benchmarks. [arXiv:2306.04557](https://arxiv.org/abs/2306.04557)
9. Woebbecke, D. M., Meyer, G. E., Von Bargen, K., Mortensen, D. A. *Color indices for weed identification under various soil, residue, and lighting conditions.* Transactions of the ASAE, 38(1), 259–269, 1995. （ExG = 2G − R − B 的原始来源）

> 提交说明：上述条目的 arXiv 编号于本轮通过 arXiv API 实际核对（见 `tools/verify_refs.py` 的核对记录）；ExG 原始文献为期刊文章，未在 arXiv 核对。