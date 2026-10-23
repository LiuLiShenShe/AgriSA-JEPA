# Stage 5B — 后续 JEPA 预训练计划（Stage 5C 预备）

> 配置：`configs/stage5c_field_jepa_candidate.yaml`
> 本轮**没有运行正式长训练**，只做了数据读取与配置兼容性 smoke test。
> 状态：Field-D1 仍是 CANDIDATE，启动正式训练前必须先通过冻结门禁。

## 1. 固定项（所有实验臂一致）

| 项目 | 取值 |
|---|---|
| 初始化 | 官方 I-JEPA ViT-H/14 `IN1K-vit.h.14-300e.pth.tar`（SHA-256 `0382013c…aae22`） |
| Patch / 分辨率 | 14 / 224 |
| 预测器 | depth 12，dim 384 |
| Mask | 官方随机多块（`src/masks/random.py`，未修改） |
| 损失 | Smooth-L1 |
| EMA | 0.996 → 1.0 |
| 精度 | BF16 |
| 优化器 | AdamW，lr 2e-4，wd 0.04 |
| 有效 batch | 16 × grad_accum 8 = **128** |
| 硬件 | 单卡 A6000 48 GB（`CUDA_VISIBLE_DEVICES=1`） |

## 2. 实验矩阵

| Arm | 预训练语料 | 采样 | 目标 |
|---|---|---|---|
| A | 官方 I-JEPA，无农业继续训练 | — | 通用基线 |
| B0 | `agrifield40k-v1`（12,567 train） | natural | 历史农业基线 |
| B1 | Field-D1（15,118 train） | natural | 新田间来源是否带来增益 |
| B2 | Field-D1 | source_capped(cap=0.5) | 来源分布的影响 |
| B3 | Mixed-D1（条件满足后） | natural | 受控图像加入的影响 |

B2 的采样器**尚未实现**。`configs/stage5c_field_jepa_candidate.yaml` 中的
`data.sampling: natural` 是唯一已接线取值；在采样器实现并测试之前，不允许用
`source_balanced` / `source_capped` 启动训练。

## 3. 预算与公平性

统一预算：**2,000 optimizer updates**，有效 batch 128 → **256,000 次图像曝光**。

| 语料 | 唯一图像 | 曝光 / 唯一图像（epoch 等价） |
|---|---:|---:|
| v1 (B0) | 12,567 | 20.4 |
| Field-D1 (B1, natural) | 15,118 | 16.9 |
| Field-D1 (B2, capped) | 15,118（加权后等效） | 依权重而定 |

关键区别，必须在论文中分开陈述：

1. **相同 optimizer updates**：B0 与 B1 都跑 2,000 步，但 B1 每张图被看到的次数更少。
   如果 B1 更好，不能直接归因于"数据更多"。
2. **相同曝光次数**：两者曝光次数本来就相同（256k）；差别在"每张唯一图像的曝光"。
3. **相同 epoch 数**：若要让 B1 也跑 20.4 epoch，需要 2,420 步左右。这属于另一个对照，
   本轮不预设哪个更公平；两个都跑才说得清。
4. 不允许把"更多更新次数"带来的收益写成"更多数据"的收益。

## 4. 数据规模分析

- 当前 Field-D1 train 15,118 张，唯一来源 8 个，最大来源 59.1%。
- 新增 3,138 张只占 20.2%，无法把 LUCASVision 稀释到 50% 以下。
- 若目标是把最大来源压到 50%，需要约 **11,200** 张等效新增（`lucas` 11,183 →
  在总量约 22,400 时占 50%），即再增加约 8,000 张合格田间图。
- 若目标是来源均衡（`source_balanced`），不需要新增数据，但小来源会被重复采样；
  `sorghum_weed` 的 172 张会被重复约 81 次，这是必须报告的偏差。
- 因此 Field-D2 的目标应是"再接入 8,000–15,000 张、至少 4 个新来源的合格田间 RGB"，
  而不是追逐 Agri-FM+ 的 147K 数字。来源多样性优先于绝对数量。

## 5. 下游验证

| 任务 | 数据 | 指标 | 状态 |
|---|---|---|---|
| 语义分割 | PhenoBench | mIoU / mAcc | Stage 3A/4A 已有协议与结果 |
| 目标检测 | GWHD 2021 | mAP50 / mAP50-95 | Stage 5A 已有 A/B 协议与结果 |
| 检测（待接入） | PlantDoc | mAP | 数据未在本机 |
| 实例分割（待接入） | GWHD Instance Segmentation | mask mAP | 数据未在本机 |

每个下游任务固定：线性/轻量 head、相同标注比例、相同 seed 集合、相同评价脚本。
新增 Field-D1 的作用必须通过 A vs B0 vs B1 的**同一协议**比较来验证，不能只看
预训练 loss。

## 6. 本轮已完成的兼容性检查

`tools/stage5b_summarize_corpus.py --smoke-test`：

- Field-D1 train manifest 可被 `AgriFieldDataset` 读取；
- 单样本返回 `(PIL image, sample_id)`；
- 经 `Resize(224) + ToTensor` 后 batch 形状为 `[2, 3, 224, 224]`，dtype `float32`；
- v1 相对路径通过 `root_path=/data/datasets/AgriField-40K` 解析成功；
- 新来源的绝对路径直接可用。

未执行：真实的 mask collator + 模型前向（需要 GPU 且属于 Stage 5C）。本轮不做。

## 7. 启动 Stage 5C 前的前置条件

1. Field-D1 从 CANDIDATE 晋升（人工确认 3,138 张新图的域标签，或按来源签字）。
2. `weedcrop` 的处置有明确结论（弃用或改判后重新审核）。
3. 为 Field-D1 重建图像缓存（`tools/build_image_cache.py`；现有 v1 缓存按 manifest
   SHA-256 校验，必然不匹配）。
4. 实现并测试 `source_capped` 采样器，否则 B2 只能是计划。
5. 确认发布用 manifest 的 SHA-256 已重算并写进配置。
