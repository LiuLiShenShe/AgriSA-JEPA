# Arm B — 农业持续预训练 I-JEPA ViT-H/14（500-step）+ 线性分割头（PhenoBench semantics）

**结论：完成 500 次农业持续预训练更新的 ViT-H/14，冻结特征在 PhenoBench
语义分割上 mIoU = 73.13 ± 1.09（3 seeds），比官方 I-JEPA 对照组高
+1.45 个百分点（按 seed 配对）。**

| 指标 | 值（mean ± std, 3 seeds） | 每个 seed | 相对 Arm A |
| --- | --- | --- | --- |
| **mIoU** | **73.13 ± 1.09** | 74.35 / 73.34 / 71.71 | **+1.45** |
| Soil IoU | 97.70 ± 0.003 | 97.7042 / 97.7074 / 97.6997 | +0.095 |
| **Crop IoU** | **82.38 ± 0.16** | 82.60 / 82.30 / 82.24 | **+0.92** |
| **Weed IoU** | **39.32 ± 3.12** | 42.75 / 40.01 / 35.20 | **+3.34** |

- 指标来源：官方 `phenobench-eval --task semantics`（val 划分，772 张）。
- 自实现指标与官方指标在每个 seed 上最大偏差 **< 0.005 个百分点**。
- 平凡基线（全部预测 soil）：mIoU **29.97**。

## 设置

| 项 | 值 |
| --- | --- |
| 骨干 | ViT-H/14，**冻结**，`runs/stage3_armB_pilot_500/jepa-latest.pth.tar` |
| 骨干 SHA-256 | `57181b8a65a1edf741b26d989e803c7640465393666308e257444ee8c31957e9` |
| 农业持续预训练 | **500 optimizer updates**，标准 I-JEPA 目标 |
| 预训练数据 | `agrifield40k-v0`（**PROVISIONAL**），12,447 张 |
| 分割头 | `Linear(1280 → 3)`，**3,843** 参数（与 Arm A 完全相同） |
| 输入 | 1024×1024 → 5×5 个 224×224 tile（stride 200，无 padding，全覆盖） |
| 优化器 / LR / WD | AdamW / 1e-3 / 0.0 |
| batch / epochs | 32 tiles / 10 |
| seeds | 0, 1, 2 |
| 训练 / 评价图片 | 1,407 / 772 |
| Probe 训练耗时 | 99.6 / 68.9 / 67.2 秒 |
| 最终训练损失 | 0.0590（3 seeds 一致；Arm A 为 0.0644） |

## 结果解读（必须与下一节一起读）

1. **crop IoU 的 +0.92 是最稳定的信号**：两个 arm 的 seed 间标准差都只有
   0.16–0.17，配对差值 +0.94 / +0.84 / +0.99，方向一致。
2. **weed IoU 的 +3.34 相对幅度最大**（35.98 → 39.32，约 +9% 相对提升），
   但 seed 间波动大（配对差值 +4.03 / +1.27 / +4.72），不能单独作为强证据。
3. **mIoU +1.45** 在 3 个 seed 上全部为正（+1.68 / +0.74 / +1.93）。
   n=3、df=2 的配对 t 只应视为描述性，不构成显著性结论。
4. **这不是"结构感知 Mask 有效"的证据。** Arm B 使用的是与官方完全相同的
   随机多块掩码，差异只来自 500 步农业持续预训练。结构感知掩码尚未实现。
5. **数据版本不一致**：Arm B 训练于 v0；v1 是 Stage 2D 之后才建立的。
   因此这是探索性结果，不能当作基于最终冻结 v1 的公平最终实验。

## 可能原因分析（按证据强弱）

| 可能原因 | 支持/反对 |
| --- | --- |
| 农业领域特征确实被改善 | crop/weed 的提升方向一致，且 crop 配对差极稳定 |
| 训练步数不足 | 仅 500 次 update（约 5.1 个 v0 epoch），提升幅度小；更像"轻微的领域适配"而非充分预训练 |
| 预训练数据分布 | v0 中 LUCASVision 占 71.8%，是田间覆盖图像；PhenoBench 是近地俯拍，分布仍有差距 |
| 来源集中 | v0 的 ACRECropWeed 仅 671 张，且 v1 修复后全部在 train；农业近地视角样本占比低 |
| 特征层选择 | 本轮只用最后一层 post-LN 特征；中间层可能对细粒度分割更合适（未实验） |
| 224×224 patch 表征分辨率 | 14×14 像素/patch，细杂草边界天然受限；weed IoU 绝对值低与此一致 |
| 掩码/目标区域特性 | 未涉及；两 arm 掩码相同（Arm A 不训练） |

## 产物

- `runs/stage3a_phenobench/armB/result.json`
- `runs/stage3a_phenobench/armB/seed{0,1,2}/predictions/semantics/*.png`
- `runs/stage3a_phenobench/armB/seed{0,1,2}/phenobench_eval_stdout.txt`
- 特征缓存：`runs/stage3a_phenobench/cache/B/`