# Arm A — 官方 I-JEPA ViT-H/14 冻结骨干 + 线性分割头（PhenoBench semantics）

**结论：官方 ImageNet I-JEPA ViT-H/14 的冻结特征在 PhenoBench 语义分割上
mIoU = 71.68 ± 1.35（3 seeds）。**

| 指标 | 值（mean ± std, 3 seeds） | 每个 seed |
| --- | --- | --- |
| **mIoU** | **71.68 ± 1.35** | 72.67 / 72.60 / 69.78 |
| Soil IoU | 97.61 ± 0.01 | 97.6186 / 97.6104 / 97.5964 |
| **Crop IoU** | **81.46 ± 0.17** | 81.66 / 81.46 / 81.25 |
| **Weed IoU** | **35.98 ± 3.89** | 38.72 / 38.74 / 30.48 |

- 指标来源：官方 `phenobench-eval --task semantics`（val 划分，772 张）。
- 自实现指标与官方指标在每个 seed 上最大偏差 **< 0.005 个百分点**，
  见 `metric_cross_check`（`runs/stage3a_phenobench/armA/result.json`）。
- 平凡基线（全部预测 soil）：mIoU **29.97**。该结果显著高于基线。

## 设置

| 项 | 值 |
| --- | --- |
| 骨干 | ViT-H/14，**冻结**，官方 `IN1K-vit.h.14-300e.pth.tar` |
| 骨干 SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` |
| 农业持续预训练 | 无 |
| 分割头 | `Linear(1280 → 3)`，**3,843** 参数 |
| 输入 | 1024×1024 → 5×5 个 224×224 tile（stride 200，无 padding，全覆盖） |
| 优化器 / LR / WD | AdamW / 1e-3 / 0.0 |
| batch / epochs | 32 tiles / 10 |
| seeds | 0, 1, 2 |
| 训练图片 | 1,407（PhenoBench 官方 train） |
| 评价图片 | 772（PhenoBench 官方 val，未参与训练） |
| GPU | 单张 RTX A6000，未见峰值显存 117.5 MiB（特征已缓存） |
| Probe 训练耗时 | 73.5 / 71.0 / 95.8 秒 |
| 最终训练损失 | 0.0644（3 seeds 一致） |

## 产物

- `runs/stage3a_phenobench/armA/result.json` — 机器可读结果（含协议、SHA、逐 seed 指标）
- `runs/stage3a_phenobench/armA/seed{0,1,2}/predictions/semantics/*.png` — 官方格式预测图
- `runs/stage3a_phenobench/armA/seed{0,1,2}/phenobench_eval_stdout.txt` — 官方评价器原始输出
- 特征缓存：`runs/stage3a_phenobench/cache/A/`（fp16，每图 25 tile）

## 备注

- 该 arm 是对照组：它回答"未经任何农业持续预训练，官方 I-JEPA ViT-H 的特征
  对农业语义分割有多好"。
- 训练损失 0.064 的量级由 soil 像素占比（约 88%）主导，不能单独反映
  crop/weed 质量；mIoU 与 per-class IoU 才是结论依据。
