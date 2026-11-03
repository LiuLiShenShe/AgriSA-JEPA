# PhenoBench 下游语义分割评价协议（Stage 3A）

**目的**：在完全相同、冻结的评价协议下，比较两套 ViT-H/14 特征对农业语义分割
的适用性。

| | Arm A（对照） | Arm B（处理） |
| --- | --- | --- |
| 骨干 | ViT-H/14 | 同一 ViT-H/14 |
| 权重 | 官方 `IN1K-vit.h.14-300e.pth.tar` | `runs/stage3_armB_pilot_500/jepa-latest.pth.tar` |
| 农业持续训练 | 无 | 500 次 optimizer update（标准 I-JEPA 目标） |
| 训练数据版本 | — | `agrifield40k-v0`（PROVISIONAL） |
| 骨干是否冻结 | 是 | 是 |

配置文件：`configs/phenobench_armA_linear.yaml`、`configs/phenobench_armB_linear.yaml`。
`tools/check_arm_protocol.py` 断言两份配置**除骨干标识外逐字段相同**。

> **数据版本不一致的声明（必须引用）**
> Arm B 的 500-step 检查点是在 v0 上训练的；Stage 2D 之后才建立 v1。
> 因此本轮对比是 **探索性诊断**，不是"在同一最终冻结数据版本上"的公平最终实验。
> 若要正式比较，必须从同一官方初始化、在统一最终数据版本上重新训练 Arm B。

## 1. 任务与类别口径

- 任务：PhenoBench Semantic Segmentation。
- 类别：`0 = soil/background`、`1 = crop`、`2 = weed`。
- 官方标签 id 还包含 `3 = partial crop`、`4 = partial weed`。
  官方评价器 `convert_partial_semantics` 把 **3→1、4→2** 后再计分，
  **不忽略**部分可见类别。本协议对训练目标施行同一映射。
- 官方指标定义来源：`phenobench/evaluation/evaluate_semantics.py`
  （`MulticlassJaccardIndex(num_classes=3, average=None)`，结果 ×100）。
- **不合并类别、不替换官方指标、不静默忽略任何类别。**

## 2. 数据

| | 用途 | 数量 |
| --- | --- | ---: |
| PhenoBench v1.1.0 train | 训练线性分割头 | 1,407 |
| PhenoBench v1.1.0 val | 评价（不参与训练） | 772 |
| PhenoBench v1.1.0 test | 不使用（无公开标注） | 693 |

- 只使用 `images` 与 `semantics`。
- 官方 val **绝不用于训练**。
- PhenoBench 图像**不进入** JEPA 自监督预训练。

## 3. 分辨率适配（224×224 Tiling + Stitching）

ViT-H/14 在 224×224 预训练（16×16 = 256 个 patch token，token 维度 1280），
而 PhenoBench 图像为 1024×1024。**不把整图压缩到 224**。

规则（所有 arm、所有图像、所有运行完全一致）：

```
n      = ceil(size / tile)              = ceil(1024/224) = 5
stride = ceil((size - tile) / (n - 1))  = ceil(800/4)    = 200
offsets = [0, 200, 400, 600, 800]       (最后一块的右/下边界正好 1024)
```

- 每张图 **5×5 = 25 块** 224×224 tile；
- **无 padding、无裁剪**：所有 tile 完整位于图像内，且最后一块正好贴边；
- 相邻 tile 重叠 **24 像素**；
- **每个像素至少被一块 tile 覆盖**（不存在未预测区域）。

拼接：
1. 每块 tile 的 16×16 patch logits 最近邻上采样 ×14 → 224×224；
2. 重叠像素对所有覆盖它的 tile 的 logits 取平均；
3. 取 argmax 得到 1024×1024 语义图。

因此自实现指标与官方指标都在**完整 1024×1024** 上计分，区域完全一致。

**与全图多尺度密集预测的差异**：本方案是单尺度 224 patch 网格的拼接，
不使用 FPN/多尺度上下文，也不做跨 tile 的注意力交互；它保留了原始像素分辨率，
但相邻 tile 之间没有信息交换。这属于第一版可解释基线，不是密集预测架构。

## 4. 分割头

- 结构：**Linear(1280 → 3)**，逐 patch token 分类；
- 参数量：3,843（1280×3 + 3）；
- 所有 arm 使用完全相同的头，不存在为某一 arm 单独调整解码器的情况；
- 本轮**不做**轻量 Decoder、不做 Full Fine-tuning。

### 损失

最近邻上采样后逐像素交叉熵。因为同一 patch 的 14×14 像素共享 logits，该损失
可精确写成 patch 级类别直方图形式（`src/downstream/probe.py: patch_ce_loss`）：

```
loss = Σ_patch Σ_c count[patch,c] · −log_softmax(logits_patch)[c] / Σ count
```

数学上与原式**完全等价**（测试 `test_patch_loss_equals_upsampled_pixel_loss`
同时校验 loss 与梯度），但不需要构造 10 GB 的 `[N,224,224]` 标签张量。

## 5. 训练协议（两 arm 完全相同）

| 项 | 值 |
| --- | --- |
| 骨干 | 冻结，`eval()`，`requires_grad=False` |
| 输入 | 224×224，ImageNet mean/std 归一化 |
| 优化器 | AdamW |
| 学习率 | 1e-3（固定，无调度） |
| weight decay | 0.0 |
| batch size | 32 tiles |
| epochs | 10（每 epoch 全部训练 tile 一遍） |
| 随机种子 | 0, 1, 2 |
| 数据增强 | 无（特征已缓存，评估协议固定） |
| 精度 | 特征提取用 BF16，head 训练用 FP32 |
| 设备 | 单张 RTX A6000 |

每个 epoch 的 tile 顺序由 seed 决定；同一 seed 下两 arm 的 tile 顺序相同。
Probe 只在 `train` 划分上训练，`val` 只做前向。

## 6. 指标

必须输出：

- mIoU（3 类平均）
- 每类 IoU：soil / crop / weed
- 训练损失曲线
- 模型参数量
- GPU 显存峰值
- Probe 训练耗时

**官方评价器**：

```bash
phenobench-eval --task semantics \
  --phenobench_dir /data/datasets/PhenoBench/PhenoBench-v110 \
  --prediction_dir <run>/predictions --split val
```

官方工具遍历 **全部 772 张** GT 并 `zip` 预测文件；因此本轮在跑官方评价时
**必须为全部 772 张 val 图像生成预测**，不能只预测子集（只预测子集时官方工具
`zip` 静默截断，产出的数字只代表子集，不能用于报告）。

**交叉校验**：自实现 `IoUAccumulator` 与官方 `MulticlassJaccardIndex`
在同一预测图上逐类比较，差值必须 < 0.01 个百分点；若不同，先调查原因，
**不得选用其中较好的一个**。

## 7. 平凡基线

val 的类别像素占比为 soil 89.92% / crop 9.55% / weed 0.53%。
"全部预测 soil" 得到 soil IoU = 89.92、crop IoU = 0、weed IoU = 0，
**mIoU = 29.97%**。任何结果都必须与该基线比较。

## 8. 执行顺序

1. **数据 Smoke Test**：读取真实图像与标签，验证 tile/重建尺寸、官方评价器可运行。
2. **小样本过拟合**：16 张训练图，30 epochs，确认监督路径正确、训练损失下降。
   （不是正式结果）
3. **完整 train 划分 Frozen Probe**：Arm A / Arm B 各 3 个 seed。
4. **官方 val 评价**：自实现 + 官方交叉校验。
5. 汇总 `reports/phenobench_armA_results.md` / `armB_results.md` / 对比表。

## 9. 已知限制（写入结果报告）

- 单尺度 224 tile 拼接，tile 间无注意力交互；
- 线性头表达力有限，指标低于全监督分割网络属预期；
- Arm B 仅 500 次 update，且训练于 v0（PROVISIONAL）；
- 类别极度不平衡（weed 仅 0.50% 像素），patch 级预测对细杂草召回天然不利；
- 未使用 PhenoBench 的 test 划分。

## 复现

```bash
# 1) 特征缓存（每 arm 一次）
./.venv/bin/python -m src.downstream.phenobench_probe --arm A --stage features
./.venv/bin/python -m src.downstream.phenobench_probe --arm B --stage features

# 2) 训练 + 评价（逐 seed）
./.venv/bin/python -m src.downstream.phenobench_probe --arm A --stage train_eval --seed 0
# ...

# 3) 协议一致性
./.venv/bin/python -m tools.check_arm_protocol
```