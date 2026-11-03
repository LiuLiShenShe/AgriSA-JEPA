# Stage 5A — GWHD 2021 检测协议（预注册）

> 本协议在训练开始前写入。所有 Arm 使用**完全相同**的输入、划分、Head、预算、
> 增强、种子与评价。唯一允许的差异是 Backbone 权重（A: 官方 I-JEPA；B: 农业 JEPA 2k）。
> 数据审计见 `reports/stage5a_gwhd_data_audit.md`。

## 1. 数据

| 项 | 取值 |
|---|---|
| 数据集 | Global Wheat Head Dataset 2021（Zenodo 5092309） |
| 许可 | CC-BY-4.0 |
| 划分 | **官方 train / val / test**，原样使用，不重划分 |
| 标签 | 单类别（wheat head）边界框，COCO 格式 |
| 保留策略 | 全部 GWHD 图像（含 train）禁止进入任何预训练语料 |

- 官方 CSV 只用于读取标签；图像名重复、标签图像缺失、未标注图像均为硬错误。
- 退化框与完全越界框在**训练时**丢弃，部分越界框裁剪到图像内；
  审计报告保留原始缺陷计数，不做静默修复。

## 2. 输入分辨率：为什么不是 224，也不是 1008

GWHD 图像为 1024×1024，麦穗是小目标。直接把 1024 缩到 224 会丢失小目标细节，
不可接受。两个候选方案在单张 RTX A6000（BF16 + SDPA，冻结编码器、无梯度）上实测：

| 方案 | 输入 | Token 网格 | 吞吐 | 每图编码次数 | 训练特征缓存（fp16） |
|---|---|---:|---:|---:|---:|
| 224 原尺寸 | 224 | 16×16 | 204 img/s | 1 | 25 tiles × 256 × 1280 × 2 B ≈ 81 MB/图 |
| **提高分辨率（选用）** | **518** | **37×37** | **37 img/s** | **1** | 无（在线前向） |
| 提高分辨率 | 700 | 50×50 | 18 img/s | 1 | 无 |
| 提高分辨率 | 1008 | 72×72 | 7.4 img/s | 1 | 无 |
| 224 tiling（stride 168） | 224×36 tiles | 每 tile 16×16 | 204/36 ≈ 5.7 图/s | 36 | ≈154 GB（train split） |

**正式协议：518×518（37×37 patch tokens），位置编码做双三次插值。**

理由：

1. 相对 224，线性细节提升 2.3×，token 数提升 5.3×；
2. 单次前向，训练/评价吞吐 37 img/s，20K iterations（batch 2）约 30 分钟/臂；
3. 1008 只提升 1.95× 线性细节，却慢 5×；tiling 慢 36×，且需要 ~154 GB 缓存，
   在 518 GB 可用磁盘上虽可放下，但会挤占数据与 checkpoint 空间；
4. 位置编码插值是 I-JEPA ViT 的标准做法，且只作用于冻结副本，不改变预训练代码。

**已知局限**：518 下麦穗约 15–40 px（1–3 个 patch），仍然偏小。
因此评价必须报告 COCO 的 `AP_small` / `AP_medium` / `AP_large`，
不得只报总体 AP 掩盖小目标问题。若 A/B 都接近 0，结论应是“该冻结分辨率不适合
小目标”，而不是“农业预训练无效”。

## 3. 检测结构（两臂完全相同）

```
Frozen ViT-H/14 @518 (37×37 tokens, D=1280)
        │ 1×1 lateral conv
        ▼
   P3 (37×37, 256) ──► P2 (74×74, 256)   [nearest ×2]
        │
        ├──► P4 (19×19, 256)             [maxpool ×2]
        └──► P5 (10×10, 256)             [maxpool ×2]
        ▼
Faster R-CNN (torchvision)
  RPN anchors: 4 levels × sizes (16,32,64,128) × ratios (0.5,1,2)
  ROI: MultiScaleRoIAlign(0,1,2,3 → 7×7, sampling 2)
  num_classes=2 (background + wheat head)
```

- Backbone：`requires_grad=False`，`eval()`；forward 使用 BF16 autocast，输出转 fp32。
- Neck：1×1 lateral + 4 个 3×3 smooth conv（256 通道），从头训练。
- Head：Faster R-CNN 默认 RPN/ROI head，从头训练。
- 可训练参数：**17.19 M**（Neck + RPN + ROI + box predictor）；冻结参数 ≈ 632 M。
- 两臂共享同一 `build_detector()`；有断言禁止 `backbone.encoder.*` 出现可训练参数。

## 4. 训练协议

| 项 | 取值 |
|---|---|
| 输入 | 1024×1024 → 518×518（双线性、antialias） |
| 归一化 | ImageNet mean/std |
| 增强 | 仅随机水平翻转（p=0.5），对框精确 |
| Batch | 2 图/iteration |
| 迭代 | 20,000（≈10 epochs of 4,000 images，与 Detectron2 常用 20K 对齐） |
| 优化器 | SGD，momentum 0.9，weight decay 1e-4，nesterov=False |
| LR | 0.0025，线性 warmup 500 iters，cosine 退火到 0 |
| 梯度裁剪 | 10.0 |
| 种子 | 训练 seed 0；资源允许时补 1、2 |
| 选模 | **固定预算，取最终 checkpoint**；不用 val/test 反复挑点 |
| 评估 | 每 2,000 iters 在 val 上评估；**test 只在最后评估一次** |

> 说明：20K iterations 是相对 Agri-FM+ 的缩小预算（他们用 Detectron2 20K、
> batch 2、ResNet-50），因此绝对 mAP 不可与论文数字直接比较；A/B 在同一协议下可比。

## 5. 指标

| 指标 | 定义 |
|---|---|
| AP50:95 | COCO IoU 0.50:0.95，101 点插值（主指标） |
| AP50 / AP75 | COCO |
| AP_small / AP_medium / AP_large | COCO 面积区间（原图像素） |
| AR100 | COCO maxDets=100 |
| Precision@0.5 / Recall@0.5 | score ≥ 0.5 时的贪心一对一匹配（IoU ≥ 0.5） |
| 延迟 | 每图 eval ms（同 batch=2、同设备） |
| 成本 | 训练秒数、峰值显存、可训练参数量 |

工具：`torchmetrics.detection.MeanAveragePrecision(backend='pycocotools')`。
不使用任何自定义 mAP 定义；若与原论文不同，只并列说明，不混用。

## 6. 隔离与泄漏控制

- GWHD 为下游 holdout：`data/registry/downstream_holdout_sources.csv` 明确列出；
- D1 候选语料构建时整来源排除 GWHD（`src/registry/freeze.py`）；
- 官方 train/val/test 的图像名与采集域（domain prefix）交叉检查，发现重叠即拒绝运行；
- 预训练语料（AgriField-40K v1 / PlantVillage）与 GWHD 的 SHA-256 / pixel hash /
  pHash 交叉筛查结果写入数据审计报告；
- 测试集只用于最终报告，不参与超参或 checkpoint 选择。

## 7. 公平性断言

`tools/stage5a_compare_gwhd.py` 在比较 A/B 前重新读取两份 `results.json`，
逐字段比较 `input_size / iterations / batch_size / lr / momentum / weight_decay /
warmup / eval_every / min_box_size / hflip / score_thresh / seed / architecture /
anchors / optimizer / scheduler`。任一不同即**拒绝比较**；backbone SHA 相同也拒绝。

## 8. 预注册的结论规则

- 只报告一次固定协议的结果，不因结果不理想而改分辨率/超参重新跑；
- 若 B < A：结论为“在该冻结特征 + 该检测头 + 该分辨率下，农业持续预训练没有
  检测收益”，并分析任务差异/小目标/冻结特征容量，而不是宣称算法失败；
- 若 B ≈ A：报告为“无显著差异”，不以小数点后噪声下结论；
- 若 B > A：仍需第二个检测数据集或独立预训练 seed 复核，才可写入论文主结论；
- 单训练 seed 的差值一律标注为探索性。
