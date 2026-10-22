# Stage 3B — 标准 JEPA 基线（Arm B）协议与结果

> `agrifield40k-v1` 仍为 **PROVISIONAL**；本节结果为**探索性**方法比较，不是论文定稿数据。

## 1. 目的

在进入结构感知 Mask 原型之前，先建立一个**与 Stage 3A 不同预算**的标准基线：

- Stage 3A 的 B 组只训练了 **500** 次 optimizer updates；
- Stage 3B 的 B 组按协议训练 **2,000** 次 optimizer updates，并采用 **2,000-step 的完整调度曲线**。

两者**不是**同一学习率轨迹，第 500 步也不能跨实验直接等价比较（协议 §4.2 明确要求）。

## 2. 配置

| 项目 | 取值 |
|---|---|
| Backbone | ViT-H/14，224×224，patch 14，16×16 grid |
| 初始化 | 官方 `IN1K-vit.h.14-300e.pth.tar` |
| 初始化 SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` |
| 数据 | `data/frozen/agrifield40k-v1_train.csv`（12,567 张） |
| manifest SHA-256 | `d63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435` |
| Micro-batch | 16 |
| 梯度累积 | 8 |
| 有效 batch | 128 |
| 优化器 | AdamW |
| LR | 2.0e-4（warmup 5%，线性衰减到 1.0e-6） |
| Weight decay | 0.04 → 0.4 |
| EMA | 0.996 → 1.0（2,000 步线性） |
| 精度 / 注意力 | BF16 / SDPA |
| Mask | 官方 `MaskCollator`，**未修改** |
| nenc / npred | 1 / 4 |
| 目标块尺度 / 长宽比 | 0.15–0.20 / 0.75–1.5 |
| 上下文块尺度 | 0.85–1.0 |
| `min_keep` | 10 |
| 总预算 | 2,000 optimizer updates |
| 保存点 | 500 / 2,000（另加 0 = 官方权重） |
| 训练 seed | 20261019 |

配置文件：`configs/stage3b_random_2000.yaml`。

## 3. 真实训练结果

| 指标 | 数值 |
|---|---:|
| 真实 optimizer updates | **2,000** |
| micro-steps | 15,860 |
| 每个 epoch 的 updates | 99 |
| 实际 epoch 数 | ~21 |
| Wall clock | **6,622 s（1 h 50 min）** |
| 每 update 耗时 | 3.31 s |
| 吞吐（effective batch 128） | 38.7 images/s |
| 纯计算 step 时间（均值） | 267.3 ms |
| GPU 峰值显存 | 23,298 MiB（22.8 GiB） |

> 完整四组对照见 `stage4a_training_benchmark.md`。

## 4. 正确性核查

| 检查项 | 结果 |
|---|---|
| 官方初始化精确加载（encoder / predictor / target encoder） | 389 / 152 / 389 张量，全部匹配 |
| 初始化 SHA-256 校验 | 通过 |
| manifest SHA-256 校验 | 通过 |
| Loss 有限 | 是（首次 0.1838，末次 0.0970） |
| EMA 公式验证 | 通过（`ema_formula_verified: true`） |
| encoder / predictor 均被更新 | 是 |
| 预算精确到达 2,000 | 是（`optimizer_steps_this_run: 2000`） |
| 第 500 步快照来自 2,000-step 调度 | 是（同一 run 中途保存） |
| 保存快照未重置调度 / 优化器 / 随机流 | 是（快照在 update 之后写入，不触碰状态） |
| 是否静默复用旧目录 | 否（目录新建） |

## 5. 产出文件

| 文件 | 用途 | 大小 |
|---|---|---:|
| `jepa-step500.pth.tar` | 可恢复完整 checkpoint @500 | 10.36 GB |
| `jepa-encoder-step500.pth` | 轻量评价快照 @500 | 2.52 GB |
| `jepa-latest.pth.tar` | 最终完整 checkpoint @2000 | 10.36 GB |
| `jepa-encoder-step2000.pth` | 轻量评价快照 @2000 | 2.52 GB |
| `jepa_r0.csv` | 逐 micro-step 日志 | 578 KB |
| `training_verification.yaml` | 机器可读核查记录 | 1.6 KB |

关键 SHA-256：

- `jepa-latest.pth.tar` = `9bbefabdd0538438273b9006a7fdd18eff613af088ea8ac73d641b14381472ae`
- `jepa-encoder-step2000.pth` = `1c4e1c484d89245ea3702c9603c62355bdfcd07be8594df020e6e3cf4d94d23f`

## 6. 与 Stage 3A 的关系

- Stage 3A 的 500-step B 组**不复用**、**不续训**。Stage 3B 的 B 组从官方初始化重新开始，独立完成 2,000 步。
- Stage 3A 的 A 组（官方权重）结果按要求复用，不重新训练。

## 7. 关于训练 loss 的说明

本轮 B 组的训练 loss 从 0.184 降到 0.097。**该数值不用于与其他 Arm 比较**：不同 Mask 策略改变了预测目标的难度，loss 更低不等于表征更好（协议 §10.4）。所有方法的优劣判断只基于 PhenoBench 下游指标。