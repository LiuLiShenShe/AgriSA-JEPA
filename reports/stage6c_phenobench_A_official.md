# Stage 6C — Arm A（官方 I-JEPA ViT-H/14）baseline 复用报告

**结论**：复用历史 Stage 3A Arm A baseline，属于**已核实的同协议 reference**，
不是重新训练。全部机器可读证据齐全，且与当前 Stage 6C 协议逐字段一致。

---

## 1. 决策：复用，而非重跑

| 核查项 | 结果 |
| --- | --- |
| `runs/stage3a_phenobench/armA/result.json` 存在且可解析 | 是 |
| checkpoint SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`，与预期一致 |
| checkpoint 文件当前在盘 | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar`，10,358,004,345 字节，SHA 复算一致 |
| 协议字段 | 与 `configs/phenobench_armA_linear.yaml` 及 Stage 6C 协议逐字段一致（见 §3） |
| 3 个 probe seeds | 0 / 1 / 2 齐全 |
| 每 seed val 预测 | 772 / 772 张，文件名与官方 val 列表一致 |
| 官方评价器原始输出 | `seed{0,1,2}/phenobench_eval_stdout.txt` 齐全 |
| 自实现 vs 官方指标 | 最大偏差 0.0043 / 0.0047 / 0.0048 pp，均 < 0.01 pp |
| 数据版本与清单 | PhenoBench v1.1.0；清单 SHA `13fa2810…bd5d`；13,074 项逐文件尺寸校验 0 不符 |

因此**无需重跑 A**。重跑会重复消耗约 35 GB 特征缓存和数十分钟 GPU 时间，
且不会增加证据强度。若上述任一项缺失或不可验证，则应按协议用官方 checkpoint
在同一脚本下重跑至 `runs/stage6c_phenobench/A_official/`；本轮未触发该分支。

---

## 2. 历史 A 指标（直接实测值）

指标来源：官方 `phenobench-eval --task semantics`（val，772 张）。
soil 由自实现 `IoUAccumulator` 给出（官方评价器不打印 soil 行），
其 mIoU 恒等式与官方输出一致。

| seed | Soil IoU | Crop IoU | Weed IoU | mIoU | 训练 loss | probe 耗时 (s) | 峰值显存 (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 97.6186 | 81.66 | 38.72 | 72.67 | 0.064382 | 73.5 | 117.5 |
| 1 | 97.6104 | 81.46 | 38.74 | 72.60 | 0.064353 | 71.0 | 117.5 |
| 2 | 97.5964 | 81.25 | 30.48 | 69.78 | 0.064399 | 95.8 | 117.5 |
| **mean ± std** | **97.6085 ± 0.0092** | **81.4567 ± 0.1674** | **35.98 ± 3.8891** | **71.6833 ± 1.3462** | 0.064378 | 80.1 | 117.5 |

- 训练更新数：11,000（1,407 图 × 25 tile / batch 32 × 10 epochs）。
- mIoU 的 seed 间波动（std 1.35）几乎全部由 weed IoU 的波动（std 3.89）驱动；
  crop IoU 非常稳定（std 0.17），soil IoU 几乎不变。
- 平凡基线（全部预测 soil）：mIoU **29.97**。A 明显高于该基线。

---

## 3. 协议一致性证据

历史 A 的 `result.json.protocol` 与 Stage 6C 要求逐字段相同：

```json
{
  "backbone": "ViT-H/14, frozen, 224x224 tiles, 16x16 tokens, embed 1280",
  "head": "Linear(1280 -> 3) over patch tokens, nearest upsample x14",
  "tiling": "overlapping 224x224, stride 200, 5x5 per 1024 image, no padding, every pixel covered, overlaps logit-averaged",
  "optimizer": "AdamW",
  "lr": 0.001,
  "weight_decay": 0.0,
  "epochs": 10,
  "batch_size": 32,
  "seeds": [0, 1, 2],
  "loss": "patch cross-entropy, exact equivalent of nearest-upsampled pixel CE",
  "train_images": 1407,
  "val_images": 772
}
```

`tools/compare_stage6c_phenobench.py` 在聚合前重新断言该协议与 B1/B2 完全一致。

---

## 4. 证据边界

- 这是**历史复用 baseline**，不是本阶段重新生成的数字；其原始训练发生在
  Stage 3A。
- probe seeds 0/1/2 只衡量分割头/训练随机性；每个 arm 仍只有一个预训练 checkpoint。
- 历史 A 的 `metric_cross_check.rows` 中 soil 行因官方输出无 soil 行而为 NaN；
  比较脚本只对有限的 crop/weed/mIoU 偏差取最大值，不把 NaN 当作一致。
- 未使用 test split；未在 test 上调参。

---

## 5. 产物路径

| 产物 | 路径 |
| --- | --- |
| 机器可读结果 | `runs/stage3a_phenobench/armA/result.json` |
| 每 seed 预测 PNG | `runs/stage3a_phenobench/armA/seed{0,1,2}/predictions/semantics/*.png` |
| 官方评价器原始输出 | `runs/stage3a_phenobench/armA/seed{0,1,2}/phenobench_eval_stdout.txt` |
| 人读报告 | `reports/phenobench_armA_results.md` |
| 特征缓存（历史） | `runs/stage3a_phenobench/cache/A/` |
