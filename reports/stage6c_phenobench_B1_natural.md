# Stage 6C — Arm B1（natural sampling，2000 updates）PhenoBench 结果

**arm label**：`stage6c_b1_natural`
**checkpoint**：`runs/stage6_b1_natural_2000/jepa-encoder-step2000.pth`
**checkpoint SHA-256**：`19796cebc65e29a07eafd7222f5a642d4ef9f28c33f2e4f4f2b65e34a2bfde6b`
**continued training**：2000 optimizer updates；官方 I-JEPA 目标；自然源采样
**预训练数据**：R3v3_license_verified train；147,277 张；审计状态 **CONDITIONAL**

---

## 1. 直接实测值（每 seed，官方评价器 + 自实现 soil）

指标定义：crop / weed / mIoU 来自官方 `phenobench-eval --task semantics`
（val，772 张）；soil 来自自实现 `IoUAccumulator`（官方不打印 soil 行）。

| seed | Soil IoU | Crop IoU | Weed IoU | mIoU | 训练 loss | probe 耗时 (s) | 峰值显存 (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 97.6899 | 82.35 | 40.35 | 73.46 | 0.060963 | 77.6 | 2639.3 |
| 1 | 97.6937 | 82.29 | 40.53 | 73.50 | 0.060904 | 75.9 | 2639.3 |
| 2 | 97.6757 | 82.01 | 35.12 | 71.60 | 0.060976 | 63.2 | 2639.3 |

## 2. 汇总计算值（3 个 probe seeds）

| 指标 | mean ± std |
| --- | ---: |
| Soil IoU | 97.6864 ± 0.0078 |
| Crop IoU | 82.2167 ± 0.1482 |
| Weed IoU | 38.6667 ± 2.5089 |
| mIoU | **72.8533 ± 0.8864** |
| 最终训练 loss | 0.060948 |
| probe 耗时 | 72.2 s（mean） |
| 峰值显存 | 2639.3 MiB |

- 训练更新数：11,000 / seed（1,407 图 × 25 tile / batch 32 × 10 epochs）。
- 自实现与官方指标最大偏差：0.0049 pp（< 0.01 pp），逐 seed 均通过。
- val 预测完整性：3 seeds × 772 张，共 2,316 张 PNG，文件名与官方 val 列表一致。

## 3. 与 Arm A（历史同协议 baseline）的配对差值

Arm A 数值复用 `runs/stage3a_phenobench/armA/result.json`（已核实的同协议
reference，见 `reports/stage6c_phenobench_A_official.md`）。

| seed | ΔSoil | ΔCrop | ΔWeed | ΔmIoU |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.07 | +0.69 | +1.63 | +0.79 |
| 1 | +0.08 | +0.83 | +1.79 | +0.90 |
| 2 | +0.08 | +0.76 | +4.64 | +1.82 |
| **mean** | **+0.08** | **+0.76** | **+2.69** | **+1.17** |

**方向一致性**：3 个 probe seed 上 B1 的 mIoU、crop IoU、weed IoU 全部高于 A。
soil 仅 +0.08 pp，因此 mIoU 提升不是由占比最大的 soil 类驱动，而是由
crop（+0.76 pp）和 weed（+2.69 pp）驱动。

## 4. 解释

- 直接实测：B1 在 3/3 个 probe seed 上 mIoU 高于 A，mean 差值 +1.17 pp。
- 汇总计算：mean±std 见 §2；配对差值见 §3。
- **探索性信号**：这构成“2000-update 农业持续预训练在冻结线性探针下有初步
  迁移收益”的证据，但不能宣称自然采样策略已被证明优越——B1 只有一个预训练
  checkpoint，3 个 probe seeds 只衡量分割头随机性。
- weed IoU 的 seed 间波动（std 2.51）仍是主要不确定性来源；seed 2 的 weed
  明显低于 seed 0/1，与 A 的 seed 2 模式相似，属于 probe 训练随机性而非
  预训练差异。

## 5. 协议与产物

- 协议：与 Stage 6C 固定协议逐字段一致（见
  `reports/stage6c_phenobench_protocol.md`）。
- 设备：物理 GPU1（`CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0`）。
- 特征缓存键：`stage6c_b1_natural__19796cebc65e29a0`
  （label + checkpoint SHA[:16] 绑定）。
- 缓存于本 arm 结果核验后**精确删除**：`runs/stage6c_phenobench/cache/stage6c_b1_natural__19796cebc65e29a0/`。

| 产物 | 路径 |
| --- | --- |
| 机器可读结果 | `runs/stage6c_phenobench/B1_natural/result.json` |
| 每 seed 预测 | `runs/stage6c_phenobench/B1_natural/seed{0,1,2}/predictions/semantics/*.png` |
| 官方评价器原始输出 | `runs/stage6c_phenobench/B1_natural/seed{0,1,2}/phenobench_eval_stdout.txt` |
| 控制台日志 | `runs/stage6c_phenobench/B1_natural.console.log` |
