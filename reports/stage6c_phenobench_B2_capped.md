# Stage 6C — Arm B2（capped no-backfill sampling，2000 updates）PhenoBench 结果

**arm label**：`stage6c_b2_capped_no_backfill`
**checkpoint**：`runs/stage6_b2_capped_no_backfill_2000/jepa-encoder-step2000.pth`
**checkpoint SHA-256**：`6e7a83c9156e556757016145fb804e2b67e7b94f5e8b45bda4ffd24cb5975a62`
**continued training**：2000 optimizer updates；官方 I-JEPA 目标；
Stage 5E.1 物理源 capped no-backfill 采样画像
**预训练数据**：R3v3_license_verified train；147,277 张；审计状态 **CONDITIONAL**

---

## 1. 直接实测值（每 seed，官方评价器 + 自实现 soil）

| seed | Soil IoU | Crop IoU | Weed IoU | mIoU | 训练 loss | probe 耗时 (s) | 峰值显存 (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 97.6951 | 82.37 | 40.21 | 73.43 | 0.060584 | 104.1 | 2639.3 |
| 1 | 97.6913 | 82.15 | 39.95 | 73.27 | 0.060502 | 103.9 | 2639.3 |
| 2 | 97.6763 | 82.16 | 33.87 | 71.23 | 0.060593 | 107.8 | 2639.3 |

## 2. 汇总计算值（3 个 probe seeds）

| 指标 | mean ± std |
| --- | ---: |
| Soil IoU | 97.6876 ± 0.0081 |
| Crop IoU | 82.2267 ± 0.1014 |
| Weed IoU | 38.0100 ± 2.9293 |
| mIoU | **72.6433 ± 1.0015** |
| 最终训练 loss | 0.060559 |
| probe 耗时 | 105.3 s（mean） |
| 峰值显存 | 2639.3 MiB |

- 训练更新数：11,000 / seed。
- 自实现与官方指标最大偏差：0.0050 pp（< 0.01 pp），逐 seed 均通过。
- val 预测完整性：3 seeds × 772 张，共 2,316 张 PNG。

## 3. 与 Arm A / Arm B1 的配对差值

| 差值 | Soil mean | Crop mean | Weed mean | mIoU mean | 方向（+/− seed） |
| --- | ---: | ---: | ---: | ---: | --- |
| B2−A | +0.08 | +0.77 | +2.03 | **+0.96** | soil 3/0，crop 3/0，weed 3/0，mIoU 3/0 |
| B2−B1 | +0.00 | +0.01 | −0.66 | **−0.21** | soil 2/1，crop 2/1，weed 0/3，mIoU 0/3 |

**B2 vs A**：3/3 个 seed 上 mIoU、crop、weed 全部高于 A，soil 几乎不变——
收益由 crop/weed 驱动，而非占比最大的 soil。

**B2 vs B1**：mIoU 均值 −0.21 pp（小于 B1−A 的 +1.17），且逐类方向**不一致**：
- soil 基本持平（2/1 微弱正/负，幅度 <0.01 pp）；
- crop 基本持平（2/1，幅度 ≤0.15 pp）；
- weed 3/3 为负，mean −0.66 pp，seed 2 偏差最大（−1.25 pp）。

因此 B2−B1 在 mIoU 上呈**轻微负向但不一致**：weed 稳定略低于 B1，而
soil/crop 几乎无差。B2−B1 差值的幅度（0.21 pp mIoU）远小于 probe seed 自身
的 std（B2 std 1.00，B1 std 0.89），属于探索性方向，**不能判定 capped
no-backfill 优于或劣于 natural 采样**。

## 4. 解释

- B2 相对 A 有与 B1 同方向、略小的正迁移（mIoU +0.96 vs +1.17）。
- B2 相对 B1 为轻微负向、类间方向不一致；因每种采样仅一个预训练 checkpoint，
  该结论**必须限定为探索性**，需独立预训练 seeds 验证。
- weed IoU 的 seed 间波动（std 2.93，seed 2 明显低于 seed 0/1）仍是主要
  不确定性；该模式在三臂的 seed 2 上均出现，指向 probe 训练随机性而非
  预训练差异。

## 5. 协议与产物

- 协议：与 Stage 6C 固定协议逐字段一致。
- 设备：物理 GPU1（`CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0`）。
- 特征缓存键：`stage6c_b2_capped_no_backfill__6e7a83c9156e5567`
  （label + checkpoint SHA[:16] 绑定）。
- 缓存于本 arm 结果核验后**精确删除**：
  `runs/stage6c_phenobench/cache/stage6c_b2_capped_no_backfill__6e7a83c9156e5567/`。

| 产物 | 路径 |
| --- | --- |
| 机器可读结果 | `runs/stage6c_phenobench/B2_capped/result.json` |
| 每 seed 预测 | `runs/stage6c_phenobench/B2_capped/seed{0,1,2}/predictions/semantics/*.png` |
| 官方评价器原始输出 | `runs/stage6c_phenobench/B2_capped/seed{0,1,2}/phenobench_eval_stdout.txt` |
| 控制台日志 | `runs/stage6c_phenobench/B2_capped.console.log` |
