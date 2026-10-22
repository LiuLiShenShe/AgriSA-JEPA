# Stage 4B-3 — 500-step 训练试点

## 1. 协议

* 初始化：官方 `IN1K-vit.h.14-300e.pth.tar`（SHA-256 `0382013c…`），每个 Arm 独立初始化。
* 数据：`agrifield40k-v1_train.csv`（PROVISIONAL，12,567 张）。
* 调度：**2,000 optimizer updates** 的总调度，在第 **500** 步纯中断并保存。
  `max_optimizer_steps` 仍为 2,000，因此 LR/WD/EMA 轨迹与既有 B_500 完全一致。
* 有效 batch：micro-batch 16 × grad-accum 8 = 128；BF16；SDPA；AdamW；原始 Smooth-L1。
* 单卡 NVIDIA RTX A6000（`cuda:1`）。

## 2. 健康检查与真实更新次数

| Arm | 实际 updates | 调度总长 | 中断点 | loss first → last | loss finite | EMA 公式验证 | Peak GPU (MiB) | ms/step | 训练段 wall-clock (min) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| G0 | 500 | 2000 | 500 | 0.1895 → 0.1275 | True | True | 22718 | 249 | 27.2 |
| G1 | 500 | 2000 | 500 | 0.1862 → 0.1296 | True | True | 23159 | 261 | 40.6 |
| G2 | 500 | 2000 | 500 | 0.1889 → 0.1351 | True | True | 23093 | 265 | 36.6 |
| G3 | 500 | 2000 | 500 | 0.1957 → 0.1425 | True | True | 23094 | 256 | 32.7 |

## 3. 训练时的真实 Mask 诊断（主进程，num_workers=0）

| Arm | guided | fallback | invalid | 选中重叠 | 候选重叠 | prior s/batch | selection s/batch |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 253760 | 0 | 0.0 | 0.2423 | 0.3710 | 0.000 | 176.029 |
| G1 | 253760 | 0 | 0.0 | 0.3491 | 0.3373 | 7.115 | 250.339 |
| G2 | 253760 | 0 | 0.0 | 0.3432 | 0.3405 | 17.383 | 265.646 |
| G3 | 253760 | 0 | 0.0 | 0.3220 | 0.3257 | 18.307 | 256.918 |

## 4. Mask 生成开销（隔离基准，CPU）

| Arm | ms/batch | img/s | 相对 B 开销 | prior ms/batch | selection ms/batch | fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| B | 9.3 | 1714 | 0.0% | 0.0 | 0.0 | — |
| U | 65.3 | 245 | 599.8% | 0.0 | 54.6 | 0.000 |
| G0 | 67.2 | 238 | 620.5% | 0.0 | 55.9 | 0.000 |
| G1 | 111.9 | 143 | 1099.2% | 16.8 | 75.2 | 0.000 |
| G2 | 171.5 | 93 | 1737.9% | 76.7 | 69.7 | 0.000 |
| G3 | 190.7 | 84 | 1942.8% | 93.2 | 76.6 | 0.000 |

> 基准条件：cpu，batch 16，8 个 batch。训练中 `num_workers=0` 且 prior 在 GPU 上计算，因此实际开销低于该 CPU 隔离值；真实 step time 见第 2 节。

## 5. Resume 验证（从 G1 的 step-500 checkpoint 续跑到 506）

| 字段 | 值 |
|---|---|
| optimizer_steps_total | 506 |
| total_optimizer_steps_horizon | 2000 |
| interrupt_after_optimizer_steps | 506 |
| loss_finite | True |
| ema_formula_verified | True |

恢复源是 `jepa-latest.pth.tar`（包含 optimizer / scaler / RNG / loader generator / mask_collator_step / epoch_start_state）。
`jepa-step500.pth.tar` 在修复前不包含 `epoch_start_state`，因此只能用于评价；`save_step_snapshot` 已修复，之后的 step 快照也可直接续跑。


## 6. 未执行的内容

* 没有跑 2,000-step 的四臂全长训练：本轮是机制筛查，不是最终显著性验证。
* U 没有单独训练：全量离线审计显示 U 与 B 的 Target Union / Context / 结构覆盖
  分布等价（KS 检验见门禁），因此按协议降为诊断性实验。
* 没有用 PhenoBench Val 调 β；β=3.5 只由离线 Mask 几何选择。
