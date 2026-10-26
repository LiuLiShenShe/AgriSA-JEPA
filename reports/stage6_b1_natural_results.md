# Stage 6 — B1 Natural 受控预训练结果

## 1. 运行事实（实测，非目标值）

| 项目 | 值 |
|---|---|
| 配置 | `configs/stage6_b1_natural_2000.yaml` |
| 输出目录 | `runs/stage6_b1_natural_2000` |
| 训练前 Git SHA | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981` |
| manifest | `data/stage6/stage6_r3v3_train_view_nvme.csv` |
| manifest SHA-256 | `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` |
| 初始化权重 SHA | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`（strict load） |
| 采样 | natural（DistributedSampler, shuffle, seed=20261026） |
| `optimizer_steps_total` | **2000** |
| `micro_steps_total` | 15,996 |
| `optimizer_steps_per_epoch` | 1,151 |
| `max_optimizer_steps` | 2,000 |
| loss finite | true |
| `ema_formula_verified` | true |
| `encoder_updated` / `predictor_updated` | true / true |
| peak GPU memory | 23,355 MiB |
| step 时间均值 | 276.3 ms（GPU compute 段，不含 checkpoint I/O 偶发尖峰） |
| 实际训练耗时 | 约 261 min（4.35 h） |
| GPU | RTX A6000（GPU1；GPU0 被占用见 preflight 与最终报告） |

## 2. 训练曲线（直接来自 `jepa_r0.csv`）

| 指标 | 值 |
|---|---|
| loss 首次 / 末次 | 0.1904 / 0.1289 |
| loss 最小 / 最大 | 0.1000 / 0.2039 |
| loss 方向 | 单调下降（warmup 后 LR 余弦衰减至 1e-6） |

2,000 update 的 loss 下降**仅证明优化器在收敛**，不构成农业下游泛化提升的证据（见第 6 节）。

## 3. 实际来源曝光（来自 `source_exposure_ledger.csv`，逐样本记录）

| physical_source_id | emitted | share | 自然份额（manifest） |
|---|---:|---:|---:|
| artemis_imagesafari | 194,141 | 75.86% | 75.89% |
| agrifield40k_pool | 31,800 | 12.43% | 12.38% |
| agrifm_ref58 | 26,712 | 10.44% | 10.45% |
| applebbch81 | 2,410 | 0.94% | 0.94% |
| agrifm_ref11 | 666 | 0.26% | 0.26% |
| agrifm_ref9 | 139 | 0.05% | 0.06% |
| agrifm_ref1 | 68 | 0.03% | 0.03% |
| **合计** | **255,936** | 100% | 147,277 行 |

- emitted = 255,936 = 2000 × 128 − 64（epoch 末 drop_last 丢弃），符合预期。
- 实际份额与 manifest 自然份额一致到 0.05% 以内，说明 natural 采样如实复现了自然分布。
- HHI = 0.6018，entropy = 0.7703（与 manifest 自然集中度 0.6022 / 0.7698 一致）。

## 4. 协议一致性

- 与 B2 差异**仅**在：arm 标识、采样策略、输出目录。
- 模型结构、mask、batch、grad_accum、LR/WD schedule、EMA、precision、SDPA、manifest、初始化权重全部相同（`tools/check_stage6_protocol.py` 与 `tools/compare_stage6_arms.py` 均通过）。
- 两个 arm 从**同一官方 checkpoint 独立初始化**，B2 不从 B1 续训。

## 5. Checkpoint

- 步 500 / 1000 / 1500 / 2000 快照已保存（`jepa-step{N}.pth.tar` + `jepa-encoder-step{N}.pth`）。
- `jepa-latest.pth.tar`（完整训练状态，含 RNG / loader generator / mask collator step / manifest SHA）。
- 下游评估用的 encoder 快照：`jepa-encoder-step2000.pth`。

## 6. 结论边界

- 本 arm 完成 2,000-update 受控探索性预训练，loss 正常收敛，无 NaN/Inf，EMA 公式验证通过。
- 仅 1 个 pretrain seed，未做下游 probe；不把 loss 下降当作农业表征质量的证据。

【任务已结束】
