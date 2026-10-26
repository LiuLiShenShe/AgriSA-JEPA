# Stage 6 — B2 Capped (no-backfill) 受控预训练结果

## 1. 运行事实（实测，非目标值）

| 项目 | 值 |
|---|---|
| 配置 | `configs/stage6_b2_capped_no_backfill_2000.yaml` |
| 输出目录 | `runs/stage6_b2_capped_no_backfill_2000` |
| 训练前 Git SHA | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981` |
| manifest | `data/stage6/stage6_r3v3_train_view_nvme.csv` |
| manifest SHA-256 | `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` |
| 初始化权重 SHA | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`（strict load） |
| 采样 | `profile_capped`（Stage 5E.1 `by_source` profile，no-backfill） |
| `cap_unit` / `cap_share` | `physical_source_id` / 0.5 |
| capped source / cap | `artemis_imagesafari` / 75,000 |
| `optimizer_steps_total` | **2000** |
| `micro_steps_total` | 15,992 |
| `optimizer_steps_per_epoch` | 869（capped epoch = 111,172 samples） |
| `max_optimizer_steps` | 2,000 |
| loss finite | true |
| `ema_formula_verified` | true |
| `encoder_updated` / `predictor_updated` | true / true |
| peak GPU memory | 23,355 MiB |
| step 时间均值 | 291.9 ms |
| 实际训练耗时 | 约 269 min（4.5 h） |
| GPU | RTX A6000（GPU0） |

## 2. no-backfill 规则如何落地（实测）

冻结 profile 的 cap 是相对 **nominal 150,000-exposure budget** 的 50% = 75,000。因为不回填，capped epoch 实际只发出 **111,171.64** 个样本，于是 artemis 的**实际 emitted 占比 = 75,000 / 111,171.64 ≈ 67.46%**，不是 50%。

实现位置：`src/datasets/stage6_sampling.py::ProfileExposureSampler`，由 `make_agrifield40k(sampler_spec=...)` 接入 `DataLoader`。整数 epoch（最大余数法）固定为 111,172 个样本，artemis 取整数 75,000；其余来源保持自然计数，赤字不回填（`capped_exposure(backfill=False)`）。该语义由 `tests/test_stage6_sampling.py` 全程锁定。

## 3. 训练曲线（直接来自 `jepa_r0.csv`）

| 指标 | 值 |
|---|---|
| loss 首次 / 末次 | 0.1942 / 0.1220 |
| loss 最小 / 最大 | 0.1022 / 0.2030 |
| loss 方向 | 单调下降（warmup 后 LR 余弦衰减至 1e-6） |

2,000 update 的 loss 下降仅证明优化器在收敛，不构成农业下游泛化提升的证据。

## 4. 实际来源曝光（来自 `source_exposure_ledger.csv`，逐样本记录，已去重）

| physical_source_id | emitted | share | capped nominal | 说明 |
|---|---:|---:|---:|---|
| artemis_imagesafari | 172,618 | **67.46%** | 75,000 / 111,172 | 整数 cap 75,000；占比 67.46% 非 50% |
| agrifield40k_pool | 42,756 | 16.71% | 18,577 | 自然计数，未回填 |
| agrifm_ref58 | 36,067 | 14.10% | 15,670 | 自然计数，未回填 |
| applebbch81 | 3,243 | 1.27% | 1,409 | 自然计数，未回填 |
| agrifm_ref11 | 895 | 0.35% | 389 | 自然计数，未回填 |
| agrifm_ref9 | 198 | 0.08% | 86 | 自然计数，未回填 |
| agrifm_ref1 | 95 | 0.04% | 41 | 自然计数，未回填 |
| **合计** | **255,872** | 100% | 111,172 × ~2.3 epoch | 2000 × 128 = 256,000，去重后 255,872 |

- emitted = 255,872 = 2000 × 128 − 128（末 epoch 末 batch 因 drop_last 丢弃），符合预期。
- 整数 cap 精确等于 75,000；B1 的对应计数为 194,141（自然 75.89%）。B2 把 artemis 从 **75.89%** 实际压到 **67.46%**（注意：这是相对 emitted total 的比例；相对 nominal 150,000 则为 50%）。
- HHI = 0.5031（接近 profile 的 0.5031），entropy = 0.9243（接近 profile 的 0.9242），证明 sampler 精确实现了冻结 profile。

> **口径提醒**：报告与对话中一律区分两个数字 —— nominal cap = 75,000 / 150,000 = **50%**；actual emitted share = 75,000 / 111,172 = **67.5%**。任何“实际发出样本中最大来源占比为 50%”的表述都是错误的。

## 5. 协议一致性

- 与 B1 差异**仅**在：arm 标识、采样策略、输出目录。
- 模型结构、mask、batch、grad_accum、LR/WD schedule、EMA、precision、SDPA、manifest、初始化权重全部相同（`tools/check_stage6_protocol.py` 与 `tools/compare_stage6_arms.py` 均通过）。
- 从**同一官方 checkpoint 独立初始化**，不从 B1 续训。

## 6. Checkpoint

- 步 500 / 1000 / 1500 / 2000 快照已保存（`jepa-step{N}.pth.tar` + `jepa-encoder-step{N}.pth`）。
- `jepa-latest.pth.tar`（完整训练状态）。
- 下游评估用的 encoder 快照：`jepa-encoder-step2000.pth`。

## 7. 结论边界

- 本 arm 完成 2,000-update 受控探索性预训练，loss 正常收敛，无 NaN/Inf，EMA 公式验证通过。
- 仅 1 个 pretrain seed，未做下游 probe；不把 loss 下降当作农业表征质量的证据。

【任务已结束】
