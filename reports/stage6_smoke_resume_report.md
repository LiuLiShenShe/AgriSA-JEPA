# Stage 6 — 真实图像 smoke 与 checkpoint resume 报告

**本文件的结果不属于 B1/B2 正式对照**，只用于证明 Stage 6 数据管线与中断/恢复机制在**真实 R3v3 图像**上可运行。

## 1. 运行设置

| 项目 | 值 |
|---|---|
| 配置（part 1） | `configs/stage6_smoke_resume.yaml` |
| 配置（part 2） | `configs/stage6_smoke_resume_continued.yaml` |
| 采样器 | `profile_capped`（比 natural 路径更复杂，选它做 smoke） |
| 训练预算 | 20 optimizer updates（两个 part 共用的 schedule horizon） |
| part 1 中断点 | `interrupt_after_optimizer_steps: 10` |
| part 2 | `load_checkpoint: true`，无中断点，跑满 20 |
| 有效 batch | 16 × 8 = 128 |
| `num_workers` | 0（本阶段只声明 0 worker 的 resume 一致性） |
| 图像 | 真实 R3v3 train 派生视图（NVMe 逐字节镜像） |
| GPU | RTX A6000（GPU0；GPU1 被外部任务占用，见 preflight） |

`max_optimizer_steps=20` 是 smoke 自己的 horizon，part 1 与 part 2 完全相同，因此 resume 后的 LR/WD/EMA 位置可以精确接续。正式 B1/B2 另用 2,000 的 horizon，两者不混用。

## 2. Part 1：10 次 optimizer update 后中断

实测（来自 `runs/stage6_smoke_resume/training_verification.yaml` 与 console）：

| 指标 | 值 |
|---|---|
| `optimizer_steps_total` | 10 |
| `micro_steps_total` | 80 |
| `optimizer_steps_per_epoch` | 869 |
| loss（首次 / 末次） | 0.194146 / 0.238717 |
| `loss_finite` | true |
| `ema_formula_verified` | true |
| `encoder_updated` / `predictor_updated` | true / true |
| peak GPU memory | 22,813 MiB |
| micro-step 平均耗时 | 268.9 ms |
| 实际 emitted 样本 | 1,280（10 × 128） |
| 中断日志 | `Interrupted at interrupt_after_optimizer_steps=10; saving and exiting (schedule horizon unchanged).` |
| checkpoint | `jepa-latest.pth.tar`（10.3 GB） |

中断发生在 epoch 中途：`epoch_itr=80`，`epoch_start_state` 已保存。

## 3. Part 2：从 checkpoint 恢复到总计 20 次 update

| 指标 | 值 |
|---|---|
| 恢复日志 | `Replaying 80 already-consumed micro-steps of epoch 1` |
| `optimizer_steps_total` | **20** |
| `micro_steps_total` | **160** |
| `micro_steps_this_run` | 80 |
| loss（首次 / 末次） | 0.233284 / 0.218366 |
| `loss_finite` / `ema_formula_verified` | true / true |
| 结束日志 | `Reached max_optimizer_steps=20; stopping.` |
| peak GPU memory | 约 22.3 GB |

## 4. 无重复、无丢失 update

逐个 micro-step 比对两份 CSV 的 `itr` 序列：

| 文件 | 行数 | itr 范围 |
|---|---:|---|
| `runs/stage6_smoke_resume/jepa_r0.csv` | 80 | 0 – 79 |
| `runs/stage6_smoke_resume_continued/jepa_r0.csv` | 80 | 80 – 159 |
| 合并 | 160 | 0 – 159，连续无缺口 |

同时：

- part 1 末行 itr=79，part 2 首行 itr=80，**没有重复执行**；
- 恢复后重放的 80 个 micro-step 被丢弃（不写日志、不计入 `micro_steps_this_run`）；
- `optimizer_steps_total=20` 与 20 次 update 的预算一致。

## 5. checkpoint 内容核对

恢复与导出测试读取 `jepa-latest.pth.tar`，确认包含：

`encoder`、`predictor`、`target_encoder`、`opt`、`scaler`、`epoch`、`epoch_itr`、`epoch_start_state`（含 `rng_state` / `loader_generator_state` / `mask_collator_step`）、`optimizer_steps`、`micro_steps`、`grad_accum`、`config`、`hardware`、`git_commit`、`pretrained_init`、`rng_state`、`loader_generator_state`、`mask_collator_step`、`manifest_sha256`。

`tests/test_stage6_checkpoint.py` 另外验证：

1. 中断 checkpoint 确实停在 epoch 中途（`epoch_itr > 0`）且带回放状态；
2. `manifest_sha256` 与训练视图一致；
3. 恢复后 `optimizer_steps=20`、`micro_steps=160`；
4. 恢复后的 `blocks.0.mlp.fc1.weight` **既不同于官方初始化权重、也不同于第 10 步快照**（证明恢复确实继续训练，而不是重新初始化或空转）；
5. encoder state 可用 `strict=True` 载入一个全新 ViT-H/14（导出路径可用）；
6. optimizer state 非空（4 个 param group），target encoder 已被 EMA 更新且与 encoder 不同。

## 6. 采样与来源曝光（smoke 记录，不作为结果）

part 1 实际消费 1,280 个样本，ledger 计数：

| 来源 | 实际 emitted | share |
|---|---:|---:|
| artemis_imagesafari | 863 | 67.42% |
| agrifield40k_pool | 214 | 16.72% |
| agrifm_ref58 | 180 | 14.06% |
| applebbch81 | 16 | 1.25% |
| agrifm_ref11 | 5 | 0.39% |
| agrifm_ref9 | 1 | 0.08% |
| agrifm_ref1 | 1 | 0.08% |
| **合计** | **1,280** | 100% |

这些计数来自 `source_exposure_ledger.csv`（逐样本 `sample_id` → `physical_source_id`），不是配置推断。1,280 样本上 artemis 份额 67.42%，与整数 epoch 的 67.46% 接近；小样本下的偏差已如实展示。

## 7. 已知限制

- 本阶段只验证 `num_workers=0` 的中断/恢复一致性。`num_workers>0` 时上游 `MaskCollator` 的共享计数器会导致 mask block 尺寸不确定（项目既有结论），因此未扩大声明范围。
- resume 后第一条 `PROGRESS` 行的 `ms/step` 显示 `nan`：该值是“距上一次 optimizer update 的墙钟时间”，新进程尚无上一次记录；CSV 的 `time (ms)` 列不受影响。
- 20 步 loss 的变化不能作为表征质量证据。

## 8. 结论

真实 R3v3 图像上的 **10-step → 中断 → resume → 20-step** 流程完整通过：checkpoint 字段齐全、无重复/丢失 update、EMA 公式验证通过、loss 全程 finite、encoder 可导出并 strict 载入。

【任务已结束】
