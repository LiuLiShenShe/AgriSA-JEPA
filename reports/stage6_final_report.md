# Stage 6A–6B — 受控农业持续预训练 最终验收报告

> 本阶段只完成 **2,000-update 探索性受控预训练**（B1 natural / B2 capped no-backfill）与工程验证。
> 没有做长训练，也没有做下游 probe；**不宣称农业表征质量或最终论文结论已经成立**。

---

## 1. 实际 Git 基线、执行分支与最终 commit

| 项目 | 值 |
|---|---|
| 执行前远端基线 | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981`（`origin/agri-stage5e1-evidence-reconciliation`，`git ls-remote` 核实） |
| 执行前本地 HEAD | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981`（与远端一致） |
| 执行分支 | `agri-stage6-controlled-pretraining` |
| 最终 commit SHA | 见 `reports/stage6_commit.txt`（提交后写入，避免自指） |
| 是否合并 main / force push | 否 |

未提交用户文件 `CLAUDE.md` 保留未动；未执行 reset / clean。

## 2. 本地 train/val CSV 是否存在、SHA 是否一致、路径与样本数

| 项目 | 值 |
|---|---|
| train 路径 | `data/corpus_v4/R3v3_license_verified_train.csv` |
| train 存在 | 是 |
| train SHA-256 | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08`（与冻结值**一致**） |
| train 行数 | **147,277** |
| val 路径 | `data/corpus_v4/R3v3_license_verified_val.csv` |
| val 存在 | 是 |
| val SHA-256 | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686`（与冻结值**一致**） |
| val 行数 | **22,047** |
| train/val 交集 | 0 |
| 全部 train 图像文件存在 | 147,277 / 147,277 |

`split` 列：`train` 147,275 + `TRAIN_ONLY_GROUP_UNKNOWN` 2；2 行属于 train split（R3v3 manifest 的 `counts.train=147277`），保留。

**工程阻塞与处理**：现有 `AgriFieldDataset` 要求 `sample_id/path/split/status`，冻结 R3v3 CSV 是 `image_id/relative_path/split` 且无 `status`。为此新增派生训练视图（不改 R3v3、不重排、不改成员）：

- `data/stage6/stage6_r3v3_train_view_nvme.csv`，147,277 行，SHA-256 `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2`；
- 绝对路径来自 Stage 5C ledger 的 `raw_path`（按 `image_id` join），逐文件存在性已校验。

## 3. 官方 I-JEPA checkpoint SHA 是否一致、是否 strict load

| 项目 | 值 |
|---|---|
| 路径 | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`（**一致**） |
| 加载 | `src/pretrained.py::init_from_pretrained`：`compare_state_dicts()` 逐键/逐 shape 比对后 `load_state_dict(..., strict=True)` |
| `strict=False` | **未使用** |
| 实测 | encoder 389 / predictor 152 / target_encoder 389 tensors，`predictor_inherited=true` |

## 4. cache 是否与 R3v3 manifest 对齐；是否重建；实际读取吞吐

- **未使用 `image_cache_dir`**。解码后 uint8 RGB 全量 cache 需 **1.31 TB**（435.7 G 像素 × 3），本机无此空间。
- 替代方案：`tools/stage6_mirror_images.py` 将 147,277 个文件**逐字节**复制到本机 NVMe（`data/stage6/images/`），每文件 SHA-256 与 `raw_sha256` 比对通过；仅改视图的 `path` 列，不改像素/解码/resize/crop/增强。
- **实测吞吐**：
  - 机械盘（`/data`，`rotational=1`）冷随机读 + 解码：**63.5 ms/图** → 约 8.1 s / 128 样本；
  - NVMe：约 **6–7.5 s / optimizer update**（含 8×16 样本与 GPU 计算）。
- 正式训练结束后为释放磁盘，NVMe 镜像已删除（124 GB）；它是派生产物，可用
  `python -m tools.stage6_mirror_images` 逐字节重建（源文件仍在 `/data`，SHA 可复核）。此点已在 `stage6_preflight.md` 与本节记录。

## 5. Natural 与 capped profile 是否真实接入 DataLoader；代码位置

| 臂 | 采样 | 代码位置 |
|---|---|---|
| B1 | natural | `src/datasets/agri_dataset.py::make_agrifield40k` → 既有 `DistributedSampler(shuffle=True, seed=20261026)` |
| B2 | profile_capped | `src/datasets/stage6_sampling.py::ProfileExposureSampler`，由 `make_agrifield40k(sampler_spec=...)` 构造并传给 `DataLoader`；`src/train.py` 读取 `data.sampling` |

不是“配置里有 `sampling` 就算接入”：正式 B2 运行写出的 `training_verification.yaml.sampling_profile` 与 `stage6_sampling.json` 记录了 sampler 的实际整数分配，`source_exposure_ledger.csv` 记录了实际发出的每个样本。

## 6. B2 no-backfill 如何落地；nominal 与实际 emitted share

- `capped_exposure()`：`capped[s] = min(natural[s], 0.5×150000)` 仅对 `artemis_imagesafari`；其余来源保持 natural；**赤字不回填**。
- 整数 epoch：最大余数法 → 111,172 个样本，`artemis` 固定 75,000。
- **nominal**：75,000 / 150,000 = **50%**。
- **实际 emitted（来自 ledger）**：255,872 个样本中 artemis = 172,618 → **67.46%**；HHI = 0.5031，entropy = 0.9243。
- 报告与对话中始终区分这两个口径；没有把 B2 描述成“实际训练流最大来源占比 50%”。

## 7. smoke 是否完成 10-step → resume → 总计 20-step

**是。** 真实 R3v3 图像、capped sampler、`num_workers=0`：

- part 1：10 updates / 80 micro-steps 后按 `interrupt_after_optimizer_steps=10` 中断；
- part 2：重放 80 个 micro-step，继续到 `optimizer_steps_total=20`、`micro_steps_total=160`；
- itr 序列 0–79 + 80–159，无重复/缺失；EMA 公式验证通过；loss 全程 finite；
- checkpoint 含 encoder/predictor/target/optimizer/scheduler/EMA/RNG/loader/mask/epoch 状态；encoder 可 `strict=True` 导出。

## 8. B1/B2 是否各完成 2,000 updates；有无 OOM/NaN/日志漏行/resume 问题

| 项目 | B1 | B2 |
|---|---|---|
| `optimizer_steps_total` | **2000** | **2000** |
| `micro_steps_total` | 15,996 | 15,992 |
| CSV 行数 | 15,996 | 15,992 |
| OOM | 无 | 无 |
| NaN/Inf | 无（`loss_finite=true`） | 无 |
| 日志漏行 | 无（CSV 行数 = micro-step 数） | 无 |
| resume | 正式运行未中断，无需 resume | 同 |

B2 个别 update 出现 I/O 尖峰（最大 113.7 s，来自 10 GB checkpoint 写入），不是训练错误。

## 9. 两臂配置是否只在采样策略、arm 标识、输出目录不同

**是。** `tools/check_stage6_protocol.py` 与 `tools/compare_stage6_arms.py` 均返回 `protocol_ok=true`：

- 相同：模型结构、patch、pred_depth、pred_emb_dim、mask 全部参数、batch=16、grad_accum=8、LR/WD schedule（含 `warmup_steps=100`）、EMA、bf16、SDPA、manifest 与 SHA、初始化权重与 SHA、seed/augmentation_seed、num_workers=0、save points `[500,1000,1500,2000]`。
- 不同：`meta.arm/arm_label`、`data.sampling`、`logging.folder`。

`warmup_steps` 作为绝对步数覆盖，保证两臂 epoch 长度不同（1151 vs 869 updates/epoch）时 LR/WD 轨迹**逐位相同**（有单测断言）。

## 10. 每臂真实耗时、吞吐、显存、loss 与 EMA 状态

| 指标（实测） | B1 natural | B2 capped |
|---|---:|---:|
| 实际耗时 | 269.6 min（4.49 h） | 269.3 min（4.49 h） |
| median ms / optimizer update | 7,479 | 7,248 |
| 吞吐（估算，128 样本/update） | ≈17.1 samples/s | ≈17.7 samples/s |
| GPU compute step 均值（verification） | 276.3 ms | 291.9 ms |
| peak GPU memory | 23,355 MiB | 23,355 MiB |
| loss 首次 / 末次 | 0.1904 / 0.1289 | 0.1942 / 0.1220 |
| loss min / max | 0.1000 / 0.2039 | 0.1022 / 0.2030 |
| `ema_formula_verified` | true | true |

吞吐是用 median 步时与 128 样本/update **估算**；耗时是日志墙钟实测。

## 11. 新增测试通过/失败/skipped 及 skip 原因

| 项目 | 值 |
|---|---|
| 全套件 | **761 passed / 7 skipped / 1 failed** |
| Stage 6 新增 | **53 passed**（sampling 27、protocol 20、checkpoint 6） |
| skipped | 7（既有环境变量门控）：4× `AGRISA_VITH_CHECKPOINT` 未设置、1× 同、2× `AGRISA_PHENOBENCH` 未设置 |
| failed | 1（**既有** Stage 5E.1 `test_artifact_manifest_agrees_with_git`，未修） |

skipped 不计入 passed；未删测试、未放宽断言。

## 12. R3v3 train/val 是否完全未变；语料状态

- R3v3 train/val SHA-256 与 Stage 5E.1 冻结值**完全一致**（见第 2 节）；未重建、未重排、未原地修改。
- 语料状态仍为 **CONDITIONAL**（18/20 gate PASS；`licence_evidence_backed`、`pool_upstream_licence_resolved` 未决）。
- 训练依据：Stage 5E.1 已声明允许的 **R3v3 train split**；没有把训练启动描述成“许可门禁全部通过”。

## 13. 是否有任何 optimizer update 发生在 val 图像上

**没有。** 证据链：

1. 训练视图只包含 R3v3 train 的 147,277 个 `sample_id`，与 val 22,047 个 id 交集为 0（单测断言）；
2. 视图每行 `split=train`、`status=include`；
3. `source_exposure_ledger.csv` 记录的每个 emitted `sample_id` 全部来自该视图；
4. checkpoint 的 `manifest_sha256` 绑定该视图；
5. val 从未被传入 DataLoader（只有 train 视图）。

## 14. Stage 6C 应先评估哪两个下游任务

1. **PhenoBench 语义分割**：有官方 split 与像素级标注，直接检验 encoder 的稠密空间表征；
2. **GWHD 目标检测**：检验小目标/计数场景下的表征迁移。

理由：两者分别覆盖分割与检测，且项目已有 Stage 3A/5A 的官方 split 与评价器基础；先用它们判断 B1/B2 是否值得扩大预算。

## 15. 本轮结论边界

**可以支持：**
- Stage 6 数据管线、capped sampler 与 resume 机制在真实 R3v3 图像上可运行且可复现；
- B2 精确实现了 Stage 5E.1 的 no-backfill profile（HHI/entropy/整数 cap 均吻合）；
- 两臂协议除采样外一致，且都在 2,000 updates 内稳定收敛、无 NaN/Inf。

**不能支持：**
- 不能说 B2 优于/劣于 B1（没有下游 probe，且只有 1 个 pretrain seed）；
- 不能把 loss 下降当作农业泛化提升；
- 不能把单个 seed 的 downstream probe 多个 seed 说成“多个独立预训练复现”；
- 不能宣称长训练或最终论文结论已完成。

**下一步（Stage 6C，不在本阶段执行）**：用相同官方协议比较官方 ViT-H、B1、B2；PhenoBench + GWHD；固定官方 split；报告逐 seed 均值/标准差、差值与预算；若差异小于实际意义阈值或跨任务方向不一致，不急于宣称 capped 优越，优先扩大预训练 seed 或步数；只有 2,000-step 有清晰信号后再设计 10k–30k update。

---

## 附：本阶段发现并修复的工程问题（如实记录）

1. R3v3 schema 与 loader 不兼容 → 派生训练视图（不改成员）。
2. 机械盘 I/O 过慢（约 7 h/臂）→ NVMe 逐字节镜像（SHA 校验），降至约 4.5 h/臂。
3. `source_exposure_ledger` 的 `flush()` 每个 epoch 重写全部缓存行，导致 CSV 重复；已修复（只追加新行），对已完成 run 的 CSV 按 `(part, micro_step, sample_id)` 去重修复，并加回归测试。汇总 JSON 的计数始终正确。
4. GPU1 被外部任务占用 → 经用户授权改用 GPU0；未终止任何外部进程。正式运行的 verification 两臂均记录 `visible_device_index: 0`（`CUDA_VISIBLE_DEVICES` 重映射），无法从产物判断两臂各自对应的物理卡；两臂各自显存约 23 GB。
5. 磁盘一度写满（checkpoint+镜像）导致一次测试运行 ENOSPC → 释放派生镜像后重跑，全部通过。

【任务已结束】
