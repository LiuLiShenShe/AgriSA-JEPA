# Stage 6D — 冻结预注册协议

- 协议日期：2026-10-27
- 分支：`agri-stage6d-pretrain-seed-replication`
- 代码 commit：待提交（本文件与其余 Stage 6D 代码/配置/工具一并提交后写入；先占位）
- 研究问题：**Stage 6C 的初步正迁移信号，在独立预训练随机种子下能否重复出现？**
- 本文件在启动任何新训练前冻结。事后不修改协议；不挑选有利 run；不更改评价口径。

---

## 1. 研究问题与主要/次要终点

**主要终点**：PhenoBench v1.1.0 val（772 张）mIoU，以“独立预训练 checkpoint”为重复单位。

**次要终点**：Soil IoU、Crop IoU、Weed IoU。

**判定逻辑（预注册，不依赖事后阈值）**：

- **支持进一步投入**：多个独立 B1 checkpoint 相对 A 的 mIoU 改善方向**基本一致**（多数 run 为正），非单个异常 run 驱动，Crop/Weed 变化可解释。
- **结果不确定**：不同预训练 seed 的方向不一致、跨 run 波动接近或超过平均收益，或改善主要来自单个 run。
- **不支持当前短预算下的稳定收益**：多数独立 run 未优于 A，或改善无法重复出现。
- **B1 vs B2**：仅用独立预训练运行间的配对比较；差异小于运行间波动或类别方向不一致时，结论为“无法区分”。

不预先规定固定 mIoU 提升阈值；结合运行间波动、改善方向、类别指标与后续跨任务证据作判断。

---

## 2. 三臂定义

| | Arm A（对照） | Arm B1 | Arm B2 |
|---|---|---|---|
| 骨干 | ViT-H/14 | 同一 ViT-H/14 | 同一 ViT-H/14 |
| 权重来源 | 官方 `IN1K-vit.h.14-300e.pth.tar` | 官方初始化 + 农业持续预训练 | 官方初始化 + 农业持续预训练 |
| 农业持续预训练 | 无 | 2,000 optimizer updates；natural sampling | 2,000 optimizer updates；capped no-backfill |
| 训练数据 | — | R3v3 train view（147,277 张），val 不参与 | 同 B1 |
| 骨干冻结 | 是 | 是（预训练后冻结 encoder 导出） | 是 |

- **A**：复用 Stage 6C 已核实 baseline（`runs/stage3a_phenobench/armA/result.json`），不重训；其 SHA 已核验（`0382013c…`）。
- **B1/B2**：每次新 run 从官方 checkpoint **重新初始化**，不得从历史农业 checkpoint 续训。

---

## 3. 预训练 seed 设计（配对）

| pair | seed | B1 run id | B2 run id | 可追溯 | 主分析 | 敏感性子集 |
|---|---:|---|---|---|---|---|
| P0（历史） | 20261026 | `stage6_b1_natural_2000` | `stage6_b2_capped_no_backfill_2000` | 否（caveat） | 是 | 否 |
| P1（新增） | 20261027 | `stage6d_b1_natural_seed20261027` | `stage6d_b2_capped_no_backfill_seed20261027` | 是 | 是 | 是 |
| P2（新增） | 20261028 | `stage6d_b1_natural_seed20261028` | `stage6d_b2_capped_no_backfill_seed20261028` | 是 | 是 | 是 |

- 同 pair 的 B1/B2 使用**同一 seed**（配对设计：控制数据顺序与增强随机性，只让采样策略不同）。
- 每个新 run 把 `data.seed`、`data.augmentation_seed`、（B2）`data.sampling.seed` 一起设为 pair seed，确保所有随机流一起变化。
- seed 不与历史 `20261026` 冲突；`20261027/20261028` 为顺序配对编号。

---

## 4. 预训练协议（B1/B2，与 Stage 6 base 逐字段一致，除 seed/run 标识/存储目录）

| 项 | 值 |
|---|---|
| 初始化权重 | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar`（`0382013c…`），strict load，`load_checkpoint: false` |
| 模型 | ViT-H/14，patch 14，pred_depth 12，pred_emb_dim 384 |
| 数据 manifest | `data/stage6/stage6_r3v3_train_view_nvme.csv`（`26c55431…`，147,277 行） |
| split | train（val 不参与） |
| batch / grad_accum | 16 / 8（effective batch 128） |
| max optimizer updates | **2,000**（`save_at_optimizer_steps: [500,1000,1500,2000]`） |
| epochs | 24（上限；实际由 2,000-update 预算截断） |
| LR / start_lr / final_lr | 2e-4 / 1e-5 / 1e-6 |
| weight_decay / final_wd | 0.04 / 0.4 |
| warmup / warmup_steps | 0.05 / **100**（绝对步数覆盖，保证两臂 LR/WD 轨迹逐位相同） |
| EMA | [0.996, 1.0] |
| mask | 官方 random multiblock，untouched：num_enc_masks 1、num_pred_masks 4、enc_mask_scale [0.85,1.0]、pred_mask_scale [0.15,0.2]、aspect_ratio [0.75,1.5]、min_keep 10、allow_overlap false |
| bf16 / SDPA / activation_checkpointing | true / true / false |
| num_workers | 0 |
| B1 sampling | natural（`DistributedSampler`，shuffle，seed=pair seed） |
| B2 sampling | profile_capped：cap_unit=physical_source_id、cap_share=0.5、capped_sources=[artemis_imagesafari]、epoch_length=150000、backfill=false、seed=pair seed、profile=`R3v3_license_verified_sampling_profiles.json`（`91fb23ef…`） |
| 输出目录 | `/data/agrisa_jepa_runs/stage6d/<run_id>`（根盘空间不足，见 §9） |

**不得修改**：I-JEPA 目标函数、模型结构、LR/WD schedule、EMA、图像分辨率、mask 机制、数据成员。B1/B2 除采样策略与随机 seed 外完全一致。

---

## 5. PhenoBench 冻结线性 probe 协议（每 checkpoint × seed 0/1/2）

| 项 | 值 |
|---|---|
| 数据 | PhenoBench v1.1.0；val 772 张；train 1,407 张；**test 693 张不参与** |
| 骨干 | ViT-H/14 冻结，`eval()`，`requires_grad=False` |
| 特征 | 224×224 tiles，BF16，patch tokens 16×16，embed 1280 |
| 分辨率适配 | 1024→5×5=25 个 224×224 tile，stride 200，无 padding，全覆盖，重叠 logits 平均拼接 |
| 分割头 | `Linear(1280 → 3)`，3,843 可训练参数 |
| 类别映射 | partial crop 3→crop 1、partial weed 4→weed 2 |
| 优化器 / LR / WD | AdamW / 1e-3 / 0.0 |
| batch / epochs | 32 tiles / 10 |
| probe seed | 0, 1, 2（仅衡量 head/训练随机性） |
| 增强 | 无（特征已缓存，协议固定） |
| 精度 | 特征 BF16，head FP32 |
| 评价 | 官方 `phenobench-eval --task semantics` + 自实现 `IoUAccumulator`（soil）；两者最大偏差容忍 0.01 pp，否则拒绝汇总 |
| 缓存键 | 绑定 arm/run label、checkpoint 完整 SHA、PhenoBench manifest SHA、probe seed、协议版本 |
| 物理 GPU | 物理 GPU1（`CUDA_VISIBLE_DEVICES=1` 映射为 `cuda:0`）；若 GPU1 不可用需**显式授权** |

A 复用 Stage 6C 结果，不重跑；其 checkpoint SHA、manifest SHA、probe 协议均与本阶段一致（已由 `tools/compare_stage6c_phenobench.py` 校验）。

---

## 6. 统计口径（预注册）

### 6.1 单位

- **预训练 checkpoint 是主要重复单位**。
- 同一 checkpoint 的 3 个 probe seed **不**冒充独立预训练样本。

### 6.2 逐 checkpoint

- 报告 probe seeds 0/1/2 的 Soil/Crop/Weed/mIoU；
- 计算该 checkpoint 的 3-seed mean ± std；
- 逐 checkpoint 计算相对 A 的差值。

### 6.3 跨独立预训练 run

- B1、B2 分别以“每 checkpoint 的 probe-seed 均值”为单位，计算跨 run mean ± std；
- 报告每个 run 相对 A 的 mIoU/Crop/Weed 差值；
- 配对 seed 的 B1/B2：报告每对差值与跨 pair 汇总；
- 列出逐 run 数据、预训练 run 数、probe run 数、失败/排除原因。

### 6.4 解释

- 预训练重复数为 3 时以**描述性证据**为主，不用小样本 p 值制造显著性；
- 不把 training loss 下降解释为下游提升；
- 结果方向不一致时，明确报告 unstable / mixed result，不挑选有利 run；
- 同时给出“主分析（含历史）”与“敏感性分析（仅新增 2 个可追溯 run）”两套汇总。

---

## 7. 验收条件（来自提示词 §6，逐项映射）

- [x] 已审计现有 seed 来源，明确历史运行能否计入重复（见 `stage6d_seed_provenance_audit.md`）
- [ ] B1/B2 达到可核验独立预训练运行数（新增 2 + 历史 1 = 3/arm）
- [ ] 所有新 run 从官方 I-JEPA 初始化，而非续训
- [ ] 每个新 run 完成 2,000 optimizer updates
- [ ] 数据 manifest、初始化权重、配置与 checkpoint 均有完整 SHA-256
- [ ] 每个新 checkpoint 完成 3 个 probe seeds 的 PhenoBench val 评价（772 张）
- [ ] A baseline 的复用来源与 SHA 核验通过
- [ ] 统计以预训练 checkpoint 为主要重复单位
- [ ] 未使用 test split，未事后修改协议
- [ ] 新增测试通过；历史失败如实记录
- [ ] 报告与代码可追溯，远端分支已核实

---

## 8. 测试与交付（预注册）

- 新增测试覆盖：seed 配置唯一性、`run_id` 唯一性、checkpoint SHA、manifest SHA、probe seed 完整性、缓存隔离、统计口径、失败 run 处理。
- 运行 Stage 6D 定向测试与全量测试；记录历史失败是否仍为同一失败（不删、不屏蔽、不放松）。
- `git diff --check` 通过。
- 报告/产物路径见 Phase 7；代码/配置/测试/小报告提交到 Stage 6D 分支并 push；不合并 main、不 force push、不提交 checkpoint/原图/大型缓存。
- 中文验收报告：分支/HEAD、每个 run 的 seed 与 checkpoint SHA、optimizer updates、manifest SHA、GPU/耗时/显存、逐 checkpoint 与跨 run 结果、A/B1/B2 差值、测试、遗留问题、是否达标、下一步。

---

## 9. 当前阻塞与偏差记录

| 项 | 事实 | 对协议的影响 |
|---|---|---|
| GPU1 被占用 | 外部项目 PID 2594918，14,912 MiB，100% | 新训练/probe 未启动；**不得**杀进程、不得未经授权改 GPU0 |
| 根盘空间 | `/` 可用 187 G，单 run ~58 G | 新 run 输出目录改到 `/data/agrisa_jepa_runs/stage6d/`（584 G 可用） |
| 历史代码来源 caveat | run 时 Stage 6 代码未提交；run 产物无代码哈希 | 历史 checkpoint 计入主分析但标 caveat；敏感性分析只用新增 run |
| 历史物理 GPU 不可唯一确定 | verification 只记可见序号 | 新 run 显式记录物理 GPU 序号 |

---

## 10. 未执行项（如实标注）

- 4 个新 run 的 2,000-update 预训练：**未执行**。
- 新 checkpoint 的 PhenoBench 特征缓存/线性 probe：**未执行**。
- Stage 6D 统计汇总与比较报告：**未执行**。
- 任何以 Stage 6C 历史结果冒充足 Stage 6D 新结果：**禁止**。

---

## 11. 代码 commit 回填

本文件与其余 Stage 6D 代码（`tools/`、`configs/`、`tests/`、`src/` 改动）一并提交后，将实际 commit SHA 写入本 §0 “代码 commit” 与 `stage6d_preflight` 记录。提交顺序：**先提交再训练**，避免历史 run 的代码来源问题重演。
