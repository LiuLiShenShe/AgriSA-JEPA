# Stage 6D — 执行报告（2026-10-29 完成）

> 本文件是**执行报告**，不是计划书。所有数字均来自本机实测的 checkpoint、probe 结果与统计工具输出。
> 未实际执行的步骤已在正文明确标注（本阶段没有未执行的必需步骤）。

---

## 1. 执行摘要

| 阶段 | 状态 |
|---|---|
| Phase 0 Git/环境/历史协议核验 | **已完成** |
| Phase 1 预训练 seed 来源审计 | **已完成**（`reports/stage6d_seed_provenance_audit.{md,json}`） |
| Phase 2 冻结预注册协议 | **已完成**（`reports/stage6d_protocol.md`；4 配置 `ok=true`） |
| Phase 3 独立预训练（4 个新 run） | **已完成**（4/4，2,000 updates each，逐个核验） |
| Phase 4 PhenoBench probe（4 checkpoint × seeds 0/1/2） | **已完成**（12 个 probe 结果，官方评价器 + 自实现交叉核验） |
| Phase 5 跨 checkpoint 统计 | **已完成**（主分析 + 仅可追溯敏感性） |
| Phase 6 解释 | **已完成**（见 §8） |
| 新测试 / 全量测试 | **已完成**（定向 25 passed；全量 797 passed / 7 skipped / 1 既有失败） |
| Git 交付 | 报告/结果已提交并推送；未合并 main、未 force push |

**核心结论（描述性）**：Stage 6D 的 4 个新预训练 checkpoint（B1/B2 × 2 个新配对 seed）**全部**在 PhenoBench val 772 上高于官方 I-JEPA baseline A；3 个 B1 checkpoint 方向一致，2 个仅新增可追溯 B1 checkpoint 也一致为正。B1 与 B2 的差异小于 run 间波动，**不能区分**。该结果与 Stage 6C 的初步正迁移方向一致，并在独立预训练 seed 下重复。

---

## 2. Git、分支与提交

| 项目 | 值 |
|---|---|
| 基线分支 | `agri-stage6c-phenobench-eval`（HEAD `2697091…`） |
| Stage 6D 分支 | `agri-stage6d-pretrain-seed-replication` |
| 初始代码/配置/测试/报告提交 | `3ee47d7…` |
| 每个 run 前重查 GPU1 的编排器 | `54df24e…` |
| 运行记录 / 事故记录 / run 1 记录 | `bc7c50a` → `31f6873` → `a660be8` → `8a9d50a` → `2ffaffb` |
| 本报告与结果提交 | 以 `git log -1 --format=%H` 与 `git ls-remote origin agri-stage6d-pretrain-seed-replication` 实时输出为准 |
| 是否 force push / reset / clean / 合并 main | **否** |
| 用户未跟踪文件 | `CLAUDE.md` 保留未改 |

提交内容仅代码、配置、测试与小报告；不含 checkpoint、原图、特征缓存或 `runs/`。

---

## 3. Phase 0–2（核验与冻结）

- 官方 I-JEPA 权重 SHA-256 `0382013c…`；train view manifest `26c55431…`（147,277 行）；PhenoBench manifest `13fa2810…`；R3v3 相关 CSV/采样 profile SHA 均与历史一致。
- 审计结论：历史 B1/B2 的 `data.seed=20261026` 有充分证据控制训练随机性（Python/NumPy/Torch、sampler、DataLoader、mask；增强用独立记录的 `augmentation_seed`）；历史 checkpoint 计入**一个带 caveat 的重复**（代码在 run 时未提交；硬件只记录 `visible_device_index=0`）；敏感性口径排除历史 P0。
- 新配对 seed：P1=`20261027`、P2=`20261028`；同 pair 的 B1/B2 同 seed；与历史 `20261026` 不冲突。
- 4 个配置除 seed / arm / run 标识 / 输出目录外与 Stage 6 base 逐字段一致；`tools/check_stage6d_protocol.py`：`ok=true`。

---

## 4. Phase 3 独立预训练（4/4 完成）

每个 run：从官方 I-JEPA checkpoint 重新初始化（`load_checkpoint=false`，`pretrained_sha256=0382013c…`），冻结 train view manifest `26c55431…`，2,000 optimizer updates，`grad_accum=8`，warmup 100，EMA/mask/optimizer/分辨率不変。

| run | arm | pair | seed | 物理 GPU | optimizer updates | micro steps | loss_last | wall (min) | peak MiB | encoder step2000 SHA-256 |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `stage6d_b1_natural_seed20261027` | B1 | P1 | 20261027 | GPU1¹ | 2000 | 15,996 | 0.1286 | 278.3 | 23,255 | `1dc10ab68ab48b7e4de0bfa0cd444a1328d0104589484e4454d36cbe5f389106` |
| `stage6d_b2_capped_no_backfill_seed20261027` | B2 | P1 | 20261027 | GPU0² | 2000 | 15,992 | 0.1338 | 248.8 | 23,236 | `4df238c16a4945e416ab47edf0f37d2eaa94088e440f48777d569d2b71ddeab7` |
| `stage6d_b1_natural_seed20261028` | B1 | P2 | 20261028 | GPU0² | 2000 | 15,996 | 0.1187 | 257.0 | 23,219 | `7e3cb0b67f1f53de720f5a705ef9960df7fd55f2568b4c3fd355ddf9cec82bae` |
| `stage6d_b2_capped_no_backfill_seed20261028` | B2 | P2 | 20261028 | GPU0² | 2000 | 15,992 | 0.1314 | 248.8 | 23,219 | `1650db8c5a158fae3c730b84fb38d262bbdc1ea0be10c640c1639ad30cd76eb8` |

- 编码器/预测器/target encoder 均从官方权重加载（389/152/389 tensors）；`encoder_updated=true`、`predictor_updated=true`、EMA 公式核验通过。
- `realised_source_exposure` 每个 run 均生成；B2 的 capped 采样 `max_share≈0.6746`（cap=0.5 在物理源粒度 + 无回填的整数 epoch 下实际略高，属协议已知行为）。
- 所有 run 的 `manifest_sha256=26c55431…`、`pretrained_init.sha256=0382013c…`、loss finite、seed/增强/采样 seed 传播均核验通过。
- ¹ Run 1 由先前部署的 GPU1 守护在 GPU1 短暂空闲时启动（11:05），已完成的计算未浪费；其 `git_commit=a660be8`（该 run 写入 verification 时的 HEAD）。
- ² Run 2–4 依据用户 2026-10-28「用 gpu0」的显式授权在物理 GPU0 串行执行；`git_commit=8a9d50a`。
- 失败与恢复过程（如实记录）：首个 run 曾因 `data/stage6/images/` 被外部删除而失败；随后用 `tools/stage6_mirror_images.py` 重建 147,277 文件 / 132 GB 字节级镜像，重构 manifest SHA 仍为 `26c55431…`，再重跑。详见 `reports/stage6d_preflight.md` §7。

---

## 5. Phase 4 PhenoBench 冻结 encoder 线性 probe（完成）

协议与 Stage 6C 完全一致：ViT-H/14 冻结，224×224 tile、stride 200、5×5/图，`Linear(1280→3)`，AdamW/1e-3/0.0，10 epochs，batch 32，probe seeds 0/1/2；train 1,407 图、val 772 图；官方 `phenobench-eval` 与自实现 `IoUAccumulator` 交叉核验（容差 0.01 pp）；test 未参与。

- 每个新 checkpoint 的 feature cache 绑定 `run_id|完整 checkpoint SHA|PhenoBench manifest SHA|protocol_version`。
- 每个 probe 结果均核验：checkpoint SHA 一致、seeds `[0,1,2]`、val 772、每 seed 772 张 PNG、官方 stdout 存在、self/official 交叉核验 `agrees=true`。
- 逐 checkpoint 明细：`reports/stage6d_phenobench_per_checkpoint.{json,csv}`。

---

## 6. Phase 5 统计（以预训练 checkpoint 为重复单位）

probe seed **不**计入预训练重复；每个 checkpoint 先对 3 个 probe seed 取均值，再跨 checkpoint 统计。

### 6.1 主分析（每 arm 3 个 checkpoint：历史 P0 + 新 P1/P2）

| arm | n | soil mean±std | crop mean±std | weed mean±std | mIoU mean±std |
|---|---:|---|---|---|---|
| A | 1 | 97.6085 | 81.4567 | 35.9800 | 71.6833 |
| B1 | 3 | 97.6926±0.0062 | 82.2600±0.0500 | 38.9733±0.5407 | **72.9722±0.1996** |
| B2 | 3 | 97.6922±0.0036 | 82.2567±0.0233 | 38.8144±0.5690 | **72.9233±0.1980** |

相对 A：B1 **+1.2889 pp** mIoU（crop +0.8033，weed +2.9933）；B2 **+1.2400 pp**（crop +0.8000，weed +2.8344）。

### 6.2 仅可追溯敏感性（排除历史 P0；每 arm 2 个 checkpoint）

| arm | n | mIoU mean±std | Δ vs A |
|---|---:|---|---:|
| B1 | 2 | 73.0317±0.2217 | **+1.3484** |
| B2 | 2 | 73.0633±0.0033 | **+1.3800** |

### 6.3 逐 checkpoint（每个都是独立预训练重复）

| arm | run | seed | traceable | mIoU (3 probe seeds) | Δ vs A |
|---|---|---|---:|---|---:|
| B1 | historical P0 | 20261026 | 否（caveat） | 72.8533 | +1.1700 |
| B1 | seed20261027 | 20261027 | 是 | 72.8100 | +1.1267 |
| B1 | seed20261028 | 20261028 | 是 | 73.2533 | +1.5700 |
| B2 | historical P0 | 20261026 | 否（caveat） | 72.6433 | +0.9600 |
| B2 | seed20261027 | 20261027 | 是 | 73.0600 | +1.3767 |
| B2 | seed20261028 | 20261028 | 是 | 73.0667 | +1.3834 |

### 6.4 配对 B1/B2（同 seed，仅新增 pair）

| pair | seed | mIoU B2−B1 |
|---|---:|---:|
| P1 | 20261027 | +0.2500 |
| P2 | 20261028 | −0.1867 |
| 均值 | — | **+0.0316** |

跨 pair 的 B2−B1 符号翻转；|均值| 0.03 pp 远小于 run 间波动（跨 checkpoint std ≈0.20 pp）。

---

## 7. 与 Stage 6C 的一致性

Stage 6C（历史单 seed 结果）：B1−A mIoU **+1.17**、B2−A **+0.96**。
Stage 6D 主分析：B1−A **+1.29**、B2−A **+1.24**；敏感性口径 **+1.35 / +1.38**。
方向一致，量级相近；Stage 6D 的 4 个新 checkpoint 单独看也全部为正。

---

## 8. Phase 6 解释（按冻结协议）

1. **是否支持继续投入**：是（描述性）。3 个 B1 checkpoint（含 2 个仅新增可追溯 checkpoint）方向一致为正，且每个单独高于 A；B2 同样一致为正。
2. **B1 vs B2**：**不能区分**。配对差均值 +0.03 pp，且两个 pair 符号相反；差异 < run 间波动。
3. **不是 p 值结论**：只有 3 个（敏感性 2 个）预训练重复/arm，按预注册协议只做描述性统计，不做小样本显著性检验。
4. **不能声称**：不能从本阶段声称 SOTA、首次发现或普适性；A 只有 1 个官方 checkpoint，没有 A 的预训练 seed 重复。
5. **诚实边界**：历史 P0 有代码/硬件 caveat；新 run 跨 GPU0/GPU1；PhenoBench 只做 val，未用 test。

---

## 9. 测试

- 定向：`tests/test_stage6d_seed_replication.py` **25 passed**（3.08 s）。
- 全量（Stage 6D 代码定稿后执行）：**797 passed / 7 skipped / 1 failed**（938.53 s）；唯一失败是既有 `tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`，与 Stage 6/6C 相同，未修/未放宽。
- `git diff --check`：通过；`tools/check_stage6d_protocol.py`：`ok=true`，4 runs，0 problems。
- 详见 `reports/stage6d_test_summary.md`。

---

## 10. 产物清单

| 产物 | 状态 |
|---|---|
| `reports/stage6d_seed_provenance_audit.md` / `.json` | 已生成 |
| `reports/stage6d_protocol.md` | 已生成（冻结） |
| `reports/stage6d_preflight.md` | 已生成（含失败/镜像重建/GPU 授权记录） |
| `reports/stage6d_config_plan.json` / `stage6d_protocol_check.json` | 已生成（ok=true） |
| `reports/stage6d_pretraining_runs.json` / `.csv` | **4/4 completed，含 checkpoint SHA** |
| `reports/stage6d_phenobench_per_checkpoint.json` / `.csv` | **已生成，7 个 checkpoint 实测** |
| `reports/stage6d_phenobench_comparison.json` / `.md` | **已生成（主分析）** |
| `reports/stage6d_phenobench_sensitivity.json` / `.md` | **已生成（仅可追溯）** |
| `reports/stage6d_comparison_manifest.json` | 已生成（7 entries） |
| `reports/stage6d_historical_reference.md` / `.json` | 工具回归（保留） |
| `reports/stage6d_test_summary.md` | 已生成 |
| `configs/stage6d_*.yaml` | 4 个 |
| `tools/stage6d_*.py` | 已实现并实际使用 |
| `tests/test_stage6d_seed_replication.py` | 25 passed |
| 新预训练 checkpoint | 4 个（在 `/data/agrisa_jepa_runs/stage6d/`，未入 git） |
| 新 probe 结果 | 12 个（4 checkpoint × 3 seeds） |

---

## 11. 阻塞与偏离（如实记录）

| 项 | 事实 |
|---|---|
| 首个 run 失败 | `data/stage6/images/` 被外部删除；已按字节重建（147,277 文件 / 132 GB），manifest SHA 不变，重跑成功 |
| GPU1 被外部占用 | `/fj/Agro-JEPA` 多次占用；未杀进程、未改其项目 |
| 改用 GPU0 | 用户 2026-10-28 显式授权「用 gpu0」；run 2–4 在 GPU0，run 1 在 GPU1（竞态已启动） |
| 历史 P0 caveat | 代码在 run 时未提交；硬件只记录相对可见索引；主分析保留、敏感性排除 |
| 磁盘 | 未删除任何用户/项目文件；`/` 空间紧张时仅做只读检查 |

**不会做的事**：杀外部进程、伪造 seed/结果、覆盖历史产物、声称训练/评价已完成而实际未完成。

---

## 12. Stage 6D 验收条件逐项

- [x] 已审计现有 seed 来源，并明确历史运行能否计入重复（可计入，带 caveat；敏感性分析排除）。
- [x] B1/B2 达到规则规定的可核验独立预训练运行数（每 arm 3 个 checkpoint：历史 1 + 新增 2）。
- [x] 所有新 run 从官方 I-JEPA 初始化（`pretrained_sha256=0382013c…`）。
- [x] 每个新 run 完成 2,000 optimizer updates。
- [x] 数据 manifest、初始化权重、配置与 checkpoint 均有完整 SHA-256。
- [x] 每个新 checkpoint 完成 3 个 probe seeds 的 PhenoBench val 评价（772 图，官方+自实现交叉核验）。
- [x] A baseline 的复用来源和 SHA 核验通过。
- [x] 统计以预训练 checkpoint 为主要重复单位。
- [x] 未使用 test split，未事后修改协议。
- [x] 新增测试通过（25 passed）；既有历史失败如实记录。
- [x] 报告与代码可追溯，远端分支已核实。

---

## 13. 下一步建议

1. 若要提高 B1/B2 区分的把握，需要**更多独立预训练 seed**（当前每 arm 3 个 checkpoint，配对差被 run 间波动淹没）；本阶段不足以区分采样策略。
2. 若要检验普适性，应在**独立下游数据集**上重复同一冻结 probe，而不是继续在同一下游上调参。
3. 历史 P0 的代码/硬件 caveat 无法事后修复；后续阶段应继续"先提交、再启动"（本阶段已做到 run 2–4）。
4. A 目前只有 1 个官方 checkpoint；若要把 A 也当作随机变量，需要对 A 做多次初始化/评估。
