# Stage 6D — 执行报告（截至 2026-10-27）

> 本文件是**执行报告**，不是计划书。所有未实际执行的步骤都在正文与文末明确标注。
> 本阶段没有产生任何新的预训练 checkpoint 或新的 PhenoBench probe 结果。

---

## 1. 执行摘要

| 项目 | 状态 |
|---|---|
| Phase 0 Git/环境/历史协议核验 | **已完成** |
| Phase 1 预训练 seed 来源审计 | **已完成**（`reports/stage6d_seed_provenance_audit.{md,json}`） |
| Phase 2 冻结预注册协议 | **已完成**（`reports/stage6d_protocol.md`） |
| Stage 6D 配置生成与协议检查 | **已完成**（4 个配置，`ok=true`） |
| Stage 6D 代码/工具/测试 | **已完成**（未提交，见 §2） |
| Phase 3 独立预训练（4 个新 run） | **未执行**（GPU1 被外部进程占用） |
| Phase 4 PhenoBench probe | **未执行**（无新 checkpoint；GPU 阻塞） |
| Phase 5 新结果统计 | **未执行**（无新结果） |
| Phase 7 代码/配置/报告 Git 交付 | 代码/报告已就绪；提交/推送见 §2 |
| 历史 Stage 6C 结果复用核验 | **已完成**（SHA/seed/772 预测校验通过） |
| 聚合工具真实数据回归 | **已完成**（复现 B1−A +1.17、B2−A +0.96） |

**核心结论**：Stage 6D 的审计、协议、配置、代码与测试已完成；由于物理 GPU1 被另一个项目的训练进程占用（项目约束只允许 GPU1），2,000-update 预训练与下游 probe **未执行**。因此本报告**不能**回答“正迁移信号能否在独立预训练 seed 下重复”，也不能声称 B1/B2 稳定优于 A。

---

## 2. Git、分支与提交

| 项目 | 值 |
|---|---|
| 基线分支 | `agri-stage6c-phenobench-eval` |
| 基线 HEAD（本地=远端） | `269709114e0b8b87f057234ebf7e455fc0e9fca7` |
| Stage 6D 分支 | `agri-stage6d-pretrain-seed-replication` |
| Stage 6D 初始代码/配置/测试/报告提交 | `3ee47d7027348b80de4d22ae6a324699c52e8647` |
| Stage 6D 最终代码提交（每个 run 前重查 GPU1；训练实际使用） | `54df24e3ce49db59ebf43affbc2f959f39f28eb2` |
| Stage 6D 元数据提交 | 见 `reports/stage6d_commit.txt` |
| 远端 HEAD | 以 `git ls-remote origin agri-stage6d-pretrain-seed-replication` 的实时输出为准 |
| 是否 force push / reset / clean / 合并 main | 否 |
| 用户未跟踪文件 | `CLAUDE.md` 保留未改 |

提交内容仅代码、配置、测试与小报告；不含 checkpoint、原图、特征缓存或 `runs/`（`.gitignore` 已忽略）。

---

## 3. Phase 0 核验（实测）

- 本地 HEAD 与 `git ls-remote origin agri-stage6c-phenobench-eval` 一致：`2697091…`。
- 官方 I-JEPA 权重 SHA 重新计算一致：`0382013c…`（10,358,004,345 B）。
- train view manifest SHA 一致：`26c55431…`，147,277 行。
- R3v3 train/val CSV、sampling profile、PhenoBench manifest SHA 均与历史记录一致。
- Python 3.13.16 / PyTorch 2.9.1+cu128；`phenobench-eval` CLI 可用；PhenoBench v1.1.0 存在。
- GPU1 被外部项目 `/fj/Agro-JEPA` 的 PID 2594918 占用（14,912 MiB，100%）。
- `/` 可用 187 G，`/data` 可用 584 G；单 Stage 6 run 产物约 58 G。

详见 `reports/stage6d_preflight.md`。

---

## 4. Phase 1 seed 来源审计结论

### 4.1 事实

- 历史 B1/B2 均记录 `data.seed=20261026`、`data.augmentation_seed=20261026`；B2 另记录 `data.sampling.seed=20261026`。
- 代码逐行核验：`data.seed` 传入 Python/NumPy/PyTorch CPU/CUDA、natural `DistributedSampler`、DataLoader generator、mask 位置采样；capped sampler 用 `data.sampling.seed`；增强用独立但同样记录的 `augmentation_seed`。
- 模型无 dropout/drop-path；权重由官方 checkpoint 覆盖；`num_workers=0`。
- **结论**：seed 的“记录 + 作用范围”有充分证据，seed 确实控制训练随机性。

### 4.2 两个 caveat

1. **代码来源**：历史 run 记录 `git_commit=7ec8a6b`，但该 commit 不包含 `stage6_sampling.py`；正式 run（10-26 15:12–19:57）发生在 Stage 6 代码提交 `5f02c86`（10-26 21:50）之前，使用的是未提交工作树。提交后的 `src/`、`configs/` 与当前树逐字节一致，但 run 产物没有代码文件哈希，所以是“推断一致”，不是密码学钉住。
2. **硬件来源**：`stage6_preflight.md` 记录 GPU1 被外部进程占用后经用户授权早期运行改用 GPU0；正式 run 只记录 `visible_device_index=0`，无法区分物理 GPU0/1。

### 4.3 判定与运行数量

- 历史 checkpoint 计入**一个带 caveat 的独立预训练重复**；每个 arm 新增 **2 个完全可追溯的配对 run**（共 4 个新 run）。
- 主分析：每 arm 3 个 checkpoint（历史 1 + 新增 2）。
- 预注册敏感性分析：只使用新增的 2 个可追溯 checkpoint/arm，排除历史 run。
- 新配对 seed：`20261027`、`20261028`；同 pair 的 B1/B2 同 seed；新 run 同时设置 `data.seed`、`data.augmentation_seed`、（B2）`data.sampling.seed`。

详见 `reports/stage6d_seed_provenance_audit.md` 与同名 `.json`。

---

## 5. Phase 2 协议与配置

- `reports/stage6d_protocol.md`：主要/次要终点、三臂定义、配对 seed、预训练协议、probe 协议、统计口径、判定逻辑、验收条件、阻塞记录。
- 生成 4 个配置：
  - `configs/stage6d_b1_natural_seed20261027.yaml`
  - `configs/stage6d_b2_capped_no_backfill_seed20261027.yaml`
  - `configs/stage6d_b1_natural_seed20261028.yaml`
  - `configs/stage6d_b2_capped_no_backfill_seed20261028.yaml`
- `tools/check_stage6d_protocol.py` 结果：`ok=true`，4 个 run，0 problems。逐字段断言除 seed/run 标识/输出目录外，其余与 Stage 6 base 一致。
- B1/B2 同 pair 仅采样策略、arm 标识、run id、输出目录不同。

---

## 6. Phase 3 独立预训练：未执行

- 计划：4 个新 run，每个 2,000 optimizer updates，从官方 checkpoint 重新初始化（`load_checkpoint=false`，`pretrained_sha256=0382013c…`），使用冻结 train view manifest。
- 编排器：`tools/run_stage6d_pretraining.py`，已实现并 `--dry-run` 通过；硬门禁包括：
  - 拒绝脏 tracked 工作树（要求先提交，避免重演 Stage 6 代码来源问题）；
  - 记录 HEAD、完整 `git status`、配置 SHA、物理 GPU；
  - 检查物理 GPU1 空闲内存与利用率，拒绝启动，不杀进程；
  - 每 run 独立目录，输出目录非空则拒绝；
  - 每 run 后核验 2,000 updates、loss finite、manifest SHA、初始化 SHA、checkpoint、来源 ledger、seed 传播、`git_commit` 一致性；
  - 失败即停止，不静默继续。
- **实际未启动任何训练**。未产生 `optimizer_steps_total`、checkpoint SHA、耗时、显存等新数据。
- `reports/stage6d_pretraining_runs.json/csv` 只有 `status=not_started` 的计划记录，没有伪造实测值。

---

## 7. Phase 4 PhenoBench 冻结 encoder 线性 probe：未执行

- 计划：每个新 checkpoint × probe seeds 0/1/2；val 772；官方评价器 + 自实现土壤 IoU 交叉核验；test 不参与。
- 编排器：`tools/run_stage6d_probe.py`，已实现并 `--dry-run` / `--manifest-only` 通过；包含 checkpoint SHA 校验、缓存绑定、772 预测/官方 stdout 校验、预测 manifest。
- **实际未运行 probe**。`reports/stage6d_phenobench_per_checkpoint.{json,csv}` 只有 `status=not_executed` 占位，不含任何指标。

---

## 8. Phase 5 统计：未执行；历史参考已验证工具

- 统计工具：`tools/stage6d_compare.py`，以 **checkpoint** 为重复单位，逐 checkpoint 计算 3-seed 均值、跨 checkpoint mean±std、相对 A 差值、配对 B1/B2 差值与跨 pair 汇总，并支持 `--traceable-only` 敏感性口径。
- 用真实 Stage 6C 结果做数值回归（`reports/stage6d_historical_reference.{md,json}`）：
  - B1 − A：mIoU **+1.17 pp**，Crop +0.76 pp，Weed +2.69 pp；
  - B2 − A：mIoU **+0.96 pp**，Crop +0.77 pp，Weed +2.03 pp。
  与 Stage 6C 报告一致，说明聚合工具没有口径漂移。
- **这不是 Stage 6D 新结果**，不能用于回答 Stage 6D 的可重复性问题。
- `reports/stage6d_phenobench_comparison.{json,csv,md}` 明确标注 `not_executed` / `historical_reference_only`，不含新指标。

---

## 9. 测试

- Stage 6D 定向测试：`tests/test_stage6d_seed_replication.py`，**25 passed**（config 生成/只改允许字段/seed 配对与唯一性、协议字段冻结、缓存键绑定、checkpoint 均值、跨 checkpoint 统计不混淆 probe seed、配对 B1/B2、失败/缺失拒绝、历史与敏感性子集、probe 校验、比较工具真实数据）。
- 全量测试：**797 passed / 7 skipped / 1 failed**（938.53 s）。相对 Stage 6C 基线（772/7/1），新增 25 个测试全部通过。
- 既有历史失败 `tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git` 未修、未删、未放宽；与 Stage 6/6C 完全相同。
- `git diff --check`：通过。
- 详见 `reports/stage6d_test_summary.md`。

---

## 10. 产物清单

| 产物 | 状态 |
|---|---|
| `reports/stage6d_seed_provenance_audit.md` / `.json` | 已生成 |
| `reports/stage6d_protocol.md` | 已生成 |
| `reports/stage6d_preflight.md` | 已生成 |
| `reports/stage6d_config_plan.json` | 已生成 |
| `reports/stage6d_protocol_check.json` | 已生成（ok=true） |
| `reports/stage6d_pretraining_runs.json` / `.csv` | 已生成，`not_started` |
| `reports/stage6d_phenobench_per_checkpoint.json` / `.csv` | 已生成，`not_executed` |
| `reports/stage6d_phenobench_comparison.md` / `.csv` / `.json` | 已生成，`not_executed` / 历史参考 |
| `reports/stage6d_comparison_manifest.json` | 已生成（A + 历史 B1/B2；新 run 待补） |
| `reports/stage6d_historical_reference.md` / `.json` | 已生成（工具回归） |
| `reports/stage6d_test_summary.md` | 待全量测试完成后生成 |
| `configs/stage6d_*.yaml` | 4 个已生成 |
| `tools/stage6d_build_configs.py` | 已实现 |
| `tools/check_stage6d_protocol.py` | 已实现 |
| `tools/stage6d_compare.py` | 已实现 |
| `tools/run_stage6d_pretraining.py` | 已实现，dry-run 通过 |
| `tools/run_stage6d_probe.py` | 已实现，dry-run/manifest-only 通过 |
| `tests/test_stage6d_seed_replication.py` | 25 passed |
| 新预训练 checkpoint / probe 结果 | **未产生** |

---

## 11. 阻塞与需要用户决定

| 阻塞 | 事实 | 需要 |
|---|---|---|
| GPU1 被外部项目占用 | PID 2594918，`/fj/Agro-JEPA`，14,912 MiB，100% | 等待释放；或**显式授权改用 GPU0**（Stage 6 有先例；本轮提示词要求先报告） |
| 根盘空间不足 | `/` 187 G vs 4 个新 run 约 232 G | 已默认把新 run 写到 `/data`（584 G）；若坚持写在仓库 `runs/`，需减少 checkpoint 保留或清理空间 |
| 历史代码来源 caveat | run 时未提交；无代码哈希 | 已在主分析标 caveat，并提供“仅新增 run”的敏感性口径 |

**不会做的事**：杀外部进程、未经授权改 GPU0、伪造 seed/结果、覆盖历史产物、声称训练已完成。

---

## 12. Stage 6D 验收条件逐项

- [x] 已审计现有 seed 来源，并明确历史运行能否计入重复（可计入，带 caveat；敏感性分析排除）。
- [ ] B1/B2 达到规则规定的可核验独立预训练运行数（**未执行**）。
- [ ] 所有新 run 从官方 I-JEPA 初始化（**未执行**）。
- [ ] 每个新 run 完成 2,000 optimizer updates（**未执行**）。
- [ ] 数据 manifest、初始化权重、配置与 checkpoint 均有完整 SHA-256（配置/输入已核验；新 checkpoint **未产生**）。
- [ ] 每个新 checkpoint 完成 3 个 probe seeds 的 PhenoBench val 评价（**未执行**）。
- [x] A baseline 的复用来源和 SHA 核验通过。
- [x] 统计以预训练 checkpoint 为主要重复单位（工具与测试已实现）。
- [x] 未使用 test split，未事后修改协议。
- [ ] 新增测试通过（定向 25 passed；全量待完成）；历史失败如实记录。
- [ ] 报告与代码可追溯，远端分支已核实（待提交/推送）。

---

## 13. 下一步

1. **用户决定 GPU**：等待 GPU1 释放，或授权 GPU0。
2. 提交 Stage 6D 代码/配置/测试/报告，确保工作树干净。
3. 运行 `tools/run_stage6d_pretraining --execute --device <n>`，逐 run 核验。
4. 运行 `tools/run_stage6d_probe --execute --device <n>`，生成 12 个新 probe 结果（4 checkpoint × 3 seeds）。
5. 运行 `tools/stage6d_compare`（主分析 + 敏感性），生成逐 checkpoint 与跨 run 报告。
6. 按 §12 逐项勾选，更新本报告与 `stage6d_test_summary.md`。
7. 只有在多个独立 B1 checkpoint 方向基本一致时，才讨论“支持进一步投入”；否则如实报告不稳定/mixed result。
