# Stage 6D — 训练前核查（preflight）

- 日期：2026-10-27
- 分支：`agri-stage6d-pretrain-seed-replication`
- 基线：`agri-stage6c-phenobench-eval` HEAD `269709114e0b8b87f057234ebf7e455fc0e9fca7`
- 本文件只记录**实际核实到的事实**；未执行项明确标注。

---

## 1. Git

| 项目 | 值 | 来源 |
|---|---|---|
| 仓库根 | `/fj/jepa-二维图像任务` | `git rev-parse --show-toplevel` |
| 基线本地 HEAD | `269709114e0b8b87f057234ebf7e455fc0e9fca7` | `git rev-parse HEAD` |
| 基线远端 HEAD | `269709114e0b8b87f057234ebf7e455fc0e9fca7` | `git ls-remote origin agri-stage6c-phenobench-eval` |
| 本地/远端一致 | 是 | 实测 |
| 新建分支 | `agri-stage6d-pretrain-seed-replication` | 实测 |
| 未跟踪用户文件 | `CLAUDE.md`（内容：`不允许使用conda`、`只允许使用GPU1`） | 实测，保留未改 |
| force push / reset / clean / 合并 main | 否 | 未执行 |

Stage 6D 新增/修改的文件（提交前）：

- `reports/stage6d_seed_provenance_audit.{md,json}`
- `reports/stage6d_protocol.md`
- `reports/stage6d_preflight.md`
- `reports/stage6d_config_plan.json`
- `reports/stage6d_protocol_check.json`
- `reports/stage6d_comparison_manifest.json`
- `reports/stage6d_historical_reference.{md,json}`（用真实 Stage 6C 结果验证聚合工具）
- `configs/stage6d_b1_natural_seed2026102{7,8}.yaml`
- `configs/stage6d_b2_capped_no_backfill_seed2026102{7,8}.yaml`
- `tools/stage6d_build_configs.py`
- `tools/check_stage6d_protocol.py`
- `tools/stage6d_compare.py`
- `tools/run_stage6d_pretraining.py`
- `tools/run_stage6d_probe.py`
- `tests/test_stage6d_seed_replication.py`
- `src/downstream/features.py`（新增 `stage6d_probe_key` / `STAGE6D_PROTOCOL_VERSION`，仅追加）

---

## 2. 冻结输入（实测）

| 输入 | SHA-256 | 状态 |
|---|---|---|
| `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` | 一致（本次重新哈希，64.5 s） |
| `data/stage6/stage6_r3v3_train_view_nvme.csv` | `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` | 一致（147,277 行） |
| `data/corpus_v4/R3v3_license_verified_train.csv` | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` | 一致 |
| `data/corpus_v4/R3v3_license_verified_val.csv` | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` | 一致 |
| `data/corpus_v4/R3v3_license_verified_sampling_profiles.json` | `91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505` | 一致 |
| `reports/phenobench_file_manifest.sha256.txt` | `13fa2810d9fe87c2d65b2dc5b7c7e8ac55e31812aee44af21aea98da8e98bd5d` | 一致 |

PhenoBench v1.1.0 根目录 `/data/datasets/PhenoBench/PhenoBench-v110` 存在；`phenobench-eval` CLI 可用。

---

## 3. 环境与硬件（实测）

| 项目 | 值 |
|---|---|
| Python | 3.13.16（`.venv/bin/python`） |
| PyTorch | 2.9.1+cu128 |
| CUDA 可用 | 是 |
| GPU0 | RTX A6000，49,140 MiB，占用 280 MiB，0% |
| GPU1 | RTX A6000，49,140 MiB，占用 14,912 MiB，100% |
| `/`（NVMe） | 1.9 T，已用 1.6 T，**可用 187 G** |
| `/data`（HDD） | 3.6 T，已用 2.9 T，**可用 584 G** |
| 单 Stage 6 run 产物 | 约 **58 G**（`runs/stage6_b1_natural_2000` = 58G） |

### 3.1 GPU1 占用证据

```
PID 2594918
cwd=/fj/Agro-JEPA
cmd=python3 agro_jepa/train/train_s4b2_full.py --config agro_jepa/config/s4b2_full_train.yaml \
    --condition b2_no_time --seed 1 --epochs 14 \
    --output-dir external_data/s4b2_runs/b2_no_time/seed1
started=2026-10-27 12:14:59
GPU1: 14,912 MiB / 49,140 MiB, 100% util
```

该进程属于**另一个项目**，不是本任务。按约束：不杀进程、不未经授权改 GPU0。

---

## 4. Stage 6D 配置与协议检查

- `tools/stage6d_build_configs.py` 生成 4 个配置；机械断言除 seed / run 标识 / 输出目录外，其余字段与 Stage 6 base 逐字段一致。
- `tools/check_stage6d_protocol.py` 结果：`ok=true`，4 个 run，0 problems。
- 配对 seed：P1 = 20261027，P2 = 20261028；同 pair 的 B1/B2 同 seed；与历史 20261026 不冲突。
- 每个新 run 同时设置 `data.seed`、`data.augmentation_seed`、（B2）`data.sampling.seed`。
- 新 run 输出目录：`/data/agrisa_jepa_runs/stage6d/<run_id>`（根盘空间不足）。

---

## 5. 准备好的命令（未执行）

```bash
# 0) 提交 Stage 6D 代码后，工作树必须干净
git status --porcelain

# 1) 协议检查
.venv/bin/python -m tools.check_stage6d_protocol

# 2) 预训练 dry-run（不训练）
.venv/bin/python -m tools.run_stage6d_pretraining --dry-run --device 1

# 3) 预训练（仅当 GPU1 空闲且工作树干净）
.venv/bin/python -m tools.run_stage6d_pretraining --execute --device 1

# 4) probe dry-run / 执行
.venv/bin/python -m tools.run_stage6d_probe --dry-run --device 1
.venv/bin/python -m tools.run_stage6d_probe --execute --device 1

# 5) 统计
.venv/bin/python -m tools.stage6d_compare \
  --manifest reports/stage6d_comparison_manifest.json \
  --out-prefix reports/stage6d_phenobench_comparison
.venv/bin/python -m tools.stage6d_compare \
  --manifest reports/stage6d_comparison_manifest.json \
  --out-prefix reports/stage6d_phenobench_sensitivity \
  --traceable-only
```

---

## 6. 阻塞与未执行项（如实记录）

| 项 | 状态 | 证据 |
|---|---|---|
| GPU1 空闲 | **否** | PID 2594918，100% util |
| 根盘可容纳 4 个新 run | **否** | 187 G 可用 vs 约 232 G 需求 |
| 2,000-update 预训练 | **未执行** | GPU 阻塞；不伪造 |
| PhenoBench 新 checkpoint probe | **未执行** | 无新 checkpoint；特征缓存需 GPU |
| Stage 6D 新结果统计 | **未执行** | 无新结果 |
| 历史 Stage 6C 结果复用核验 | **已执行** | `stage6d_comparison_manifest.json`；3 个历史 arm 通过 SHA/seed/772 预测校验 |
| 聚合工具真实数据回归 | **已执行** | `stage6d_historical_reference.*` 复现 B1−A +1.17、B2−A +0.96 |

**结论**：Phase 0/1/2 与可离线验证的代码/测试已完成；Phase 3/4/5 因 GPU1 被外部进程占用而**未执行**。需用户决定：等待 GPU1 释放，或显式授权改用 GPU0（Stage 6 有先例，但本轮提示词要求先报告）。
