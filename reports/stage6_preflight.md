# Stage 6 — 训练前置核查报告（preflight）

本报告只记录**实际执行前核实到的事实**。区分三类数字：
- **配置目标值**：配置文件里写的；
- **实测值**：本次在本机运行/读取得到的；
- **估算值**：由实测外推的，明确标注。

---

## 1. Git 基线

| 项目 | 值 |
|---|---|
| remote | `https://github.com/LiuLiShenShe/AgriSA-JEPA.git` |
| upstream | `https://github.com/facebookresearch/ijepa.git` |
| 执行前分支 | `agri-stage5e1-evidence-reconciliation` |
| 执行前本地 HEAD | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981` |
| `git ls-remote origin refs/heads/agri-stage5e1-evidence-reconciliation` | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981` |
| 本地 HEAD 与远端分支 HEAD 是否一致 | **一致** |
| 新建工作分支 | `agri-stage6-controlled-pretraining`（从 `7ec8a6b` 创建） |
| 未提交文件 | `CLAUDE.md`（用户文件，保留，未修改） |

未执行 reset / clean / force push。

## 2. 环境与硬件（实测）

| 项目 | 值 |
|---|---|
| Python | 3.13.12（`.venv-uv/bin/python`） |
| PyTorch | 2.9.1+cu128 |
| CUDA / cuDNN | 12.8 / 91002 |
| GPU | 2 × NVIDIA RTX A6000，各 49,140 MiB |
| CPU / RAM | 64 vCPU / 125 GiB |
| `/`（NVMe，`/dev/nvme0n1p2`） | 1.9 T，执行前可用约 385 G |
| `/data`（HDD，`/dev/sda1`） | 3.6 T，可用约 599 G |

### GPU 分配（重要偏差，已获用户授权）

执行开始时 GPU1 空闲；进入 smoke 阶段后，GPU1 被**两个不属于本任务的**外部训练进程占用：

- PID 3830083：`agro_jepa/train/train_s4.py --condition b2_no_time_constant_delta`，约 16.3 GB
- PID 3831669：`agro_jepa/train/train_s4.py --condition b3_delta`，约 16.3 GB

GPU1 仅余约 15 GB，小于 ViT-H/14 所需的约 23 GB，首次 smoke OOM。

- 未终止任何外部进程。
- 已向用户确认，**smoke 阶段及其后的早期运行改用 GPU0（`cuda:0`）**。
- 正式对照运行的 `training_verification.yaml` 两臂均记录 `visible_device_index: 0`（通过 `CUDA_VISIBLE_DEVICES` 重映射），
  无法从日志/verification 区分两臂各自落在物理 GPU0 还是 GPU1；两臂显存占用均约 23 GB，未超出单卡 49 GB。
- 该偏差在此显式记录；它不改变任何训练超参或数据语义。

## 3. 数据 manifest（实测）

| 项目 | 值 |
|---|---|
| train CSV | `data/corpus_v4/R3v3_license_verified_train.csv` |
| train 是否存在 | 是 |
| train SHA-256 | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08`（与冻结值一致） |
| train 行数 | **147,277**（与冻结值一致） |
| val CSV | `data/corpus_v4/R3v3_license_verified_val.csv` |
| val 是否存在 | 是 |
| val SHA-256 | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686`（与冻结值一致） |
| val 行数 | **22,047**（与冻结值一致） |
| train/val sample_id 交集 | **0** |
| train `stage5e_train_eligible=True` | 147,277 / 147,277 |
| train `holdout_status=OK` | 147,277 / 147,277 |
| train `split` 列 | `train` 147,275 + `TRAIN_ONLY_GROUP_UNKNOWN` 2 |

这 2 个 `TRAIN_ONLY_GROUP_UNKNOWN` 行属于 train split（R3v3 manifest 的 `counts.train=147277`，其中 `train_only_group_unknown=2`），因此保留在训练集合中，原值记录在派生视图的 `source_split` 列。

### 3.1 发现并修复的工程阻塞：当前 loader 不能直接读 R3v3 CSV

`AgriFieldDataset._read_manifest()` 要求列 `sample_id`、`path`、`split`、`status`，并过滤 `status == 'include'`。冻结 R3v3 CSV 使用 `image_id`、`relative_path`、`split`，**没有 `status` 列**，也没有绝对路径。

处理方式（不重建、不重排、不改冻结文件）：新增
`tools/stage6_build_train_manifest.py`，从 Stage 5C 的 per-source ledger（`data/corpus_v3/images/*.csv` 的 `raw_path`，按 `image_id` join）生成**派生训练视图**。

| 派生视图 | 值 |
|---|---|
| 路径 | `data/stage6/stage6_r3v3_train_view.csv` |
| 行数 | 147,277 |
| SHA-256 | `8e4050977cfded74f6cee942457cecf11e7fc21b432afdf563eb16f67e6187a8` |
| 行序 | 与 R3v3 CSV 完全一致（未重排） |
| 成员 | 与 R3v3 train 完全一致（147,277 个 sample_id） |
| 与 val 交集 | 0 |
| 图像文件存在性 | 147,277 / 147,277 存在 |

### 3.2 图像读取介质

- 解码后的 uint8 RGB cache 需要约 **1.31 TB**（435.7 G 像素 × 3），本机无此空间 → **不构建** `image_cache_dir` 缓存。
- 源图位于 `/data`（机械盘，`rotational=1`）。实测冷随机读 + 解码 **63.5 ms/图**，约 **8.1 s / 128 样本**，加上约 2.2 s GPU 计算，实测约 **9.8–10.2 s / optimizer update**（2,000 update 约 7 小时/臂）。
- 为把 I/O 降到可接受范围，新增 `tools/stage6_mirror_images.py`，把 147,277 个文件**逐字节原样**复制到本机 NVMe（`data/stage6/images/`），并对每个文件校验 SHA-256 与 manifest `raw_sha256` 一致。
  - 只改派生视图的 `path` 列，不改像素、解码、resize、crop 或增强语义。
  - 新视图：`data/stage6/stage6_r3v3_train_view_nvme.csv`，147,277 行，SHA-256 `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2`。
  - 镜像实测：147,277 文件 / **132.0 GB**，逐文件 SHA-256 全部匹配；冷启动阶段因机械盘随机 seek 只有 14–34 文件/s，改为按路径排序 + 断点续传后达到 700–790 文件/s，总耗时 192.8 s。
  - 镜像后实测约 **6–7.5 s / optimizer update**（2,000 update 约 4.5 小时/臂）。
  - **正式训练结束后已删除该镜像（124 GB）以释放磁盘**（最后一次全量测试曾因磁盘 97% 触发 ENOSPC）。它是派生产物：源文件仍在 `/data`，可用
    `.venv-uv/bin/python -m tools.stage6_mirror_images` 逐字节重建，视图 SHA 与逐文件 SHA 均可复核。
- **没有** disable 任何图像；没有改变 RGB 转换或 RandomResizedCrop。

## 4. 官方初始化权重（实测）

| 项目 | 值 |
|---|---|
| 路径 | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`（与冻结值一致） |
| 结构 | 官方 I-JEPA ViT-H/14（`vit_huge`, patch 14, pred_depth 12, pred_emb_dim 384） |
| 加载路径 | `src/pretrained.py::init_from_pretrained` |
| strict 策略 | `compare_state_dicts()` 逐键比对后 `load_state_dict(..., strict=True)`；**不存在 `strict=False` 绕过** |
| 加载结果（smoke 实测） | encoder 389 / predictor 152 / target_encoder 389 tensors，`predictor_inherited=true` |

## 5. Stage 5E.1 状态（不变）

- gate：**18/20 PASS**；未决项 `licence_evidence_backed`、`pool_upstream_licence_resolved`。
- 语料状态：**CONDITIONAL**（本阶段不改变）。
- 本阶段训练依据：Stage 5E.1 已声明的 **R3v3 train split**。
- R3v3 train/val 字节未变（SHA 见第 3 节）。

## 6. 基线测试（执行前）

执行 `.venv-uv/bin/python -m pytest tests/ -q`：

- **708 passed / 7 skipped / 1 failed**（284.69 s）
- 唯一失败：`tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`
  - 原因：`reports/stage5e1_licence_evidence.json` 在 git 中已被跟踪，但 `reports/stage5e1_artifacts.json` 将其记录为 `tracked: false`。
  - 这是 **Stage 5E.1 基线自身的不一致**，非本阶段引入；按“不修改 Stage 5E.1 历史产物”的约束，本阶段**不修**，如实记录。skipped 不计入 passed。

## 7. 采样接入核查

- B1 natural：既有 `DistributedSampler(shuffle=True, seed=20261026)`，无新增路径。
- B2 capped：`ProfileExposureSampler`（`src/datasets/stage6_sampling.py`），由 `make_agrifield40k(sampler_spec=...)` 接入 `DataLoader`。
- 采样数学、no-backfill 语义、实际 emitted 比例见 `reports/stage6_sampling_profile_audit.md`。
- `tools/check_stage6_protocol.py`：B1/B2 除 arm 标识、采样策略、输出目录外逐字段一致；冻结 profile 的 cap/unfulfilled/total 均复核。

## 8. 结论

- Git、数据、权重、环境均核实。
- 发现并解决两个真实工程阻塞：loader schema 不兼容、机械盘 I/O 过慢。
- 未改动 R3v3 成员、阈值或 Stage 5 历史产物。
- 允许继续 smoke 与 B1/B2 受控对照。

【任务已结束】
