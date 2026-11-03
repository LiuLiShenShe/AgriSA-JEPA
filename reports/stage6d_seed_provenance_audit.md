# Stage 6D — 预训练 seed 来源审计

- 审计日期：2026-10-27
- 审计分支：`agri-stage6d-pretrain-seed-replication`（从 `agri-stage6c-phenobench-eval` HEAD `269709114e0b8b87f057234ebf7e455fc0e9fca7` 新建）
- 审计对象：Stage 6 历史 B1（natural）/ B2（capped no-backfill）两个 2,000-update 预训练 run
- 证据类型标注：**[实测]** 本次在本机读取/计算；**[代码核验]** 逐行阅读仓库代码；**[推断]** 由实测与代码推出的解释，不是直接证据。

---

## 0. 结论摘要

1. 历史 B1/B2 都记录了 `data.seed = 20261026`、`data.augmentation_seed = 20261026`；B2 另记录 `data.sampling.seed = 20261026`。这些值同时出现在**配置文件**、run 启动时转储的 **`params-ijepa.yaml`**、**`console.log`** 与 **`training_verification.yaml`** 中。**[实测]**
2. 代码核验表明 `data.seed` 实际传入 Python `random`、NumPy、PyTorch CPU/CUDA、natural `DistributedSampler`、DataLoader generator，以及 mask 位置采样；capped arm 的 sampler 使用 `data.sampling.seed`。图像增强由独立的 `augmentation_seed` 控制（同样有记录）。模型无 dropout / drop-path，权重由官方 checkpoint 覆盖，`num_workers=0`。**[代码核验]**
3. 因此，**seed 的“记录 + 作用范围”有充分证据**：可以证明 seed 确实控制训练随机过程，而不是只写进配置。
4. **两个来源 caveat 必须披露**：
   - **代码来源**：历史 run 记录的 `git_commit = 7ec8a6b…` 是 Stage 6 代码提交 `5f02c86…` **之前**的基线 commit；该 commit **不包含** `src/datasets/stage6_sampling.py`。正式 B1/B2 run 使用的是**未提交工作树代码**。提交后的 `src/`、`configs/` 与当前树逐字节一致（`git diff 5f02c86 HEAD -- src/ configs/` 为空），但 run 产物没有记录代码文件哈希，所以代码来源是“推断一致”，不是密码学钉住。**[实测 + 推断]**
   - **硬件来源**：`reports/stage6_preflight.md` 记录 Stage 6 smoke 起 GPU1 被两个外部训练进程占用，经用户授权早期运行改用 GPU0；正式 run 的 `training_verification.yaml` 只记录程序可见序号 `visible_device_index: 0`，无法区分物理 GPU0/GPU1。该偏差不影响 RNG 协议，但影响“落在哪张卡”的可追溯性。**[实测]**
5. **运行数量决定**：按规则第一档处理——历史 checkpoint 计入**一个带 caveat 的独立预训练重复**；每个 arm **新增 2 个完全可追溯的配对 run**（共 4 个新 run）。主分析每个 arm 3 个 checkpoint；预注册敏感性分析只用 2 个新增 run（完全可追溯子集）。
6. **新 seed 选择**：配对 seed `20261027`、`20261028`，与历史 `20261026` 不冲突。每个新 run 同时设置 `data.seed`、`data.augmentation_seed`、（B2）`data.sampling.seed = pair seed`，确保所有随机流一起变化；同一 pair 的 B1/B2 使用同一 seed。
7. **当前阻塞（未执行训练/probe）**：
   - GPU1 被外部项目 `/fj/Agro-JEPA` 的 PID 2594918 占用（14,912 MiB，100% 利用率）；按约束不得杀进程、不得未经授权改 GPU0。
   - `/`（NVMe）仅余 **187 G**，单个 Stage 6 run 产物约 **58 G**；4 个新 run 约 **232 G**，放不下。`/data`（HDD）可用 **584 G**。

---

## 1. Phase 0 核验事实

### 1.1 Git

| 项目 | 值 | 来源 |
|---|---|---|
| 仓库根 | `/fj/jepa-二维图像任务` | [实测] `git rev-parse --show-toplevel` |
| 审计前分支 | `agri-stage6c-phenobench-eval` | [实测] |
| 审计前本地 HEAD | `269709114e0b8b87f057234ebf7e455fc0e9fca7` | [实测] |
| 远端同名分支 HEAD | `269709114e0b8b87f057234ebf7e455fc0e9fca7` | [实测] `git ls-remote origin` |
| 本地 / 远端是否一致 | **一致** | [实测] |
| 工作树未跟踪文件 | `CLAUDE.md`（仅此一项） | [实测] `git status --porcelain` |
| 新建分支 | `agri-stage6d-pretrain-seed-replication` | [实测] |
| 是否 force push / reset / clean / 合并 main | **否** | [实测] 未执行 |

`CLAUDE.md` 内容为项目约束（`不允许使用conda`、`只允许使用GPU1`），审计全程保留未改。

### 1.2 冻结输入 SHA-256

| 输入 | 记录值 | 本次实测 | 一致 |
|---|---|---|---|
| 官方 I-JEPA 初始化权重 `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` | 同左（10,358,004,345 B） | 是 |
| Stage 6 train view `data/stage6/stage6_r3v3_train_view_nvme.csv` | `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` | 同左，147,277 数据行 | 是 |
| R3v3 train CSV `data/corpus_v4/R3v3_license_verified_train.csv` | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` | 见 `reports/stage6_protocol_check.json` | 是（历史记录） |
| R3v3 val CSV `data/corpus_v4/R3v3_license_verified_val.csv` | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` | 同上 | 是（历史记录） |
| Stage 5E.1 sampling profile | `91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505` | 见 `reports/stage6_run_manifest_b2.json` | 是（历史记录） |
| PhenoBench 文件清单 `reports/phenobench_file_manifest.sha256.txt` | `13fa2810d9fe87c2d65b2dc5b7c7e8ac55e31812aee44af21aea98da8e98bd5d` | 同左 | 是 |

### 1.3 环境

| 项目 | 值 | 来源 |
|---|---|---|
| Python | 3.13.16（`.venv/bin/python`） | [实测] |
| PyTorch | 2.9.1+cu128，CUDA 可用 | [实测] |
| yaml / numpy / Pillow | 6.0.3 / 2.4.6 / 12.3.0 | [实测] |
| pytest | 9.1.1 | [实测] |
| GPU0 | RTX A6000，49,140 MiB，占用 280 MiB，0% | [实测] `nvidia-smi` |
| GPU1 | RTX A6000，49,140 MiB，占用 14,912 MiB，100% | [实测] `nvidia-smi` |
| `/`（NVMe） | 1.9 T，已用 1.6 T，可用 **187 G** | [实测] `df -h` |
| `/data`（HDD） | 3.6 T，已用 2.9 T，可用 **584 G** | [实测] `df -h` |

GPU1 上的进程：

```
PID 2594918  python3 agro_jepa/train/train_s4b2_full.py --config agro_jepa/config/s4b2_full_train.yaml \
  --condition b2_no_time --seed 1 --epochs 14 --output-dir external_data/s4b2_runs/b2_no_time/seed1
cwd=/fj/Agro-JEPA   owner=test   started=2026-10-27 12:14:59
```

该进程属于**另一个项目**，不是本仓库的 Stage 6D 任务。**[实测]**

---

## 2. 历史 B1/B2 run 的事实

| 项目 | B1 natural | B2 capped no-backfill | 来源 |
|---|---|---|---|
| 配置 | `configs/stage6_b1_natural_2000.yaml` | `configs/stage6_b2_capped_no_backfill_2000.yaml` | [实测] |
| 配置 SHA-256 | `52e4478c5b6c375f89476007954a2cbe72ac227ffb91c6e6d184f23bc77657db` | `f0d5fd7bdc6f11a0e4a4103f610aaf3d8e8556dd5e17cf211d93193879b63b01` | [实测] `sha256sum` |
| run 目录 | `runs/stage6_b1_natural_2000` | `runs/stage6_b2_capped_no_backfill_2000` | [实测] |
| `params-ijepa.yaml` SHA-256 | `e4e8d7ff20c06694938b0d31130daa0dfe7e53128594d9eec3f7a26485b43a74` | `cfed80f099a47a6c9fe95170ba020515a9785a349b428de71a7fa610e8940ebc` | [实测] |
| `training_verification.yaml` SHA-256 | `7176bc6ab7384d437b0b7047948d9870d912b897f629a33e10d49281936a1552` | `c2bd3da59803767be0bc4c2addbee2f59d28243e4d4a36757635e4a5f0fd2f3a` | [实测] |
| `jepa_r0.csv` SHA-256 | `e4fc7961a27e0b3036d2d6a06c047012ff78fa6642f2ec1b08bc3806ab439dfb` | `ee9662f70ff88c3d9e55c33a7581ac38f3200827fa1b6545a47c5e10e7fb8d6a` | [实测] |
| step2000 encoder SHA-256 | `19796cebc65e29a07eafd7222f5a642d4ef9f28c33f2e4f4f2b65e34a2bfde6b` | `6e7a83c9156e556757016145fb804e2b67e7b94f5e8b45bda4ffd24cb5975a62` | [实测] `sha256sum` |
| `optimizer_steps_total` | **2,000** | **2,000** | [实测] verification |
| `micro_steps_total` | 15,996 | 15,992 | [实测] verification |
| `optimizer_steps_per_epoch` | 1,151 | 869 | [实测] verification |
| loss finite | true | true | [实测] verification |
| peak GPU memory | 23,355 MiB | 23,355 MiB | [实测] verification |
| 记录 `git_commit` | `7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981` | 同左 | [实测] verification |
| 记录 `visible_device_index` | 0 | 0 | [实测] verification |
| run 墙钟 | 269.6 min | 269.3 min | [实测] console |
| 文件时间（params / step500 / step2000 / ledger） | 10-26 15:12 / 16:17 / 19:44 / 19:57 | 10-26 15:16 / 16:26 / 19:48 / 19:57 | [实测] 文件 mtime |

两臂均为 2,000 optimizer updates、无 OOM/NaN、无 resume。B1 实际来源曝光 255,936 样本，B2 255,872 样本；这些是 **update 预算**下的实际曝光，不是 epoch 数。**[实测]**

---

## 3. Seed 字段与传递路径

### 3.1 记录的 seed 字段

| 字段 | B1 | B2 | 记录位置 |
|---|---:|---:|---|
| `data.seed` | 20261026 | 20261026 | 配置、`params-ijepa.yaml`、`console.log` |
| `data.augmentation_seed` | 20261026 | 20261026 | 配置、`params-ijepa.yaml`、`console.log` |
| `data.sampling.seed` | —（natural 无此字段） | 20261026 | 配置、`params-ijepa.yaml`、`console.log`、`stage6_sampling.json` |
| `sampling_profile.seed`（verification） | 20261026 | 20261026 | `training_verification.yaml` |

### 3.2 代码路径（逐项）

| 随机过程 | 实际使用的 seed | 代码位置 | 证据类型 |
|---|---|---|---|
| Python `random` | `data.seed` | `src/train.py:194` | [代码核验] |
| NumPy 全局 RNG | `data.seed` | `src/train.py:195` | [代码核验] |
| PyTorch CPU RNG | `data.seed` | `src/train.py:196` | [代码核验] |
| PyTorch CUDA RNG（全部可见卡） | `data.seed` | `src/train.py:198` | [代码核验] |
| natural 采样顺序 | `DistributedSampler(seed=data.seed, shuffle=True)` | `src/train.py:428-430` → `src/datasets/agri_dataset.py:257-263` | [代码核验] |
| DataLoader batch 生成 | `torch.Generator().manual_seed(data.seed + rank)` | `src/datasets/agri_dataset.py:284-285` | [代码核验] |
| capped 采样顺序 | `ProfileExposureSampler(seed=data.sampling.seed)` | `src/datasets/agri_dataset.py:267-279` → `src/datasets/stage6_sampling.py:200-210` | [代码核验] |
| mask 块尺寸 | 每 batch 计数器 `step()` + `torch.Generator().manual_seed(step)` | `src/masks/multiblock.py:125-127` | [代码核验] |
| mask 块位置 | 全局 PyTorch RNG（`torch.randint`），即 `data.seed` | `src/masks/multiblock.py:89-90` | [代码核验] |
| 图像增强 | `augmentation_seed` → 每样本 `sha256("{aug}:{epoch}:{index}")` 派生 seed | `src/datasets/agri_dataset.py:173-175, 199-206` | [代码核验] |
| 模型 dropout / drop-path | 无（`drop_rate=0.0`、`attn_drop_rate=0.0`、`drop_path_rate=0.0`） | `src/models/vision_transformer.py:257-259, 372-374` | [代码核验] |
| 模型权重初始化 | 随机 `trunc_normal_`，随后被官方 checkpoint 覆盖 | `src/helper.py:76-84`；`src/train.py:309, 334` | [代码核验] |
| DataLoader worker | `num_workers=0`，`seed_worker` 不实际运行 | 配置；`src/datasets/agri_dataset.py:19-24` | [代码核验] |

### 3.3 关键解释

- `data.seed` 是**主预训练 seed**，控制数据顺序、mask 位置与所有全局 RNG。
- 图像增强被**刻意解耦**到 `augmentation_seed`：同一样本在同一 `(augmentation_seed, epoch, index)` 下得到逐字节相同的增强结果。这是 Stage 4 为“只比较 mask、固定增强”引入的设计。
- 因此，**只改 `data.seed` 而不改 `augmentation_seed` 的“seed 重复”不会改变增强随机流**。Stage 6D 的新 run 必须把两者（以及 B2 的 `data.sampling.seed`）一起设为同一个 pair seed，才能让所有随机流一起变化。这一点写入 Stage 6D 协议。

---

## 4. 代码来源审计（关键 caveat）

### 4.1 时间线

| 时间（2026-10-26） | 事件 | 证据 |
|---|---|---|
| 15:12 | B1 `params-ijepa.yaml` 写入（run 开始） | [实测] mtime |
| 15:16 | B2 `params-ijepa.yaml` 写入（run 开始） | [实测] mtime |
| 19:44 / 19:48 | B1 / B2 step2000 checkpoint 写入 | [实测] mtime |
| 19:57 | 两臂 source-exposure ledger 写入 | [实测] mtime |
| **21:50:39** | **Stage 6 代码提交 `5f02c86c4d3fa7389da1d5f4abbc16b6c4f0500d`** | [实测] `git log` |

历史 run 的 `training_verification.yaml` 记录 `git_commit: 7ec8a6b694e6ee66ac3c7027c3cb0d6c924cb981`。该 commit 的时间是 2026-10-25 22:02，且：

```
git cat-file -e 7ec8a6b:src/datasets/stage6_sampling.py
→ fatal: 路径 ... 不在 '7ec8a6b...' 中
```

即**运行时 Stage 6 代码尚未提交**；`7ec8a6b` 只是当时的 HEAD 基线，不是训练代码的版本。**[实测]**

### 4.2 已提交代码与当前树的一致性

`git diff --stat 5f02c86 HEAD -- src/ configs/ tools/ tests/` 只列出 Stage 6C 新增的 `tests/test_stage6c_phenobench.py` 与 `tools/compare_stage6c_phenobench.py`；`src/` 与 `configs/` 自 `5f02c86` 起**无改动**。四份关键文件在 `5f02c86` 与当前树的 SHA-256 逐字节一致：

| 文件 | `5f02c86` 内容 SHA-256 = 当前内容 SHA-256 |
|---|---|
| `src/datasets/stage6_sampling.py` | `5d8f490cd331b6d4ce9ac388560e0b99646e8a2ed9c14e2310b8106fd06cd251` |
| `src/datasets/agri_dataset.py` | `dadc876176e7599773b03ffa26645df164135ac3633898342d4118aaebfe96b0` |
| `src/train.py` | `08ba3e5394fd054444285fbee5fe5e13d7a71badcd0c7084c11a6f961074092e` |
| `src/datasets/source_ledger.py` | `4ae591bbc1aa0345cdcde21adfc684ab4e759c59ab08cc78025547e9247e9a44` |

### 4.3 结论

- 可以合理推断历史 run 使用的代码与 `5f02c86`/当前树一致（配置、params、console、产物计数与代码行为吻合）。**[推断]**
- 但**不能**从 run 产物密码学证明这一点：run 中没有代码文件哈希，`git_commit` 指向的 commit 不含 Stage 6 代码。
- 该 caveat 影响“完全可追溯”，不影响“seed 是否控制随机性”的代码级结论。Stage 6D 新 run 将**先提交代码、再启动训练**，并在启动前拒绝脏工作树，避免重复该问题。

---

## 5. 硬件来源审计（caveat）

`reports/stage6_preflight.md` 第 2 节记录：

- 执行开始时 GPU1 空闲；
- 进入 smoke 后，GPU1 被两个不属于本任务的外部训练进程占用（PID 3830083、3831669，各约 16.3 GB）；
- 首次 smoke OOM；
- 未终止任何外部进程；
- **经用户确认，smoke 阶段及其后的早期运行改用 GPU0（`cuda:0`）**；
- 正式对照运行的 `training_verification.yaml` 两臂均记录 `visible_device_index: 0`，无法从日志区分物理 GPU0/GPU1；两臂显存均约 23 GB。

结论：历史 B1/B2 的**物理 GPU 归属不可从产物唯一确定**；这不改变训练协议或 RNG（`torch.cuda.manual_seed_all` 对可见设备生效），但属于来源 caveat。Stage 6D 将在每个新 run 的 verification 中显式记录**物理 GPU 序号**（通过 `nvidia-smi` 与 `CUDA_VISIBLE_DEVICES` 解析），而不是只记录程序可见序号。**[实测 + 推断]**

---

## 6. 历史 checkpoint 能否计入独立预训练重复

### 6.1 判定依据

| 条件 | 结果 | 说明 |
|---|---|---|
| 记录了可复核的 seed | 是 | 三处配置/日志 + verification |
| seed 传给 Python/NumPy/PyTorch/CUDA | 是 | `src/train.py:194-198` |
| seed 传给 DataLoader worker | 不适用（`num_workers=0`）；generator 已 seed | `src/datasets/agri_dataset.py:284-285` |
| seed 传给采样器 | 是（natural `DistributedSampler`；capped `ProfileExposureSampler`） | 代码核验 |
| seed 传给 mask 随机过程 | 是（位置用全局 torch RNG） | `src/masks/multiblock.py:89-90` |
| 代码版本可密码学钉住 | **否**（run 时未提交；提交后文件逐字节一致，但 run 无代码哈希） | caveat |
| 物理 GPU 可唯一确定 | **否**（verification 只记可见序号） | caveat |

### 6.2 判定

- **seed 与作用范围**：充分证据，seed 确实控制训练随机性。→ 适用规则第一档。
- **代码/硬件来源**：存在 caveat，但不构成“seed 不可核验”或“无法证明 seed 控制随机性”。
- **因此**：历史 checkpoint 计入**一个带 caveat 的独立预训练重复**，不“冒充”为完全可追溯重复；最终报告会把它标为 `code_provenance: inferred_from_byte_identical_post_hoc_commit`，并同时给出“只用新增 run”的敏感性分析。
- 若用户/审稿人要求更严格，可把历史 checkpoint 完全移出主分析，只保留为 Stage 6C 历史证据；协议已预留该敏感性口径。

---

## 7. 新 seed 选择与配对方案

### 7.1 设计

| pair | seed | B1 run id | B2 run id | 用途 |
|---|---:|---|---|---|
| P0（历史） | 20261026 | `stage6_b1_natural_2000` | `stage6_b2_capped_no_backfill_2000` | 主分析（caveat）；敏感性分析中排除 |
| P1（新增） | 20261027 | `stage6d_b1_natural_seed20261027` | `stage6d_b2_capped_no_backfill_seed20261027` | 主分析 + 完全可追溯子集 |
| P2（新增） | 20261028 | `stage6d_b1_natural_seed20261028` | `stage6d_b2_capped_no_backfill_seed20261028` | 主分析 + 完全可追溯子集 |

- 同一 pair 的 B1/B2 使用**同一 seed**（配对设计，控制增强与数据顺序随机性，只让采样策略不同）。
- 不同 pair 的 seed 不同，构成独立预训练重复。
- 新 seed 与历史 `20261026` 不冲突；`20261027`、`20261028` 为顺序配对编号，选择依据是“与已有运行不冲突 + 可读 + 成对”，不是事后挑选。
- 每个新 run 同时写入：`data.seed`、`data.augmentation_seed`、（B2）`data.sampling.seed = pair seed`；并在 `meta.pretraining_seed`、`meta.pair_id`、`meta.run_id` 中显式记录。

### 7.2 运行数量

- 历史：B1 1 个、B2 1 个（带 caveat）。
- 新增：B1 2 个、B2 2 个（完全可追溯）。
- 每个 arm 最终 **3 个可核验 checkpoint**；每个 checkpoint 做 probe seeds 0/1/2。
- 预训练重复单位是 **checkpoint**，probe seed 只衡量下游 head 训练随机性，绝不冒充预训练重复。

---

## 8. 当前阻塞（训练/probe 未执行）

| 阻塞 | 事实 | 影响 | 处理 |
|---|---|---|---|
| GPU1 被占用 | 外部项目 PID 2594918，14,912 MiB，100% | 无法在 GPU1 启动 2,000-update 训练或 PhenoBench 特征提取 | 等待释放；或经用户**显式授权**改用 GPU0（Stage 6 有先例，但本轮提示词要求先报告） |
| 磁盘不足 | `/` 可用 187 G；单 run ~58 G；4 个新 run ~232 G | 无法把全部新 run 产物放在仓库所在 NVMe | 新 run 放 `/data/agrisa_jepa_runs/stage6d/`（可用 584 G）；或只保留 step2000 快照（需用户确认，因改变产物保留策略） |
| 无新 checkpoint | 训练未执行 | Phase 4/5 无法产出新 probe/统计 | 明确标注“未执行”，不伪造任何数字 |

**未执行的操作**：2,000-update 预训练、PhenoBench 特征缓存/线性 probe、新增 checkpoint 的统计汇总。**不得**在报告中把历史 Stage 6C 结果说成 Stage 6D 新结果。

---

## 9. 证据路径清单

| 证据 | 路径 |
|---|---|
| B1/B2 配置 | `configs/stage6_b1_natural_2000.yaml`、`configs/stage6_b2_capped_no_backfill_2000.yaml` |
| run 转储配置 | `runs/stage6_b1_natural_2000/params-ijepa.yaml`、`runs/stage6_b2_capped_no_backfill_2000/params-ijepa.yaml` |
| run 验证 | 两 run 的 `training_verification.yaml` |
| run 日志 | 两 run 的 `console.log`、`jepa_r0.csv` |
| 来源曝光 | 两 run 的 `source_exposure_ledger.csv`、`.summary.json`、B2 的 `stage6_sampling.json` |
| checkpoint | 两 run 的 `jepa-encoder-step2000.pth`、`jepa-step2000.pth.tar` |
| run manifest | `reports/stage6_run_manifest_b1.json`、`reports/stage6_run_manifest_b2.json` |
| 协议检查 | `reports/stage6_protocol_check.json` |
| 预检（GPU 偏差） | `reports/stage6_preflight.md` |
| 历史结果 | `reports/stage6_b1_natural_results.md`、`reports/stage6_b2_capped_results.md`、`reports/stage6_final_report.md` |
| Stage 6C 比较 | `reports/stage6c_phenobench_comparison.{md,csv,json}`、`reports/stage6c_phenobench_*` |
| 代码 | `src/train.py`、`src/datasets/agri_dataset.py`、`src/datasets/stage6_sampling.py`、`src/masks/multiblock.py`、`src/models/vision_transformer.py`、`src/helper.py` |

---

## 10. 给 Phase 2/3 的约束

1. 新 run 必须**从官方 checkpoint 重新初始化**（`meta.load_checkpoint: false`、`meta.read_checkpoint: null`、`meta.pretrained_checkpoint` = 官方权重、`pretrained_sha256` = `0382013c…`），不得从历史 B1/B2 step2000 续训。
2. 严格 2,000 optimizer updates；`max_optimizer_steps: 2000`、`grad_accum: 8`、`batch_size: 16`、`warmup_steps: 100`、LR/WD/EMA/mask 与 Stage 6 逐字段一致。
3. train manifest 必须是 `data/stage6/stage6_r3v3_train_view_nvme.csv`，SHA `26c55431…`；val 不参与预训练。
4. 除 seed/run 标识/存储目录外，B1/B2 与 Stage 6 base 配置的协议字段必须完全一致。
5. 每个 run 先提交代码、再启动；启动前记录 `git rev-parse HEAD`、`git status --porcelain`、`git diff --check` 与配置 SHA；运行中记录物理 GPU、seed、来源曝光、loss/LR/WD/EMA、耗时、显存、checkpoint SHA。
6. 每个 run 完成后先做完整性核验（2,000 updates、loss finite、checkpoint SHA、manifest SHA、来源曝光 ledger），再启动下一个；失败不得静默跳过。
7. probe 缓存键至少绑定 run label、checkpoint 完整 SHA、PhenoBench manifest SHA、probe seed、协议版本；SHA 不一致拒绝汇总。
