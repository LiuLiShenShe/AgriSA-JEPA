# Stage 6C — PhenoBench 冻结线性探针评价协议

**分支**：`agri-stage6c-phenobench-eval`（从 `agri-stage6-controlled-pretraining`
HEAD `f057721530d687fc37cba864106b39f484298ca5` 切出）

**目的**：在完全相同、冻结的评价协议下，比较三套 ViT-H/14 冻结特征在
PhenoBench 语义分割（semantics）上的表现，回答 Stage 6 末提出的三个研究问题：

1. B1（natural）/ B2（capped no-backfill）是否优于官方 I-JEPA 的 PhenoBench
   冻结特征？
2. B2 相对 B1 的表现方向如何？
3. 是否值得进入 GWHD 检测、独立预训练 seed 或更长训练？

---

## 1. 三臂定义

| | Arm A（对照） | Arm B1 | Arm B2 |
| --- | --- | --- | --- |
| 骨干 | ViT-H/14 | 同一 ViT-H/14 | 同一 ViT-H/14 |
| 权重 | 官方 `IN1K-vit.h.14-300e.pth.tar` | `runs/stage6_b1_natural_2000/jepa-encoder-step2000.pth` | `runs/stage6_b2_capped_no_backfill_2000/jepa-encoder-step2000.pth` |
| 农业持续预训练 | 无 | 2000 optimizer updates；自然源采样 | 2000 optimizer updates；Stage 5E.1 物理源 capped no-backfill |
| 训练数据版本 | — | R3v3_license_verified train；147,277 张；审计状态 CONDITIONAL | 同 B1 |
| 骨干是否冻结 | 是 | 是 | 是 |

**R3v3 训练语料状态仍为 CONDITIONAL**。本阶段不更改该状态。审计报告见
`reports/phenobench_data_audit.md` 与 `reports/phenobench_leakage_check.json`。
PhenoBench 不进入任何 JEPA 自监督预训练语料。

---

## 2. 固定评价协议（三臂逐字段一致）

| 项 | 值 |
| --- | --- |
| 骨干 | ViT-H/14，冻结，`eval()`，`requires_grad=False` |
| 特征提取 | 224×224 tiles，BF16（CUDA 可用时），patch tokens 16×16，embed 1280 |
| 分辨率适配 | 1024→5×5=25 个 224×224 tile，stride 200，无 padding，全覆盖，重叠 logits 平均拼接 |
| 分割头 | `Linear(1280 → 3)`，3,843 可训练参数 |
| 类别映射 | partial crop 3→crop 1、partial weed 4→weed 2（官方 `convert_partial_semantics`） |
| 优化器 / LR / WD | AdamW / 1e-3 / 0.0 |
| batch / epochs | 32 tiles / 10 |
| probe seeds | 0, 1, 2（仅衡量分割头/训练随机性；每个 arm 只有一个预训练 checkpoint） |
| 数据增强 | 无（特征已缓存，协议固定） |
| 精度 | 特征 BF16，head FP32 |
| 设备 | 物理 GPU1（RTX A6000，`CUDA_VISIBLE_DEVICES=1`，映射为 `cuda:0`） |
| 训练 | 只用官方 train（1,407） |
| 评价 | 只用官方 val（772），不使用 test（693，无公开标注） |

`tools/compare_stage6c_phenobench.py` 在聚合前逐字段断言三臂协议一致，并拒绝
任何协议偏差（包括 checkpoint SHA、seed 数、val 预测数、自实现/官方指标偏差）。

---

## 3. 指标与交叉核验

- 必须报告：mIoU、Soil IoU、Crop IoU、Weed IoU。
- 每个 seed 的指标、mean±std、配对 seed 差值，以及 B1−A、B2−A、B2−B1。
- 训练 loss、probe 耗时、显存峰值、模型参数量。
- checkpoint SHA、数据清单 SHA、代码 commit、协议参数、输出 SHA。
- 平凡基线（全部预测 soil）mIoU = 29.97（审计记录；本地重算偏差须在 ±0.05 内）。
- **官方评价器**（`phenobench-eval --task semantics`）与**自实现逐类指标**
  逐 seed 偏差必须小于 0.01 个百分点；若不一致，先调查，不得挑选更有利的指标。
  官方评价器只打印 crop/weed/mIoU 三行，soil 取自习实现 `IoUAccumulator`，
  并通过 mIoU 恒等式间接交叉核验。

---

## 4. 缓存与隔离

- 特征缓存按 `arm-label + checkpoint SHA[:16]` 绑定（`src/downstream/features.cache_key`），
  绝不复用旧 Stage 3B/4B/5A 特征。
- 每臂 train+val 特征缓存约 35–36 GB；一次只处理一个新 arm。
- 缓存目录：`runs/stage6c_phenobench/cache/`
- 输出：`A_official/`、`B1_natural/`、`B2_capped/` 各自独立目录。
- 每臂完成后确认 result JSON、3 个 seed 结果、全部 772 张 val 预测、官方评价器
  原始输出和 SHA 记录落盘，再精确删除该 arm 的 `cache/<cache_key>/` 子目录
  （不删除 cache root、不删除用户数据或历史产物）。

---

## 5. 禁止事项（硬约束）

- 不改 tile size、stride、类别映射、head、optimizer、LR、epochs 或 checkpoint 选择规则。
- 不做 backbone 微调，不加 Adapter/Decoder，不使用 test 调参。
- 不将 3 个 probe seeds 写成 3 个独立 pretraining seeds。
- 本阶段**不启动任何新的 JEPA 预训练**。
- 旧 Arm B（500-step，v0）和 Stage 3B/4B 结果不混入 A/B1/B2 统计。

---

## 6. 来源依据

| 依据 | 路径 |
| --- | --- |
| 评价协议 | `reports/phenobench_evaluation_protocol.md` |
| 配置 | `configs/phenobench_armA_linear.yaml`、`configs/phenobench_armB_linear.yaml` |
| 实现 | `src/downstream/phenobench_probe.py`、`probe.py`、`features.py`、`phenobench_data.py`、`tiling.py` |
| 比较脚本 | `tools/compare_stage6c_phenobench.py` |
| 测试 | `tests/test_stage6c_phenobench.py` |
| 数据审计 | `reports/phenobench_data_audit.md`、`phenobench_file_manifest.sha256.txt`、`phenobench_leakage_check.json` |
| 历史 A baseline | `runs/stage3a_phenobench/armA/result.json`、`reports/phenobench_armA_results.md` |
| Stage 6 manifest | `reports/stage6_run_manifest_b1.json`、`stage6_run_manifest_b2.json` |
