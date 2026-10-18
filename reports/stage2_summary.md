# AgriSA-JEPA Stage 2 — 最终报告

分支 `agri-stage2`。
本阶段代码与数据的交付 commit 为 `4982b55f9bb22f2a322743fc967cf7ac225695ed`（父 commit 为 Stage 1 的 `df6093a`）；本报告的 commit 为其后的分支尖端。
本报告只记录本阶段**实际执行并测量**的结果；未执行的工作在 Q17 明确列出。

**最终状态：`CONDITIONAL`** —— 见文末判据。

---

## A. 模型

### 1. 官方 ViT-H 权重是否正确加载？—— 是，且可验证为“逐张量精确”

| 项目 | 值 |
| --- | --- |
| 权重 URL | `https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| 本地路径 | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar`（在 Git 之外） |
| 字节数 | 10,358,004,345 |
| SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` |
| 源训练信息 | epoch 299，loss 0.16633，batch 32，world_size 64，lr 1e-3 |

该文件在本机已存在，因此本阶段**没有下载**，仅做完整性校验。摘要已固化进 configs 的 `meta.pretrained_sha256`，日后被替换会被立刻发现。

反序列化使用 `torch.load(..., weights_only=True)`，**成功**，即不反序列化任意 Python 对象；加载器会**报告**它实际使用的模式，而不是静默降级。

逐张量比对（去掉 DDP `module.` 前缀后）：

| 模块 | 模型张量 | 官方张量 | missing | unexpected | shape mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| encoder | 389 | 389 | 0 | 0 | 0 |
| predictor | 152 | 152 | 0 | 0 | 0 |
| target_encoder | 389 | 389 | 0 | 0 | 0 |

加载后与官方文件的 max-abs-diff 为 `encoder 0.0 / predictor 0.0 / target_encoder 0.0`。

关键设计：`src/pretrained.py` 先算出**显式 diff**，只要有任一 missing / unexpected / shape mismatch 就直接抛 `RuntimeError('Refusing to load ...')`。也就是说本项目**不存在**用未解释的 `strict=False` 掩盖不匹配、或只加载一小部分、其余随机初始化却宣称“用了官方权重”的路径。

证据文件：[`vith_checkpoint_loading.md`](vith_checkpoint_loading.md)、[`vith_checkpoint_loading.json`](vith_checkpoint_loading.json)。

### 2. Encoder、Target Encoder、Predictor 分别如何初始化？

- **Encoder**：Operation A —— 整份官方 encoder（389 张量）全部覆盖，无随机残留。
- **Target Encoder**：整份官方 `target_encoder`（389 张量）全部覆盖；随后按官方规则冻结、由 EMA 更新。注意：**没有**用 encoder 的权重去初始化 target encoder，而是直接采用官方各自的 EMA 副本。
- **Predictor**：官方 `predictor` **152 个张量全部继承**（`predictor_inherited: true`）。predictor 在本仓库中被重建为 `pred_depth=12 / pred_emb_dim=384`，与官方结构一致，因此可以逐个张量对齐 —— 这一点已被测试断言（见 Q15）。
- **优化器 / LR-WD 调度器 / GradScaler / epoch 计数器**：Operation A 阶段**有意全部丢弃**，让农业实验从干净状态开始。这与 Operation B（训练续跑）是两条不同代码路径，checkpoint 中分别记录。

### 3. 是否成功在单张 A6000 上执行全参数训练？—— 是

真实农业图像（`agrifield40k` 冻结前的 `pretrain_manifest.csv`），非合成数据。全参数训练，未冻结任何骨干层。两个真实运行：

| 运行 | 配置 | micro steps | optimizer steps | loss 首→末 | 峰值显存 |
| --- | --- | ---: | ---: | --- | ---: |
| 1（Operation A 起点） | `configs/stage2_vith_10steps.yaml` | 80 | 10 | 0.14814 → 0.14736 | 20,106 MiB |
| 2（Operation B 续跑） | `configs/stage2_vith_10steps_resume.yaml` | 80 | 10 | 0.14782 → 0.16089 | 21,623 MiB |

第 2 次运行的 `loss_first ≈ 第 1 次的 loss_last`，说明续跑是**接着**跑的，不是重新开始。
两次运行都验明：`loss_finite: true`、`encoder_updated: true`、`predictor_updated: true`、`ema_formula_verified: true`。

**注意**：loss 数值下降**不是**任何下游性能证据，本阶段也不据此宣称任何训练成功。它只证明梯度和优化器路径是活的、数值是有限的。

### 4. 是否启用了 SDPA、Activation Checkpointing 或梯度累积？

| 机制 | 状态 | 依据 |
| --- | --- | --- |
| SDPA（`F.scaled_dot_product_attention`） | **启用** | 数值等价验证：输出 max-abs-diff `0.0`，梯度 max-abs-diff `2.2e-11`；峰值 alloc 3058 → 2995 MiB；速度 1.008× |
| 梯度累积 | **启用**，`grad_accum=8` | 累积窗口是确定性的（按 `itr` 对齐窗口起点），窗口末尾才 `optimizer.step()`；有专门的等价性测试 |
| BF16 autocast + GradScaler | **启用** | `scaler.scale(loss / grad_accum).backward()`，反缩放后再 step；未出现被跳过的 step |
| Activation Checkpointing | **实现但关闭** | 全部 3 个模型（enc+target+predictor）都在显存里时峰值仅 ~25 GB，远低于 48 GB，开启只会白白付出重算代价。仍保留在 `meta.activation_checkpointing` 后 |

SDPA 的数值等价性很重要：它证明打开 SDPA **没有改变训练数学**，只是更省显存。证据：[`sdpa_equivalence.json`](sdpa_equivalence.json)。

### 5. 是否保持原始 I-JEPA 预测目标？—— 是，未做任何修改

- context encoder → predictor → 在 latent 空间预测 **EMA target encoder** 的目标块特征；
- 损失仍是官方 **smooth-L1**（`src/train.py` 沿用上游实现），未替换、未加权、未加项；
- mask 仍是官方 `MaskCollator` 的**随机多块**方案（1 个 context 块 + 4 个 target 块，比例与官方一致）；
- EMA 仍是官方的 momentum schedule。

本阶段**没有**实现任何农业结构感知 mask，**没有**改动预测目标。这是刻意的：结构感知 mask 属于后续阶段的可比性实验，必须建立在干净基线上。

---

## B. 性能

### 6. Batch 1、2、4 分别是否成功？—— 全部成功，且一直测到 32

见 Q7 表。1/2/4/8/16/32 **全部无 OOM**。

### 7. 各配置的峰值显存和 Images/sec

RTX A6000 48 GB，`CUDA_VISIBLE_DEVICES=1`（物理 1 号卡，程序内为 `cuda:0`；0 号卡负责桌面，全程未占用）。
固定 mask 几何（1 个 81 patch context 块 + 4 个 42 patch target 块），5 次 warmup + 10 个计时窗口，CUDA event + `synchronize()` 计时。

| Micro-batch | 等效 batch | optimizer step (ms) | micro step (ms) | Images/sec | 峰值 alloc (MiB) | 峰值 reserved (MiB) | loss 有限 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 1175.5 | 135.5 | 6.81 | 20,077 | 21,198 | 是 |
| 2 | 16 | 1170.9 | 135.0 | 13.66 | 20,085 | 21,522 | 是 |
| 4 | 32 | 1200.8 | 138.7 | 26.65 | 20,077 | 21,708 | 是 |
| 8 | 64 | 1330.4 | 155.2 | 48.11 | 20,506 | 22,736 | 是 |
| 16 | 128 | 2014.8 | 240.7 | 63.53 | 22,118 | 24,296 | 是 |
| 32 | 256 | 3394.8 | 413.2 | 75.41 | 25,376 | 26,706 | 是 |

**没有 OOM。** 峰值显存从未超过 ~25.4 GB（48 GB 的一半略多），吞吐随 batch 增长到 32（75.4 img/s，约为 batch 1 的 11 倍）。显存**不是**瓶颈。

诊断结论：峰值显存由**静态**占用主导 —— 参数 + 梯度 + Adam 一阶/二阶动量（encoder+predictor ≈ 13 GB）外加 target encoder 的 630 M 参数，因此 batch 1→4 的峰值几乎不动（激活占比很小）。

**Stage 1 ViT-B 的 152 ms/step 不可外推到 ViT-H。** ViT-H 的 encoder+predictor 约 1.28 B 可训练参数（ViT-B 约 86 M），且 GPU 上同时驻留三个完整模型。ViT-H 的实测 optimizer step 时间约 2.0 s（batch 16）。

证据：[`vith_single_gpu_benchmark.md`](vith_single_gpu_benchmark.md)、[`vith_benchmark.csv`](vith_benchmark.csv)、[`vith_benchmark.json`](vith_benchmark.json)。

### 8. 20 次真实 optimizer updates 和 checkpoint 恢复是否成功？—— 是

- 第一次运行：10 次真实 optimizer update，写出 checkpoint；
- 第二次运行（Operation B）：读回该 checkpoint，再跑 10 次，`optimizer_steps_total: 20`、`micro_steps_total: 160`、`checkpoint_epoch: 2`。
- 恢复内容全部核对：encoder/target/predictor 权重、优化器状态（4 个 param group）、GradScaler（5 个键）、epoch、micro_step、optimizer_step、EMA 进度、RNG 状态（python/numpy/torch/cuda 四路）、`loader_generator_state`、`mask_collator_step`、manifest SHA-256、模型配置、初始化权重 SHA-256、Git commit、硬件描述。
- 恢复后 encoder/predictor/target 相对 checkpoint 的变化量在 ~1e-4 量级，即**只有训练增量**，不存在被静默重置的情况。
- 恢复时若 manifest SHA-256 与当前 manifest 不符，会**直接报错**而不是继续。

### 9. Stage 3 的预计计算成本

按推荐配置（micro-batch 16、grad_accum 8、BF16、SDPA、关 checkpointing，实测 ~2.0 s/optimizer step）：

```
30,000 optimizer steps × ~2.0 s ≈ 60,000 s ≈ 16.7 小时（单张 A6000）
```

这是**基于 10 个计时窗口的估算**，不是已验证的完整运行；最终取决于 Stage 3 协议确定的等效 batch 与 step 数。本阶段**故意没有**执行任何长时间预训练。量级判断：单卡单日内可完成，不属于“多日战役”。

---

## C. 数据

### 10. 来源待核查样本减少了多少？—— 采用证据但**未自动采纳**：14,743 行仍全部隔离

- 语料 39,963 张图；`pending_audit` 行数 **14,743** 从未进入 include manifest，也从未进入冻结划分。
- 本阶段对这 10 个前缀逐一取证，结果：
  - **9 个前缀**（`gc`、`vcd`、`pags`、`rn`、`weds`、`mw`、`rl`、`saw`、`per`）拿到了**字面文件名匹配** —— 观测到的原始 ID 逐字符出现在该来源**自己发布的索引**里（Mendeley/figshare API、Kaggle 文件清单、作者仓库、Aalborg 官方档案及其 metadata CSV、Zenodo 记录）。
  - **1 个前缀**（`mc`/MuST-C）为 `supporting_only`：文件名每个组成部分都对得上 BonnData 官方档案，但官方命名是嵌套路径，而 AgriField 存的是扁平化字符串，因此**不是**字面匹配。
- **因此被采纳而减少的行数 = 0。**

这是刻意的。冻结由 Stage 1 制品的 `status`/`source` 两列驱动，`evidence_class` 只是**记录**，不是**决定**；标注证据**无法**让任何一行被静默纳入。冻结划分为 `PROVISIONAL`，`gate_met: false`，`pending_rows_confirmed: 0`。

理由（why 不直接采纳）：
1. 部分证据来自 HTTP Range 读取归档中央目录，本会话无法逐一独立复现；
2. `weds` 依赖镜像佐证 —— `zenodo.org` 在会话中途开始返回 403；
3. `mc` 本身只是 supporting-only；
4. 纳入 14,743 张图会改变**数据版本**和 Stage 3/4 的比较基准 —— 这是策展决策，不是工程开关。

因此把这条路径规范化为 5 步的晋升流程，记录在 [`stage2_source_index_evidence.md`](stage2_source_index_evidence.md)：策展人显式确认前缀映射 → 重跑来源/许可审计 → 对新增行重跑去重筛查（它们此前从未在 include manifest 中，从未被去重）→ 在新数据版本下重新冻结 → 在 Stage 3 协议中记录变更。

### 11. 重复候选人工复核材料是否完成？—— 材料**已完成**，人工复核**未完成**

- `data/duplicate_review.csv`：**784** 条候选边（776 条感知近重复 + 8 条精确重复），`review_status` **全部**为 `pending_manual_review`，`reviewer` 列**全为空**（0 条被预填）。
  - 其中跨划分 247 条，影响 manifest 11 条，同划分筛查 526 条；跨划分或影响 manifest 的共 258 条被隔离。
- `reports/duplicate_review/index.html` + `thumbs/`（1,568 张缩略图，0 失败）：逐边并排对照，供人工裁决。
- `reports/duplicate_review_protocol.md`：记录判定标准、必须回答的问题、以及“如何算同一张图”的规则。
- **没有任何一条**被自动填成“人工确认为重复”。这是硬约束，已遵守。

隔离规则（`stage2b_isolation_rule`）已执行：所有跨划分或影响 manifest 且仍 `pending_manual_review` 的候选边，其**两个端点**都被移出冻结划分 —— 共标记 405 个样本，其中 11 个确实来自 include manifest（11 个均为 train，0 个 val）。这是一个**隔离规则**，不是完成的去重。

### 12. 当前数据能否正式冻结？—— **不能，只能 `provisional`**

冻结产物已生成且自洽：

| 项目 | 值 |
| --- | --- |
| 数据版本 | `agrifield40k-v0` |
| 状态 | `PROVISIONAL`（`gate_met: false`） |
| 冻结 train / val | 12,447 / 3,145（共 15,592） |
| train ∩ val | 空（写盘时断言） |
| 固定 revision | HF `7f530c67572b6a572467adab840cdbbd7029b112` |
| train SHA-256 | `d8a179b81f2052c6d66d2f43d26505652e95b1090560d13c1e9d46efe3b27037` |
| val SHA-256 | `4a7338d9f105164dd97fbf5719adef4b03d0e61e45d337cea07f24fbc1905a45` |

门禁未通过的原因：14,743 行仍 `pending_audit`，784 条重复候选边未经人工裁决，iNatWeeds 的逐图署名/观测字段尚未解决，发布方的序列级划分规则未被本审计独立验证。

冻结 manifest 中另记录 `unresolved_prefix_evidence` / `unresolved_prefix_adoption` 两列：证据等级与“是否采纳”，前者已填，后者**全部为 `not_adopted`**。

（另：本次重跑生成的冻结 CSV 与上一版**逐字节相同**，说明审计工具是确定性的，证据记录不影响划分。）

### 13. 是否仍有许可、来源和测试集泄漏风险？—— 三者都仍有

- **来源**：见 Q10。9 个前缀证据可采纳但未采纳；`mc` 只有支持性证据。
- **许可**：`gc` 若确认则为 CC-BY-SA-4.0（**传染性**许可，会影响整份合集的再分发条款）；`per` 为 MIT；其余候选多为 CC-BY-4.0 / CC0-1.0。**这些候选许可尚未加到任何一行**，`confirmed_sources` 仍然只有 Stage 1 的 6 个来源。iNatWeeds 是 `per_image_metadata`，逐图署名/观测字段尚未解决。
- **测试集泄漏**：已知风险，且已在冻结 manifest 的 `known_leakage_risks` 中记录：
  - 发布方的 80/20 划分是**其自身声明**，本审计**未**逐图独立验证（尤其序列级来源）；
  - PhenoBench 相关行**整体排除**（9,606 行，含任何接触到 PhenoBench 的连通分量），以避免与下游评测集重叠；
  - 405 个候选重复端点被隔离（其中 11 个原本在 include manifest 内）；
  - 247 条跨划分候选边意味着**同一采集序列**可能同时出现在 train 与 val —— 这是最需要人工裁决的一类。

### 附：数据分布的一个关键事实（任务 F 要求）

**LUCASVision 占训练集的 8,944 / 12,458 = 71.79%**（验证集 2,245 / 3,145 = 71.38%）。第二大来源 VegAnn 仅 9.99%，LUCASVision 是其 7.2 倍。最小两个来源为 SorghumWeed(137)、RadishWheat(427)，跨度近两个数量级。LUCASVision 也是唯一被拆到多个前缀（12 个 `lvb*`）的来源。

报告中给出了两种配比方案（上限 5,000 → 58.73%；上限 1,000 → 23.61%）供选择，但**本阶段未擅自改变数据配比**。详见 [`data_distribution.md`](data_distribution.md)。

---

## D. 工程

### 14. 新增或修改了哪些文件？

**修改（3）**
- `src/train.py` —— Operation A/B 分离、BF16 scaler、SDPA、activation checkpointing、确定性梯度累积、按 **optimizer step** 计数的 LR/WD/EMA 调度、按 optimizer step 快进的续跑、checkpoint 字段扩展、`git_commit()` / `hardware_report()`。
- `src/models/vision_transformer.py` —— `Attention(use_sdpa=...)`；`Block(activation_checkpointing=...)`（`use_reentrant=False`，仅 `self.training and x.requires_grad` 时启用）；`model_architecture()`、`VIT_EMBED_DIMS`。
- `src/helper.py` —— `init_model(..., use_sdpa, activation_checkpointing)` 透传到 encoder 与 predictor。

**新增（代码 / 配置 / 测试）**
- `src/pretrained.py` —— Operation A 加载器（SHA-256、去 `module.` 前缀、显式 diff、拒绝加载）。
- `tools/bench_vith.py`、`tools/verify_sdpa.py`、`tools/verify_vith_pretrained.py`、`tools/stage2_data_audit.py`。
- `configs/stage2_vith_pilot.yaml`、`stage2_vith_10steps.yaml`、`stage2_vith_10steps_resume.yaml`、`stage3_baseline_candidate.yaml`。
- `tests/test_stage2_vith.py`。

**新增（数据 / 报告）**
- `data/frozen/agrifield40k-v0_train.csv`、`_val.csv`、`freeze_manifest.json`；`data/duplicate_review.csv`。
- `reports/`：`vith_checkpoint_loading.{md,json}`、`vith_single_gpu_benchmark.md`、`vith_benchmark.{csv,json}`、`vith_benchmark_sweep.{csv,json}`、`sdpa_equivalence.json`、`stage2_data_audit.{md,json}`、`stage2_source_index_evidence.md`、`data_distribution.md`、`data_freeze.md`、`duplicate_review_protocol.md`、`duplicate_review/`（HTML + 1,568 缩略图）、`stage3_baseline_protocol.md`、本文件。

**没有进入 Git 的**：10 GB 官方权重、10 GB checkpoint、数据集图像、`runs/`（已在 `.gitignore`）。本次提交 20.76 MB，最大单文件 6.9 MB（冻结 train CSV）。

### 15. 新增测试及全部 pytest 结果

- 全套（CPU）：**18 passed, 4 skipped —— 12.28 s**。
  - 4 个 skipped 是**门控**测试（需 `AGRISA_VITH_CHECKPOINT` 指向 10 GB 官方权重才跑）。
- 带门控：`AGRISA_VITH_CHECKPOINT=... pytest tests/test_stage2_vith.py` → **17 passed —— 549.82 s**。
- `tests/test_stage2_vith.py` 覆盖：ViT-H 结构（dim/depth/heads/patch/predictor 深度与宽度）、`model_architecture()`、`compare_state_dicts`、`strip_module_prefix`、`describe_payload`、**`_load_module` 在 mismatch 时拒绝加载**、mask 索引合法性（`min_keep`、上界）、encoder 与 predictor 每个预测块**逐块不相交**（25 个 seed）、EMA 更新规则、梯度累积的 loss/梯度等价性。

### 16. 当前 Git 分支和 commit

- 分支：`agri-stage2`，已推送至 `origin/agri-stage2`（新建分支，非强推）。
- 交付 commit：`4982b55f9bb22f2a322743fc967cf7ac225695ed`。
- `agri-stage1`（`df6093a`）与 `main`（`52c1ae9`）**未被改写、未被合并**。

### 17. 哪些事项尚未完成（如实列出）

1. **784 条重复候选边的人工裁决** —— 0 条已复核，材料齐备。
2. **10 个前缀的来源/许可采纳** —— 证据已记录，采纳需策展人决策 + 新数据版本。
3. **iNatWeeds 逐图署名/观测字段** —— 尚未解决（数据已保留在 `data/all_samples.csv`）。
4. **发布方序列级划分规则的独立验证** —— 未做。
5. **正式数据冻结** —— 目前为 `provisional`。
6. **农业结构感知 mask** —— 未实现（**按指令禁止实现**）。
7. **长时间/多 epoch 预训练** —— 未执行（**按指令禁止执行**）。
8. **Stage 3 三臂（A/B/C）协议** —— 仅输出协议与候选配置，**未训练**。
9. `configs/stage3_baseline_candidate.yaml` 仍是**候选**，未经运行验证；其中原先写入的 `max_optimizer_steps` 因 `src/train.py` 未实现该预算开关而**被删除**，以免暗示一个不存在的控制项。

### 18. 是否建议进入 Stage 3 正式标准 JEPA 预训练？

**建议：可以进入 Stage 3 的协议定稿与短程试点，但不要直接启动 Stage 3 的全量正式预训练。**

理由：
- **工程侧已通过**：初始化精确、单卡全参数训练可跑、20 步真实 update 与续跑可恢复、checkpoint 字段完整、预算量级（~16.7 h/30k step）单日可承担、无 OOM。
- **数据侧未通过门禁**：14,743 行来源待定、784 条重复候选未裁决、LUCASVision 占 71.8% 的严重不均衡尚未决策。在这些问题定稿前启动正式预训练，等于把数据版本的不确定性固化进模型权重。

---

## 最终状态：`CONDITIONAL`

**判据：**

- ✅ 模型初始化：官方 ViT-H 三份权重**逐张量精确加载**（diff = 0.0），`strict=True`，无掩盖。
- ✅ 单卡训练：单张 A6000 上全参数训练成功，batch 1–32 均无 OOM。
- ✅ Checkpoint 与恢复：20 次真实 optimizer update，续跑正确，字段完整（含 RNG/scaler/EMA/manifest 哈希/git commit）。
- ✅ 计算预算：~2.0 s/step，30k step ≈ 16.7 h，单卡单日可承担。
- ✅ 预测目标：保持官方 I-JEPA 目标，未加农业结构 mask，未改损失。
- ❌ 数据版本：**未通过**来源/许可门禁（`gate_met: false`，14,743 行待定，784 条重复候选待人工裁决）。

**为什么不是 READY**：`READY` 要求“单卡训练、初始化、Checkpoint **和数据版本**均通过预定验收”。数据版本一项明确未通过，因此不能给出 READY。**特别说明：本结论不是基于“没有 OOM”得出的** —— 恰恰相反，显存是最不成问题的一环（峰值仅为 48 GB 的一半）；卡住 READY 的是数据可追溯性，不是算力。

**为什么不是 BLOCKED**：`BLOCKED` 指“ViT-H 单卡训练或关键数据条件尚未通过”。ViT-H 单卡训练**已经通过并实测**；数据条件虽有缺口，但缺口是**已知的、被隔离的、有明确晋升路径的**（14,743 行不在冻结划分内，784 条候选已隔离，路径见 `stage2_source_index_evidence.md`），而非未知或不可解。

**解除 CONDITIONAL 的最小条件**：① 裁决 784 条重复候选边（至少完成 247 条跨划分边）；② 由策展人对 9 个可采纳前缀给出显式映射决策并重跑筛查；③ 在这两项完成后升级数据版本并重新冻结。三者完成后即可进入 Stage 3 正式预训练。
