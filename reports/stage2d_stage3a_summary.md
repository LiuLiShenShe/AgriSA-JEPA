# Stage 2D + Stage 3A 总结

**核心问题：官方 I-JEPA ViT-H/14 经过 500 步农业持续预训练后，
在独立农业语义分割任务上是否获得可测量的表征收益？**

**答案：是，但幅度小且需谨慎解读。** 在同一冻结骨干 + 同一线性分割头 +
同一 224 tiling/stitching + 同一 3 seeds 的协议下，农业持续预训练模型的
mIoU 比官方模型高 **+1.45 个百分点**，3 个 seed 全部为正；
crop IoU 的 +0.92 最为稳定。

这**不是**"结构感知 Mask 有效"的证据——Arm B 用的仍是官方随机多块掩码。

---

## 1. 一句话结论

| 问题 | 答案 |
| --- | --- |
| v1 是否修复了两处结构泄漏 | 是：ACW 全序列入 train；iNatWeeds 观测级划分，跨 split observation 31 → **0** |
| v1 是否可宣称 READY | **否**，仍为 PROVISIONAL（784 候选未人工审查、14,743 待定来源未采纳） |
| PhenoBench 是否完整获得 | 是，官方 v1.1.0，train 1,407 / val 772，官方评价器已跑通 |
| 官方 ViT-H mIoU | **71.68 ± 1.35** |
| 农业 500-step mIoU | **73.13 ± 1.09** |
| 差值 | **+1.45**（配对 +1.68 / +0.74 / +1.93） |
| 是否支持扩大预算 | **A（谨慎版）：显示积极趋势，值得进入更大预算基线实验** |

---

## 2. Stage 2D：数据划分修复

详见 `reports/stage2d_split_repair.md`、`reports/stage2d_leakage_check.md`。

### 2.1 ACRECropWeed

- 791 张，跨 2022-10-06（159 张）与 2022-10-10（632 张），帧间最大间隔约 87 分钟。
- 无法证明是独立采集 session（同机器人、同任务、同地块），
  **没有按任意时间阈值人为制造独立序列**。
- 采用最保守且明确记录的方案：**整个序列只进入训练划分**，
  v0 val 中的 120 张移入 train。
- 代价：v1 val 不再包含 ACRECropWeed。已明确记录。

### 2.2 iNatWeeds

- 分组单位：`observation_url` 解析出的 `observation_id`。
- 1,309 张图像，**1,227 个 observation**，全部有可解析 id（缺失 0）。
- 固定种子 `20261019`，排序后洗牌，贪心填充 val 至来源比例
  （278/1309 = 21.24%，目标 278 张）。
- 结果：val 278 / train 1,031，与 v0 比例一致；
  **跨 split observation：31 → 0**。

### 2.3 版本与数量

| | v0 | v1 |
| --- | ---: | ---: |
| train | 12,447 | **12,567** |
| val | 3,145 | **3,025** |
| 合计 | 15,592 | 15,592 |

| 文件 | SHA-256 |
| --- | --- |
| `agrifield40k-v1_train.csv` | `d63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435` |
| `agrifield40k-v1_val.csv` | `86efe5156291950b25af692d41b5f1b4cee437dd3ba2a44ce39e2cf18984f944` |

v0 文件字节级未改动（哈希与 `reports/data_freeze.md` 一致）。
连续两次构建产物哈希完全一致（确定性已验证）。

### 2.4 仍未解决

- 784 条重复候选 **0 条人工审查**；
- 14,743 张 `pending_audit` **未采纳**；
- hash 阈值以下的同场景不同次采集无法排除；
- 状态保持 **PROVISIONAL**，未为通过门禁而修改。

---

## 3. Stage 3A：PhenoBench 下游评价

详见 `reports/phenobench_data_audit.md`、`reports/phenobench_evaluation_protocol.md`、
`reports/phenobench_armA_results.md`、`reports/phenobench_armB_results.md`。

### 3.1 数据

- PhenoBench **v1.1.0**，官方 zip 7,630,658,167 字节，
  SHA-256 `a310c1c1dcfec4ca4897097fd38c02af39ea24112f658fd9408d5deb3c7c8bfc`。
- train 1,407（images + semantics + 4 类实例/可见性标注），val 772，test 693（无标注）。
- 许可：页面写 CC-BY-SA-4.0，包内 README 写 **CC BY-NC-SA 4.0**；
  按更严格的 NC 处理，仅本机科研使用，不分发。
- 官方 val **未用于训练**；PhenoBench 图像**未进入** JEPA 预训练。

### 3.2 分辨率适配

1024×1024 → **5×5 = 25 个 224×224 tile**：
`n=ceil(1024/224)=5`，`stride=ceil(800/4)=200`，offset `0,200,400,600,800`。
**无 padding、无裁剪、每个像素都被覆盖**，相邻 tile 重叠 24 px。
拼接时对重叠像素的 logits 取平均再 argmax。
因此自实现指标与官方指标都在完整 1024×1024 上计分。

### 3.3 核心结果

官方 `phenobench-eval --task semantics`，val 772 张，3 seeds：

| 指标 | Arm A 官方 ViT-H | Arm B 农业 500-step | 差值 (B-A) | 配对差值 (3 seeds) |
| --- | --- | --- | ---: | --- |
| **mIoU** | **71.68 ± 1.35** | **73.13 ± 1.09** | **+1.45** | +1.68 / +0.74 / +1.93 |
| Soil IoU | 97.61 ± 0.01 | 97.70 ± 0.003 | +0.095 | +0.086 / +0.097 / +0.103 |
| **Crop IoU** | **81.46 ± 0.17** | **82.38 ± 0.16** | **+0.92** | +0.94 / +0.84 / +0.99 |
| **Weed IoU** | **35.98 ± 3.89** | **39.32 ± 3.12** | **+3.34** | +4.03 / +1.27 / +4.72 |

- 平凡基线（全预测 soil）mIoU = **29.97**；两 arm 都远高于它。
- 自实现与官方指标在每个 seed 上最大偏差 **< 0.005 pp**（双向交叉校验通过）。
- 分割头参数量 3,843（两 arm 相同），峰值显存 117.5 MiB（特征已缓存），
  probe 训练 ~70–100 s/seed。

### 3.4 如何解读

- **支持进一步做更大预算实验**：mIoU、crop、weed 三个指标在 3 个 seed 上
  方向全部一致；crop 的配对标准差只有 0.08，是非常稳定的信号。
- **不能过度解读**：500 步 ≈ 5.1 个 v0 epoch，提升 1.45 pp，量级上更像轻微
  领域适配；weed 波动大；n=3 不足以做显著性推断。
- **不是结构感知 Mask 的功劳**：Arm B 使用官方随机多块掩码。
- **数据版本不一致**：Arm B 训练于 v0，v1 在其后建立。最终公平实验必须
  从同一官方初始化、在最终统一数据版本上重新训练 Arm B。

---

## 4. Stage 2C 工程遗留修复

### 4.1 预算终点日志缺失

**已修复。** `src/train.py` 中 `log_stats()` 现在在 budget/interrupt 的
`break` **之前**执行，因此执行最后一次 optimizer update 的 micro-step
一定写入 CSV。模型更新、scheduler、EMA、停止时机语义均未改变。

回归测试：
- 静态顺序断言（`log_stats()` 在 break 前、在 `optimizer_steps += 1` 后）；
- **真实 CPU 训练**：8 张图、batch 2、grad_accum 2、budget 3 更新 →
  CSV 恰好 6 行，最后一行是 epoch 2；interrupt 路径同样验证。

### 4.2 CSVLogger 静默追加

**已修复。** 新实验（`load_model=False`）会**截断**日志文件；
续训（`load_model=True`）才追加，且先校验表头一致，不一致直接报错。

### 4.3 Resume 一致性

- 原有 Stage 2C 的逐位一致性结论仅在 `num_workers: 0` 下验证过；
  本轮没有推翻该结论，新增的续训测试使用 `num_workers: 0`。
- 测试验证：中断于 update 2（4 行）→ 续训到 budget 10，总日志 20 行，
  `(epoch, itr)` 无重复、无丢失。
- **未验证** `num_workers > 0` 的逐位一致性；不得沿用该结论。

---

## 5. 测试

完整 pytest（见最终验收报告）：原有 + 新增共 150+ 项通过。
新增覆盖：

1. ACRECropWeed 组划分不交叉 ✓
2. iNatWeeds observation_id 不交叉 ✓
3. v1 manifest 确定性与 Hash ✓
4. v1 保守重复隔离 ✓
5. PhenoBench 图片/标签加载 ✓
6. 语义标签 ID 处理（与官方 devkit 交叉校验）✓
7. ViT-H token 空间形状 ✓
8. Tiling / 全覆盖 / Stitching ✓
9. Segmentation Probe forward/backward ✓
10. 官方评价格式 ✓
11. Arm A/B 权重正确加载 ✓
12. 训练日志预算终点记录 ✓
13. Checkpoint 恢复兼容性 ✓

---

## 6. 下一阶段是否具备结构感知 Mask 原型的工程条件

**具备（有条件）。**

已就绪：
- v1 确定性冻结划分 + 可复用的分组/泄漏检查工具；
- 官方 ViT-H 权重加载与 500-step 训练/恢复闭环；
- 独立下游评价管线（PhenoBench 官方 split + 官方评价器 + 交叉校验）；
- 两 arm 协议一致性断言工具，可防止后续实验偷偷改配置；
- 日志与恢复回归测试。

仍需先做：
1. 从同一官方初始化、在 **v1** 上重训 Arm B，并与本轮探索性结果对照
   （这是把 +1.45 pp 变成可引用结论的前提）。
2. 扩大预算（至少数千到数万 update），并保持 Arm B/C 同预算独立训练。
3. 完成 784 条重复候选的人工审查，决定 v1 是否可升为 READY。
4. 若要评价 Mask 设计本身，必须新增 Arm C（结构感知 Mask），
   与 Arm B 同初始化、同预算、同数据版本、同下游协议。

---

## 7. 复现命令

```bash
# Stage 2D
./.venv/bin/python -m tools.build_split_v1
./.venv/bin/python -m tools.verify_split_version \
  --train data/frozen/agrifield40k-v1_train.csv \
  --val   data/frozen/agrifield40k-v1_val.csv --label agrifield40k-v1

# Stage 3A
./.venv/bin/python -m src.downstream.phenobench_probe --arm A --stage features
./.venv/bin/python -m src.downstream.phenobench_probe --arm B --stage features
./.venv/bin/python -m src.downstream.phenobench_probe --arm A --stage train_eval --seed 0 1 2
./.venv/bin/python -m src.downstream.phenobench_probe --arm B --stage train_eval --seed 0 1 2
./.venv/bin/python -m tools.summarize_stage3a
./.venv/bin/python -m tools.check_arm_protocol

# 测试
./.venv/bin/python -m pytest tests/ -q
```