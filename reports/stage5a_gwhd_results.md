# Stage 5A — GWHD 2021 目标检测真实 A/B 结果

> 运行日期：2026-10-23。设备：仅 GPU 1（RTX A6000 48 GB），`CUDA_VISIBLE_DEVICES=1`。
> 协议预注册见 `reports/stage5a_gwhd_protocol.md`；数据审计见
> `reports/stage5a_gwhd_data_audit.md`；机器可读对比见
> `reports/stage5a_gwhd_comparison.{csv,json}` 与
> `reports/stage5a_gwhd_ab_summary.md`。
>
> 两臂唯一差异是 Backbone 权重：A = 官方 I-JEPA ViT-H/14（IN1K 300e），
> B = 农业 JEPA v1（AgriField-40K 2,000 次更新）。输入、划分、Neck/Head、
> 预算、增强、优化器、评价脚本完全相同。
>
> 共 4 次 20K 运行：seed 0（预注册主结果）+ seed 1（一致性复核）。

## 0. 结论先行

1. 检测管线**真实跑通**：20,000 iterations × 2 图/步，官方 train 训练，
   官方 val 每 2,000 步评估，官方 test 只在最后评估一次。4 个运行目录全部
   含 `protocol.json` / `train_log.csv` / `eval.jsonl` / `results.json` / `head.pth`。
2. **val**：A AP50:95 = 0.0759 / AP50 = 0.2652；B = 0.0828 / 0.2737（seed 0）。
   两 seed 平均：A 0.0753±0.0006 / 0.2630±0.0022，B 0.0837±0.0009 / 0.2780±0.0043。
3. **test**：A AP50:95 = 0.0155 / AP50 = 0.0564；B = 0.0187 / 0.0785（seed 0）。
   两 seed 平均：A 0.0146±0.0009 / 0.0582±0.0018，B 0.0189±0.0002 / 0.0786±0.0001。
4. **两个 seed 的差值方向一致**（test ΔAP 分别为 +0.0032 / +0.0053，
   ΔAP50 +0.0220 / +0.0187），因此这不是单次运行的方向抖动。
   但 **AP 幅度仍然很小**（val +0.0084、test +0.0043 AP），且 A 自身的
   seed 间波动（test 0.0155 vs 0.0137）与臂间差同量级。
   本阶段的正确表述是“**方向在 2 seed 上一致、AP 幅度小；AP50 的差值相对更稳**”，
   不能写成“检测取得显著提升”。
5. Test AP 远低于 val AP（A：0.0759 → 0.0155；B：0.0828 → 0.0187），说明
   GWHD 2021 官方 test 存在明显跨域偏移（test 含 18 个采集域）。
   这是数据集与协议的属性，不是某一臂的问题。
6. 小目标是主要瓶颈：AP_small ≈ 0.0004–0.0031，而 AP_large ≈ 0.016–0.119。
   518 输入下麦穗仅 1–3 个 patch，冻结特征的表达能力是上限。
7. 发现并修复了一个 checkpoint 保存缺陷（丢失 Neck 权重），不影响任何已报告
   指标，详见 §8.1。

## 1. 运行清单

| run | Backbone | 预训练 | seed | 机器可读产物 | 状态 |
|---|---|---|---:|---|---|
| A_seed0 | 官方 I-JEPA ViT-H/14 | IN1K, 300 epochs | 0 | `reports/stage5a_gwhd_runs/A_seed0/` | 完成 |
| B_seed0 | 农业 JEPA v1 | AgriField-40K, 2,000 updates | 0 | `reports/stage5a_gwhd_runs/B_seed0/` | 完成 |
| A_seed1 | 同上 A | 同上 | 1 | `reports/stage5a_gwhd_runs/A_seed1/` | 完成 |
| B_seed1 | 同上 B | 同上 | 1 | `reports/stage5a_gwhd_runs/B_seed1/` | 完成 |

每个目录含 `protocol.json`（含数据与权重 SHA-256）、`train_log.csv`（每 50 步）、
`eval.jsonl`（每次 val）、`results.json`（最终 val + test + 成本）。
完整目录（含 `head.pth`）在 `runs/stage5a_gwhd/<run>/`，该目录被 `.gitignore`
忽略（约 230 MB 的权重副本），机器可读的配置与指标已复制到上面的 reports 路径。

Backbone SHA-256：

- A `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`
  （`/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar`）
- B `1c4e1c484d89245ea3702c9603c62355bdfcd07be8594df020e6e3cf4d94d23f`
  （`/home/test/agrisa_runs/stage3b_random_2000/jepa-encoder-step2000.pth`）

`tools/stage5a_compare_gwhd.py` 在比较前逐字段校验
`input_size / iterations / batch_size / lr / momentum / weight_decay /
warmup_iterations / eval_every / min_box_size / hflip_prob / score_thresh /
seed / architecture / anchors / optimizer / scheduler`，并拒绝
`backbone_sha256` 相同的“对比”；任一字段不同直接拒绝输出。

## 2. 协议摘要（与预注册一致，未改动）

| 项 | 取值 |
|---|---|
| 数据 | GWHD 2021 官方 train / val / test（3,655 / 1,476 / 1,381） |
| 输入 | 1024×1024 → 518×518，双线性 + antialias，ImageNet 归一化 |
| ViT 位置编码 | 16×16 → 37×37，双三次二维插值（仅在冻结副本上） |
| Neck | 1×1 lateral + 4×3×3 smooth，256 通道，Simple Feature Pyramid（P2–P5） |
| Head | Faster R-CNN，anchors 4 层 (16,32,64,128) × (0.5,1,2)，ROI 7×7 |
| 可训练参数 | 17,188,121（Neck 2,688,256 + RPN + ROI + box predictor） |
| 冻结参数 | 632,186,880（`backbone.encoder.*`，有断言） |
| 预算 | 20,000 iterations × batch 2，SGD lr 2.5e-3，warmup 500，cosine→0 |
| 增强 | 仅水平翻转 p=0.5 |
| 选模 | 固定预算，取最终 checkpoint；test 只评一次 |

### 2.1 评价 GT 的已知处理（两臂完全一致）

- 训练与评价使用同一套标注合法性过滤：**部分越界框裁剪到图像内**，
  **完全越界框与退化框丢弃**（`keep_valid_boxes`，min_size=2 px）。
- 受影响规模（来自 `reports/stage5a_gwhd_data_audit.json`）：train 0 个越界框、
  val 0 个、test 7 个完全越界 + 12 个部分越界；退化框 0 个。
- 因此 test 的 1,381 张图中有 19 个框的 GT 与“直接对原始标注跑 COCOeval”不同。
  这不影响 A vs B 的公平性（两臂同一过滤），但**绝对 AP 不能**当作对官方
  COCOeval 的逐位复现。
- COCO AP 由 `torchmetrics.MeanAveragePrecision(backend='pycocotools')` 计算，
  未自定义 mAP 定义；P/R@0.5 是额外的贪心一对一匹配，与 COCO 指标分开报告。

## 3. 主结果

### 3.1 seed 0（预注册主结果）

| run | split | AP50:95 | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | P@0.5 | R@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | val | 0.0759 | 0.2652 | 0.0192 | 0.0005 | 0.0593 | 0.1062 | 0.1977 | 0.3662 | 0.3026 |
| B | val | **0.0828** | **0.2737** | **0.0282** | **0.0009** | **0.0645** | **0.1157** | **0.2047** | 0.3495 | **0.3212** |
| A | test | 0.0155 | 0.0564 | **0.0101** | 0.0030 | 0.0184 | 0.0163 | 0.0604 | 0.2383 | 0.0755 |
| B | test | **0.0187** | **0.0785** | 0.0040 | 0.0024 | **0.0230** | **0.0250** | **0.0752** | **0.2830** | **0.0964** |

### 3.2 seed 1（一致性复核）

| run | split | AP50:95 | AP50 | AP75 | AP_small | AP_medium | AP_large | AR100 | P@0.5 | R@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | val | 0.0748 | 0.2608 | 0.0180 | 0.0003 | 0.0593 | 0.1038 | 0.1959 | 0.3619 | 0.3034 |
| B | val | **0.0846** | **0.2822** | **0.0257** | **0.0027** | **0.0655** | **0.1193** | **0.2074** | **0.3635** | **0.3204** |
| A | test | 0.0137 | 0.0600 | 0.0017 | 0.0031 | 0.0157 | 0.0181 | 0.0617 | 0.2439 | 0.0789 |
| B | test | **0.0190** | **0.0787** | 0.0039 | 0.0024 | **0.0238** | **0.0226** | **0.0760** | **0.2822** | **0.0963** |

### 3.3 两 seed 平均（mean ± pstdev）

| split | 指标 | A | B | Δ(B−A) | 每 seed 差值 |
|---|---|---:|---:|---:|---|
| val | AP50:95 | 0.0753 ± 0.0006 | 0.0837 ± 0.0009 | **+0.0084** | +0.0070 / +0.0098 |
| val | AP50 | 0.2630 ± 0.0022 | 0.2780 ± 0.0043 | **+0.0150** | +0.0085 / +0.0215 |
| val | AP75 | 0.0186 ± 0.0006 | 0.0270 ± 0.0012 | **+0.0084** | +0.0090 / +0.0078 |
| val | AP_small | 0.0004 ± 0.0001 | 0.0018 ± 0.0009 | +0.0014 | +0.0004 / +0.0024 |
| val | AP_medium | 0.0593 ± 0.0000 | 0.0650 ± 0.0005 | +0.0057 | +0.0052 / +0.0061 |
| val | AP_large | 0.1050 ± 0.0012 | 0.1175 ± 0.0018 | +0.0125 | +0.0095 / +0.0155 |
| val | AR100 | 0.1968 ± 0.0009 | 0.2061 ± 0.0014 | +0.0093 | +0.0070 / +0.0115 |
| val | P@0.5 | 0.3641 ± 0.0022 | 0.3565 ± 0.0070 | −0.0076 | −0.0167 / +0.0016 |
| val | R@0.5 | 0.3030 ± 0.0004 | 0.3208 ± 0.0004 | **+0.0178** | +0.0186 / +0.0170 |
| test | AP50:95 | 0.0146 ± 0.0009 | 0.0189 ± 0.0002 | **+0.0043** | +0.0032 / +0.0053 |
| test | AP50 | 0.0582 ± 0.0018 | 0.0786 ± 0.0001 | **+0.0204** | +0.0220 / +0.0187 |
| test | AP75 | 0.0059 ± 0.0042 | 0.0040 ± 0.0001 | −0.0020 | −0.0061 / +0.0021 |
| test | AP_small | 0.0031 ± 0.0001 | 0.0024 ± 0.0000 | −0.0006 | −0.0006 / −0.0007 |
| test | AP_medium | 0.0170 ± 0.0014 | 0.0234 ± 0.0004 | +0.0064 | +0.0046 / +0.0081 |
| test | AP_large | 0.0172 ± 0.0009 | 0.0238 ± 0.0012 | +0.0066 | +0.0087 / +0.0045 |
| test | AR100 | 0.0610 ± 0.0007 | 0.0756 ± 0.0004 | **+0.0146** | +0.0149 / +0.0144 |
| test | P@0.5 | 0.2411 ± 0.0028 | 0.2826 ± 0.0004 | **+0.0415** | +0.0447 / +0.0383 |
| test | R@0.5 | 0.0772 ± 0.0017 | 0.0963 ± 0.0001 | **+0.0191** | +0.0208 / +0.0174 |

### 3.4 val 训练曲线

Arm A（官方 I-JEPA），seed 0 / seed 1：

| iter | A0 AP | A0 AP50 | A1 AP | A1 AP50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0233 | 0.0871 | 0.0202 | 0.0855 |
| 4000 | 0.0412 | 0.1537 | 0.0382 | 0.1430 |
| 6000 | 0.0456 | 0.1662 | 0.0553 | 0.1974 |
| 8000 | 0.0656 | 0.2315 | 0.0578 | 0.2049 |
| 10000 | 0.0705 | 0.2471 | 0.0685 | 0.2422 |
| 12000 | 0.0745 | 0.2514 | 0.0683 | 0.2415 |
| 14000 | 0.0669 | 0.2370 | 0.0675 | 0.2398 |
| 16000 | 0.0760 | 0.2646 | 0.0754 | 0.2602 |
| 18000 | 0.0753 | 0.2623 | 0.0722 | 0.2544 |
| 20000 | 0.0759 | 0.2652 | 0.0748 | 0.2608 |

Arm B（农业 JEPA 2k），seed 0 / seed 1：

| iter | B0 AP | B0 AP50 | B1 AP | B1 AP50 |
|---:|---:|---:|---:|---:|
| 2000 | 0.0335 | 0.1284 | 0.0363 | 0.1327 |
| 4000 | 0.0537 | 0.1885 | 0.0455 | 0.1619 |
| 6000 | 0.0481 | 0.1781 | 0.0707 | 0.2348 |
| 8000 | 0.0709 | 0.2382 | 0.0682 | 0.2343 |
| 10000 | 0.0779 | 0.2559 | 0.0759 | 0.2536 |
| 12000 | 0.0812 | 0.2675 | 0.0808 | 0.2669 |
| 14000 | 0.0708 | 0.2439 | 0.0803 | 0.2697 |
| 16000 | 0.0873 | 0.2838 | 0.0825 | 0.2775 |
| 18000 | 0.0830 | 0.2733 | 0.0837 | 0.2778 |
| 20000 | 0.0828 | 0.2737 | 0.0846 | 0.2822 |

## 4. 这个差值可信吗

**分成两个指标看。**

**AP50（0.0582 → 0.0786，+0.0204）比 AP50:95 更稳。**
两 seed 的每-seed 差值分别是 +0.0220 和 +0.0187，非常接近；而 A 臂自身的
seed 间标准差只有 0.0018，B 臂只有 0.0001。**差值（0.019–0.022）比两臂的
seed 间波动（≤0.002）大约 10 倍**，所以 AP50 上的这个差在本协议内是
可复现的。同理 R@0.5（+0.0170 / +0.0208）、AR100（+0.0144 / +0.0149）、
P@0.5（+0.0383 / +0.0447）也都满足“每 seed 差值稳定”。

**AP50:95（+0.0043）仍然弱。**
每 seed 差值 +0.0032 / +0.0053，A 臂 seed 间标准差 0.0009 —— 差值约为
波动的 3–5 倍，方向一致但幅度只有 AP50 差值的五分之一。这符合预期：
IoU≥0.75 的定位精度依赖小目标回归，而小目标在本分辨率下几乎不可见。

**仍不足以写成论文主结论**，理由是覆盖度：只有 1 个检测数据集、1 个预训练 seed
（农业 JEPA 2k 是 Stage 3B 的单一 seed 产物）、2 个训练 seed。按预注册规则，
“B > A” 需要第二个检测数据集或独立预训练 seed 复核。因此：

> 在冻结 ViT-H + 同一 Faster R-CNN Head + 518 输入 + 20K 更新的固定协议下，
> 农业持续预训练（2,000 步）在 GWHD 2021 上的 AP50 / AR100 / R@0.5 上取得
> **2 seed 一致、幅度约 +2.0 pp 的复现性提升**；AP50:95 上方向一致但幅度仅
> +0.4 pp。绝对 mAP 极低（小目标上限），结论只限“同协议内 A vs B”。

## 5. 小目标与分辨率分析

- AP_small 在两臂、两个划分上都接近 0（A test 0.0031，B test 0.0024）。
  518 输入下 1024 像素图像被压缩 1.98×，麦穗典型尺寸 20–80 原像素 →
  10–40 输入像素，只占 1–3 个 14-pixel patch。
- AP_medium / AP_large 明显高于 AP_small（val：0.0593/0.1050 对 0.0004），
  说明检测头本身能学到东西，瓶颈在小目标与冻结特征的空间精度。
- 预注册时已写明：若两臂都接近 0，结论应是“该冻结分辨率不适合小目标”，
  而不是“农业预训练无效”。实测 AP50 ≈ 0.28 非零，但 AP_small 确实接近 0，
  符合这一预注册判断。
- 未采用 tiling：36 tiles/图、约 154 GB 特征缓存、36× 计算开销；在
  单卡与 20K 预算下不现实。该选择在训练前写入协议，不是事后调整。

## 6. 成本

| 项 | A seed0 | B seed0 | A seed1 | B seed1 |
|---|---:|---:|---:|---:|
| 训练墙钟（20,000 iter） | 3,567 s | 4,301 s | 8,566 s | 8,753 s |
| 峰值显存 | 2.9 GB | 2.9 GB | 2.9 GB | 2.9 GB |
| test 推理延迟 | 45.50 ms/图 | 45.49 ms/图 | 137.43 ms/图 | 68.73 ms/图 |

- seed1 的两个运行是**并行**跑在同一张 GPU 上的，墙钟与延迟都被拉长约 2.3×，
  因此墙钟/延迟只在 seed0（独占）与 seed1（共享）内部可比，跨组不可比。
- seed0 独占条件下：A 11.6 img/s，B 9.7 img/s；测试推理两臂均为 45.5 ms/图
  （模型属性一致）。
- 训练成本约 **1 GPU·小时/臂**（独占条件）；完整 4 次复现约 4 GPU·小时。
- 参数量：可训练 17,188,121；冻结 632,186,880；两臂相同。

## 7. 与 Agri-FM+ 数字的关系

Agri-FM+ 用 Detectron2、ResNet-50、20K iterations 报告 GWHD 结果；本文用
冻结 ViT-H + Faster R-CNN + 518 输入。**绝对 mAP 不可直接对比**，只能比
同协议内的 A vs B。报告中不会把 0.0189 与论文数字并列成“更好/更差”。

## 8. 复现命令

```bash
# Arm A（官方 I-JEPA）
CUDA_VISIBLE_DEVICES=1 .venv-uv/bin/python -m src.downstream.detection_train \
  --arm A --data-root /data/datasets/GWHD2021/extracted \
  --out-dir runs/stage5a_gwhd/A_seed0 \
  --iterations 20000 --batch-size 2 --seed 0 --workers 8

# Arm B（农业 JEPA 2,000-step）
CUDA_VISIBLE_DEVICES=1 .venv-uv/bin/python -m src.downstream.detection_train \
  --arm B --data-root /data/datasets/GWHD2021/extracted \
  --out-dir runs/stage5a_gwhd/B_seed0 \
  --iterations 20000 --batch-size 2 --seed 0 --workers 8

# seed 1（把 --seed 换成 1，--out-dir 换成 A_seed1 / B_seed1）

# 比较（协议不同会直接拒绝）
.venv-uv/bin/python tools/stage5a_compare_gwhd.py \
  --a runs/stage5a_gwhd/A_seed0/results.json \
  --b runs/stage5a_gwhd/B_seed0/results.json \
  --seed-results seed=1:runs/stage5a_gwhd/A_seed1/results.json,runs/stage5a_gwhd/B_seed1/results.json \
  --out-prefix reports/stage5a_gwhd
```

产物：`protocol.json`（含数据与权重 SHA-256）、`train_log.csv`（每 50 步）、
`eval.jsonl`（每次 val）、`results.json`（最终 val + test + 成本）、
`head.pth`（Neck + RPN + ROI 的状态字典）。

### 8.1 已发现并修复的保存缺陷（影响 checkpoint，不影响指标）

4 次运行使用的保存过滤是 `not key.startswith('backbone.')`，它把
**Neck（`backbone.lateral` / `backbone.smooth`，2,688,256 参数）** 一起过滤掉了：
现有 `head.pth` 里只有 RPN + ROI（14,499,865 参数），比实际训练的
17,188,121 少 2,688,256。

- **指标不受影响**：所有 AP / P / R 都是在训练进程内用完整模型（编码器 + Neck + Head）
  计算并写入 `results.json` 的。
- **影响**：仅凭这些 `head.pth` 不能重建当时被评价的模型；要精确复评必须重跑。
- **修复**：改为 `trainable_state_dict()`，只排除 `backbone.encoder.*`，
  并加入单元测试 `test_trainable_state_dict_keeps_neck_and_drops_encoder`
  （断言保存的参数总数 == 可训练参数总数）。修复晚于 4 次运行，
  因此这 4 个目录里是旧格式 checkpoint，不做返工重跑。

## 9. 下一步（GWHD 相关）

1. 若要把 GWHD 写成论文主结论，还需要：第三个训练 seed、第二个预训练 seed
   的农业 JEPA 权重、以及第二个体积相近的检测数据集
   （PlantDoc / MinneApple / DeepWeeds）。
2. 若目标是提升绝对检测精度，应先解决小目标：512/768 tiling 或
   更高分辨率 + 位置编码插值的系统对比，而不是继续在 518 上调检测头。
3. 任何后续改动都必须在新的预注册协议下进行，并同时重跑 A 与 B。