# PhenoBench 数据审计（Stage 3A）

**结论：已获得官方完整数据，版本 v1.1.0，train / val 的 images 与 5 类标注齐全。
未上传任何原始图像到 Git。**

## 1. 官方来源与版本

| 项目 | 值 |
| --- | --- |
| 官方页面 | <https://www.phenobench.org/dataset.html> |
| 下载地址 | <https://www.phenobench.org/data/PhenoBench-v110.zip> |
| 版本 | **PhenoBench v1.1.0**（2023-12，`PhenoBench/README.MD` 明示） |
| 本地路径 | `/data/datasets/PhenoBench/PhenoBench-v110` |
| 压缩包大小 | **7,630,658,167** 字节 |
| 压缩包 SHA-256 | `a310c1c1dcfec4ca4897097fd38c02af39ea24112f658fd9408d5deb3c7c8bfc` |
| 解压后大小 | ≈ 7.2 GB |
| 下载日期 | 2026-10-19 |
| 数据清单 | `reports/phenobench_file_manifest.sha256.txt`（13,074 个文件的相对路径 + 字节数） |
| 数据清单 SHA-256 | `13fa2810d9fe87c2d65b2dc5b7c7e8ac55e31812aee44af21aea98da8e98bd5d` |

解压后完整性用 `unzip -t` 验证通过（`ZIP OK`）。文件级清单按
`split/subdir/filename<TAB>bytes` 排序生成，哈希见上。

## 2. 许可与署名

- 官方数据集页面标注 **CC-BY-SA-4.0**；
- 但压缩包内 `README.MD` 写明 **CC BY-NC-SA 4.0**
  （Attribution-NonCommercial-ShareAlike）。
- **两者不一致，必须按更严格的 NC 条款处理**：本仓库仅将其用于本机科研评价，
  不用于商业用途，不再分发原始图像。此差异已记录，未擅自择一。
- 署名要求（来自 README，原文引用）：

  > Weyler, Magistri, Marks, Chong, Sodano, Roggiolani, Chebrolu, Stachniss,
  > Behley. *PhenoBench — A Large Dataset and Benchmarks for Semantic Image
  > Interpretation in the Agricultural Domain*. arXiv preprint, 2023.

## 3. 文件与样本数量

| Split | images | semantics | plant_instances | leaf_instances | plant_visibility | leaf_visibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,407 | 1,407 | 1,407 | 1,407 | 1,407 | 1,407 |
| val | 772 | 772 | 772 | 772 | 772 | 772 |
| test | 693 | — | — | — | — | — |

- 文件名完全对齐（`diff <(ls images) <(ls semantics)` 为空）。
- 图像尺寸全部为 **1024×1024 RGB**。
- test 划分没有公开标注，本轮不使用。

## 4. 语义标签定义（官方 Devkit 口径）

官方 `PhenoBench/README.MD` 与 `phenobench/phenobench_loader.py`：

| id | 含义 |
| ---: | --- |
| 0 | background / soil |
| 1 | crop |
| 2 | weed |
| 3 | partial crop（可见像素 < 50%） |
| 4 | partial weed |

官方语义指标使用 `phenobench/evaluation/auxiliary/convert.py` 的
`convert_partial_semantics`：**3→1，4→2**，然后用
`MulticlassJaccardIndex(num_classes=3, average=None)` 评分。
也就是说，部分可见作物/杂草被计入其父类，**不被忽略**。
本项目的训练目标使用完全相同的映射。

### 标签像素分布（实测全量）

| Split | soil(0) | crop(1) | weed(2) | partial crop(3) | partial weed(4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,293,238,516 (87.66%) | 153,463,715 (10.40%) | 7,078,797 (0.48%) | 21,310,522 (1.44%) | 254,882 (0.017%) |
| val | 727,889,307 (89.92%) | 66,418,271 (8.20%) | 4,088,634 (0.51%) | 10,916,383 (1.35%) | 188,077 (0.023%) |

映射到 3 类后：

| Split | soil | crop | weed |
| --- | ---: | ---: | ---: |
| train | 87.66% | 11.85% | 0.50% |
| val | 89.92% | 9.55% | 0.53% |

**这是高度不平衡的 3 类问题。** 一个"全部预测 soil"的平凡解在 val 上得到
soil IoU = 0.8992、crop IoU = 0、weed IoU = 0，即 **mIoU ≈ 29.97%**。
报告任何 mIoU 时必须与本平凡基线比较，否则数字没有意义。
该基线在 `reports/phenobench_armA_results.md` 中列出。

统计文件：`reports/phenobench_label_stats.json`。

## 5. 数据使用限制（本轮遵守）

- PhenoBench 官方 **Val 未用于训练分割头**；Probe 只在官方 train 上训练。
- PhenoBench 图像**未加入**农业 JEPA 自监督预训练（Arm B 的 500-step 训练
  早于本轮，且使用 v0，不含任何 PhenoBench 行）。
- 原始图像与预测图不进入 Git（`runs/` 已在 `.gitignore`）。
- 若未来加入新来源，重复筛查入口是
  `tools/phenobench_leakage_check.py` 与 `tools/verify_split_version.py`。

## 6. 与 AgriField 的重叠检查

见 `reports/phenobench_leakage_check.json` 与
`reports/stage2d_leakage_check.md`。要点：

- AgriField v0/v1 冻结划分中 **source=PhenoBench 的行数为 0**。
- AgriField 汇编本身已把 9,606 行标为 `source=PhenoBench`、`status=exclude`，
  v0/v1 均不包含它们。
- 这 9,606 行是 PhenoBench **未标注 patch 发布版**的 1024×1024 裁剪
  （原始 ID 形如 `20200425_val_nh_P0024183_4178_1024`），覆盖 7 个采集日，
  其中 **2020-05-15 / 05-26 / 06-05 三个日期与 v1.1.0 标注版重合**。
- 对 v1 的 15,592 行与 PhenoBench 的 2,179 张图像做：
  - 文件 SHA-256 相同：**0**
  - 解码像素 SHA-256 相同：**0**
  - dHash ≤ 8 的近重复：356 对；≤ 4：9 对；≤ 2：1 对
- 对最近的候选（d=2、d=3）做了人工/视觉复核（Agnes 3.0 Flash 识图）：
  两者土壤颜色、作物形态、耕作痕迹均明显不同，判定为**不同地块/不同日期**，
  属于 dHash 在俯拍农田上的高假阳性，而非真实重复。
- 结论：**v1 与 PhenoBench 没有发现像素级或感知级真实重叠**；
  "同一地块不同日期"的语义重叠无法由 hash 排除，但日期与来源不同
  （AgriField 的 pnb 行已全部排除，其余来源为 2006–2022 的其他数据集），
  因此不构成训练/评价泄漏。

## 复现

```bash
# 下载与校验（本机不重复下载）
sha256sum /data/datasets/PhenoBench/PhenoBench-v110.zip
# 结构与标签
./.venv/bin/python - <<'PY'
from src.downstream.phenobench_data import dataset_summary
print(dataset_summary('/data/datasets/PhenoBench/PhenoBench-v110'))
PY
# 重叠检查
./.venv/bin/python -m tools.phenobench_leakage_check \
  --agrifield data/frozen/agrifield40k-v1_train.csv \
              data/frozen/agrifield40k-v1_val.csv \
  --phenobench /data/datasets/PhenoBench/PhenoBench-v110 \
  --out reports/phenobench_leakage_check.json
```