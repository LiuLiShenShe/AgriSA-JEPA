# Stage 2D — agrifield40k-v1 数据划分修复

**状态：PROVISIONAL。** 本版本只修复了两个 Stage 2C 已测量的结构性泄漏，
没有人工复核 784 条重复候选，也没有采纳任何 14,743 张 `pending_audit` 图像，
因此 source/licence gate 仍未达成，版本保持 PROVISIONAL。

- 新版本：`agrifield40k-v1`
- 前身：`agrifield40k-v0`（未被修改，字节级保持不变，可用于复现）
- 生成工具：`tools/build_split_v1.py`
- 验证工具：`tools/verify_split_version.py`
- 固定随机种子：`20261019`

## 文件

| 文件 | 行数 | SHA-256 |
| --- | ---: | --- |
| `data/frozen/agrifield40k-v1_train.csv` | 12,567 | `d63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435` |
| `data/frozen/agrifield40k-v1_val.csv` | 3,025 | `86efe5156291950b25af692d41b5f1b4cee437dd3ba2a44ce39e2cf18984f944` |
| `data/frozen/agrifield40k-v1_freeze_manifest.json` | — | 含上述哈希与全部统计 |

v0 参考（未改动，校验值与前次冻结记录一致）：

| 文件 | 行数 | SHA-256 |
| --- | ---: | --- |
| `data/frozen/agrifield40k-v0_train.csv` | 12,447 | `d8a179b81f2052c6d66d2f43d26505652e95b1090560d13c1e9d46efe3b27037` |
| `data/frozen/agrifield40k-v0_val.csv` | 3,145 | `4a7338d9f105164dd97fbf5719adef4b03d0e61e45d337cea07f24fbc1905a45` |

## 1. 样本数量变化

| | v0 | v1 | 变化 |
| --- | ---: | ---: | ---: |
| train | 12,447 | **12,567** | +120 |
| val | 3,145 | **3,025** | −120 |
| 合计 | 15,592 | 15,592 | 0 |

没有任何样本被删除或新增：v1 的样本集与 v0 完全相同（写入时以
`train_ids | val_ids == v0 sample set` 断言），只重新分配了 split。

### 来源分布

| Source | v0 train | v0 val | v1 train | v1 val |
| --- | ---: | ---: | ---: | ---: |
| LUCASVision | 8,938 | 2,245 | 8,938 | 2,245 |
| VegAnn | 1,243 | 360 | 1,243 | 360 |
| iNatWeeds | 1,031 | 278 | 1,031 | 278 |
| ACRECropWeed | 671 | 120 | **791** | **0** |
| RadishWheat | 427 | 107 | 427 | 107 |
| SorghumWeed | 137 | 35 | 137 | 35 |
| 合计 | 12,447 | 3,145 | 12,567 | 3,025 |

### 移动明细（`data/frozen/agrifield40k-v1_freeze_manifest.json`）

- `ACRECropWeed: val→train`：**120** 张（ACRECropWeed 全部 791 张进入 train）
- `iNatWeeds: train→val`：**218** 张
- `iNatWeeds: val→train`：**218** 张
- 其他来源：0 张移动

## 2. ACRECropWeed 如何重新划分

**方案：整个采集序列只进入训练划分，验证集中的 120 张全部移入训练集。**

事实依据（Stage 2C 已测量，本阶段复核）：

- 791 张 ACRECropWeed 图像的时间戳范围是 `rgb-2022-10-06-17-16-49` 到
  `rgb-2022-10-10-17-30-48`，跨越约 96 小时；
- 日期分布：2022-10-06 共 159 张，2022-10-10 共 632 张；
- v0 中最近的 train/val 帧相距 **2 秒**（val `rgb-2022-10-06-17-39-59` vs
  train `rgb-2022-10-06-17-40-01`），112/120 张 val 帧在 60 秒内有 train 帧；
- 官方 dataset card 声称"对序列来源按序列划分"，Stage 2C 已证伪。

**为什么不按 session 分组。** 帧间最大间隔约 87 分钟
（`rgb-2022-10-06-17-19-42` → `rgb-2022-10-06-17-39-39`），
以及约 3.85 天的跨日间隔。这些间隔不足以证明是独立采集 session：
同一机器人、同一任务、同一地块，间隔可能只是移动或暂停。
按任意时间阈值人为制造"独立序列"会把泄漏来源变成看似干净的划分，
属于伪造独立性。因此本版本不引入 session 划分。

**代价。** v1 的验证集中不再有 ACRECropWeed，来自该来源的机器人近地视角
杂草/作物外观不在独立验证中出现。这一点在评估下游结果时必须说明：
PhenoBench 是独立的评价集，不受此影响；但 AgriField v1 val 上的自监督验证
指标不再覆盖 ACRECropWeed。

## 3. iNatWeeds 如何重新划分

**方案：以 `observation_id` 为分组单位，同一 observation 的所有图像进入同一 split。**

1. `observation_id` 从 `observation_url` 解析（`/observations/(\d+)`）。
   v1 覆盖的 1,309 张 iNatWeeds 图像全部有可解析的 observation_id，
   缺失数 **0**（缺失者若出现，策略是强制进 train，绝不出现在 val）。
2. 排序 observation id → 用固定种子 `random.Random(20261019)` 洗牌 →
   贪心放入 val，直到该来源的 v0 val 比例
   （278/1309 = 21.24%）达到目标图像数 **278**。
3. 该算法是 `(rows, target_fraction, seed)` 的纯函数，且排序在洗牌之前，
   因此与 manifest 行序无关。已验证：连续两次构建产物字节完全相同
   （train `d63183af…`，val `86efe515…`）。

结果：

- 独立 observation 数：**1,227**（与 v0 相同，没有合并或拆分 observation）
- val 目标图像 278，实际分配 val 278、train 1,031 —— 与 v0 的
  1,031/278 完全一致
- v0 中跨 split 的 **31 个 observation（36 train / 31 val）已全部消除**，
  跨 split observation 数现为 **0**

## 4. 一致性验证结果

`reports/stage2d_leakage_check.json` / `.md`，由
`tools/verify_split_version.py` 生成。结论 **PASS**：

| 检查 | 结果 |
| --- | ---: |
| train/val sample_id 交集 | **0** |
| train/val path 交集 | **0** |
| 同 split 内文件 SHA-256 完全重复组 | train 0 / val 0 |
| 同 split 内解码像素 SHA-256 完全重复组 | train 0 / val 0 |
| 采集序列跨 split（60s 窗口） | **无**（两个 split 中没有同源带时间戳图像） |
| iNatWeeds observation_id 跨 split | **0**（1,227 个 key） |
| PhenoBench 来源行 | train 0 / val 0 —— **PASS** |

重复候选（`data/duplicate_review.csv`，784 条边 = 526 `same_split_screening`
+ 247 `cross_split` + 11 `manifest_affecting`，全部 `pending_manual_review`）：

- 784 条边的 **963 个端点中，0 个存在于 v0/v1 冻结划分中**；
- 端点状态分布：925 `pending_audit` + 27 `exclude` + **11 `include`**；
- 那 11 个 `include` 端点正是 v0 中 `manifest_affecting` 的 11 条边，
  已在 v0 冻结时被双边剔除（不计入 12,447/3,145）；
- 其余 772 条边的端点全部落在 `pending_audit` / `exclude`（含 16 个
  PhenoBench 端点）里，本来就进不了冻结划分。

因此 v1 的重复候选隔离状态与 v0 **逐条延续**：没有任何一条候选边在
重新划分后重新变成"划分内"或"跨划分"，没有任何候选被人工判定。
**没有任何候选被人工判定为重复或非重复。**

## 5. 仍未解决的风险

1. **784 条重复候选未人工复核。** 0 条已审查。真实重复率未知。
   保守隔离规则在 v0 已应用；v1 未改变样本集，隔离结论延续。
2. **14,743 张 `pending_audit` 图像仍未采纳。** 本轮不自动采纳，
   不因未采纳阻断探索性实验。source/licence gate 未达成。
3. **Hash 阈值以下的同场景不同次采集。** 同一地块不同日期、或更宽裁剪的
   同一场景，可能落在 pHash/dHash 半径之外，任何 hash 筛器都看不到。
   无缓解措施。
4. **未解析前缀的跨 split 重复。** 247 条跨 split 感知候选的两个端点都在
   include manifest 之外，全部涉及未解析前缀。它们被隔离，不进入冻结划分，
   但如果未来采纳这些来源，必须重新筛。
5. **ACRECropWeed 不在 v1 val 中。** 这是显式代价，见上文。

## 6. 为什么仍然是 PROVISIONAL

source/licence gate 的两个前提都没有改变：14,743 行来源未确认、
784 条重复候选未人工审查。修复两个已知泄漏是必要的，但不充分。
**没有为了通过门禁而修改状态。**

## 复现

```bash
./.venv/bin/python -m tools.build_split_v1            # 生成 v1 三个文件
./.venv/bin/python -m tools.verify_split_version \
    --train data/frozen/agrifield40k-v1_train.csv \
    --val   data/frozen/agrifield40k-v1_val.csv \
    --label agrifield40k-v1 \
    --json reports/stage2d_leakage_check.json \
    --md   reports/stage2d_leakage_check.md
```

已验证：连续两次 `tools.build_split_v1` 产物 SHA-256 完全一致；
`data/frozen/agrifield40k-v0_*` 哈希与 `reports/data_freeze.md` 记录一致，
未被改动。