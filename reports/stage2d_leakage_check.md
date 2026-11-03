# agrifield40k-v1 一致性验证

本工具只读，不修改任何 manifest。所有数字来自下方 JSON。

## 1. Train / Val 样本 ID 与路径交集

- 行数：train 12,567 / val 3,025
- sample_id 交集：**0**
- path 交集：**0**

## 2. 完全重复检测（文件 Hash 与解码像素 Hash）

| Split | 行数 | 文件 SHA-256 重复组 | 重复样本 | 像素 SHA-256 重复组 | 重复样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 12,567 | 0 | 0 | 0 | 0 |
| val | 3,025 | 0 | 0 | 0 | 0 |

同一 split 内完全重复保留在训练集是已知的、无害的；这里只统计，不删除。

## 3. 采集序列跨 Split 检查（窗口 60s）

两个 split 中都没有同源带时间戳的图像 —— 无采集序列跨 split。
## 4. iNatWeeds observation_id 跨 Split 检查

- 独立 grouping key：**1,227**
- 同时出现在 train 和 val 的 key：**0** (0 train / 0 val)

## 5. PhenoBench 来源重叠检查

- 冻结版本中的 PhenoBench 行：train 0 / val 0 —— **PASS**

## 6. pHash/dHash 重复候选隔离状态

- 接触本版本的候选边：0
- 重新落在同一 split 内的边：0（其中 v0 时为跨 split 的：0）
- Candidate endpoints are almost entirely pending_audit/exclude rows that no frozen split contains; the 11 include endpoints were the manifest-affecting edges isolated in v0 and are still outside v1. No candidate was adjudicated by a human.

