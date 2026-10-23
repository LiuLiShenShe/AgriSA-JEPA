# Stage 5B — 来源分布与采样策略

> 机器可读：`reports/stage5b_source_distribution.json`
> 自动生成表：由 `tools/stage5b_summarize_corpus.py` 写入本文件的上半部分
> 工具：`tools/stage5b_summarize_corpus.py`

## 1. Field-D1 候选版本实际组成

| 划分 | 图像 | 来源 | 最大来源占比 | 来源熵 |
|---|---:|---:|---:|---:|
| train | 15,118 | 8 | 59.1% | 2.036 bit |
| val | 3,612 | 7 | 62.2% | 1.845 bit |
| 合计 | 18,730 | 8 | — | — |
| v1 参考 | 15,592 | 6 | 71.7% | 1.438 bit |

v1 的来源熵按 6 个来源计算（把 `lvb11/lvb32/...` 等批次前缀归并为 `lucas_vision`），
否则会把同一来源的不同文件批次误当成不同来源。

train 按来源：

| 来源 | 图像 | 占比 | 域 |
|---|---:|---:|---|
| `lucas_vision` | 8,938 | 59.1% | FIELD_IN_SITU |
| `applebbch81` | 1,511 | 10.0% | ORCHARD_IN_SITU |
| `vegann` | 1,243 | 8.2% | FIELD_IN_SITU |
| `sesame_weed` | 1,040 | 6.9% | FIELD_IN_SITU |
| `inat_weeds` | 1,031 | 6.8% | FIELD_IN_SITU |
| `acre_crop_weed` | 791 | 5.2% | FIELD_IN_SITU |
| `radish_wheat` | 427 | 2.8% | FIELD_IN_SITU |
| `sorghum_weed` | 137 | 0.9% | FIELD_IN_SITU |

按采集域：FIELD_IN_SITU 90.0%、ORCHARD_IN_SITU 10.0%。
按设备：地面相机/手持 100%（本版本无 UAV、无卫星、无受控台拍）。

## 2. 单一来源占比问题

去掉 PlantVillage 后，`lucas_vision` 仍是最大来源（train 59.1%），仅比 v1 的 71.7%
下降 12.6 个百分点。本轮新增 3,138 张净图像只占 20.2%，不足以把任何来源压到 50%
以下。

要真正接近 50% 目标，还需要 5,000–15,000 张**新来源**的田间图像。这属于 Field-D2
的范围（见 `reports/stage5b_pretraining_plan.md` 的数据规模分析）。

本轮**不通过删除合法数据来凑比例**。LUCASVision 图像是合法的 CC-BY 田间数据，
删除它们会让 v1 的历史可比性失效。

## 3. 三种采样策略

`src/registry/source_balance.py` 给出三种透明策略的每来源权重与采样后占比。
三者都只改采样权重，不删数据。

### 3.1 natural

语料即分布。可复现，但与来源规模成正比。`lucas_vision` 占 59.7%。

### 3.2 source_balanced

每来源等权（各 12.5%）。`sorghum_weed` 的 172 张会被重复约 13.6 倍，
`radish_wheat` 约 4.4 倍。作为对照臂是干净的，但重复曝光可能损害泛化，
所以它不是默认设置。

### 3.3 source_capped(cap=0.5)

把最大来源压到 50%，其余按比例上浮（权重统一 1.24）。小来源的重复倍数比
balanced 小一个数量级，同时没有单一来源主导采样质量。

**注意**：`source_capped` 的采样器**尚未在 DataLoader 中实现**。
`configs/stage5c_field_jepa_candidate.yaml` 的 `data.sampling` 目前只接受
`natural`。权重表是 Stage 5C 的实现规格，不是已生效的代码。

## 4. 采集组

| 来源 | 图像 | 组 | 每组件数 |
|---|---:|---:|---:|
| `applebbch81` | 1,838 | 140 | 13.1 |
| 其余 7 个来源 | 16,892 | 16,892 | 1.0 |

只有 `applebbch81` 提供真实的多视图采集单元（树）。v1 的 `component` 列是逐图 id，
不是观测/会话 id；这是继承的局限，记录在
`reports/stage5b_duplicate_audit.md` 第 4 节。

## 5. 结论

1. Field-D1 相比 v1：来源熵 1.438 → 2.036 bit，最大来源占比 71.7% → 59.1%，
   采集域从单一域变成两个近地域，设备仍是全地面相机。三项都是净改善。
2. 单一来源占比 59.1% 仍超过 50% 参考目标，原因是新增数据量（3,138 张）不足以稀释
   一个 11,183 张的来源。这是规模问题，不是策略问题。
3. 推荐 Stage 5C 至少跑两条：B1（natural）与 B2（source_capped, cap=0.5）。
   Balanced 保留为可选第三臂，因为它会显著增加小来源的重复曝光。
