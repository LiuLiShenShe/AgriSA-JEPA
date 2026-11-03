# Stage 5B — 重复与采集组审计

> 产物：`data/registry/field_duplicate_candidates.csv`（5,960 行）、
> `reports/stage5b_duplicate_audit.json`
> 工具：`tools/stage5b_group_duplicate_candidates.py`

## 1. 方法与阈值

| 层级 | 判据 | 处理 |
|---|---|---|
| 字节重复 | `sha256` 相同 | 直接排除第二份 |
| 解码重复 | `pixel_sha256` 相同（重新编码/转格式） | 直接排除第二份 |
| 感知候选 | pHash 汉明距离 ≤ 4（聚类用） | **只生成候选**，不删除 |
| 感知支持 | dHash ≤ 4 | 只作佐证，不参与聚类 |
| 采集组 | 来源自带的树号/原始图号 | 整组不跨 train/val |

dHash 在重复的俯拍土壤图上连通性过强，会把上千张图连成一个巨型簇（本轮若用 dHash
聚类，最大簇达 1,337——已实测）。因此聚类只用 pHash，dHash 仅作支持证据；
簇大小超过 50 时标记 `chaining_suspected`，不自动给出处理结论。

## 2. 精确重复

| 来源 | 文件数 | 来源内字节重复对 | 解码重复对 | 与已采纳语料重复 |
|---|---:|---:|---:|---:|
| `sesame_weed` | 1,300 | 0 | 0 | 0 |
| `applebbch81` | 1,838 | 0 | 0 | 0 |
| `weedcrop` | 2,822 | 0 | 0 | 0 |

`sesame_weed` 与 AgriField-40K `saw_*` 行有 1,300 个精确命中，但那些行属于
`pending_audit`、不在 v1 语料中；工具将其记为 `upstream_pending_match`（来源映射证据），
而不是 `cross_source_duplicate`。详见下载审计第 2.1 节。

## 3. 感知候选与连通分量

| 指标 | 数值 |
|---|---:|
| 参与筛查图像 | 5,960 |
| pHash 候选对（≤4） | 1,049 |
| dHash 支持对（≤4） | 2,777 |
| 多图连通分量 | 428 |
| 最大分量 | 55 |
| 分量规模分布 | 2:568, 3:270, 4:144, 5:60, 6:24, 23:23, 55:55 |

| 风险层 | 图像数 | 建议动作 |
|---|---:|---|
| `singleton` | 4,816 | 保留 |
| `cross_group_perceptual` | 1,013 | 人工复核 |
| `same_group_perceptual` | 76 | 每组保留一张 |
| `chaining_suspected` | 55 | 使用前复核（疑似哈希链式合并） |

### 3.1 分来源

| 来源 | 单例 | 跨组候选 | 同组候选 | 链式疑似 |
|---|---:|---:|---:|---:|
| `sesame_weed` | 477 | 823 | 0 | 0 |
| `applebbch81` | 1,838 | 0 | 0 | 0 |
| `weedcrop` | 2,501 | 190 | 76 | 55 |

`sesame_weed` 的高候选数来自俯拍土壤画面的视觉相似性：这些图像**不是**重复，
而是"同一类田块的不同位置"。pHash 无法区分，因此必须由人看。它们不阻塞候选版本，
但会阻塞"重复审计已完成"的结论。

`applebbch81` 在 pHash 上完全干净（0 对），因为果园背景、树行与果实朝向差异大。

`weedcrop` 的 76 张同组候选正是"同一原始图的多个增强版本"；如果将来启用该来源，
应按 `group_key`（原始图号）每组只保留一张。

## 4. 采集组结构

| 来源 | 图像 | 采集组 | 每组件数 | 组定义 |
|---|---:|---:|---:|---|
| `applebbch81` | 1,838 | 140 | 13.1 | 树号（`DSC_<树号>_...`），每树多视图 |
| `weedcrop` | 2,822 | 1,544 | 1.8 | 原始图号（增强版本共享组） |
| `sesame_weed` | 1,300 | 1,300 | 1.0 | 无会话元数据，按单图成组（保守） |
| `lucas_vision` | 11,183 | 11,183 | 1.0 | v1 `component` 列（见下） |
| `acre_crop_weed` | 791 | 791 | 1.0 | v1 `component` 列 |
| `inat_weeds` | 1,309 | 1,309 | 1.0 | v1 `component` 列 |
| `vegann` | 1,603 | 1,603 | 1.0 | v1 `component` 列 |
| `radish_wheat` | 534 | 534 | 1.0 | v1 `component` 列 |
| `sorghum_weed` | 172 | 172 | 1.0 | v1 `component` 列 |

**已知局限（继承自 v1）**：v1 的 `component` 列是逐图 id，不是观测/会话 id。v1 的
观测级划分（iNat 的 1,227 个 observation、ACRE 的整段采集序列）是在 Stage 2D 生成
划分时施加的，并没有以组件列的形式保存在 manifest 里。本轮不重写 v1 数据，也不
声称能从这个列恢复观测分组。对新增来源，组键是真实采集单元。

## 5. 跨划分泄漏

- Field-D1 的 train/val 采集组交集为 **0**（`split_integrity.cross_split_groups = 0`）。
- 新增来源的组只出现在一个划分中：`sesame_weed` 全部为单图组；
  `applebbch81` 的 140 棵树按组划分，112 棵在 train、28 棵在 val。
- v1 的 train/val 划分保持字节级不变（SHA-256 与 Stage 2D 记录一致）。

## 6. 仍需人工审核

1. `sesame_weed` 823 张跨组感知候选（多为同类土壤画面，需确认非重复）。
2. `weedcrop` 55 张链式疑似 + 76 张同组候选。
3. 继承的 v1 784 条感知候选（Stage 5A 遗留，未解决，仍阻碍 v1 冻结）。
4. PlantVillage 12,436 条候选（不进入 Field-D1，不阻塞本轮）。
