# Stage 5B — 许可证与署名审计

> 机器可读：`reports/stage5b_source_verification.json`、
> `data/registry/field_source_download.csv`
> 原则：许可证只按**原始发布页/数据卡**记录；无法核实的写 `unverified`，
> 不根据数据集常见许可推断。

## 1. 进入 Field-D1 的 8 个来源

| 来源 | 许可证 | 证据 | 署名要求 | 本轮处理 |
|---|---|---|---|---|
| `lucas_vision` | CC-BY-4.0 | 固定 revision 的 AgriField-40K 数据卡 | 是 | 在 manifest 的 `attribution` 列保留 |
| `acre_crop_weed` | CC-BY-4.0 | 同上 | 是 | 同上 |
| `vegann` | CC-BY-1.0 | 同上 | 是 | 同上 |
| `radish_wheat` | CC-BY-4.0 | 同上 | 是 | 同上 |
| `sorghum_weed` | CC-BY-4.0 | 同上 | 是 | 同上 |
| `inat_weeds` | 逐图 CC-BY / CC0 | 同上 | 部分 | 逐图许可字段保留 |
| `sesame_weed` | CC0-1.0 | Kaggle 数据集页许可字段 "CC0: Public Domain" | 否 | 仍记录发布者 |
| `applebbch81` | CC-BY-4.0 | Kaggle 页 "Attribution 4.0 International" + 论文致谢 | 是 | 记录 Kodors et al. 2024 引用 |

所有 CC-BY 来源都需要在论文/数据声明中署名。Field-D1 manifest 的 `license` 与
`attribution` 列逐行保留这两项，未做归并或简化。

## 2. 被排除来源的许可证状态

| 来源 | 许可证 | 结论 |
|---|---|---|
| PlantVillage (`agrifm_ref93`) | Zenodo 元数据 `cc-by-4.0`，描述写 "CC BY-SA 3.0 US" | 按更严格的 CC-BY-SA-3.0-US 记录；`images_approved` 保持为空 |
| `weedcrop` | CC0-1.0（Kaggle），上游 Ronin 为 CC-BY-4.0 | 即使许可可用，域标签为 `UNKNOWN`，不纳入 Field-D1 |
| `agrifm_ref57` | CC-BY-4.0 | 许可可核实，但采集域混合且文件下载受阻 |
| `perennial_plants` | 台账写 MIT，Kaggle 页写 "Other (specified in description)" | **冲突未解决**，未下载、未纳入 |
| `vcd`、`we3ds`、`grass_clover`、`palmer_amaranth`、`rumex_leaves` | CC-BY-4.0 / CC-BY-4.0 / CC-BY-SA-4.0 / CC-BY-4.0 / CC-BY-4.0 | 许可清楚，但文件无法取得（见下载审计） |

## 3. PlantVillage 的处置

- 54,305 张历史文件完整保留在 `/data/datasets/PlantVillage/extracted`；
- 54,284 张去重候选的 manifest、哈希与审计结果保留在
  `data/registry/ingested/plantvillage_manifest.csv` 与
  `reports/stage5a_plantvillage_audit.json`；
- 12,436 条感知相似候选未复核的状态保持不变；
- 本轮**未删除、未修改**任何 PlantVillage 文件或记录；
- 它不出现在 `agri-field-d1-candidate` 中（`test_field_d1_manifest_has_no_controlled_or_unknown_source` 固化）；
- 未来用途：Controlled-D1 归档、Mixed-D1 消融候选。启用前必须解决
  CC-BY-SA-3.0-US 的传染性条款与署名方式。

## 4. 数据集编译许可

AgriField-40K 的编译许可为 CC-BY-SA-4.0，而其中个别来源是 CC-BY-4.0 或 CC-BY-1.0。
v1 manifest 已同时记录 `license`（原图许可）与 `compilation_license`（编译许可），
本轮继承这两个字段，没有把它们合并成一个"似乎没问题"的值。

新增来源不经过 AgriField-40K 编译层，直接来自原始发布页，因此只有单一许可。

## 5. 未决事项

1. `perennial_plants` 的 MIT vs "Other" 冲突需要向作者确认后才能在台账里定案。
2. PlantVillage 的 CC-BY-SA 传染性需要在论文的数据可用性声明中明确处理。
3. `inat_weeds` 的逐图许可需要在下游发布时保留逐图署名；本轮只在 manifest 中
   保留字段，尚未生成逐图署名清单。
