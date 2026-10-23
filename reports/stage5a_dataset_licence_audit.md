# Stage 5A — 数据许可证审计

> 许可证逐行记录在 `data/registry/agricultural_sources.csv` 的 `license` / `license_evidence` / `attribution_required` 字段。

## 1. 按许可证汇总

| 许可证 | 来源数 | source_id |
|---|---:|---|
| unverified | 26 | agrifm_ref1, agrifm_ref10, agrifm_ref11, agrifm_ref15, agrifm_ref18, agrifm_ref2, agrifm_ref29, agrifm_ref3, agrifm_ref35, agrifm_ref46, agrifm_ref48, agrifm_ref5, agrifm_ref58, agrifm_ref6, agrifm_ref65, agrifm_ref7, agrifm_ref73, agrifm_ref74, agrifm_ref75, agrifm_ref76, agrifm_ref8, agrifm_ref81, agrifm_ref82, agrifm_ref9, agrifm_ref90, agrifm_ref95 |
| CC-BY-4.0 | 20 | acre_crop_weed, agrifm_ref20, agrifm_ref31, agrifm_ref57, agrifm_ref60, agrifm_ref79, deepweeds, growliflower, gwhd_2021, lucas_vision, maize_weed, must_c, palmer_amaranth, plantdoc, radish_wheat, ronin, rumex_leaves, sorghum_weed, vcd, we3ds |
| CC-BY-SA-4.0 | 5 | agrifm_ref37, agrifm_ref88, cropandweed, grass_clover, phenobench |
| CC-BY-4.0 (Roboflow dataset page) | 3 | gwhd_instance_seg, roboflow_crop_detection, roboflow_disease_detection |
| MIT | 2 | agrifm_ref4, perennial_plants |
| CC-BY-1.0 | 1 | vegann |
| CC0-1.0 | 1 | sesame_weed |
| per-image CC-BY / CC0 | 1 | inat_weeds |
| CC-BY-SA-3.0-US (Zenodo also lists CC-BY-4.0) | 1 | agrifm_ref93 |
| research use (see repository) | 1 | minneapple |
| see repository | 1 | plantseg |
| CC-BY-4.0 (figshare record) | 1 | multiclass_weeds |

## 2. 结论

- 多数来源为 CC-BY-4.0，需保留署名；PhenoBench/GrassClover/CropAndWeed 为 CC-BY-SA-4.0，衍生作品需同许可共享；Perennial Plants 为 MIT；Sesame&Weed 为 CC0。
- PlantVillage 的 Zenodo 记录同时出现 `cc-by-4.0` 与正文 CC BY-SA 3.0 US，存在元数据冲突 → 以更严格的 **CC-BY-SA-3.0-US** 记录，未经人工确认不进入冻结语料。
- 许可证未核实的来源一律 `allowed_in_project=pending`，且状态不超过 `URL_VERIFIED`，不进入任何训练清单。
- 本阶段**没有**把任何来源标记为 `APPROVED`，因此没有新增“已授权并审核”语料；D1 candidate 的纳入决定见总结报告。

