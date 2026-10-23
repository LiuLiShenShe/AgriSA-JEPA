# Stage 5A — 农业数据源目录（真实台账）

> 本文件由 `tools/stage5a_build_registries.py` 生成，机器可读版本为 `data/registry/agricultural_sources.csv`。
> 生成时间：2026-10-23。所有数量分列：论文报告 / 远端文件 / 实际下载 / 解码通过 / 审核通过。

## 1. 来源总览

- 台账行数：**63**（AgriField-40K 17 个原始来源、Agri-FM+ Table 1 35 个来源、12 个下游数据集；同一物理数据集在不同目录中保留独立行，不合并计数）。
- 目录分组：{'agrifield40k': 17, 'agrifmplus': 49, 'project_v1': 5, 'downstream': 12, 'project_d1': 1}
- 状态分布：{'DISCOVERED': 1, 'URL_VERIFIED': 26, 'LICENSE_VERIFIED': 28, 'DOWNLOADED': 0, 'DECODED': 0, 'DUPLICATE_SCREENED': 1, 'SPLIT_CHECKED': 7, 'APPROVED': 0, 'FROZEN': 0, 'BLOCKED': 0}
- 下游 holdout：**12** 个数据集，全部禁止进入预训练语料。

### 状态含义

`DISCOVERED` → `URL_VERIFIED` → `LICENSE_VERIFIED` → `DOWNLOADED` → `DECODED` → `DUPLICATE_SCREENED` → `SPLIT_CHECKED` → `APPROVED` → `FROZEN`。
任何一行都必须逐级推进，且每一级都需要对应证据字段；`BLOCKED` 单独标记。

## 2. 全部来源

| source_id | 数据集 | 分组 | 报告数 | 远端 | 已下载 | 解码 | 许可 | 状态 | holdout |
|---|---|---|---|---|---|---|---|---|---|
| grass_clover | GrassClover Image Dataset | agrifield40k | 435 |  |  |  | CC-BY-SA-4.0 | LICENSE_VERIFIED | no |
| maize_weed | Maize-Weed (Development of maize plant dataset) | agrifield40k | 843 | 255 |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| palmer_amaranth | Palmer Amaranth Growth Stage (PAGS8) | agrifield40k | 614 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| rumex_leaves | RumexLeaves / Zoom-in-on-the-Plant | agrifield40k | 809 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| sesame_weed | Crop and Weed Detection Data with Bounding Boxes | agrifield40k | 1300 |  |  |  | CC0-1.0 | LICENSE_VERIFIED | no |
| vcd | VCD (annotated vegetable crops, early stage) | agrifield40k | 2258 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| we3ds | WE3DS RGB-D dataset | agrifield40k | 2568 | 1553 |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| lucas_vision | LUCASVision (LUCAS crop cover photos) | agrifield40k;agrifmplus | 15876 | 11195 | 11195 | 11195 | CC-BY-4.0 | SPLIT_CHECKED | no |
| must_c | MuST-C (Multi-Sensor Multi-Temporal Crops) | agrifield40k;agrifmplus | 7242 |  |  |  | CC-BY-4.0 | DISCOVERED | no |
| perennial_plants | Perennial Plants Detection | agrifield40k;agrifmplus | 392 | 240 |  |  | MIT | LICENSE_VERIFIED | no |
| ronin | Ronin / annotated food crops and weeds (Latvia) | agrifield40k;agrifmplus | 1176 | 135 |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| phenobench | PhenoBench v1.1.0 | agrifield40k;agrifmplus;downstream | 9606 | 9606 | 9606 | 9606 | CC-BY-SA-4.0 | SPLIT_CHECKED | yes |
| acre_crop_weed | ACRE Crop-Weed Dataset | agrifield40k;project_v1 | 1000 | 791 | 791 | 791 | CC-BY-4.0 | SPLIT_CHECKED | no |
| inat_weeds | iNatWeeds (iNaturalist weed observations) | agrifield40k;project_v1 | 1315 | 1315 | 1315 | 1315 | per-image CC-BY / CC0 | SPLIT_CHECKED | no |
| radish_wheat | RadishWheatDataset | agrifield40k;project_v1 | 552 | 534 | 534 | 534 | CC-BY-4.0 | SPLIT_CHECKED | no |
| sorghum_weed | SorghumWeedDataset | agrifield40k;project_v1 | 252 | 172 | 172 | 172 | CC-BY-4.0 | SPLIT_CHECKED | no |
| vegann | VegAnn | agrifield40k;project_v1 | 3775 | 1607 | 1607 | 1607 | CC-BY-1.0 | SPLIT_CHECKED | no |
| agrifm_ref1 | Wheat Spike Dataset (voglinio) | agrifmplus | 670 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref10 | Classification Maize & Cocoa Dataset (clarentjd) | agrifmplus | 40457 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref11 | Maize Phenology Images (NDVI camera) | agrifmplus | 2950 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref15 | Maize Seed Dataset (dsv/8681789) | agrifmplus | 17724 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref18 | Deep Fruit Detection in Orchards (ACFR) | agrifmplus | 3694 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref2 | Corn field mapped with eBee Ag in Brazil (AgEagle) | agrifmplus | 267 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref20 | Winter Wheat Leaf Images Dataset | agrifmplus | 1081 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref29 | Maize in Field Dataset (dsv/3603983) | agrifmplus | 2355 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref3 | Mixed-use agricultural fields eBee Ag (AgEagle) | agrifmplus | 182 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref31 | Global Wheat Head Dataset 2021 (GWHD) | agrifmplus | 6514 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref35 | Cassava Leaf Disease Classification | agrifmplus | 21397 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref37 | V2 Plant Seedlings Dataset | agrifmplus | 5539 |  |  |  | CC-BY-SA-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref4 | Perennial Plants Detection | agrifmplus | 392 |  |  |  | MIT | LICENSE_VERIFIED | no |
| agrifm_ref46 | Maize Plant Abnormality from UAV Images | agrifmplus | 12354 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref48 | Drone-based Agricultural Dataset for Crop Yield Estimation | agrifmplus | 14870 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref5 | Makerere Fall Armyworm Crop Challenge | agrifmplus | 2599 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref57 | Crop Pest and Disease Detection Dataset | agrifmplus | 24881 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref58 | Makerere University Beans Image Dataset | agrifmplus | 15402 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref6 | Leaf Images Dataset (hamedetezadi) | agrifmplus | 900 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref60 | DeepWeeds | agrifmplus | 17509 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref65 | Paddy Doctor | agrifmplus | 13878 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref7 | Agriculture Crops Dataset (osamajalilhassan) | agrifmplus | 829 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref73 | Weed Spraying Object Detection Dataset | agrifmplus | 4402 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref74 | Apple Orchards Aerial Detection Dataset | agrifmplus | 2400 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref75 | Corn Stalk Disease (dsv/6848682) | agrifmplus | 3000 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref76 | Corn Leaf Disease (dsv/6341289) | agrifmplus | 4000 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref79 | Annotated Food Crops and Weed Images (Latvia) | agrifmplus | 2822 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref8 | Maize Images Dataset (devanshujoshi01) | agrifmplus | 251 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref81 | InsectBase Soybean Crop Insect Dataset | agrifmplus | 3824 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref82 | Multispectral Potato Plants Images | agrifmplus | 360 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref88 | PhenoBench | agrifmplus | 129000 |  |  |  | CC-BY-SA-4.0 | LICENSE_VERIFIED | no |
| agrifm_ref9 | PinotNoirGrapes (nicolaasregnier) | agrifmplus | 100 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref90 | Tomato Plant Factory Dataset | agrifmplus | 520 |  |  |  | unverified | URL_VERIFIED | no |
| agrifm_ref95 | Sugar Beet Nutrient Deficiency RGB | agrifmplus | 3600 |  |  |  | unverified | URL_VERIFIED | no |
| deepweeds | DeepWeeds | agrifmplus;downstream | 17509 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | yes |
| gwhd_2021 | Global Wheat Head Dataset 2021 | agrifmplus;downstream | 6514 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | yes |
| gwhd_instance_seg | GWHD Instance Segmentation (Roboflow wheat-seg) | agrifmplus;downstream | 582 |  |  |  | CC-BY-4.0 (Roboflow dataset page) | LICENSE_VERIFIED | yes |
| minneapple | MinneApple | agrifmplus;downstream | 1001 |  |  |  | research use (see repository) | LICENSE_VERIFIED | yes |
| multiclass_weeds | Multiclass Weeds Dataset | agrifmplus;downstream | 7872 |  |  |  | CC-BY-4.0 (figshare record) | LICENSE_VERIFIED | yes |
| plantdoc | PlantDoc | agrifmplus;downstream | 2569 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | yes |
| plantseg | PlantSeg | agrifmplus;downstream | 11458 |  |  |  | see repository | LICENSE_VERIFIED | yes |
| roboflow_crop_detection | Roboflow Crops (maize, cassava, sugarcane vs grass) | agrifmplus;downstream | 570 |  |  |  | CC-BY-4.0 (Roboflow dataset page) | LICENSE_VERIFIED | yes |
| roboflow_disease_detection | Roboflow Detecting-Diseases | agrifmplus;downstream | 5493 |  |  |  | CC-BY-4.0 (Roboflow dataset page) | LICENSE_VERIFIED | yes |
| agrifm_ref93 |  | agrifmplus;project_d1 | 20639 | 54305 | 54305 | 54305 | CC-BY-SA-3.0-US (Zenodo also lists CC-BY-4.0) | DUPLICATE_SCREENED | no |
| cropandweed | CropAndWeed | downstream | 5616 |  |  |  | CC-BY-SA-4.0 | LICENSE_VERIFIED | yes |
| growliflower | GrowliFlower | downstream | 1400 |  |  |  | CC-BY-4.0 | LICENSE_VERIFIED | yes |

## 3. URL 核验结果

- 受检 URL：58；`ok`：40；`reachable_but_blocked`：9；失败：9。
- 403/429 表示主机存在但拒绝自动访问，**不**据此提升为已验证；DNS/连接失败保持 `DISCOVERED` 并记录原因。

## 4. 数量口径（关键）

- **论文报告数量**只代表原论文陈述，绝不等于本机已有数据。
- **远端文件数量**只在真实列出归档目录后填写。
- **实际已取得数量**以本地目录为准；**解码通过**以 Pillow 实际解码为准；**最终纳入**必须通过去重、隔离与人工复核。
- 当前 **没有任何一行达到 `APPROVED`/`FROZEN`**；D1 candidate 的冻结在 `reports/stage5a_summary.md` 中单独报告。

## 5. 未找到 / 不可达资源

- `https://ageagle.com/data-set/corn-field-mapped-with-ebee-ag-in-brazil/` — unreachable / URLError: <urlopen error timed out>
- `https://doi.org/10.1007/978-3-031-45725-8_43` — http_404
- `https://doi.org/10.1016/j.dib.2024.109935` — http_404
- `https://github.com/WHU-Sensing/PlantSeg` — http_404
- `https://huggingface.co/datasets/KaraAgroAI/Drone-based-Agricultural-Dataset-for-Crop-Yield-Estimation` — unreachable / URLError: <urlopen error [Errno 101] Network is unreachable>
- `https://vegann.inrae.fr/` — unreachable / URLError: <urlopen error [Errno -2] Name or service not known>
- `https://www.ipb.uni-bonn.de/data/growliflower/` — unreachable / URLError: <urlopen error timed out>
- `https://www.ipb.uni-bonn.de/data/must-c/` — unreachable / URLError: <urlopen error timed out>
- `https://www.phenobench.org/` — unreachable / URLError: <urlopen error timed out>

## 6. 证据约定

- AgriField-40K 行：计数与许可证来自固定 revision 的官方数据卡；v1 中已确认的 6 个来源另附本地解码/去重/划分证据路径。
- Agri-FM+ 行：仅以 Table 1 的参考文献编号与参考文献列表中的 URL 为证据；论文没有公开的下载镜像不会被补造。
- 下游行：许可证来自 Zenodo API / 官方仓库 / 数据页；无法核实的字段写 `see repository` 而不是猜测。

