# Stage 5C — 35 来源清点与下载计划（Gate A）

> 生成时间：2026-10-24T05:21:10Z
> 机器可读：`data/corpus_v2/download_plan.csv`、`data/corpus_v2/sources.csv`
> 本文所有 HTTP 结果均为本机实测，非论文转述。

## 1. 覆盖

- Agri-FM+ Table 1 引用编号：35/35
- 缺失：无
- 物理数据集（合并同源引用后）：37
- BLOCKED：18 个

## 2. 逐来源状态

| # | 来源 | 域 | 论文报数 | 远端索引 | 计划动作 | 实测状态 | 预算字节 |
|---:|---|---|---:|---:|---|---|---:|
| 1 | Wheat Spike Dataset (voglinio) | UNKNOWN | 670 | 670 | DOWNLOAD_FULL | RANGE_206 | 546032010 |
| 2 | Corn field mapped with eBee Ag in Brazil (AgEagle) | UAV_AERIAL | 267 | — | MANUAL_REVIEW | 200 | — |
| 3 | Mixed-use agricultural fields eBee Ag (AgEagle) | UAV_AERIAL | 182 | — | BLOCKED | None | — |
| 4 | Perennial Plants Detection | UNKNOWN | 392 | 392 | DOWNLOAD_FULL | RANGE_206 | 5577501925 |
| 5 | Makerere Fall Armyworm Crop Challenge | FIELD_IN_SITU | 2599 | — | MANUAL_REVIEW | 200 | — |
| 6 | Leaf Images Dataset (hamedetezadi) | CONTROLLED_LAB | 900 | 900 | DOWNLOAD_FULL | RANGE_206 | 14105730 |
| 7 | Agriculture Crops Dataset (osamajalilhassan) | UNKNOWN | 829 | 829 | DOWNLOAD_FULL | RANGE_206 | 82830492 |
| 8 | Maize Images Dataset (devanshujoshi01) | UNKNOWN | 251 | 251 | DOWNLOAD_FULL | RANGE_206 | 273842393 |
| 9 | PinotNoirGrapes (nicolaasregnier) | ORCHARD_IN_SITU | 100 | 100 | DOWNLOAD_FULL | RANGE_206 | 1701005180 |
| 10 | Classification Maize & Cocoa Dataset (clarentjd) | UNKNOWN | 40457 | 40457 | DOWNLOAD_FULL | RANGE_206 | 19759726347 |
| 11 | Maize Phenology Images (NDVI camera) | FIELD_IN_SITU | 2950 | 2950 | DOWNLOAD_FULL | RANGE_206 | 6590889394 |
| 15 | Maize Seed Dataset (dsv/8681789) | CONTROLLED_LAB | 17724 | — | BLOCKED | NO_METHOD | — |
| 18 | Deep Fruit Detection in Orchards (ACFR) | ORCHARD_IN_SITU | 3694 | — | MANUAL_REVIEW | 200 | — |
| 20 | Winter Wheat Leaf Images Dataset | CONTROLLED_LAB | 1081 | 1081 | DOWNLOAD_FULL | LISTED | 8346980998 |
| 29 | Maize in Field Dataset (dsv/3603983) | FIELD_IN_SITU | 2355 | — | BLOCKED | NO_METHOD | — |
| 31 | Global Wheat Head Dataset 2021 (GWHD) | FIELD_IN_SITU | 6514 | 6515 | REUSE_LOCAL | LOCAL_PRESENT | — |
| 35 | Cassava Leaf Disease Classification | FIELD_IN_SITU | 21397 | — | BLOCKED | NO_METHOD | — |
| 37 | V2 Plant Seedlings Dataset | CONTROLLED_LAB | 5539 | — | BLOCKED | NO_METHOD | — |
| 46 | Maize Plant Abnormality from UAV Images | UAV_AERIAL | 12354 | — | BLOCKED | 404 | — |
| 48 | Drone-based Agricultural Dataset for Crop Yield Estimation | UAV_AERIAL | 14870 | — | BLOCKED | NO_METHOD | — |
| 57 | Crop Pest and Disease Detection Dataset | UNKNOWN | 24881 | 130472 | DOWNLOAD_FULL | LISTED | 8424664368 |
| 58 | Makerere University Beans Image Dataset | UNKNOWN | 15402 | 7 | DOWNLOAD_FULL | LISTED | 11378096691 |
| 60 | DeepWeeds | FIELD_IN_SITU | 17509 | 17509 | ARCHIVE_ONLY_NOT_DOWNLOADED | NOT_PRESENT | — |
| 65 | Paddy Doctor | FIELD_IN_SITU | 13878 | — | BLOCKED | 403 | — |
| 73 | Weed Spraying Object Detection Dataset | FIELD_IN_SITU | 4402 | — | MANUAL_REVIEW | 200 | — |
| 74 | Apple Orchards Aerial Detection Dataset | UAV_AERIAL | 2400 | — | MANUAL_REVIEW | 200 | — |
| 75 | Corn Stalk Disease (dsv/6848682) | UNKNOWN | 3000 | — | BLOCKED | NO_METHOD | — |
| 76 | Corn Leaf Disease (dsv/6341289) | UNKNOWN | 4000 | — | BLOCKED | NO_METHOD | — |
| 79 | Annotated Food Crops and Weed Images (Latvia) | FIELD_IN_SITU | 2822 | 2822 | REUSE_LOCAL | LOCAL_PRESENT | — |
| 81 | InsectBase Soybean Crop Insect Dataset | UNKNOWN | 3824 | — | MANUAL_REVIEW | 202 | — |
| 82 | Multispectral Potato Plants Images | FIELD_IN_SITU | 360 | — | BLOCKED | 403 | — |
| 88 | PhenoBench | FIELD_IN_SITU | 129000 | 13767 | REUSE_LOCAL | LOCAL_PRESENT | — |
| 90 | Tomato Plant Factory Dataset | PROTECTED_CULTIVATION | 520 | 1 | DOWNLOAD_FULL | LISTED | 3402876899 |
| 93 | PlantVillage (colour images) | CONTROLLED_LAB | 20639 | 54304 | REUSE_LOCAL | LOCAL_PRESENT | — |
| 95 | Sugar Beet Nutrient Deficiency RGB | FIELD_IN_SITU | 3600 | — | BLOCKED | 403 | — |

## 3. 扩展来源

| 来源 | 域 | 许可 | 远端规模 | 动作 |
|---|---|---|---:|---|
| Artemis and Image Safari crop imagery (AWS Open Data) | FIELD_IN_SITU | CC-BY-SA-4.0 | 7469218 | s3_prefix |
| AgriField-40K curated pool (local copy) | MIXED | CC-BY-4.0 | 39551 | local_reuse |
| USER_OWNED (interface only, no images supplied) | UNKNOWN | unknown | — | none |

## 4. 阻塞证据

- **agrifm_ref2**：Publisher page returns HTTP 403 to automated clients; no direct data URL.
- **agrifm_ref3**：Publisher page returns HTTP 403 / connection reset; no direct data URL.
- **agrifm_ref5**：Zindi competition requires an authenticated account and accepted competition rules.
- **agrifm_ref15**：Kaggle dsv endpoint returns HTTP 403 without account credentials; ACCESS_REQUIRES_USER_ACTION.
- **agrifm_ref18**：Paper page reachable but no machine-readable archive endpoint for the orchard fruit dataset was located; ACCESS_REQUIRES_USER_ACTION.
- **agrifm_ref29**：Kaggle dsv endpoint returns HTTP 403 without account credentials.
- **agrifm_ref35**：Kaggle competition download returns HTTP 401; competition rules require an authenticated account.
- **agrifm_ref37**：Kaggle competition download returns HTTP 401; competition rules require an authenticated account.
- **agrifm_ref46**：Springer chapter landing page returned HTTP 404 from this network; no data repository link.
- **agrifm_ref48**：HF repository lists only README.md and .gitattributes (no image files); the 14,870 images are not published through the mirror. UAV source, ARCHIVE_ONLY.
- **agrifm_ref65**：ACM DL returns HTTP 403; Paddy Doctor data are distributed through a competition that needs an account.
- **agrifm_ref73**：arXiv paper only; no public archive URL in the reference row; data provider page not resolvable.
- **agrifm_ref74**：arXiv paper only; no public archive URL in the reference row.
- **agrifm_ref75**：Kaggle dsv endpoint returns HTTP 403 without account credentials.
- **agrifm_ref76**：Kaggle dsv endpoint returns HTTP 403 without account credentials.
- **agrifm_ref81**：figshare API/ndownloader returns HTTP 403 from this network.
- **agrifm_ref82**：University page now returns HTTP 403 (redirects to uidaho.edu/404.htm).
- **agrifm_ref95**：MDPI returns HTTP 403; the data-availability statement points to a request-based channel, not an archive.

## 5. 边界说明

- “远端索引”来自 API 列表或字节范围探测；它不等于本机下载数量。
- 无法合法自动获取的来源保留 BLOCKED 与实测证据，不阻挡其他来源。
- 下游 Holdout（PhenoBench、GWHD、DeepWeeds 等）只建立隔离归档，永不进入 R2/R3。
