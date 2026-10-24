# Stage 5C — 下载审计（Gate A）

> 生成：2026-10-24T20:38:23Z
> 机器可读：`data/corpus_v2/download_plan.csv`、`/data/datasets/Stage5C/download_manifest.csv`

本表只记录本机实际存在的文件。论文报数、远端索引数与本机下载数分列，不混算。

## 1. 逐来源状态

| 来源 | 访问方式 | 下载状态 | 归档字节 | 归档 SHA-256 | 解压图像 | 备注 |
|---|---|---|---:|---|---:|---|
| agrifm_ref1 | kaggle_dataset | DOWNLOADED_FULL | 546032010 | `1ec9fe29c54d4402…` | 670 | 670 wheat-spike images; spike annotations. |
| agrifm_ref2 | url | BLOCKED | — | `…` | 0 | Publisher page returns HTTP 403 to automated clients; no direct data URL. |
| agrifm_ref3 | url | BLOCKED | — | `…` | 0 | Publisher page returns HTTP 403 / connection reset; no direct data URL. |
| perennial_plants | kaggle_dataset | DOWNLOADED_FULL | — | `…` | 397 | Kaggle licence field says "Other" while the Stage 5A ledger says MIT; licence conflict ->  |
| agrifm_ref5 | url | BLOCKED | — | `…` | 0 | Zindi competition requires an authenticated account and accepted competition rules. |
| agrifm_ref6 | kaggle_dataset | DOWNLOADED_FULL | 14105730 | `5559c381cda0959b…` | 900 | Leaf images, controlled capture. |
| agrifm_ref7 | kaggle_dataset | DOWNLOADED_FULL | 82830492 | `0bf4a77f591576de…` | 829 |  |
| agrifm_ref8 | kaggle_dataset | DOWNLOADED_FULL | 273842393 | `751004a72f583f55…` | 251 |  |
| agrifm_ref9 | kaggle_dataset | DOWNLOADED_FULL | 1701005180 | `143568be45f81a8c…` | 115 |  |
| agrifm_ref10 | kaggle_dataset | NOT_STARTED | — | `…` | 0 |  |
| agrifm_ref11 | kaggle_dataset | DOWNLOADED_FULL | — | `…` | 5899 | NDVI camera; not pure RGB. |
| agrifm_ref15 | kaggle_dsv | BLOCKED | — | `…` | 0 | Kaggle dsv endpoint returns HTTP 403 without account credentials; ACCESS_REQUIRES_USER_ACT |
| agrifm_ref18 | url | BLOCKED | — | `…` | 0 | Paper page reachable but no machine-readable archive endpoint for the orchard fruit datase |
| agrifm_ref20 | mendeley | DOWNLOADED_FULL | — | `…` | 1081 | 1,081 individual JPEGs; controlled winter-wheat leaf capture. |
| agrifm_ref29 | kaggle_dsv | BLOCKED | — | `…` | 0 | Kaggle dsv endpoint returns HTTP 403 without account credentials. |
| gwhd_2021 | local_reuse | DOWNLOADED_FULL | — | `…` | 6515 | Downstream holdout; already on this machine, SHA-verified in Stage 5A. |
| agrifm_ref35 | kaggle_competition | BLOCKED | — | `…` | 0 | Kaggle competition download returns HTTP 401; competition rules require an authenticated a |
| agrifm_ref37 | kaggle_competition | BLOCKED | — | `…` | 0 | Kaggle competition download returns HTTP 401; competition rules require an authenticated a |
| agrifm_ref46 | url | BLOCKED | — | `…` | 0 | Springer chapter landing page returned HTTP 404 from this network; no data repository link |
| agrifm_ref48 | none | BLOCKED | — | `…` | 0 | HF repository lists only README.md and .gitattributes (no image files); the 14,870 images  |
| agrifm_ref57 | mendeley | DOWNLOADED_PARTIAL | 1971986776 | `394c2b500007cc24…` | 31500 | 130,472 individual JPEGs across white/dark/illuminated/real backgrounds (paper reports 24, |
| agrifm_ref58 | dataverse | DOWNLOADED_FULL | — | `…` | 23205 | 6 ZIPs (ALS + bean rust + healthy), 15,402 images, CC0-1.0. |
| deepweeds | local_reuse | ARCHIVE_ONLY | — | `…` | 0 | Downstream holdout (DeepWeeds). No verified machine-readable image archive on this network |
| agrifm_ref65 | url | BLOCKED | — | `…` | 0 | ACM DL returns HTTP 403; Paddy Doctor data are distributed through a competition that need |
| agrifm_ref73 | url | BLOCKED | — | `…` | 0 | arXiv paper only; no public archive URL in the reference row; data provider page not resol |
| agrifm_ref74 | url | BLOCKED | — | `…` | 0 | arXiv paper only; no public archive URL in the reference row. |
| agrifm_ref75 | kaggle_dsv | BLOCKED | — | `…` | 0 | Kaggle dsv endpoint returns HTTP 403 without account credentials. |
| agrifm_ref76 | kaggle_dsv | BLOCKED | — | `…` | 0 | Kaggle dsv endpoint returns HTTP 403 without account credentials. |
| ronin | local_reuse | DOWNLOADED_FULL | — | `…` | 2822 | Ronin (Sudars et al. 2020). Full original not reachable; Stage 5B holds a Ronin-derived au |
| agrifm_ref81 | url | BLOCKED | — | `…` | 0 | figshare API/ndownloader returns HTTP 403 from this network. |
| agrifm_ref82 | url | BLOCKED | — | `…` | 0 | University page now returns HTTP 403 (redirects to uidaho.edu/404.htm). |
| phenobench | local_reuse | DOWNLOADED_FULL | — | `…` | 13767 | Downstream holdout; already on this machine. |
| agrifm_ref90 | mendeley_zip | DOWNLOADED_FULL | — | `…` | 520 | Protected cultivation (plant factory); ARCHIVE_ONLY. Verified byte-range download. |
| plantvillage | local_reuse | DOWNLOADED_FULL | — | `…` | 54305 | Controlled laboratory source; never admitted to R2/R3. |
| agrifm_ref95 | url | BLOCKED | — | `…` | 0 | MDPI returns HTTP 403; the data-availability statement points to a request-based channel,  |
| artemis_imagesafari | s3_prefix | DOWNLOADED_PARTIAL | 122389034611 | `c53e4632b0a8492b…` | 135623 | Sampled, not complete: PARTIAL_SOURCE_DOWNLOAD by design. |
| agrifield40k_pool | local_reuse | DOWNLOADED_FULL | — | `…` | 39963 | Derived pool: images retain upstream identity via filename prefix; PhenoBench and other ho |
| sesame_weed | local_reuse | DOWNLOADED_FULL | — | `…` | 1300 | Stage 5B download; 1,300 images are byte-identical to agrifield40k saw_* rows. |
| applebbch81 | local_reuse | DOWNLOADED_FULL | — | `…` | 1838 | Stage 5B download; 140 trees / 150 sessions. |
| user_owned | none | NOT_STARTED | — | `…` | 0 | No user-owned images exist on this machine; the metadata template is defined in data/corpu |

## 2. 完整性

- 归档在下载时流式计算 SHA-256；重跑时若目标文件大小匹配则标记 `SKIPPED_EXISTING`。
- zip 归档在解压前执行 `ZipFile.testzip()`，损坏归档记为 `CORRUPT`，不进入解压目录。
- 原始归档只读保留，筛选只生成索引与派生资产。

## 3. 未取得来源

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

## 4. 口径

| 口径 | 说明 |
|---|---|
| `reported_images` | 论文/来源页声称的数量 |
| `remotely_indexed_images` | API 实际列出的远端文件数 |
| `downloaded_images` | 本机实际文件数 |
| `decoded_images` | 通过 PIL 解码的数量 |
| `final_accepted` | 通过全部门禁的数量 |

这五个数字由不同工具产生，任何报告都不得相互替代。
