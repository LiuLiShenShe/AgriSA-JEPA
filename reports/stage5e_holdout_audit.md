# Stage 5E 下游 holdout 文件级索引与交叉审计（Work Package B）

## 注册实体与索引状态

| dataset | 名称 | 版本 | 状态 | 索引文件数 | 内容哈希覆盖 |
|---|---|---|---|---:|---:|
| `phenobench` | PhenoBench | v1.1.0 | FILE_LEVEL_INDEXED | 13767 | 13767 |
| `gwhd_2021` | Global Wheat Head Dataset 2021 | 2021 | FILE_LEVEL_INDEXED | 6515 | 6515 |
| `plantdoc` | PlantDoc | original (test + train, 2,579 images) | FILE_LEVEL_INDEXED | 2579 | 2579 |
| `minneapple` | MinneApple | v1 (official release) | BLOCKED_HOST_REJECTED | 0 | 0 |
| `plantseg` | PlantSeg | current | BLOCKED_AUTH | 0 | 0 |
| `deepweeds` | DeepWeeds | 2018 (original release) | UNRESOLVED_FILE_LEVEL | 0 | 0 |
| `growliflower` | GrowliFlower | current | BLOCKED_NETWORK | 0 | 0 |
| `cropandweed` | CropAndWeed | WACV 2023 release | UNRESOLVED_FILE_LEVEL | 0 | 0 |
| `multiclass_weeds` | Multiclass Weeds Dataset | figshare record 11924543 | BLOCKED_HOST_REJECTED | 0 | 0 |
| `roboflow_disease_detection` | Roboflow - Detecting Diseases (Plant Disease Detection) | registry row (unversioned) | UNRESOLVED_FILE_LEVEL | 0 | 0 |
| `roboflow_crop_detection` | Roboflow - Crops (maize, cassava, sugarcane vs grass) | registry row (unversioned) | UNRESOLVED_FILE_LEVEL | 0 | 0 |
| `gwhd_instance_seg` | GWHD Instance Segmentation (Roboflow wheat-seg) | registry row (unversioned) | UNRESOLVED_FILE_LEVEL | 0 | 0 |

## 交叉检查结果

- 精确文件 SHA-256 命中：**0**
- 规范化像素 SHA-256 命中：**0**
- pHash 近邻候选（Hamming ≤ 6）：**14**
- 语料中出现的已注册 holdout 上游数据集名：`{}`

## 适用范围与未知项

- 上述零命中只覆盖**已取得图像字节**的范围。对 BLOCKED_* / INDEX_ONLY / UNRESOLVED_FILE_LEVEL 的实体，本阶段没有文件级证据，不得宣称无泄漏。
- PlantDoc 本次真实取得并哈希了 2,579 张图像（约 1.0 GB 归档），是唯一在本阶段从"来源级"提升到"文件级"的 holdout。
- 其余 holdout 的阻塞原因逐条记录在 `stage5e_holdout_evidence.json` 的 `registry` 字段（403 / DNS 失败 / 需要登录 / 需要账号），不做任何绕过。
- pHash 候选一律标记为 `CANDIDATE_REQUIRES_REVIEW`，不自动删除，需要嵌入确认后才可定论。
- Stage 5E.1 复核后 14 个 pHash 候选全部定论（2 个 `RESAMPLED_COPY_SAME_CONTENT`、12 个 `HASH_COLLISION_DIFFERENT_IMAGE`），`holdout_crossmatch.csv` 的 `conclusion` 列已回填为 `VERDICT_*`，不再存在 pending 候选。
