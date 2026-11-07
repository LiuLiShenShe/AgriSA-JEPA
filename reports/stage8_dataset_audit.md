# Stage 8 — 数据集审计

> 生成时间（UTC）：2026-11-07T08:12:55+00:00

- 预训练 manifest：`stage6_r3v3_train_view_nvme.csv`（147277 张图）。

| dataset | 任务 | 状态 | 本地图数 | 可解码 | 备注 |
|---|---|---|---|---|---|
| `gwhd2021_det` | detection | READY_WITH_MODIFICATION | 6515 | 6515 | local images audited |
| `plantdoc_cls` | classification | READY_WITH_MODIFICATION | 2579 | 2579 | local images audited |
| `minneapple_det` | detection | BLOCKED | 0 | 0 | not_downloaded |
| `cropandweed_sem` | segmentation | READY_WITH_MODIFICATION | 8034 | 8034 | local images audited |
| `growliflower_sem` | segmentation | BLOCKED | 0 | 0 | not_downloaded |
| `plantseg_sem` | segmentation | BLOCKED | 0 | 0 | not_downloaded |
| `deepweeds_cls` | classification | READY_WITH_MODIFICATION | 17509 | 17509 | local images audited |
| `phenobench_sem` | segmentation | READY_WITH_MODIFICATION | 13767 | 13767 | local images audited |

## 每数据集说明

### `gwhd2021_det`（detection）

- 本地路径：`/data/datasets/GWHD2021/extracted/gwhd_2021/images`
- 注释：three official CSVs (competition_{train,val,test}.csv); reuse Stage 5A pipeline
- 官方切分：train / val / test (official)
- 许可：CC-BY-4.0（Zenodo 5092309）
- 泄漏风险：excluded from all pretraining manifests (holdout)
- 状态：**READY_WITH_MODIFICATION** — local images audited

### `plantdoc_cls`（classification）

- 本地路径：`/fj/jepa-二维图像任务/data/holdout_raw/plantdoc_extract/PlantDoc-Dataset-master`
- 注释：class-directory structure (train/{28 classes}, test/{28 classes}); no bbox annotations shipped, so detection is BLOCKED locally; classification reuses class directory labels
- 官方切分：train / test (class dirs), val carved from train with fixed seed
- 许可：CC-BY-4.0（GitHub pratikkayal/PlantDoc-Dataset）
- 泄漏风险：not in pretraining sources (holdout)
- 状态：**READY_WITH_MODIFICATION** — local images audited

### `minneapple_det`（detection）

- 本地路径：`（未下载）`
- 注释：not downloaded; license "research use (repository terms)" must be accepted before local audit
- 官方切分：train / val / test (official)
- 许可：research use (repository terms)（GitHub nicolaihaeni/MinneApple）
- 泄漏风险：none
- 状态：**BLOCKED** — not_downloaded

### `cropandweed_sem`（segmentation）

- 本地路径：`/data/datasets/cropandweed/images`
- 注释：bboxes/labelIds/params per image (8,034 each); semantic masks are labelIds/{CropAndWeed}/*.png; splits derived from ave_/vwg_ recording-session prefix (no official CSV split)
- 官方切分：ave_ (Application Set) / vwg_ (Experimental Set); session-grouped
- 许可：AIT non-commercial research（LICENCE.txt (local, AIT)）
- 泄漏风险：upstream ACRECropWeed (790 images) in pretraining corpus is a DIFFERENT dataset -- verify no name/hash overlap
- 状态：**READY_WITH_MODIFICATION** — local images audited

### `growliflower_sem`（segmentation）

- 本地路径：`（未下载）`
- 注释：not downloaded; must check sequence/session leakage before use
- 官方切分：train / val / test (time series)
- 许可：CC-BY-4.0（ipb.uni-bonn.de/data/growliflower）
- 泄漏风险：none
- 状态：**BLOCKED** — not_downloaded

### `plantseg_sem`（segmentation）

- 本地路径：`（未下载）`
- 注释：not downloaded; license "see repository" must be read before use
- 官方切分：train / test (per disease)
- 许可：see repository（arXiv:2409.04038）
- 泄漏风险：none
- 状态：**BLOCKED** — not_downloaded

### `deepweeds_cls`（classification）

- 本地路径：`/data/datasets/DeepWeeds/images`
- 注释：labels.csv (Filename,Label,Species) + 5-fold train/val/test subset CSVs; no single official split -- use fold-0 as the fixed split
- 官方切分：5-fold stratified CV (paper)
- 许可：CC-BY-4.0（GitHub AlexOlsen/DeepWeeds）
- 泄漏风险：appears in Agri-FM+ pretraining (ref [60]); excluded from our pretraining manifests
- 状态：**READY_WITH_MODIFICATION** — local images audited

### `phenobench_sem`（segmentation）

- 本地路径：`/data/datasets/PhenoBench/PhenoBench-v110`
- 注释：official download + official eval tool installed; Stage 7B probes already cover this dataset
- 官方切分：train / val (+ test server)
- 许可：CC-BY-SA-4.0（phenobench.org）
- 泄漏风险：excluded from all pretraining manifests (holdout)
- 状态：**READY_WITH_MODIFICATION** — local images audited

