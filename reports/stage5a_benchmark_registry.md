# Stage 5A — 农业多任务 Bench Registry

> 机器可读：`data/registry/agricultural_benchmarks.csv`。覆盖 Agri-FM+ 的 8 项基准与 AgriMAE 的 4 项下游资源。

| benchmark_id | 任务 | 数据集 | 指标 | 官方工具 | 重叠风险 | 本项目状态 |
|---|---|---|---|---|---|---|
| gwhd2021_det | object_detection | Global Wheat Head Dataset 2021 | mAP (COCO definition) / AP50 / AP50:95 | pycocotools COCOeval (official split files) | HIGH in Agri-FM+ (ref [31] in Agri-147K); excluded from all of our pretraining manifests | NOT_RUN |
| plantdoc_det | object_detection | PlantDoc | mAP | COCOeval | not in our pretraining sources | NOT_RUN |
| minneapple_det | object_detection | MinneApple | mAP | COCOeval | none | NOT_RUN |
| disease_det | object_detection | Roboflow Detecting-Diseases | mAP | COCOeval | none | NOT_RUN |
| crop_det | object_detection | Roboflow Crops (maize/cassava/sugarcane vs grass) | mAP | COCOeval | none | NOT_RUN |
| plantseg_sem | semantic_segmentation | PlantSeg | mIoU | MMSegmentation mIoU | none | NOT_RUN |
| multiclass_weeds_sem | semantic_segmentation | Multiclass Weeds Dataset | mIoU | MMSegmentation mIoU | none | NOT_RUN |
| gwhd_instance_seg | instance_segmentation | GWHD Instance Segmentation (Roboflow wheat-seg) | mask AP | COCOeval (segm) | none | NOT_RUN |
| deepweeds_cls | classification | DeepWeeds | accuracy / macro-F1 | official split accuracy | appears in Agri-FM+ pretraining (ref [60]) and in AgriMAE; hold out if used | NOT_RUN |
| phenobench_sem | semantic_segmentation | PhenoBench v1.1.0 | mIoU (official semantics task) | phenobench-eval --task semantics | excluded from all pretraining manifests | RUN |
| growliflower_sem | semantic_segmentation | GrowliFlower | mIoU | official segmentation metric | none | NOT_RUN |
| cropandweed_sem | semantic_segmentation | CropAndWeed | mIoU | official segmentation metric | none | NOT_RUN |

## 关键区分

- **GWHD 2021 目标检测**（6,514 张，COCO bbox）与 **GWHD Instance Segmentation**（582 张，Roboflow mask）是两个不同资源，不能把 bbox 当 mask。
- Agri-FM+ 的 GWHD 结果使用 Detectron2 Faster R-CNN + 20K iterations；与我们的冻结 ViT-H + 检测头协议不可直接比数字，只能比“同协议内 A vs B”。
- PhenoBench 已有本项目真实结果（A 71.68 / B_2000 73.10 mIoU）；其余 11 项本阶段均为 `NOT_RUN`。

## 统一评价接口

`src/downstream/registry.py` 定义 Backbone/数据/Head/指标/cache key；`src/downstream/evaluation_registry.py` 定义每个 benchmark 的官方指标与结果路径。新增任务必须先在 registry 注册，才能进入训练入口。

