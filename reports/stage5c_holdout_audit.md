# Stage 5C — 保留集隔离审计

R2/R3 在构建时按来源 ID 与文件前缀双重排除下游基准。`pnb_` 前缀用于排除混入 AgriField-40K 目录的 PhenoBench 图像。

| 保留集 | 隔离方式 | R2/R3 图像数 |
|---|---|---:|
| R1 | 不隔离（重建层） | 60,603 |
| R2 | 来源 ID + 前缀 | 20,944 |
| R3 | 来源 ID + 前缀 | 177,396 |

保留集名单（代码常量 `HOLDOUT_IDS`）：

- `cropandweed`
- `deepweeds`
- `growliflower`
- `gwhd_2021`
- `gwhd_instance_seg`
- `minneapple`
- `multiclass_weeds`
- `phenobench`
- `plantdoc`
- `plantseg`
- `roboflow_crop_detection`
- `roboflow_disease_detection`

任何 R2/R3 冻结若 `no_holdout` 门禁为 false，则状态不会升级为 `AUTO_FROZEN`。

