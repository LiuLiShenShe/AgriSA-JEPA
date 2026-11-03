# Stage 5C 数据集构成

> 由 `tools/stage5c_generate_tables.py` 从 `stage5c_dataset_composition.csv` 渲染；数字为程序统计值。

## R0 — R0_agrifm35_source_pool

- 状态：`AUDIT_ONLY`（AUDIT_ONLY）
- 图像：243,694
- 物理来源：14
- 已知采集组：29,380；无组元数据图像：149,630
- train / val：206,971 / 36,723
- 最大单一来源占比：55.7%
- 跨 split 已知组：0

| 采集域 | 图像 | 占比 |
|---|---:|---:|
| FIELD_IN_SITU | 198,376 | 81.4% |
| UNKNOWN | 40,829 | 16.8% |
| ORCHARD_IN_SITU | 1,988 | 0.8% |
| CONTROLLED_LAB | 1,981 | 0.8% |
| PROTECTED_CULTIVATION | 520 | 0.2% |

## R1 — R1_agrifm147_rebuild

- 状态：`PROTOCOL_INFORMED_RECONSTRUCTION`（PROTOCOL_INFORMED_RECONSTRUCTION）
- 图像：60,603
- 物理来源：9
- 已知采集组：3；无组元数据图像：60,003
- train / val：51,617 / 8,986
- 最大单一来源占比：48.0%
- 跨 split 已知组：0

| 采集域 | 图像 | 占比 |
|---|---:|---:|
| FIELD_IN_SITU | 29,788 | 49.2% |
| UNKNOWN | 29,728 | 49.1% |
| CONTROLLED_LAB | 958 | 1.6% |
| ORCHARD_IN_SITU | 129 | 0.2% |

## R2 — R2_field_strict

- 状态：`AUTO_FROZEN`（programmatic gates satisfied; NOT human-verified）
- 图像：20,944
- 物理来源：3
- 已知采集组：0；无组元数据图像：20,944
- train / val：17,780 / 3,164
- 最大单一来源占比：73.5%
- 跨 split 已知组：0

| 采集域 | 图像 | 占比 |
|---|---:|---:|
| FIELD_IN_SITU | 20,854 | 99.6% |
| ORCHARD_IN_SITU | 90 | 0.4% |

## R3 — R3_field_extended

- 状态：`AUTO_FROZEN`（programmatic gates satisfied; NOT human-verified）
- 图像：177,396
- 物理来源：6
- 已知采集组：27,370；无组元数据图像：100,387
- train / val：150,613 / 26,783
- 最大单一来源占比：73.8%
- 跨 split 已知组：0

| 采集域 | 图像 | 占比 |
|---|---:|---:|
| FIELD_IN_SITU | 175,557 | 99.0% |
| ORCHARD_IN_SITU | 1,839 | 1.0% |

## 逐来源（R2 / R3 贡献）

| 来源 | 下载 | 解码 | 基础QC | IQA | 域 | 保留集排除 | 许可阻断 | R1 | R2 | R3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| artemis_imagesafari | 135,623 | 135,623 | 131,683 | 134,123 | 135,623 | 0 | 0 | 0 | 0 | 130,961 |
| agrifield40k_pool | 39,963 | 39,963 | 39,722 | 39,798 | 31,658 | 7,703 | 0 | 0 | 0 | 23,742 |
| agrifm_ref57 | 31,500 | 31,500 | 28,725 | 0 | 0 | 0 | 0 | 29,066 | 0 | 0 |
| agrifm_ref58 | 23,205 | 23,205 | 23,203 | 23,199 | 23,205 | 0 | 0 | 23,203 | 15,396 | 15,396 |
| agrifm_ref11 | 5,899 | 5,899 | 5,828 | 5,468 | 5,899 | 0 | 0 | 5,899 | 5,458 | 5,458 |
| applebbch81 | 1,838 | 1,838 | 1,778 | 1,805 | 1,838 | 0 | 0 | 0 | 0 | 1,749 |
| sesame_weed | 1,300 | 1,300 | 1,300 | 1,300 | 1,300 | 0 | 0 | 0 | 0 | 0 |
| agrifm_ref20 | 1,081 | 1,081 | 44 | 1,050 | 0 | 0 | 0 | 58 | 0 | 0 |
| agrifm_ref6 | 900 | 900 | 900 | 896 | 0 | 0 | 900 | 900 | 0 | 0 |
| agrifm_ref7 | 829 | 829 | 431 | 783 | 49 | 0 | 829 | 441 | 0 | 0 |
| agrifm_ref1 | 670 | 670 | 670 | 0 | 670 | 0 | 0 | 670 | 0 | 0 |
| agrifm_ref90 | 520 | 520 | 508 | 516 | 0 | 0 | 520 | 0 | 0 | 0 |
| agrifm_ref8 | 251 | 251 | 251 | 251 | 7 | 0 | 251 | 251 | 0 | 0 |
| agrifm_ref9 | 115 | 115 | 106 | 99 | 115 | 0 | 0 | 115 | 90 | 90 |

