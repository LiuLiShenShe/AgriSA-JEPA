# Stage 4B — 几何匹配门禁

门禁结论：**PASS**（required checks；control checks 只报告，不作为放行条件）

## Required checks

| 检查 | 结果 | 细节 |
|---|---|---|
| `G1:structure_preference` | ok | `{"metric": "mean_vegetation_target_union", "delta_vs_B": 0.06731659585080074, "expected": "greater"}` |
| `G2:structure_preference` | ok | `{"metric": "mean_boundary_target_union", "delta_vs_B": 0.01623073625466004, "expected": "greater"}` |
| `G1:geometry_mean_match` | ok | `{"target_cells": {"delta_vs_B": -0.02713455876501314, "within_5": true, "within_3": true}, "context_cells": {"delta_vs_B": -1.7481499164478436, "within_5": true, "within_3": true}, "duplicated_target_area": {"delta_vs_B": 0.027134558765020245, "within_5": true, "within_3": true}, "target_overlap_fraction": {"delta_vs_B": -0.00037645442649614225, "within_5": true, "within_3": true}}` |
| `G2:geometry_mean_match` | ok | `{"target_cells": {"delta_vs_B": 0.8799236094533285, "within_5": true, "within_3": true}, "context_cells": {"delta_vs_B": -1.6388159465266199, "within_5": true, "within_3": true}, "duplicated_target_area": {"delta_vs_B": -0.8799236094533285, "within_5": true, "within_3": true}, "target_overlap_fraction": {"delta_vs_B": -0.005418408389234741, "within_5": true, "within_3": true}}` |
| `G1:geometry_distribution_match` | ok | `{"target_cells": {"ks": 0.012731757778308261, "within_tolerance": true}, "context_cells": {"ks": 0.0776637224476805, "within_tolerance": true}, "duplicated_target_area": {"ks": 0.021325694278666346, "within_tolerance": true}}` |
| `G2:geometry_distribution_match` | ok | `{"target_cells": {"ks": 0.027134558765019523, "within_tolerance": true}, "context_cells": {"ks": 0.07511737089201875, "within_tolerance": true}, "duplicated_target_area": {"ks": 0.02705498527890504, "within_tolerance": true}}` |
| `U:geometry_distribution_match` | ok | `{"target_cells": {"ks": 0.012493037319964984, "within_tolerance": true}, "context_cells": {"ks": 0.021643988223124022, "within_tolerance": true}, "duplicated_target_area": {"ks": 0.01297047823665154, "within_tolerance": true}}` |
| `G1:overlap_fraction_distribution` | ok | `{"ks": 0.024667780695472263, "tolerance": 0.08}` |
| `G2:overlap_fraction_distribution` | ok | `{"ks": 0.024190339778785708, "tolerance": 0.08}` |
| `U:equivalent_to_B` | ok | `{"target_cells": {"ks": 0.012493037319964984, "delta_vs_B": 0.16232991167342448, "ok": true}, "context_cells": {"ks": 0.021643988223124022, "delta_vs_B": 0.05522399936340605, "ok": true}, "duplicated_target_area": {"ks": 0.01297047823665154, "delta_vs_B": -0.16232991167342448, "ok": true}, "mean_vegetation_target_union": {"ks": 0.007479907694756149, "delta_vs_B": 0.00047984846740406484, "ok": true` |
| `no_nan_masks` | ok | `{"B": 0, "C1": 0, "C2": 0, "C3": 0, "U": 0, "G0": 0, "G1": 0, "G2": 0, "G3": 0}` |

## Control checks

| 检查 | 结果 | 细节 |
|---|---|---|
| `G3:structure_preference` | ok | `{"metric": "mean_boundary_target_union", "delta_vs_B": -0.01083462955018405, "expected": "not_greater"}` |
| `G3:geometry_mean_match` | NOTE | `{"target_cells": {"delta_vs_B": 4.536802737327918, "within_5": true, "within_3": false}, "context_cells": {"delta_vs_B": -6.749263945253446, "within_5": false, "within_3": false}, "duplicated_target_area": {"delta_vs_B": -4.536802737327918, "within_5": true, "within_3": false}, "target_overlap_fraction": {"delta_vs_B": -0.026009624128954467, "within_5": true, "within_3": true}}` |
| `G0:geometry_mean_match` | NOTE | `{"target_cells": {"delta_vs_B": 15.7004853982653, "within_5": false, "within_3": false}, "context_cells": {"delta_vs_B": -11.871409246439086, "within_5": false, "within_3": false}, "duplicated_target_area": {"delta_vs_B": -15.700485398265293, "within_5": false, "within_3": false}, "target_overlap_fraction": {"delta_vs_B": -0.08941700527500634, "within_5": true, "within_3": true}}` |
| `G3:geometry_distribution_match` | NOTE | `{"target_cells": {"ks": 0.10877695551842126, "within_tolerance": false}, "context_cells": {"ks": 0.2431765735656879, "within_tolerance": false}, "duplicated_target_area": {"ks": 0.09652263865679955, "within_tolerance": false}}` |
| `G0:geometry_distribution_match` | NOTE | `{"target_cells": {"ks": 0.35386329275085543, "within_tolerance": false}, "context_cells": {"ks": 0.47489456513089834, "within_tolerance": false}, "duplicated_target_area": {"ks": 0.3188509588605077, "within_tolerance": false}}` |
| `G3:overlap_fraction_distribution` | NOTE | `{"ks": 0.10694676533778946, "tolerance": 0.08}` |
| `G0:overlap_fraction_distribution` | NOTE | `{"ks": 0.36030874512612393, "tolerance": 0.08}` |
| `G3:shift_keeps_local_structure` | ok | `{"value_multiset_preserved_fraction": 1.0, "mean_abs_correlation_with_original": 0.08969484016453498, "lag1_retention_ratio": 0.9809568941449189, "fraction_abs_correlation_above_0.5": 0.006, "seam_to_interior_jump": 1.509835475214095}` |

## 必须同时成立的判据

"结构选择发生变化"与"几何混杂明显减小"必须同时为真，才能把
G1/G2 称为有效的结构引导几何控制原型。上面的
`G1:structure_preference` / `G2:structure_preference` 与
`G*:geometry_mean_match` / `G*:geometry_distribution_match`
就是这两个判据。
