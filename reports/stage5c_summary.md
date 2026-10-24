# Stage 5C — 执行总结

> 目标阈值：R3 严格唯一可训练图 > 147,286
> 实际状态：**TARGET_MET**

## 版本规模

| 版本 | 图像 | 来源 | 状态 |
|---|---:|---:|---|
| R0 | 243,694 | 14 | `AUDIT_ONLY` |
| R1 | 60,603 | 9 | `PROTOCOL_INFORMED_RECONSTRUCTION` |
| R2 | 20,944 | 3 | `AUTO_FROZEN` |
| R3 | 177,396 | 6 | `AUTO_FROZEN` |

## 质量与隔离

### R2 门禁

- PASS — `no_holdout`
- PASS — `no_license_blocked`
- PASS — `groups_disjoint`
- PASS — `no_duplicate_representatives`
- PASS — `no_unconfirmed_near_duplicates`
- PASS — `no_giant_clusters`
- PASS — `all_qc_labelled`
- PASS — `all_domain_labelled`

### R3 门禁

- PASS — `no_holdout`
- PASS — `no_license_blocked`
- PASS — `groups_disjoint`
- PASS — `no_duplicate_representatives`
- PASS — `no_unconfirmed_near_duplicates`
- PASS — `no_giant_clusters`
- PASS — `all_qc_labelled`
- PASS — `all_domain_labelled`

## 结论

机器判定只能给出 `AUTO_FROZEN`；本报告不含任何“专家人工审核”声称。若 R3 低于目标，状态为 `BELOW_SCALE_TARGET`，并如实报告缺口。

