# Stage 6D — 测试汇总

## 1. 总览

执行命令：

```
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

| 项目 | 数量 |
|---|---:|
| passed | **797** |
| skipped | **7** |
| failed | **1** |
| 耗时 | 938.53 s（15 min 38 s） |

相对 Stage 6C 基线（772 passed / 7 skipped / 1 failed）：新增 **25** 个 Stage 6D 测试，全部通过；skipped 数量不变；failed 仍是同一个**既有**问题（inherited baseline failure）。

## 2. Stage 6D 定向测试

执行命令：

```
.venv/bin/python -m pytest tests/test_stage6d_seed_replication.py -q
```

结果：**25 passed in 2.42 s**。

| # | 测试 | 覆盖点 |
|---:|---|---|
| 1 | `test_config_plan_has_paired_seeds_and_unique_run_ids` | 配对 seed、run id 唯一、与历史 seed 不冲突 |
| 2 | `test_config_plan_changes_only_allowed_fields` | 只改 seed/运行标识/输出目录；其余字段冻结 |
| 3 | `test_config_plan_detects_historical_seed_collision` | 历史 seed 冲突被检出 |
| 4 | `test_generated_configs_match_plan` | 已提交配置与生成计划逐字节一致 |
| 5 | `test_stage6d_probe_key_binds_every_component` | 缓存键绑定 run/完整 checkpoint SHA/manifest/protocol/probe seed |
| 6 | `test_stage6d_probe_key_rejects_short_sha` | 拒绝截断 SHA |
| 7 | `test_legacy_cache_key_still_short` | 旧缓存键行为不变 |
| 8 | `test_checkpoint_mean_is_probe_seed_mean` | 单 checkpoint 的 3-seed 均值 |
| 9 | `test_aggregation_uses_checkpoint_means_not_probe_seeds` | 跨 run 统计以 checkpoint 为单位，不把 probe seed 当重复 |
| 10 | `test_compare_refuses_missing_probe_seed` | probe seed 缺失拒绝汇总 |
| 11 | `test_compare_refuses_wrong_val_count` | val 不是 772 拒绝汇总 |
| 12 | `test_paired_b1_b2_uses_same_seed` | 配对 B1/B2 差值 |
| 13 | `test_paired_b1_b2_rejects_mismatched_seed` | 配对 seed 不一致被拒 |
| 14 | `test_sensitivity_subset_excludes_historical` | 敏感性子集排除历史 P0 |
| 15 | `test_run_id_uniqueness_and_arm_identity` | run id / pair / seed / arm 一致性 |
| 16 | `test_protocol_check_passes_on_generated_configs` | 4 个配置通过冻结协议检查 |
| 17 | `test_protocol_check_detects_lr_change` | LR 篡改被检出 |
| 18 | `test_protocol_check_detects_manifest_sha_change` | manifest SHA 篡改被检出 |
| 19 | `test_check_pair_rejects_seed_change` | 同 pair seed 不一致被检出 |
| 20 | `test_check_pair_allows_sampling_only` | 仅采样差异被允许 |
| 21 | `test_probe_label_binds_inputs` | probe 标签绑定 run/checkpoint/manifest |
| 22 | `test_probe_verify_rejects_missing_predictions` | 预测覆盖不足被拒 |
| 23 | `test_probe_verify_rejects_wrong_checkpoint_sha` | checkpoint SHA 不符被拒 |
| 24 | `test_comparison_manifest_includes_historical_and_new` | 比较清单含 A + 历史 + 新 run |
| 25 | `test_compare_accepts_manifest_dict` | 比较工具接受带 `entries` 的清单 |

## 3. skipped 列表（不计入 passed）

| # | 文件 | 原因 |
|---:|---|---|
| 1–4 | `test_stage2_vith.py:154/167/180/194` | `AGRISA_VITH_CHECKPOINT` 未设置为可读 checkpoint |
| 5 | `test_stage3a_downstream.py:330` | `AGRISA_VITH_CHECKPOINT` 未设置 |
| 6–7 | `test_stage3a_downstream.py:493/501` | `AGRISA_PHENOBENCH` 未设置为 PhenoBench 根目录 |

这些是既有的环境变量门控 skip，与 Stage 6D 无关；未计入 passed。

## 4. 唯一的 failed（既有，不修）

`tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`

```
AssertionError: reports/stage5e1_licence_evidence.json
assert False == True
```

- 与 Stage 6 / Stage 6C 的失败完全相同（同一测试、同一原因）。
- 按“不修改历史 Stage 5E.1 产物”的约束，本阶段不修、不删测试、不放宽断言。
- Stage 6D 新增测试**没有**引入新的失败。

## 5. 其他检查

- `git diff --check`：通过（无空白错误）。
- 未提交原始图像、预测 PNG 大批量文件、特征缓存或 checkpoint（`runs/` 已在 `.gitignore`）。
- 未修改历史 Stage 3A/4A/4B/5A/5E.1/6/6C 报告。
- 未执行训练或 probe；本文件不声称任何新结果。

## 6. 结论

- 新增 25 个 Stage 6D 测试全部通过；
- 全套件 797 passed / 7 skipped / 1 failed；
- 没有删除测试、没有放宽断言；
- skipped 逐条列出且未计入 passed；
- 既有 1 个失败如实保留并说明原因。
