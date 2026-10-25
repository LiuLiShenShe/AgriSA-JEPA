# Stage 5E 测试结果

命令：`.venv-uv/bin/python -m pytest tests/ -q`

```
657 passed, 7 skipped, 0 failed  (471.98 s)
```

## 新增测试

`tests/test_stage5e_corpus.py`：61 个用例，覆盖本阶段要求的全部 25 项：

| 要求 | 对应用例（节选） |
|---|---|
| 四个许可来源状态 | `test_agrifm_ref58_is_cc0_from_dataverse_api`、`test_agrifm_ref1_is_cc0_from_kaggle_api`、`test_agrifm_ref11_is_apache2`、`test_agrifm_ref9_is_nc_and_restricted_not_verified` |
| 每个许可决定都有证据 | `test_each_licence_decision_has_archived_evidence`、`test_evidence_index_records_the_archived_bytes` |
| 证据丢失/冲突 | `test_evidence_mismatch_is_detected` |
| unresolved 拒绝准入 | `test_unknown_source_is_never_train_eligible`、`test_blocked_sources_are_not_train_eligible` |
| 训练许可与再分发许可独立 | `test_train_and_redistribution_are_independent_fields`、`test_redistribution_allowed_does_not_imply_train_allowed`、`test_train_and_redistribution_columns_are_both_present_and_distinct` |
| 只有文件名无内容哈希 | `test_filename_only_is_not_a_content_check` |
| 403/404/timeout/auth 区分 | `test_blocked_holdout_records_the_real_failure_mode`、`test_blocked_statuses_are_distinct` |
| Roboflow 拆成具体实体 | `test_registry_splits_roboflow_into_concrete_entities` |
| 未索引 holdout 不得宣称通过 | `test_no_holdout_entity_claims_more_than_its_status_supports`、`test_only_file_level_entities_are_content_verified` |
| 精确重复与近重复候选 | `test_no_exact_hash_match_between_holdout_and_corpus`、`test_near_candidates_are_flagged_not_deleted` |
| 哈希碰撞复核 | `test_stage5d_collision_verdict_was_wrong_and_is_corrected`、`test_confirmed_holdout_images_are_absent_from_train` |
| 确认命中不删文件 | `test_collision_review_does_not_delete_files` |
| unknown group split | `test_unknown_group_images_are_train_only` |
| 计数守恒 | `test_r3v3_counts_conserve` |
| 采样不改变实体数 | `test_sampling_never_changes_entities`、`test_real_sampling_profiles_preserve_train_entities` |
| capped profile 预算 | `test_capped_profile_reports_unfulfilled_budget`、`test_capped_profile_can_fill_the_budget` |
| 历史文件不变 | `test_stage5d_files_are_unchanged`、`test_stage5c_files_are_unchanged` |
| 零训练 | `test_no_stage5e_tool_trains`、`test_gates_report_zero_optimizer_updates` |
| 门禁引擎语义 | `test_gate_engine_blocks_on_failure`、`test_gate_engine_conditional_when_unresolved`、`test_gate_engine_ready_only_when_all_green` |

## 7 个 skip 的真实原因（skip 不计为 pass）

| 文件 | 数量 | 原因 |
|---|---:|---|
| `tests/test_stage2_vith.py` | 4 | 未设置 `AGRISA_VITH_CHECKPOINT`（指向可读的 ViT-H checkpoint） |
| `tests/test_stage3a_downstream.py` | 1 | 未设置 `AGRISA_VITH_CHECKPOINT` |
| `tests/test_stage3a_downstream.py` | 2 | 未设置 `AGRISA_PHENOBENCH`（指向 PhenoBench 根目录） |

这些都是历史遗留的环境变量依赖，与 Stage 5E 无关；本阶段没有新增 skip。
