# Stage 6C — 测试汇总

## 1. 总览

执行命令：

```
.venv-uv/bin/python -m pytest tests/ -q
```

| 项目 | 数量 |
|---|---:|
| passed | **772** |
| skipped | **7** |
| failed | **1** |
| 耗时 | 940.78 s（15 min 40 s） |

相对 Stage 6B 基线（761 passed / 7 skipped / 1 failed）：新增 **11** 个
Stage 6C 测试，全部通过；skipped 数量不变；failed 仍是同一个**既有**问题
（inherited baseline failure）。

## 2. Stage 6C 定向测试

执行命令：

```
.venv-uv/bin/python -m pytest tests/test_stage6c_phenobench.py -q
```

结果：**11 passed in 1.61 s**。

| # | 测试 | 覆盖点 |
|---:|---|---|
| 1 | `test_metric_values_prefers_official_for_three_classes_and_self_soil` | 指标来源口径（官方 crop/weed/mIoU，自实现 soil） |
| 2 | `test_protocol_problems_accepts_identical_and_flags_lr` | 三臂协议一致性；LR 篡改被检出 |
| 3 | `test_validate_arm_accepts_good_fixture` | 合法 arm 通过全部校验 |
| 4 | `test_validate_arm_rejects_wrong_checkpoint_sha` | checkpoint SHA 不符被拒 |
| 5 | `test_validate_arm_rejects_missing_seed` | probe seed 缺失被拒 |
| 6 | `test_validate_arm_rejects_missing_prediction` | val 预测缺图被拒（拒绝汇总） |
| 7 | `test_validate_arm_rejects_self_official_disagreement` | 自实现/官方指标不一致被拒 |
| 8 | `test_summarise_and_paired_deltas` | mean/std 与配对 seed 差值 |
| 9 | `test_build_comparison_and_writers` | 端到端聚合、CSV/Markdown 写出、guardrail 文本 |
| 10 | `test_explicit_checkpoint_never_falls_back_to_legacy_arm` | 显式 checkpoint 不回退旧 A/B；缓存键按 label+SHA 绑定 |
| 11 | `test_trivial_baseline_recomputes_from_labels` | 全 soil 平凡基线本地重算 |

## 3. skipped 列表（不计入 passed）

| # | 文件 | 原因 |
|---:|---|---|
| 1–4 | `test_stage2_vith.py:154/167/180/194` | `AGRISA_VITH_CHECKPOINT` 未设置为可读 checkpoint |
| 5 | `test_stage3a_downstream.py:330` | `AGRISA_VITH_CHECKPOINT` 未设置 |
| 6–7 | `test_stage3a_downstream.py:493/501` | `AGRISA_PHENOBENCH` 未设置为 PhenoBench 根目录 |

这些是既有的环境变量门控 skip，与 Stage 6C 无关；它们**没有**被计为 passed，
也不代表相应端到端路径已通过验证。新增测试文件不含任何 skip 标记。

## 4. 唯一的 failed（既有，不修）

`tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`

```
AssertionError: reports/stage5e1_licence_evidence.json
assert False == True
```

- 现象：`reports/stage5e1_licence_evidence.json` 在 git 中已被跟踪，但
  `reports/stage5e1_artifacts.json` 记录 `tracked: false`。
- 这是 **Stage 5E.1 基线自身的不一致**，在 Stage 6B 时同样失败
  （`reports/stage6_test_summary.md` 已记录）。
- 按“不修改历史 Stage 5E.1 产物”的约束，本阶段不修、不删测试、不放宽断言。
- 本阶段新增测试**没有**引入新的失败。

## 5. 其他检查

- `git diff --check`：通过（无空白错误）。
- 未提交原始图像、预测 PNG 大批量文件、特征缓存或 checkpoint（`runs/` 已被
  `.gitignore` 忽略）。
- 未修改历史 Stage 3A/4A/4B/5A/5E.1/Stage 6 报告。

## 6. 结论

- 新增 11 个测试全部通过；
- 全套件 772 passed / 7 skipped / 1 failed；
- 没有删除测试、没有放宽断言；
- skipped 逐条列出且未计入 passed；
- 既有 1 个失败如实保留并说明原因。
