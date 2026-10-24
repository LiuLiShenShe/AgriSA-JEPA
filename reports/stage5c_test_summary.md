# Stage 5C — 测试与门禁记录

> 生成：2026-10-25（CST）
> 命令：`.venv-uv/bin/python -m pytest tests/ -q --no-header`
> 环境：Python 3.13.12 / uv 虚拟环境，未使用 conda

## 1. 结果

| 指标 | 数值 |
|---|---:|
| 收集测试 | 554 |
| 通过 | 547 |
| 失败 | 0 |
| 跳过 | 7 |
| 用时 | 300.4 s |

本轮新增 Stage 5C 测试 47 个（`tests/test_stage5c_corpus.py`，44 个测试函数，
其中 `test_stage5c_corpus.py` 内 3 个参数化展开）。继承的 Stage 1–5B 测试全部通过，
7 个跳过项与 5B 末尾一致（需要外部分布式数据或未下载的归档）。

跳过项均为环境依赖型（大语料/外部归档未落盘），不是失败被吞掉：
pytest 以 `skip` 记录并在 `-q` 汇总中单列。

## 2. 新增测试覆盖的主题

| 主题 | 代表测试 |
|---|---|
| 35 个 ref 全量登记 | `test_all_35_refs_registered` |
| 五种计数口径不混算 | `test_source_and_total_counts_agree` |
| R1/R2 语义分离 | `test_r1_and_r2_have_distinct_semantics` |
| 扩展来源不得进入 R1/R2 | `test_extension_sources_never_enter_r1_or_r2` |
| 扩展来源同样过全部门禁 | `test_extension_rows_still_face_every_strict_gate` |
| 旧帧缺省 `in_pool` 兼容 | `test_in_pool_defaults_to_true_for_legacy_frames` |
| 缺失 IQA 不得当作通过 | `test_missing_iqa_is_never_a_pass` |
| 漏斗计数守恒 | `test_funnel_counts_conserve` |
| AUTO_FROZEN ≠ HUMAN_VERIFIED | `test_auto_frozen_semantics` |
| 保留集文件级隔离 | `test_holdout_flag_marks_phenobench_rows_in_pool` |
| 许可不确定不得进入 R2/R3 | `test_uncertain_licence_blocks_entry` |
| 旧冻结 manifest 不变 | `test_legacy_frozen_manifests_unchanged` |
| 两次构建结果一致 | `test_two_builds_agree` |
| 单来源失败不影响其他来源 | `test_blocked_sources_do_not_stop_others` |

## 3. 门禁状态（R2 / R3）

构建命令：`tools/stage5c_build_versions.py --sources ALL --r3 --no-embeddings`

| 版本 | 图像 | 状态 | 门禁 |
|---|---:|---|---|
| R0 | 243,694 | AUDIT_ONLY | — |
| R1 | 60,603 | PROTOCOL_INFORMED_RECONSTRUCTION | — |
| R2 | 20,944 | AUTO_FROZEN | 8/8 通过 |
| R3 | 177,396 | AUTO_FROZEN | 8/8 通过 |

八项门禁：`no_holdout` / `no_license_blocked` / `groups_disjoint` /
`no_duplicate_representatives` / `no_unconfirmed_near_duplicates` /
`no_giant_clusters` / `all_qc_labelled` / `all_domain_labelled`。

`AUTO_FROZEN` 的确切含义是“全部程序化门禁通过”，**不是**“专家人工审核通过”。

## 4. 复现命令

```bash
.venv-uv/bin/python -m pytest tests/ -q --no-header
.venv-uv/bin/python tools/stage5c_build_versions.py --sources ALL --r3 --no-embeddings
.venv-uv/bin/python tools/stage5c_generate_tables.py
.venv-uv/bin/python tools/stage5c_final_report.py
```

## 5. Git

- 分支：`agri-stage5c-agrifm147k-auto-corpus`
- 代码与报告提交：见 `reports/stage5c_final_report.md` 第 25 问与分支 HEAD
- 未提交（按 `.gitignore`）：`data/corpus_v2/images/`（逐图 ledger，356 MB）、
  `data/corpus_v2/work/`、`*_manifest.csv`（R0–R3 共 221 MB）。
  这些文件的 SHA-256 记录在对应 `*_manifest.json` 中，可校验。
