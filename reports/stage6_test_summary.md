# Stage 6 — 测试汇总

## 1. 总览

执行命令：

```
.venv-uv/bin/python -m pytest tests/ -q -p no:cacheprovider
```

| 项目 | 数量 |
|---|---:|
| passed | **761** |
| skipped | **7** |
| failed | **1** |
| 耗时 | 1028.53 s（17 min 08 s） |

相对 Stage 5E.1 基线（708 passed / 7 skipped / 1 failed）：新增 **53** 个 Stage 6 测试，全部通过；skipped 数量不变；failed 仍是同一个**既有**问题。

## 2. Stage 6 定向测试

| 文件 | 用例数 | 结果 |
|---|---:|---|
| `tests/test_stage6_sampling.py` | 27 | 27 passed |
| `tests/test_stage6_protocol.py` | 20 | 20 passed |
| `tests/test_stage6_checkpoint.py` | 6 | 6 passed |
| **合计** | **53** | **53 passed** |

（单独执行三份套件：`53 passed in 735.28s`。）

### 2.1 sampler 测试覆盖

- 固定 seed 时 schedule 完全一致；重复迭代不改变结果；
- 换 seed 顺序变化；
- `set_epoch(0)` 可复现，换 epoch 顺序变化但来源构成不变；
- 每个 emitted sample 可回溯到 manifest 行与 `physical_source_id`；
- 统计计数与真实 emitted stream 一致；
- no-backfill 规则：未 cap 来源的 capped exposure 严格等于 natural exposure；capped 来源固定 75,000；
- 实际 max share ≈ 67.46%，并显式断言**不是** 50%；
- 256,000 样本前缀的来源比例贴合 profile；
- 不对 manifest / split / 图像内容做任何修改。

### 2.2 protocol 测试覆盖

- 两臂配置逐字段比对，仅允许 arm 标识、采样策略、输出目录不同；
- 篡改 `lr` 或 `backfill` 会被检出（负向测试）；
- manifest SHA 与配置一致；冻结 profile 的 cap / unfulfilled / total 复核；
- `warmup_steps` 覆盖后，不同 `optimizer_steps_per_epoch` 的两臂产生完全相同的 LR/WD 轨迹；
- 派生训练视图成员 = R3v3 train（147,277），与 val 交集为 0，行序不变；
- ledger 计数与 emitted stream 一致；flush 不重复写行（针对本次发现的 flush bug 的回归测试）。

### 2.3 checkpoint / resume 测试覆盖

- 中断 checkpoint 停在 epoch 中途且带回放状态；
- manifest SHA 绑定训练视图；
- 恢复后 `optimizer_steps=20`、`micro_steps=160`；
- 恢复后的可训练权重既不同于官方初始化、也不同于 step-10 快照；
- encoder 可用 `strict=True` 载入全新 ViT-H/14；
- optimizer 4 个 param group、target encoder 已被 EMA 更新；
- 两份 CSV 的 itr 连续（0–79 + 80–159），无重复/缺失。

## 3. skipped 列表（不计入 passed）

| # | 文件 | 原因 |
|---:|---|---|
| 1–4 | `test_stage2_vith.py:154/167/180/194` | `AGRISA_VITH_CHECKPOINT` 未设置为可读 checkpoint |
| 5 | `test_stage3a_downstream.py:330` | `AGRISA_VITH_CHECKPOINT` 未设置 |
| 6–7 | `test_stage3a_downstream.py:493/501` | `AGRISA_PHENOBENCH` 未设置为 PhenoBench 根目录 |

这些是既有的环境变量门控 skip，与本阶段无关；它们**没有**被计为 passed，也不代表相应端到端路径已通过验证。

## 4. 唯一的 failed（既有，不修）

`tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`

- 现象：`reports/stage5e1_licence_evidence.json` 在 git 中已被跟踪，但 `reports/stage5e1_artifacts.json` 记录 `tracked: false`。
- 这是 **Stage 5E.1 基线自身的不一致**，在本阶段开始前就已存在（基线 708 passed / 1 failed 时同样失败）。
- 按“不修改 Stage 5E.1 历史产物”的约束，本阶段不修、不删测试、不放宽断言。

## 5. 结论

- 新增 53 个测试全部通过；
- 没有删除测试、没有放宽断言；
- skipped 逐条列出且未计入 passed；
- 既有 1 个失败如实保留并说明原因。

【任务已结束】
