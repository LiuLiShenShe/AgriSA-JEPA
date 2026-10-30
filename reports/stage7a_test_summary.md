# Stage 7A — 测试结果汇总

## 定向测试（Stage 7A 新增）

`tests/test_stage7a_checkpoint_curve.py` — **25 passed**（合成、快速、无 GPU）

覆盖：
- 配置计划：预算字段（max=50000 / interrupt=20000 / save_at=[10000,20000,50000] / warmup=2500）、
  `epochs=50 >= 44`、仅允许字段变更、stage1→stage2 仅恢复字段差异。
- 协议检查：拒绝对 `lr`、`max_optimizer_steps`、`warmup_steps`、`epochs` 的漂移；50k 调度 horizon。
- 调度连续性：LR/WD/EMA 在 20k 处为 50k 余弦中段，下一步平滑（无重置/跳变）。
- 决策规则：CONTINUE / STOP（>0.3pp 下降、无正向信号、weed 退化 >1pp）/ INCONCLUSIVE（阈值边缘、crop-weed 冲突、缺失 10k/20k）。
- 比较聚合：参考行（A、2k）与轨迹行（10k/20k/50k）分离，含 50k 时纳入。

## Smoke（GPU1，真实图像，小规模）

`tools/stage7a_smoke` — **通过**（`/data/agrisa_jepa_runs/stage7a_smoke/smoke_report.json`，`ok=true`）

| 检查项 | 结果 |
|---|---|
| part1 中断点 | 精确 12 optimizer updates |
| 快照 | step6/step12 全量 + encoder-only 均写出 |
| part2 续训 | 从 step12 恢复，重放 95 个已消费 micro-step，到达 24 |
| LR@中断（24 步调度） | 1.1466e-04（与调度镜像一致） |
| LR@终点 | 1.0000e-06（= final_lr） |
| step12 SHA 跨阶段 | 不变（`b393aec2…`） |
| 结论 | 中断/快照/恢复/继续且 LR/WD/EMA 无重置、无跳变 |

> 说明：smoke 使用 24-update horizon 以分钟级运行，验证的是 train.py 的中断/恢复代码路径；
> 50,000-update horizon 的调度算术由 `check_stage7a_protocol` 与单元测试单独核验。

## 全量测试（历史回归）

`pytest tests/` — **822 passed, 1 failed, 7 skipped**（45 分钟）

已知历史失败（与 Stage 7A 无关，未修改、未放宽）：

- `tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`
  - 原因：`reports/stage5e1_artifacts.json` 将 `reports/stage5e1_licence_evidence.json`
    记录为 `tracked: false`，但该文件实际已在 `7ec8a6b` 提交、`git ls-files` 可查。
    即清单过期，属于 Stage 5E.1 历史产物，不在本阶段职责内修复。

## 结论

Stage 7A 代码、配置、协议检查、smoke 与单元测试通过；唯一全量失败为既有历史清单不一致。
可以启动第一阶段（20k）真实训练。
