# Stage 3B + Stage 4A 总结报告

> 生成时间：2026-10-22  
> 分支：`agri-stage3b-4a`，代码基线 commit `e86c900`（本轮工作尚未提交）  
> 运行环境：uv（Python 3.13.12，torch 2.9.1+cu128），单卡 RTX A6000（GPU1）  
> 数据：`agrifield40k-v1`（**PROVISIONAL**，12,567 train / 3,025 val）

## 0. 最终结论

**C — Negative**

在相同的官方初始化、农业数据、2,000 optimizer updates 预算和 PhenoBench 评价协议下，植被引导（C1）、结构边界引导（C2）与打乱对照（C3）都没有超过标准随机多块 Mask（B）。2,000 步时 C1/C2/C3 的 mIoU 相对 B 分别为 **−0.27 / −0.39 / −0.17 pp**；打乱空间对应的 C3 与 C1/C2 同级，说明“结构先验必须与图像空间对齐”在本轮没有得到支持。

农业持续预训练本身是有效的：B 相对官方权重 **+1.42 pp mIoU、+3.29 pp Weed IoU**，3/3 seeds 一致。

## 1. 本轮实际完成的工作

1. 将仓库依赖从 conda 环境迁移为 uv 管理：
   - `pyproject.toml`、`.python-version`（3.13）、`uv.lock`
   - 环境位于 `.venv-uv`（原 conda `.venv` 未删除，仅不再使用）
   - 索引：USTC PyPI 镜像 + PyTorch cu128 官方索引
2. 续跑两个被中断的 Arm 到完整 2,000 步（GPU1）：
   - C1 vegetation：1683 → 2000（+317 updates）
   - C3 shuffled：1287 → 2000（+713 updates）
3. 补齐 4 个缺失评价点（C1_2000、C2_2000、C3_500、C3_2000）与 2 个此前中断的评价点（C1_500、C2_500）。
4. 重新生成完整对比表：9 个评价点 × 3 Probe seeds。
5. 新增实时训练日志（`PROGRESS` 行）与资源监控。
6. 全量测试：**246 passed, 7 skipped**；四臂协议检查通过。

## 2. 训练结果

| Arm | Mask | 真实 updates | 本轮耗时 | Peak GPU | 最终 loss | 产物 |
|---|---|---:|---:|---:|---:|---|
| B | random | 2,000 | 6,622 s（连续） | 23,298 MiB | — | step500 / step2000 |
| C1 | vegetation | 2,000 | 续跑 317 步 17.4 min | 23,821 MiB | 0.1026 | step500 / step2000 |
| C2 | boundary | 2,000 | 8,403 s（连续） | 23,882 MiB | — | step500 / step2000 |
| C3 | shuffled | 2,000 | 续跑 713 步 38.6 min | 23,583 MiB | 0.1086 | step500 / step2000 |

说明：
- C1/C3 的原始训练段曾与机器上其他任务共享 GPU，wall-clock 不可与 B/C2 直接逐项比较；上表只报告本轮续跑段实测值。
- 四组 `verification` 均为 `loss_finite=true`、`ema_formula_verified=true`，`optimizer_steps_total=2000`。
- C1/C3 续跑使用 `configs/stage4a_*_2000_resume.yaml`，与原配置仅 3 处不同（`load_checkpoint`、`read_checkpoint`、`save_at_optimizer_steps`），已逐字段 diff 验证。

### Encoder SHA-256

| Label | SHA-256 |
|---|---|
| B_500 | `ae9bc1bec28c5454ed1d61ca76fa2245cd8d1b130d3b4bb146601d58a8b49cd4` |
| B_2000 | `1c4e1c484d89245ea3702c9603c62355bdfcd07be8594df020e6e3cf4d94d23f` |
| C1_500 | `0386d4282e683a426b7b934e139de881aef1d3dd66fdc8b3ccfb7968e703840b` |
| C1_2000 | `cbedb7a3caafd90e085f18dfc0015d45d4da2f789ca5d230fbfb7cf632b498a5` |
| C2_500 | `3decac1ee47e72040f4cfb53e5ab887ed3a9dd599d01ac34c8742d065d0a7b77` |
| C2_2000 | `0a7d695c7faddc2e793ca2c5c3d1700aae6d39da9b7061fc58f41bdcdc184f7c` |
| C3_500 | `5fa120b3342a8b35e315ae40d3a3e3b2cddb790039779baa9c4ebf937f66e6cc` |
| C3_2000 | `9f13d3e179e6261f0eb83268371c8291d298a6f775db745b87816948fd70bfa8` |

## 3. PhenoBench 结果（官方评价器，3 Probe seeds）

| Label | Crop IoU | Weed IoU | mIoU |
|---|---:|---:|---:|
| A_0 | 81.4533 ± 0.1634 | 35.9633 ± 3.8774 | 71.6767 ± 1.3413 |
| B_500 | 82.4033 ± 0.1464 | 37.7867 ± 2.6954 | 72.6433 ± 0.9402 |
| B_2000 | **82.3433 ± 0.1644** | **39.2500 ± 1.7033** | **73.1000 ± 0.5967** |
| C1_500 | 82.2100 ± 0.3059 | 38.4833 ± 2.3186 | 72.7967 ± 0.8755 |
| C1_2000 | 82.2333 ± 0.3564 | 38.5767 ± 2.9559 | 72.8333 ± 1.0973 |
| C2_500 | 82.1333 ± 0.2346 | 37.5233 ± 4.1397 | 72.4500 ± 1.4440 |
| C2_2000 | 82.1033 ± 0.3021 | 38.3400 ± 3.2651 | 72.7067 ± 1.1296 |
| C3_500 | 82.3467 ± 0.2735 | 39.0467 ± 3.1590 | 73.0333 ± 1.1268 |
| C3_2000 | 82.1867 ± 0.2855 | 38.9267 ± 2.3429 | 72.9333 ± 0.8696 |

所有评价点 `cross_check_agrees=True`（自实现指标 vs 官方，最大差 0.005 pp）。

详细对比与消融见：
- `reports/stage4a_phenobench_comparison.md`
- `reports/stage4a_ablation.md`

## 4. 方法实现（回答协议第十七节 A 部分）

1. **C1 植被先验**：对增强后的 224×224 RGB 还原到 [0,1]，计算 `ExG = 2G − R − B`，逐图做稳健归一化（分位数裁剪），均值池化到 16×16 patch grid。
2. **C2 边界先验**：灰度梯度/Sobel 边缘强度与 ExG 响应梯度组合，mean-pool 到 16×16；论文中记录原始边缘、植被相关边缘与最终结构图。
3. **C3 打乱**：按 `sha256(base_seed:sample_id)` 对 16×16 先验单元做置换，保持数值分布、破坏与图像的空间对应。
4. **采样**：所有 Arm 使用同一候选块采样器；每个 target 生成 `num_candidates=16` 个合法位置，`p_i = softmax(λ · score_i)`，`λ=6.0`；`λ=0` 退化为均匀采样（已有测试）。
5. **几何公平性**：块尺寸/长宽比/数量、Context-Target 排斥、`min_keep`、batch 共同长度裁剪均沿用官方语义；C1/C3 的非法候选率为 0。
6. **目标函数**：仍为原始 JEPA Smooth-L1 与 EMA，未改变 backbone/predictor/target encoder。

## 5. 工程与环境

- uv 环境：`UV_PROJECT_ENVIRONMENT=.venv-uv uv run --no-active ...`
- 训练入口：`main.py --fname <config> --devices cuda:1`
- 实时日志：`/home/test/agrisa_runs/logs/*.log`，每个 optimizer update 一行 `PROGRESS step/N loss lr wd gpu ms/step elapsed eta`
- 资源采样：`/home/test/agrisa_runs/logs/resources_gpu1*.csv`
- 启动脚本：`tools/run_stage4a_resume.sh`；监控脚本：`tools/monitor_training.sh`
- 评价队列：`tools.queue_stage4a_evals`（机器级 flock，单卡顺序执行）

## 6. 尚未完成 / 风险

- 本轮只有 **1 个预训练 seed**；+ 3 个 Probe seeds 属于探索性比较，不能替代多预训练 seed 的稳定性验证。
- `agrifield40k-v1` 仍为 PROVISIONAL：784 条重复候选未全部人工裁决，14,743 张待定图像未采纳。
- PhenoBench 验证集参与了本轮方法消融，存在验证集选择偏差；论文需要独立任务/数据集做泛化验证。
- 结构先验（颜色/边缘）对黄化、干枯、非绿色作物与特殊光照有已知局限；本轮 C1/C2 未显示收益，下一步更值得做的是扩大预训练预算/多种子，而不是继续调先验强度。
- 本轮改动尚未 commit/push。

## 7. 复现

```bash
cd /fj/jepa-二维图像任务
export UV_PROJECT_ENVIRONMENT=.venv-uv

# 测试与协议
uv run --no-active pytest -q
uv run --no-active python tools/check_stage3b4a_protocol.py

# 续跑（若需重来）
tools/run_stage4a_resume.sh

# 评价与汇总
uv run --no-active python -u -m tools.queue_stage4a_evals \
  --device cuda:1 \
  --labels C1_500,C1_2000,C2_500,C2_2000,C3_500,C3_2000 \
  --runs-root /home/test/agrisa_runs
uv run --no-active python -m tools.compare_stage4a_phenobench \
  --results runs/stage4a_phenobench/results
```
