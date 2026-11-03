# Stage 7A — 最终验收报告

> 生成时间（UTC）：2026-11-03T10:42:35+00:00

## 1. Git 基线

- 分支：`agri-stage7a-20k-checkpoint-curve`
- 基线 commit：`6b9cb8502f0501c2f37054337b777347005b6162`
- 新提交：见 `git log --oneline -5`（本报告之后提交）
- 工作树状态：仅未跟踪的 `CLAUDE.md`（保留），无强制回退、无覆盖历史产物

## 2. 冻结输入

- train_csv：`9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` (通过)
- val_csv：`f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` (通过)
- sampling_profile：`91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505` (通过)
- train_view：`26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` (通过)

## 3. 配置与调度

- 阶段1配置：`configs/stage7a_b1_natural_50k.yaml`（SHA `18f713d9c5ab6d90…`）
- 阶段2配置：`configs/stage7a_b1_natural_50k_resume.yaml`（SHA `d63fcf4223e9bc2e…`）
- 总调度长度：`max_optimizer_steps=50000`，`epochs=50`（最低 44）
- 阶段性中断：`interrupt_after_optimizer_steps=20000`
- 保存点：`[10000, 20000, 50000]`
- 有效 batch：16 x 8 = 128
- 协议检查：**通过**（0 个问题）

## 4. 训练轨迹

| 阶段 | optimizer steps | micro steps | loss 首→末 | 峰值显存(MiB) | 耗时(小时) |
|---|---|---|---|---|---|
| 10k/20k（阶段1） | 20000 | 159932 | 0.1747→0.1353 | 23474.6513671875 | 35.78 |
| 50k（阶段2） | 50000 | 399821 | 0.1302→0.1046 | 23544.11572265625 | 53.98 |

- 阶段1 实测耗时：2146.8 分钟
- 阶段2 实测耗时：3238.5 分钟

## 5. Checkpoint 与 SHA-256

| 文件 | 大小(GB) | SHA-256 |
|---|---|---|
| jepa-step10000.pth.tar | 9.65 | `094b7d0d03b1358f8f22188b…` |
| jepa-step20000.pth.tar | 9.65 | `29cd7a5228ddc6c9e80212b4…` |
| jepa-step50000.pth.tar | 9.65 | `f10e83aa4e74d3ab5305d5cb…` |
| jepa-encoder-step10000.pth | 2.35 | `ee8ce11eab1023f0a620eedf…` |
| jepa-encoder-step20000.pth | 2.35 | `a66f1fa42dcdc44718c9e79c…` |
| jepa-encoder-step50000.pth | 2.35 | `5f302e4789593024548c7fb2…` |
| jepa-latest.pth.tar | 9.65 | `d0162877a7b04a6d122089e8…` |

## 6. source exposure 实际分布

**阶段1（part 0，10k/20k）**：
- artemis_imagesafari：1941885（75.89%）
- agrifield40k_pool：316914（12.38%）
- agrifm_ref58：267293（10.45%）
- applebbch81：24020（0.94%）
- agrifm_ref11：6636（0.26%）
- agrifm_ref9：1466（0.06%）
- agrifm_ref1：698（0.03%）
- 总样本：2558912；熵：0.7698147608527589；HHI：0.6022300817327654

**合计（阶段1+阶段2，parts=['0', '20000']，6397136 样本）**：
- artemis_imagesafari：4854534（75.89%）
- agrifield40k_pool：792214（12.38%）
- agrifm_ref58：668372（10.45%）
- applebbch81：60032（0.94%）
- agrifm_ref11：16596（0.26%）
- agrifm_ref9：3655（0.06%）
- agrifm_ref1：1733（0.03%）

## 7. PhenoBench 冻结 encoder probe

### 参考行

- baseline（A）：soil 97.6085，crop 81.4567，weed 35.98，mIoU 71.6833
- ref_2k_stage6d（B1）：soil 97.6902，crop 82.2333，weed 38.52，mIoU 72.81

### 同轨迹

- step10k：soil 97.7342，crop 82.3333，weed 35.9867，mIoU 72.0167
- step20k：soil 97.7424，crop 82.44，weed 37.37，mIoU 72.5167
- step50k：soil 97.7415，crop 82.4733，weed 37.8267，mIoU 72.68
- **决策：CONTINUE_TO_50k** — 20k is at least as good as 10k within tolerance and shows a clear positive signal; continue the same 50k schedule from the 20k checkpoint.

> 诚实披露：probe 跨 seed 的 weed 标准差较大（10k ±3.22pp、20k ±2.56pp、50k ±0.56pp），而 weed 的 20k−10k 信号为 +1.38pp、20k mIoU 信号 +0.50pp。决策严格按预注册阈值执行；信号幅度接近 probe 噪声量级，应视为“同轨迹内持续改善”而非强结论。

## 8. 是否继续到 50k

- 结论：**CONTINUE_TO_50k**
- 理由：20k is at least as good as 10k within tolerance and shows a clear positive signal; continue the same 50k schedule from the 20k checkpoint.
- training_verified：True
- miou_20k_minus_10k：0.5
- crop_20k_minus_10k：0.1067
- weed_20k_minus_10k：1.3833
- soil_20k_minus_10k：0.0082
- miou_not_worse_than_minus_0_3：True
- positive_signal：True
- crop_and_weed_not_below_minus_1_0：True
- cross_check_agrees：True
- borderline：False
- conflicting：False
- thresholds：{'miou_drop_max': 0.3, 'miou_up_min': 0.2, 'crop_weed_up_min': 0.5, 'crop_weed_drop_max': 1.0}

## 9. 测试与偏离

- 单元测试：`tests/test_stage7a_checkpoint_curve.py`（25 项，见 test_summary）
- 历史测试：未修改、未删除、未放宽
- 已知偏离：无

## 10. 结论

- **已完成**：单条 50k schedule 上 10k/20k/50k 三点的冻结 encoder probe（mIoU：step10k 72.02 → step20k 72.52 → step50k 72.68），同轨迹内单调改善。
- 20k 决策门通过后已续训至 50k，最终步精确 50000。

**可支持**：
- 在本单轨迹上，预算从 10k→50k 时 PhenoBench mIoU 与 weed 持续改善。
**不可支持**：
- 不能把 10k/20k/50k 计为三个独立预训练 seed（同一轨迹）。
- 单条轨迹不能支持“预算越多越好”的普遍性结论。
- weed 跨 seed 方差（±0.6–3.2pp）不小，不能宣称 weed 增益具有统计显著性。

## 11. 下一阶段建议

- 本阶段已完成 10k/20k/50k 单轨迹预算曲线；若需强结论，应增加独立预训练 seed 复现（不同 `data.seed`）而非延长同一条轨迹。
- 可用 50k checkpoint 作为后续下游任务（检测/分割/分类）的初始化基线。

