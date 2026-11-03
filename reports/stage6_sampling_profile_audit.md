# Stage 6 — 采样 profile 审计（B1 Natural / B2 Capped no-backfill）

本文件只回答一个问题：**Stage 5E.1 冻结的 `by_source` profile 是否被真实接入 DataLoader，以及它到底会发出多少、什么来源比例的样本。** 所有数字都从冻结 profile 与本机生成的 schedule 重新计算，不引用配置中的期望值。

## 1. 输入与代码路径

| 项目 | 值 |
|---|---|
| profile 文件 | `data/corpus_v4/R3v3_license_verified_sampling_profiles.json` |
| profile SHA-256 | `91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505` |
| 正式 profile | `by_source`（`cap_unit = physical_source_id`） |
| 训练 manifest | `data/stage6/stage6_r3v3_train_view_nvme.csv`（R3v3 train 的派生视图，成员与行序不变） |
| 实现 | `src/datasets/stage6_sampling.py` |
| 接入点 | `src/datasets/agri_dataset.py::make_agrifield40k(sampler_spec=...)`；`src/train.py` 传入 `data.sampling` |
| B1 | `mode: natural` → 既有 `DistributedSampler`（均匀打乱 147,277 行） |
| B2 | `mode: profile_capped` → `ProfileExposureSampler` |

B1 没有新增采样代码路径：它就是 Stage 3B/4A 已经在用的 `DistributedSampler(shuffle=True, seed=data.seed)`。B2 才需要新实现。

## 2. 冻结 profile 的数学复现

设 manifest 行数为 `N = 147,277`，`epoch_length = L = 150,000`，则每个来源的自然曝光为

```
E_s = count_s * L / N
```

`artemis_imagesafari` 的 cap 为 `0.5 * L = 75,000`，其余来源保持 `E_s`，**赤字不回填**。

本地 `capped_exposure()` 与冻结 JSON 的逐项对比（abs diff 全部为 0）：

| physical_source_id | manifest 行数 | natural exposure | capped exposure |
|---|---:|---:|---:|
| artemis_imagesafari | 111,762 | 113,828.3642388153 | **75,000.0000** |
| agrifield40k_pool | 18,240 | 18,577.238808503706 | 18,577.238808503706 |
| agrifm_ref58 | 15,386 | 15,670.471288795943 | 15,670.471288795943 |
| applebbch81 | 1,383 | 1,408.5702451842446 | 1,408.5702451842446 |
| agrifm_ref11 | 382 | 389.06278645002277 | 389.06278645002277 |
| agrifm_ref9 | 84 | 85.55307346021442 | 85.55307346021442 |
| agrifm_ref1 | 40 | 40.7395587905783 | 40.7395587905783 |
| **合计** | **147,277** | **150,000.0000** | **111,171.6357611847** |

派生量：

- `unfulfilled_exposure = 38,828.364238815295`（与冻结值一致）
- `budget_filled = false`
- capped HHI = 0.5030953299、capped entropy = 0.9242341620（与冻结值一致）

## 3. 关键区分：50% 是 nominal budget 的比例，不是实际发出样本的比例

冻结 profile 的 50% 是相对于 **nominal 150,000 exposure budget** 的上限。因为不回填：

```
capped 实际发出量 = 111,171.64
artemis 实际占比   = 75,000 / 111,171.64 = 67.46%
```

因此 **不能**把 B2 描述为“实际训练流中最大来源占比为 50%”。本阶段报告一律写作：
nominal cap = 75,000 / 150,000 = 50%；实际 emitted share ≈ 67.5%。

## 4. 从 fractional profile 到确定性整数训练流

profile 的曝光是小数（如 18,577.2388），而 DataLoader 只能发出整数个样本。转换规则固定如下：

1. **整数分配**：对每个 epoch，用最大余数法（largest remainder）把 `capped_exposure` 分配到 `round(111,171.6357) = 111,172` 个样本。结果：

   | 来源 | integer allocation |
   |---|---:|
   | artemis_imagesafari | 75,000 |
   | agrifield40k_pool | 18,577 |
   | agrifm_ref58 | 15,670 |
   | applebbch81 | 1,409 |
   | agrifm_ref11 | 389 |
   | agrifm_ref9 | 86 |
   | agrifm_ref1 | 41 |
   | **合计** | **111,172** |

   被 cap 的来源保持整数 cap 75,000，不因取整而增加；取整误差只落在未 cap 的来源上，赤字不回填。

2. **来源交错**：smooth weighted round-robin（Bresenham 族）。任何前缀的来源构成都在“理想比例 ±1 个样本”内，因此 2,000 update（256,000 样本）不必跑完整个 epoch 也具有代表性。

3. **来源内顺序**：每个来源用自己的 seeded permutation；需求超过该来源行数时（nominal 150k epoch 对**所有**来源都会超，因为 150,000 > 147,277）进入下一次新 permutation。使用 NumPy legacy `RandomState`，其随机流跨版本稳定。

4. **可复现性**：采样器是 `(seed, epoch)` 的纯函数，没有需要 checkpoint 的可变 RNG 状态；resume 时重放 epoch 得到完全相同的 index 流。

## 5. 实测（不是配置推断）

`ProfileExposureSampler` 在真实 147,277 行 manifest 上构建：

- `len(sampler) = 111,172`（= 整数 epoch）
- epoch 0 发出的来源计数 **精确等于** 上面的 integer allocation
- 同一 seed 两次构建 `list(sampler)` 完全相同；`set_epoch(0)` 可复现
- 换 seed 顺序变化
- 每个 emitted index 都能回溯到 manifest 行与 `physical_source_id`
- 实测 max source share = 0.6746（artemis），非 0.50

2,000 update = 16,000 micro-step × 16 = **256,000 个 emitted 样本**（2 个完整 epoch + 第三个 epoch 的前 33,656 个）。按前缀比例，预期来源份额：

| 来源 | 256k 中出现次数 | 实际份额 |
|---|---:|---:|
| artemis_imagesafari | 172,705 | 67.46% |
| agrifield40k_pool | 42,778 | 16.71% |
| agrifm_ref58 | 36,083 | 14.09% |
| applebbch81 | 3,245 | 1.27% |
| agrifm_ref11 | 896 | 0.35% |
| agrifm_ref9 | 198 | 0.08% |
| agrifm_ref1 | 95 | 0.04% |

这些是**由 schedule 直接数出来的**，不是配置值。正式运行时另由 `source_exposure_ledger.csv` 记录每个实际消费样本的 `sample_id` 与 `physical_source_id`，报告中的 finally 数字以 ledger 为准。

## 6. 对照：B1 natural

B1 使用 `DistributedSampler(shuffle=True, seed=20261026)`，每个 epoch 均匀打乱全部 147,277 行并按 batch 截断。自然比例（相对 train manifest）：

| 来源 | 行数 | natural share |
|---|---:|---:|
| artemis_imagesafari | 111,762 | 75.89% |
| agrifield40k_pool | 18,240 | 12.38% |
| agrifm_ref58 | 15,386 | 10.45% |
| applebbch81 | 1,383 | 0.94% |
| agrifm_ref11 | 382 | 0.26% |
| agrifm_ref9 | 84 | 0.06% |
| agrifm_ref1 | 40 | 0.03% |

B1 同样写 source-exposure ledger，所以“natural 的真实发出比例”也不用配置推断。

## 7. 结论

1. B2 的 capped profile **已真实接入** DataLoader（`ProfileExposureSampler`），不是只有配置项。
2. no-backfill 语义由测试锁定：未 cap 来源的 capped exposure 严格等于其 natural exposure；capped 来源固定 75,000；总发出量 < 150,000。
3. 实际最大来源占比约 67.5%，不是 50%；报告不得混用这两个口径。
4. natural 与 capped 两臂使用**同一个** manifest 与 sample 集合，仅曝光频率不同。

【任务已结束】
