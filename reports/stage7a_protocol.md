# Stage 7A — 协议检查

- 状态：**通过**
- 运行标识：stage7a_b1_natural_50k（pair=S7A，seed=20261029）
- 基础配置：configs/stage6_b1_natural_2000.yaml（SHA-256 `52e4478c5b6c375f…`）
- 阶段1配置：configs/stage7a_b1_natural_50k.yaml
- 阶段2配置：configs/stage7a_b1_natural_50k_resume.yaml

## 预算与调度

| 字段 | 值 |
|---|---|
| max_optimizer_steps | 50000 |
| interrupt_after_optimizer_steps | 20000 |
| save_at_optimizer_steps | [10000, 20000, 50000] |
| warmup_steps | 2500 |
| epochs | 50（最低 44） |
| optimizer_steps_per_epoch | 1151 |

## 调度器 horizon 核验（LR / WD / EMA 均在 50k 处结束）

- LR cosine：`warmup=2500` → `T_max=47500`，在 step 50000 到达 final_lr
- WD cosine：`T_max=50000`，在 step 50000 到达 final_wd
- EMA：`0.996 → 1.0` 在 step 50000
- 一致性：通过

## 冻结输入文件

| 文件 | 状态 | SHA-256 |
|---|---|---|
| train_csv | 通过 | `9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08` |
| val_csv | 通过 | `f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686` |
| sampling_profile | 通过 | `91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505` |
| train_view | 通过 | `26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2` |

未发现协议偏离。
