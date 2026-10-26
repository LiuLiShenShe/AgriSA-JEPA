# Stage 6 — B1 / B2 对照汇总

| 指标 | B1 natural | B2 capped (no backfill) |
|---|---:|---:|
| optimizer_steps_total | 2000 | 2000 |
| micro_steps_total | 15996 | 15992 |
| csv_rows | 15996 | 15992 |
| emitted_samples | 255936 | 255872 |
| loss_first | 0.19038 | 0.19415 |
| loss_last | 0.12885 | 0.12195 |
| loss_min | 0.09998 | 0.10219 |
| loss_max | 0.2039 | 0.20303 |
| peak_memory_mib | 23355.33984375 | 23355.33984375 |
| step_time_ms_mean | 276.31644414086225 | 291.85121879868836 |
| hhi | 0.6018294175302173 | 0.5030854219919451 |
| entropy | 0.7702830748713404 | 0.9242986107618288 |
| max_share | 0.7585529194798699 | 0.6746263756878439 |
| max_source | artemis_imagesafari | artemis_imagesafari |
| optimizer_steps_per_epoch | 1151 | 869 |

## 非采样协议字段一致性

除采样策略、arm 标识与输出目录外，所有协议字段一致。
