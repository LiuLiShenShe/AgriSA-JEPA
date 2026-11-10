# Stage 8 — 泄漏审计

> 生成时间（UTC）：2026-11-07T08:12:55+00:00

- 与预训练 manifest（`stage6_r3v3_train_view_nvme.csv`，147277 张图）的 raw_sha256 / pixel_sha256 精确比对。

- **精确 raw 命中** = 与预训练某张图字节完全一致；

- **精确 pixel 命中** = 与预训练某张图解码后像素完全一致（重编码、重压缩、无损重存都会改变 raw 但保持 pixel）。

| dataset | 扫描 | 可解码 | raw 命中 | pixel 命中 | 结论 |
|---|---|---|---|---|---|
| `gwhd2021_det` | 6515 | 6515 | 0 | 0 | CLEAN_exact |
| `plantdoc_cls` | 2579 | 2579 | 0 | 0 | CLEAN_exact |
| `cropandweed_sem` | 8034 | 8034 | 0 | 0 | CLEAN_exact |
| `deepweeds_cls` | 17509 | 17509 | 0 | 0 | CLEAN_exact |
| `phenobench_sem` | 13767 | 13767 | 0 | 0 | CLEAN_exact |

## 说明

- 只做精确哈希比对；phash / dhash 近重复筛查见 Stage 5E（`data/corpus_v4/holdout_crossmatch.csv`），其结论已覆盖 gwhd_2021 / plantdoc / phenobench。

- 历史结论：GWHD 有 2 张图与 VegAnn（预训练上游）像素同图，对应 corpus 样本已在 R3v3 排除（`HOLDOUT_CONFIRMED_PIXEL_MATCH`）。
- 本审计对最终 R3v3 训练集重新做精确哈希比对，若某数据集出现新命中则 Phase 2 必须先排除泄漏图再冻结协议。

