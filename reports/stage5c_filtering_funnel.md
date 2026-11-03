# Stage 5C 过滤漏斗

> 单因归因：每张图只计入第一次失败的步骤，因此各步排除数之和等于输入增量。

## R1

| 步骤 | 输入 | 本步排除 | 剩余 | 判定 |
|---|---:|---:|---:|---|
| downloaded | 64,970 | 0 | 64,970 | on-disk file |
| decoded | 64,970 | 0 | 64,970 | PIL decode |
| paper_rules_p0 | 64,970 | 3,847 | 61,123 | paper-disclosed rules only |
| license | 61,123 | 0 | 61,123 | licence not blocked |
| holdout_archive | 61,123 | 520 | 60,603 | holdout / archive-only |
| strict_eligible | 60,603 | 0 | 60,603 | final strict set |

## R2

| 步骤 | 输入 | 本步排除 | 剩余 | 判定 |
|---|---:|---:|---:|---|
| downloaded | 64,970 | 0 | 64,970 | on-disk file |
| decoded | 64,970 | 0 | 64,970 | PIL decode |
| min_resolution | 64,970 | 428 | 64,542 | short side >= 224 |
| basic_qc | 64,542 | 2,811 | 61,731 | blur/exposure/blank |
| iqa | 61,731 | 30,755 | 30,976 | MUSIQ + NIQE |
| domain | 30,976 | 2,194 | 28,782 | field / orchard evidence |
| exact_dedup | 28,782 | 7,801 | 20,981 | byte/pixel identity |
| near_dedup | 20,981 | 0 | 20,981 | confirmed collapse / uncertainty isolation |
| holdout_license_archive | 20,981 | 29 | 20,952 | holdout / licence / archive-only |
| strict_eligible | 20,952 | 0 | 20,952 | final strict set |

## R3

| 步骤 | 输入 | 本步排除 | 剩余 | 判定 |
|---|---:|---:|---:|---|
| downloaded | 243,694 | 0 | 243,694 | on-disk file |
| decoded | 243,694 | 0 | 243,694 | PIL decode |
| min_resolution | 243,694 | 428 | 243,266 | short side >= 224 |
| basic_qc | 243,266 | 5,892 | 237,374 | blur/exposure/blank |
| iqa | 237,374 | 31,805 | 205,569 | MUSIQ + NIQE |
| domain | 205,569 | 10,487 | 195,082 | field / orchard evidence |
| exact_dedup | 195,082 | 9,109 | 185,973 | byte/pixel identity |
| near_dedup | 185,973 | 0 | 185,973 | confirmed collapse / uncertainty isolation |
| holdout_license_archive | 185,973 | 7,678 | 178,295 | holdout / licence / archive-only |
| strict_eligible | 178,295 | 0 | 178,295 | final strict set |

