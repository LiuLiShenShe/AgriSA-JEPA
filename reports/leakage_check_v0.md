# Structural leakage check

Metadata-only checks that pixel hashing cannot perform. This tool reports; it does not gate and it never modifies a manifest.

- Train manifest: `data/frozen/agrifield40k-v0_train.csv` (12,447 rows)
- Val manifest: `data/frozen/agrifield40k-v0_val.csv` (3,145 rows)

## 1. Capture-sequence separation

| Source | Train (timestamped) | Val (timestamped) | Val within window | Nearest gap (s) |
| --- | ---: | ---: | ---: | ---: |
| ACRECropWeed | 671 | 120 | 112 | 2 |

## 2. Grouping-key separation

- Distinct grouping keys: **1,227**
- Keys present in BOTH train and val: **31** (36 train / 31 val samples)

