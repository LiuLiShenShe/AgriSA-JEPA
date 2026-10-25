# Stage 5D - capture groups and splits

| source | accepted | official | documented path | weak | unknown | train | val | unknown in val | cross-known |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| artemis_imagesafari | 130,270 | 130,268 | 0 | 0 | 2 | 111,760 | 18,508 | 0 | 0 |
| agrifield40k_pool | 21,418 | 0 | 21,418 | 0 | 0 | 18,242 | 3,176 | 0 | 0 |
| agrifm_ref58 | 15,386 | 0 | 0 | 15,386 | 0 | 15,386 | 0 | 0 | 0 |
| applebbch81 | 1,746 | 0 | 1,746 | 0 | 0 | 1,383 | 363 | 0 | 0 |
| agrifm_ref11 | 382 | 0 | 0 | 382 | 0 | 382 | 0 | 0 | 0 |
| agrifm_ref9 | 84 | 0 | 0 | 84 | 0 | 84 | 0 | 0 | 0 |
| agrifm_ref1 | 40 | 0 | 0 | 40 | 0 | 40 | 0 | 0 | 0 |

- Strong groups: 47,674; cross-split strong groups: 0
- Unknown-group images: 2; of them in val: 0
- Weak-proxy images (train only): 15,892
- Group-disjoint validation available: True

Only OFFICIAL_GROUP and DOCUMENTED_PATH_GROUP support a group-disjoint
claim.  UNKNOWN_GROUP images are kept for training and never used for
internal validation; the separate DIAGNOSTIC_RANDOM_VAL split exists for
diagnostics only and must never be reported as group-disjoint.

