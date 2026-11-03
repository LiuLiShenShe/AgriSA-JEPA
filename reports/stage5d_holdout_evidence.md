# Stage 5D - downstream holdout evidence

## What was checked

- Hash index built from 20,282 holdout files ({'phenobench': 13767, 'gwhd_2021': 6515}).
- Registered holdout sources present in the new corpus: **none**.

## The Stage 5C leak

- `agrifield40k_pool` contains **9,606** PhenoBench-prefixed
  images: {'train': 7703, 'val': 1903}.
- The Stage 5C ledger flagged only 7,703 of them as holdout
  (the `train/` prefix), so the `val/` half escaped.
- Old R3 contained **1,864** PhenoBench-prefixed images.
- New corpus contains **0** exact hash matches and **11** near matches.

## Limits

The following holdouts are not on this machine, so only source/identity
level evidence exists for them: plantdoc, minneapple, plantseg, multiclass_weeds, growliflower, cropandweed, deepweeds, roboflow_*.
No claim of "PROVEN_NO_LEAKAGE" is made for those.
