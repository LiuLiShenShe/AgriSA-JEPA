# Stage 5D - legacy funnel reconciliation

Re-ran the Stage 5C functions on the Stage 5C ledgers, using the Stage 5C
loader exactly (including the mtime rule that chose the unenriched ledger
for two sources).

| version | funnel final | manifest | funnel-only | manifest-only |
|---|---:|---:|---:|---:|
| R1 | 60603 | (60,603 reported) | - | - |
| R2 | 20952 | 20944 | 8 | 0 |
| R3 | 178295 | 177396 | 899 | 0 |

## Why the funnel kept images the manifest dropped

- R2: {"qc_status_QUARANTINE": 8}
- R3: {"qc_status_QUARANTINE": 899, "p0_rejected_but_not_a_funnel_step": 30, "p0_BLURRY_APPROXIMATION": 30}

The funnel never evaluated the P0 (paper-disclosed) predicate, while the
selection mask required `selected_for_R1`.  Images that passed every
funnel step but failed the paper-disclosed Laplacian rule survived the
funnel and were dropped from the manifest.  Every such image is listed in
`stage5d_funnel_mismatches.csv`.

## Legacy policy integrity

- `data/corpus_v2/policy.yaml` SHA-256: `45379ed1a669edb64778b97c1e9dd59ba9f3d8d5983418f642a2542b437a565c`
- recorded in the Stage 5C manifests: `45379ed1a669edb64778b97c1e9dd59ba9f3d8d5983418f642a2542b437a565c`
- unchanged: **True**
