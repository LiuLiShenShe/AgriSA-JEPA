# Stage 5D - IQA score-scale and threshold trace

- Legacy policy file: `data/corpus_v2/policy.yaml`
- YAML thresholds: accept `0.35`, reject `0.22`
- Stage 5C code defaults: accept `45.0`, reject `25.0`
- Measured MUSIQ range: **15.834 - 79.125** over 211,524 scored images

## Verdict

- YAML thresholds would pass **211,524 / 211,524** (1.0000).
- Code defaults pass 209,288, quarantine 2,128, reject 108.
- Conclusion: the YAML values are on a different scale and were never in force; the effective thresholds were 45/25.

## NIQE

- NIQE measured on 2,095 / 243,694 rows (0.0086).
- NIQE never entered the admission decision: `iqa_accept()` reads only `iqa_musiq`.  Stage 5C reports that said "MUSIQ + NIQE" overstated the gate.

## Ledger selection under the old loader

| source | old loader chose | rows | MUSIQ scored | min | max | NIQE scored |
|---|---|---:|---:|---:|---:|---:|
| agrifield40k_pool | agrifield40k_pool_qc.csv | 39,963 | 39,963 | 20.040691375732422 | 78.843017578125 | 0 |
| agrifm_ref1 | agrifm_ref1.csv | 670 | 0 | - | - | 0 |
| agrifm_ref11 | agrifm_ref11_qc.csv | 5,899 | 5,899 | 20.91962242126465 | 75.32869720458984 | 0 |
| agrifm_ref20 | agrifm_ref20_qc.csv | 1,081 | 1,081 | 29.93341636657715 | 77.81155395507812 | 0 |
| agrifm_ref57 | agrifm_ref57.csv | 31,500 | 0 | - | - | 0 |
| agrifm_ref58 | agrifm_ref58_qc.csv | 23,205 | 23,205 | 32.60713577270508 | 79.1254653930664 | 0 |
| agrifm_ref6 | agrifm_ref6_qc.csv | 900 | 900 | 40.032325744628906 | 73.71937561035156 | 900 |
| agrifm_ref7 | agrifm_ref7_qc.csv | 829 | 829 | 19.2172908782959 | 77.77117919921875 | 829 |
| agrifm_ref8 | agrifm_ref8_qc.csv | 251 | 251 | 61.64869689941406 | 71.7670669555664 | 251 |
| agrifm_ref9 | agrifm_ref9_qc.csv | 115 | 115 | 31.94976043701172 | 68.88580322265625 | 115 |
| agrifm_ref90 | agrifm_ref90_qc.csv | 520 | 520 | 39.509925842285156 | 76.24457550048828 | 0 |
| applebbch81 | applebbch81_qc.csv | 1,838 | 1,838 | 28.882537841796875 | 73.8553237915039 | 0 |
| artemis_imagesafari | artemis_imagesafari_qc.csv | 135,623 | 135,623 | 15.834232330322266 | 78.87425994873047 | 0 |
| sesame_weed | sesame_weed_qc.csv | 1,300 | 1,300 | 50.09016418457031 | 77.40180206298828 | 0 |

Two sources (`agrifm_ref1`, `agrifm_ref57`) had a newer *plain* ledger, so the old loader read the unenriched file with empty `iqa_musiq`.  Those rows became `IQA_MISSING` and were excluded from R2/R3.  Stage 5D always reads the enriched ledger when it exists.
