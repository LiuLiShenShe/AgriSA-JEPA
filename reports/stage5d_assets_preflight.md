# Stage 5D - assets preflight

## Repository

- Branch: `agri-stage5d-corpus-validation`
- HEAD: `2867e7f4b09814c268612e893c34216506cbe55e`
- Stage 5C recorded HEAD: `2867e7f4b09814c268612e893c34216506cbe55e`
- Match: **True**
- Working tree: ['M .gitignore', '?? CLAUDE.md', '?? data/corpus_v3/', '?? reports/stage5d_assets_preflight.json', '?? reports/stage5d_assets_preflight.md', '?? reports/stage5d_build_summary.json', '?? reports/stage5d_dataset_card.md', '?? reports/stage5d_dataset_composition.csv', '?? reports/stage5d_dataset_composition.md', '?? reports/stage5d_duplicate_audit.json', '?? reports/stage5d_duplicate_audit.md', '?? reports/stage5d_figures/', '?? reports/stage5d_final_report.md', '?? reports/stage5d_funnel_mismatches.csv', '?? reports/stage5d_funnel_table.csv', '?? reports/stage5d_gates.json', '?? reports/stage5d_groups_and_splits.csv', '?? reports/stage5d_groups_and_splits.json', '?? reports/stage5d_groups_and_splits.md', '?? reports/stage5d_groups_split.csv', '?? reports/stage5d_holdout_evidence.json', '?? reports/stage5d_holdout_evidence.md', '?? reports/stage5d_legacy_funnel_reconciliation.json', '?? reports/stage5d_legacy_funnel_reconciliation.md', '?? reports/stage5d_loader_smoke.json', '?? reports/stage5d_metadata_coverage.csv', '?? reports/stage5d_metadata_coverage.md', '?? reports/stage5d_modality_audit.json', '?? reports/stage5d_modality_audit.md', '?? reports/stage5d_qc_calibration.csv', '?? reports/stage5d_qc_policy_diff.md', '?? reports/stage5d_rebuild_check.json', '?? reports/stage5d_source_balance.csv', '?? reports/stage5d_source_composition.csv', '?? reports/stage5d_synthetic_qc.csv', '?? reports/stage5d_synthetic_qc.json', '?? reports/stage5d_threshold_sensitivity.json', '?? reports/stage5d_threshold_trace.json', '?? reports/stage5d_threshold_trace.md', '?? src/registry/stage5d_dedup.py', '?? src/registry/stage5d_funnel.py', '?? src/registry/stage5d_gates.py', '?? src/registry/stage5d_group_policy.py', '?? src/registry/stage5d_metadata.py', '?? src/registry/stage5d_metadata_schema.py', '?? src/registry/stage5d_modality.py', '?? src/registry/stage5d_qc_policy.py', '?? src/registry/stage5d_qc_validation.py', '?? tests/test_stage5d_corpus.py', '?? tools/stage5d_audit_score_scales.py', '?? tools/stage5d_build_corpus.py', '?? tools/stage5d_compute_missing_iqa.py', '?? tools/stage5d_final_report.py', '?? tools/stage5d_generate_tables.py', '?? tools/stage5d_holdout_audit.py', '?? tools/stage5d_loader_smoke.py', '?? tools/stage5d_preflight.py', '?? tools/stage5d_publish_materials.py', '?? tools/stage5d_rebuild_check.py', '?? tools/stage5d_reconcile_funnel.py', '?? tools/stage5d_run_all.sh', '?? tools/stage5d_sampling_profiles.py', '?? tools/stage5d_sensitivity.py', '?? tools/stage5d_synthetic_qc.py', '?? tools/stage5d_validate_modality.py']

## Environment

- Python: 3.13.12 at `/fj/jepa-二维图像任务/.venv-uv`
- CUDA: torch 2.9.1+cu128, cuda 12.8, available=True
- GPUs: ['NVIDIA RTX A6000', 'NVIDIA RTX A6000']
- Conda prefix detected: False
- Disk free: 445.7 GB

## Data root

- `/data/datasets/Stage5C` exists: **True**
  - `agrifm_ref1`: 1,680 files
  - `agrifm_ref11`: 5,900 files
  - `agrifm_ref20`: 1,081 files
  - `agrifm_ref57`: 31,500 files
  - `agrifm_ref58`: 35,849 files
  - `agrifm_ref6`: 901 files
  - `agrifm_ref7`: 830 files
  - `agrifm_ref8`: 252 files
  - `agrifm_ref9`: 233 files
  - `agrifm_ref90`: 1,561 files
  - `artemis_imagesafari`: 135,655 files
  - `perennial_plants`: 799 files

## Assets present and hashed

| asset | exists | bytes | SHA-256 |
|---|---|---:|---|
| `R0_agrifm35_source_pool_manifest.csv` | True | 115,276,824 | `8cbd124525d43dd2...` |
| `R1_agrifm147_rebuild_manifest.csv` | True | 24,157,373 | `fdb0aab21080269a...` |
| `R2_field_strict_manifest.csv` | True | 8,662,021 | `3d0519cf849948ad...` |
| `R3_field_extended_manifest.csv` | True | 87,803,200 | `ce32e5c819d0c1e4...` |
| `images/artemis_imagesafari.csv` | True | 119,926,547 | `5167fc62ec2a02b5...` |
| `images/artemis_imagesafari_qc.csv` | True | 122,209,670 | `75b3816a11c068e6...` |
| `images/agrifm_ref11.csv` | True | 4,268,165 | `136e4cb51a417772...` |
| `images/agrifm_ref11_qc.csv` | True | 4,369,548 | `5bcb513a4faba786...` |
| `images/agrifm_ref58.csv` | True | 15,808,573 | `68a7764c6d88ea02...` |
| `images/agrifm_ref58_qc.csv` | True | 16,199,419 | `2c2cd76575f0d428...` |
| `images/agrifield40k_pool.csv` | True | 28,601,294 | `0463ecedfcfdaebd...` |
| `images/agrifield40k_pool_qc.csv` | True | 29,275,953 | `6a4e632ee1745c3f...` |
| `R0_agrifm35_source_pool_manifest.json` | True | 1,772 | `3df8e80ccd5a500c...` |
| `R1_agrifm147_rebuild_manifest.json` | True | 1,607 | `b414224a9dc7b3d5...` |
| `R2_field_strict_manifest.json` | True | 1,376 | `16780dc9f89c1dd8...` |
| `R3_field_extended_manifest.json` | True | 1,493 | `f01c148d9e108140...` |
| `policy.yaml` | True | 2,272 | `45379ed1a669edb6...` |
| `artemis/remote_index.csv` | True | 1,862,148,285 | `8ad734602cd64340...` |

Missing assets: **none**

## Path hygiene

- Legacy manifests embed absolute paths: 4
- Stage 5D rule: relative_path + logical data root (AGRI_DATA_ROOT / --data-root)
