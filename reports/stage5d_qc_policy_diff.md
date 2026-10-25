# Stage 5D - policy diff (Stage 5C v1 -> Stage 5D v2)

- Old policy: `/fj/jepa-二维图像任务/data/corpus_v2/policy.yaml` (version `stage5c-policy-v1`)
- New policy: `/fj/jepa-二维图像任务/data/corpus_v3/policy.yaml` (version `stage5d-policy-v2`)

## The 0.35 / 0.22 question

- Old YAML: `quality.iqa.accept_min = 0.35`, `reject_max = 0.22` per metric `musiq`.
- These values are on a 0-1 scale, but pyIQA MUSIQ emits 0-100.
- Measured MUSIQ over 212,194 scored images spans 12.58-79.13, so the
  interval [0.22, 0.35] contains **zero** images: as written the old YAML
  could not have rejected anything.
- The Stage 5C freeze code used 45.0 / 25.0, i.e. the 0-100 scale, and the
  acceptance counts are only reproducible with those values.
- Conclusion: the effective old thresholds were 45 / 25; the YAML fields
  were dead configuration.  They are not copied forward.

## New metric definitions

| metric | direction | range | accept | reject | role | model SHA-256 |
|---|---|---|---:|---:|---|---|
| musiq | higher_is_better | [0.0, 100.0] | 45.0 | 25.0 | GATE | `e95806b9eae5f381...` |
| niqe | lower_is_better | [0.0, 100.0] | 8.0 | 12.0 | DIAGNOSTIC_ONLY | `fa2aa4d12dd9595c...` |

MUSIQ is a gate.  NIQE is `DIAGNOSTIC_ONLY`, because Stage 5C measured it
on 3,776 images (1.8% of the pool); it never participated in the old
acceptance decision either.

## Changed keys

| key | old | new |
|---|---|---|
| `created` | `2026-10-24` | `2026-10-25` |
| `paths.data_root` | `/data/datasets/Stage5C` | `STAGE5D_ROOT` |
| `version` | `stage5c-policy-v1` | `stage5d-policy-v2` |

## Moved keys (same value, new location)

| old key | new key | value |
|---|---|---|
| `quality.blank_pixel_fraction_max` | `quality.basic.blank_pixel_fraction_max` | `0.85` |
| `quality.bright_mean_min` | `quality.basic.bright_mean_min` | `238.0` |
| `quality.dark_mean_max` | `quality.basic.dark_mean_max` | `22.0` |
| `quality.entropy_min` | `quality.basic.entropy_min` | `2.6` |
| `quality.extreme_aspect_ratio_max` | `quality.basic.extreme_aspect_ratio_max` | `8.0` |
| `quality.laplacian_var_min` | `quality.basic.laplacian_var_min` | `35.0` |
| `quality.min_short_side` | `quality.basic.min_short_side` | `224` |
| `quality.quarantine_on_conflict` | `quality.basic.quarantine_on_conflict` | `True` |
| `quality.tenengrad_min` | `quality.basic.tenengrad_min` | `25.0` |

The basic-QC thresholds moved from `quality.*` to `quality.basic.*` with
identical values, so the numeric rule did not change; only its location
became explicit.

- Removed keys: `budget.max_artemis_images`, `budget.max_raw_download_bytes`, `budget.target_r3_strict_images`, `dedup.dhash_confirmation_distance`, `dedup.embedding.batch_size`, `dedup.embedding.cosine_similarity`, `dedup.embedding.device`, `dedup.embedding.enabled`, `dedup.embedding.model`, `dedup.phash_candidate_distance`, `dedup.pixel_sha_exact`, `domain.apply_vision_to`, `domain.strict_excluded`, `domain.vision_accept_min`, `domain.vision_conflict_margin`, `domain.vision_model`, `download.chunk_bytes`, `download.max_parallel_files`, `download.retries`, `download.timeout_seconds`, `download.user_agent`, `groups.hash_salt`, `groups.min_group_size_for_split`, `paths.crops`, `paths.embeddings`, `paths.work`, `quality.cleanvision.analyzers`, `quality.cleanvision.enabled`, `quality.cleanvision.sample_per_source`, `quality.iqa.accept_min`, `quality.iqa.batch_size`, `quality.iqa.conflict_gap`, `quality.iqa.device`, `quality.iqa.enabled`, `quality.iqa.metric`, `quality.iqa.reject_max`, `quality.require_decode`
- Added keys: `dedup.aspect_tolerance`, `dedup.confirm_phash_max`, `dedup.dhash_max_distance`, `dedup.embedding_cosine_min`, `dedup.phash_max_distance`, `dedup.use_embeddings`, `domain.excluded_labels`, `domain.quarantine_labels`, `groups.salt`, `groups.unknown_group_split`, `holdout.source_ids`, `modality.accept_status`, `modality.quarantine_status`, `modality.reject_status`, `notes`, `paths.agrifield_samples`, `paths.artemis_index`, `quality.basic.near_threshold_band`, `quality.iqa.metrics.musiq.accept_threshold`, `quality.iqa.metrics.musiq.direction`, `quality.iqa.metrics.musiq.model`, `quality.iqa.metrics.musiq.model_sha256`, `quality.iqa.metrics.musiq.reject_threshold`, `quality.iqa.metrics.musiq.role`, `quality.iqa.metrics.musiq.scale_evidence`, `quality.iqa.metrics.musiq.score_range`, `quality.iqa.metrics.niqe.accept_threshold`, `quality.iqa.metrics.niqe.direction`, `quality.iqa.metrics.niqe.model`, `quality.iqa.metrics.niqe.model_sha256`, `quality.iqa.metrics.niqe.reject_threshold`, `quality.iqa.metrics.niqe.role`, `quality.iqa.metrics.niqe.scale_evidence`, `quality.iqa.metrics.niqe.score_range`, `quality.iqa.missing_score_action`, `supersedes`, `target.r3v2_strict_images`

