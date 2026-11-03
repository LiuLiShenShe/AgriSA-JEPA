# AgriField-Extended (internal working name) - dataset card

> This is an internal name.  It is **not** Agri-147K and must not be cited
> as the official Agri-FM+ corpus.

## 1. Identity

- Version: `R3v2_strict_rgb`
- Status: `AUTO_VALIDATED`
- Meaning: programmatic gates only; NOT human-verified and NOT PROVEN_NO_LEAKAGE
- Policy: `stage5d-policy-v2` / SHA-256 `f89e1a4515c4f7353618c73da77bebf77d1b78e570266a4349177f7fd2bfbc97`
- Accepted images: **169,326** (train 147,279 / val 22,047)
- Quarantined: 42,085; rejected: 9,301; holdout: 9,606; licence-blocked: 2,500

## 2. Relationship to upstream datasets

Agri-FM+ (CVPR Workshops 2025) publishes the *construction principles* of
Agri-147K but not a per-image list.  This corpus is a protocol-informed
reconstruction plus documented extensions, not a reproduction.

| source | accepted | upstream dataset(s) | license | redistribution | official entry |
|---|---:|---|---|---|---|
| artemis_imagesafari | 130,270 | Artemis;ImageSafari | CC-BY-SA-4.0 | yes-share-alike | https://registry.opendata.aws/artemis-image-safari/ |
| agrifield40k_pool | 21,418 | ACRECropWeed;GrassClover;LUCASVision;Maize-Weed;MuST-C;PalmerAmaranth;PerennialPlants;PhenoBench;RadishWheat;Ronin;RumexLeaves;SesameWeed;SorghumWeed;VCD;VegAnn;WE3DS;iNatWeeds | CC-BY-4.0 | yes | https://dtu-pas.github.io/agrifield40k/ |
| agrifm_ref58 | 15,386 | agrifm_ref58 | CC0-1.0 | unknown | https://doi.org/10.7910/DVN/TCKVEW |
| applebbch81 | 1,746 | AppleBBCH81 | CC-BY-4.0 | yes | https://www.kaggle.com/datasets/projectlzp201910094/applebbch81 |
| agrifm_ref11 | 382 | agrifm_ref11 | Apache-2.0 | unknown | https://www.kaggle.com/datasets/zzzzqi/maize-phenology-images-acquired-using-ndvi-camera |
| agrifm_ref9 | 84 | agrifm_ref9 | CC-BY-NC-4.0 | unknown | https://www.kaggle.com/datasets/nicolaasregnier/pinotnoirgrapes/data |
| agrifm_ref1 | 40 | agrifm_ref1 | CC0-1.0 | unknown | https://www.kaggle.com/datasets/voglinio/wheat-dataset-original |

## 3. Composition

- By source: {"artemis_imagesafari": 130270, "agrifield40k_pool": 21418, "agrifm_ref58": 15386, "applebbch81": 1746, "agrifm_ref11": 382, "agrifm_ref9": 84, "agrifm_ref1": 40}
- By crop (accepted; `unknown` = no evidence): {"common-bean": 41809, "unknown": 39058, "sorghum": 27031, "potato": 13930, "cowpea": 8271, "pearl-millet": 7158, "groundnut": 5094, "african-yam": 5091, "pigeon-pea": 4874, "banana": 4367, "finger-millet": 3404, "rice": 2371, "sweet-potato": 2210, "chickpea": 789, "lentil": 785, "cassava": 780, "wheat": 778, "soybean": 776, "maize": 750}
- By country (accepted; `unknown` = no evidence): {"unknown": 113552, "UGA": 18598, "COL": 15415, "USA": 15393, "TZA": 6368}
- By modality: {"RGB_VISIBLE": 169326}
- By group tier: {"OFFICIAL_GROUP": 130268, "DOCUMENTED_PATH_GROUP": 23164, "WEAK_PROXY": 15892, "UNKNOWN_GROUP": 2}

## 4. Quality pipeline

| metric | direction | range | accept | reject | role | model SHA-256 |
|---|---|---|---:|---:|---|---|
| musiq | higher_is_better | [0.0, 100.0] | 45.0 | 25.0 | GATE | `e95806b9eae5f381...` |
| niqe | lower_is_better | [0.0, 100.0] | 8.0 | 12.0 | DIAGNOSTIC_ONLY | `fa2aa4d12dd9595c...` |

MUSIQ is measured on a 0-100 scale; this was verified empirically (see
`reports/stage5d_threshold_trace.md`).  NIQE is diagnostic only, because it
was measured on a small minority of sources in Stage 5C.

Synthetic stress test (constructed degradations, **not** human labels):

- 210 base images from 7 source/domain strata, 3360 degraded variants, 16 degradations per base;
- clean retention 1.000, clean false-removal 0.000;
- full-pipeline recall 0.391; basic rules alone 0.227; IQA alone 0.332.
- strongest: `gaussian_blur` / severe blur (0.981); weakest: `occlusion` / blocked view (0.067).
- these numbers measure sensitivity to the constructed damage only; no natural-image precision/FPR is claimed.

## 5. Splits and independence

- Strong groups: 47,674
- Cross-split strong groups: 0
- Unknown-group images in the accepted set: 2
- Weak-proxy images (train only): 15,892
- Group-disjoint validation available: True
- Diagnostic random split (NOT group-disjoint) among accepted: 23,603

Unknown-group images never enter the validation split.  They are labelled
`TRAIN_ONLY_GROUP_UNKNOWN`; a separate diagnostic random split exists but is
not the group-disjoint validation set.

## 6. Deduplication

- Exact duplicate images: 9,384
- Confirmed near-duplicate pairs: 15,806
- Unresolved candidates (quarantined): 66,668
- Giant-cluster (texture collision) images: 0

## 7. Holdout

- PhenoBench-prefixed images in the mixed pool: 9,606 (blocked by Stage 5D: 9,606)
- Old R3 contained 1,864 PhenoBench-prefixed images; Stage 5C's ledger had flagged only 7,703.
- Exact holdout-hash matches in the new master: 0.
- pHash-neighbour candidates (distance <= 6) in the new master: 11; these are perceptual-hash collisions and were spot-checked, not auto-deleted.

- Levels *not* verifiable on this machine (PlantDoc, MinneApple,
  PlantSeg, MulticlassWeeds, GrowliFlower, CropAndWeed, DeepWeeds and
  the Roboflow family) were screened at source level only.

## 8. Not suitable for

* disease diagnosis or clinical/plant-pathology decisions;
* satellite or UAV remote sensing;
* any claim of official Agri-147K reproduction.

### Redistribution

Image files may be redistributed only where the upstream licence permits
it: Artemis / Image Safari (CC-BY-SA-4.0), AgriField-40K (CC-BY-4.0) and
AppleBBCH81 (CC-BY-4.0).  The registry records `redistribution = unknown`
for agrifm_ref1, agrifm_ref58, agrifm_ref11 and agrifm_ref9; for those the
publication ships only source ids, hashes and rebuild scripts, not pixels.

## 9. Hashes and reproduction

- Train CSV SHA-256: `f33b7cc24eee2d04e4f8b53d6b1026cfebe8fac5854971377216f3edb37abc30`
- Val CSV SHA-256: `a64a1222b27ddbdaa835a5c836a5330ceb1a71d1e18f0392f3f0b10a4d2e22d2`
- Policy SHA-256: `f89e1a4515c4f7353618c73da77bebf77d1b78e570266a4349177f7fd2bfbc97`

```bash
export AGRI_DATA_ROOT=/data/datasets/Stage5C
bash tools/stage5d_run_all.sh          # full pipeline (uses the GPU)
STAGE5D_SKIP_EMBEDDINGS=1 bash tools/stage5d_run_all.sh  # CPU-only build
```

## 10. Limits still open

- The corpus is automatically validated, never human-verified.
- Country is recovered only where the published key encodes it (Artemis);
  Image Safari country remains unknown.
- Crop is unknown for the small Agri-FM+ reference sources and for the
  mixed AgriField pool; `unknown` is kept as an explicit value.
- Holdouts that are not on this machine were screened only at source level.
- The QC gate is strong on severe blur and dark frames but weak on noise,
  JPEG recompression, colour cast and partial occlusion (see the synthetic
  stress table); those images are not certified clean by this pipeline.
- 76.9% of accepted images come from Artemis / Image Safari and 75.9% of
  the train split; use a capped or balanced sampling profile rather than
  the natural one if source balance matters for a run.
