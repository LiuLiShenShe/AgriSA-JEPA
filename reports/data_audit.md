# AgriField-40K Data Audit

## Scope and provenance

The audit used the actual archive `/data/datasets/AgriField-40K/AgriField-40K.zip` and extracted files under `/data/datasets/AgriField-40K/agrifield40k/{train,val}`. It audited the pinned Hugging Face dataset revision `7f530c67572b6a572467adab840cdbbd7029b112`; the full archive SHA-256 matched the official Git LFS object hash: `63eec0b4f652340feddf9011c2ee26e2df341096018bdf7f5756cc29674e8976` (11,900,356,365 bytes).

The pinned dataset card states that AgriField-40K contains 39,963 field images from 17 source datasets, standardizes filenames as `[dataset_source]_[id]`, and is a CC BY-SA 4.0 compilation while each underlying image remains subject to its source license. This audit treats that card as the authoritative source registry, but does not infer an origin from a short prefix and matching count alone. iNatWeeds rows additionally join by exact photo ID and original filename to the pinned per-image metadata; attribution, observation URL, and the row's CC BY 4.0 or CC0 license are retained.

## File and split checks

| Check | Result |
| --- | ---: |
| Images audited | 39,963 |
| Original train / validation split | 32,136 / 7,827 |
| JPEG / PNG | 27,078 / 12,885 |
| Dimensions | 39,963 at 512 x 512 |
| Decoded RGB | 39,963 |
| Decode failures | 0 |

Every image was opened and fully decoded. Per-file records include relative path, source prefix, candidate and confirmed source, original ID and split, dimensions, mode, format, byte count, file SHA-256, decoded-pixel SHA-256, perceptual hashes, license fields, dataset-card URL/revision, and audit status/reason.

## Source decisions

| Source/status | Training rows retained | Validation rows retained | Decision |
| --- | ---: | ---: | --- |
| ACRECropWeed | 671 | 120 | Confirmed by explicit source token, original ID convention, and official retained count |
| LUCASVision | 8,944 | 2,245 | Confirmed by original LUCAS/year identifiers and aggregate official count |
| RadishWheat | 427 | 107 | Confirmed by original wheat/radish identifiers and official count |
| SorghumWeed | 137 | 35 | Confirmed by original split/source identifiers and official count |
| VegAnn | 1,245 | 360 | Confirmed by original `VegAnn_` identifiers and official count |
| iNatWeeds | 1,034 | 278 | Confirmed by exact per-image metadata join; individual license and attribution retained |
| PhenoBench | 0 | 0 | All 9,606 rows excluded |
| Pending source audit | 11,964 | 2,779 | Not included in either training manifest |

This leaves 15,603 confirmed, non-PhenoBench, deduplicated images in the two manifests: 12,458 training and 3,145 validation. The audit conservatively leaves 14,743 rows pending because the abbreviated prefix/original ID evidence was insufficient to establish the underlying source. Their candidate names are recorded, not treated as facts: MuST-C (7,242), VCD (2,258), WE3DS (1,553), SesameWeed (1,300), RumexLeaves (809), PalmerAmaranth (516), GrassClover (435), Maize-Weed (255), PerennialPlants (240), and Ronin (135). These rows have pending source and license fields and are excluded from pretraining manifests.

The source-level licenses recorded for confirmed rows follow the pinned card: CC BY 4.0 for ACRECropWeed, LUCASVision, RadishWheat, and SorghumWeed; CC BY 1.0 for VegAnn; and per-image CC BY 4.0 or CC0 for iNatWeeds. The card's compilation license is CC BY-SA 4.0. These records support traceability, not legal advice or a substitute for reviewing the original source terms and attribution requirements before redistribution or publication.

## PhenoBench exclusion

All 9,606 `pnb_` rows were excluded from both splits. Evidence combines the pinned AgriField card's `[dataset_source]_[id]` naming convention and PhenoBench retained count with inspection of the official PhenoBench unlabeled-patch sample ZIP central directory. The original patch names contain the documented acquisition date, split, patch category and coordinates; observed AgriField `pnb_` original names match that pattern. Any connected duplicate component touching a PhenoBench row is excluded as well. This is not a generic substring-only filter.

Official PhenoBench publishes unlabeled patch data for seven acquisition dates from April 25 through July 2, 2020, specifically including data for self-supervised pretraining; its site declares CC BY-SA 4.0. The downstream split overlap risk is therefore handled by excluding the entire identified PhenoBench source rather than retaining its nominally unlabeled patches.

## Duplicate screening

Exact file SHA-256, decoded RGB pixel SHA-256, and a two-screener perceptual candidate rule were used. A perceptual edge requires pHash Hamming distance at most 4 and dHash distance at most 6; the intersection is used to reduce smooth-scene collisions. The scan found 8 exact file-hash edges, 776 perceptual-candidate edges, 389 multi-image connected components, and 247 candidate edges crossing the original train/validation split. For non-PhenoBench confirmed rows, one deterministic representative per component is kept, preferring validation; 11 non-PhenoBench rows were excluded as nonrepresentatives. Included train and validation components are disjoint.

Perceptual matches are screening candidates, not proof that two photos depict the same scene. The 776 candidate edges have not all received manual visual adjudication. Review those candidates before making claims about semantic duplication or final benchmark leakage guarantees.

## Generated files

- `data/all_samples.csv`: one row for each of the 39,963 audited images.
- `data/exclusion_ledger.csv`: excluded and pending rows with reason codes.
- `data/duplicate_pairs.csv`: exact and perceptual candidate edges, including split crossing.
- `pretrain_manifest.csv`: include-only training samples, 12,458 rows.
- `val_manifest.csv`: include-only validation samples, 3,145 rows.
- `data/source_registry.json`: pinned evidence and conservative prefix decisions.
- `data/iNatWeeds_metadata.json`: tracked per-image attribution source metadata.
- `reports/data_audit.json`: machine-readable audit summary.

SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `pretrain_manifest.csv` | `3b60e1af8e3050e2354f2d21121dad51024e29eb4f1d06e6380aac21566cb204` |
| `val_manifest.csv` | `4a7338d9f105164dd97fbf5719adef4b03d0e61e45d337cea07f24fbc1905a45` |
| `data/all_samples.csv` | `c418a7f09ee2fb18f2f4f197f82e1012e8e2561b9680f66a284dfa40c7d0492c` |
| `data/iNatWeeds_metadata.json` | `8808f91f0be13832876d6e237a3b7cd870dd97a29b725b84bbb5e6ac9b9667a3` |

Re-run from the repository root after the archive has been extracted:

```bash
./.venv/bin/python -u -m tools.audit_data --workers 8 --verify-archive
```

## Sources

- AgriField-40K pinned card: https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/7f530c67572b6a572467adab840cdbbd7029b112/README.md
- AgriField-40K dataset page: https://huggingface.co/datasets/dtu-pcas/AgriField-40K
- AgriField-40K project: https://github.com/DTU-PAS/agrifield40k
- PhenoBench official data and license page: https://www.phenobench.org/dataset.html
