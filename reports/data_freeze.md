# Data freeze record

**Status: PROVISIONAL.** The source and license gate is not met, so this freeze is provisional. Read the gate note below before using these files for anything that gets reported.

- Data version: `agrifield40k-v0`
- Pinned HF revision: `7f530c67572b6a572467adab840cdbbd7029b112`
- Publisher card: <https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/7f530c67572b6a572467adab840cdbbd7029b112/README.md>
- Generated at: `(not recorded)`

## Included sample count

| Split | Source manifest | Frozen | Removed by isolation |
| --- | ---: | ---: | ---: |
| train | 12,458 | **12,447** | 11 |
| val | 3,145 | **3,145** | 0 |
| **total** | **15,603** | **15,592** | **11** |

The frozen split holds 15,592 of the 15,603 samples in the include manifests, which is 99.93% of them. Frozen train and frozen val are disjoint by construction and asserted at write time.

## Per-source breakdown

| Source | Frozen train | Frozen train share | Frozen val | Frozen val share |
| --- | ---: | ---: | ---: | ---: |
| LUCASVision | 8,938 | 71.81% | 2,245 | 71.38% |
| VegAnn | 1,243 | 9.99% | 360 | 11.45% |
| iNatWeeds | 1,031 | 8.28% | 278 | 8.84% |
| ACRECropWeed | 671 | 5.39% | 120 | 3.82% |
| RadishWheat | 427 | 3.43% | 107 | 3.40% |
| SorghumWeed | 137 | 1.10% | 35 | 1.11% |

## License and review status

- Compilation license: `CC-BY-SA-4.0`
- Source/license gate met: **no**
- 14,743 of 39,963 images remain pending_audit. For 9 prefixes a literal source-index match now exists and for 1 the match is components-only, but none has been adopted: their sources/licences are not confirmed in the dataset version, so the source/license gate is not met and the freeze stays provisional.

| Source | License |
| --- | --- |
| ACRECropWeed | CC-BY-4.0 |
| LUCASVision | CC-BY-4.0 |
| RadishWheat | CC-BY-4.0 |
| SorghumWeed | CC-BY-4.0 |
| VegAnn | CC-BY-1.0 |
| iNatWeeds | per_image_metadata |

Unresolved prefixes, all excluded from the freeze: `gc`, `mc`, `mw`, `pags`, `per`, `rl`, `rn`, `saw`, `vcd`, `weds`. 14,743 rows. Their candidate licenses are recorded in `reports/stage2_data_audit.md` but are not applied to any row.

Provisioning note: 9 of those 10 prefixes now have a literal filename match against their named source's own published index, and 1 (`mc`) has components-only support; see `reports/stage2_source_index_evidence.md`. That file is evidence, not adoption — every one of those rows still carries `status=pending_audit` and stayed out of this freeze. The exclusion is a policy choice pending a curator's prefix map and a data-version bump, not a claim that the sources are unfindable.

## Exclusion reasons (all audited images)

`stage2b_isolation_unreviewed_candidate` counts only the samples that actually came out of the include manifests, so this table is a partition of the 39,963 audited images into included / excluded / pending. The candidate edges also name further samples that were already outside both manifests; those are not counted twice here.

| Reason | Rows |
| --- | ---: |
| unconfirmed_source_mapping | 14,743 |
| phenobench_source | 9,606 |
| duplicate_component_nonrepresentative | 11 |
| stage2b_isolation_unreviewed_candidate | 11 |
| duplicate_component_touches_phenobench | 0 |
| **included in frozen split** | **15,592** |
| **total** | **39,963** |

## Train/validation split rule

- Origin: publisher 80/20 split in the pinned HF revision (32,136 train / 7,827 val images before any filtering)
- Publisher statement: 80% training, 20% validation; split at sequence level for sequence-derived sources, random split for independent-image sources (<https://arxiv.org/abs/2608.07984>)
- Stage 1: one deterministic representative per connected component, preferring validation; any component touching PhenoBench excluded in full
- Stage 2B: both endpoints of every split-crossing or manifest-affecting candidate edge that is still pending_manual_review are excluded from the frozen split
- Frozen train/val disjoint: True

## SHA-256

### Frozen files

| File | SHA-256 |
| --- | --- |
| `data/frozen/agrifield40k-v0_train.csv` | `d8a179b81f2052c6d66d2f43d26505652e95b1090560d13c1e9d46efe3b27037` |
| `data/frozen/agrifield40k-v0_val.csv` | `4a7338d9f105164dd97fbf5719adef4b03d0e61e45d337cea07f24fbc1905a45` |

### Source manifests (inputs to this freeze)

| File | SHA-256 |
| --- | --- |
| `data/all_samples.csv` | `c418a7f09ee2fb18f2f4f197f82e1012e8e2561b9680f66a284dfa40c7d0492c` |
| `data/duplicate_pairs.csv` | `85ea70c5f7a734fa2c8caf5ebf98e3fa2d13a1a485881c5cf07e8482f8ea5c08` |
| `data/exclusion_ledger.csv` | `56547d1c1e12df8de27454e294252dd678bc220657737bba0b94bc7cf4c84303` |
| `data/source_registry.json` | `5123726d5c9198a6c50209d338dd645bb32650408fc7d4737af3d267a52843d0` |
| `pretrain_manifest.csv` | `3b60e1af8e3050e2354f2d21121dad51024e29eb4f1d06e6380aac21566cb204` |
| `val_manifest.csv` | `4a7338d9f105164dd97fbf5719adef4b03d0e61e45d337cea07f24fbc1905a45` |

`data/frozen/freeze_manifest.json` carries the same values in machine-readable form. Any mismatch between a frozen file and its recorded hash means the frozen file was edited after freezing and must be discarded.

## Known leakage risks

**unreviewed perceptual duplicate candidates.** 776 perceptual_candidate edges and 8 exact edges exist. The 258 split-crossing or manifest-affecting ones are isolated, but manual visual adjudication has not been performed on any of them, so the true duplicate rate is unknown.
  - Mitigation: conservative isolation rule; see reports/duplicate_review_protocol.md

**sequence-level split not independently verified.** The publisher states the split is at sequence level for sequence-derived sources. This audit verified pixel identity, not that the publisher actually honoured the sequence rule. Near-duplicate frames from the same sequence could therefore still straddle the frozen split.
  - Mitigation: manual review of the perceptual candidates is the only way to close this; the screener rule (pHash<=4 AND dHash<=6) is a bound, not a guarantee

**same-scene different-capture pairs below the screener threshold.** A pair of the same field on different days, or a wider crop of the same scene, can fall outside the pHash/dHash radius and never appear as an edge at all. No hash-based screener can find these.
  - Mitigation: none in this audit; requires manual or learned semantic matching

**unresolved sources could contain cross-split duplicates.** 247 cross-split perceptual candidates have both endpoints outside the include manifests, and all of them involve unresolved prefixes. They are quarantined, so they do not leak into the frozen split, but their provenance is unverified and they would need re-screening if ever admitted.
  - Mitigation: excluded from the frozen split entirely

## Isolation rule applied

258 candidate edges triggered isolation (247 cross-split, 11 manifest-affecting). Both endpoints of each triggered edge were dropped from the frozen split where they were present: 11 samples came out of the include manifests (11 train, 0 val). The edges also name 394 further samples that were already outside both manifests, so dropping them was a no-op for the manifests.

Frozen training manifest therefore holds 12,447 of the 12,458 samples in `pretrain_manifest.csv`. The validation manifest is untouched at 3,145 rows: every manifest-affecting edge it touches had its manifest endpoint on the training side.

**This is not completed manual deduplication.** It is a conservative isolation rule applied in the absence of manual deduplication. It removes any pair the screener flagged that could straddle the frozen split, and it pays for that by discarding images that are very likely fine. The true number of duplicates among the isolated samples is unknown until a human reviews `reports/duplicate_review/`.

## Unfinished items

- manual adjudication of all 784 duplicate candidate edges (0 reviewed, 784 pending_manual_review)
- source/license confirmation for the 10 unresolved prefixes (14,743 rows) — evidence now recorded (reports/stage2_source_index_evidence.md) but adoption is a curator decision
- resolution of the iNatWeeds per-image attribution/observation fields for redistribution (already retained in data/all_samples.csv)
- independent verification of the publisher sequence-level split rule

## Files

| File | Contents |
| --- | --- |
| `data/frozen/agrifield40k-v0_train.csv` | frozen training manifest, 12,447 rows |
| `data/frozen/agrifield40k-v0_val.csv` | frozen validation manifest, 3,145 rows |
| `data/frozen/freeze_manifest.json` | this record in machine-readable form |
| `data/duplicate_review.csv` | one row per candidate edge, all `pending_manual_review` |
| `reports/stage2_data_audit.md` | source/license audit for the unresolved prefixes |
| `reports/data_distribution.md` | manifest distribution and cap analysis |
| `reports/duplicate_review_protocol.md` | the human review protocol |
| `reports/duplicate_review/index.html` | offline contact sheet |

Regenerate everything with:

```bash
./.venv/bin/python -u -m tools.stage2_data_audit
```

## How these numbers were verified

Every count in this record is read from the Stage 1 artefacts, not typed in. The commands run to produce and check them:

```bash
# input shape
wc -l data/all_samples.csv data/exclusion_ledger.csv data/duplicate_pairs.csv \
       pretrain_manifest.csv val_manifest.csv
# status / pending-prefix / manifest-membership counts
./.venv/bin/python -u -c "import csv,collections; \
  R={r['sample_id']:r for r in csv.DictReader(open('data/all_samples.csv',newline='',encoding='utf-8'))}; \
  print(collections.Counter(r['status'] for r in R.values()))"
# regenerate this freeze
./.venv/bin/python -u -m tools.stage2_data_audit
# confirm the frozen split is disjoint and matches the recorded hashes
sha256sum data/frozen/*.csv
```

Asserted at write time: every edge endpoint resolves to an audited sample, no `sample_id` appears in both manifests, the frozen train and val sets are disjoint, and the frozen train partition plus the isolated samples exactly covers the training manifest. A run that fails any of these raises instead of writing output.

