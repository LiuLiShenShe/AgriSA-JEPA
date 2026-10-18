# Stage 2B manifest distribution analysis

Analysis only. Nothing in this section changes `pretrain_manifest.csv` or `val_manifest.csv`; the cap schemes below are worked examples, not applied weighting.

## Per-source counts

Training manifest: **12,458** samples. Validation manifest: **3,145** samples.

| Source | Train | Train share | Val | Val share | Val/(Train+Val) | License |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LUCASVision | 8,944 | 71.79% | 2,245 | 71.38% | 20.06% | CC-BY-4.0 |
| VegAnn | 1,245 | 9.99% | 360 | 11.45% | 22.43% | CC-BY-1.0 |
| iNatWeeds | 1,034 | 8.30% | 278 | 8.84% | 21.19% | per_image_metadata |
| ACRECropWeed | 671 | 5.39% | 120 | 3.82% | 15.17% | CC-BY-4.0 |
| RadishWheat | 427 | 3.43% | 107 | 3.40% | 20.04% | CC-BY-4.0 |
| SorghumWeed | 137 | 1.10% | 35 | 1.11% | 20.35% | CC-BY-4.0 |
| **Total** | **12,458** | **100.00%** | **3,145** | **100.00%** | **20.16%** | |

Per LUCASVision prefix (`lvb*`), since it is the only source split across several prefixes:

| Prefix | Train | Val |
| --- | ---: | ---: |
| `lvb11` | 1,254 | 314 |
| `lvb12` | 229 | 58 |
| `lvb13` | 1,081 | 271 |
| `lvb14` | 636 | 160 |
| `lvb15` | 492 | 123 |
| `lvb16` | 908 | 228 |
| `lvb21` | 661 | 166 |
| `lvb22` | 696 | 175 |
| `lvb31` | 381 | 96 |
| `lvb32` | 1,152 | 289 |
| `lvb33` | 538 | 135 |
| `lvb55` | 916 | 230 |

## LUCASVision share of the training manifest

**LUCASVision is 8,944 of 12,458 training samples = 71.79%.** In validation it is 2,245 of 3,145 = 71.38%. Combined across both manifests it is 71.71% of the included corpus.

The next largest source, VegAnn, is 1,245 samples (9.99%). LUCASVision therefore outweighs it 7.2x. The two smallest sources are SorghumWeed (137) and RadishWheat (427), so the corpus spans roughly two orders of magnitude between its largest and smallest contributor.

Two further facts matter for interpretation. First, the publisher records LUCASVision as a handheld-camera source of 12 crop species (<https://arxiv.org/abs/2608.07984>, Table 1), so it is a wide, geographically and seasonally spread survey rather than a single acquisition burst; its dominance is not an artefact of one shoot. Second, its audited rows arrive through 12 distinct `lvb*` prefixes (1,156-1,569 rows each), which is the only source in the corpus split across several prefixes. Both cuts are shown above. What the audit cannot tell you is how close LUCASVision is to natural-image photography in distribution: that is an empirical question, and the visual evidence for it is in the contact sheet.

## Cap schemes (worked examples, NOT applied)

A cap is applied by keeping the first N samples of the capped source in manifest order. Because the manifests are written in sorted `sample_id` order by Stage 1, "first N" is deterministic and reproducible.

### Scheme A — cap LUCASVision at 5,000

The mildest option. Every source keeps all of its samples except the one dominant source, which is trimmed to 5,000.

| Source | Natural | Capped | Natural share | Capped share |
| --- | ---: | ---: | ---: | ---: |
| LUCASVision | 8,944 | 5,000 | 71.79% | 58.73% |
| VegAnn | 1,245 | 1,245 | 9.99% | 14.62% |
| iNatWeeds | 1,034 | 1,034 | 8.30% | 12.14% |
| ACRECropWeed | 671 | 671 | 5.39% | 7.88% |
| RadishWheat | 427 | 427 | 3.43% | 5.02% |
| SorghumWeed | 137 | 137 | 1.10% | 1.61% |
| **Total** | **12,458** | **8,514** | | |

### Scheme B — cap every source at 1,000

Equal-volume per source. This is the strongest flattening option and it throws away the most data: 3 of the 6 confirmed sources sit above 1,000, so only the sources already below the cap keep all of their samples.

| Source | Natural | Capped | Natural share | Capped share |
| --- | ---: | ---: | ---: | ---: |
| LUCASVision | 8,944 | 1,000 | 71.79% | 23.61% |
| VegAnn | 1,245 | 1,000 | 9.99% | 23.61% |
| iNatWeeds | 1,034 | 1,000 | 8.30% | 23.61% |
| ACRECropWeed | 671 | 671 | 5.39% | 15.84% |
| RadishWheat | 427 | 427 | 3.43% | 10.08% |
| SorghumWeed | 137 | 137 | 1.10% | 3.23% |
| **Total** | **12,458** | **4,235** | | |

| | Natural | Scheme A (LUCASVision<=5000) | Scheme B (all<=1000) |
| --- | ---: | ---: | ---: |
| Training samples retained | 12,458 | 8,514 | 4,235 |
| Samples discarded | 0 | 3,944 | 8,223 |
| Largest source share | 71.79% | 58.73% | 23.61% |

## Risk assessment

**Single-source dominance — present, and material.** 71.79% of the training manifest is one source. For I-JEPA-style pretraining this is not automatically wrong — LUCASVision is real field imagery and more data generally helps — but it does mean the learned prior is mostly a LUCAS prior. Any downstream evaluation on a source that is visually distinct from LUCAS (robot, drone, stereo RGB-D) is a transfer test, not an in-distribution test. It also means a change in how the LUCAS rows were selected would move the whole pretraining result.

**Small-source oversampling — present.** With a natural sampler the six sources contribute between 137 and 8,944 training samples. SorghumWeed, at 1.10% of the corpus, is unlikely to move the loss at all; for practical purposes it is decorative. If a downstream benchmark is drawn from a small source, the pretraining signal from that source is negligible regardless of how relevant it is.

**Recommendation.** Keep the frozen manifest as the default, so the Stage 1 audit and the freeze record stay the reference. For the Stage 2 pretraining runs, use a source-balanced sampler (per-source sampling weight proportional to `1/sqrt(source_count)`) rather than a hard cap: it preserves every image, flattens the effective distribution far more than Scheme B, and does not require editing the manifests. If a hard cap is wanted for a specific ablation, Scheme A is the defensible one — it discards 3,944 images (31.66%) and keeps LUCASVision as the largest single source, while Scheme B discards 66.01% of the corpus and makes LUCASVision a minority source by volume.

