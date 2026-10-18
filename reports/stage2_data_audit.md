# Stage 2B source and license audit

Pinned dataset revision: `7f530c67572b6a572467adab840cdbbd7029b112`. Publisher card: <https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/7f530c67572b6a572467adab840cdbbd7029b112/README.md>.

## Headline

- Audited images: **39,963**
- Rows `pending_audit` before Stage 2B: **14,743**
- Rows that moved to `confirmed` in Stage 2B: **0**
- Rows still `pending_audit`: **14,743**

**No unresolved prefix was auto-confirmed into the frozen split.** The audit is conservative by design: provenance evidence is recorded but the freeze is driven by the Stage 1 artefact columns (`status`/`source`), so recording an evidence class can never silently admit a row. What the Stage 2B evidence pass found:

- **9 of 10 prefixes** now have a literal filename match against the named source's OWN published archive/file index (`evidence_class: original_source_index`): `gc`, `vcd`, `pags`, `rn`, `weds`, `mw`, `rl`, `saw`, `per`. Their observed original IDs appear verbatim in the source dataset the card names.
- **1 prefix** (`mc`/MuST-C) is `supporting_only`: every component of its filename matches the official BonnData archive, but the official naming is a nested path, not the flattened string AgriField stores.
- **0 prefix(es)** have no evidence.

These prefixes remain `pending_audit` in the freeze and stay out of the frozen split until a curator opts in by promoting them. The recommendation and the mechanics for doing that are in `reports/data_freeze.md`.

## What the publisher materials do establish

- <https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/7f530c67572b6a572467adab840cdbbd7029b112/README.md> — AgriField-40K contains 39,963 images from 17 source datasets and renames them [dataset_source]_[id]; the compilation is CC BY-SA 4.0 and each image stays under its source license.
  - Establishes: The naming convention, the 17-source list, and each source license and retained count.
  - Does not establish: It does not publish the short-prefix to source map. It never states that "mc" means MuST-C.
- <https://arxiv.org/abs/2608.07984> — Paper Sec. 3 / Table 1 list all 17 sources with retained counts and licenses; the split is 80/20, split at sequence level for sequence-derived sources and randomly for independent-image sources.
  - Establishes: Retained counts, licenses, and the publisher split rule. It also states that only RGB images were kept and annotations discarded.
  - Does not establish: It contains no prefix map either. The split rule is the publisher's stated intent, not something this audit verifies per image.

## Per-prefix review

| Prefix | Candidate source | Rows | Card retained | License if confirmed | Evidence found | Decision |
| --- | --- | ---: | ---: | --- | --- | --- |
| `gc` | GrassClover | 435 | 435 | CC-BY-SA-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `mc` | MuST-C | 7,242 | 7,242 | CC-BY-4.0 | supporting_only | **pending_audit (supporting evidence only)** |
| `mw` | Maize-Weed | 255 | 255 | CC-BY-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `pags` | PalmerAmaranth | 516 | 516 | CC-BY-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `per` | PerennialPlants | 240 | 240 | MIT | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `rl` | RumexLeaves | 809 | 809 | CC-BY-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `rn` | Ronin | 135 | 135 | CC-BY-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `saw` | SesameWeed | 1,300 | 1,300 | CC0-1.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `vcd` | VCD | 2,258 | 2,258 | CC-BY-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |
| `weds` | WE3DS | 1,553 | 1,553 | CC-BY-4.0 | original_source_index | **pending_audit (adoptable: literal source-index match)** |

### Detail

#### `gc` — GrassClover (435 rows)

- Original-ID pattern observed: `biomass_image_test_0000` (435 distinct original IDs)
- Card retained count: 435 (matches the audited row count)
- Checked: The official GrassClover biomass archive (`biomass_data.zip`, download_id 4771, published by the Aalborg University AILab) contains the literal entries `test/images/biomass_image_test_0100.jpg` and `test/images/biomass_image_test_0101.jpg`, and its metadata CSV `test/biomass_test_data.csv` literally lists `biomass_image_test_0100.jpg`. The archive holds 261 train + 174 test = 435 images, matching the card.
- URL: <https://theailab.dk/grassclover-dataset/>
  - source index: <https://theailab.dk/grassclover-dataset/>
  - source index: <https://theailab.dk/?sdm_process_download=1&download_id=4771>
  - source index: <https://vision.eng.au.dk/grass-clover-dataset/>
- Establishes: The exact original filename stem `biomass_image_test_0100` is a literal entry in the named source's own archive AND in its own metadata table, and the image count equals the card retained count.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `mc` — MuST-C (7,242 rows)

- Original-ID pattern observed: `plot162_day230601_pos00_nikon_0` (7242 distinct original IDs)
- Card retained count: 7,242 (matches the audited row count)
- Checked: BonnData record 10.60507/FK2/OX9XTM (MuST-C, Chong et al., Scientific Data) publishes an image archive whose central directory contains the nested paths `images/UGV-RGB/230601/plot162/pos01/nikon_0.jpeg`, with `pos00`-`pos03`, dates `230601/230613/230616`, and `nikon_0`-`nikon_19`.
- URL: <https://doi.org/10.60507/FK2/OX9XTM>
  - source index: <https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/OX9XTM>
  - source index: <https://bonndata.uni-bonn.de/api/access/datafile/8066>
  - source index: <https://github.com/PRBonn/MuST-C>
- Establishes: Every component of the observed original ID (plot number, date token, position, camera index) corresponds to a real path component of the official MuST-C archive, and the archive count is consistent with the card retained count.
- Does not establish: The official MuST-C naming convention is a NESTED PATH (`<date>/plotNNN/posNN/nikon_N.jpeg`), not the flattened string AgriField stores (`plot162_day230601_pos01_nikon_0`). The `day` token has no official counterpart. So the mapping is strongly supported but NOT a literal filename match.
- **Evidence: `supporting_only`** — strong match on components, but not a literal filename match; stays `pending_audit`.

#### `mw` — Maize-Weed (255 rows)

- Original-ID pattern observed: `C00020826_output` (255 distinct original IDs)
- Card retained count: 255 (matches the audited row count)
- Checked: Mendeley record 10.17632/jjbfcckrsp "Maize-Weed Image Dataset" publishes `Maize-Weed Image Dataset.rar`, whose listing contains `Maize-Weed Image Dataset/Dry Season Maize Weed Images/C00020826_output.jpeg` and `.../C00020908_output.jpeg`. The v2 file API independently lists the same `C0002xxxx` namespace.
- URL: <https://data.mendeley.com/public-api/datasets/jjbfcckrsp>
  - source index: <https://data.mendeley.com/public-api/datasets/jjbfcckrsp>
  - source index: <https://data.mendeley.com/api/datasets/jjbfcckrsp/files>
- Establishes: The exact original filename stem `C00020826_output` is a literal entry in the named source's own archive.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `pags` — PalmerAmaranth (516 rows)

- Original-ID pattern observed: `19345061_ext_cotton_20210825_125203` (516 distinct original IDs)
- Card retained count: 516 (matches the audited row count)
- Checked: The PAGS8 author repository `geezacoleman/Palmer-detection` contains `data/xml/19345061_ext_cotton_20210825_125203.xml`, whose `<filename>` element is literally `19345061_ext_cotton_20210825_125203.jpg`. The README ties those labels to the PAGS8 images published via Weed-AI.
- URL: <https://github.com/geezacoleman/Palmer-detection>
  - source index: <https://raw.githubusercontent.com/geezacoleman/Palmer-detection/main/data/xml/19345061_ext_cotton_20210825_125203.xml>
  - source index: <https://weed-ai.sydney.edu.au/datasets/5c78d067-8750-4803-9cbe-57df8fae55e4>
- Establishes: The exact original filename stem `19345061_ext_cotton_20210825_125203` is a literal entry in the named source's own repository.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `per` — PerennialPlants (240 rows)

- Original-ID pattern observed: `DSCF8389` (240 distinct original IDs)
- Card retained count: 240 (matches the audited row count)
- Checked: The official Kaggle file listing for `benediktgeisler/perrenial-plants-detection` contains the literal rows `raw_images/raw_images/DSCF8400.JPG` (13,706,901 B) and `raw_images/raw_images/DSCF8401.JPG` (13,849,520 B).
- URL: <https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection>
  - source index: <https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection>
- Establishes: The exact original filename stems `DSCF8400` and `DSCF8401` are literal entries in the named source's own file listing, with matching byte sizes.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `rl` — RumexLeaves (809 rows)

- Original-ID pattern observed: `10105` (809 distinct original IDs)
- Card retained count: 809 (matches the audited row count)
- Checked: DTU figshare record 23659524 (`RumexLeaves.zip`, doi 10.11583/DTU.23659524.v2) contains the literal entries `RumexLeaves/iNaturalist/10105.jpg`, `RumexLeaves/iNaturalist/10122.jpg` and `RumexLeaves/iNaturalist/10296.jpg`. Structure is iNaturalist (698) + RoboRumex (308) = 809, matching the card.
- URL: <https://api.figshare.com/v2/articles/23659524>
  - source index: <https://api.figshare.com/v2/articles/23659524>
  - source index: <https://doi.org/10.11583/DTU.23659524.v2>
- Establishes: The exact original filename stems `10105` and `10122` are literal entries in the named source's own archive.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `rn` — Ronin (135 rows)

- Original-ID pattern observed: `12114` (135 distinct original IDs)
- Card retained count: 135 (matches the audited row count)
- Checked: Mendeley record 10.17632/nj4vtk4tt6.1 "Data for: Dataset of annotated food crops and weed images for robotic computer vision control" (Ronin, Sudars et al.) publishes `Ronin_OPEN_DB.zip`, whose central directory contains `Ronin_OPEN_DB/raw images/12202.jpg`, `12229.jpg` and `32160.jpg`, each with a matching `annotations/*.xml`.
- URL: <https://data.mendeley.com/public-api/datasets/nj4vtk4tt6>
  - source index: <https://data.mendeley.com/public-api/datasets/nj4vtk4tt6>
  - source index: <https://data.mendeley.com/public-files/datasets/nj4vtk4tt6/files/0d3e34be-9f29-4d8d-bd4c-0915ec623847/file_downloaded>
- Establishes: The exact original filename stems `12202` and `12229` are literal entries in the named source's own archive.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `saw` — SesameWeed (1,300 rows)

- Original-ID pattern observed: `agri_0_1009` (1300 distinct original IDs)
- Card retained count: 1,300 (matches the audited row count)
- Checked: The official Kaggle file listing for `ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes` (Sesame&Weed) contains the literal rows `agri_data/data/agri_0_1009.jpeg` (+ matching `.txt`) and `agri_data/data/agri_0_1017.jpeg` (+ `.txt`).
- URL: <https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes>
  - source index: <https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes>
- Establishes: The exact original filename stem `agri_0_1009` is a literal entry in the named source's own file listing.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).
- Note: The Mendeley DOI 10.17632/6xj66pvdp8 suggested as a candidate for SesameWeed does NOT exist (DataCite/Mendeley 404). Kaggle is the real confirming index.

#### `vcd` — VCD (2,258 rows)

- Original-ID pattern observed: `2019-05-23_montoldre_bean_1_000000_jpg.rf.1243bff29eccd3bacbd5994f897a3cbe` (2258 distinct original IDs)
- Card retained count: 2,258 (matches the audited row count)
- Checked: Mendeley record 10.17632/d7kbzjr83k.1 "Vegetable Crops Dataset for Proximal Sensing (VCD)" publishes `DB.zip`, whose central directory contains `DB_Mendeley/dataset/2019-05-23_montoldre_bean_1_000000.jpg` plus matching `.xml`/`.json` sidecars. The token `montoldre` occurs 1681x.
- URL: <https://data.mendeley.com/public-api/datasets/d7kbzjr83k>
  - source index: <https://data.mendeley.com/public-api/datasets/d7kbzjr83k>
  - source index: <https://data.mendeley.com/public-files/datasets/d7kbzjr83k/files/e87c7db7-0537-4c2e-b020-e1c1101cbafb/file_downloaded>
- Establishes: The exact original filename stem `2019-05-23_montoldre_bean_1_000000` is a literal entry in the official VCD archive. The observed `.rf.<hex>` suffix is a Roboflow export artifact layered on that stem.
- Does not establish: Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).

#### `weds` — WE3DS (1,553 rows)

- Original-ID pattern observed: `img_00000` (1553 distinct original IDs)
- Card retained count: 1,553 (matches the audited row count)
- Checked: Zenodo record 7457983 publishes `WE3DS.zip`, whose central directory contains `images/img_00100.png` and `images/img_00102.png` among 2568 files `img_00000.png`-`img_02567.png`, matching the card retained count.
- URL: <https://doi.org/10.5281/zenodo.7457983>
  - source index: <https://zenodo.org/api/records/7457983>
  - source index: <https://doi.org/10.5281/zenodo.7457983>
  - source index: <https://huggingface.co/datasets/Voxel51/WE3DS>
  - source index: <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10007111/fullTextXML>
- Establishes: The exact original filename stem `img_00100` is a literal entry in the named source's own archive.
- Does not establish: The direct archive read could not be re-verified from this host (Zenodo 403). Independent corroboration came from a DERIVED re-host (HuggingFace `Voxel51/WE3DS` `samples.json`, which lists `data/img_00100.png`), plus the official MDPI paper (PMC10007111) confirming 2568 RGB-D images. Weaker provenance than the other prefixes; treat as confirmed but flag it.
- **Evidence: `original_source_index`** — the observed original ID appears verbatim in the named source's own published index. Adoptable by a curator, but the freeze keeps it `pending_audit` until then (see reports/data_freeze.md).
- Reproduction caveat: Zenodo began returning HTTP 403 ("unusual traffic from your IP") for this host partway through the session, so the direct read of the official archive could not be independently repeated here.

## Confirmed sources (unchanged from Stage 1)

| Source | Train include | Val include | License |
| --- | ---: | ---: | --- |
| ACRECropWeed | 671 | 120 | CC-BY-4.0 |
| LUCASVision | 8,944 | 2,245 | CC-BY-4.0 |
| RadishWheat | 427 | 107 | CC-BY-4.0 |
| SorghumWeed | 137 | 35 | CC-BY-4.0 |
| VegAnn | 1,245 | 360 | CC-BY-1.0 |
| iNatWeeds | 1,034 | 278 | per_image_metadata |

## Web access during this audit

The prefix evidence above was gathered with `open-websearch` plus direct `curl` and archive-index reads. Search engines were unreliable from this host: `duckduckgo` worked intermittently, `bing` is region-locked and returned unrelated Chinese-language results for quoted filename queries, `baidu` returned `302 engine_error`, `brave` was `429` rate-limited, `sogou` served an anti-bot page, `startpage` failed token extraction, and `exa` returned zero rows. Where a search engine failed, the evidence above came from the source dataset's own API or archive central directory (fetched by HTTP Range where the archive was large), which is a stronger index than a search hit.

Host-level access notes: `zenodo.org` began returning `403` mid-session (this affects the `weds` re-check, noted above); `grassclover.multicrop.dk` failed the TLS handshake, so the GrassClover evidence uses the Aalborg AILab re-host; `mdpi.com` and `sciencedirect.com` returned `403`; `api.github.com` was rate limited (raw.githubusercontent.com was not).

## Consequence

14,743 images (36.89% of the corpus) remain quarantined. They are present in `data/exclusion_ledger.csv` with reason `unconfirmed_source_mapping` and are absent from both include manifests and from the frozen split.

This quarantine is now a *policy* choice rather than an evidence gap for 9 of the 10 prefixes: the evidence to admit them exists and is recorded above. Promoting them would raise the frozen corpus from 15,603 to ~30,000 images and would require a new data version, re-screening of the affected duplicate candidates, and a fresh freeze. That is a curation decision, so the default remains the conservative `agrifield40k-v0` freeze.

