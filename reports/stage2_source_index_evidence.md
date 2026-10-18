# Stage 2 original-source-index evidence (prefix provenance)

This file records, for each AgriField-40K short prefix that the publisher card leaves unmapped, whether the observed original filename appears verbatim in the named source dataset's **own published file index**. It is evidence only. The frozen split is unchanged: every prefix below stays `pending_audit` until a curator promotes it (see `reports/data_freeze.md`).

Naming context: the card states AgriField renames images to `[dataset_source]_[id]`. The audit observes the `[id]` part and asks whether that exact string exists in the named upstream source.

## Summary — 9 of 10 prefixes adoptable, 1 supporting-only

| Prefix | Source | Adoptable rows | Evidence class | Index read |
| --- | --- | ---: | --- | --- |
| `gc` | GrassClover | 435 | literal match | <https://theailab.dk/grassclover-dataset/> |
| `mc` | MuST-C | 7,242 | components only | <https://doi.org/10.60507/FK2/OX9XTM> |
| `mw` | Maize-Weed | 255 | literal match | <https://data.mendeley.com/public-api/datasets/jjbfcckrsp> |
| `pags` | PalmerAmaranth | 516 | literal match | <https://github.com/geezacoleman/Palmer-detection> |
| `per` | PerennialPlants | 240 | literal match | <https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection> |
| `rl` | RumexLeaves | 809 | literal match | <https://api.figshare.com/v2/articles/23659524> |
| `rn` | Ronin | 135 | literal match | <https://data.mendeley.com/public-api/datasets/nj4vtk4tt6> |
| `saw` | SesameWeed | 1,300 | literal match | <https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes> |
| `vcd` | VCD | 2,258 | literal match | <https://data.mendeley.com/public-api/datasets/d7kbzjr83k> |
| `weds` | WE3DS | 1,553 | literal match | <https://doi.org/10.5281/zenodo.7457983> |

## Per-prefix evidence and caveats

### `gc` — GrassClover

The official GrassClover biomass archive (`biomass_data.zip`, download_id 4771, published by the Aalborg University AILab) contains the literal entries `test/images/biomass_image_test_0100.jpg` and `test/images/biomass_image_test_0101.jpg`, and its metadata CSV `test/biomass_test_data.csv` literally lists `biomass_image_test_0100.jpg`. The archive holds 261 train + 174 test = 435 images, matching the card.

- **Establishes:** The exact original filename stem `biomass_image_test_0100` is a literal entry in the named source's own archive AND in its own metadata table, and the image count equals the card retained count.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://theailab.dk/grassclover-dataset/>, <https://theailab.dk/grassclover-dataset/>, <https://theailab.dk/?sdm_process_download=1&download_id=4771>, <https://vision.eng.au.dk/grass-clover-dataset/>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `mc` — MuST-C

BonnData record 10.60507/FK2/OX9XTM (MuST-C, Chong et al., Scientific Data) publishes an image archive whose central directory contains the nested paths `images/UGV-RGB/230601/plot162/pos01/nikon_0.jpeg`, with `pos00`-`pos03`, dates `230601/230613/230616`, and `nikon_0`-`nikon_19`.

- **Establishes:** Every component of the observed original ID (plot number, date token, position, camera index) corresponds to a real path component of the official MuST-C archive, and the archive count is consistent with the card retained count.
- **Does not establish:** The official MuST-C naming convention is a NESTED PATH (`<date>/plotNNN/posNN/nikon_N.jpeg`), not the flattened string AgriField stores (`plot162_day230601_pos01_nikon_0`). The `day` token has no official counterpart. So the mapping is strongly supported but NOT a literal filename match.
- **Index reads:** <https://doi.org/10.60507/FK2/OX9XTM>, <https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/OX9XTM>, <https://bonndata.uni-bonn.de/api/access/datafile/8066>, <https://github.com/PRBonn/MuST-C>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `mw` — Maize-Weed

Mendeley record 10.17632/jjbfcckrsp "Maize-Weed Image Dataset" publishes `Maize-Weed Image Dataset.rar`, whose listing contains `Maize-Weed Image Dataset/Dry Season Maize Weed Images/C00020826_output.jpeg` and `.../C00020908_output.jpeg`. The v2 file API independently lists the same `C0002xxxx` namespace.

- **Establishes:** The exact original filename stem `C00020826_output` is a literal entry in the named source's own archive.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://data.mendeley.com/public-api/datasets/jjbfcckrsp>, <https://data.mendeley.com/public-api/datasets/jjbfcckrsp>, <https://data.mendeley.com/api/datasets/jjbfcckrsp/files>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `pags` — PalmerAmaranth

The PAGS8 author repository `geezacoleman/Palmer-detection` contains `data/xml/19345061_ext_cotton_20210825_125203.xml`, whose `<filename>` element is literally `19345061_ext_cotton_20210825_125203.jpg`. The README ties those labels to the PAGS8 images published via Weed-AI.

- **Establishes:** The exact original filename stem `19345061_ext_cotton_20210825_125203` is a literal entry in the named source's own repository.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://github.com/geezacoleman/Palmer-detection>, <https://raw.githubusercontent.com/geezacoleman/Palmer-detection/main/data/xml/19345061_ext_cotton_20210825_125203.xml>, <https://weed-ai.sydney.edu.au/datasets/5c78d067-8750-4803-9cbe-57df8fae55e4>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `per` — PerennialPlants

The official Kaggle file listing for `benediktgeisler/perrenial-plants-detection` contains the literal rows `raw_images/raw_images/DSCF8400.JPG` (13,706,901 B) and `raw_images/raw_images/DSCF8401.JPG` (13,849,520 B).

- **Establishes:** The exact original filename stems `DSCF8400` and `DSCF8401` are literal entries in the named source's own file listing, with matching byte sizes.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection>, <https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `rl` — RumexLeaves

DTU figshare record 23659524 (`RumexLeaves.zip`, doi 10.11583/DTU.23659524.v2) contains the literal entries `RumexLeaves/iNaturalist/10105.jpg`, `RumexLeaves/iNaturalist/10122.jpg` and `RumexLeaves/iNaturalist/10296.jpg`. Structure is iNaturalist (698) + RoboRumex (308) = 809, matching the card.

- **Establishes:** The exact original filename stems `10105` and `10122` are literal entries in the named source's own archive.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://api.figshare.com/v2/articles/23659524>, <https://api.figshare.com/v2/articles/23659524>, <https://doi.org/10.11583/DTU.23659524.v2>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `rn` — Ronin

Mendeley record 10.17632/nj4vtk4tt6.1 "Data for: Dataset of annotated food crops and weed images for robotic computer vision control" (Ronin, Sudars et al.) publishes `Ronin_OPEN_DB.zip`, whose central directory contains `Ronin_OPEN_DB/raw images/12202.jpg`, `12229.jpg` and `32160.jpg`, each with a matching `annotations/*.xml`.

- **Establishes:** The exact original filename stems `12202` and `12229` are literal entries in the named source's own archive.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://data.mendeley.com/public-api/datasets/nj4vtk4tt6>, <https://data.mendeley.com/public-api/datasets/nj4vtk4tt6>, <https://data.mendeley.com/public-files/datasets/nj4vtk4tt6/files/0d3e34be-9f29-4d8d-bd4c-0915ec623847/file_downloaded>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `saw` — SesameWeed

The official Kaggle file listing for `ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes` (Sesame&Weed) contains the literal rows `agri_data/data/agri_0_1009.jpeg` (+ matching `.txt`) and `agri_data/data/agri_0_1017.jpeg` (+ `.txt`).

- **Establishes:** The exact original filename stem `agri_0_1009` is a literal entry in the named source's own file listing.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Note:** The Mendeley DOI 10.17632/6xj66pvdp8 suggested as a candidate for SesameWeed does NOT exist (DataCite/Mendeley 404). Kaggle is the real confirming index.
- **Index reads:** <https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes>, <https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `vcd` — VCD

Mendeley record 10.17632/d7kbzjr83k.1 "Vegetable Crops Dataset for Proximal Sensing (VCD)" publishes `DB.zip`, whose central directory contains `DB_Mendeley/dataset/2019-05-23_montoldre_bean_1_000000.jpg` plus matching `.xml`/`.json` sidecars. The token `montoldre` occurs 1681x.

- **Establishes:** The exact original filename stem `2019-05-23_montoldre_bean_1_000000` is a literal entry in the official VCD archive. The observed `.rf.<hex>` suffix is a Roboflow export artifact layered on that stem.
- **Does not establish:** Nothing further is needed for provenance; this is a literal filename match against the named source's own index.
- **Index reads:** <https://data.mendeley.com/public-api/datasets/d7kbzjr83k>, <https://data.mendeley.com/public-api/datasets/d7kbzjr83k>, <https://data.mendeley.com/public-files/datasets/d7kbzjr83k/files/e87c7db7-0537-4c2e-b020-e1c1101cbafb/file_downloaded>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

### `weds` — WE3DS

Zenodo record 7457983 publishes `WE3DS.zip`, whose central directory contains `images/img_00100.png` and `images/img_00102.png` among 2568 files `img_00000.png`-`img_02567.png`, matching the card retained count.

- **Establishes:** The exact original filename stem `img_00100` is a literal entry in the named source's own archive.
- **Does not establish:** The direct archive read could not be re-verified from this host (Zenodo 403). Independent corroboration came from a DERIVED re-host (HuggingFace `Voxel51/WE3DS` `samples.json`, which lists `data/img_00100.png`), plus the official MDPI paper (PMC10007111) confirming 2568 RGB-D images. Weaker provenance than the other prefixes; treat as confirmed but flag it.
- **Reproduction caveat:** Zenodo began returning HTTP 403 ("unusual traffic from your IP") for this host partway through the session, so the direct read of the official archive could not be independently repeated here.
- **Index reads:** <https://doi.org/10.5281/zenodo.7457983>, <https://zenodo.org/api/records/7457983>, <https://doi.org/10.5281/zenodo.7457983>, <https://huggingface.co/datasets/Voxel51/WE3DS>, <https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10007111/fullTextXML>
- **Adoption:** `not_adopted` — rows remain `pending_audit` in `agrifield40k-v0`.

## Promotion path (curator action, not automated)

Admitting the evidence-backed prefixes is a data-version bump, not a flag flip. The required steps, in order:

1. Curator records an explicit `prefix -> source` decision for each prefix in the prefix map, citing the index read above.
2. Re-run the source/license audit so the affected rows move from `pending_audit` to `confirmed` with a license attached.
3. Re-run duplicate screening over the newly admitted rows — their near-duplicates are currently unscreened because the rows were never in an include manifest.
4. Regenerate the include manifests and re-freeze under a new data version (`agrifield40k-v1` or later), with fresh SHA-256 digests.
5. Record the change and its evidence in the Stage 3 baseline protocol before any training run uses the new split, so the Stage 3/4 comparison basis stays explicit.

Promoting all 9 adoptable prefixes plus `mc` would raise the frozen corpus from 15,603 to roughly 30,000 images. Until that is done deliberately, the conservative `agrifield40k-v0` freeze stands.

