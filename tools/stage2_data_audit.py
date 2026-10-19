"""Stage 2B data audit: duplicate-review materials, distribution analysis, data freeze.

Stage 1 (`tools/audit_data.py`) audited the real pixels, quarantined uncertain
origins, and emitted the include-only manifests. This script is read-only with
respect to every Stage 1 artefact: it never rewrites `data/all_samples.csv`,
`data/exclusion_ledger.csv`, `data/duplicate_pairs.csv`, `data/source_registry.json`,
`pretrain_manifest.csv`, `val_manifest.csv`, or `src/`. It only adds new files.

What it produces
----------------
1. `reports/stage2_data_audit.md`      human-readable source/license state for the
                                       unresolved prefixes, with the web evidence
                                       that was and was not found.
2. `reports/stage2_data_audit.json`    machine-readable summary of the same.
3. `data/duplicate_review.csv`        one row per candidate edge from
                                       `data/duplicate_pairs.csv`, enriched with
                                       split/source/manifest context and a
                                       review column that starts at
                                       `pending_manual_review` for every row.
4. `reports/duplicate_review/`        offline HTML contact sheet + thumbnails so a
                                       human can actually adjudicate the pairs.
5. `reports/duplicate_review_protocol.md`
6. `reports/data_distribution.md`     per-source manifest distribution, LUCASVision
                                       share, concrete cap schemes, dominance risk.
7. `data/frozen/*`                    versioned frozen manifests + freeze manifest
                                       with SHA-256 of every frozen file.
8. `reports/data_freeze.md`           the freeze record.

Determinism
-----------
Every output is derived only from the pinned Stage 1 CSV/JSON inputs, the fixed
web-evidence constant below, and the image bytes on disk. No randomness, no
network access, no clock reads except the `--generated-at` override that is
written verbatim into the freeze record. Re-running with the same inputs
reproduces byte-identical outputs.

The one input that is NOT reproducible from disk is the web evidence log
(`WEB_EVIDENCE`). It records what was fetched on the date in `EVIDENCE_FETCHED_ON`
and must be re-verified before it is relied on for anything other than reporting
the state of the audit on that date.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

DATA_VERSION = 'agrifield40k-v0'
FREEZE_REVISION = '7f530c67572b6a572467adab840cdbbd7029b112'
CARD_URL = f'https://huggingface.co/datasets/dtu-pcas/AgriField-40K/raw/{FREEZE_REVISION}/README.md'
DATASET_PAGE = 'https://huggingface.co/datasets/dtu-pcas/AgriField-40K'
PROJECT_REPO = 'https://github.com/DTU-PAS/agrifield40k'
PROJECT_PAGE = 'https://dtu-pas.github.io/agrifield40k/'
PAPER_URL = 'https://arxiv.org/abs/2608.07984'
DEFAULT_IMAGE_ROOT = Path('/data/datasets/AgriField-40K')

# Stage 1 fields (data/all_samples.csv) reused verbatim; do not re-derive.
SAMPLE_FIELDS = [
    'sample_id', 'path', 'source_prefix', 'source', 'candidate_source', 'source_confirmed',
    'original_id', 'split', 'original_split', 'width', 'height', 'mode', 'format', 'bytes',
    'sha256', 'pixel_sha256', 'phash64', 'dhash64', 'decode_ok', 'license',
    'compilation_license', 'source_url', 'source_revision', 'attribution',
    'observation_url', 'status', 'reason', 'component',
]
PAIR_FIELDS = ['sample_a', 'sample_b', 'kind', 'phash_distance', 'dhash_distance', 'cross_split']
REVIEW_FIELDS = [
    'edge_id', 'sample_a', 'sample_b', 'kind', 'phash_distance', 'dhash_distance',
    'cross_split', 'split_a', 'split_b', 'source_a', 'source_b', 'path_a', 'path_b',
    'in_train_manifest', 'in_val_manifest', 'review_status', 'reviewer', 'review_notes',
    'reviewed_at', 'isolation_category', 'thumb_a', 'thumb_b',
]

PENDING_REVIEW = 'pending_manual_review'

# ---------------------------------------------------------------------------
# Web evidence, fixed for reproducibility.
#
# Each entry records a claim, the URL it rests on, and what the URL actually
# does and does not establish. `evidence_class` is deliberately strict:
#   publisher_card   -- the AgriField-40K publisher's own statement
#   publisher_paper  -- the AgriField-40K publisher's own paper
#   original_source_index -- the ORIGINAL source dataset's own published file
#                       index / archive listing literally contains the observed
#                       original filename. This is the strong evidence class the
#                       brief calls "official file IDs and mapping rules".
#   no_evidence      -- checked, nothing found; the prefix stays pending_audit
#
# IMPORTANT: `evidence_class` is a RECORD, not a decision. The freeze is driven
# by the `source` / `status` columns of the Stage 1 artefacts, so annotating an
# entry here can never silently admit a row. Rows whose evidence is
# `original_source_index` still carry `status=pending_audit` and remain
# quarantined until a curator opts in. See the `adoption` field on each entry.
# ---------------------------------------------------------------------------
WEB_EVIDENCE = [
    {
        'prefix': None,
        'evidence_class': 'publisher_card',
        'url': CARD_URL,
        'claim': 'AgriField-40K contains 39,963 images from 17 source datasets and renames them '
                 '[dataset_source]_[id]; the compilation is CC BY-SA 4.0 and each image stays '
                 'under its source license.',
        'establishes': 'The naming convention, the 17-source list, and each source license and '
                       'retained count.',
        'does_not_establish': 'It does not publish the short-prefix to source map. It never '
                              'states that "mc" means MuST-C.',
    },
    {
        'prefix': None,
        'evidence_class': 'publisher_paper',
        'url': PAPER_URL,
        'claim': 'Paper Sec. 3 / Table 1 list all 17 sources with retained counts and licenses; '
                 'the split is 80/20, split at sequence level for sequence-derived sources and '
                 'randomly for independent-image sources.',
        'establishes': 'Retained counts, licenses, and the publisher split rule. It also states '
                       'that only RGB images were kept and annotations discarded.',
        'does_not_establish': 'It contains no prefix map either. The split rule is the '
                              "publisher's stated intent, not something this audit verifies "
                              'per image.',
    },
    {
        'prefix': 'mc',
        'evidence_class': 'supporting_only',
        'adoption': 'not_adopted',
        'url': 'https://doi.org/10.60507/FK2/OX9XTM',
        'claim': 'BonnData record 10.60507/FK2/OX9XTM (MuST-C, Chong et al., Scientific Data) '
                 'publishes an image archive whose central directory contains the nested paths '
                 '`images/UGV-RGB/230601/plot162/pos01/nikon_0.jpeg`, with `pos00`-`pos03`, '
                 'dates `230601/230613/230616`, and `nikon_0`-`nikon_19`.',
        'establishes': 'Every component of the observed original ID (plot number, date token, '
                       'position, camera index) corresponds to a real path component of the '
                       'official MuST-C archive, and the archive count is consistent with the '
                       'card retained count.',
        'does_not_establish': 'The official MuST-C naming convention is a NESTED PATH '
                              '(`<date>/plotNNN/posNN/nikon_N.jpeg`), not the flattened string '
                              'AgriField stores (`plot162_day230601_pos01_nikon_0`). The `day` '
                              'token has no official counterpart. So the mapping is strongly '
                              'supported but NOT a literal filename match.',
        'evidence_urls': [
            'https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/OX9XTM',
            'https://bonndata.uni-bonn.de/api/access/datafile/8066',
            'https://github.com/PRBonn/MuST-C',
        ],
    },
    {
        'prefix': 'vcd',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://data.mendeley.com/public-api/datasets/d7kbzjr83k',
        'claim': 'Mendeley record 10.17632/d7kbzjr83k.1 "Vegetable Crops Dataset for Proximal '
                 'Sensing (VCD)" publishes `DB.zip`, whose central directory contains '
                 '`DB_Mendeley/dataset/2019-05-23_montoldre_bean_1_000000.jpg` plus matching '
                 '`.xml`/`.json` sidecars. The token `montoldre` occurs 1681x.',
        'establishes': 'The exact original filename stem '
                       '`2019-05-23_montoldre_bean_1_000000` is a literal entry in the official '
                       'VCD archive. The observed `.rf.<hex>` suffix is a Roboflow export '
                       'artifact layered on that stem.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://data.mendeley.com/public-api/datasets/d7kbzjr83k',
            'https://data.mendeley.com/public-files/datasets/d7kbzjr83k/files/e87c7db7-0537-4c2e-b020-e1c1101cbafb/file_downloaded',
        ],
    },
    {
        'prefix': 'pags',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://github.com/geezacoleman/Palmer-detection',
        'claim': 'The PAGS8 author repository `geezacoleman/Palmer-detection` contains '
                 '`data/xml/19345061_ext_cotton_20210825_125203.xml`, whose `<filename>` element '
                 'is literally `19345061_ext_cotton_20210825_125203.jpg`. The README ties those '
                 'labels to the PAGS8 images published via Weed-AI.',
        'establishes': 'The exact original filename stem '
                       '`19345061_ext_cotton_20210825_125203` is a literal entry in the '
                       'named source\'s own repository.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://raw.githubusercontent.com/geezacoleman/Palmer-detection/main/data/xml/19345061_ext_cotton_20210825_125203.xml',
            'https://weed-ai.sydney.edu.au/datasets/5c78d067-8750-4803-9cbe-57df8fae55e4',
        ],
    },
    {
        'prefix': 'gc',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://theailab.dk/grassclover-dataset/',
        'claim': 'The official GrassClover biomass archive (`biomass_data.zip`, download_id 4771, '
                 'published by the Aalborg University AILab) contains the literal entries '
                 '`test/images/biomass_image_test_0100.jpg` and '
                 '`test/images/biomass_image_test_0101.jpg`, and its metadata CSV '
                 '`test/biomass_test_data.csv` literally lists `biomass_image_test_0100.jpg`. '
                 'The archive holds 261 train + 174 test = 435 images, matching the card.',
        'establishes': 'The exact original filename stem `biomass_image_test_0100` is a literal '
                       'entry in the named source\'s own archive AND in its own metadata table, '
                       'and the image count equals the card retained count.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://theailab.dk/grassclover-dataset/',
            'https://theailab.dk/?sdm_process_download=1&download_id=4771',
            'https://vision.eng.au.dk/grass-clover-dataset/',
        ],
    },
    {
        'prefix': 'rn',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://data.mendeley.com/public-api/datasets/nj4vtk4tt6',
        'claim': 'Mendeley record 10.17632/nj4vtk4tt6.1 "Data for: Dataset of annotated food '
                 'crops and weed images for robotic computer vision control" (Ronin, Sudars et '
                 'al.) publishes `Ronin_OPEN_DB.zip`, whose central directory contains '
                 '`Ronin_OPEN_DB/raw images/12202.jpg`, `12229.jpg` and `32160.jpg`, each with a '
                 'matching `annotations/*.xml`.',
        'establishes': 'The exact original filename stems `12202` and `12229` are literal entries '
                       'in the named source\'s own archive.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://data.mendeley.com/public-api/datasets/nj4vtk4tt6',
            'https://data.mendeley.com/public-files/datasets/nj4vtk4tt6/files/0d3e34be-9f29-4d8d-bd4c-0915ec623847/file_downloaded',
        ],
    },
    {
        'prefix': 'weds',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'reproduction_caveat': 'Zenodo began returning HTTP 403 ("unusual traffic from your IP") '
                               'for this host partway through the session, so the direct read of '
                               'the official archive could not be independently repeated here.',
        'url': 'https://doi.org/10.5281/zenodo.7457983',
        'claim': 'Zenodo record 7457983 publishes `WE3DS.zip`, whose central directory contains '
                 '`images/img_00100.png` and `images/img_00102.png` among 2568 files '
                 '`img_00000.png`-`img_02567.png`, matching the card retained count.',
        'establishes': 'The exact original filename stem `img_00100` is a literal entry in the '
                       'named source\'s own archive.',
        'does_not_establish': 'The direct archive read could not be re-verified from this host '
                              '(Zenodo 403). Independent corroboration came from a DERIVED '
                              're-host (HuggingFace `Voxel51/WE3DS` `samples.json`, which lists '
                              '`data/img_00100.png`), plus the official MDPI paper (PMC10007111) '
                              'confirming 2568 RGB-D images. Weaker provenance than the other '
                              'prefixes; treat as confirmed but flag it.',
        'evidence_urls': [
            'https://zenodo.org/api/records/7457983',
            'https://doi.org/10.5281/zenodo.7457983',
            'https://huggingface.co/datasets/Voxel51/WE3DS',
            'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10007111/fullTextXML',
        ],
    },
    {
        'prefix': 'mw',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://data.mendeley.com/public-api/datasets/jjbfcckrsp',
        'claim': 'Mendeley record 10.17632/jjbfcckrsp "Maize-Weed Image Dataset" publishes '
                 '`Maize-Weed Image Dataset.rar`, whose listing contains '
                 '`Maize-Weed Image Dataset/Dry Season Maize Weed Images/C00020826_output.jpeg` '
                 'and `.../C00020908_output.jpeg`. The v2 file API independently lists the same '
                 '`C0002xxxx` namespace.',
        'establishes': 'The exact original filename stem `C00020826_output` is a literal entry in '
                       'the named source\'s own archive.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://data.mendeley.com/public-api/datasets/jjbfcckrsp',
            'https://data.mendeley.com/api/datasets/jjbfcckrsp/files',
        ],
    },
    {
        'prefix': 'rl',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://api.figshare.com/v2/articles/23659524',
        'claim': 'DTU figshare record 23659524 (`RumexLeaves.zip`, doi 10.11583/DTU.23659524.v2) '
                 'contains the literal entries `RumexLeaves/iNaturalist/10105.jpg`, '
                 '`RumexLeaves/iNaturalist/10122.jpg` and `RumexLeaves/iNaturalist/10296.jpg`. '
                 'Structure is iNaturalist (698) + RoboRumex (308) = 809, matching the card.',
        'establishes': 'The exact original filename stems `10105` and `10122` are literal entries '
                       'in the named source\'s own archive.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://api.figshare.com/v2/articles/23659524',
            'https://doi.org/10.11583/DTU.23659524.v2',
        ],
    },
    {
        'prefix': 'saw',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'dead_doi_note': 'The Mendeley DOI 10.17632/6xj66pvdp8 suggested as a candidate for '
                         'SesameWeed does NOT exist (DataCite/Mendeley 404). Kaggle is the '
                         'real confirming index.',
        'url': 'https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
        'claim': 'The official Kaggle file listing for '
                 '`ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes` (Sesame&Weed) '
                 'contains the literal rows `agri_data/data/agri_0_1009.jpeg` (+ matching '
                 '`.txt`) and `agri_data/data/agri_0_1017.jpeg` (+ `.txt`).',
        'establishes': 'The exact original filename stem `agri_0_1009` is a literal entry in the '
                       'named source\'s own file listing.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
        ],
    },
    {
        'prefix': 'per',
        'evidence_class': 'original_source_index',
        'adoption': 'not_adopted',
        'url': 'https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection',
        'claim': 'The official Kaggle file listing for '
                 '`benediktgeisler/perrenial-plants-detection` contains the literal rows '
                 '`raw_images/raw_images/DSCF8400.JPG` (13,706,901 B) and '
                 '`raw_images/raw_images/DSCF8401.JPG` (13,849,520 B).',
        'establishes': 'The exact original filename stems `DSCF8400` and `DSCF8401` are literal '
                       'entries in the named source\'s own file listing, with matching byte '
                       'sizes.',
        'does_not_establish': 'Nothing further is needed for provenance; this is a literal '
                              'filename match against the named source\'s own index.',
        'evidence_urls': [
            'https://www.kaggle.com/datasets/benediktgeisler/perrenial-plants-detection',
        ],
    },
]

# Licenses recorded for the unresolved prefixes, per the pinned publisher card.
# These are *card-recorded* licenses for the candidate source. They are NOT
# applied to any manifest row, because no row is admitted while its source is
# unconfirmed. They exist so the review can record what would apply if the
# mapping is later confirmed.
CANDIDATE_LICENSES = {
    'mc': 'CC-BY-4.0', 'vcd': 'CC-BY-4.0', 'pags': 'CC-BY-4.0', 'gc': 'CC-BY-SA-4.0',
    'rn': 'CC-BY-4.0', 'weds': 'CC-BY-4.0', 'mw': 'CC-BY-4.0', 'rl': 'CC-BY-4.0',
    'saw': 'CC0-1.0', 'per': 'MIT',
}

# Freeze provenance for confirmed sources (from the Stage 1 registry + card).
CONFIRMED_LICENSES = {
    'ACRECropWeed': 'CC-BY-4.0', 'LUCASVision': 'CC-BY-4.0', 'RadishWheat': 'CC-BY-4.0',
    'SorghumWeed': 'CC-BY-4.0', 'VegAnn': 'CC-BY-1.0', 'iNatWeeds': 'per_image_metadata',
}


def sha256_file(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\r\n')
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fmt_int(value) -> str:
    return f'{int(value):,}'


def fmt_pct(part, whole, digits=2) -> str:
    return f'{100.0 * part / whole:.{digits}f}%' if whole else 'n/a'


# ---------------------------------------------------------------------------
# contact-sheet thumbnails
# ---------------------------------------------------------------------------

def make_thumbnail(source: Path, destination: Path, width: int) -> bool:
    """Centre-crop square, resize to `width`, save as JPEG. Returns success."""
    try:
        with Image.open(source) as im:
            im.load()
            rgb = im.convert('RGB')
            side = min(rgb.width, rgb.height)
            left = (rgb.width - side) // 2
            top = (rgb.height - side) // 2
            rgb = rgb.crop((left, top, left + side, top + side))
            rgb = rgb.resize((width, width), Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(destination, format='JPEG', quality=82, optimize=True)
        return True
    except Exception:
        return False


def thumb_name(sample_id: str, suffix: str) -> str:
    # Sample IDs are at most 76 chars; prefix with the counter to stay unique
    # and to keep filenames short and filesystem-safe.
    digest = hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:16]
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in sample_id)[:40]
    return f'{digest}{suffix}.jpg'


def generate_thumbs(review_rows, samples, image_root: Path, out_dir: Path, thumb_width: int,
                    enabled: bool):
    """Write a thumbnail per edge endpoint and record its filename on the row.

    Returns (written, failed). The filename is deterministic, so the HTML writer
    can be run independently afterwards.
    """
    thumbs_dir = out_dir / 'thumbs'
    ok = failed = 0
    for row in review_rows:
        for sample_id, suffix, field in ((row['sample_a'], '_a', 'thumb_a'),
                                         (row['sample_b'], '_b', 'thumb_b')):
            name = thumb_name(sample_id, suffix)
            if not enabled:
                row[field] = ''
                continue
            rel_path = samples[sample_id]['path']
            if make_thumbnail(image_root / rel_path, thumbs_dir / name, thumb_width):
                row[field] = name
                ok += 1
            else:
                row[field] = ''
                failed += 1
        if not enabled:
            continue
    if failed:
        print(f'  {failed} thumbnail(s) could not be generated', flush=True)
    return ok, failed


def build_contact_sheet(review_rows, samples, out_dir: Path, thumb_width: int,
                        thumbs_written: int, thumbs_failed: int):
    """Write the single offline index.html contact sheet."""

    def category_of(row):
        if row['cross_split'] == 'True':
            return 'cross_split'
        if row['isolation_category'] == 'manifest_affecting':
            return 'manifest_affecting'
        return 'same_split_screening'

    groups = defaultdict(list)
    for row in review_rows:
        groups[category_of(row)].append(row)

    order = ['cross_split', 'manifest_affecting', 'same_split_screening']
    titles = {
        'cross_split': 'A. Cross-split candidates (train <-> val)',
        'manifest_affecting': 'B. Manifest-affecting candidates (one endpoint in an include manifest)',
        'same_split_screening': 'C. Same-split screening candidates',
    }
    descriptions = {
        'cross_split': 'These pairs straddle the original train/validation split. If they are '
                       'real duplicates they are a direct leakage path between pretraining and '
                       'validation. Review these first.',
        'manifest_affecting': 'One endpoint is in an include manifest and the other is not. If the '
                              'pair is a real duplicate, the excluded endpoint is confirmed as '
                              'redundant; if it is not, the excluded row may have been dropped '
                              'unnecessarily. Review these second.',
        'same_split_screening': 'Both endpoints are outside the include manifests, so nothing in '
                                'the frozen manifests changes either way. Lowest priority.',
    }

    parts = []
    parts.append('<!DOCTYPE html>')
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append('<title>AgriField-40K duplicate review pack</title>')
    parts.append('<style>')
    parts.append(
        'body{font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;'
        'margin:0;padding:24px;background:#f6f6f4;color:#1a1a1a}'
        'h1{font-size:22px;margin:0 0 4px}'
        'h2{font-size:17px;margin:32px 0 4px;padding-top:12px;border-top:2px solid #c9c9c4}'
        '.sub{color:#555;margin:0 0 4px;max-width:1100px}'
        '.grp{color:#333;font-size:13px;margin:0 0 16px;max-width:1100px}'
        '.count{font-weight:600;color:#111}'
        '.nav{position:sticky;top:0;background:#f6f6f4;padding:8px 0 10px;z-index:5;'
        'border-bottom:1px solid #d5d5d0;margin-bottom:8px}'
        '.nav a{margin-right:16px;color:#0b5fa5;text-decoration:none;font-weight:600}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}'
        '.card{border:1px solid #d5d5d0;border-radius:6px;background:#fff;padding:10px}'
        '.card h3{font-size:12px;margin:0 0 8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'word-break:break-all;color:#444}'
        '.pair{display:flex;gap:10px;align-items:flex-start}'
        '.pair figure{margin:0;text-align:center}'
        '.pair img{width:160px;height:160px;object-fit:cover;border:1px solid #bbb;display:block;'
        'background:#eee}'
        '.pair figcaption{font-size:11px;color:#666;margin-top:3px;word-break:break-all;'
        'max-width:170px}'
        '.vs{align-self:center;font-weight:700;color:#999;padding:0 4px}'
        'table.meta{border-collapse:collapse;width:100%;margin-top:8px;font-size:11px}'
        'table.meta th,table.meta td{border:1px solid #e0e0dc;padding:2px 5px;text-align:left}'
        'table.meta th{background:#fafaf8;color:#555;font-weight:600}'
        'table.meta td{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}'
        '.verdict{margin-top:8px;font-size:11px;color:#555}'
        '.verdict label{margin-right:14px;white-space:nowrap}'
        '.verdict input[type=text]{width:220px;font:inherit;font-size:11px}'
        '.tag{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;font-weight:700;'
        'letter-spacing:.02em}'
        '.tag.cross{background:#fde2e2;color:#8a1c1c}'
        '.tag.man{background:#fdf0d5;color:#7a5310}'
        '.tag.same{background:#e6ecf5;color:#1d4470}'
        '.note{background:#fffbe8;border:1px solid #e8d99a;padding:10px 12px;border-radius:5px;'
        'max-width:1100px;font-size:12px;margin:0 0 20px}'
        'code{background:#eef;padding:1px 3px;border-radius:3px}'
        '.export{max-width:1100px;margin:0 0 20px}'
        '.export button{font:inherit;font-weight:600;padding:6px 12px;border:1px solid #999;'
        'border-radius:5px;background:#fff;cursor:pointer}'
        '.export textarea{display:block;width:100%;height:120px;margin-top:8px;'
        'font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}'
    )
    parts.append('</style></head><body>')
    parts.append('<h1>AgriField-40K duplicate review pack</h1>')
    parts.append(f'<p class="sub">{esc(fmt_int(len(review_rows)))} candidate edges from '
                 f'<code>data/duplicate_pairs.csv</code>, covering '
                 f'{esc(fmt_int(len({r["sample_a"] for r in review_rows} | {r["sample_b"] for r in review_rows})))} '
                 'distinct images. All 512x512 images were centre-cropped square and resized to '
                 f'{thumb_width}px for this sheet ({thumbs_written} thumbnails written, '
                 f'{thumbs_failed} failed). Works offline.</p>')
    parts.append('<p class="sub"><strong>Every row starts at '
                 '<code>pending_manual_review</code>.</strong> Nothing in this pack is a '
                 'confirmed duplicate. A hash distance is a screening signal, not proof. '
                 'Tick a verdict per row and write it back to '
                 '<code>data/duplicate_review.csv</code> (see '
                 '<code>reports/duplicate_review_protocol.md</code>).</p>')
    parts.append('<div class="nav">')
    for key in order:
        parts.append(f'<a href="#sec-{key}">{esc(titles[key])} '
                     f'<span class="count">({len(groups.get(key, []))})</span></a>')
    parts.append('</div>')
    parts.append('<div class="note">Exact byte-identical pairs (<code>kind=sha256</code>) are '
                 'certain duplicates. <code>kind=perceptual_candidate</code> means pHash Hamming '
                 '&le;4 <em>and</em> dHash Hamming &le;6, which is a two-screener rule chosen to '
                 'reduce false hits on visually uniform field scenes. It still produces false '
                 'positives, which is exactly what this review exists to resolve.</div>')
    parts.append('<div class="export">'
                 '<button type="button" onclick="afExport()">Collect verdicts as JSON</button> '
                 '<button type="button" onclick="afClear()">Clear</button>'
                 '<textarea id="afOut" readonly placeholder="Tick verdicts below, then click '
                 'Collect verdicts. Copy the JSON out and fold it into '
                 'data/duplicate_review.csv as described in '
                 'reports/duplicate_review_protocol.md."></textarea></div>')

    for key in order:
        rows = groups.get(key, [])
        tag = {'cross_split': 'cross', 'manifest_affecting': 'man', 'same_split_screening': 'same'}[key]
        parts.append(f'<h2 id="sec-{key}">{esc(titles[key])} '
                     f'<span class="count">({len(rows)})</span></h2>')
        parts.append(f'<p class="grp">{esc(descriptions[key])}</p>')
        if not rows:
            parts.append('<p class="grp">None.</p>')
            continue
        parts.append('<div class="grid">')
        for row in rows:
            parts.append(f'<div class="card" data-edge="{esc(row["edge_id"])}">')
            parts.append(f'<h3>{esc(row["edge_id"])}</h3>')
            parts.append('<div class="pair">')
            for sample_id, thumb_key, label in (
                    (row['sample_a'], 'thumb_a', 'A'), (row['sample_b'], 'thumb_b', 'B')):
                name = row[thumb_key]
                parts.append('<figure>')
                if name:
                    parts.append(f'<img src="thumbs/{esc(name)}" alt="{esc(sample_id)}" '
                                 f'loading="lazy" width="160" height="160">')
                else:
                    parts.append('<div style="width:160px;height:160px;border:1px solid #bbb;'
                                 'background:#eee;display:flex;align-items:center;justify-content:'
                                 'center;color:#888;font-size:11px">image unavailable</div>')
                parts.append(f'<figcaption><strong>{label}</strong> '
                             f'{esc(samples[sample_id]["split"])} &middot; '
                             f'{esc(samples[sample_id]["source_prefix"])}<br>'
                             f'{esc(samples[sample_id]["original_id"])}</figcaption>')
                parts.append('</figure>')
                if label == 'A':
                    parts.append('<div class="vs">vs</div>')
            parts.append('</div>')
            parts.append('<table class="meta"><tr>'
                         '<th>kind</th><th>pHash</th><th>dHash</th><th>cross_split</th>'
                         '<th>split</th><th>source</th><th>in train</th><th>in val</th>'
                         f'<th>category</th></tr><tr>'
                         f'<td>{esc(row["kind"])}</td>'
                         f'<td>{esc(row["phash_distance"])}</td>'
                         f'<td>{esc(row["dhash_distance"])}</td>'
                         f'<td>{esc(row["cross_split"])}</td>'
                         f'<td>{esc(row["split_a"])} / {esc(row["split_b"])}</td>'
                         f'<td>{esc(row["source_a"])} / {esc(row["source_b"])}</td>'
                         f'<td>{esc(row["in_train_manifest"])}</td>'
                         f'<td>{esc(row["in_val_manifest"])}</td>'
                         f'<td><span class="tag {tag}">{esc(row["isolation_category"])}</span></td>'
                         '</tr></table>')
            parts.append('<div class="verdict">verdict: '
                         '<label><input type="checkbox" data-v="confirmed_duplicate"> duplicate</label>'
                         '<label><input type="checkbox" data-v="not_duplicate"> not duplicate</label>'
                         '<label><input type="checkbox" data-v="near_duplicate"> near duplicate</label><br>'
                         'reviewer: <input type="text" data-r="reviewer"> &nbsp; '
                         'date: <input type="text" data-r="reviewed_at"> &nbsp; '
                         'notes: <input type="text" data-r="review_notes"></div>')
            parts.append('</div>')
        parts.append('</div>')
    # Inline export helper. Verdicts live only in the browser until exported,
    # which keeps the pack a pure static file with no server and no storage.
    parts.append('<script>')
    parts.append(
        "function afClear(){"
        "document.querySelectorAll('.card input').forEach(function(i){"
        "if(i.type==='checkbox'){i.checked=false}else{i.value=''}});"
        "document.getElementById('afOut').value='';}"
        "function afExport(){"
        "var out=[];"
        "document.querySelectorAll('.card').forEach(function(card){"
        "var v=card.querySelector('input[type=checkbox]:checked');"
        "var reviewer=card.querySelector('[data-r=reviewer]').value.trim();"
        "var reviewed=card.querySelector('[data-r=reviewed_at]').value.trim();"
        "var notes=card.querySelector('[data-r=review_notes]').value.trim();"
        "if(!v&&!reviewer&&!reviewed&&!notes){return;}"
        "out.push({edge_id:card.getAttribute('data-edge'),"
        "review_status:v?v.getAttribute('data-v'):'pending_manual_review',"
        "reviewer:reviewer,reviewed_at:reviewed,review_notes:notes});});"
        "document.getElementById('afOut').value=out.length?JSON.stringify(out,null,1):"
        "'No verdicts entered yet.';}")
    parts.append('</script>')
    parts.append('</body></html>')
    write_text(out_dir / 'index.html', '\n'.join(parts) + '\n')

# ---------------------------------------------------------------------------
# report writers
# ---------------------------------------------------------------------------

def write_source_audit_report(path: Path, json_path: Path, audit, registry):
    pend = audit['pending_by_prefix']
    total_pending = audit['pending_rows']
    moved = audit['pending_rows_confirmed']
    lines = []
    lines.append('# Stage 2B source and license audit')
    lines.append('')
    lines.append(f'Pinned dataset revision: `{FREEZE_REVISION}`. '
                 'Publisher card: <' + CARD_URL + '>.')
    lines.append('')
    lines.append('## Headline')
    lines.append('')
    lines.append(f'- Audited images: **{fmt_int(audit["images"])}**')
    lines.append(f'- Rows `pending_audit` before Stage 2B: **{fmt_int(total_pending)}**')
    lines.append(f'- Rows that moved to `confirmed` in Stage 2B: **{fmt_int(moved)}**')
    lines.append(f'- Rows still `pending_audit`: **{fmt_int(total_pending - moved)}**')
    lines.append('')
    lines.append('**No unresolved prefix was auto-confirmed into the frozen split.** The audit is '
                 'conservative by design: provenance evidence is recorded but the freeze is driven '
                 'by the Stage 1 artefact columns (`status`/`source`), so recording an evidence '
                 'class can never silently admit a row. What the Stage 2B evidence pass found:')
    lines.append('')
    n_idx = sum(1 for p, i in pend.items() if i['evidence_class'] == 'original_source_index')
    n_support = sum(1 for p, i in pend.items() if i['evidence_class'] == 'supporting_only')
    n_none = sum(1 for p, i in pend.items()
                 if i['evidence_class'] in ('no_evidence', 'not_checked'))
    lines.append(f'- **{n_idx} of 10 prefixes** now have a literal filename match against the '
                 f'named source\'s OWN published archive/file index (`evidence_class: '
                 f'original_source_index`): `gc`, `vcd`, `pags`, `rn`, `weds`, `mw`, `rl`, '
                 f'`saw`, `per`. Their observed original IDs appear verbatim in the source dataset '
                 f'the card names.')
    lines.append(f'- **1 prefix** (`mc`/MuST-C) is `supporting_only`: every component of its '
                 f'filename matches the official BonnData archive, but the official naming is a '
                 f'nested path, not the flattened string AgriField stores.')
    lines.append(f'- **{n_none} prefix(es)** have no evidence.')
    lines.append('')
    lines.append('These prefixes remain `pending_audit` in the freeze and stay out of the frozen '
                 'split until a curator opts in by promoting them. The recommendation and the '
                 'mechanics for doing that are in `reports/data_freeze.md`.')
    lines.append('')
    lines.append('## What the publisher materials do establish')
    lines.append('')
    for entry in WEB_EVIDENCE:
        if entry['evidence_class'].startswith('publisher'):
            lines.append(f'- <{entry["url"]}> — {entry["claim"]}')
            lines.append(f'  - Establishes: {entry["establishes"]}')
            lines.append(f'  - Does not establish: {entry["does_not_establish"]}')
    lines.append('')
    lines.append('## Per-prefix review')
    lines.append('')
    lines.append('| Prefix | Candidate source | Rows | Card retained | License if confirmed | '
                 'Evidence found | Decision |')
    lines.append('| --- | --- | ---: | ---: | --- | --- | --- |')
    for prefix, info in sorted(pend.items()):
        cls = info['evidence_class']
        decision = ('pending_audit (adoptable: literal source-index match)'
                    if cls == 'original_source_index'
                    else 'pending_audit (supporting evidence only)' if cls == 'supporting_only'
                    else 'pending_audit')
        lines.append(
            f'| `{prefix}` | {info["candidate_source"] or "-"} | {fmt_int(info["rows"])} | '
            f'{fmt_int(info["expected_count"])} | {CANDIDATE_LICENSES.get(prefix, "-")} | '
            f'{cls} | **{decision}** |')
    lines.append('')
    lines.append('### Detail')
    lines.append('')
    for prefix, info in sorted(pend.items()):
        entry = info['web_entry']
        lines.append(f'#### `{prefix}` — {info["candidate_source"]} ({fmt_int(info["rows"])} rows)')
        lines.append('')
        lines.append(f'- Original-ID pattern observed: `{info["id_example"]}` '
                     f'({info["id_pattern_note"]})')
        examples = info.get('id_examples') or []
        if len(examples) > 1:
            lines.append(f'- Example original IDs: '
                         + ', '.join(f'`{e}`' for e in examples))
        lines.append(f'- Card retained count: {fmt_int(info["expected_count"])} '
                     f'({"matches" if info["count_matches"] else "DOES NOT match"} the audited '
                     f'row count)')
        lines.append(f'- Checked: {entry["claim"]}')
        lines.append(f'- URL: <{entry["url"]}>')
        for u in entry.get('evidence_urls', []):
            lines.append(f'  - source index: <{u}>')
        lines.append(f'- Establishes: {entry["establishes"]}')
        lines.append(f'- Does not establish: {entry["does_not_establish"]}')
        cls = info['evidence_class']
        if cls == 'original_source_index':
            lines.append(f'- **Evidence: `original_source_index`** — the observed original ID '
                         f'appears verbatim in the named source\'s own published index. '
                         f'Adoptable by a curator, but the freeze keeps it `pending_audit` until '
                         f'then (see reports/data_freeze.md).')
        elif cls == 'supporting_only':
            lines.append(f'- **Evidence: `supporting_only`** — strong match on components, but '
                         f'not a literal filename match; stays `pending_audit`.')
        else:
            lines.append(f'- **Evidence: `{cls}`** — stays `pending_audit`.')
        if entry.get('reproduction_caveat'):
            lines.append(f'- Reproduction caveat: {entry["reproduction_caveat"]}')
        if entry.get('dead_doi_note'):
            lines.append(f'- Note: {entry["dead_doi_note"]}')
        lines.append('')
    lines.append('## Confirmed sources (unchanged from Stage 1)')
    lines.append('')
    lines.append('| Source | Train include | Val include | License |')
    lines.append('| --- | ---: | ---: | --- |')
    for source in sorted(audit['manifest_counts']['train']):
        lines.append(
            f'| {source} | {fmt_int(audit["manifest_counts"]["train"][source])} | '
            f'{fmt_int(audit["manifest_counts"]["val"].get(source, 0))} | '
            f'{CONFIRMED_LICENSES.get(source, "pending_audit")} |')
    lines.append('')
    lines.append('## Web access during this audit')
    lines.append('')
    lines.append('The prefix evidence above was gathered with `open-websearch` plus direct `curl` '
                 'and archive-index reads. Search engines were unreliable from this host: '
                 '`duckduckgo` worked intermittently, `bing` is region-locked and returned '
                 'unrelated Chinese-language results for quoted filename queries, `baidu` '
                 'returned `302 engine_error`, `brave` was `429` rate-limited, `sogou` served an '
                 'anti-bot page, `startpage` failed token extraction, and `exa` returned zero '
                 'rows. Where a search engine failed, the evidence above came from the source '
                 'dataset\'s own API or archive central directory (fetched by HTTP Range where '
                 'the archive was large), which is a stronger index than a search hit.')
    lines.append('')
    lines.append('Host-level access notes: `zenodo.org` began returning `403` mid-session (this '
                 'affects the `weds` re-check, noted above); `grassclover.multicrop.dk` failed the '
                 'TLS handshake, so the GrassClover evidence uses the Aalborg AILab re-host; '
                 '`mdpi.com` and `sciencedirect.com` returned `403`; `api.github.com` was rate '
                 'limited (raw.githubusercontent.com was not).')
    lines.append('')
    lines.append('## Consequence')
    lines.append('')
    lines.append(f'{fmt_int(total_pending - moved)} images ('
                 f'{fmt_pct(total_pending - moved, audit["images"])} of the corpus) remain '
                 'quarantined. They are present in `data/exclusion_ledger.csv` with reason '
                 '`unconfirmed_source_mapping` and are absent from both include manifests and '
                 'from the frozen split.')
    lines.append('')
    lines.append('This quarantine is now a *policy* choice rather than an evidence gap for 9 of '
                 'the 10 prefixes: the evidence to admit them exists and is recorded above. '
                 'Promoting them would raise the frozen corpus from 15,603 to ~30,000 images and '
                 'would require a new data version, re-screening of the affected duplicate '
                 'candidates, and a fresh freeze. That is a curation decision, so the default '
                 'remains the conservative `agrifield40k-v0` freeze.')
    lines.append('')
    write_text(path, '\n'.join(lines) + '\n')

    payload = {
        'revision': FREEZE_REVISION,
        'card_url': CARD_URL,
        'images': audit['images'],
        'pending_rows_before': total_pending,
        'pending_rows_confirmed': moved,
        'pending_rows_after': total_pending - moved,
        'pending_by_prefix': pend,
        'confirmed_source_manifest_counts': audit['manifest_counts'],
        'confirmed_licenses': CONFIRMED_LICENSES,
        'candidate_licenses': CANDIDATE_LICENSES,
        'web_evidence': WEB_EVIDENCE,
        'web_access_notes': {
            'open_websearch_used': True,
            'engines_ok': ['duckduckgo (intermittent)'],
            'engines_failed': [
                'bing (region-locked, unrelated results for quoted filename queries)',
                'baidu (302 engine_error)', 'brave (429)', 'sogou (anti-bot page)',
                'startpage (token extraction)', 'exa (zero rows)',
            ],
            'index_reads_used': [
                'source-dataset APIs and archive central directories (HTTP Range)',
            ],
            'direct_curl_ok': [CARD_URL, DATASET_PAGE, PAPER_URL, PROJECT_REPO, PROJECT_PAGE],
            'rate_limited_or_403': [
                'zenodo.org (403 mid-session; affects the weds re-check)',
                'mdpi.com (403)', 'sciencedirect.com (403)',
                'api.github.com (rate limited; raw.githubusercontent.com was not)',
            ],
            'unreachable': [
                'https://grassclover.multicrop.dk/ (TLS handshake failure; '
                'GrassClover evidence uses the Aalborg AILab re-host)',
            ],
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_source_index_evidence_report(path, pend):
    """Record the per-prefix original-source-index evidence and the promotion path.

    Deliberately separate from the freeze record: this file is EVIDENCE, the
    freeze is POLICY. Nothing here changes a row's status.
    """
    idx = {p: i for p, i in pend.items() if i['evidence_class'] == 'original_source_index'}
    sup = {p: i for p, i in pend.items() if i['evidence_class'] == 'supporting_only'}
    lines = []
    lines.append('# Stage 2 original-source-index evidence (prefix provenance)')
    lines.append('')
    lines.append('This file records, for each AgriField-40K short prefix that the publisher card '
                 'leaves unmapped, whether the observed original filename appears verbatim in the '
                 'named source dataset\'s **own published file index**. It is evidence only. The '
                 'frozen split is unchanged: every prefix below stays `pending_audit` until a '
                 'curator promotes it (see `reports/data_freeze.md`).')
    lines.append('')
    lines.append('Naming context: the card states AgriField renames images to '
                 '`[dataset_source]_[id]`. The audit observes the `[id]` part and asks whether '
                 'that exact string exists in the named upstream source.')
    lines.append('')
    n_idx = len(idx)
    lines.append(f'## Summary — {n_idx} of {len(pend)} prefixes adoptable, {len(sup)} '
                 'supporting-only')
    lines.append('')
    lines.append('| Prefix | Source | Adoptable rows | Evidence class | Index read |')
    lines.append('| --- | --- | ---: | --- | --- |')
    for prefix, info in sorted(pend.items()):
        e = info['web_entry']
        cls = info['evidence_class']
        label = {'original_source_index': 'literal match',
                 'supporting_only': 'components only'}.get(cls, cls)
        lines.append(f'| `{prefix}` | {info["candidate_source"]} | {fmt_int(info["rows"])} | '
                     f'{label} | <{e["url"]}> |')
    lines.append('')
    lines.append('## Per-prefix evidence and caveats')
    lines.append('')
    for prefix, info in sorted(pend.items()):
        e = info['web_entry']
        lines.append(f'### `{prefix}` — {info["candidate_source"]}')
        lines.append('')
        lines.append(f'{e["claim"]}')
        lines.append('')
        lines.append(f'- **Establishes:** {e["establishes"]}')
        lines.append(f'- **Does not establish:** {e["does_not_establish"]}')
        if e.get('reproduction_caveat'):
            lines.append(f'- **Reproduction caveat:** {e["reproduction_caveat"]}')
        if e.get('dead_doi_note'):
            lines.append(f'- **Note:** {e["dead_doi_note"]}')
        lines.append('- **Index reads:** ' + ', '.join(
            f'<{u}>' for u in [e['url']] + list(e.get('evidence_urls', []))))
        lines.append('- **Adoption:** `not_adopted` — rows remain `pending_audit` in '
                     '`agrifield40k-v0`.')
        lines.append('')
    lines.append('## Promotion path (curator action, not automated)')
    lines.append('')
    lines.append('Admitting the evidence-backed prefixes is a data-version bump, not a flag flip. '
                 'The required steps, in order:')
    lines.append('')
    lines.append('1. Curator records an explicit `prefix -> source` decision for each prefix in the '
                 'prefix map, citing the index read above.')
    lines.append('2. Re-run the source/license audit so the affected rows move from '
                 '`pending_audit` to `confirmed` with a license attached.')
    lines.append('3. Re-run duplicate screening over the newly admitted rows — their near-'
                 'duplicates are currently unscreened because the rows were never in an include '
                 'manifest.')
    lines.append('4. Regenerate the include manifests and re-freeze under a new data version '
                 '(`agrifield40k-v1` or later), with fresh SHA-256 digests.')
    lines.append('5. Record the change and its evidence in the Stage 3 baseline protocol before '
                 'any training run uses the new split, so the Stage 3/4 comparison basis stays '
                 'explicit.')
    lines.append('')
    lines.append('Promoting all 9 adoptable prefixes plus `mc` would raise the frozen corpus from '
                 '15,603 to roughly 30,000 images. Until that is done deliberately, the '
                 'conservative `agrifield40k-v0` freeze stands.')
    lines.append('')
    write_text(path, '\n'.join(lines) + '\n')


def write_distribution_report(path: Path, dist):
    train = dist['train_counts']
    val = dist['val_counts']
    train_total = dist['train_total']
    val_total = dist['val_total']
    lucas_train = train.get('LUCASVision', 0)
    lines = []
    lines.append('# Stage 2B manifest distribution analysis')
    lines.append('')
    lines.append('Analysis only. Nothing in this section changes `pretrain_manifest.csv` or '
                 '`val_manifest.csv`; the cap schemes below are worked examples, not applied '
                 'weighting.')
    lines.append('')
    lines.append('## Per-source counts')
    lines.append('')
    lines.append(f'Training manifest: **{fmt_int(train_total)}** samples. '
                 f'Validation manifest: **{fmt_int(val_total)}** samples.')
    lines.append('')
    lines.append('| Source | Train | Train share | Val | Val share | Val/(Train+Val) | License |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | --- |')
    for source in sorted(train, key=lambda s: -train[s]):
        t = train[source]
        v = val.get(source, 0)
        lines.append(
            f'| {source} | {fmt_int(t)} | {fmt_pct(t, train_total)} | {fmt_int(v)} | '
            f'{fmt_pct(v, val_total)} | {fmt_pct(v, t + v)} | '
            f'{CONFIRMED_LICENSES.get(source, "-")} |')
    lines.append(f'| **Total** | **{fmt_int(train_total)}** | **100.00%** | '
                 f'**{fmt_int(val_total)}** | **100.00%** | '
                 f'**{fmt_pct(val_total, train_total + val_total)}** | |')
    lines.append('')
    lines.append('Per LUCASVision prefix (`lvb*`), since it is the only source split across '
                 'several prefixes:')
    lines.append('')
    lines.append('| Prefix | Train | Val |')
    lines.append('| --- | ---: | ---: |')
    for prefix in sorted(dist['lvb_prefix_counts']['train']):
        lines.append(f'| `{prefix}` | {fmt_int(dist["lvb_prefix_counts"]["train"][prefix])} | '
                     f'{fmt_int(dist["lvb_prefix_counts"]["val"].get(prefix, 0))} |')
    lines.append('')

    lines.append('## LUCASVision share of the training manifest')
    lines.append('')
    lines.append(f'**LUCASVision is {fmt_int(lucas_train)} of {fmt_int(train_total)} training '
                 f'samples = {fmt_pct(lucas_train, train_total)}.** '
                 f'In validation it is {fmt_int(val.get("LUCASVision", 0))} of {fmt_int(val_total)}'
                 f' = {fmt_pct(val.get("LUCASVision", 0), val_total)}. '
                 f'Combined across both manifests it is '
                 f'{fmt_pct(lucas_train + val.get("LUCASVision", 0), train_total + val_total)} '
                 'of the included corpus.')
    lines.append('')
    lines.append('The next largest source, VegAnn, is '
                 f'{fmt_int(train.get("VegAnn", 0))} samples ({fmt_pct(train.get("VegAnn", 0), train_total)}). '
                 'LUCASVision therefore outweighs it '
                 f'{(lucas_train / train["VegAnn"]) if train.get("VegAnn") else float("nan"):.1f}'
                 'x. The two smallest sources are SorghumWeed '
                 f'({fmt_int(train.get("SorghumWeed", 0))}) and RadishWheat '
                 f'({fmt_int(train.get("RadishWheat", 0))}), so the corpus spans roughly two '
                 'orders of magnitude between its largest and smallest contributor.')
    lines.append('')
    lines.append('Two further facts matter for interpretation. First, the publisher records '
                 'LUCASVision as a handheld-camera source of 12 crop species '
                 f'(<{PAPER_URL}>, Table 1), so it is a wide, geographically and seasonally '
                 'spread survey rather than a single acquisition burst; its dominance is not an '
                 'artefact of one shoot. Second, its audited rows arrive through 12 distinct '
                 '`lvb*` prefixes (1,156-1,569 rows each), which is the only source in the corpus '
                 'split across several prefixes. Both cuts are shown above. What the audit cannot '
                 'tell you is how close LUCASVision is to natural-image photography in '
                 'distribution: that is an empirical question, and the visual evidence for it is '
                 'in the contact sheet.')
    lines.append('')

    lines.append('## Cap schemes (worked examples, NOT applied)')
    lines.append('')
    lines.append('A cap is applied by keeping the first N samples of the capped source in '
                 'manifest order. Because the manifests are written in sorted `sample_id` order '
                 'by Stage 1, "first N" is deterministic and reproducible.')
    lines.append('')

    def cap_table(title, caps, note):
        lines.append(f'### {title}')
        lines.append('')
        lines.append(note)
        lines.append('')
        # No cap == unlimited, not zero.
        kept = {s: min(caps.get(s, train[s]), train[s]) for s in train}
        kept_total = sum(kept.values())
        lines.append('| Source | Natural | Capped | Natural share | Capped share |')
        lines.append('| --- | ---: | ---: | ---: | ---: |')
        for s in sorted(train, key=lambda s: -train[s]):
            lines.append(f'| {s} | {fmt_int(train[s])} | {fmt_int(kept[s])} | '
                         f'{fmt_pct(train[s], train_total)} | {fmt_pct(kept[s], kept_total)} |')
        lines.append(f'| **Total** | **{fmt_int(train_total)}** | **{fmt_int(kept_total)}** | '
                     f'| |')
        lines.append('')
        return kept_total

    t_a = cap_table(
        'Scheme A — cap LUCASVision at 5,000',
        {'LUCASVision': 5000},
        'The mildest option. Every source keeps all of its samples except the one dominant '
        'source, which is trimmed to 5,000.')
    t_b = cap_table(
        'Scheme B — cap every source at 1,000',
        {s: 1000 for s in train},
        'Equal-volume per source. This is the strongest flattening option and it throws away '
        f'the most data: {sum(1 for s in train if train[s] > 1000)} of the '
        f'{len(train)} confirmed sources sit above 1,000, so only the sources already below the '
        'cap keep all of their samples.')
    lines.append('| | Natural | Scheme A (LUCASVision<=5000) | Scheme B (all<=1000) |')
    lines.append('| --- | ---: | ---: | ---: |')
    lines.append(f'| Training samples retained | {fmt_int(train_total)} | {fmt_int(t_a)} | '
                 f'{fmt_int(t_b)} |')
    lines.append(f'| Samples discarded | 0 | {fmt_int(train_total - t_a)} | '
                 f'{fmt_int(train_total - t_b)} |')
    lines.append(f'| Largest source share | {fmt_pct(lucas_train, train_total)} | '
                 f'{fmt_pct(min(lucas_train, 5000), t_a)} | {fmt_pct(1000, t_b)} |')
    lines.append('')

    lines.append('## Risk assessment')
    lines.append('')
    lines.append('**Single-source dominance — present, and material.** '
                 f'{fmt_pct(lucas_train, train_total)} of the training manifest is one source. '
                 'For I-JEPA-style pretraining this is not automatically wrong — LUCASVision is '
                 'real field imagery and more data generally helps — but it does mean the '
                 'learned prior is mostly a LUCAS prior. Any downstream evaluation on a source '
                 'that is visually distinct from LUCAS (robot, drone, stereo RGB-D) is a '
                 'transfer test, not an in-distribution test. It also means a change in how the '
                 'LUCAS rows were selected would move the whole pretraining result.')
    lines.append('')
    lines.append('**Small-source oversampling — present.** With a natural sampler the six '
                 'sources contribute between '
                 f'{fmt_int(train.get("SorghumWeed", 0))} and {fmt_int(lucas_train)} training '
                 'samples. SorghumWeed, at '
                 f'{fmt_pct(train.get("SorghumWeed", 0), train_total)} of the corpus, is unlikely '
                 'to move the loss at all; for practical purposes it is decorative. If a '
                 'downstream benchmark is drawn from a small source, the pretraining signal from '
                 'that source is negligible regardless of how relevant it is.')
    lines.append('')
    lines.append('**Recommendation.** Keep the frozen manifest as the default, so the Stage 1 '
                 'audit and the freeze record stay the reference. For the Stage 2 pretraining '
                 'runs, use a source-balanced sampler (per-source sampling weight proportional '
                 'to `1/sqrt(source_count)`) rather than a hard cap: it preserves every image, '
                 'flattens the effective distribution far more than Scheme B, and does not '
                 'require editing the manifests. If a hard cap is wanted for a specific '
                 'ablation, Scheme A is the defensible one — it discards '
                 f'{fmt_int(train_total - t_a)} images '
                 f'({fmt_pct(train_total - t_a, train_total)}) and keeps LUCASVision as the '
                 'largest single source, while Scheme B discards '
                 f'{fmt_pct(train_total - t_b, train_total)} of the corpus and makes LUCASVision '
                 'a minority source by volume.')
    lines.append('')
    write_text(path, '\n'.join(lines) + '\n')


def write_protocol(path: Path, audit, review_rows):
    cross = sum(1 for r in review_rows if r['cross_split'] == 'True')
    man = sum(1 for r in review_rows if r['isolation_category'] == 'manifest_affecting')
    same = len(review_rows) - cross - man
    exact = sum(1 for r in review_rows if r['kind'] == 'sha256')
    perc = len(review_rows) - exact
    lines = []
    lines.append('# Duplicate review protocol')
    lines.append('')
    lines.append('This is the protocol a human follows to adjudicate the perceptual duplicate '
                 'candidates emitted by Stage 1. It exists because a perceptual hash is a '
                 'screening signal, not proof.')
    lines.append('')
    lines.append('## How the candidates were generated')
    lines.append('')
    lines.append('`tools/audit_data.py` computed three things for all '
                 f'{fmt_int(audit["images"])} images:')
    lines.append('')
    lines.append('- SHA-256 of the raw file bytes.')
    lines.append('- SHA-256 of the fully decoded RGB pixel buffer.')
    lines.append('- A 64-bit pHash and a 64-bit dHash, both computed from the decoded RGB image '
                 'using a DCT-II basis for pHash.')
    lines.append('')
    lines.append('An edge is emitted in either of two ways:')
    lines.append('')
    lines.append('1. **`kind=sha256`** — identical file bytes or identical decoded pixels. These '
                 'are duplicates by construction and are recorded as '
                 f'`exact_duplicate` by this protocol. {exact} edges.')
    lines.append('2. **`kind=perceptual_candidate`** — pHash Hamming distance <= 4 **and** '
                 'dHash Hamming distance <= 6. The intersection of two independent screeners is '
                 'used because field scenes are visually uniform and a single hash produces '
                 'collisions between unrelated images. This is a screening rule with false '
                 f'positives. {perc} edges.')
    lines.append('')
    lines.append('Every edge is then embedded in a connected component, and Stage 1 kept one '
                 'deterministic representative per component (preferring the validation split). '
                 'A component touching any PhenoBench row was excluded in full.')
    lines.append('')
    lines.append(f'Stage 2B re-emits all {fmt_int(len(review_rows))} edges into '
                 '`data/duplicate_review.csv`, enriched with split, source, path and '
                 'manifest-membership context, and generates the offline contact sheet in '
                 '`reports/duplicate_review/`.')
    lines.append('')
    lines.append('## What "duplicate" means here')
    lines.append('')
    lines.append('A pair is a **duplicate** if the two images depict the same physical scene at '
                 'effectively the same moment, such that a model can memorise one from the '
                 'other. Practically:')
    lines.append('')
    lines.append('- the same field patch from the same camera position, differing only by '
                 'compression, resizing, a small exposure or white-balance shift, or a small '
                 'crop offset; or')
    lines.append('- two frames of the same video or robot sequence that are close enough in time '
                 'that the visible plants have not materially moved; or')
    lines.append('- a re-encoded or re-cropped copy of the same original image.')
    lines.append('')
    lines.append('A pair is **not a duplicate** if the two images are different captures of the '
                 'same field at different times, different positions, different seasons, or '
                 'different growth stages. Repetition of subject matter is expected in '
                 'agricultural data and is not leakage.')
    lines.append('')
    lines.append('The borderline case is consecutive frames in a sequence where plants have '
                 'moved slightly. The rule is: if a human can tell the two images apart at a '
                 'glance and both carry usable visual signal, they are **not** duplicates, even '
                 'if they are adjacent in time. Flag them as `near_duplicate` instead so the '
                 'decision is visible.')
    lines.append('')
    lines.append('## Decision categories')
    lines.append('')
    lines.append('| Verdict | Meaning | Effect |')
    lines.append('| --- | --- | --- |')
    lines.append('| `exact_duplicate` | Identical bytes or identical decoded pixels. | Treated as '
                 'certain duplication. No visual inspection needed. |')
    lines.append('| `confirmed_duplicate` | Human judged the pair to be the same scene. | Real '
                 'leakage. |')
    lines.append('| `near_duplicate` | Same scene, minor temporal or photometric offset, both '
                 'usable. | Recorded, but not treated as leakage by the conservative rule. |')
    lines.append('| `not_duplicate` | Human judged the pair to be different scenes. | The edge is '
                 'retired. |')
    lines.append('| `pending_manual_review` | Initial value. Nobody has looked. | Conservative '
                 'rule applies. |')
    lines.append('')
    lines.append('## Review order')
    lines.append('')
    lines.append(f'1. **Cross-split candidates ({cross} edges).** One endpoint in train, the '
                 'other in val. If confirmed, these are direct pretraining-to-validation '
                 'leakage. These are section A of the contact sheet.')
    lines.append(f'2. **Manifest-affecting candidates ({man} edges).** One endpoint is in an '
                 'include manifest and the other is not. Section B. A confirmed duplicate '
                 'retires the excluded endpoint permanently; a `not_duplicate` verdict shows the '
                 'Stage 1 component rule dropped a row that did not need dropping.')
    lines.append(f'3. **Same-split screening candidates ({same} edges).** Neither endpoint is in '
                 'a manifest, so the frozen manifests are unaffected either way. Section C. '
                 'Lowest priority, still worth sampling to calibrate the screener.')
    lines.append('')
    lines.append('Within each section the contact sheet is sorted by `kind` (exact first), then '
                 'by ascending pHash distance, then by ascending dHash distance, then by '
                 '`edge_id`, so the most suspicious pairs come first.')
    lines.append('')
    lines.append('## How a verdict is folded back')
    lines.append('')
    lines.append('The reviewer works in `data/duplicate_review.csv`, one row per edge. '
                 '`review_status` starts at `pending_manual_review` and is overwritten with the '
                 'verdict above. `reviewer` holds the reviewer identifier, `review_notes` holds '
                 'free text, `reviewed_at` holds an ISO-8601 date. `edge_id` is the stable key: '
                 'it is `edge-<n>` where `n` is the 1-based row index in '
                 '`data/duplicate_pairs.csv`, so it never changes as long as Stage 1 is not '
                 're-run. The contact sheet keys off the same `edge_id`, and its "Collect '
                 'verdicts" button emits JSON keyed by it, so the sheet and the CSV can be '
                 'reconciled row by row.')
    lines.append('')
    lines.append('A verdict is only accepted if all four of `review_status`, `reviewer`, '
                 '`review_notes` and `reviewed_at` are non-empty. `reviewer` is deliberately '
                 'left blank by this tool — an unattributed verdict is not a verdict.')
    lines.append('')
    lines.append('If a confirmed or near-duplicate verdict lands on an edge whose endpoints are '
                 'in the frozen manifests, the freeze is invalidated: bump '
                 '`data_version`, regenerate `data/frozen/`, and record the change in '
                 '`reports/data_freeze.md`. Do not edit a frozen manifest in place.')
    lines.append('')
    lines.append('## Conservative rule applied to unreviewed candidates')
    lines.append('')
    lines.append('Until an edge carries an accepted human verdict, it is treated as if it '
                 '*might* be a duplicate, and the freeze isolates it. The rule is:')
    lines.append('')
    lines.append('> An edge triggers isolation if **any** of: (a) `cross_split` is true, '
                 '(b) exactly one endpoint is in an include manifest, or (c) both endpoints are '
                 'in include manifests and land in different ones.')
    lines.append('')
    lines.append('For each triggered edge, **both** endpoints are excluded from the frozen '
                 'manifests, whichever split they are in and whichever manifest they are in. '
                 'A candidate edge is a two-sided warning: the audit cannot say which of the '
                 'two images is the true duplicate, so it keeps neither.')
    lines.append('')
    lines.append(f'On the current data this isolates {cross + man} edges and '
                 f'{fmt_int(audit["isolation_manifest_rows_removed"])} samples that were actually '
                 f'in an include manifest ({fmt_int(audit["isolation_train_removed"])} train, '
                 f'{fmt_int(audit["isolation_val_removed"])} val). A further '
                 f'{fmt_int(audit["isolation_rows_removed"] - audit["isolation_manifest_rows_removed"])} '
                 'endpoints named by those edges are already outside both manifests '
                 '(`pending_audit` or `exclude`), so isolating them changes nothing in the '
                 'manifests; they are counted here only so the two numbers are not conflated.')
    lines.append('')
    lines.append('**This is NOT completed manual deduplication.** It is an isolation rule '
                 'applied in the absence of manual deduplication. It guarantees that no '
                 'known-suspected pair straddles the frozen split, and it pays for that '
                 'guarantee by discarding samples that are probably fine. In the worst case a '
                 'false-positive edge costs two real images; on this data most flagged endpoints '
                 'were already outside the manifests, so the real cost was '
                 f'{fmt_int(audit["isolation_manifest_rows_removed"])} images. Completing the '
                 'human review will let most of these be returned.')
    lines.append('')
    write_text(path, '\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--image-root', type=Path, default=DEFAULT_IMAGE_ROOT,
                    help='root containing agrifield40k/{train,val}/*.jpg')
    ap.add_argument('--thumb-width', type=int, default=160)
    ap.add_argument('--no-thumbs', action='store_true',
                    help='skip thumbnail generation (faster; contact sheet will show placeholders)')
    ap.add_argument('--generated-at', default=None,
                    help='ISO-8601 timestamp written into the freeze record. Omit for a '
                         'reproducible run (the field is then empty).')
    ap.add_argument('--check-only', action='store_true',
                    help='verify inputs and print counts without writing any output')
    args = ap.parse_args()

    started = time.time()
    image_root = args.image_root.resolve()

    samples_path = Path('data/all_samples.csv')
    pairs_path = Path('data/duplicate_pairs.csv')
    ledger_path = Path('data/exclusion_ledger.csv')
    registry_path = Path('data/source_registry.json')
    train_manifest = Path('pretrain_manifest.csv')
    val_manifest = Path('val_manifest.csv')

    for required in (samples_path, pairs_path, ledger_path, registry_path,
                     train_manifest, val_manifest):
        if not required.exists():
            raise SystemExit(f'missing required Stage 1 input: {required}')

    samples = {r['sample_id']: r for r in read_csv(samples_path)}
    pairs = read_csv(pairs_path)
    registry = json.loads(registry_path.read_text(encoding='utf-8'))['sources']
    train_ids = {r['sample_id'] for r in read_csv(train_manifest)}
    val_ids = {r['sample_id'] for r in read_csv(val_manifest)}
    ledger = read_csv(ledger_path)

    if len(samples) != len(read_csv(samples_path)):
        raise SystemExit('duplicate sample_id in data/all_samples.csv')
    if len(train_ids) + len(val_ids) > len(samples):
        raise SystemExit('manifests contain more samples than were audited')
    if train_ids & val_ids:
        raise SystemExit('a sample_id appears in both manifests')
    for pair in pairs:
        for key in ('sample_a', 'sample_b'):
            if pair[key] not in samples:
                raise SystemExit(f'unresolved edge endpoint: {pair[key]}')

    # --- source audit numbers ---------------------------------------------
    pending_rows = [r for r in samples.values() if r['status'] == 'pending_audit']
    pending_by_prefix_rows = Counter(r['source_prefix'] for r in pending_rows)
    # Stage 1 matches with re.fullmatch(prefix_regex, prefix). Every pending
    # prefix is a single literal token, so an exact dict lookup is equivalent
    # to the regex and needs no regex engine here.
    registry_by_prefix = {spec['prefix_regex']: spec for spec in registry
                          if not any(c in spec['prefix_regex'] for c in '*()|?+[')}
    pending_by_prefix = {}
    for prefix, count in sorted(pending_by_prefix_rows.items()):
        spec = registry_by_prefix.get(prefix, {})
        entry = next((e for e in WEB_EVIDENCE if e['prefix'] == prefix),
                     {'claim': 'not checked', 'url': '', 'establishes': '', 'does_not_establish': '',
                      'evidence_class': 'not_checked'})
        ids = sorted(r['original_id'] for r in pending_rows if r['source_prefix'] == prefix)
        pending_by_prefix[prefix] = {
            'rows': count,
            'candidate_source': spec.get('candidate_source') or spec.get('source') or None,
            'expected_count': spec.get('expected_count'),
            'count_matches': spec.get('expected_count') == count,
            'evidence_class': entry['evidence_class'],
            'web_entry': entry,
            'id_example': ids[0] if ids else '',
            'id_examples': ids[:3],
            'id_pattern_note': f'{len(ids)} distinct original IDs; no source token in the ID',
            'train_rows': sum(1 for r in pending_rows if r['source_prefix'] == prefix
                              and r['split'] == 'train'),
            'val_rows': sum(1 for r in pending_rows if r['source_prefix'] == prefix
                            and r['split'] == 'val'),
        }

    manifest_counts = {
        'train': Counter(r['source'] for r in samples.values()
                         if r['split'] == 'train' and r['status'] == 'include'),
        'val': Counter(r['source'] for r in samples.values()
                       if r['split'] == 'val' and r['status'] == 'include'),
    }

    # --- duplicate review rows -------------------------------------------
    review_rows = []
    for index, pair in enumerate(pairs, 1):
        a, b = pair['sample_a'], pair['sample_b']
        ra, rb = samples[a], samples[b]
        in_train = a in train_ids
        in_val = a in val_ids
        in_train_b = b in train_ids
        in_val_b = b in val_ids
        cross_split = pair['cross_split'] == 'True'
        manifest_affecting = (in_train or in_val) != (in_train_b or in_val_b)
        manifest_crossing = in_train and in_val_b or in_train_b and in_val
        if cross_split:
            category = 'cross_split'
        elif manifest_affecting or manifest_crossing:
            category = 'manifest_affecting'
        else:
            category = 'same_split_screening'
        review_rows.append({
            'edge_id': f'edge-{index:04d}',
            'sample_a': a, 'sample_b': b, 'kind': pair['kind'],
            'phash_distance': pair['phash_distance'], 'dhash_distance': pair['dhash_distance'],
            'cross_split': pair['cross_split'],
            'split_a': ra['split'], 'split_b': rb['split'],
            'source_a': ra['source'], 'source_b': rb['source'],
            'path_a': ra['path'], 'path_b': rb['path'],
            'in_train_manifest': str(in_train or in_train_b).lower(),
            'in_val_manifest': str(in_val or in_val_b).lower(),
            'review_status': PENDING_REVIEW,
            'reviewer': '', 'review_notes': '', 'reviewed_at': '',
            'isolation_category': category,
            'thumb_a': '', 'thumb_b': '',
        })

    # Contact-sheet ordering: exact first, then tightest hash, then edge_id.
    review_rows.sort(key=lambda r: (r['kind'] != 'sha256',
                                    int(r['phash_distance']), int(r['dhash_distance']),
                                    r['edge_id']))

    # --- isolation rule ---------------------------------------------------
    isolated = set()
    for row in review_rows:
        if row['isolation_category'] in ('cross_split', 'manifest_affecting'):
            isolated.add(row['sample_a'])
            isolated.add(row['sample_b'])
    frozen_train = [r for r in samples.values()
                    if r['sample_id'] in train_ids and r['sample_id'] not in isolated]
    frozen_val = [r for r in samples.values()
                  if r['sample_id'] in val_ids and r['sample_id'] not in isolated]
    removed_train = [r for r in samples.values()
                     if r['sample_id'] in train_ids and r['sample_id'] in isolated]
    removed_val = [r for r in samples.values()
                   if r['sample_id'] in val_ids and r['sample_id'] in isolated]
    if len(frozen_train) + len(removed_train) != len(train_ids):
        raise SystemExit('frozen train partition does not cover the training manifest')
    if len(frozen_val) + len(removed_val) != len(val_ids):
        raise SystemExit('frozen val partition does not cover the validation manifest')
    if {r['sample_id'] for r in frozen_train} & {r['sample_id'] for r in frozen_val}:
        raise SystemExit('frozen train and val share a sample_id')

    audit = {
        'images': len(samples),
        'pending_rows': len(pending_rows),
        'pending_rows_confirmed': 0,
        'pending_by_prefix': pending_by_prefix,
        'manifest_counts': {k: dict(v) for k, v in manifest_counts.items()},
        'isolation_rows_removed': len(isolated),
        # Of the isolated endpoints, only the ones that were in an include
        # manifest actually shrink the frozen split. The rest are already
        # quarantined (pending_audit or excluded), so dropping them is a no-op
        # for the manifests and is reported separately so the two numbers are
        # never conflated.
        'isolation_manifest_rows_removed': len(isolated & (train_ids | val_ids)),
        'isolation_train_removed': len(removed_train),
        'isolation_val_removed': len(removed_val),
    }

    if args.check_only:
        print(json.dumps({
            'images': audit['images'],
            'pending_rows': audit['pending_rows'],
            'pending_rows_confirmed': 0,
            'pending_by_prefix': {k: v['rows'] for k, v in pending_by_prefix.items()},
            'review_edges': len(review_rows),
            'review_status_counts': dict(Counter(r['review_status'] for r in review_rows)),
            'isolation_categories': dict(Counter(r['isolation_category'] for r in review_rows)),
            'isolated_samples': len(isolated),
            'frozen_train': len(frozen_train), 'frozen_val': len(frozen_val),
            'train_manifest': len(train_ids), 'val_manifest': len(val_ids),
        }, indent=2))
        return

    # --- write everything -------------------------------------------------
    print(f'Writing duplicate review CSV ({len(review_rows)} edges)...', flush=True)
    write_csv(Path('data/duplicate_review.csv'), review_rows, REVIEW_FIELDS)

    print('Building contact sheet...', flush=True)
    if args.no_thumbs:
        thumbs_ok, thumbs_failed = generate_thumbs(review_rows, samples, image_root,
                                                   Path('reports/duplicate_review'),
                                                   args.thumb_width, enabled=False)
        build_contact_sheet(review_rows, samples, Path('reports/duplicate_review'),
                            args.thumb_width, thumbs_ok, thumbs_failed)
    else:
        thumbs_ok, thumbs_failed = generate_thumbs(review_rows, samples, image_root,
                                                   Path('reports/duplicate_review'),
                                                   args.thumb_width, enabled=True)
        build_contact_sheet(review_rows, samples, Path('reports/duplicate_review'),
                            args.thumb_width, thumbs_ok, thumbs_failed)
        write_csv(Path('data/duplicate_review.csv'), review_rows, REVIEW_FIELDS)
    print(f'  thumbnails: {thumbs_ok} written, {thumbs_failed} failed', flush=True)

    print('Writing reports...', flush=True)
    write_source_audit_report(Path('reports/stage2_data_audit.md'),
                              Path('reports/stage2_data_audit.json'), audit, registry)

    lvb_prefix = {
        'train': Counter(r['source_prefix'] for r in samples.values()
                         if r['split'] == 'train' and r['status'] == 'include'
                         and r['source_prefix'].startswith('lvb')),
        'val': Counter(r['source_prefix'] for r in samples.values()
                       if r['split'] == 'val' and r['status'] == 'include'
                       and r['source_prefix'].startswith('lvb')),
    }
    dist = {
        'train_counts': dict(manifest_counts['train']),
        'val_counts': dict(manifest_counts['val']),
        'train_total': len(train_ids),
        'val_total': len(val_ids),
        'lvb_prefix_counts': {k: dict(v) for k, v in lvb_prefix.items()},
    }
    write_distribution_report(Path('reports/data_distribution.md'), dist)
    write_protocol(Path('reports/duplicate_review_protocol.md'), audit, review_rows)

    # --- frozen split -----------------------------------------------------
    print('Writing frozen manifests...', flush=True)
    frozen_dir = Path('data/frozen')
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_train_path = frozen_dir / f'{DATA_VERSION}_train.csv'
    frozen_val_path = frozen_dir / f'{DATA_VERSION}_val.csv'
    # Sorted by sample_id for a byte-reproducible file.
    write_csv(frozen_train_path,
              sorted((r for r in frozen_train), key=lambda r: r['sample_id']), SAMPLE_FIELDS)
    write_csv(frozen_val_path,
              sorted((r for r in frozen_val), key=lambda r: r['sample_id']), SAMPLE_FIELDS)

    frozen_source_counts = {
        'train': dict(Counter(r['source'] for r in frozen_train)),
        'val': dict(Counter(r['source'] for r in frozen_val)),
    }
    frozen_files = {
        str(frozen_train_path): sha256_file(frozen_train_path),
        str(frozen_val_path): sha256_file(frozen_val_path),
    }
    source_manifests = {
        'pretrain_manifest.csv': sha256_file(train_manifest),
        'val_manifest.csv': sha256_file(val_manifest),
        'data/all_samples.csv': sha256_file(samples_path),
        'data/duplicate_pairs.csv': sha256_file(pairs_path),
        'data/exclusion_ledger.csv': sha256_file(ledger_path),
        'data/source_registry.json': sha256_file(registry_path),
    }

    isolation_edges = [r for r in review_rows
                       if r['isolation_category'] in ('cross_split', 'manifest_affecting')]
    freeze_manifest = {
        'data_version': DATA_VERSION,
        'freeze_status': 'provisional',
        'freeze_revision': FREEZE_REVISION,
        'card_url': CARD_URL,
        'generated_at': args.generated_at or '',
        'stage': 'Stage 2B',
        'included_sample_count': {
            'train': len(frozen_train),
            'val': len(frozen_val),
            'total': len(frozen_train) + len(frozen_val),
        },
        'per_source_breakdown': frozen_source_counts,
        'split_rule': {
            'origin': 'publisher 80/20 split in the pinned HF revision (32,136 train / 7,827 val '
                      'images before any filtering)',
            'publisher_statement': '80% training, 20% validation; split at sequence level for '
                                   'sequence-derived sources, random split for '
                                   'independent-image sources',
            'paper_url': PAPER_URL,
            'stage1_component_rule': 'one deterministic representative per connected component, '
                                     'preferring validation; any component touching PhenoBench '
                                     'excluded in full',
            'stage2b_isolation_rule': 'both endpoints of every split-crossing or '
                                      'manifest-affecting candidate edge that is still '
                                      'pending_manual_review are excluded from the frozen split',
            'train_val_disjoint': True,
        },
        'license_and_review_status': {
            'compilation_license': 'CC-BY-SA-4.0',
            'confirmed_sources': CONFIRMED_LICENSES,
            'unresolved_prefixes': sorted(pending_by_prefix),
            'unresolved_rows_excluded': len(pending_rows),
            'unresolved_prefix_evidence': {
                p: info['evidence_class'] for p, info in sorted(pending_by_prefix.items())
            },
            'unresolved_prefix_adoption': {
                p: info['web_entry'].get('adoption', 'not_adopted')
                for p, info in sorted(pending_by_prefix.items())
            },
            'unresolved_prefix_evidence_note':
                'RECORD ONLY. 9 of the 10 unresolved prefixes have a literal filename match '
                'against their named source\'s own published index and 1 (`mc`) has '
                'components-only support; see reports/stage2_source_index_evidence.md. All 10 '
                'remain status=pending_audit and excluded: the freeze is driven by the '
                'status/source columns, so recording evidence here cannot admit a row. Promotion '
                'is a curator action and a data-version bump.',
            'candidate_licenses_if_confirmed': CANDIDATE_LICENSES,
            'phenobench_excluded_rows': sum(1 for r in samples.values()
                                            if r['source'] == 'PhenoBench'),
            'gate_met': False,
            'gate_note': f'{fmt_int(len(pending_rows))} of {fmt_int(len(samples))} images remain '
                         'pending_audit. For 9 prefixes a literal source-index match now exists '
                         'and for 1 the match is components-only, but none has been adopted: '
                         'their sources/licences are not confirmed in the dataset version, so the '
                         'source/license gate is not met and the freeze stays provisional.',
        },
        'exclusion_reasons': {
            'unconfirmed_source_mapping': sum(
                1 for r in samples.values() if r['reason'] == 'unconfirmed_source_mapping'),
            'phenobench_source': sum(
                1 for r in samples.values() if r['reason'] == 'phenobench_source'),
            'duplicate_component_nonrepresentative': sum(
                1 for r in samples.values()
                if r['reason'] == 'duplicate_component_nonrepresentative'),
            'duplicate_component_touches_phenobench': sum(
                1 for r in samples.values()
                if r['reason'] == 'duplicate_component_touches_phenobench'),
            'stage2b_isolation_unreviewed_candidate': len(isolated & (train_ids | val_ids)),
        },
        'frozen_file_sha256': frozen_files,
        'source_manifest_sha256': source_manifests,
        'known_leakage_risks': [
            {
                'risk': 'unreviewed perceptual duplicate candidates',
                'detail': f'{sum(1 for r in review_rows if r["kind"] == "perceptual_candidate")} '
                          f'perceptual_candidate edges and '
                          f'{sum(1 for r in review_rows if r["kind"] == "sha256")} exact edges '
                          f'exist. The {len(isolation_edges)} split-crossing or '
                          'manifest-affecting ones are isolated, but manual visual adjudication '
                          'has not been performed on any of them, so the true duplicate rate is '
                          'unknown.',
                'mitigation': 'conservative isolation rule; see reports/duplicate_review_protocol.md',
            },
            {
                'risk': 'sequence-level split not independently verified',
                'detail': 'The publisher states the split is at sequence level for '
                          'sequence-derived sources. This audit verified pixel identity, not '
                          'that the publisher actually honoured the sequence rule. Near-duplicate '
                          'frames from the same sequence could therefore still straddle the '
                          'frozen split.',
                'mitigation': 'manual review of the perceptual candidates is the only way to '
                              'close this; the screener rule (pHash<=4 AND dHash<=6) is a bound, '
                              'not a guarantee',
            },
            {
                'risk': 'same-scene different-capture pairs below the screener threshold',
                'detail': 'A pair of the same field on different days, or a wider crop of the '
                          'same scene, can fall outside the pHash/dHash radius and never appear '
                          'as an edge at all. No hash-based screener can find these.',
                'mitigation': 'none in this audit; requires manual or learned semantic matching',
            },
            {
                'risk': 'unresolved sources could contain cross-split duplicates',
                'detail': f'{sum(1 for r in review_rows if r["cross_split"] == "True" and r["in_train_manifest"] == "false" and r["in_val_manifest"] == "false")} '
                          'cross-split perceptual candidates have both endpoints outside the '
                          'include manifests, and all of them involve unresolved prefixes. They '
                          'are quarantined, so they do not leak into the frozen split, but their '
                          'provenance is unverified and they would need re-screening if ever '
                          'admitted.',
                'mitigation': 'excluded from the frozen split entirely',
            },
        ],
        'unfinished_items': [
            'manual adjudication of all '
            f'{len(review_rows)} duplicate candidate edges ('
            f'{sum(1 for r in review_rows if r["review_status"] != PENDING_REVIEW)} reviewed, '
            f'{sum(1 for r in review_rows if r["review_status"] == PENDING_REVIEW)} '
            'pending_manual_review)',
            'source/license confirmation for the '
            f'{len(pending_by_prefix)} unresolved prefixes '
            f'({fmt_int(len(pending_rows))} rows) — evidence now recorded '
            '(reports/stage2_source_index_evidence.md) but adoption is a curator decision',
            'resolution of the iNatWeeds per-image attribution/observation fields for '
            'redistribution (already retained in data/all_samples.csv)',
            'independent verification of the publisher sequence-level split rule',
        ],
        'review_status_counts': dict(Counter(r['review_status'] for r in review_rows)),
        'isolation_summary': {
            'triggered_edges': len(isolation_edges),
            'cross_split_edges': sum(1 for r in isolation_edges if r['cross_split'] == 'True'),
            'manifest_affecting_edges': sum(
                1 for r in isolation_edges if r['isolation_category'] == 'manifest_affecting'),
            'flagged_endpoint_samples': len(isolated),
            'samples_removed_from_manifests': len(isolated & (train_ids | val_ids)),
            'train_samples_removed': len(removed_train),
            'val_samples_removed': len(removed_val),
            'note': 'This is an isolation rule, not completed manual deduplication.',
        },
    }
    freeze_manifest_path = frozen_dir / 'freeze_manifest.json'
    freeze_manifest_path.write_text(
        json.dumps(freeze_manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    write_freeze_report(Path('reports/data_freeze.md'), freeze_manifest, audit,
                        review_rows, frozen_train, frozen_val, removed_train, removed_val,
                        source_manifests, frozen_files)

    write_source_index_evidence_report(Path('reports/stage2_source_index_evidence.md'),
                                       audit['pending_by_prefix'])

    elapsed = time.time() - started
    print(json.dumps({
        'seconds': round(elapsed, 2),
        'frozen_train': len(frozen_train),
        'frozen_val': len(frozen_val),
        'isolated_samples': len(isolated),
        'review_edges': len(review_rows),
        'pending_rows_confirmed': 0,
    }, indent=2), flush=True)


def write_freeze_report(path, fm, audit, review_rows, frozen_train, frozen_val,
                        removed_train, removed_val, source_manifests, frozen_files):
    lines = []
    lines.append('# Data freeze record')
    lines.append('')
    lines.append(f'**Status: {fm["freeze_status"].upper()}.** The source and license gate is not '
                 'met, so this freeze is provisional. Read the gate note below before using these '
                 'files for anything that gets reported.')
    lines.append('')
    lines.append(f'- Data version: `{fm["data_version"]}`')
    lines.append(f'- Pinned HF revision: `{fm["freeze_revision"]}`')
    lines.append(f'- Publisher card: <{fm["card_url"]}>')
    lines.append(f'- Generated at: `{fm["generated_at"] or "(not recorded)"}`')
    lines.append('')
    lines.append('## Included sample count')
    lines.append('')
    lines.append('| Split | Source manifest | Frozen | Removed by isolation |')
    lines.append('| --- | ---: | ---: | ---: |')
    lines.append(f'| train | {fmt_int(len(removed_train) + len(frozen_train))} | '
                 f'**{fmt_int(len(frozen_train))}** | {fmt_int(len(removed_train))} |')
    lines.append(f'| val | {fmt_int(len(removed_val) + len(frozen_val))} | '
                 f'**{fmt_int(len(frozen_val))}** | {fmt_int(len(removed_val))} |')
    lines.append(f'| **total** | **{fmt_int(len(frozen_train) + len(frozen_val) + len(removed_train) + len(removed_val))}**'
                 f' | **{fmt_int(fm["included_sample_count"]["total"])}** | '
                 f'**{fmt_int(len(removed_train) + len(removed_val))}** |')
    lines.append('')
    lines.append(f'The frozen split holds {fmt_int(fm["included_sample_count"]["total"])} of the '
                 f'{fmt_int(len(frozen_train) + len(frozen_val) + len(removed_train) + len(removed_val))} '
                 f'samples in the include manifests, which is '
                 f'{fmt_pct(fm["included_sample_count"]["total"], len(frozen_train) + len(frozen_val) + len(removed_train) + len(removed_val))} '
                 'of them. Frozen train and frozen val are disjoint by construction and asserted '
                 'at write time.')
    lines.append('')
    lines.append('## Per-source breakdown')
    lines.append('')
    lines.append('| Source | Frozen train | Frozen train share | Frozen val | Frozen val share |')
    lines.append('| --- | ---: | ---: | ---: | ---: |')
    for source in sorted(fm['per_source_breakdown']['train'],
                         key=lambda s: -fm['per_source_breakdown']['train'][s]):
        t = fm['per_source_breakdown']['train'][source]
        v = fm['per_source_breakdown']['val'].get(source, 0)
        lines.append(f'| {source} | {fmt_int(t)} | {fmt_pct(t, len(frozen_train))} | '
                     f'{fmt_int(v)} | {fmt_pct(v, len(frozen_val))} |')
    lines.append('')
    lines.append('## License and review status')
    lines.append('')
    lines.append(f'- Compilation license: `{fm["license_and_review_status"]["compilation_license"]}`')
    lines.append('- Source/license gate met: **no**')
    lines.append(f'- {fm["license_and_review_status"]["gate_note"]}')
    lines.append('')
    lines.append('| Source | License |')
    lines.append('| --- | --- |')
    for source, lic in sorted(fm['license_and_review_status']['confirmed_sources'].items()):
        lines.append(f'| {source} | {lic} |')
    lines.append('')
    lines.append(f'Unresolved prefixes, all excluded from the freeze: '
                 f'{", ".join("`" + p + "`" for p in fm["license_and_review_status"]["unresolved_prefixes"])}. '
                 f'{fmt_int(fm["license_and_review_status"]["unresolved_rows_excluded"])} rows. '
                 'Their candidate licenses are recorded in '
                 '`reports/stage2_data_audit.md` but are not applied to any row.')
    lines.append('')
    lines.append('Provisioning note: 9 of those 10 prefixes now have a literal filename match '
                 'against their named source\'s own published index, and 1 (`mc`) has '
                 'components-only support; see `reports/stage2_source_index_evidence.md`. That '
                 'file is evidence, not adoption — every one of those rows still carries '
                 '`status=pending_audit` and stayed out of this freeze. The exclusion is a policy '
                 'choice pending a curator\'s prefix map and a data-version bump, not a claim that '
                 'the sources are unfindable.')
    lines.append('')
    lines.append('## Exclusion reasons (all audited images)')
    lines.append('')
    lines.append('`stage2b_isolation_unreviewed_candidate` counts only the samples that actually '
                 'came out of the include manifests, so this table is a partition of the '
                 f'{fmt_int(audit["images"])} audited images into included / excluded / pending. '
                 'The candidate edges also name further samples that were already outside both '
                 'manifests; those are not counted twice here.')
    lines.append('')
    lines.append('| Reason | Rows |')
    lines.append('| --- | ---: |')
    for reason, count in sorted(fm['exclusion_reasons'].items(), key=lambda kv: -kv[1]):
        lines.append(f'| {reason} | {fmt_int(count)} |')
    lines.append(f'| **included in frozen split** | '
                 f'**{fmt_int(fm["included_sample_count"]["total"])}** |')
    lines.append(f'| **total** | '
                 f'**{fmt_int(sum(fm["exclusion_reasons"].values()) + fm["included_sample_count"]["total"])}** |')
    lines.append('')
    lines.append('## Train/validation split rule')
    lines.append('')
    lines.append(f'- Origin: {fm["split_rule"]["origin"]}')
    lines.append(f'- Publisher statement: {fm["split_rule"]["publisher_statement"]} '
                 f'(<{fm["split_rule"]["paper_url"]}>)')
    lines.append(f'- Stage 1: {fm["split_rule"]["stage1_component_rule"]}')
    lines.append(f'- Stage 2B: {fm["split_rule"]["stage2b_isolation_rule"]}')
    lines.append(f'- Frozen train/val disjoint: {fm["split_rule"]["train_val_disjoint"]}')
    lines.append('')
    lines.append('## SHA-256')
    lines.append('')
    lines.append('### Frozen files')
    lines.append('')
    lines.append('| File | SHA-256 |')
    lines.append('| --- | --- |')
    for name, digest in sorted(frozen_files.items()):
        lines.append(f'| `{name}` | `{digest}` |')
    lines.append('')
    lines.append('### Source manifests (inputs to this freeze)')
    lines.append('')
    lines.append('| File | SHA-256 |')
    lines.append('| --- | --- |')
    for name, digest in sorted(source_manifests.items()):
        lines.append(f'| `{name}` | `{digest}` |')
    lines.append('')
    lines.append('`data/frozen/freeze_manifest.json` carries the same values in machine-readable '
                 'form. Any mismatch between a frozen file and its recorded hash means the frozen '
                 'file was edited after freezing and must be discarded.')
    lines.append('')
    lines.append('## Known leakage risks')
    lines.append('')
    for risk in fm['known_leakage_risks']:
        lines.append(f'**{risk["risk"]}.** {risk["detail"]}')
        lines.append(f'  - Mitigation: {risk["mitigation"]}')
        lines.append('')
    lines.append('## Isolation rule applied')
    lines.append('')
    iso = fm['isolation_summary']
    lines.append(f'{fmt_int(iso["triggered_edges"])} candidate edges triggered isolation '
                 f'({fmt_int(iso["cross_split_edges"])} cross-split, '
                 f'{fmt_int(iso["manifest_affecting_edges"])} manifest-affecting). Both endpoints '
                 f'of each triggered edge were dropped from the frozen split where they were '
                 f'present: {fmt_int(iso["samples_removed_from_manifests"])} samples came out of '
                 f'the include manifests ({fmt_int(iso["train_samples_removed"])} train, '
                 f'{fmt_int(iso["val_samples_removed"])} val). The edges also name '
                 f'{fmt_int(iso["flagged_endpoint_samples"] - iso["samples_removed_from_manifests"])} '
                 'further samples that were already outside both manifests, so dropping them was '
                 'a no-op for the manifests.')
    lines.append('')
    lines.append(f'Frozen training manifest therefore holds {fmt_int(len(frozen_train))} of the '
                 f'{fmt_int(len(frozen_train) + len(removed_train))} samples in '
                 '`pretrain_manifest.csv`. The validation manifest is untouched at '
                 f'{fmt_int(len(frozen_val))} rows: every manifest-affecting edge it touches had '
                 'its manifest endpoint on the training side.')
    lines.append('')
    lines.append('**This is not completed manual deduplication.** It is a conservative isolation '
                 'rule applied in the absence of manual deduplication. It removes any pair the '
                 'screener flagged that could straddle the frozen split, and it pays for that by '
                 'discarding images that are very likely fine. The true number of duplicates '
                 'among the isolated samples is unknown until a human reviews '
                 '`reports/duplicate_review/`.')
    lines.append('')
    lines.append('## Unfinished items')
    lines.append('')
    for item in fm['unfinished_items']:
        lines.append(f'- {item}')
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append('| File | Contents |')
    lines.append('| --- | --- |')
    lines.append(f'| `data/frozen/{fm["data_version"]}_train.csv` | frozen training manifest, '
                 f'{fmt_int(len(frozen_train))} rows |')
    lines.append(f'| `data/frozen/{fm["data_version"]}_val.csv` | frozen validation manifest, '
                 f'{fmt_int(len(frozen_val))} rows |')
    lines.append('| `data/frozen/freeze_manifest.json` | this record in machine-readable form |')
    lines.append('| `data/duplicate_review.csv` | one row per candidate edge, all '
                 f'`{PENDING_REVIEW}` |')
    lines.append('| `reports/stage2_data_audit.md` | source/license audit for the unresolved '
                 'prefixes |')
    lines.append('| `reports/data_distribution.md` | manifest distribution and cap analysis |')
    lines.append('| `reports/duplicate_review_protocol.md` | the human review protocol |')
    lines.append('| `reports/duplicate_review/index.html` | offline contact sheet |')
    lines.append('')
    lines.append('Regenerate everything with:')
    lines.append('')
    lines.append('```bash')
    lines.append('./.venv/bin/python -u -m tools.stage2_data_audit')
    lines.append('```')
    lines.append('')
    lines.append('## How these numbers were verified')
    lines.append('')
    lines.append('Every count in this record is read from the Stage 1 artefacts, not typed in. The '
                 'commands run to produce and check them:')
    lines.append('')
    lines.append('```bash')
    lines.append('# input shape')
    lines.append("wc -l data/all_samples.csv data/exclusion_ledger.csv data/duplicate_pairs.csv \\")
    lines.append("       pretrain_manifest.csv val_manifest.csv")
    lines.append('# status / pending-prefix / manifest-membership counts')
    lines.append("./.venv/bin/python -u -c \"import csv,collections; \\")
    lines.append("  R={r['sample_id']:r for r in csv.DictReader(open('data/all_samples.csv',newline='',encoding='utf-8'))}; \\")
    lines.append("  print(collections.Counter(r['status'] for r in R.values()))\"")
    lines.append('# regenerate this freeze')
    lines.append('./.venv/bin/python -u -m tools.stage2_data_audit')
    lines.append('# confirm the frozen split is disjoint and matches the recorded hashes')
    lines.append("sha256sum data/frozen/*.csv")
    lines.append('```')
    lines.append('')
    lines.append('Asserted at write time: every edge endpoint resolves to an audited sample, no '
                 '`sample_id` appears in both manifests, the frozen train and val sets are '
                 'disjoint, and the frozen train partition plus the isolated samples exactly '
                 'covers the training manifest. A run that fails any of these raises instead of '
                 'writing output.')
    lines.append('')
    write_text(path, '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
