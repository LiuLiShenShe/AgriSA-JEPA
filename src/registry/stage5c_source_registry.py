"""Stage 5C source registry.

The Stage 5A/5B registry (``agricultural_sources_v2.csv``) is the authority for
*what the 35 Agri-FM+ Table 1 references are*.  This module does not reinvent
that mapping: it reads the 35 ``agrifm_ref*`` rows, attaches a physical source
identity (several references are the same physical dataset), and adds the
Stage 5C download/access plan.

Three counting rules are enforced here and in every report:

* ``reported_images``   -- the number printed in the source paper;
* ``remotely_indexed``  -- what the remote archive lists (checked live);
* ``downloaded_images`` -- files actually present under the local data root.

They are never derived from each other.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_CSV = ROOT / 'data/registry/agricultural_sources_v2.csv'

REF_NUMBERS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 18, 20, 29, 31, 35, 37, 46, 48,
    57, 58, 60, 65, 73, 74, 75, 76, 79, 81, 82, 88, 90, 93, 95,
]

def ref_id(number: int) -> str:
    return f'agrifm_ref{number}'

REF_IDS: List[str] = [ref_id(n) for n in REF_NUMBERS]

#: Download states.  ``ARCHIVE_ONLY`` is a *use* flag, not a state: a holdout or
#: UAV source can still be DOWNLOADED_FULL while being barred from R2/R3.
STATES = (
    'DISCOVERED',
    'URL_CHECKED',
    'LICENSE_RECORDED',
    'REMOTE_INDEXED',
    'DOWNLOADING',
    'DOWNLOADED_FULL',
    'DOWNLOADED_PARTIAL',
    'BLOCKED',
)

#: References that share one physical dataset.  Only the canonical id counts.
SAME_PHYSICAL_DATASET: Dict[str, str] = {
    'agrifm_ref4': 'perennial_plants',
    'agrifm_ref79': 'ronin',
    'agrifm_ref88': 'phenobench',
    'agrifm_ref31': 'gwhd_2021',
    'agrifm_ref93': 'plantvillage',
    'agrifm_ref60': 'deepweeds',
}

#: Downstream holdouts.  Present in R0 for audit, never in R2/R3.
HOLDOUT_IDS = {
    'phenobench', 'gwhd_2021', 'plantdoc', 'minneapple',
    'roboflow_disease_detection', 'roboflow_crop_detection', 'plantseg',
    'multiclass_weeds', 'gwhd_instance_seg', 'deepweeds', 'growliflower',
    'cropandweed',
}


@dataclass
class AccessPlan:
    """How a source can be obtained from this machine (verified live)."""

    method: str                       # kaggle_dataset | kaggle_competition |
                                      # kaggle_dsv | dataverse | mendeley |
                                      # hf_mirror | local_reuse | s3_prefix | none
    ref: str = ''                     # slug / dataset id / local path
    archive_name: str = ''
    expected_bytes: Optional[int] = None
    remote_files: Optional[int] = None
    archive_only: bool = False
    blocked_reason: str = ''
    notes: str = ''
    probe_evidence: str = ''


#: Live-checked access plans (2026-10-24).  Evidence is recorded even when the
#: answer is "blocked": an unreachable host is a fact, not a missing value.
ACCESS: Dict[str, AccessPlan] = {
    # ---- Kaggle public dataset API: anonymous download verified (HTTP 206) ----
    'agrifm_ref1': AccessPlan('kaggle_dataset', 'voglinio/wheat-dataset-original',
                              'ref1_wheat.zip', 546032010, 670,
                              notes='670 wheat-spike images; spike annotations.'),
    'agrifm_ref4': AccessPlan('kaggle_dataset', 'benediktgeisler/perrenial-plants-detection',
                              'ref4_perennial_plants.zip', 5577501925, 392, archive_only=True,
                              notes='Kaggle licence field says "Other" while the Stage 5A ledger says MIT; licence conflict -> ARCHIVE_ONLY for strict corpus.'),
    'agrifm_ref6': AccessPlan('kaggle_dataset', 'hamedetezadi/leaf-image-dataset',
                              'ref6_leaf_images.zip', 14105730, 900,
                              notes='Leaf images, controlled capture.'),
    'agrifm_ref7': AccessPlan('kaggle_dataset', 'osamajalilhassan/agriculture-crops-dataset',
                              'ref7_agriculture_crops.zip', 82830492, 829),
    'agrifm_ref8': AccessPlan('kaggle_dataset', 'devanshujoshi01/maize-images-dataset',
                              'ref8_maize_images.zip', 273842393, 251),
    'agrifm_ref9': AccessPlan('kaggle_dataset', 'nicolaasregnier/pinotnoirgrapes',
                              'ref9_pinot_noir_grapes.zip', 1701005180, 100),
    'agrifm_ref10': AccessPlan('kaggle_dataset', 'clarentjd/classification-maize-and-cocoa-datasets',
                               'ref10_maize_cocoa.zip', 19759726347, 40457),
    'agrifm_ref11': AccessPlan('kaggle_dataset', 'zzzzqi/maize-phenology-images-acquired-using-ndvi-camera',
                               'ref11_maize_ndvi.zip', 6590889394, 2950,
                               notes='NDVI camera; not pure RGB.'),
    # ---- Kaggle endpoints that require an authenticated account ----
    'agrifm_ref15': AccessPlan('kaggle_dsv', '8681789', blocked_reason='Kaggle dsv endpoint returns HTTP 403 without account credentials; ACCESS_REQUIRES_USER_ACTION.'),
    'agrifm_ref29': AccessPlan('kaggle_dsv', '3603983', blocked_reason='Kaggle dsv endpoint returns HTTP 403 without account credentials.'),
    'agrifm_ref35': AccessPlan('kaggle_competition', 'cassava-leaf-disease-classification', blocked_reason='Kaggle competition download returns HTTP 401; competition rules require an authenticated account.'),
    'agrifm_ref37': AccessPlan('kaggle_competition', 'plant-seedlings-classification', blocked_reason='Kaggle competition download returns HTTP 401; competition rules require an authenticated account.'),
    'agrifm_ref75': AccessPlan('kaggle_dsv', '6848682', blocked_reason='Kaggle dsv endpoint returns HTTP 403 without account credentials.'),
    'agrifm_ref76': AccessPlan('kaggle_dsv', '6341289', blocked_reason='Kaggle dsv endpoint returns HTTP 403 without account credentials.'),
    # ---- other hosts ----
    'agrifm_ref2': AccessPlan('url', 'https://eaglenxt.com/data-set/corn-field-mapped-with-ebee-ag-in-brazil/', blocked_reason='Publisher page returns HTTP 403 to automated clients; no direct data URL.'),
    'agrifm_ref3': AccessPlan('url', 'https://eaglenxt.com/data-set/mixed-use-agricultural-fields-ebee-ag/', blocked_reason='Publisher page returns HTTP 403 / connection reset; no direct data URL.'),
    'agrifm_ref5': AccessPlan('url', 'https://zindi.world/competitions/makerere-fall-armyworm-crop-challenge/data', blocked_reason='Zindi competition requires an authenticated account and accepted competition rules.'),
    'agrifm_ref18': AccessPlan('url', 'https://data.acfr.usyd.edu.au/', blocked_reason='Paper page reachable but no machine-readable archive endpoint for the orchard fruit dataset was located; ACCESS_REQUIRES_USER_ACTION.'),
    'agrifm_ref46': AccessPlan('url', 'https://doi.org/10.1007/978-3-031-45725-8_43', blocked_reason='Springer chapter landing page returned HTTP 404 from this network; no data repository link.'),
    'agrifm_ref48': AccessPlan('none', 'KaraAgroAI/Drone-based-Agricultural-Dataset-for-Crop-Yield-Estimation',
                               blocked_reason='HF repository lists only README.md and .gitattributes (no image files); the 14,870 images are not published through the mirror. UAV source, ARCHIVE_ONLY.'),

    'agrifm_ref20': AccessPlan('mendeley', 'gj72pjfvb2', notes='1,081 individual JPEGs; controlled winter-wheat leaf capture.'),
    'agrifm_ref57': AccessPlan('mendeley', 'bwh3zbpkpv', notes='130,472 individual JPEGs across white/dark/illuminated/real backgrounds (paper reports 24,881); image-level domain separation required.'),
    'agrifm_ref58': AccessPlan('dataverse', 'doi:10.7910/DVN/TCKVEW', notes='6 ZIPs (ALS + bean rust + healthy), 15,402 images, CC0-1.0.'),
    'agrifm_ref60': AccessPlan('local_reuse', '', archive_only=True, remote_files=17509,
                               notes='Downstream holdout (DeepWeeds). No verified machine-readable image archive on this network; ARCHIVE_ONLY.'),
    'agrifm_ref65': AccessPlan('url', 'https://doi.org/10.1145/3570991.3571042', blocked_reason='ACM DL returns HTTP 403; Paddy Doctor data are distributed through a competition that needs an account.'),
    'agrifm_ref73': AccessPlan('url', 'https://arxiv.org/abs/2109.11048', blocked_reason='arXiv paper only; no public archive URL in the reference row; data provider page not resolvable.'),
    'agrifm_ref74': AccessPlan('url', 'https://arxiv.org/abs/2110.12331', blocked_reason='arXiv paper only; no public archive URL in the reference row.'),
    'agrifm_ref81': AccessPlan('url', 'https://figshare.com/articles/dataset/13077221', blocked_reason='figshare API/ndownloader returns HTTP 403 from this network.'),
    'agrifm_ref82': AccessPlan('url', 'https://www.webpages.uidaho.edu/vakanski/Multispectral_Images_Dataset.html', blocked_reason='University page now returns HTTP 403 (redirects to uidaho.edu/404.htm).'),
    'agrifm_ref79': AccessPlan('local_reuse', '/data/datasets/Stage5B/weedcrop/extracted',
                               archive_only=True, remote_files=2822,
                               notes='Ronin (Sudars et al. 2020). Full original not reachable; Stage 5B holds a Ronin-derived augmented copy (2,822 files, 1,544 originals). ARCHIVE_ONLY: derived copies cannot be counted as independent captures.'),
    'agrifm_ref90': AccessPlan('mendeley_zip', '8h3s6jkyff', 'ref90_tomato_plant_factory.zip', 3402876899, 520,
                               archive_only=True,
                               notes='Protected cultivation (plant factory); ARCHIVE_ONLY. Verified byte-range download.'),
    'agrifm_ref95': AccessPlan('url', 'https://doi.org/10.3390/s20205893', blocked_reason='MDPI returns HTTP 403; the data-availability statement points to a request-based channel, not an archive.'),
    # ---- local reuse (no re-download) ----
    'agrifm_ref31': AccessPlan('local_reuse', '/data/datasets/GWHD2021/extracted/gwhd_2021', remote_files=6515,
                               archive_only=True, notes='Downstream holdout; already on this machine, SHA-verified in Stage 5A.'),
    'agrifm_ref88': AccessPlan('local_reuse', '/data/datasets/PhenoBench/PhenoBench-v110', remote_files=13767,
                               archive_only=True, notes='Downstream holdout; already on this machine.'),
    'agrifm_ref93': AccessPlan('local_reuse', '/data/datasets/PlantVillage/extracted', remote_files=54304,
                               archive_only=True, notes='Controlled laboratory source; never admitted to R2/R3.'),
}

#: Licences read live from the publisher page (2026-10-24).  ``Unknown`` is
#: recorded as unverified, not silently promoted to a permissive licence.
LICENSE_LIVE: Dict[str, Dict[str, str]] = {
    'agrifm_ref1': {'license': 'CC0-1.0', 'evidence': 'https://www.kaggle.com/datasets/voglinio/wheat-dataset-original (JSON-LD license)'},
    'agrifm_ref6': {'license': 'Unknown', 'evidence': 'https://www.kaggle.com/datasets/hamedetezadi/leaf-image-dataset (JSON-LD license = Unknown)'},
    'agrifm_ref7': {'license': 'Unknown', 'evidence': 'https://www.kaggle.com/datasets/osamajalilhassan/agriculture-crops-dataset (JSON-LD license = Unknown)'},
    'agrifm_ref8': {'license': 'Unknown', 'evidence': 'https://www.kaggle.com/datasets/devanshujoshi01/maize-images-dataset (JSON-LD license = Unknown)'},
    'agrifm_ref9': {'license': 'CC-BY-NC-4.0', 'evidence': 'https://www.kaggle.com/datasets/nicolaasregnier/pinotnoirgrapes (JSON-LD license)'},
    'agrifm_ref10': {'license': 'LGPL-3.0', 'evidence': 'https://www.kaggle.com/datasets/clarentjd/classification-maize-and-cocoa-datasets (JSON-LD license)'},
    'agrifm_ref11': {'license': 'Apache-2.0', 'evidence': 'https://www.kaggle.com/datasets/zzzzqi/maize-phenology-images-acquired-using-ndvi-camera (JSON-LD license)'},
    'agrifm_ref58': {'license': 'CC0-1.0', 'evidence': 'Dataverse API licence field (CC0 1.0)'},
    'agrifm_ref57': {'license': 'CC-BY-4.0', 'evidence': 'https://data.mendeley.com/datasets/bwh3zbpkpv/1 data_licence'},
    'agrifm_ref20': {'license': 'CC-BY-4.0', 'evidence': 'https://data.mendeley.com/datasets/gj72pjfvb2/1 data_licence'},
}

#: Metadata that the Stage 5A ledger does not carry.
EXTRA_META: Dict[str, Dict[str, str]] = {
    'agrifm_ref1': {'crop': 'wheat', 'country': 'unknown', 'platform': 'handheld'},
    'agrifm_ref4': {'crop': 'perennial plants', 'country': 'unknown', 'platform': 'unknown'},
    'agrifm_ref6': {'crop': 'leaf (multi-species)', 'country': 'unknown', 'platform': 'handheld'},
    'agrifm_ref7': {'crop': 'mixed crops', 'country': 'unknown', 'platform': 'unknown'},
    'agrifm_ref8': {'crop': 'maize', 'country': 'unknown', 'platform': 'unknown'},
    'agrifm_ref9': {'crop': 'grape', 'country': 'unknown', 'platform': 'handheld'},
    'agrifm_ref10': {'crop': 'maize; cocoa', 'country': 'unknown', 'platform': 'unknown'},
    'agrifm_ref11': {'crop': 'maize', 'country': 'unknown', 'platform': 'NDVI camera'},
    'agrifm_ref48': {'crop': 'cashew; cocoa; coffee', 'country': 'unknown', 'platform': 'UAV'},
    'agrifm_ref57': {'crop': 'cashew; cassava; maize; tomato', 'country': 'unknown', 'platform': 'handheld'},
    'agrifm_ref58': {'crop': 'bean', 'country': 'Uganda', 'platform': 'handheld'},
    'agrifm_ref90': {'crop': 'tomato', 'country': 'Japan', 'platform': 'fixed camera'},
    'agrifm_ref88': {'crop': 'sugar beet; weed', 'country': 'Germany', 'platform': 'ground robot'},
    'agrifm_ref31': {'crop': 'wheat', 'country': 'multi-country', 'platform': 'handheld'},
    'agrifm_ref93': {'crop': 'multi-species', 'country': 'controlled', 'platform': 'lab'},
}


def _read_registry() -> Dict[str, Dict[str, str]]:
    with open(REGISTRY_CSV, newline='', encoding='utf-8') as handle:
        return {row['source_id']: row for row in csv.DictReader(handle)}


def physical_source_id(ref: str) -> str:
    return SAME_PHYSICAL_DATASET.get(ref, ref)


def _int(value: Optional[str]) -> Optional[int]:
    if value in (None, ''):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


@dataclass
class SourceRow:
    physical_source_id: str
    source_display_name: str
    original_ref_ids: List[str]
    groups: List[str]
    paper_url: str = ''
    official_data_url: str = ''
    original_version: str = ''
    remote_version: str = ''
    modality: str = 'RGB'
    capture_domain: str = 'UNKNOWN'
    crop_category: str = ''
    disease_weed_fruit_tags: str = ''
    country_region: str = ''
    platform: str = ''
    license_id: str = 'unverified'
    license_evidence: str = ''
    attribution: str = ''
    redistribution_allowed: str = 'unknown'
    access_status: str = 'DISCOVERED'
    download_status: str = 'NOT_STARTED'
    archive_sha256: str = ''
    reported_images: Optional[int] = None
    remotely_indexed_images: Optional[int] = None
    downloaded_images: int = 0
    decoded_images: int = 0
    basic_qc_pass: int = 0
    iqa_pass: int = 0
    domain_pass: int = 0
    exact_dedup_removed: int = 0
    near_dedup_removed: int = 0
    holdout_removed: int = 0
    license_blocked: int = 0
    quarantined: int = 0
    final_accepted: int = 0
    split_train: int = 0
    split_val: int = 0
    group_count: int = 0
    final_status: str = 'NOT_BUILT'
    failure_reason: str = ''
    access_method: str = ''
    access_ref: str = ''
    archive_only: bool = False
    holdout: bool = False
    local_root: str = ''
    notes: str = ''

    def as_dict(self) -> Dict[str, object]:
        data = dict(self.__dict__)
        data['original_ref_ids'] = ';'.join(self.original_ref_ids)
        data['groups'] = ';'.join(self.groups)
        data['archive_only'] = 'yes' if self.archive_only else 'no'
        data['holdout'] = 'yes' if self.holdout else 'no'
        return data


def thirty_five() -> List[SourceRow]:
    """Build the 35-row Table 1 registry, merging physical duplicates."""
    registry = _read_registry()
    missing = [r for r in REF_IDS if r not in registry]
    if missing:
        raise KeyError(f'registry is missing reference rows: {missing}')

    merged: Dict[str, SourceRow] = {}
    order: List[str] = []
    for ref in REF_IDS:
        base = registry[ref]
        physical = physical_source_id(ref)
        plan = ACCESS.get(ref, AccessPlan('none', blocked_reason='no Stage 5C access plan recorded'))
        extra = EXTRA_META.get(physical) or EXTRA_META.get(ref, {})
        holdout = physical in HOLDOUT_IDS
        if physical not in merged:
            merged[physical] = SourceRow(
                physical_source_id=physical,
                source_display_name=base['official_name'],
                original_ref_ids=[ref],
                groups=['AGRI_FM35'],
                paper_url=base.get('paper_url', ''),
                official_data_url=base.get('download_url', ''),
                original_version=base.get('year', ''),
                modality=base.get('modality', 'RGB') or 'RGB',
                capture_domain=base.get('acquisition_domain', 'UNKNOWN') or 'UNKNOWN',
                crop_category=extra.get('crop', ''),
                country_region=extra.get('country', 'unknown'),
                platform=extra.get('platform', base.get('platform', '')),
                license_id=(LICENSE_LIVE.get(ref, {}).get('license')
                            or base.get('license', 'unverified') or 'unverified'),
                license_evidence=(LICENSE_LIVE.get(ref, {}).get('evidence')
                                  or base.get('license_evidence', '')),
                attribution=base.get('attribution_required', ''),
                redistribution_allowed='yes' if base.get('license', '').startswith(('CC', 'MIT', 'CC0')) else 'unknown',
                reported_images=_int(base.get('images_reported')),
                remotely_indexed_images=plan.remote_files,
                access_method=plan.method,
                access_ref=plan.ref,
                archive_only=plan.archive_only or holdout,
                holdout=holdout,
                local_root=plan.ref if plan.method == 'local_reuse' else '',
                notes=plan.notes,
                failure_reason=plan.blocked_reason,
                access_status='BLOCKED' if plan.method in ('none',) or plan.blocked_reason else 'URL_CHECKED',
                download_status='BLOCKED' if plan.blocked_reason and plan.method not in ('local_reuse',) else ('DOWNLOADED_FULL' if plan.method == 'local_reuse' else 'NOT_STARTED'),
            )
            order.append(physical)
        else:
            row = merged[physical]
            row.original_ref_ids.append(ref)
            if plan.remote_files:
                row.remotely_indexed_images = max(row.remotely_indexed_images or 0, plan.remote_files)
            if not row.failure_reason and plan.blocked_reason:
                row.failure_reason = plan.blocked_reason
        if holdout:
            merged[physical].holdout = True
            merged[physical].archive_only = True
    return [merged[k] for k in order]


EXTENSION_SOURCES: List[Dict[str, object]] = [
    {
        'physical_source_id': 'artemis_imagesafari',
        'source_display_name': 'Artemis and Image Safari crop imagery (AWS Open Data)',
        'original_ref_ids': [],
        'groups': ['EXTENSION'],
        'paper_url': 'https://doi.org/10.31220/agriRxiv.2026.00491',
        'official_data_url': 'https://registry.opendata.aws/artemis-image-safari/',
        'original_version': '2026-09 snapshot',
        'modality': 'RGB',
        'capture_domain': 'FIELD_IN_SITU',
        'crop_category': 'bean; cowpea; sorghum; soybean; maize; cassava; wheat; potato; rice; banana; yam; millet; groundnut; lentil; chickpea; sweet potato',
        'country_region': 'sub-Saharan Africa; Latin America; United States',
        'platform': 'handheld; ground phenotyping',
        'license_id': 'CC-BY-SA-4.0',
        'license_evidence': 'https://registry.opendata.aws/artemis-image-safari/ (license field)',
        'attribution': 'Alliance of Bioversity International and CIAT; cite Mutuvi et al. agriRxiv 2026',
        'redistribution_allowed': 'yes-share-alike',
        'reported_images': 7469218,
        'access_method': 's3_prefix',
        'access_ref': 's3://alliance-artemis-imagesafari/',
        'notes': 'Sampled, not complete: PARTIAL_SOURCE_DOWNLOAD by design.',
    },
    {
        'physical_source_id': 'agrifield40k_pool',
        'source_display_name': 'AgriField-40K curated pool (local copy)',
        'original_ref_ids': [],
        'groups': ['EXTENSION', 'AGRIFIELD40K'],
        'paper_url': 'https://arxiv.org/abs/2608.07984',
        'official_data_url': 'https://dtu-pas.github.io/agrifield40k/',
        'original_version': 'AgriField-40K (2026)',
        'modality': 'RGB',
        'capture_domain': 'MIXED',
        'crop_category': 'multi-crop',
        'country_region': 'multi-country',
        'platform': 'handheld; robot; UAV subset',
        'license_id': 'CC-BY-4.0',
        'license_evidence': 'pinned dataset card dtu-pcas/AgriField-40K@7f530c6',
        'attribution': 'Tzouras, Pegios, Nalpantidis (2026), arXiv:2608.07984',
        'redistribution_allowed': 'yes',
        'reported_images': 39551,
        'access_method': 'local_reuse',
        'access_ref': '/data/datasets/AgriField-40K/agrifield40k',
        'notes': 'Derived pool: images retain upstream identity via filename prefix; PhenoBench and other holdout prefixes are excluded.',
    },
    {
        'physical_source_id': 'sesame_weed',
        'source_display_name': 'Crop and Weed Detection Data with Bounding Boxes (Sesame)',
        'original_ref_ids': [],
        'groups': ['EXTENSION'],
        'paper_url': 'https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
        'official_data_url': 'https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes',
        'original_version': '2023',
        'modality': 'RGB',
        'capture_domain': 'FIELD_IN_SITU',
        'crop_category': 'sesame',
        'country_region': 'India',
        'platform': 'handheld',
        'license_id': 'CC0-1.0',
        'license_evidence': 'Kaggle dataset page licence field',
        'attribution': 'Ravirajsinh Dabhi (Kaggle), CC0',
        'redistribution_allowed': 'yes',
        'reported_images': 1300,
        'access_method': 'local_reuse',
        'access_ref': '/data/datasets/Stage5B/sesame_weed/extracted/agri_data/data',
        'notes': 'Stage 5B download; 1,300 images are byte-identical to agrifield40k saw_* rows.',
    },
    {
        'physical_source_id': 'applebbch81',
        'source_display_name': 'AppleBBCH81 (orchard apple fruit, BBCH 81-85)',
        'original_ref_ids': [],
        'groups': ['EXTENSION'],
        'paper_url': 'https://doi.org/10.3390/drones8120734',
        'official_data_url': 'https://www.kaggle.com/datasets/projectlzp201910094/applebbch81',
        'original_version': '2024',
        'modality': 'RGB',
        'capture_domain': 'ORCHARD_IN_SITU',
        'crop_category': 'apple',
        'country_region': 'Latvia',
        'platform': 'handheld',
        'license_id': 'CC-BY-4.0',
        'license_evidence': 'Kaggle dataset page licence field; Kodors et al. 2024',
        'attribution': 'Kodors et al. (2024), Drones 8(12):734',
        'redistribution_allowed': 'yes',
        'reported_images': 1838,
        'access_method': 'local_reuse',
        'access_ref': '/data/datasets/Stage5B/applebbch81/extracted',
        'notes': 'Stage 5B download; 140 trees / 150 sessions.',
    },
    {
        'physical_source_id': 'user_owned',
        'source_display_name': 'USER_OWNED (interface only, no images supplied)',
        'original_ref_ids': [],
        'groups': ['USER_OWNED'],
        'official_data_url': '',
        'modality': 'RGB',
        'capture_domain': 'UNKNOWN',
        'license_id': 'unknown',
        'redistribution_allowed': 'unknown',
        'reported_images': 0,
        'access_method': 'none',
        'access_ref': '',
        'notes': 'No user-owned images exist on this machine; the metadata template is defined in data/corpus_v2/user_owned_template.csv. No count is fabricated.',
    },
]

SOURCE_FIELDS: List[str] = list(SourceRow('', '', [], []).as_dict().keys())


def all_rows() -> List[SourceRow]:
    rows = thirty_five()
    for spec in EXTENSION_SOURCES:
        row = SourceRow(**spec)
        if row.access_method == 'local_reuse' and row.access_ref:
            row.local_root = row.access_ref
        rows.append(row)
    return rows


def write_sources_csv(path: Path, rows: Optional[List[SourceRow]] = None) -> Path:
    rows = rows or all_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def coverage_report(rows: Optional[List[SourceRow]] = None) -> Dict[str, object]:
    rows = rows or all_rows()
    covered = {ref for row in rows for ref in row.original_ref_ids}
    return {
        'reference_ids_expected': len(REF_IDS),
        'reference_ids_covered': len(covered & set(REF_IDS)),
        'missing': sorted(set(REF_IDS) - covered),
        'extra': sorted(covered - set(REF_IDS)),
        'physical_sources': len([r for r in rows if r.groups and r.physical_source_id != 'user_owned']),
        'blocked': sorted(r.physical_source_id for r in rows if r.download_status == 'BLOCKED'),
        'archive_only': sorted(r.physical_source_id for r in rows if r.archive_only),
    }


if __name__ == '__main__':
    out = write_sources_csv(ROOT / 'data/corpus_v2/sources.csv')
    print(json.dumps(coverage_report(), indent=2, ensure_ascii=False))
    print(out)
