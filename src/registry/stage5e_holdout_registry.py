"""Stage 5E holdout registry.

Every downstream evaluation set is registered as one or more *concrete
entities*: a dataset at a specific version or release.  Stage 5D carried a
single ``downstream_holdout_sources.csv`` row per label; "Roboflow series" was
one row standing for several different datasets and versions, which is not a
verifiable claim.

Each entity below records only what was actually reachable from this machine.
When a host refused the connection the status is one of ``BLOCKED_*`` /
``INDEX_ONLY`` / ``UNRESOLVED_FILE_LEVEL`` - never ``VERIFIED``.

Status vocabulary
-----------------
``FILE_LEVEL_INDEXED``
    The image bytes were obtained and hashed.  Content-level cross-checking
    against the pretraining corpus is real evidence for these files.
``INDEX_ONLY``
    An official file list / hash manifest exists, but the image bytes were not
    retrieved.  No content-level claim is possible.
``BLOCKED_AUTH``
    Requires an account, token, or an accepted terms-of-use click-through.
``BLOCKED_NETWORK``
    DNS / TCP / TLS failure - the host is unreachable from this machine.
``BLOCKED_HOST_REJECTED``
    The host answered, but refused the automated request (403 etc.).
``UNRESOLVED_FILE_LEVEL``
    Registered for evaluation but no file-level evidence could be produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

FILE_LEVEL_INDEXED = 'FILE_LEVEL_INDEXED'
INDEX_ONLY = 'INDEX_ONLY'
BLOCKED_AUTH = 'BLOCKED_AUTH'
BLOCKED_NETWORK = 'BLOCKED_NETWORK'
BLOCKED_HOST_REJECTED = 'BLOCKED_HOST_REJECTED'
UNRESOLVED_FILE_LEVEL = 'UNRESOLVED_FILE_LEVEL'

STATUSES = (FILE_LEVEL_INDEXED, INDEX_ONLY, BLOCKED_AUTH, BLOCKED_NETWORK,
            BLOCKED_HOST_REJECTED, UNRESOLVED_FILE_LEVEL)

#: Statuses for which a content-level (hash/pHash) claim may be published.
CONTENT_VERIFIED_STATUSES = (FILE_LEVEL_INDEXED,)


@dataclass(frozen=True)
class HoldoutEntity:
    dataset_id: str
    dataset_name: str
    version: str
    publisher: str
    task: str
    licence: str
    official_url: str
    licence_evidence_url: str
    access_url: str
    status: str
    observed: str
    images_expected: str
    files_indexed: int = 0
    files_hashed: int = 0
    notes: str = ''
    mirror_urls: List[str] = field(default_factory=list)

    def as_row(self) -> Dict[str, object]:
        row = {k: v for k, v in self.__dict__.items()}
        row['mirror_urls'] = ';'.join(self.mirror_urls)
        row['content_verified'] = self.status in CONTENT_VERIFIED_STATUSES
        return row


#: Registry entries.  ``status`` / ``observed`` are filled by the indexer at
#: runtime; the values below are the registration-time facts only.
ENTITIES: List[HoldoutEntity] = [
    HoldoutEntity(
        dataset_id='phenobench',
        dataset_name='PhenoBench',
        version='v1.1.0',
        publisher='University of Bonn (PhenoRob)',
        task='semantic segmentation',
        licence='CC-BY-SA-4.0',
        official_url='https://www.phenobench.org/',
        licence_evidence_url='https://www.phenobench.org/',
        access_url='https://www.phenobench.org/',
        status=FILE_LEVEL_INDEXED,
        observed='local copy /data/datasets/PhenoBench/PhenoBench-v110',
        images_expected='13,767',
        notes='Present on disk from an earlier stage; index reused.',
    ),
    HoldoutEntity(
        dataset_id='gwhd_2021',
        dataset_name='Global Wheat Head Dataset 2021',
        version='2021',
        publisher='University of Saskatchewan / CGIAR',
        task='object detection',
        licence='CC-BY-4.0',
        official_url='https://zenodo.org/records/5092304',
        licence_evidence_url='https://zenodo.org/records/5092304',
        access_url='https://zenodo.org/records/5092304',
        status=FILE_LEVEL_INDEXED,
        observed='local copy /data/datasets/GWHD2021/extracted',
        images_expected='6,515',
        notes='Present on disk from an earlier stage; index reused.',
    ),
    HoldoutEntity(
        dataset_id='plantdoc',
        dataset_name='PlantDoc',
        version='original (test + train, 2,579 images)',
        publisher='pratikkayal (GitHub mirror of the PlantDoc dataset)',
        task='object detection / image classification',
        licence='CC-BY-4.0',
        official_url='https://github.com/pratikkayal/PlantDoc-Dataset',
        licence_evidence_url='https://github.com/pratikkayal/PlantDoc-Dataset/blob/master/LICENSE.txt',
        access_url='https://codeload.github.com/pratikkayal/PlantDoc-Dataset/tar.gz/refs/heads/master',
        status=FILE_LEVEL_INDEXED,
        observed=('tarball downloaded and extracted; 2,579 images decoded and hashed from real '
                  'image bytes'),
        images_expected='2,579',
        notes=('Stage 5E first observed a GitHub Tree API blob list (2,579 blobs) while the '
               'tarball download was incomplete; a git blob SHA is not a content hash, so that '
               'listing was never treated as file-level evidence. The full download later '
               'completed and every image was decoded and hashed here.'),
        mirror_urls=['https://universe.roboflow.com/ (redirects; 403 to automated requests)'],
    ),
    HoldoutEntity(
        dataset_id='minneapple',
        dataset_name='MinneApple',
        version='v1 (official release)',
        publisher='University of Minnesota Digital Conservancy',
        task='object detection / instance segmentation',
        licence='research use (repository terms)',
        official_url='https://conservancy.umn.edu/handle/11299/206813',
        licence_evidence_url='https://conservancy.umn.edu/handle/11299/206813',
        access_url='https://conservancy.umn.edu/handle/11299/206813',
        status=BLOCKED_HOST_REJECTED,
        observed='HTTP 403 from conservancy.umn.edu (automated request refused)',
        images_expected='~1,000',
        notes='Official landing page reachable only via a browser session. No file list obtained.',
        mirror_urls=['https://github.com/nicolaihaeni/MinneApple (MIT code repo, not the images)'],
    ),
    HoldoutEntity(
        dataset_id='plantseg',
        dataset_name='PlantSeg',
        version='current',
        publisher='gitlab.com/sauthier',
        task='semantic segmentation',
        licence='see repository',
        official_url='https://gitlab.com/sauthier/plant_seg',
        licence_evidence_url='https://gitlab.com/sauthier/plant_seg',
        access_url='https://gitlab.com/sauthier/plant_seg',
        status=BLOCKED_AUTH,
        observed='HTTP 403 -> redirect to gitlab.com/users/sign_in (repository requires sign-in)',
        images_expected='unknown',
        notes='Repository visibility requires an authenticated GitLab account. Not bypassed.',
        mirror_urls=['https://github.com/tqwei05/PlantSeg (79 stars, licence not declared)'],
    ),
    HoldoutEntity(
        dataset_id='deepweeds',
        dataset_name='DeepWeeds',
        version='2018 (original release)',
        publisher='Alex Olsen / James Cook University',
        task='weed classification',
        licence='CC BY 4.0',
        official_url='https://github.com/AlexOlsen/DeepWeeds',
        licence_evidence_url='https://github.com/AlexOlsen/DeepWeeds#readme',
        access_url='https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj',
        status=UNRESOLVED_FILE_LEVEL,
        observed='README reachable (200); images hosted on Google Drive, request timed out',
        images_expected='17,509',
        notes=('Licence confirmed from the repository README ("images and annotations are licensed '
               'under CC BY 4.0"). Images are a 468 MB Google Drive archive; the download did not '
               'finish within the session budget, so no image bytes were hashed.'),
        mirror_urls=['https://www.tensorflow.org/datasets/catalog/deep_weeds (TFDS, blocked here)'],
    ),
    HoldoutEntity(
        dataset_id='growliflower',
        dataset_name='GrowliFlower',
        version='current',
        publisher='Natalia Gonzalez / Zenodo',
        task='semantic segmentation',
        licence='CC-BY-4.0',
        official_url='https://zenodo.org/records/4744656',
        licence_evidence_url='https://zenodo.org/records/4744656',
        access_url='https://zenodo.org/records/4744656',
        status=BLOCKED_NETWORK,
        observed='zenodo.org connection refused (curl exit 7)',
        images_expected='~1,400',
        notes='Host unreachable from this machine; licence is the registry record, not verified here.',
        mirror_urls=['https://github.com/nataliag7/GrowliFlower (no licence declared)'],
    ),
    HoldoutEntity(
        dataset_id='cropandweed',
        dataset_name='CropAndWeed',
        version='WACV 2023 release',
        publisher='vitro-testing.com (dataset host of the WACV 2023 authors)',
        task='semantic segmentation',
        licence='CC-BY-SA-4.0',
        official_url='https://github.com/cropandweed/cropandweed-dataset',
        licence_evidence_url='https://github.com/cropandweed/cropandweed-dataset/blob/main/LICENCE',
        access_url='https://vitro-testing.com/wp-content/uploads/2022/12/cropandweed_images1of4.tar',
        status=UNRESOLVED_FILE_LEVEL,
        observed='image tars 2.98 GB each (4 shards); annotations tar downloaded',
        images_expected='~30,000',
        notes=('Repository reachable, annotations archive downloaded. Image shards are 2.98 GB each '
               '(~12 GB total); the full image download was not attempted in this session, so no '
               'image bytes were hashed.'),
        mirror_urls=[],
    ),
    HoldoutEntity(
        dataset_id='multiclass_weeds',
        dataset_name='Multiclass Weeds Dataset',
        version='figshare record 11924543',
        publisher='figshare',
        task='semantic segmentation',
        licence='CC-BY-4.0 (figshare record)',
        official_url='https://figshare.com/articles/dataset/Multiclass_weeds_dataset/11924543',
        licence_evidence_url='https://figshare.com/articles/dataset/Multiclass_weeds_dataset/11924543',
        access_url='https://api.figshare.com/v2/articles/11924543',
        status=BLOCKED_HOST_REJECTED,
        observed='figshare landing page HTTP 202; api.figshare.com HTTP 403 to automated requests',
        images_expected='~17,000',
        notes='Record page loads in a browser; the API and file list are refused to automated clients.',
    ),
    HoldoutEntity(
        dataset_id='roboflow_disease_detection',
        dataset_name='Roboflow - Detecting Diseases (Plant Disease Detection)',
        version='registry row (unversioned)',
        publisher='Roboflow Universe',
        task='object detection',
        licence='CC-BY-4.0 (Roboflow dataset page)',
        official_url='https://universe.roboflow.com/',
        licence_evidence_url='https://universe.roboflow.com/',
        access_url='https://public.roboflow.com/detection/plantdoc/1/download',
        status=UNRESOLVED_FILE_LEVEL,
        observed='public.roboflow.com redirected to universe.roboflow.com (403)',
        images_expected='unknown',
        notes=('Stage 5D registered "Roboflow series" as one row. The registry does not name a '
               'specific workspace/project/version, so this is not a verifiable dataset identity.'),
    ),
    HoldoutEntity(
        dataset_id='roboflow_crop_detection',
        dataset_name='Roboflow - Crops (maize, cassava, sugarcane vs grass)',
        version='registry row (unversioned)',
        publisher='Roboflow Universe',
        task='object detection',
        licence='CC-BY-4.0 (Roboflow dataset page)',
        official_url='https://universe.roboflow.com/',
        licence_evidence_url='https://universe.roboflow.com/',
        access_url='https://universe.roboflow.com/',
        status=UNRESOLVED_FILE_LEVEL,
        observed='universe.roboflow.com HTTP 403 to automated requests',
        images_expected='unknown',
        notes=('Same registry problem as roboflow_disease_detection: the row names a description, '
               'not a dataset id and version.'),
    ),
    HoldoutEntity(
        dataset_id='gwhd_instance_seg',
        dataset_name='GWHD Instance Segmentation (Roboflow wheat-seg)',
        version='registry row (unversioned)',
        publisher='Roboflow Universe',
        task='instance segmentation',
        licence='CC-BY-4.0 (Roboflow dataset page)',
        official_url='https://universe.roboflow.com/',
        licence_evidence_url='https://universe.roboflow.com/',
        access_url='https://universe.roboflow.com/',
        status=UNRESOLVED_FILE_LEVEL,
        observed='universe.roboflow.com HTTP 403 to automated requests',
        images_expected='unknown',
        notes=('Derived from GWHD 2021; the Roboflow project id and version are not recorded in '
               'the registry, so the exact entity cannot be identified.'),
    ),
]


def by_id() -> Dict[str, HoldoutEntity]:
    return {entity.dataset_id: entity for entity in ENTITIES}


def rows() -> List[Dict[str, object]]:
    return [entity.as_row() for entity in ENTITIES]


def status_summary() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entity in ENTITIES:
        counts[entity.status] = counts.get(entity.status, 0) + 1
    return counts
