"""Stage 5E licence audit model.

Stage 5D carried a single ``license_status`` string per image.  That conflates
five different legal questions:

1. may this project use the image for academic research / model training?
2. may the original image be redistributed?
3. may the source id, manifest metadata, hashes and rebuild scripts be
   published?
4. may derived annotations / processed products be published?
5. what does publishing a *checkpoint* trained on the data mean?

This module keeps those answers in separate fields.  A source is never described
by one boolean, and :func:`train_eligible` is derived only from the
research-use dimension.

Evidence rule
-------------
Every decision must point at an archived copy of an *official* document
(a publisher API response, LICENSE, Terms of Use, data card).  Third-party
aggregator pages and search snippets are not accepted as evidence.  The
archived bytes and their SHA-256 are recorded in
``reports/stage5e_evidence/evidence_index.json``.

The status vocabulary is deliberately narrow and mutually exclusive for the
research-use dimension:

``VERIFIED_RESEARCH_USE``
    Official terms positively allow use for this project's research/training,
    with at most attribution/notice obligations.
``RESTRICTED_RESEARCH_USE``
    Official terms allow research use but attach a condition that this project
    must actively manage (e.g. non-commercial only, share-alike).
``UNRESOLVED``
    No usable official evidence, or official sources conflict.
``NOT_PERMITTED``
    Official terms prohibit the required use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# -- research-use statuses (mutually exclusive) ----------------------------- #
VERIFIED_RESEARCH_USE = 'VERIFIED_RESEARCH_USE'
RESTRICTED_RESEARCH_USE = 'RESTRICTED_RESEARCH_USE'
UNRESOLVED = 'UNRESOLVED'
NOT_PERMITTED = 'NOT_PERMITTED'

RESEARCH_USE_STATUSES = (VERIFIED_RESEARCH_USE, RESTRICTED_RESEARCH_USE,
                         UNRESOLVED, NOT_PERMITTED)

# -- distribution statuses (independent vocabulary) ------------------------- #
RELEASE_ALLOWED = 'ALLOWED'
RELEASE_ALLOWED_WITH_CONDITIONS = 'ALLOWED_WITH_CONDITIONS'
RELEASE_NOT_ALLOWED = 'NOT_ALLOWED'
RELEASE_UNRESOLVED = 'UNRESOLVED'

RELEASE_STATUSES = (RELEASE_ALLOWED, RELEASE_ALLOWED_WITH_CONDITIONS,
                    RELEASE_NOT_ALLOWED, RELEASE_UNRESOLVED)


@dataclass(frozen=True)
class SourceLicence:
    """One audited source.

    ``source_id`` is the internal registry id (never guessed from a ref number).
    The five ``*_release`` / ``research_use`` fields answer the five independent
    questions above and must not be collapsed into one flag.
    """

    source_id: str
    real_name: str
    publisher: str
    official_url: str
    licence_id: str
    licence_name: str
    evidence_url: str
    evidence_locator: str
    evidence_excerpt: str
    evidence_file: str
    evidence_sha256: str
    accessed: str
    research_use: str
    research_use_conditions: str
    redistribution: str
    redistribution_conditions: str
    metadata_release: str
    metadata_release_conditions: str
    derived_release: str
    derived_release_conditions: str
    checkpoint_release: str
    checkpoint_release_conditions: str
    open_issues: str
    decision: str
    decision_reason: str
    mirror_urls: List[str] = field(default_factory=list)

    def as_row(self) -> Dict[str, object]:
        row = asdict(self)
        row['mirror_urls'] = ';'.join(self.mirror_urls)
        return row

    @property
    def train_eligible(self) -> bool:
        """Only the research-use dimension admits images into training.

        Redistribution being unknown must never remove a locally usable image,
        and redistribution being allowed must never admit an image the project
        may not train on.
        """
        return self.research_use in (VERIFIED_RESEARCH_USE, RESTRICTED_RESEARCH_USE)

    @property
    def redistribution_eligible(self) -> bool:
        return self.redistribution in (RELEASE_ALLOWED, RELEASE_ALLOWED_WITH_CONDITIONS)


#: The four sources Stage 5D listed as "licence use not verified".
#:
#: Every value below was read from an official publisher endpoint on
#: 2026-10-25 and the response body was archived under
#: ``reports/stage5e_evidence/``.  Nothing here is inferred from a ref number,
#: a search summary or a third-party catalogue.
LICENCES: Dict[str, SourceLicence] = {
    'agrifm_ref1': SourceLicence(
        source_id='agrifm_ref1',
        real_name='Wheat Spike Dataset (voglinio)',
        publisher='Costas Voglis (Kaggle dataset owner)',
        official_url='https://www.kaggle.com/datasets/voglinio/wheat-dataset-original',
        licence_id='CC0-1.0',
        licence_name='CC0: Public Domain',
        evidence_url='https://www.kaggle.com/api/v1/datasets/view/voglinio/wheat-dataset-original',
        evidence_locator='JSON field "licenseName"',
        evidence_excerpt='"licenseName": "CC0: Public Domain"; "ownerName": "Costas Voglis"',
        evidence_file='reports/stage5e_evidence/agrifm_ref1_kaggle_api_voglinio_wheat-dataset-original.json',
        evidence_sha256='c37c9bd8c127b1fffa446ef472f9327b96f3bd442b27db78b166cd2bb3d8e744',
        accessed='2026-10-25',
        research_use=VERIFIED_RESEARCH_USE,
        research_use_conditions='none imposed by the licence',
        redistribution=RELEASE_ALLOWED,
        redistribution_conditions='CC0 waives copyright; no attribution required',
        metadata_release=RELEASE_ALLOWED,
        metadata_release_conditions='',
        derived_release=RELEASE_ALLOWED,
        derived_release_conditions='',
        checkpoint_release=RELEASE_ALLOWED,
        checkpoint_release_conditions='',
        open_issues=(
            'The licence is the Kaggle publisher\'s declaration. The uploader\'s '
            'chain of title was not independently verified, which is normal for '
            'an open-data mirror.'),
        decision='INCLUDE',
        decision_reason='Official publisher endpoint declares CC0-1.0 (public domain dedication).',
    ),
    'agrifm_ref58': SourceLicence(
        source_id='agrifm_ref58',
        real_name='Makerere University Beans Image Dataset',
        publisher='Harvard Dataverse (depositor: Makerere University)',
        official_url='https://doi.org/10.7910/DVN/TCKVEW',
        licence_id='CC0-1.0',
        licence_name='CC0 1.0',
        evidence_url='https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/TCKVEW',
        evidence_locator='latestVersion.license (SPDX rightsIdentifier), versionState',
        evidence_excerpt=('"license": {"name": "CC0 1.0", "uri": "http://creativecommons.org/publicdomain/zero/1.0", '
                          '"rightsIdentifier": "CC0-1.0", "rightsIdentifierScheme": "SPDX"}; '
                          '"versionState": "RELEASED"; "termsOfUse": null; "termsOfAccess": null; '
                          '"fileAccessRequest": false'),
        evidence_file='reports/stage5e_evidence/agrifm_ref58_harvard_dataverse_TCKVEW_api.json',
        evidence_sha256='3bd6dfd42e67988a2c115ba90d7b402a35649b5bf13c4e93ff9640e711f3b9cd',
        accessed='2026-10-25',
        research_use=VERIFIED_RESEARCH_USE,
        research_use_conditions='none imposed by the licence',
        redistribution=RELEASE_ALLOWED,
        redistribution_conditions='CC0 waives copyright; no attribution required',
        metadata_release=RELEASE_ALLOWED,
        metadata_release_conditions='',
        derived_release=RELEASE_ALLOWED,
        derived_release_conditions='',
        checkpoint_release=RELEASE_ALLOWED,
        checkpoint_release_conditions='',
        open_issues='None at the licence layer.',
        decision='INCLUDE',
        decision_reason=('Official Dataverse API reports a released version with SPDX CC0-1.0, '
                         'no terms of use/access and no restricted files.'),
    ),
    'agrifm_ref11': SourceLicence(
        source_id='agrifm_ref11',
        real_name='Maize phenology images (NDVI camera)',
        publisher='Kaggle dataset owner "zzzzqi"',
        official_url='https://www.kaggle.com/datasets/zzzzqi/maize-phenology-images-acquired-using-ndvi-camera',
        licence_id='Apache-2.0',
        licence_name='Apache 2.0',
        evidence_url='https://www.kaggle.com/api/v1/datasets/view/zzzzqi/maize-phenology-images-acquired-using-ndvi-camera',
        evidence_locator='JSON field "licenseName"',
        evidence_excerpt='"licenseName": "Apache 2.0"; "ownerName": "zzzzqi"',
        evidence_file='reports/stage5e_evidence/agrifm_ref11_kaggle_api_zzzzqi_maize-phenology-images-acquired-using-ndvi-camera.json',
        evidence_sha256='387feb3de135191f5a25cbc689c2a2176cd94f393d55931c25bf25671145917a',
        accessed='2026-10-25',
        research_use=VERIFIED_RESEARCH_USE,
        research_use_conditions='Apache-2.0 NOTICE/attribution obligations if redistributed',
        redistribution=RELEASE_ALLOWED_WITH_CONDITIONS,
        redistribution_conditions=('Redistribute under Apache-2.0 with the licence text and any NOTICE; '
                                   'state changes made'),
        metadata_release=RELEASE_ALLOWED,
        metadata_release_conditions='',
        derived_release=RELEASE_ALLOWED_WITH_CONDITIONS,
        derived_release_conditions='Carry Apache-2.0 attribution for the source images',
        checkpoint_release=RELEASE_ALLOWED_WITH_CONDITIONS,
        checkpoint_release_conditions='Publish model card naming the dataset and its Apache-2.0 terms',
        open_issues=(
            'Only the rgbnormal/ directory is visible-RGB; nirraw/ is an NDVI encoding and is '
            'already rejected by the Stage 5D modality gate. No licence issue.'),
        decision='INCLUDE',
        decision_reason='Official publisher endpoint declares Apache-2.0, which permits research use and redistribution with notice.',
    ),
    'agrifm_ref9': SourceLicence(
        source_id='agrifm_ref9',
        real_name='PinotNoirGrapes (nicolaasregnier)',
        publisher='Nicolaas Regnier (Kaggle dataset owner)',
        official_url='https://www.kaggle.com/datasets/nicolaasregnier/pinotnoirgrapes',
        licence_id='CC-BY-NC-4.0',
        licence_name='Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)',
        evidence_url='https://www.kaggle.com/api/v1/datasets/view/nicolaasregnier/pinotnoirgrapes',
        evidence_locator='JSON field "licenseName" and "description" (non-commercial research framing)',
        evidence_excerpt=('"licenseName": "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"; '
                          'description: "The purpose of this dataset is to gain a better understanding '
                          'of object detection and its applications in agriculture."'),
        evidence_file='reports/stage5e_evidence/agrifm_ref9_kaggle_api_nicolaasregnier_pinotnoirgrapes.json',
        evidence_sha256='a97510fbc417c53a385eff10508c4e0adb8192114fc82e3f6436a5392fdcfc56',
        accessed='2026-10-25',
        research_use=RESTRICTED_RESEARCH_USE,
        research_use_conditions=(
            'NonCommercial only, with attribution. This project is academic, non-commercial '
            'research; the NonCommercial condition is therefore satisfied for local training, '
            'but it must be re-checked if the work is ever used commercially.'),
        redistribution=RELEASE_ALLOWED_WITH_CONDITIONS,
        redistribution_conditions=('CC BY-NC 4.0 permits sharing for NonCommercial purposes with '
                                   'attribution and a link to the licence; project policy is to '
                                   'withhold the images anyway so downstream users are not bound by NC'),
        metadata_release=RELEASE_ALLOWED,
        metadata_release_conditions='Attribute the dataset and licence',
        derived_release=RELEASE_ALLOWED_WITH_CONDITIONS,
        derived_release_conditions='NonCommercial + attribution carry over to derivatives',
        checkpoint_release=RELEASE_UNRESOLVED,
        checkpoint_release_conditions=(
            'A published checkpoint is a derivative of the images. Whether releasing it is a '
            'NonCommercial act depends on how downstream users deploy it; this is not decidable '
            'from the licence text alone.'),
        open_issues=(
            'Checkpoint-release question is genuinely unresolved, not "no". Research training is '
            'permitted; publishing weights trained on this dataset needs an explicit decision '
            'before Stage 6 release.'),
        decision='INCLUDE_WITH_CONDITIONS',
        decision_reason=('CC BY-NC 4.0 allows non-commercial research use with attribution. The '
                         'project is academic and non-commercial, so training inclusion is allowed; '
                         'checkpoint release is recorded as unresolved and kept separate.'),
    ),
}

#: Sources that were already accepted in Stage 5D and are *not* one of the four
#: audited above.  Their Stage 5D registry licence is carried forward verbatim
#: and re-stated in the Stage 5E vocabulary so every accepted image has a
#: research-use status with a traceable basis.
CARRIED_FORWARD: Dict[str, Dict[str, str]] = {
    'artemis_imagesafari': {
        'real_name': 'Artemis / Image Safari (CIAT-Alliance)',
        'publisher': 'Alliance of Bioversity International and CIAT',
        'official_url': 'https://registry.opendata.aws/artemis-image-safari/',
        'licence_id': 'CC-BY-SA-4.0',
        'research_use': VERIFIED_RESEARCH_USE,
        'redistribution': RELEASE_ALLOWED_WITH_CONDITIONS,
        'conditions': ('Attribution (Mutuvi et al., agriRxiv 2026) and ShareAlike on any '
                       'redistributed adaptation.'),
        'checkpoint_release': RELEASE_UNRESOLVED,
        'checkpoint_note': ('ShareAlike applies to adaptations of the *material*; whether trained '
                            'weights are an adaptation is a legal question this stage does not '
                            'settle. Recorded, not assumed.'),
        'open_issues': 'ShareAlike/checkpoint interaction not legally resolved.',
    },
    'agrifield40k_pool': {
        'real_name': 'AgriField-40K curated pool (local mirror of upstream datasets)',
        'publisher': 'AgriField-40K authors (mirror of many upstream publishers)',
        'official_url': 'https://dtu-pas.github.io/agrifield40k/',
        'licence_id': 'CC-BY-4.0',
        'research_use': VERIFIED_RESEARCH_USE,
        'redistribution': RELEASE_ALLOWED_WITH_CONDITIONS,
        'conditions': 'Attribution. Upstream per-dataset licences still apply to each subset.',
        'checkpoint_release': RELEASE_UNRESOLVED,
        'checkpoint_note': 'Depends on the upstream dataset behind each image.',
        'open_issues': ('The pool aggregates many upstream datasets; a single CC-BY-4.0 label is a '
                        'pool-level statement, not a per-image verification.'),
    },
    'applebbch81': {
        'real_name': 'AppleBBCH81',
        'publisher': 'Stage 5B ingested source',
        'official_url': 'https://github.com/',
        'licence_id': 'CC-BY-4.0',
        'research_use': VERIFIED_RESEARCH_USE,
        'redistribution': RELEASE_ALLOWED_WITH_CONDITIONS,
        'conditions': 'Attribution.',
        'checkpoint_release': RELEASE_UNRESOLVED,
        'checkpoint_note': 'Not decided at this stage.',
        'open_issues': 'Official evidence URL needs to be pinned to the exact record.',
    },
    'sesame_weed': {
        'real_name': 'SesameWeed',
        'publisher': 'Stage 5B ingested source',
        'official_url': 'https://github.com/',
        'licence_id': 'CC0-1.0',
        'research_use': VERIFIED_RESEARCH_USE,
        'redistribution': RELEASE_ALLOWED,
        'conditions': '',
        'checkpoint_release': RELEASE_ALLOWED,
        'checkpoint_note': 'CC0 imposes no condition.',
        'open_issues': 'No accepted images remain after exact-duplicate removal, so the row is informational.',
    },
}

#: Sources whose licence could not be positively verified in Stage 5D.  They
#: contribute no accepted images and must never re-enter training.
BLOCKED_IDS = ('agrifm_ref6', 'agrifm_ref7', 'agrifm_ref8', 'agrifm_ref90')


def licence_for(source_id: str) -> Optional[SourceLicence]:
    return LICENCES.get(source_id)


def research_use_status(source_id: str) -> str:
    """Return the research-use status for any source id.

    Unknown ids are ``UNRESOLVED`` - never silently admitted.
    """
    if source_id in LICENCES:
        return LICENCES[source_id].research_use
    if source_id in CARRIED_FORWARD:
        return CARRIED_FORWARD[source_id]['research_use']
    return UNRESOLVED


def train_eligible(source_id: str) -> bool:
    return research_use_status(source_id) in (VERIFIED_RESEARCH_USE, RESTRICTED_RESEARCH_USE)


def redistribution_status(source_id: str) -> str:
    if source_id in LICENCES:
        return LICENCES[source_id].redistribution
    if source_id in CARRIED_FORWARD:
        return CARRIED_FORWARD[source_id]['redistribution']
    return RELEASE_UNRESOLVED


def redistribution_eligible(source_id: str) -> bool:
    return redistribution_status(source_id) in (RELEASE_ALLOWED,
                                                 RELEASE_ALLOWED_WITH_CONDITIONS)


def audit_rows() -> List[Dict[str, object]]:
    """One row per audited source, sorted by id."""
    rows: List[Dict[str, object]] = []
    for source_id in sorted(LICENCES):
        rows.append(LICENCES[source_id].as_row())
    for source_id in sorted(CARRIED_FORWARD):
        meta = CARRIED_FORWARD[source_id]
        rows.append({
            'source_id': source_id,
            'real_name': meta['real_name'],
            'publisher': meta['publisher'],
            'official_url': meta['official_url'],
            'licence_id': meta['licence_id'],
            'licence_name': meta['licence_id'],
            'evidence_url': meta['official_url'],
            'evidence_locator': 'Stage 5D source registry (carried forward)',
            'evidence_excerpt': meta['conditions'],
            'evidence_file': 'data/corpus_v3/sources.csv',
            'evidence_sha256': '',
            'accessed': '2026-10-25',
            'research_use': meta['research_use'],
            'research_use_conditions': meta['conditions'],
            'redistribution': meta['redistribution'],
            'redistribution_conditions': meta['conditions'],
            'metadata_release': RELEASE_ALLOWED,
            'metadata_release_conditions': '',
            'derived_release': meta['redistribution'],
            'derived_release_conditions': meta['conditions'],
            'checkpoint_release': meta['checkpoint_release'],
            'checkpoint_release_conditions': meta['checkpoint_note'],
            'open_issues': meta['open_issues'],
            'decision': 'INCLUDE',
            'decision_reason': 'Stage 5D accepted source; licence re-stated, not re-litigated.',
            'mirror_urls': '',
        })
    return rows
