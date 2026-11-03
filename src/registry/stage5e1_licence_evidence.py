"""Stage 5E.1 licence-evidence grading.

Stage 5E's ``licence_evidence_backed`` gate claimed that *every* accepted
source had "an archived official document whose SHA-256 still matches".  That
was not true: four sources had a freshly archived publisher response, while
four others merely re-stated a Stage 5D registry row.  The gate description,
the gate code and the report therefore disagreed.

This module makes the distinction explicit and machine-readable.  Every
accepted source gets exactly one grade:

``OFFICIAL_EVIDENCE_ARCHIVED_AND_HASHED``
    This stage fetched an official publisher response and archived its bytes;
    the SHA-256 is recorded and re-checked.
``CARRIED_FORWARD_WITH_PRIOR_EVIDENCE``
    No new fetch this stage, but a *prior* pinned artifact (the Stage 5D source
    registry, the pinned compilation card, or the per-image iNaturalist
    metadata) is named, present, and hashed.
``REGISTRY_URL_ONLY``
    Only a registry URL / prose citation exists.  The official document was not
    archived, so this stage does not claim to have re-verified it.
``UNRESOLVED``
    Neither an archived document nor a usable prior artifact pins the licence.

The grade is about *evidence*, never about whether the images are usable.  A
``REGISTRY_URL_ONLY`` or ``UNRESOLVED`` grade lowers the Stage 6 status to
``CONDITIONAL``; it never silently upgrades a source to "verified".

For ``agrifield40k_pool`` the grade is computed per upstream dataset, because a
single pool-level CC-BY tag cannot grant rights over third-party images.  The
Creative Commons guidance is the general principle used here: a licence only
grants the rights the licensor is able to grant, and third-party material may
need separate permission.  We do not extend that general principle into any
unverified legal conclusion about a specific upstream.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent

EVIDENCE_ARCHIVED = 'OFFICIAL_EVIDENCE_ARCHIVED_AND_HASHED'
EVIDENCE_CARRIED = 'CARRIED_FORWARD_WITH_PRIOR_EVIDENCE'
EVIDENCE_URL_ONLY = 'REGISTRY_URL_ONLY'
EVIDENCE_UNRESOLVED = 'UNRESOLVED'

GRADES = (EVIDENCE_ARCHIVED, EVIDENCE_CARRIED, EVIDENCE_URL_ONLY, EVIDENCE_UNRESOLVED)

#: Grades that are strong enough to be called "evidence-backed" in a report.
EVIDENCE_BACKED_GRADES = (EVIDENCE_ARCHIVED, EVIDENCE_CARRIED)

#: Stage 5D registry that the carried-forward rows cite.
STAGE5D_REGISTRY = 'data/corpus_v3/sources.csv'

#: Pinned source registry for the AgriField-40K compilation.
POOL_SOURCE_REGISTRY = 'data/source_registry.json'
POOL_PINNED_REVISION = '7f530c67572b6a572467adab840cdbbd7029b112'

#: Per-image iNaturalist licence metadata used by the iNatWeeds subset.
INAT_PER_IMAGE_METADATA = 'data/iNatWeeds_metadata.json'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Per-source grading for the top-level physical sources
# --------------------------------------------------------------------------

#: Sources whose official response was archived during Stage 5E.
ARCHIVED_SOURCES = ('agrifm_ref1', 'agrifm_ref9', 'agrifm_ref11', 'agrifm_ref58')

#: Sources carried forward from the Stage 5D registry, with the field that
#: carries the licence statement.
CARRIED_SOURCES = {
    'artemis_imagesafari': {
        'prior_artifact': STAGE5D_REGISTRY,
        'prior_locator': 'Stage 5D row: license_evidence = AWS Open Data registry licence field',
        'checked_this_stage': False,
        'notes': ('The AWS Open Data registry page was not re-fetched in Stage 5E.1; the '
                  'Stage 5D row is hashed and carried forward verbatim.'),
    },
    'applebbch81': {
        'prior_artifact': STAGE5D_REGISTRY,
        'prior_locator': 'Stage 5D row: license_evidence = Kaggle dataset page licence field',
        'checked_this_stage': False,
        'notes': ('The licence module\'s evidence_url is the generic "https://github.com/" '
                  'placeholder, not a dataset-specific record. No archived official body '
                  'exists, so this stage cannot claim to have re-verified it.'),
        'force_url_only': True,
    },
    'sesame_weed': {
        'prior_artifact': STAGE5D_REGISTRY,
        'prior_locator': 'Stage 5D row: license_evidence = Kaggle dataset page licence field',
        'checked_this_stage': False,
        'notes': ('No accepted images remain after exact-duplicate removal; the row is '
                  'informational only.'),
        'force_url_only': True,
    },
}


def grade_source(source_id: str, accepted_images: int) -> Dict[str, object]:
    """Return the evidence grade for one physical source."""
    result = _grade_source(source_id, accepted_images)
    result['accepted_images'] = accepted_images
    return result


def _grade_source(source_id: str, accepted_images: int) -> Dict[str, object]:
    if accepted_images <= 0:
        return {'source_id': source_id, 'grade': EVIDENCE_URL_ONLY,
                'evidence_file': '', 'evidence_sha256': '',
                'basis': 'no accepted images; informational row',
                'affected_accepted_images': 0, 'affected_train_images': 0,
                'checked_this_stage': False}

    if source_id in ARCHIVED_SOURCES:
        audit = json.loads((ROOT / 'reports/stage5e_license_audit.json').read_text())
        row = next((r for r in audit['sources'] if r['source_id'] == source_id), {})
        evidence_file = str(row.get('evidence_file') or '')
        recorded = str(row.get('evidence_sha256') or '')
        path = ROOT / evidence_file
        actual = sha256_file(path) if path.exists() else ''
        ok = bool(recorded) and actual == recorded
        return {'source_id': source_id,
                'grade': EVIDENCE_ARCHIVED if ok else EVIDENCE_UNRESOLVED,
                'evidence_file': evidence_file, 'evidence_sha256': actual,
                'basis': 'official publisher endpoint archived during Stage 5E',
                'checked_this_stage': True,
                'notes': '' if ok else 'archived evidence missing or SHA-256 mismatch'}

    if source_id == 'agrifield40k_pool':
        # Graded per upstream below; the source-level grade is the weakest
        # accepted upstream grade.
        return {'source_id': source_id, 'grade': EVIDENCE_UNRESOLVED,
                'evidence_file': POOL_SOURCE_REGISTRY,
                'evidence_sha256': sha256_file(ROOT / POOL_SOURCE_REGISTRY),
                'basis': 'graded per upstream dataset; see pool_upstream_audit',
                'checked_this_stage': False,
                'notes': 'Source-level grade is UNRESOLVED while any accepted upstream is unresolved.'}

    meta = CARRIED_SOURCES.get(source_id)
    if meta:
        artifact = ROOT / meta['prior_artifact']
        grade = EVIDENCE_URL_ONLY if meta.get('force_url_only') else EVIDENCE_CARRIED
        return {'source_id': source_id, 'grade': grade,
                'evidence_file': meta['prior_artifact'],
                'evidence_sha256': sha256_file(artifact) if artifact.exists() else '',
                'basis': meta['prior_locator'],
                'checked_this_stage': bool(meta.get('checked_this_stage')),
                'notes': meta['notes']}

    return {'source_id': source_id, 'grade': EVIDENCE_UNRESOLVED,
            'evidence_file': '', 'evidence_sha256': '',
            'basis': 'no archived document and no prior artifact',
            'checked_this_stage': False,
            'notes': 'Unregistered source; must not be treated as verified.'}


# --------------------------------------------------------------------------
# Per-upstream grading for agrifield40k_pool
# --------------------------------------------------------------------------

#: Publishers recovered from the AgriField-40K paper (arXiv:2608.07984, Table 1)
#: and the pinned compilation card.  ``registry_state`` is the value recorded in
#: ``data/source_registry.json``: ``confirmed`` means the source registry pinned
#: an exact prefix/label mapping and the pool-declared licence; ``pending_audit``
#: means the mapping/licence was explicitly left unconfirmed.
POOL_UPSTREAMS: Dict[str, Dict[str, object]] = {
    'LUCASVision': dict(year=2023, pool_declared_license='CC-BY-4.0',
                        publisher='LUCAS (Eurostat) / LUCASVision authors',
                        registry_state='confirmed'),
    'VegAnn': dict(year=2022, pool_declared_license='CC-BY-1.0',
                   publisher='VegAnn authors',
                   registry_state='confirmed'),
    'ACRECropWeed': dict(year=2023, pool_declared_license='CC-BY-4.0',
                         publisher='Bertoglio et al. (ACRE)',
                         registry_state='confirmed'),
    'RadishWheat': dict(year=2022, pool_declared_license='CC-BY-4.0',
                        publisher='Rayner (weed-ai.sydney.edu.au)',
                        registry_state='confirmed'),
    'iNatWeeds': dict(year=2026, pool_declared_license='per-image CC BY / CC0',
                      publisher='iNaturalist observers (per-image licences)',
                      registry_state='confirmed',
                      per_image_metadata=INAT_PER_IMAGE_METADATA),
    'VCD': dict(year=2022, pool_declared_license='CC-BY-4.0',
                publisher='VCD authors', registry_state='pending_audit'),
    'WE3DS': dict(year=2023, pool_declared_license='CC-BY-4.0',
                  publisher='Kitzler et al.', registry_state='pending_audit'),
    'SesameWeed': dict(year=2020, pool_declared_license='CC0',
                       publisher='Sesame & Weed dataset authors',
                       registry_state='pending_audit'),
    'RumexLeaves': dict(year=2024, pool_declared_license='CC-BY-4.0',
                        publisher='RumexWeeds authors', registry_state='pending_audit'),
    'Maize-Weed': dict(year=2022, pool_declared_license='CC-BY-4.0',
                       publisher='Maize-Weed authors', registry_state='pending_audit'),
    'GrassClover': dict(year=2019, pool_declared_license='CC-BY-SA',
                        publisher='GrassClover authors', registry_state='pending_audit'),
    # Upstreams with zero accepted images; kept for completeness.
    'MuST-C': dict(year=2026, pool_declared_license='CC-BY-4.0',
                   publisher='MuST-C authors', registry_state='pending_audit'),
    'PalmerAmaranth': dict(year=2023, pool_declared_license='CC-BY-4.0',
                           publisher='Palmer Amaranth authors', registry_state='pending_audit'),
    'Ronin': dict(year=2021, pool_declared_license='CC-BY-4.0',
                  publisher='Ronin authors', registry_state='pending_audit'),
    'PerennialPlants': dict(year=2021, pool_declared_license='MIT',
                            publisher='PerennialPlants authors', registry_state='pending_audit'),
    'PhenoBench': dict(year=2026, pool_declared_license='CC-BY-SA-4.0',
                       publisher='University of Bonn (PhenoRob)', registry_state='confirmed'),
    'SorghumWeed': dict(year=2023, pool_declared_license='CC-BY-4.0',
                        publisher='SorghumWeed authors', registry_state='confirmed'),
}


def pool_license_declarations() -> Dict[str, str]:
    """The pool's own licence label, and its known internal inconsistency."""
    return {
        'stage5d_registry': 'CC-BY-4.0',
        'agrifield40k_paper_table1': 'CC-BY-SA-4.0',
        'note': ('The Stage 5D registry row says CC-BY-4.0; the AgriField-40K paper '
                 '(arXiv:2608.07984, Table 1 and Section 3) says CC-BY-SA-4.0. The '
                 'discrepancy is reported, not silently resolved. Either way it is a '
                 'pool-level label for the compilation.'),
    }


def grade_pool_upstream(upstream: str, accepted_images: int, train_images: int,
                        per_image_counts: Optional[Dict[str, int]] = None
                        ) -> Dict[str, object]:
    meta = POOL_UPSTREAMS.get(upstream, {})
    state = str(meta.get('registry_state', 'pending_audit'))
    per_image = meta.get('per_image_metadata')

    if accepted_images <= 0:
        return dict(upstream_dataset=upstream, pool_declared_license=meta.get(
            'pool_declared_license', 'unknown'), publisher=meta.get('publisher', 'unknown'),
            year=meta.get('year'), accepted_images=0, train_images=0,
            upstream_asset_license_grade=EVIDENCE_URL_ONLY,
            upstream_asset_license=meta.get('pool_declared_license', 'unknown'),
            identity_basis='pool paper Table 1 (retained count only; no accepted images)',
            evidence_file=POOL_SOURCE_REGISTRY, notes='No accepted images; informational.')

    if per_image:
        counts = per_image_counts or {}
        return dict(upstream_dataset=upstream,
                    pool_declared_license=meta['pool_declared_license'],
                    publisher=meta['publisher'], year=meta.get('year'),
                    accepted_images=accepted_images, train_images=train_images,
                    upstream_asset_license_grade=EVIDENCE_CARRIED,
                    upstream_asset_license=meta['pool_declared_license'],
                    identity_basis='pinned source registry + per-image metadata join',
                    evidence_file=per_image,
                    evidence_sha256=sha256_file(ROOT / per_image) if (ROOT / per_image).exists() else '',
                    per_image_license_counts=counts,
                    notes=('Each accepted image carries an individual CC BY or CC0 licence '
                           'record in the pinned metadata; the licence is per image, not '
                           'inferred from the pool label.'))

    if state == 'confirmed':
        return dict(upstream_dataset=upstream,
                    pool_declared_license=meta['pool_declared_license'],
                    publisher=meta['publisher'], year=meta.get('year'),
                    accepted_images=accepted_images, train_images=train_images,
                    upstream_asset_license_grade=EVIDENCE_CARRIED,
                    upstream_asset_license=meta['pool_declared_license'],
                    identity_basis=('pinned source registry prefix mapping; retained count '
                                    'matches the pool paper Table 1 exactly'),
                    evidence_file=POOL_SOURCE_REGISTRY,
                    evidence_sha256=sha256_file(ROOT / POOL_SOURCE_REGISTRY),
                    notes=('Licence is the upstream licence declared by the pinned AgriField-40K '
                           'compilation card. It is a prior pinned artifact, not a freshly '
                           'archived upstream LICENSE file.'))

    return dict(upstream_dataset=upstream,
                pool_declared_license=meta.get('pool_declared_license', 'unknown'),
                publisher=meta.get('publisher', 'unknown'), year=meta.get('year'),
                accepted_images=accepted_images, train_images=train_images,
                upstream_asset_license_grade=EVIDENCE_UNRESOLVED,
                upstream_asset_license='unconfirmed (pool paper declares %s)' % meta.get(
                    'pool_declared_license', 'unknown'),
                identity_basis=('prefix abbreviation plus retained-count match only; the pinned '
                                'source registry marks this mapping pending_audit'),
                evidence_file=POOL_SOURCE_REGISTRY,
                evidence_sha256=sha256_file(ROOT / POOL_SOURCE_REGISTRY),
                notes=('No archived upstream LICENSE / data card, and the pinned registry left '
                       'the mapping and licence unconfirmed. The pool-level CC-BY label is not '
                       'used to infer a third-party licence.'))


def build_pool_audit(upstream_counts: Dict[str, Dict[str, int]]) -> Dict[str, object]:
    """``upstream_counts`` maps upstream -> {'accepted': n, 'train': n}."""
    rows: List[Dict[str, object]] = []
    per_image = {}
    if (ROOT / INAT_PER_IMAGE_METADATA).exists():
        data = json.loads((ROOT / INAT_PER_IMAGE_METADATA).read_text())
        per_image = {}
        for item in data:
            key = str(item.get('license', 'unknown')).lower()
            per_image[key] = per_image.get(key, 0) + 1

    for upstream in sorted(POOL_UPSTREAMS):
        counts = upstream_counts.get(upstream, {})
        row = grade_pool_upstream(upstream, int(counts.get('accepted', 0)),
                                  int(counts.get('train', 0)), per_image)
        row.setdefault('version', f"release {POOL_UPSTREAMS[upstream].get('year', 'unknown')}")
        rows.append(row)

    unresolved = [r for r in rows if r['upstream_asset_license_grade'] == EVIDENCE_UNRESOLVED
                  and r['accepted_images'] > 0]
    carried = [r for r in rows if r['upstream_asset_license_grade'] == EVIDENCE_CARRIED
               and r['accepted_images'] > 0]
    return {
        'pool_license': pool_license_declarations(),
        'pool_pinned_revision': POOL_PINNED_REVISION,
        'rule': ('A pool-level CC-BY label is a statement about the compilation, not proof that '
                 'the pool maintainer could relicense third-party images. Creative Commons\' '
                 'guidance is that a licence only grants the rights the licensor is able to '
                 'grant, and third-party material may need separate permission. That is the '
                 'general principle used here; each upstream is graded on its own evidence, '
                 'without extending it into a specific legal conclusion.'),
        'upstreams': rows,
        'summary': {
            'upstreams_with_accepted_images': len(carried) + len(unresolved),
            'upstreams_carried_forward': len(carried),
            'upstreams_unresolved': len(unresolved),
            'accepted_images_carried_forward': int(sum(r['accepted_images'] for r in carried)),
            'accepted_images_unresolved': int(sum(r['accepted_images'] for r in unresolved)),
            'train_images_carried_forward': int(sum(r['train_images'] for r in carried)),
            'train_images_unresolved': int(sum(r['train_images'] for r in unresolved)),
            'unresolved_upstreams': [r['upstream_dataset'] for r in unresolved],
        },
    }
