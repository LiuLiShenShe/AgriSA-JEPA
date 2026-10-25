"""Canonical Stage 5D metadata schema.

Every field carries an explicit provenance.  The schema exists so that a
value which comes from an official source (an S3 key, a published sidecar) can
never be silently mixed with a value guessed from a filename or predicted by a
model.

The enumerations below are the *only* accepted values.  Reports must use them
verbatim; free-text evidence belongs in ``*_evidence`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Provenance tiers.  Ordered from strongest to weakest.
# --------------------------------------------------------------------------- #
PROVENANCE_OFFICIAL = 'OFFICIAL_METADATA'          # published sidecar / index
PROVENANCE_DATASET_LAYOUT = 'DATASET_LAYOUT'       # channel folder / key layout
PROVENANCE_PATH = 'PATH_INFERRED'                  # structured path convention
PROVENANCE_FILENAME = 'FILENAME_INFERRED'          # filename convention
PROVENANCE_MODEL = 'MODEL_INFERRED'                # CLIP / DINOv2 prediction
PROVENANCE_MISSING = 'MISSING'

PROVENANCE_TIERS = {
    PROVENANCE_OFFICIAL: 0,
    PROVENANCE_DATASET_LAYOUT: 1,
    PROVENANCE_PATH: 2,
    PROVENANCE_FILENAME: 3,
    PROVENANCE_MODEL: 4,
    PROVENANCE_MISSING: 5,
}


def strongest(*values: str) -> str:
    """Return the strongest (lowest tier) non-empty provenance value."""
    present = [v for v in values if v in PROVENANCE_TIERS]
    if not present:
        return PROVENANCE_MISSING
    return min(present, key=lambda v: PROVENANCE_TIERS[v])


# --------------------------------------------------------------------------- #
# Modality.  Visible RGB is the only modality admitted to R3v2_strict_rgb.
# --------------------------------------------------------------------------- #
MODALITY_RGB_VISIBLE = 'RGB_VISIBLE'
MODALITY_NDVI_PSEUDOCOLOR = 'NDVI_PSEUDOCOLOR'
MODALITY_FALSE_COLOR = 'FALSE_COLOR'
MODALITY_MULTISPECTRAL = 'MULTISPECTRAL'
MODALITY_NIR_RAW = 'NIR_RAW'
MODALITY_GRAYSCALE = 'GRAYSCALE'
MODALITY_UNKNOWN = 'UNKNOWN_MODALITY'

VISIBLE_RGB_MODALITIES = {MODALITY_RGB_VISIBLE}

MODALITY_STATUS_EVIDENCED = 'VISIBLE_RGB_EVIDENCED'
MODALITY_STATUS_NOT_VISIBLE = 'NOT_VISIBLE_RGB'
MODALITY_STATUS_UNKNOWN = 'UNKNOWN_MODALITY'

#: Evidence types accepted for a positive visible-RGB claim.  A JPEG that
#: merely decodes to three channels is deliberately *not* on this list.
MODALITY_EVIDENCE_STRENGTH = {
    'OFFICIAL_DOCUMENTATION': 0,
    'DATASET_CHANNEL_LAYOUT': 1,
    'REGISTRY_DECLARED_RGB': 2,
    'PIL_MODE_ONLY': 98,
    'NONE': 99,
}


# --------------------------------------------------------------------------- #
# Capture-group trust levels.
# --------------------------------------------------------------------------- #
GROUP_OFFICIAL = 'OFFICIAL_GROUP'
GROUP_DOCUMENTED_PATH = 'DOCUMENTED_PATH_GROUP'
GROUP_WEAK_PROXY = 'WEAK_PROXY'
GROUP_UNKNOWN = 'UNKNOWN_GROUP'

GROUP_STRONG = {GROUP_OFFICIAL, GROUP_DOCUMENTED_PATH}

#: Only strong groups may contribute to a GROUP_DISJOINT_VAL.
SPLIT_TRAIN = 'train'
SPLIT_VAL = 'val'
SPLIT_TRAIN_ONLY_UNKNOWN = 'TRAIN_ONLY_GROUP_UNKNOWN'
SPLIT_DIAGNOSTIC_RANDOM_VAL = 'DIAGNOSTIC_RANDOM_VAL'


# --------------------------------------------------------------------------- #
# Canonical column list written to the master ledger.
# --------------------------------------------------------------------------- #
CANONICAL_FIELDS: List[str] = [
    'image_id',
    'physical_source_id',
    'original_dataset_id',
    'original_image_id',
    'original_uri',
    'source_version',
    'original_path',
    'relative_path',
    'raw_sha256',
    'pixel_sha256',
    'phash',
    'dhash',
    'width',
    'height',
    'channels',
    'format',
    'mode',
    'provenance_method',
    'modality',
    'modality_evidence',
    'modality_status',
    'acquisition_domain',
    'domain_evidence_type',
    'domain_confidence',
    'crop',
    'crop_evidence',
    'country',
    'country_evidence',
    'region',
    'farm_id',
    'plot_id',
    'plant_id',
    'session_id',
    'capture_timestamp',
    'sensor_id',
    'device_id',
    'original_split',
    'group_key',
    'group_evidence',
    'group_confidence',
    'metadata_join_status',
    'iqa_musiq',
    'iqa_niqe',
    'short_side',
    'license_status',
    'holdout_status',
    'archive_only',
    'split',
    'decision',
    'primary_reason',
    'secondary_reasons',
]


@dataclass
class JoinAudit:
    """Row counts for an exact-key metadata join."""

    reference_rows: int = 0
    candidate_rows: int = 0
    matched_rows: int = 0
    unmatched_rows: int = 0
    one_to_one: int = 0
    one_to_many: int = 0
    many_to_one: int = 0
    key_collisions: int = 0
    conflict_rows: int = 0
    conflict_fields: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            'reference_rows': self.reference_rows,
            'candidate_rows': self.candidate_rows,
            'matched_rows': self.matched_rows,
            'unmatched_rows': self.unmatched_rows,
            'one_to_one': self.one_to_one,
            'one_to_many': self.one_to_many,
            'many_to_one': self.many_to_one,
            'key_collisions': self.key_collisions,
            'conflict_rows': self.conflict_rows,
            'conflict_fields': dict(self.conflict_fields),
        }


#: Crop normalisation.  Only aliases that are documented in the upstream
#: dataset metadata are listed; anything else is passed through lower-cased so
#: that an unknown label stays visibly unknown rather than being guessed into
#: an existing class.
CROP_ALIASES: Dict[str, str] = {
    'common bean': 'common-bean',
    'commonbean': 'common-bean',
    'pearl millet': 'pearl-millet',
    'pigeon pea': 'pigeon-pea',
    'sweet potato': 'sweet-potato',
    'finger millet': 'finger-millet',
    'african yam': 'african-yam',
    'ground nut': 'groundnut',
    'maize ': 'maize',
}


def normalize_crop(value: Optional[str]) -> str:
    if value is None:
        return 'unknown'
    text = str(value).strip().lower()
    if not text or text in ('nan', 'none', 'unknown', ''):
        return 'unknown'
    return CROP_ALIASES.get(text, text)
