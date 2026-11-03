"""Per-source acquisition-domain ledger for the Stage 5B corpus.

Every row here is a *judgement with a citation*.  The reason column must name
the evidence (official description, paper sentence, dataset card) or say that
there is none.  ``confidence`` is deliberately harsh:

* ``high``   -- the source's own documentation states the capture setting and
                the statement is unambiguous;
* ``medium`` -- the documentation implies the setting (e.g. "images of weeds
                in soybean fields") but does not spell out the platform;
* ``low``    -- only the dataset name or a third-party description suggests it;
* ``unknown``-- no usable evidence.  These rows can never enter Field-D1.

Nothing in this table is derived from looking at the images.  Image-level
evidence lives in ``data/registry/field_image_review.csv``.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

Ledger = Dict[str, Dict[str, str]]

#: ``source_id -> {acquisition_domain, evidence_type, confidence, reason, platform}``
SOURCE_DOMAINS: Ledger = {
    # ---- AgriField-40K sources present in v1 -------------------------------
    'acre_crop_weed': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground robot',
        'reason': ('ACRE Crop-Weed dataset: RGB frames recorded by an agricultural '
                   'robot in a real crop field, released with per-frame capture '
                   'timestamps and a session date.'),
    },
    'vegann': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in field',
        'reason': ('VegAnn: vegetable crop images annotated for instance '
                   'segmentation, captured on working vegetable farms in France.'),
    },
    'lucas_vision': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera, ground-level frame',
        'reason': ('LUCAS topsoil survey photographs: a camera on a frame points '
                   'at the ground surface of the sampled field parcel; the survey '
                   'protocol records the parcel and the observation date.'),
    },
    'sorghum_weed': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'handheld camera in field plots',
        'reason': ('SorghumWeedDataset: sorghum and weed images from field plots; '
                   'the dataset description names field plots but not the camera '
                   'platform in detail.'),
    },
    'radish_wheat': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'ground-level field imagery',
        'reason': ('RadishWheatDataset: radish and wheat plants photographed in '
                   'field trial plots for plant/weed discrimination.'),
    },
    'inat_weeds': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'citizen-science handheld camera',
        'reason': ('iNaturalist weed observations: opportunistic photographs of '
                   'plants in situ, each carrying a licensed observation record. '
                   'A minority of observations are from gardens or cultivation, so '
                   'the label is medium rather than high and images stay reviewable.'),
    },
    # ---- other AgriField-40K sources ---------------------------------------
    'grass_clover': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera over field plots',
        'reason': ('GrassClover Image Dataset (Aarhus University): images of '
                   'grass-clover field plots taken from above the canopy, with '
                   'plot and biomass records.'),
    },
    'maize_weed': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in field',
        'reason': ('Maize-Weed dataset (Data in Brief 2023): maize and weed images '
                   'captured in the field across growth stages.'),
    },
    'palmer_amaranth': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'field imagery (weed-ai record)',
        'reason': ('Palmer Amaranth Growth Stage dataset: growth-stage images of '
                   'Palmer amaranth in field conditions, distributed through weed-ai.'),
    },
    'rumex_leaves': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'ground robot',
        'reason': ('RumexLeaves / Zoom-in-on-the-Plant: close-up leaf imagery '
                   'collected for robotic weed control in pasture; close-ups are '
                   'kept because the capture context is the field, not a bench.'),
    },
    'sesame_weed': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in field',
        'reason': ('Crop and Weed Detection Data with Bounding Boxes: sesame crop '
                   'and weed images with bounding boxes, described by the publisher '
                   'as field data; bytes are identical to the upstream AgriField-40K '
                   'saw_* rows, which confirms the source mapping.'),
    },
    'vcd': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'shrouded platform over field plots',
        'reason': ('VCD (Lac et al., Data in Brief 2022): early-stage vegetable '
                   'crop images acquired with a shrouded platform over field plots, '
                   'with plot and acquisition grouping.'),
    },
    'we3ds': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground platform with RGB-D sensor',
        'reason': ('WE3DS (Sensors 2023): RGB-D images of crops and weeds recorded '
                   'in the field for semantic segmentation. Only the RGB channel is '
                   'in scope for this 2D task; the depth channel is not used.'),
    },
    'must_c': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'robot; UAV; multi-sensor',
        'reason': ('MuST-C is explicitly multi-sensor and multi-platform (ground '
                   'robot and UAV). It cannot receive one domain label; image-level '
                   'separation would be required before any use.'),
    },
    'perennial_plants': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'official_documentation',
        'confidence': 'low',
        'platform': 'nursery / small-batch cultivation',
        'reason': ('Perennial Plants Detection: the description says >2000 plant '
                   'varieties are grown in small batches with weekly manual weed '
                   'removal. That reads as a nursery or protected cultivation site, '
                   'but the source does not state the setting, so it is UNKNOWN.'),
    },
    'ronin': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'ground robot',
        'reason': ('Ronin / annotated food crops and weeds (Sudars et al., Data in '
                   'Brief 2020): images for robotic weed control, captured in '
                   ' cultivated fields in Latvia.'),
    },
    # ---- Agri-FM+ Table 1 sources -----------------------------------------
    'agrifm_ref1': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Wheat spike image collection with no capture-setting documentation on the dataset page.',
    },
    'agrifm_ref10': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': ('Maize & cocoa classification collection; the page does not state '
                   'whether images were taken in fields, markets or on a bench.'),
    },
    'agrifm_ref11': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'NDVI camera in field',
        'reason': ('Maize phenology images from an NDVI camera in field trials. The '
                   'modality is a vegetation-index camera, so RGB suitability must be '
                   'checked before use.'),
    },
    'agrifm_ref15': {
        'acquisition_domain': 'CONTROLLED_LAB',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'bench photography',
        'reason': ('Maize seed dataset: seeds photographed on a uniform background '
                   'for classification; not a growing crop in a field.'),
    },
    'agrifm_ref18': {
        'acquisition_domain': 'ORCHARD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in orchard',
        'reason': ('Deep Fruit Detection in Orchards (ACFR): apples photographed on '
                   'trees in commercial orchards.'),
    },
    'agrifm_ref2': {
        'acquisition_domain': 'UAV_AERIAL',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'fixed-wing UAV (eBee Ag)',
        'reason': 'Corn field mapped with eBee Ag: aerial survey imagery.',
    },
    'agrifm_ref20': {
        'acquisition_domain': 'CONTROLLED_LAB',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'bench photography',
        'reason': ('Winter wheat leaf images: detached leaves photographed for '
                   'disease classification, without field-context metadata.'),
    },
    'agrifm_ref29': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in field',
        'reason': 'Maize in Field dataset: maize photographed in the field.',
    },
    'agrifm_ref3': {
        'acquisition_domain': 'UAV_AERIAL',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'fixed-wing UAV (eBee Ag)',
        'reason': 'Mixed-use agricultural fields mapped with eBee Ag: aerial imagery.',
    },
    'agrifm_ref31': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in field',
        'reason': ('GWHD 2021: wheat head images taken in fields across many '
                   'countries. Domain is field, but the source is a downstream '
                   'holdout and never enters pretraining.'),
    },
    'agrifm_ref35': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'farmer smartphone',
        'reason': ('Cassava Leaf Disease Classification: leaf photographs submitted '
                   'by farmers from cassava fields, so real backgrounds dominate; '
                   'individual images still need review.'),
    },
    'agrifm_ref37': {
        'acquisition_domain': 'CONTROLLED_LAB',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'controlled imaging rig',
        'reason': ('V2 Plant Seedlings: seedlings imaged on a uniform soil/background '
                   'rig for classification, not in situ field scenes.'),
    },
    'agrifm_ref4': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'official_documentation',
        'confidence': 'low',
        'platform': 'nursery / small-batch cultivation',
        'reason': ('Same physical dataset as perennial_plants; the setting is not '
                   'stated, so it stays UNKNOWN and is not double-counted.'),
    },
    'agrifm_ref46': {
        'acquisition_domain': 'UAV_AERIAL',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'UAV',
        'reason': 'Maize Plant Abnormality from UAV Images: explicitly aerial.',
    },
    'agrifm_ref48': {
        'acquisition_domain': 'UAV_AERIAL',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'drone',
        'reason': 'Drone-based Agricultural Dataset for Crop Yield Estimation: explicitly drone imagery.',
    },
    'agrifm_ref5': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'smallholder smartphone',
        'reason': ('Makerere Fall Armyworm Crop Challenge: photographs of maize '
                   'damage taken by smallholder farmers in their fields.'),
    },
    'agrifm_ref57': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'handheld camera',
        'reason': ('Crop Pest and Disease Detection (Ghana): the Mendeley description '
                   'states images were "captured under various conditions and with '
                   'different backgrounds such as white, dark, illuminated, and real '
                   'backgrounds". That is a mixed field/laboratory source; a single '
                   'source label would be false, so it stays UNKNOWN pending '
                   'image-level separation.'),
    },
    'agrifm_ref58': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Makerere Beans Image Dataset: capture setting not stated in the repository record.',
    },
    'agrifm_ref6': {
        'acquisition_domain': 'CONTROLLED_LAB',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'bench photography',
        'reason': ('Leaf Images Dataset (hamedetezadi): detached leaves on plain '
                   'backgrounds for classification.'),
    },
    'agrifm_ref60': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground vehicle',
        'reason': ('DeepWeeds: weed images recorded by a ground vehicle in northern '
                   'Australian rangeland. Holdout source; never pretraining data.'),
    },
    'agrifm_ref65': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'farmer smartphone',
        'reason': ('Paddy Doctor: rice leaf photographs taken by farmers in paddy '
                   'fields for disease diagnosis.'),
    },
    'agrifm_ref7': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Agriculture Crops Dataset: no capture-setting documentation.',
    },
    'agrifm_ref73': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'field sprayer camera',
        'reason': 'Weed Spraying Object Detection: imagery captured for in-field weed spraying.',
    },
    'agrifm_ref74': {
        'acquisition_domain': 'UAV_AERIAL',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'UAV',
        'reason': 'Apple Orchards Aerial Detection Dataset: explicitly aerial.',
    },
    'agrifm_ref75': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Corn Stalk Disease collection: capture setting not documented.',
    },
    'agrifm_ref76': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Corn Leaf Disease collection: capture setting not documented.',
    },
    'agrifm_ref79': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'ground robot',
        'reason': 'Same physical dataset as ronin (Latvian crops and weeds); counted once.',
    },
    'agrifm_ref8': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Maize Images Dataset: no capture-setting documentation.',
    },
    'agrifm_ref81': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': ('InsectBase soybean crop insect dataset: insect photographs may be '
                   'lab specimens or field sightings; not stated.'),
    },
    'agrifm_ref82': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'multispectral rig in field',
        'reason': ('Multispectral potato plant images from field trials; the modality '
                   'is not plain RGB, so it is out of scope for this 2D RGB corpus.'),
    },
    'agrifm_ref9': {
        'acquisition_domain': 'ORCHARD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'handheld camera in vineyard',
        'reason': 'PinotNoirGrapes: grape clusters photographed in a vineyard.',
    },
    'agrifm_ref90': {
        'acquisition_domain': 'PROTECTED_CULTIVATION',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'plant factory',
        'reason': ('Tomato Plant Factory Dataset: images from an indoor plant factory, '
                   'i.e. protected cultivation, not open-field agriculture.'),
    },
    'agrifm_ref95': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'field imaging',
        'reason': 'Sugar Beet Nutrient Deficiency RGB: sugar beet photographed in field trials.',
    },
    # ---- downstream holdouts (domain recorded, never used for pretraining) --
    'phenobench': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground robot over sugar beet plots',
        'reason': 'PhenoBench: UAV/robot field imagery of sugar beet plots; downstream holdout.',
    },
    'gwhd_2021': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in field',
        'reason': 'Global Wheat Head Dataset 2021; downstream holdout.',
    },
    'gwhd_instance_seg': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'handheld camera in field',
        'reason': 'Roboflow wheat instance-segmentation remix of GWHD; downstream holdout.',
    },
    'minneapple': {
        'acquisition_domain': 'ORCHARD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'handheld camera in orchard',
        'reason': 'MinneApple: apples on trees in an orchard; downstream holdout.',
    },
    'plantdoc': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'mixed web collection',
        'reason': ('PlantDoc: leaf images collected for disease detection, a mix of '
                   'field and detached-leaf photos; downstream holdout, so the '
                   'ambiguity does not affect pretraining.'),
    },
    'roboflow_disease_detection': {
        'acquisition_domain': 'UNKNOWN',
        'evidence_type': 'none',
        'confidence': 'unknown',
        'platform': '',
        'reason': 'Roboflow community disease dataset; capture protocol undocumented. Downstream holdout.',
    },
    'roboflow_crop_detection': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'field imagery',
        'reason': 'Roboflow crops-vs-grass detection set of field crops; downstream holdout.',
    },
    'plantseg': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'field imagery',
        'reason': 'PlantSeg: plant segmentation imagery from field conditions; downstream holdout.',
    },
    'multiclass_weeds': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'medium',
        'platform': 'field imagery',
        'reason': 'Multiclass Weeds dataset; downstream holdout.',
    },
    'deepweeds': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground vehicle',
        'reason': 'DeepWeeds; downstream holdout.',
    },
    'growliflower': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'UAV and field camera over cauliflower',
        'reason': 'GrowliFlower: cauliflower field imagery; downstream holdout.',
    },
    'cropandweed': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground robot',
        'reason': 'CropAndWeed: field imagery for robotic weeding; downstream holdout.',
    },
    'agrifm_ref88': {
        'acquisition_domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'ground robot over sugar beet plots',
        'reason': 'Agri-FM+ reference to PhenoBench; same physical dataset, counted once.',
    },
    'agrifm_ref93': {
        'acquisition_domain': 'CONTROLLED_LAB',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'platform': 'bench photography, uniform background',
        'reason': ('PlantVillage: detached leaves photographed on uniform grey/white '
                   'backgrounds in a controlled setting. It is the canonical '
                   'CONTROLLED_LAB source and is excluded from Field-D1 by construction.'),
    },
}

#: Sources that are re-uploads of another row and must not be counted twice.
SAME_PHYSICAL_DATASET = {
    'agrifm_ref4': 'perennial_plants',
    'agrifm_ref79': 'ronin',
    'agrifm_ref88': 'phenobench',
    'agrifm_ref31': 'gwhd_2021',
    'weedcrop': 'ronin',
}


def label_for(source_id: str) -> Dict[str, str]:
    """Return the ledger entry, or a conservative UNKNOWN row."""
    entry = SOURCE_DOMAINS.get(source_id)
    if entry is None:
        return {
            'acquisition_domain': 'UNKNOWN',
            'evidence_type': 'none',
            'confidence': 'unknown',
            'platform': '',
            'reason': 'no acquisition-domain evidence recorded for this source',
        }
    return dict(entry)


def field_sources(ledger: Optional[Mapping[str, Mapping[str, str]]] = None) -> List[str]:
    ledger = ledger or SOURCE_DOMAINS
    return sorted(sid for sid, entry in ledger.items()
                  if entry.get('acquisition_domain') in ('FIELD_IN_SITU', 'ORCHARD_IN_SITU'))
