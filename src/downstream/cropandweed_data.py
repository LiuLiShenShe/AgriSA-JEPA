"""CropAndWeed loading, official CropsOrWeed9 label mapping, and split logic.

Label mapping
-------------
The official ``cnw/utilities/datasets.py`` (vendored verbatim under
``data/holdout_raw/cropandweed_official_code/``) defines the evaluation variant
``CropsOrWeed9``: eight crop classes plus one merged ``Weed`` class, each
mapping a list of source label IDs.  Two source IDs have **no** mapping in
CropsOrWeed9 and are therefore ignore for both the loss and the metric:

* ``0``   Soil
* ``255`` Vegetation (fallback class of the Experimental Set)

The mapping below is transcribed from that file; a unit test cross-checks it
against the vendored source so a transcription error cannot survive.

Splits
------
The dataset ships no split CSV.  The official README semantics are:

* ``ave_*`` Application Set — collected for training;
* ``vwg_*`` Experimental Set — collected for evaluation.

Because the recording-session id (the 4-digit field after the prefix) is
independently numbered in the two sets, the group key is the full prefixed
session (``ave-0540``), never the bare number.  Training uses the full
Application Set.  Evaluation uses one image per Experimental-Set session,
sampled with a fixed seed, so every session is represented and no session
dominates the metric.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

NUM_CLASSES = 9
IGNORE_ID = -1

CLASS_NAMES = ('Maize', 'Sugar beet', 'Soy', 'Sunflower', 'Potato', 'Pea',
               'Bean', 'Pumpkin', 'Weed')

# Transcribed from data/holdout_raw/cropandweed_official_code/datasets.py,
# DATASETS['CropsOrWeed9'].  (label_id, name, [source ids]).
_CROPS_OR_WEED9_SOURCES: Tuple[Tuple[int, str, Tuple[int, ...]], ...] = (
    (0, 'Maize', (1, 2, 3, 4, 5, 6)),
    (1, 'Sugar beet', (7, 8, 9, 10, 11, 12)),
    (2, 'Soy', (94,)),
    (3, 'Sunflower', (24,)),
    (4, 'Potato', (18,)),
    (5, 'Pea', (13,)),
    (6, 'Bean', (26, 27)),
    (7, 'Pumpkin', (15,)),
    (8, 'Weed', (31, 48, 62, 65, 68, 69, 74, 75, 81, 84, 86, 32, 29, 33, 37,
                 49, 30, 44, 66, 87, 89, 91, 61, 79, 34, 41, 52, 35, 36, 78,
                 38, 39, 71, 72, 88, 42, 45, 70, 47, 51, 54, 58, 60, 80, 83,
                 96, 22, 63, 85, 56, 57, 64, 77, 50, 59, 67, 76)),
)


def build_source_to_class() -> Dict[int, int]:
    """source label ID -> CropsOrWeed9 class ID (0..8); Soil/Vegetation absent."""
    mapping: Dict[int, int] = {}
    for class_id, _name, sources in _CROPS_OR_WEED9_SOURCES:
        for source in sources:
            if source in mapping:
                raise ValueError(f'source id {source} mapped twice')
            mapping[source] = class_id
    return mapping


#: Full lookup table for uint8 labelIds masks.
def map_label_ids(mask: np.ndarray) -> np.ndarray:
    """Map a raw labelIds mask to CropsOrWeed9 ids; unmapped -> IGNORE_ID."""
    source_to_class = build_source_to_class()
    out = np.full(mask.shape, IGNORE_ID, dtype=np.int16)
    for source, class_id in source_to_class.items():
        out[mask == source] = class_id
    return out


def session_key(image_name: str) -> str:
    """``ave-0540-0012.jpg`` -> ``ave-0540`` (prefix + recording session)."""
    stem = Path(image_name).stem
    parts = stem.split('-')
    if len(parts) != 3:
        raise ValueError(f'unexpected CropAndWeed image name: {image_name}')
    return f'{parts[0]}-{parts[1]}'


def list_images(images_root: Path, prefix: str) -> List[str]:
    """Sorted image names with the given ``ave``/``vwg`` prefix."""
    return sorted(p.name for p in images_root.iterdir()
                  if p.is_file() and p.name.startswith(f'{prefix}-')
                  and p.suffix.lower() == '.jpg')


def session_stratified_sample(names: Sequence[str], per_session: int,
                              seed: int) -> List[str]:
    """One (or ``per_session``) image per recording session, seeded and sorted."""
    by_session: Dict[str, List[str]] = {}
    for name in names:
        by_session.setdefault(session_key(name), []).append(name)
    rng = random.Random(seed)
    picked: List[str] = []
    for session in sorted(by_session):
        candidates = sorted(by_session[session])
        if per_session >= len(candidates):
            picked.extend(candidates)
        else:
            picked.extend(sorted(rng.sample(candidates, per_session)))
    return sorted(picked)


def build_split(images_root: Path, sampling_seed: int,
                per_session: int = 1) -> Dict[str, List[str]]:
    """Train = full ave_ set; eval = seeded session-stratified vwg_ sample."""
    train = list_images(images_root, 'ave')
    vwg_all = list_images(images_root, 'vwg')
    eval_names = session_stratified_sample(vwg_all, per_session, sampling_seed)
    sessions_train = {session_key(n) for n in train}
    sessions_eval = {session_key(n) for n in eval_names}
    overlap = sessions_train & sessions_eval
    # prefixed session keys are disjoint by construction; a hit means the
    # naming convention changed and the split must not be trusted.
    if overlap:
        raise ValueError(f'session leak between ave_/vwg_: {sorted(overlap)[:5]}')
    return {'train': train, 'eval': eval_names,
            'eval_pool_size': len(vwg_all),
            'eval_sessions': len(sessions_eval)}


def load_image_and_mask(images_root: Path, annotation_root: Path,
                        name: str) -> Tuple[np.ndarray, np.ndarray]:
    """RGB uint8 image and CropsOrWeed9-mapped label (IGNORE_ID where unmapped)."""
    image_path = images_root / name
    mask_path = annotation_root / f'{Path(name).stem}.png'
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    image = np.asarray(Image.open(image_path).convert('RGB'))
    raw_mask = np.asarray(Image.open(mask_path))
    if image.shape[:2] != raw_mask.shape[:2]:
        raise ValueError(f'{name}: image {image.shape[:2]} vs mask '
                         f'{raw_mask.shape[:2]}')
    return image, map_label_ids(raw_mask)


def split_counts(split: Dict[str, List[str]]) -> Dict[str, int]:
    return {'train_images': len(split['train']),
            'eval_images': len(split['eval']),
            'eval_pool': split.get('eval_pool_size', len(split['eval']))}


def write_split_csv(split: Dict[str, List[str]], path: Path) -> None:
    """Persist the split decision so Phase 4 evaluates the exact same names."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'image', 'session'])
        for part in ('train', 'eval'):
            for name in split[part]:
                writer.writerow([part, name, session_key(name)])