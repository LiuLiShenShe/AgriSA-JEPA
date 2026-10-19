"""PhenoBench semantic-segmentation data access for frozen-backbone probing.

The class ids, the partial-annotation rule and the evaluation split come from the
official PhenoBench development kit (``pip install phenobench``), not from this
project.  This module only reads pixels and labels and hands them to a probe; it
does not redefine the task.

Official semantic ids (``phenobench/phenobench_loader.py`` and
``evaluation/auxiliary/convert.py``)::

    0 - soil / background
    1 - crop
    2 - weed
    3 - partial crop
    4 - partial weed

The official semantic metric evaluates THREE classes.  ``convert_partial_semantics``
maps ``3 -> 1`` and ``4 -> 2`` before scoring, so "partial" plants are scored as
their parent class rather than ignored.  This module applies exactly that mapping
when building training targets, so probe supervision and the official metric agree
on what a label means.  Nothing is silently dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Official PhenoBench semantic label ids.
SOIL = 0
CROP = 1
WEED = 2
PARTIAL_CROP = 3
PARTIAL_WEED = 4

#: Number of classes the official semantic metric scores.
NUM_CLASSES = 3

#: Ordered class names as reported by ``phenobench-eval --task semantics``.
CLASS_NAMES = ('soil', 'crop', 'weed')

#: Image / label size of the official v1.1.0 release.
IMAGE_SIZE = 1024


def map_partial_to_parent(label: np.ndarray) -> np.ndarray:
    """Reproduce the official ``convert_partial_semantics`` mapping.

    ``3 -> 1`` (partial crop -> crop) and ``4 -> 2`` (partial weed -> weed).
    Any other value passes through unchanged.  Pure function.
    """
    out = label.astype(np.int16, copy=True)
    out[out == PARTIAL_CROP] = CROP
    out[out == PARTIAL_WEED] = WEED
    return out


def label_histogram(label: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(label, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def list_split_files(root: str | Path, split: str) -> List[str]:
    """Sorted image filenames of a split; the evaluator joins on these."""
    image_dir = Path(root) / split / 'images'
    if not image_dir.is_dir():
        raise FileNotFoundError(f'PhenoBench image dir not found: {image_dir}')
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix == '.png')
    if not names:
        raise ValueError(f'No PNG images in {image_dir}')
    return names


def load_image_and_label(root: str | Path, split: str, name: str
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Load one RGB image and its raw (unmapped) semantic label.

    Returns ``(image[H,W,3] uint8, label[H,W] int16)``.  The label keeps its
    official ids (including 3/4) so callers can decide the mapping themselves.
    """
    from PIL import Image

    root = Path(root)
    image_path = root / split / 'images' / name
    label_path = root / split / 'semantics' / name
    if not image_path.is_file():
        raise FileNotFoundError(f'Missing PhenoBench image: {image_path}')
    if not label_path.is_file():
        raise FileNotFoundError(f'Missing PhenoBench semantics: {label_path}')
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert('RGB'), dtype=np.uint8)
    with Image.open(label_path) as handle:
        label = np.asarray(handle, dtype=np.int16)
    if image.shape[:2] != label.shape[:2]:
        raise ValueError(
            f'Image/label shape mismatch for {name}: {image.shape} vs {label.shape}')
    return image, label


class PhenoBenchSemantic:
    """Minimal index-style dataset over the official train/val splits.

    Only ``images`` and ``semantics`` are read: this stage evaluates semantic
    segmentation, and instance/bbox targets are not used.  This is a data access
    object, not a re-definition of the benchmark.
    """

    def __init__(self, root: str | Path, split: str = 'val',
                 names: Optional[Sequence[str]] = None) -> None:
        if split not in ('train', 'val'):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.root = Path(root)
        self.split = split
        self.filenames = list(names) if names is not None else list_split_files(self.root, split)
        if not self.filenames:
            raise ValueError(f'Empty file list for split {split}')

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Dict[str, object]:
        name = self.filenames[index]
        image, label = load_image_and_label(self.root, self.split, name)
        return {'name': name, 'image': image, 'label': label}

    def label_histogram(self, limit: Optional[int] = None) -> Dict[int, int]:
        """Raw-id histogram over the split (None if a file cannot be read)."""
        total: Dict[int, int] = {}
        for name in self.filenames[:limit]:
            _, label = load_image_and_label(self.root, self.split, name)
            for value, count in label_histogram(label).items():
                total[value] = total.get(value, 0) + count
        return dict(sorted(total.items()))


def dataset_summary(root: str | Path) -> Dict[str, object]:
    """Counts of images / semantics / other targets per split."""
    root = Path(root)
    summary: Dict[str, object] = {'root': str(root), 'splits': {}}
    for split in ('train', 'val', 'test'):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        entry = {}
        for sub in sorted(p.name for p in split_dir.iterdir() if p.is_dir()):
            files = [p.name for p in (split_dir / sub).iterdir() if p.suffix == '.png']
            entry[sub] = len(files)
        summary['splits'][split] = entry
    return summary