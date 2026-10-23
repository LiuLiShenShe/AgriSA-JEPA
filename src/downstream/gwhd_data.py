"""GWHD 2021 loading, label validation, and official split handling.

The Zenodo release (record 5092309) ships one archive with the images and three
labelled CSVs.  We never invent a split: the three official files are used
verbatim, and a missing/duplicated image name is a hard error.

Label format
------------
The release stores boxes as a string per image (a Python/JSON list of
``[x1, y1, x2, y2]``).  ``parse_boxes`` also accepts ``x1,y1,x2,y2;`` groups so
a format change cannot silently produce empty targets.

Validation
----------
``validate_boxes`` does not silently fix training data.  It reports, per split:

* boxes with non-positive width/height (degenerate)
* boxes outside ``[0, width] x [0, height]`` (out of bounds)
* empty annotations
* duplicate boxes

Training drops degenerate and fully-outside boxes (they cannot be regressed),
and clips partially outside boxes; the audit report keeps the raw counts.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

BOX_COLUMNS = ('bbox', 'boxes', 'BoxesString', 'box', 'annotations')
NAME_COLUMNS = ('image_name', 'image', 'filename', 'file_name', 'img_name')
SPLIT_PATTERNS = {
    'train': re.compile(r'(^|[_-])(train)([_-]|$)', re.I),
    'val': re.compile(r'(^|[_-])(val|validation)([_-]|$)', re.I),
    'test': re.compile(r'(^|[_-])(test)([_-]|$)', re.I),
}


def parse_boxes(value) -> List[Tuple[float, float, float, float]]:
    """Parse the many textual box encodings into a list of xyxy float tuples."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        if items and isinstance(items[0], (int, float)) and len(items) == 4:
            items = [items]
        return [tuple(float(v) for v in item[:4]) for item in items]
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', '[]', '{}'):
        return []
    try:
        parsed = json.loads(text)
        return parse_boxes(parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    boxes: List[Tuple[float, float, float, float]] = []
    for group in re.split(r'[;|]', text):
        numbers = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', group)]
        for start in range(0, len(numbers) - 3, 4):
            boxes.append(tuple(numbers[start:start + 4]))  # type: ignore[arg-type]
    return boxes


def discover_split_files(root: Path) -> Dict[str, Path]:
    """Find the three official label CSVs and map them to split names."""
    found: Dict[str, Path] = {}
    for path in sorted(Path(root).rglob('*.csv')):
        name = path.stem.lower()
        for split, pattern in SPLIT_PATTERNS.items():
            if pattern.search(name):
                # competition_train.csv, train.csv, gwhd_train_labels.csv ...
                if split not in found:
                    found[split] = path
    if set(found) != {'train', 'val', 'test'}:
        raise FileNotFoundError(
            f'expected train/val/test CSVs under {root}, found {sorted(found)}')
    return found


def _column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


@dataclass
class GWHDSplit:
    name: str
    csv_path: Path
    images: List[str]
    boxes: List[List[Tuple[float, float, float, float]]]
    widths: List[int]
    heights: List[int]

    def __len__(self) -> int:
        return len(self.images)


def read_split(csv_path: Path, root: Path, name: str) -> GWHDSplit:
    images: List[str] = []
    boxes: List[List[Tuple[float, float, float, float]]] = []
    widths: List[int] = []
    heights: List[int] = []
    with open(csv_path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'{csv_path} has no header')
        box_col = _column(reader.fieldnames, BOX_COLUMNS)
        name_col = _column(reader.fieldnames, NAME_COLUMNS)
        width_col = _column(reader.fieldnames, ('width', 'img_width'))
        height_col = _column(reader.fieldnames, ('height', 'img_height'))
        if box_col is None or name_col is None:
            raise ValueError(
                f'{csv_path}: cannot find box/name columns in {reader.fieldnames}')
        for row in reader:
            image = str(row[name_col]).strip()
            if not image:
                continue
            images.append(image)
            boxes.append(parse_boxes(row.get(box_col)))
            widths.append(int(float(row[width_col])) if width_col and row.get(width_col) else 0)
            heights.append(int(float(row[height_col])) if height_col and row.get(height_col) else 0)
    if len({p.lower() for p in images}) != len(images):
        raise ValueError(f'{csv_path}: duplicate image names inside one split')
    return GWHDSplit(name, csv_path, images, boxes, widths, heights)


@dataclass
class BoxValidation:
    images: int
    boxes: int
    images_without_boxes: int
    degenerate: int
    out_of_bounds: int
    partially_outside: int
    duplicate_boxes: int
    suspicious_xy_swap: int
    examples: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return {
            'images': self.images, 'boxes': self.boxes,
            'images_without_boxes': self.images_without_boxes,
            'degenerate': self.degenerate,
            'out_of_bounds': self.out_of_bounds,
            'partially_outside': self.partially_outside,
            'duplicate_boxes': self.duplicate_boxes,
            'suspicious_xy_swap': self.suspicious_xy_swap,
            'examples': self.examples,
        }


def validate_boxes(split: GWHDSplit, width: int = 1024, height: int = 1024,
                   max_examples: int = 20) -> BoxValidation:
    """Count label defects without altering the split."""
    degenerate = out_of_bounds = partially_outside = duplicates = swapped = 0
    empty = 0
    examples: List[Dict[str, object]] = []
    for image, boxes, w, h in zip(split.images, split.boxes,
                                  split.widths, split.heights):
        w = w or width
        h = h or height
        if not boxes:
            empty += 1
            continue
        seen = set()
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:
                degenerate += 1
                if len(examples) < max_examples:
                    examples.append({'image': image, 'box': box,
                                     'issue': 'degenerate'})
                continue
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                    out_of_bounds += 1
                    issue = 'fully_outside'
                else:
                    partially_outside += 1
                    issue = 'partially_outside'
                if len(examples) < max_examples:
                    examples.append({'image': image, 'box': box, 'issue': issue,
                                     'width': w, 'height': h})
            key = tuple(round(v, 2) for v in box)
            if key in seen:
                duplicates += 1
            seen.add(key)
            # A box wider than it is tall by >4x, on an upright wheat head, is
            # suspicious but not necessarily wrong; count only.
            if (x2 - x1) > 4 * (y2 - y1):
                swapped += 1
    return BoxValidation(len(split.images), sum(len(b) for b in split.boxes),
                         empty, degenerate, out_of_bounds, partially_outside,
                         duplicates, swapped, examples)


def clip_boxes(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2].clamp_(min=0, max=width)
    boxes[:, 1::2].clamp_(min=0, max=height)
    return boxes


def keep_valid_boxes(boxes: torch.Tensor, width: int, height: int,
                     min_size: float = 2.0) -> torch.Tensor:
    """Clip to the image and drop boxes with < ``min_size`` side after clipping."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    boxes = clip_boxes(boxes, width, height)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (widths >= min_size) & (heights >= min_size)
    return boxes[keep]


def image_group(image_name: str) -> str:
    """GWHD image names start with the acquisition domain, e.g. ``ethz_1_...``."""
    stem = Path(image_name).stem
    match = re.match(r'([A-Za-z]+)', stem)
    return match.group(1).lower() if match else stem.split('_')[0].lower()


class GWHDDataset(Dataset):
    """One official GWHD split.

    Images are resized to ``image_size`` (default 518) *only* through the
    detector's own transform at evaluation time, so the dataset returns the
    original image tensor and original-pixel boxes.  Horizontal flip is the
    only training augmentation: it is exact for boxes and identical for all
    arms.
    """

    def __init__(self, root: Path, split: GWHDSplit, train: bool = False,
                 hflip_prob: float = 0.5, min_box_size: float = 2.0) -> None:
        self.root = Path(root)
        self.split = split
        self.train = train
        self.hflip_prob = hflip_prob
        self.min_box_size = min_box_size

    def __len__(self) -> int:
        return len(self.split)

    def _image_path(self, name: str) -> Path:
        direct = self.root / name
        if direct.is_file():
            return direct
        matches = list(self.root.rglob(Path(name).name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f'{name}: expected exactly one match under {self.root}, '
                f'found {len(matches)}')
        return matches[0]

    def __getitem__(self, index: int):
        name = self.split.images[index]
        boxes = torch.tensor(self.split.boxes[index], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        with Image.open(self._image_path(name)) as image:
            image = image.convert('RGB')
            width, height = image.size
            if self.train and torch.rand(()) < self.hflip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if boxes.numel():
                    x1 = boxes[:, 0].clone()
                    boxes[:, 0] = width - boxes[:, 2]
                    boxes[:, 2] = width - x1
            array = np.ascontiguousarray(image, dtype=np.uint8)
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
        boxes = keep_valid_boxes(boxes, width, height, self.min_box_size)
        target = {
            'boxes': boxes,
            'labels': torch.ones((boxes.shape[0],), dtype=torch.int64),
            'image_id': torch.tensor([index]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    if boxes.numel() else torch.zeros((0,)),
            'iscrowd': torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def split_isolation_report(splits: Dict[str, GWHDSplit]) -> Dict[str, object]:
    """Check that no image (or acquisition domain) leaks across official splits."""
    names = {k: {n.lower() for n in v.images} for k, v in splits.items()}
    groups = {k: {image_group(n) for n in v.images} for k, v in splits.items()}
    overlaps = {}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        shared_names = names[a] & names[b]
        shared_groups = groups[a] & groups[b]
        overlaps[f'{a}_{b}'] = {
            'shared_image_names': len(shared_names),
            'shared_groups': sorted(shared_groups),
            'shared_group_count': len(shared_groups),
        }
    return {
        'counts': {k: len(v) for k, v in splits.items()},
        'groups': {k: len(v) for k, v in groups.items()},
        'overlaps': overlaps,
    }
