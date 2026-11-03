#!/usr/bin/env python
"""DataLoader compatibility smoke test for the published manifest.

Reads ``data/corpus_v3/R3v2_strict_rgb_train.csv``, resolves each image through
the per-source ledger, and pulls **two batches** through a minimal torch
Dataset.  No model is built and no optimizer step is taken; this only proves
that the manifest paths decode and collate.

Outputs ``reports/stage5d_loader_smoke.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ManifestDataset(Dataset):
    def __init__(self, ids: List[str], paths: Dict[str, str], size: int = 224):
        self.ids = ids
        self.paths = paths
        self.size = size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        with Image.open(self.paths[image_id]) as image:
            rgb = image.convert('RGB').resize((self.size, self.size),
                                              Image.Resampling.BILINEAR)
            array = torch.from_numpy(
                __import__('numpy').asarray(rgb, dtype='uint8').copy())
        return array.permute(2, 0, 1).float() / 255.0, image_id


def main() -> int:
    train = pd.read_csv(ROOT / 'data/corpus_v3/R3v2_strict_rgb_train.csv',
                        low_memory=False)
    wanted = set(train['image_id'].astype(str))
    paths: Dict[str, str] = {}
    for path in sorted((ROOT / 'data/corpus_v3/images').glob('*.csv')):
        frame = pd.read_csv(path, low_memory=False,
                            usecols=['image_id', 'raw_path'])
        frame = frame[frame['image_id'].astype(str).isin(wanted)]
        paths.update(dict(zip(frame['image_id'].astype(str),
                              frame['raw_path'].astype(str))))
    missing = len(wanted) - len(paths)
    ids = sorted(paths)
    dataset = ManifestDataset(ids, paths)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    batches = []
    for index, (images, batch_ids) in enumerate(loader):
        if index >= 2:
            break
        assert images.shape[1:] == (3, 224, 224), images.shape
        assert torch.isfinite(images).all()
        batches.append({'batch': index,
                        'shape': list(images.shape),
                        'min': float(images.min()),
                        'max': float(images.max()),
                        'first_id': str(batch_ids[0])})
    payload = {
        'train_rows': int(len(train)),
        'resolved_paths': int(len(paths)),
        'unresolved': int(missing),
        'batches': batches,
        'optimizer_updates': 0,
        'note': 'two batches only; no model, no optimizer, no pretraining',
    }
    (ROOT / 'reports/stage5d_loader_smoke.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0 if not missing and len(batches) == 2 else 1


if __name__ == '__main__':
    raise SystemExit(main())
