#!/usr/bin/env python
"""Compute missing MUSIQ scores for a Stage 5C source without touching it.

Stage 5C loaded the *unenriched* ledger for two sources because the plain CSV
had a newer mtime, which silently turned every row into ``IQA_MISSING``.  This
tool recomputes the missing scores into a new Stage 5D directory and leaves
``data/corpus_v2`` untouched.

Usage
-----
    .venv-uv/bin/python tools/stage5d_compute_missing_iqa.py --source agrifm_ref1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import pandas as pd

DEFAULT_MAX_SIDE = 512


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ''
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--max-side', type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    import importlib.util
    import torch
    import pyiqa
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from torchvision import transforms

    source = args.source
    ledger = ROOT / f'data/corpus_v2/images/{source}.csv'
    if not ledger.exists():
        raise FileNotFoundError(ledger)
    frame = pd.read_csv(ledger, low_memory=False)
    if args.limit:
        frame = frame.head(args.limit)
    paths = frame['raw_path'].astype(str).tolist()
    print(f'{source}: scoring {len(paths)} images on {args.device}', flush=True)

    to_tensor = transforms.ToTensor()

    def load(path: str):
        try:
            with Image.open(path) as image:
                try:
                    image.draft('RGB', (args.max_side, args.max_side))
                except Exception:
                    pass
                picture = image.convert('RGB')
            if max(picture.size) > args.max_side:
                picture.thumbnail((args.max_side, args.max_side), Image.Resampling.BILINEAR)
            return picture
        except Exception:
            return None

    metric = pyiqa.create_metric('musiq', device=args.device)
    scores = []
    batch_size = 64
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            tensors = list(pool.map(load, batch))
            values = []
            for tensor in tensors:
                if tensor is None:
                    values.append(float('nan'))
                    continue
                with torch.inference_mode():
                    values.append(float(metric(to_tensor(tensor).unsqueeze(0).to(args.device))))
            scores.extend(values)
            done = min(start + batch_size, len(paths))
            if done % 320 == 0 or done == len(paths):
                print(f'  musiq {done}/{len(paths)}', flush=True)
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()

    frame['iqa_musiq'] = scores
    frame['iqa_niqe'] = float('nan')
    out_dir = ROOT / 'data/corpus_v3/derived_iqa'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{source}.csv'
    frame.to_csv(out_path, index=False)
    provenance = {
        'source_id': source,
        'rows': len(frame),
        'scored': int(pd.to_numeric(frame['iqa_musiq'], errors='coerce').notna().sum()),
        'device': args.device,
        'max_side': args.max_side,
        'model': 'musiq',
        'model_sha256': sha256_file(Path.home() / '.cache/torch/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth'),
        'pyiqa_version': pyiqa.__version__,
        'output': str(out_path),
        'output_sha256': sha256_file(out_path),
        'note': 'derived to repair the Stage 5C mtime-loader defect; corpus_v2 untouched',
    }
    (out_dir / f'{source}.provenance.json').write_text(
        json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
