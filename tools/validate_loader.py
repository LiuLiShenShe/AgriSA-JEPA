"""Validate the audited real-image loader and official mask interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.datasets.agri_dataset import AgriFieldDataset, make_agrifield40k, seed_worker
from src.masks.multiblock import MaskCollator
from src.transforms import make_transforms


def collect_deterministic_batches(manifest: Path, root: Path, seed: int, workers: int):
    transform = make_transforms(
        crop_size=224,
        crop_scale=(0.3, 1.0),
        horizontal_flip=True,
    )
    dataset = AgriFieldDataset(manifest, transform=transform, split='train',
                               max_samples=8, root_path=root)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return [(images.clone(), list(ids)) for images, ids in loader]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('pretrain_manifest.csv'))
    parser.add_argument('--root', type=Path, default=Path('/data/datasets/AgriField-40K'))
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('reports/loader_validation.json'))
    args = parser.parse_args()

    first = collect_deterministic_batches(args.manifest, args.root, args.seed, args.workers)
    second = collect_deterministic_batches(args.manifest, args.root, args.seed, args.workers)
    repeatable = all(
        torch.equal(left_images, right_images) and left_ids == right_ids
        for (left_images, left_ids), (right_images, right_ids) in zip(first, second)
    )
    if not repeatable:
        raise AssertionError('deterministic real-image loader batches differ')

    mask_collator = MaskCollator(
        input_size=224, patch_size=16, enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5),
        nenc=1, npred=4, min_keep=10, allow_overlap=False,
    )
    transform = make_transforms(crop_size=224, crop_scale=(0.3, 1.0))
    _, loader, _ = make_agrifield40k(
        transform=transform, batch_size=4, collator=mask_collator,
        pin_mem=False, num_workers=args.workers, manifest_path=args.manifest,
        root_path=args.root, split='train', seed=args.seed, max_samples=8,
    )
    batch, masks_enc, masks_pred = next(iter(loader))
    images, sample_ids = batch
    if not torch.isfinite(images).all():
        raise AssertionError('real image batch contains non-finite tensor values')
    if len(sample_ids) != images.shape[0]:
        raise AssertionError('sample metadata does not match batch dimension')
    num_patches = (224 // 16) ** 2
    mask_summary = []
    for name, masks in [('enc', masks_enc), ('pred', masks_pred)]:
        for mask in masks:
            if mask.ndim != 2 or mask.shape[0] != images.shape[0]:
                raise AssertionError(f'{name} mask shape {tuple(mask.shape)}')
            if mask.dtype != torch.int64:
                raise AssertionError(f'{name} mask dtype {mask.dtype}')
            if int(mask.min()) < 0 or int(mask.max()) >= num_patches:
                raise AssertionError(f'{name} mask index out of range')
            if any(len(torch.unique(row)) != len(row) for row in mask):
                raise AssertionError(f'{name} mask contains duplicate patch indices')
            mask_summary.append({'kind': name, 'shape': list(mask.shape),
                                 'min': int(mask.min()), 'max': int(mask.max())})

    summary = {
        'manifest': str(args.manifest.resolve()),
        'root': str(args.root.resolve()),
        'real_images_checked': 8,
        'image_shape': list(images.shape),
        'sample_ids': list(sample_ids),
        'repeatable_with_workers': args.workers,
        'repeatable': repeatable,
        'mask_summary': mask_summary,
        'mask_interface': 'official MaskCollator -> (batch, masks_enc, masks_pred)',
        'augmentation': 'RandomResizedCrop(224), optional RandomHorizontalFlip, ToTensor, ImageNet normalization',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
