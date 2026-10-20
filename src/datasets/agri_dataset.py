"""Manifest-backed agricultural image dataset for I-JEPA pretraining."""

from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy transforms from the PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_manifest(
    manifest_path: str | os.PathLike[str],
    split: Optional[str],
    max_samples: Optional[int],
    root_path: Optional[str | os.PathLike[str]],
) -> List[Dict[str, str]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    image_root = Path(root_path).expanduser().resolve() if root_path else path.parent
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        required = {"sample_id", "path", "split", "status"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing columns {sorted(missing)}: {path}")

        rows: List[Dict[str, str]] = []
        for row in reader:
            if row.get("status", "").strip().lower() != "include":
                continue
            if split is not None and row.get("split", "").strip() != split:
                continue
            image_path = Path(row["path"]).expanduser()
            if not image_path.is_absolute():
                image_path = (image_root / image_path).resolve()
            row["path"] = str(image_path)
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        split_text = f" and split={split!r}" if split is not None else ""
        raise ValueError(f"Manifest has no included samples{split_text}: {path}")
    return rows


class AgriFieldDataset(Dataset):
    """Read RGB images listed by an audited manifest.

    The second return value is a sample identifier, not a class target. The
    upstream I-JEPA collator expects an ImageFolder-like ``(image, target)``
    sample and the training loop ignores this metadata.

    ``augmentation_seed`` isolates the augmentation random stream from the
    mask-sampling random stream.  When it is set, the transform is executed
    with the global torch RNG temporarily replaced by a generator seeded from
    ``(augmentation_seed, epoch, index)``; the global RNG state is restored
    afterwards, so whatever the mask collator draws cannot shift the crops.
    The augmentation *protocol* is unchanged -- only the source of its
    randomness moves -- and different arms therefore see byte-identical
    augmented images, which is what makes a mask-only comparison fair.
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        transform: Optional[Callable] = None,
        split: Optional[str] = "train",
        max_samples: Optional[int] = None,
        root_path: Optional[str | os.PathLike[str]] = None,
        augmentation_seed: Optional[int] = None,
    ) -> None:
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.transform = transform
        self.samples = _read_manifest(self.manifest_path, split, max_samples, root_path)
        self.augmentation_seed = augmentation_seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive the per-sample augmentation seed."""
        self._epoch = int(epoch)

    def _derived_seed(self, index: int) -> int:
        digest = hashlib.sha256(
            f"{self.augmentation_seed}:{self._epoch}:{index}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[object, str]:
        row = self.samples[index]
        image_path = Path(row["path"])
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to decode RGB image at {image_path} "
                f"(sample_id={row.get('sample_id', index)}): {exc}"
            ) from exc

        if self.transform is not None:
            try:
                if self.augmentation_seed is None:
                    image = self.transform(image)
                else:
                    state = torch.get_rng_state()
                    try:
                        torch.manual_seed(self._derived_seed(index))
                        image = self.transform(image)
                    finally:
                        torch.set_rng_state(state)
            except Exception as exc:
                raise RuntimeError(
                    f"Transform failed for {image_path} "
                    f"(sample_id={row.get('sample_id', index)}): {exc}"
                ) from exc
        return image, row.get("sample_id", str(index))


def make_agrifield40k(
    transform: Optional[Callable],
    batch_size: int,
    collator=None,
    pin_mem: bool = True,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    manifest_path: str | os.PathLike[str] = "pretrain_manifest.csv",
    root_path: Optional[str | os.PathLike[str]] = None,
    split: str = "train",
    drop_last: bool = True,
    seed: int = 0,
    max_samples: Optional[int] = None,
    augmentation_seed: Optional[int] = None,
):
    """Build an I-JEPA-compatible loader from an audited manifest."""
    dataset = AgriFieldDataset(
        manifest_path=manifest_path,
        transform=transform,
        split=split,
        max_samples=max_samples,
        root_path=root_path,
        augmentation_seed=augmentation_seed,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        seed=seed,
        drop_last=drop_last,
    )
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader, sampler


__all__ = ["AgriFieldDataset", "make_agrifield40k", "seed_worker"]
