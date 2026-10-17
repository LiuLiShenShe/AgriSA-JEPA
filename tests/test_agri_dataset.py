from __future__ import annotations

import csv

import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.datasets.agri_dataset import AgriFieldDataset, make_agrifield40k
from src.masks.multiblock import MaskCollator


def write_manifest(path, image_path, status="include"):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "path", "split", "source", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-0001",
                "path": str(image_path),
                "split": "train",
                "source": "test",
                "status": status,
            }
        )


def test_dataset_converts_rgb_and_returns_sample_id(tmp_path):
    image_path = tmp_path / "gray.png"
    Image.new("L", (32, 24), color=128).save(image_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, image_path)

    dataset = AgriFieldDataset(
        manifest,
        transform=transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()]),
    )
    image, sample_id = dataset[0]
    assert image.shape == (3, 16, 16)
    assert sample_id == "sample-0001"


def test_loader_matches_ijepa_mask_interface(tmp_path):
    image_paths = []
    for index in range(4):
        image_path = tmp_path / f"image-{index}.jpg"
        Image.new("RGB", (32, 32), color=(index, 10, 20)).save(image_path)
        image_paths.append(image_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "path", "split", "source", "status"],
        )
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            writer.writerow(
                {
                    "sample_id": f"sample-{index}",
                    "path": str(image_path),
                    "split": "train",
                    "source": "test",
                    "status": "include",
                }
            )

    collator = MaskCollator(
        input_size=16,
        patch_size=4,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.4),
        aspect_ratio=(0.5, 2.0),
        nenc=1,
        npred=2,
        min_keep=2,
    )
    _, loader, _ = make_agrifield40k(
        transform=transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()]),
        batch_size=2,
        collator=collator,
        pin_mem=False,
        num_workers=0,
        manifest_path=manifest,
        split="train",
        seed=7,
    )
    batch, masks_enc, masks_pred = next(iter(loader))
    assert batch[0].shape == (2, 3, 16, 16)
    assert len(masks_enc) == 1
    assert len(masks_pred) == 2
    assert masks_enc[0].shape[0] == 2
    assert masks_pred[0].shape[0] == 2
    assert masks_enc[0].dtype == torch.int64


def test_decode_errors_are_explicit(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, tmp_path / "missing.jpg")
    dataset = AgriFieldDataset(manifest)
    with pytest.raises(RuntimeError, match="Failed to decode RGB image"):
        dataset[0]


def test_excluded_rows_are_not_loaded(tmp_path):
    image_path = tmp_path / "excluded.jpg"
    Image.new("RGB", (8, 8), color="red").save(image_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, image_path, status="exclude")
    with pytest.raises(ValueError, match="no included samples"):
        AgriFieldDataset(manifest)


def test_max_samples_must_be_positive(tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8), color="red").save(image_path)
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, image_path)
    with pytest.raises(ValueError, match="max_samples must be positive"):
        AgriFieldDataset(manifest, max_samples=0)
