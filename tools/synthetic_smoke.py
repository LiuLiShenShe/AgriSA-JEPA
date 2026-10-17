#!/usr/bin/env python
"""Run a small synthetic I-JEPA optimization and checkpoint smoke test."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.helper import init_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="reports/synthetic_smoke.json")
    parser.add_argument("--checkpoint", default="runs/synthetic_smoke/checkpoint.pth")
    return parser.parse_args()


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(batch_size: int, crop_size: int, device: torch.device):
    images = torch.randn(batch_size, 3, crop_size, crop_size, device=device)
    samples = [(images[index].cpu(), f"synthetic-{index}") for index in range(batch_size)]
    return images, samples


def take_step(
    encoder,
    predictor,
    target_encoder,
    optimizer,
    collator,
    images,
    samples,
    device,
    momentum,
):
    _, masks_enc, masks_pred = collator(samples)
    masks_enc = [mask.to(device, non_blocking=True) for mask in masks_enc]
    masks_pred = [mask.to(device, non_blocking=True) for mask in masks_pred]

    with torch.no_grad():
        target = target_encoder(images)
        target = F.layer_norm(target, (target.size(-1),))
        target = apply_masks(target, masks_pred)
        target = repeat_interleave_batch(target, len(images), repeat=len(masks_enc))

    context = encoder(images, masks_enc)
    prediction = predictor(context, masks_enc, masks_pred)
    loss = F.smooth_l1_loss(prediction, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        for query, key in zip(encoder.parameters(), target_encoder.parameters()):
            key.data.mul_(momentum).add_((1.0 - momentum) * query.detach().data)

    return float(loss.detach().cpu()), len(masks_enc[0][0]), len(masks_pred[0][0])


def main() -> None:
    args = parse_args()
    if args.steps < 10:
        raise ValueError("The smoke test requires at least 10 optimizer updates")
    seed_all()

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = requested_device
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    encoder, predictor = init_model(
        device=device,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_emb_dim=384,
        model_name="vit_base",
    )
    target_encoder = copy.deepcopy(encoder)
    for parameter in target_encoder.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters()},
            {"params": predictor.parameters()},
        ],
        lr=1e-4,
    )
    collator = MaskCollator(
        input_size=args.crop_size,
        patch_size=args.patch_size,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
    )

    initial_encoder = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    initial_target = {key: value.detach().cpu().clone() for key, value in target_encoder.state_dict().items()}
    losses = []
    mask_sizes = []
    step_times_ms = []
    images, samples = make_batch(args.batch_size, args.crop_size, device)
    for _ in range(args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss, enc_keep, pred_keep = take_step(
            encoder,
            predictor,
            target_encoder,
            optimizer,
            collator,
            images,
            samples,
            device,
            momentum=0.996,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times_ms.append((time.perf_counter() - start) * 1000.0)
        losses.append(loss)
        mask_sizes.append({"enc_keep": enc_keep, "pred_keep": pred_keep})
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {len(losses)}: {loss}")

    encoder_changed = any(
        not torch.equal(initial_encoder[key], value.detach().cpu())
        for key, value in encoder.state_dict().items()
    )
    target_changed = any(
        not torch.equal(initial_target[key], value.detach().cpu())
        for key, value in target_encoder.state_dict().items()
    )
    if not encoder_changed or not target_changed:
        raise RuntimeError(
            f"Parameter update check failed: encoder_changed={encoder_changed}, "
            f"target_changed={target_changed}"
        )

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.steps,
            "loss": losses[-1],
        },
        checkpoint_path,
    )

    encoder2, predictor2 = init_model(
        device=device,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        pred_depth=args.pred_depth,
        pred_emb_dim=384,
        model_name="vit_base",
    )
    target_encoder2 = copy.deepcopy(encoder2)
    optimizer2 = torch.optim.AdamW(
        [
            {"params": encoder2.parameters()},
            {"params": predictor2.parameters()},
        ],
        lr=1e-4,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder2.load_state_dict(checkpoint["encoder"])
    predictor2.load_state_dict(checkpoint["predictor"])
    target_encoder2.load_state_dict(checkpoint["target_encoder"])
    optimizer2.load_state_dict(checkpoint["optimizer"])
    if checkpoint["step"] != args.steps:
        raise RuntimeError("Checkpoint step metadata did not round-trip")
    for left, right in zip(encoder.parameters(), encoder2.parameters()):
        if not torch.equal(left, right):
            raise RuntimeError("Encoder checkpoint reload mismatch")
    for left, right in zip(target_encoder.parameters(), target_encoder2.parameters()):
        if not torch.equal(left, right):
            raise RuntimeError("Target encoder checkpoint reload mismatch")

    if device.type == "cuda":
        peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory_mib = None
    summary = {
        "kind": "synthetic_engineering_smoke_test",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": "vit_base",
        "crop_size": args.crop_size,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "optimizer_updates": args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_all_finite": bool(all(np.isfinite(loss) for loss in losses)),
        "encoder_updated": encoder_changed,
        "ema_target_updated": target_changed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload_verified": True,
        "peak_memory_mib": peak_memory_mib,
        "step_time_ms_mean": float(np.mean(step_times_ms)),
        "step_time_ms_last": step_times_ms[-1],
        "mask_sizes": mask_sizes,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
