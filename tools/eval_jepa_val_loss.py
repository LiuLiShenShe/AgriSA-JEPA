#!/usr/bin/env python
"""Offline I-JEPA validation-loss diagnostic for a checkpoint.

Computes the SAME smooth-L1 masked-feature-prediction objective the training loop
uses, on a held-out manifest, with the masks held FIXED across checkpoints so two
models are compared on identical masking.

IMPORTANT — what this number is and is not
------------------------------------------
It is a *diagnostic* self-supervised loss. It is NOT a downstream accuracy and
NOT a clean generalisation measure: `data/frozen/agrifield40k-v0_val.csv` has
known cross-split provenance leakage (see reports/leakage_check_v0.md —
ACRECropWeed frames 2 s apart across the split; 31 iNatWeeds observations
straddling). A lower value therefore means "the model predicts masked features
on this manifest better", not "the model generalises better". Never report it as
a result.

Usage:
    ./.venv/bin/python tools/eval_jepa_val_loss.py \
        --checkpoint runs/stage3_armB_pilot_500/jepa-latest.pth.tar \
        --manifest data/frozen/agrifield40k-v0_val.csv \
        --out reports/jepa_val_loss_pilot500.json \
        [--num-batches 32] [--batch-size 16] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.agri_dataset import make_agrifield40k  # noqa: E402
from src.helper import init_model  # noqa: E402
from src.masks.multiblock import MaskCollator as MBMaskCollator  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402
from src.transforms import make_transforms  # noqa: E402
from src.utils.tensors import repeat_interleave_batch  # noqa: E402

# -- mask geometry / transforms must match the training configs exactly so the
#    number is comparable across arms.
MASK = dict(patch_size=14, crop_size=224, allow_overlap=False,
            num_enc_masks=1, num_pred_masks=4, min_keep=10,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5))


def build_models(device, use_sdpa=True):
    encoder, predictor = init_model(
        device=device, patch_size=MASK['patch_size'], crop_size=MASK['crop_size'],
        pred_depth=12, pred_emb_dim=384, model_name='vit_huge', use_sdpa=use_sdpa)
    return encoder, predictor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', default='data/frozen/agrifield40k-v0_val.csv')
    parser.add_argument('--root-path', default='/data/datasets/AgriField-40K')
    parser.add_argument('--out', default=None)
    parser.add_argument('--num-batches', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-sdpa', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    # -- FIXED masks across every checkpoint: seed the collator deterministically
    torch.manual_seed(args.seed)
    mask_collator = MBMaskCollator(
        input_size=MASK['crop_size'], patch_size=MASK['patch_size'],
        pred_mask_scale=MASK['pred_mask_scale'], enc_mask_scale=MASK['enc_mask_scale'],
        aspect_ratio=MASK['aspect_ratio'], nenc=MASK['num_enc_masks'],
        npred=MASK['num_pred_masks'], allow_overlap=MASK['allow_overlap'],
        min_keep=MASK['min_keep'])
    transform = make_transforms(
        crop_size=MASK['crop_size'], crop_scale=(0.3, 1.0), gaussian_blur=False,
        horizontal_flip=False, color_distortion=False, color_jitter=0.0)

    _, loader, _ = make_agrifield40k(
        transform=transform, batch_size=args.batch_size, collator=mask_collator,
        pin_mem=False, num_workers=args.num_workers, world_size=1, rank=0,
        manifest_path=args.manifest, root_path=args.root_path, split='val',
        drop_last=True, seed=args.seed)

    encoder, predictor = build_models(device, use_sdpa=args.use_sdpa)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    enc_state = {k.replace('module.', '', 1) if k.startswith('module.') else k:
                 v for k, v in checkpoint['encoder'].items()}
    pred_state = {k.replace('module.', '', 1) if k.startswith('module.') else k:
                  v for k, v in checkpoint['predictor'].items()}
    enc_msg = encoder.load_state_dict(enc_state, strict=True)
    pred_msg = predictor.load_state_dict(pred_state, strict=True)
    tgt_state = {k.replace('module.', '', 1) if k.startswith('module.') else k:
                 v for k, v in checkpoint['target_encoder'].items()}
    target_encoder, _ = build_models(device, use_sdpa=args.use_sdpa)
    tgt_msg = target_encoder.load_state_dict(tgt_state, strict=True)

    for module in (encoder, predictor, target_encoder):
        module.eval().to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    losses = []
    with torch.no_grad():
        for itr, (udata, masks_enc, masks_pred) in enumerate(loader):
            if itr >= args.num_batches:
                break
            imgs = udata[0].to(device)
            m_enc = [m.to(device) for m in masks_enc]
            m_pred = [m.to(device) for m in masks_pred]
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == 'cuda'):
                h = target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))
                B = len(h)
                h = apply_masks(h, m_pred)
                h = repeat_interleave_batch(h, B, repeat=len(m_enc))
                z = encoder(imgs, m_enc)
                z = predictor(z, m_enc, m_pred)
                loss = F.smooth_l1_loss(z, h)
            losses.append(float(loss.item()))

    result = {
        'checkpoint': args.checkpoint,
        'manifest': args.manifest,
        'num_batches': len(losses),
        'batch_size': args.batch_size,
        'seed': args.seed,
        'val_loss_mean': float(np.mean(losses)) if losses else None,
        'val_loss_std': float(np.std(losses)) if losses else None,
        'val_loss_first': losses[0] if losses else None,
        'val_loss_last': losses[-1] if losses else None,
        'encoder_load': str(enc_msg),
        'predictor_load': str(pred_msg),
        'target_load': str(tgt_msg),
        'warning': ('Diagnostic only. The val manifest has known cross-split '
                    'provenance leakage; this is NOT a generalisation metric.'),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n',
                                  encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())