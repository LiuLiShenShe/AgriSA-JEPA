# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.vision_transformer as vit
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(
    device,
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
):
    checkpoint = torch.load(
        r_path, map_location=torch.device('cpu'), weights_only=False)
    epoch = checkpoint['epoch']

    msg = encoder.load_state_dict(checkpoint['encoder'])
    logger.info(f'loaded encoder from epoch {epoch}: {msg}')
    msg = predictor.load_state_dict(checkpoint['predictor'])
    logger.info(f'loaded predictor from epoch {epoch}: {msg}')
    if target_encoder is not None:
        msg = target_encoder.load_state_dict(checkpoint['target_encoder'])
        logger.info(f'loaded target encoder from epoch {epoch}: {msg}')

    opt.load_state_dict(checkpoint['opt'])
    if scaler is not None and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    logger.info(f'loaded optimizer from epoch {epoch}, read-path: {r_path}')

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(
    device,
    patch_size=16,
    model_name='vit_base',
    crop_size=224,
    pred_depth=6,
    pred_emb_dim=384,
    use_sdpa=False,
    activation_checkpointing=False
):
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        use_sdpa=use_sdpa,
        activation_checkpointing=activation_checkpointing)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
        use_sdpa=use_sdpa,
        activation_checkpointing=activation_checkpointing)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
    total_steps=None,
    warmup_steps_override=None
):
    """Build optimizer and schedules.

    ``warmup_steps_override`` makes the warmup length an absolute optimizer-
    update count.  It exists for arm comparisons whose epochs contain different
    numbers of updates (e.g. Stage 6 natural vs capped sampling): deriving
    warmup from ``iterations_per_epoch`` would silently give the arms different
    LR trajectories.  When it is ``None`` the historical
    ``int(warmup * iterations_per_epoch)`` rule is used unchanged.
    """
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    # -- `total_steps`, when given, is an EXACT optimizer-update budget and wins
    #    over the epoch-derived horizon. It is used so that LR/WD cosine decay
    #    ends precisely at the configured update count rather than at whatever
    #    num_epochs * iterations_per_epoch happens to produce.
    horizon = int(total_steps) if total_steps is not None \
        else int(ipe_scale * num_epochs * iterations_per_epoch)
    if horizon < 1:
        raise ValueError(f'Scheduler horizon must be >= 1, got {horizon}')
    warmup_steps = (int(warmup_steps_override) if warmup_steps_override is not None
                    else int(warmup*iterations_per_epoch))
    if warmup_steps < 0 or warmup_steps >= horizon:
        raise ValueError(
            f'warmup_steps must be in [0, horizon), got {warmup_steps} '
            f'with horizon {horizon}')
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=horizon)
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=horizon)
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
