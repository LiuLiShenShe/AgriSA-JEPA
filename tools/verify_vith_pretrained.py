"""Verify official I-JEPA ViT-H/14 weight loading (Stage 2, Operation A).

Checks, in order:

1. Official checkpoint structure (top-level fields, tensor counts, prefixes).
2. Exact per-tensor key/shape compatibility against a freshly built ViT-H/14
   encoder, predictor and target encoder.
3. That the transfer actually changed the randomly initialized weights.
4. That the loaded weights equal the checkpoint tensors exactly.
5. Legal random multi-block mask indices.
6. A forward pass (encoder -> predictor -> smooth-L1 target loss) free of
   NaN/Inf, at several batch sizes.

Run from the repository root, for example::

    ./.venv/bin/python -u -m tools.verify_vith_pretrained \
        --checkpoint /data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar \
        --json reports/vith_checkpoint_loading.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.helper import init_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.models.vision_transformer import model_architecture
from src.pretrained import (
    compare_state_dicts,
    describe_payload,
    init_from_pretrained,
    sha256_file,
    strip_module_prefix,
)
from src.utils.tensors import repeat_interleave_batch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('verify_vith')


def build_model(model_name, patch_size, crop_size, pred_depth, pred_emb_dim,
                use_sdpa=False, activation_checkpointing=False):
    encoder, predictor = init_model(
        device=torch.device('cpu'),
        patch_size=patch_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        use_sdpa=use_sdpa,
        activation_checkpointing=activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)
    return encoder, predictor, target_encoder


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def check_masks(height, width, patch_size, mask_collator_path, batch_size=2):
    """Sample masks through the real collator and validate the indices.

    ``height``/``width`` are GRID sizes (crop_size // patch_size), not pixels.
    """
    collator = MaskCollator(
        input_size=(height * patch_size, width * patch_size),
        patch_size=patch_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
    )
    fake_batch = [(torch.zeros(3, height * patch_size, width * patch_size), f'sample-{i}')
                  for i in range(batch_size)]
    _, masks_enc, masks_pred = collator(fake_batch)
    num_patches = height * width
    report = {'num_patches': num_patches, 'nenc': len(masks_enc), 'npred': len(masks_pred)}
    for label, masks in (('enc', masks_enc), ('pred', masks_pred)):
        report[f'{label}_shapes'] = [tuple(m.shape) for m in masks]
        for mask in masks:
            assert mask.dtype == torch.int64, f'{label} mask dtype {mask.dtype}'
            assert int(mask.min()) >= 0, f'{label} mask has negative index'
            assert int(mask.max()) < num_patches, f'{label} mask index out of range'
            for row in mask:
                assert len(set(row.tolist())) == len(row), f'{label} mask has duplicate index'
        report[f'{label}_min_size'] = int(min(m.shape[1] for m in masks))
        report[f'{label}_max_size'] = int(max(m.shape[1] for m in masks))

    # Context block must not intersect ANY target block (allow_overlap=False).
    # Target blocks may overlap each other, so compare per block, not vs. their union.
    overlapping_blocks = 0
    for image_index in range(batch_size):
        enc = set(masks_enc[0][image_index].tolist())
        for block in masks_pred:
            if enc & set(block[image_index].tolist()):
                overlapping_blocks += 1
    report['enc_pred_overlapping_blocks'] = overlapping_blocks
    report['enc_pred_disjoint'] = overlapping_blocks == 0
    return report


def forward_probe(encoder, predictor, target_encoder, cfg, batch_size, crop_size,
                  patch_size, device, use_bfloat16, masks=None):
    """One JEPA forward pass: returns loss and finiteness diagnostics."""
    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    images = torch.rand(batch_size, 3, crop_size, crop_size, device=device)
    if masks is None:
        collator = MaskCollator(
            input_size=crop_size, patch_size=patch_size,
            enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
            aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10)
        _, masks_enc, masks_pred = collator([(torch.zeros(3, crop_size, crop_size), str(i))
                                             for i in range(batch_size)])
    else:
        masks_enc, masks_pred = masks
    masks_enc = [m.to(device) for m in masks_enc]
    masks_pred = [m.to(device) for m in masks_pred]

    with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
        h = target_encoder(images)
        h = F.layer_norm(h, (h.size(-1),))
        B = len(h)
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        z = encoder(images, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        loss = F.smooth_l1_loss(z, h)
    return {
        'batch_size': batch_size,
        'z_shape': tuple(z.shape),
        'h_shape': tuple(h.shape),
        'loss': float(loss),
        'z_finite': bool(torch.isfinite(z).all()),
        'h_finite': bool(torch.isfinite(h).all()),
        'loss_finite': bool(torch.isfinite(loss)),
        'target_feature_std': float(h.float().std()),
        'context_norm': float(z.float().norm(dim=-1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar'))
    parser.add_argument('--model-name', default='vit_huge')
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--pred-depth', type=int, default=12)
    parser.add_argument('--pred-emb-dim', type=int, default=384)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--json', type=Path, default=None)
    parser.add_argument('--skip-sha256', action='store_true',
                        help='skip hashing the 10 GB checkpoint (use a cached digest)')
    parser.add_argument('--sha256', default=None, help='precomputed checkpoint digest')
    args = parser.parse_args()

    device = torch.device('cpu')
    report = {'checkpoint': str(args.checkpoint), 'architecture': model_architecture(
        args.model_name, args.patch_size, args.crop_size, args.pred_depth, args.pred_emb_dim)}

    if args.sha256:
        report['checkpoint_sha256'] = args.sha256
    elif args.skip_sha256:
        report['checkpoint_sha256'] = None
    else:
        logger.info('Hashing checkpoint (this is a 10 GB file)...')
        report['checkpoint_sha256'] = sha256_file(args.checkpoint)
    report['checkpoint_bytes'] = args.checkpoint.stat().st_size

    # -- 1. structure -------------------------------------------------------
    from src.pretrained import load_pretrained_payload
    payload, load_mode = load_pretrained_payload(args.checkpoint)
    report['load_mode'] = load_mode
    report['payload'] = describe_payload(payload)

    # -- 2. build + compare + load -----------------------------------------
    encoder, predictor, target_encoder = build_model(
        args.model_name, args.patch_size, args.crop_size, args.pred_depth, args.pred_emb_dim)

    diffs = {}
    for label, module in (('encoder', encoder), ('predictor', predictor),
                          ('target_encoder', target_encoder)):
        diffs[label] = compare_state_dicts(
            module.state_dict(), strip_module_prefix(payload[label]))
    report['state_dict_diff'] = {
        label: {
            'is_exact': d['is_exact'],
            'matched': d['matched'],
            'model_tensors': d['model_tensors'],
            'checkpoint_tensors': d['checkpoint_tensors'],
            'missing': d['missing'],
            'unexpected': d['unexpected'],
            'shape_mismatch': d['shape_mismatch'],
        } for label, d in diffs.items()
    }
    report['all_exact'] = all(d['is_exact'] for d in diffs.values())

    # -- snapshot random init to prove the load changes weights -------------
    random_before = {
        'encoder.pos_embed': encoder.pos_embed.detach().clone(),
        'encoder.blocks.0.attn.qkv.weight': encoder.blocks[0].attn.qkv.weight.detach().clone(),
        'predictor.mask_token': predictor.mask_token.detach().clone(),
    }

    # encoder/target_encoder agreement inside the official checkpoint itself
    off_enc = strip_module_prefix(payload['encoder'])
    off_tgt = strip_module_prefix(payload['target_encoder'])
    max_enc_tgt = max(float((off_enc[k].float() - off_tgt[k].float()).abs().max())
                      for k in off_enc if off_enc[k].is_floating_point())
    report['official_encoder_vs_target_max_abs_diff'] = max_enc_tgt
    logger.info('Official encoder vs target_encoder max |diff| = %.3e', max_enc_tgt)

    load_report = init_from_pretrained(
        args.checkpoint, encoder=encoder, predictor=predictor, target_encoder=target_encoder,
        device=device)
    report['operation_a'] = {
        'load_mode': load_report['load_mode'],
        'encoder_keys_loaded': load_report['encoder_keys_loaded'],
        'predictor_keys_loaded': load_report['predictor_keys_loaded'],
        'target_encoder_keys_loaded': load_report['target_encoder_keys_loaded'],
        'predictor_inherited': load_report['predictor_inherited'],
        'source_epoch': load_report['epoch_in_source_checkpoint'],
        'sha256': load_report['sha256'],
    }

    # -- 3. the load must have changed the random weights -------------------
    #    pos_embed is a fixed 2D-sincos buffer (requires_grad=False) regenerated
    #    deterministically, so it is expected to be UNCHANGED. The learnable
    #    tensors below are the meaningful check.
    changes = {}
    after_enc_qkv = encoder.blocks[0].attn.qkv.weight.detach()
    after_pred_mask = predictor.mask_token.detach()
    changes['encoder.pos_embed_changed'] = not torch.equal(
        random_before['encoder.pos_embed'], encoder.pos_embed.detach())
    changes['encoder.pos_embed_note'] = (
        'fixed 2D-sincos buffer; unchanged by design, not a learnable weight')
    changes['encoder.qkv_changed'] = not torch.equal(
        random_before['encoder.blocks.0.attn.qkv.weight'], after_enc_qkv)
    changes['predictor.mask_token_changed'] = not torch.equal(
        random_before['predictor.mask_token'], after_pred_mask)
    report['weights_changed_by_init'] = changes

    # -- 4. loaded weights equal the checkpoint tensors exactly -------------
    exact = {}
    for key in ('encoder', 'predictor', 'target_encoder'):
        module = {'encoder': encoder, 'predictor': predictor, 'target_encoder': target_encoder}[key]
        ck_state = strip_module_prefix(payload[key])
        model_state = module.state_dict()
        worst = 0.0
        for name, tensor in ck_state.items():
            if tensor.is_floating_point():
                worst = max(worst, float((model_state[name].float() - tensor.float()).abs().max()))
        exact[key] = worst
    report['loaded_vs_checkpoint_max_abs_diff'] = exact

    # -- 5. masks -----------------------------------------------------------
    grid = args.crop_size // args.patch_size
    report['masks'] = check_masks(
        grid, grid, args.patch_size, None, batch_size=2)
    # rebuild masks on the fly for the forward probe
    collator = MaskCollator(
        input_size=args.crop_size, patch_size=args.patch_size,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10)
    _, masks_enc, masks_pred = collator(
        [(torch.zeros(3, args.crop_size, args.crop_size), str(i)) for i in range(2)])

    # -- 6. forward passes --------------------------------------------------
    fwd = {}
    for bs in args.batch_sizes:
        width = 1 if bs == 1 else min(bs, len(masks_enc[0]))
        enc_m = [m[:width] for m in masks_enc]
        pred_m = [m[:width] for m in masks_pred]
        fwd[f'batch_{bs}'] = forward_probe(
            encoder, predictor, target_encoder, report['architecture'], bs,
            args.crop_size, args.patch_size, device, use_bfloat16=True,
            masks=(enc_m, pred_m))
    report['forward'] = fwd

    # free the 10 GB payload before finishing
    del payload
    del off_enc, off_tgt

    report['param_counts'] = {
        'encoder': parameter_count(encoder),
        'predictor': parameter_count(predictor),
        'target_encoder': parameter_count(target_encoder),
    }

    print(json.dumps(report, indent=2, default=str))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
        logger.info('Wrote %s', args.json)


if __name__ == '__main__':
    main()
