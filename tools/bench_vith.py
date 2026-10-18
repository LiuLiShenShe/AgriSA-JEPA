"""Micro-benchmark official I-JEPA ViT-H/14 training on ONE GPU.

Measures, for each requested micro-batch size:
  - forward + backward success or the OOM point,
  - peak allocated / reserved CUDA memory,
  - CUDA-event-timed micro-step and optimizer-step latency (with proper sync),
  - images/sec,
  - loss finiteness,
  - whether a checkpoint round-trip works.

Run from the repo root, e.g.::

    CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u tools/bench_vith.py \
        --batch-sizes 1 2 4 --grad-accum 8 --warmup 5 --measure 10 \
        --csv reports/vith_benchmark.csv --json reports/vith_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

# -- make the repository root importable however this script is launched
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.helper import init_model, init_opt
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.pretrained import init_from_pretrained
from src.utils.distributed import init_distributed
from src.utils.tensors import repeat_interleave_batch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('bench_vith')

CSV_FIELDS = [
    'config_id', 'model', 'patch_size', 'resolution', 'micro_batch_size',
    'grad_accum', 'effective_batch_size', 'precision', 'sdpa', 'activation_checkpointing',
    'context_patches', 'target_patches',
    'device_name', 'visible_device_index', 'encoder_params', 'target_encoder_params',
    'predictor_params', 'params_total',
    'peak_allocated_mib', 'peak_reserved_mib', 'oom_stage', 'oom',
    'step_time_ms_mean', 'step_time_ms_median', 'step_time_ms_min', 'step_time_ms_max',
    'micro_step_ms', 'images_per_sec', 'optimizer_step_ms',
    'loss_first', 'loss_last', 'loss_finite', 'checkpoint_saved', 'notes',
]


def build(model_name, patch_size, crop_size, pred_depth, pred_emb_dim, use_sdpa,
          activation_checkpointing, device, pretrained, pretrained_sha256):
    import copy as _copy
    from src.pretrained import load_pretrained_payload, strip_module_prefix

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        use_sdpa=use_sdpa,
        activation_checkpointing=activation_checkpointing,
    )
    # A fresh target encoder: never shares storage with the context encoder.
    target_encoder = _copy.deepcopy(encoder)

    report = {}
    if pretrained is not None:
        if pretrained_sha256:
            from src.pretrained import sha256_file
            if sha256_file(pretrained) != pretrained_sha256:
                raise ValueError('Pretrained checkpoint SHA-256 mismatch')
        payload, _ = load_pretrained_payload(pretrained)
        target_encoder.load_state_dict(
            strip_module_prefix(payload['target_encoder']), strict=True)
        del payload
        # Encoder and predictor keep the strict, verified Operation A path.
        report = init_from_pretrained(
            pretrained, encoder=encoder, predictor=predictor,
            target_encoder=None, device=device)
        target_encoder.to(device)

    # DDP exactly like src/train.py, so the benchmark reflects the real pipeline
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    return encoder, predictor, target_encoder, report


def benchmark(args):
    # -- the real pipeline wraps the model in DDP, so mirror it exactly
    #    (single rank, single process, one GPU).
    world_size, rank = init_distributed(rank_and_world_size=(0, 1))
    if world_size != 1:
        raise RuntimeError(f'benchmark expects a single process, got world_size={world_size}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        visible_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(visible_index)
        total_mib = torch.cuda.get_device_properties(visible_index).total_memory / 1024 ** 2
        world_visible = torch.cuda.device_count()
    else:
        visible_index, device_name, total_mib, world_visible = None, 'cpu', None, 0
    logger.info('Benchmark device: %s (visible index %s, %s GPUs visible, %.0f MiB)',
                device_name, visible_index, world_visible, total_mib or 0)

    collator = MaskCollator(
        input_size=args.crop_size, patch_size=args.patch_size,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10)

    encoder, predictor, target_encoder, pretrained_report = build(
        args.model_name, args.patch_size, args.crop_size, args.pred_depth,
        args.pred_emb_dim, args.sdpa, args.activation_checkpointing, device,
        args.pretrained, args.pretrained_sha256)

    param_counts = {
        'encoder': sum(p.numel() for p in encoder.parameters()),
        'target_encoder': sum(p.numel() for p in target_encoder.parameters()),
        'predictor': sum(p.numel() for p in predictor.parameters()),
    }
    param_counts['total'] = sum(param_counts.values())
    logger.info('Parameters: %s', param_counts)

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder, predictor=predictor, wd=0.04, final_wd=0.04,
        start_lr=args.lr, ref_lr=args.lr, final_lr=args.lr,
        iterations_per_epoch=1, warmup=0, num_epochs=1,
        ipe_scale=1.0, use_bfloat16=True)
    momentum_scheduler = iter([args.ema] * 10 ** 9)

    # -- Fixed masks: sample ONE set for the largest batch and slice rows for
    #    smaller batches.  Mask size drives attention cost, so re-sampling per
    #    call would confound the batch-size comparison.
    torch.manual_seed(20261018)
    results = []
    max_batch = max(args.batch_sizes)
    _, _me, _mp = collator(
        [(torch.zeros(3, args.crop_size, args.crop_size), str(i))
         for i in range(max_batch)])
    master_masks = (
        [m.to(device) for m in _me],
        [m.to(device) for m in _mp])

    def get_masks(batch_size):
        return (
            [m[:batch_size] for m in master_masks[0]],
            [m[:batch_size] for m in master_masks[1]])

    def one_micro_step(images, masks_enc, masks_pred):
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        def closure():
            def forward_target():
                with torch.no_grad():
                    h = target_encoder(images)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred)
                    return repeat_interleave_batch(h, len(h), repeat=len(masks_enc))

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                h = forward_target()
                z = encoder(images, masks_enc)
                z = predictor(z, masks_enc, masks_pred)
                loss = F.smooth_l1_loss(z, h)
            # mirrors src/train.py: BF16 uses the upstream GradScaler path
            scaler.scale(loss / args.grad_accum).backward()
            return loss.detach().item()

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        loss = closure()
        end.record()
        torch.cuda.synchronize()
        return loss, start.elapsed_time(end)

    for batch_size in args.batch_sizes:
        row = {
            'config_id': f'vith14-b{batch_size}-ga{args.grad_accum}-'
                         f'{"sdpa" if args.sdpa else "manual"}-'
                         f'{"ac" if args.activation_checkpointing else "noac"}',
            'model': f'{args.model_name}/{args.patch_size}',
            'patch_size': args.patch_size,
            'resolution': f'{args.crop_size}x{args.crop_size}',
            'micro_batch_size': batch_size,
            'grad_accum': args.grad_accum,
            'effective_batch_size': batch_size * args.grad_accum,
            'precision': 'bf16-autocast',
            'sdpa': args.sdpa,
            'activation_checkpointing': args.activation_checkpointing,
            'device_name': device_name,
            'visible_device_index': visible_index,
            'encoder_params': param_counts['encoder'],
            'target_encoder_params': param_counts['target_encoder'],
            'predictor_params': param_counts['predictor'],
            'params_total': param_counts['total'],
            'oom': False, 'oom_stage': '',
            'notes': '',
        }
        images = torch.randn(batch_size, 3, args.crop_size, args.crop_size, device=device)
        masks_enc, masks_pred = get_masks(batch_size)
        # -- record the mask geometry so numbers are reproducible
        row['context_patches'] = int(masks_enc[0].shape[1])
        row['target_patches'] = int(masks_pred[0].shape[1])

        optimizer.zero_grad(set_to_none=True)
        oom_stage = ''
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            # -- warm-up (discarded); measures the steady-state kernel mix
            for _ in range(args.warmup):
                one_micro_step(images, masks_enc, masks_pred)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # -- one full accumulation window so the peak reflects a real update
            for _ in range(args.grad_accum):
                loss, _ = one_micro_step(images, masks_enc, masks_pred)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            oom_stage = 'update'

            # -- timed passes: measure full optimizer windows, CUDA-synced
            window_ms = []
            micro_ms = []
            for _ in range(args.measure):
                torch.cuda.synchronize()
                wall_start = time.perf_counter()
                for _ in range(args.grad_accum):
                    _, micro = one_micro_step(images, masks_enc, masks_pred)
                    micro_ms.append(micro)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                window_ms.append((time.perf_counter() - wall_start) * 1000.0)
                oom_stage = 'update'

            row['optimizer_step_ms'] = float(sum(window_ms) / len(window_ms))
            row['step_time_ms_median'] = float(sorted(window_ms)[len(window_ms) // 2])
            row['step_time_ms_mean'] = row['optimizer_step_ms']
            row['step_time_ms_min'] = float(min(window_ms))
            row['step_time_ms_max'] = float(max(window_ms))
            row['micro_step_ms'] = float(sum(micro_ms) / len(micro_ms))
            row['images_per_sec'] = 1000.0 * (args.grad_accum * batch_size) / row['optimizer_step_ms']
            row['loss_first'] = loss
            row['loss_last'] = loss
            row['loss_finite'] = bool(math.isfinite(loss))

            if device.type == 'cuda':
                row['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                row['peak_reserved_mib'] = torch.cuda.max_memory_reserved(device) / 1024 ** 2

            # -- checkpoint round trip
            import tempfile, os
            ck = {
                'encoder': encoder.state_dict(), 'predictor': predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(), 'opt': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'epoch': 1,
            }
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'probe.pth.tar')
                torch.save(ck, path)
                back = torch.load(path, map_location='cpu', weights_only=False)
            del back
            row['checkpoint_saved'] = True

        except torch.cuda.OutOfMemoryError as exc:
            row['oom'] = True
            row['oom_stage'] = oom_stage or 'forward'
            row['notes'] = f'CUDA OOM: {str(exc)[:200]}'
            logger.warning('Batch %d OOM at %s: %s', batch_size, row['oom_stage'], str(exc)[:200])
            try:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                row['oom'] = True
                row['oom_stage'] = oom_stage or 'forward'
                row['notes'] = f'CUDA OOM: {str(exc)[:200]}'
                logger.warning('Batch %d OOM at %s', batch_size, row['oom_stage'])
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                row['notes'] = f'{type(exc).__name__}: {str(exc)[:200]}'
                logger.exception('Batch %d failed', batch_size)

        for key, default in (('step_time_ms_mean', None), ('step_time_ms_median', None),
                             ('step_time_ms_min', None), ('step_time_ms_max', None),
                             ('micro_step_ms', None),
                             ('optimizer_step_ms', None), ('images_per_sec', None),
                             ('peak_allocated_mib', None), ('peak_reserved_mib', None),
                             ('loss_first', None), ('loss_last', None), ('loss_finite', None),
                             ('checkpoint_saved', False)):
            row.setdefault(key, default)
        results.append(row)
        logger.info('-> %s : oom=%s peak_alloc=%s MiB step=%s ms img/s=%s',
                    row['config_id'], row['oom'],
                    f"{row['peak_allocated_mib']:.0f}" if row['peak_allocated_mib'] else 'n/a',
                    f"{row['step_time_ms_mean']:.1f}" if row['step_time_ms_mean'] else 'n/a',
                    f"{row['images_per_sec']:.2f}" if row['images_per_sec'] else 'n/a')
        del images, masks_enc, masks_pred
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            if row['oom']:
                break

    return results, param_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=Path,
                        default=Path('/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar'))
    parser.add_argument('--pretrained-sha256',
                        default='0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22')
    parser.add_argument('--model-name', default='vit_huge')
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--pred-depth', type=int, default=12)
    parser.add_argument('--pred-emb-dim', type=int, default=384)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--measure', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--ema', type=float, default=0.996)
    parser.add_argument('--sdpa', action='store_true', default=True)
    parser.add_argument('--no-sdpa', dest='sdpa', action='store_false')
    parser.add_argument('--activation-checkpointing', action='store_true', default=False)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_const',
                        const=None,
                        help='skip official-weight loading (fast plumbing check only)')
    parser.add_argument('--samples', type=int, default=0,
                        help='if >0, cap sampled images (unused; benchmark synthesizes on device)')
    parser.add_argument('--csv', type=Path, default=None)
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args()

    rows, params = benchmark(args)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({'rows': rows, 'param_counts': params}, indent=2))
    print(json.dumps({'param_counts': params, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
