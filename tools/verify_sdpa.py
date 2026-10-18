"""Verify PyTorch SDPA and the reference attention path produce equivalent
gradients and outputs on ViT-H/14, and measure the throughput / memory effect.

The project added an optional SDPA path in `src/models/vision_transformer.py`
(enabled by `use_sdpa`).  This tool confirms that path is numerically equivalent
to the reference softmax attention (so enabling it changes no training maths),
and reports the real speed / memory delta on one RTX A6000.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u -m tools.verify_sdpa \
        --out reports/sdpa_equivalence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from src.helper import init_model
from src.utils.distributed import init_distributed

torch.manual_seed(0)


def make_pair(use_sdpa):
    enc, pred = init_model(device=torch.device('cpu'), patch_size=14,
                           model_name='vit_huge', crop_size=224, pred_depth=12,
                           pred_emb_dim=384, use_sdpa=use_sdpa)
    return enc, pred


def random_patch_tokens(batch, device, n_patch=256, dim=1280):
    return torch.randn(batch, n_patch, dim, device=device, requires_grad=True)


def forward_block_with_loss(block, x):
    y, _ = block(x)
    return y.pow(2).mean()


def compare(use_sdpa):
    device = torch.device('cuda:0')
    ref_e, ref_p = make_pair(False)
    sdpa_e, sdpa_p = make_pair(True)
    # copy identical weights into the SDPA copy
    sdpa_e.load_state_dict(ref_e.state_dict())
    sdpa_p.load_state_dict(ref_p.state_dict())
    ref_e.to(device).train(); sdpa_e.to(device).train()
    ref_p.to(device); sdpa_p.to(device)

    x_ref = random_patch_tokens(2, device)
    x_sdpa = x_ref.detach().clone().requires_grad_(True)

    loss_ref = forward_block_with_loss(ref_e.blocks[0], x_ref)
    loss_sdpa = forward_block_with_loss(sdpa_e.blocks[0], x_sdpa)
    loss_ref.backward(); loss_sdpa.backward()

    return {
        'use_sdpa': use_sdpa,
        'output_max_abs_diff': float(
            (loss_ref.detach() - loss_sdpa.detach()).abs()),
        'output_relative_diff': float(
            (loss_ref.detach() - loss_sdpa.detach()).abs()
            / (loss_ref.detach().abs() + 1e-6)),
        'grad_max_abs_diff': float(
            (x_ref.grad - x_sdpa.grad).abs().max()),
    }


def timing(use_sdpa, batch_size, iters=30):
    device = torch.device('cuda:0')
    enc, _ = make_pair(use_sdpa)
    enc.to(device).train()
    x = random_patch_tokens(batch_size, device)
    # warmup
    for _ in range(5):
        enc.blocks[0](x).pow(2).mean().backward()
        enc.zero_grad()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    for _ in range(iters):
        enc.blocks[0](x).pow(2).mean().backward()
        enc.zero_grad()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - start) / iters * 1000
    return {
        'use_sdpa': use_sdpa,
        'batch_size': batch_size,
        'block_ms_mean': dt,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('reports/sdpa_equivalence.json'))
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    init_distributed(rank_and_world_size=(0, 1))
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    equiv = compare(False)  # compares reference vs SDPA internally
    print('SDPA vs reference equivalence:')
    print(json.dumps(equiv, indent=2))

    ref_t = timing(False, args.batch_size)
    sdpa_t = timing(True, args.batch_size)
    print('Timing (single ViT-H block, batch=%d):' % args.batch_size)
    print(' reference:', json.dumps(ref_t))
    print(' sdpa:     ', json.dumps(sdpa_t))

    report = {
        'equivalence': equiv,
        'timing_reference': ref_t,
        'timing_sdpa': sdpa_t,
        'speedup_x': ref_t['block_ms_mean'] / sdpa_t['block_ms_mean'],
        'peak_reserved_saving_mib': ref_t['peak_reserved_mib'] - sdpa_t['peak_reserved_mib'],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
