# ViT-H/14 Single-GPU Benchmark — RTX A6000 48 GB (Stage 2, Task E)

Machine-readable data: [`vith_benchmark.csv`](vith_benchmark.csv), [`vith_benchmark.json`](vith_benchmark.json).
Tool: [`tools/bench_vith.py`](../../tools/bench_vith.py). SDPA equivalence: [`reports/sdpa_equivalence.json`](sdpa_equivalence.json).

## Setup

| Item | Value |
| --- | --- |
| GPU (physical) | RTX A6000, **physical index 1** (the idle card; index 0 drives the desktop) |
| `CUDA_VISIBLE_DEVICES` | `1` → program sees `cuda:0` |
| VRAM | 48 GB total, ~1.3 GB free at start |
| PyTorch / CUDA | 2.9.1+cu128 / 12.8 build |
| Backbone | ViT-H/14, 224×224, 256 patches |
| Precision | BF16 autocast + GradScaler |
| Attention | PyTorch SDPA (`use_sdpa=True`) |
| Activation checkpointing | off (not needed — see below) |
| Micro-batch / Grad-accum | swept 1→32 / 8 |
| Steps measured per config | 5 warm-up + 10 timed optimizer windows |
| Timing | CUDA events + `torch.cuda.synchronize()` around each window |

All three models (encoder, target encoder, predictor) are fully materialized on the
GPU; the target encoder is a DDP-wrapped frozen copy, so its 630 M params count
against VRAM even though it does not train. This mirrors the real
[`src/train.py`](../../src/train.py) pipeline exactly (DDP-wrapped, single rank).

## Results

| Micro-batch | Eff. batch | Opt. step (ms) | Micro-step (ms) | Images/sec | Peak alloc (MiB) | Peak reserved (MiB) | Loss finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 8 | 1175.5 | 135.5 | 6.81 | 20,077 | 21,198 | yes |
| 2 | 16 | 1170.9 | 135.0 | 13.66 | 20,085 | 21,522 | yes |
| 4 | 32 | 1200.8 | 138.7 | 26.65 | 20,077 | 21,708 | yes |
| 8 | 64 | 1330.4 | 155.2 | 48.11 | 20,506 | 22,736 | yes |
| 16 | 128 | 2014.8 | 240.7 | 63.53 | 22,118 | 24,296 | yes |
| 32 | 256 | 3394.8 | 413.2 | 75.41 | 25,376 | 26,706 | yes |

**No OOM at any tested size.** Peak VRAM never exceeds ~25.4 GB of the 48 GB
available — roughly half. Throughput scales with batch size up to batch 32
(75.4 img/s ≈ 11× the batch-1 rate), confirming the model fits comfortably and
there is headroom to go higher if desired.

Mask geometry was held **fixed** across every batch size (one sampled context
block of 81 patches + four target blocks of 42 patches), so the numbers compare
cleanly — re-sampling masks per call would otherwise confound step time, because
attention cost depends on the number of patches.

## SDPA and activation checkpointing

- **SDPA equivalence** (`reports/sdpa_equivalence.json`): output max-abs diff
  `0.0`, gradient max-abs diff `2.2e-11` versus the reference softmax path.
  Enabling SDPA changes no training mathematics; it is a strict win on memory
  (peak alloc 3058 → 2995 MiB on a single block at batch 16) and is effectively
  free on speed (1.008×). The training configs use `use_sdpa=True`.
- **Activation checkpointing**: not required. The full un-checkpointed ViT-H
  with all three models in VRAM peaks at ~25 GB, well under the 48 GB ceiling,
  so enabling it would only cost recompute time for no VRAM benefit at these
  batch sizes. It remains available behind `meta.activation_checkpointing` if a
  larger batch or resolution is attempted later.

Mixed precision (BF16) is used throughout; the GradScaler path (`scaler.scale`
on backward, unscale at the optimizer step) is active. No skipped optimizer steps
were observed for any finite-loss window.

## Recommended single-GPU configuration

Given the headroom, the binding constraint is **wall-clock and corpus size**, not
VRAM. Recommended starting point:

| Setting | Value | Reason |
| --- | --- | --- |
| Micro-batch | **16** | 63.5 img/s at 22 GB; comfortable margin, good GPU saturation |
| Grad-accum | 8 | effective batch 128, matches the absorbed update scale |
| Precision | BF16 autocast | verified finite, matches upstream |
| SDPA | on | numerically identical, lower memory |
| Activation checkpointing | off | unnecessary at this size |
| Reserved VRAM headroom | ~26 GB free | stays far from the OOM instability zone |

If the run is I/O bound on the dataloader rather than compute bound, raise
micro-batch to 32 (75 img/s) — VRAM still leaves ~22 GB free.

## Estimated Stage 3 compute cost — measured vs. assumed

The Stage 1 ViT-B run cost **152 ms/step**. **That number must not be extrapolated
to ViT-H.** ViT-H has ~1.28 B trainable parameters (encoder + predictor) versus
ViT-B's ~86 M, and three full models live on the GPU instead of one. The measured
ViT-H optimizer-step time at the recommended batch-16 setting is **~2.0 s/step**.

Illustrative Stage 3 budget (provisional; the protocol fixes the exact number):
if Stage 3 runs, say, **30,000 optimizer steps** at batch-16 (~2.0 s/step):

```
30,000 steps × ~2.0 s/step ≈ 60,000 s ≈ 16.7 wall-clock hours on one A6000
```

That is a single-GPU, single-day run, not a multi-day campaign. It is an
**estimate from a 10-window measurement**, not a validated full run — the actual
cost depends on the final effective batch and step count chosen in the Stage 3
protocol. Stage 2 deliberately did **not** execute any long pretraining, so this
is a planning figure only.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u tools/bench_vith.py \
    --batch-sizes 1 2 4 8 16 32 --grad-accum 8 --warmup 5 --measure 10 \
    --csv reports/vith_benchmark.csv --json reports/vith_benchmark.json

CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u -m tools.verify_sdpa \
    --out reports/sdpa_equivalence.json
```

Both run on physical GPU 1 only and never touch GPU 0 (the desktop card).
