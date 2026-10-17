# I-JEPA Real-Image Smoke Test

## Configuration

- Upstream I-JEPA checkout: `52c1ae95d05f743e000e8f10a1f3a79b10cff048`.
- Model: random-initialized `vit_base` (ViT-B/16); no MAE or other pretrained weights.
- Input and patches: 224 x 224, patch size 16 (196 image patches).
- Objective and masks: upstream Smooth L1 JEPA objective and upstream random multi-block mask collator, one encoder mask and four predictor masks.
- Data: 40 real images from the confirmed ACRECropWeed portion of `pretrain_manifest.csv`; no labels are read.
- Batch size 4, two workers, BF16 autocast, physical GPU 1.

## Results

| Run | Epoch | Updates this run | Cumulative updates | Loss first -> last | Peak allocated GPU memory | Mean step time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial training | 1 | 10 | 10 | 0.47786 -> 0.27236 | 2,877 MiB | 152.1 ms |
| Checkpoint continuation | 2 | 10 | 20 | 0.26683 -> 0.25830 | 2,876 MiB | 160.2 ms |

Both runs reported finite loss, verified that encoder and predictor parameters changed, and checked the target encoder against the configured EMA equation. The second run loaded encoder, predictor, target encoder, and optimizer from the epoch 1 checkpoint. It also checked the manifest SHA-256 before continuing. The final checkpoint records epoch 2 and 20 total optimizer updates.

The saved GradScaler state was `scale=65536`, `_growth_tracker=10` at epoch 1 and `_growth_tracker=20` after resume (growth interval 2,000). No overflow skip was recorded in either 10-step segment, corroborating that all 20 counted updates were applied.

Separately, the synthetic engineering check completed 10 updates on random tensors, saved and reloaded a checkpoint, and verified finite loss plus encoder/target changes. It used 2,919 MiB peak memory and averaged 105.1 ms per step. This synthetic result is only a code-path smoke test; it is not agricultural data or a model-performance result. Its summary is tracked at `reports/synthetic_smoke.json`; the generated binary checkpoint remains under the ignored `runs/` directory.

This was a clean epoch-boundary checkpoint/resume test, not a forced process kill in the middle of an optimizer update. It validates the saved-state continuation path without claiming bitwise-identical data order across a mid-epoch crash.

Artifacts under the ignored local `runs/` directory:

- `runs/original_smoke_final/jepa_r0.csv`
- `runs/original_smoke_final/training_verification.yaml`
- `runs/original_smoke_final/jepa-latest.pth.tar` (epoch 1)
- `runs/original_smoke_final_resumed/jepa_r0.csv`
- `runs/original_smoke_final_resumed/training_verification.yaml`
- `runs/original_smoke_final_resumed/jepa-latest.pth.tar` (epoch 2)

The checkpoint files are intentionally not committed to Git. They are generated binary run outputs; code, configs, tests, manifests, audit records and this report are tracked separately.

## Loader and tests

`python -m pytest -q` passed all 5 dataset tests. The real-image loader validation read 8 images with two workers, returned deterministic batches across repeated seeded reads, and produced a finite `(4, 3, 224, 224)` batch. The original mask collator returned one encoder mask of shape `(4, 67)` and four predictor masks of shape `(4, 36)`; all indices were unique per sample and in the 0..195 range.

The detailed loader output is in `reports/loader_validation.json`. The training CSVs and verification YAMLs retain per-step losses, masks, timing, optimizer counts, EMA result, manifest hash, and peak memory.
