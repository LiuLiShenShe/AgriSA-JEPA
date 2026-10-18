# ViT-H/14 Official Checkpoint Loading (Stage 2, Operation A)

Machine-readable evidence: [`vith_checkpoint_loading.json`](vith_checkpoint_loading.json).

## Source and integrity

| Field | Value |
| --- | --- |
| URL | `https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar` |
| Path | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` (outside Git) |
| Bytes | 10,358,004,345 |
| SHA-256 | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` |
| Source epoch | 299 |
| Source loss / batch / world_size / lr | 0.16633 / 32 / 64 / 1e-3 |

The checkpoint already existed locally at the pinned path, so no download was
performed in this stage. Its digest is recorded above and pinned in the configs
as `meta.pretrained_sha256`, so any future replacement is detected.

The file is deserialized with `torch.load(..., weights_only=True)`, so no
arbitrary Python objects are unpickled. This succeeded on the official file, and
the loader reports the mode it used rather than silently downgrading.

## Checkpoint structure (actual, not assumed)

Top-level keys:

```
encoder, predictor, target_encoder, opt, scaler, epoch, loss, batch_size, world_size, lr
```

| Field | Tensors | `module.` prefix |
| --- | ---: | --- |
| `encoder` | 389 | yes |
| `predictor` | 152 | yes |
| `target_encoder` | 389 | yes |
| `opt` | optimizer state dict | — |
| `scaler` | GradScaler state (5 keys) | — |

All three weight fields carry the DDP `module.` prefix from the original 64-GPU
run. `src/pretrained.strip_module_prefix` removes it before comparison.

## Operation A — Pretrained Initialization

`src/pretrained.init_from_pretrained` transfers **only** the model weights.
Optimizer, LR/weight-decay schedulers, GradScaler and epoch counters are
deliberately discarded so a new agricultural experiment starts clean.

Exact key/shape audit (prefix-normalized, `strict=True` after explicit diff):

| Module | Model tensors | Checkpoint tensors | Missing | Unexpected | Shape mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| encoder | 389 | 389 | 0 | 0 | 0 |
| predictor | 152 | 152 | 0 | 0 | 0 |
| target_encoder | 389 | 389 | 0 | 0 | 0 |

**Predictor inheritance:** complete. All 152 predictor tensors were inherited
(`predictor_inherited: true`). Nothing is randomly initialized and then
presented as an official-weight run.

Verified after loading:

| Check | Result |
| --- | --- |
| max abs diff, loaded encoder vs checkpoint | `0.0` |
| max abs diff, loaded predictor vs checkpoint | `0.0` |
| max abs diff, loaded target encoder vs checkpoint | `0.0` |
| random `blocks.0.attn.qkv.weight` changed by init | yes |
| random `predictor.mask_token` changed by init | yes |

A fresh encoder's `pos_embed` is unchanged by the load **by design** — it is a
fixed 2D-sincos buffer (`requires_grad=False`) regenerated deterministically, not
a learnable weight. The learnable tensors above are the meaningful check.

The official checkpoint's own `encoder` and `target_encoder` differ by
`max|Δ| = 9.63e-03`, which is the expected residue of EMA tracking over 300
epochs. Both are loaded independently, so the agricultural target encoder starts
from the official EMA state, not from a copy of the context encoder.

## Architecture confirmed against the official config

| Setting | Value |
| --- | --- |
| Backbone | `vit_huge`, embed 1280, depth 32, heads 16, MLP ratio 4 |
| Patch | 14×14 → 224/14 = 16 → **256 patches** |
| Predictor | depth 12, embed 384 |
| Encoder params | 630,762,240 |
| Predictor params | 22,377,728 |
| Target encoder params | 630,762,240 |

## Masks

Official random multi-block collator (`allow_overlap=False`, 1 context block at
scale 0.85–1.0, 4 target blocks at scale 0.15–0.2, AR 0.75–1.5, `min_keep=10`):

- All indices are `int64`, within `[0, 256)`, and unique within each block.
- Context and target blocks are disjoint per block over the sampled seeds
  (`enc_pred_overlapping_blocks = 0`). Target blocks may legitimately overlap
  each other, so disjointness is checked per block rather than against their union.

## Forward pass

| Batch | `z` shape | `h` shape | loss | finite | target-feature std |
| --- | --- | --- | ---: | --- | ---: |
| 1 | (4, 42, 1280) | (4, 42, 1280) | 0.1503 | yes | 1.0000 |
| 2 | (8, 42, 1280) | (8, 42, 1280) | 0.1529 | yes | 1.0000 |

The predictor output and the target features have identical shapes (4 prediction
blocks concatenated along the batch axis), and the smooth-L1 loss is finite with
no NaN/Inf. A target-feature standard deviation of ≈1.0 is expected after the
official per-token `layer_norm`.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" ./.venv/bin/python -u -m tools.verify_vith_pretrained \
    --checkpoint /data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar \
    --json reports/vith_checkpoint_loading.json
```

The full command re-hashes the 10 GB checkpoint; pass
`--sha256 0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`
to reuse the recorded digest.

## Operation A vs Operation B

| | Operation A (pretrained init) | Operation B (training resume) |
| --- | --- | --- |
| Source | official ImageNet-1K I-JEPA | this project's Stage 2 checkpoint |
| Loads | encoder, predictor, target_encoder | encoder, predictor, target_encoder, opt, scaler, schedules, RNG |
| Optimizer | new | restored |
| LR/WD schedule | new | fast-forwarded to the resume step |
| Epoch / step counters | reset | restored |
| Code | `src/pretrained.py` | `src/helper.load_checkpoint` + `src/train.py` |
| Config trigger | `meta.pretrained_checkpoint`, `meta.load_checkpoint: false` | `meta.load_checkpoint: true` |

The two are separate code paths and are never mixed: when
`meta.load_checkpoint` is true the pretrained-init path is skipped entirely.
