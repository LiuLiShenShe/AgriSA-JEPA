# AgriSA-JEPA Stage 3 — Standard JEPA Baseline Protocol (Candidate)

**Status:** candidate protocol. Nothing here has been validated by a full run;
every hyperparameter below is a proposal for Stage 3 review, not a confirmed
final recipe. The Stage 2 measurements in
[`vith_single_gpu_benchmark.md`](vith_single_gpu_benchmark.md) fix what the
single A6000 can sustain.

## Goal

Fix the comparison arms and control every variable that could otherwise be
blamed for a difference, so that the Stage 4 agricultural structure-aware mask
can be evaluated against a standard I-JEPA baseline under an equal budget.

## Arm definitions

| Arm | Backbone | Initialization | Continued training | Mask |
| --- | --- | --- | --- | --- |
| **A** | ViT-H/14 @ 224 | Official `IN1K-vit.h.14-300e.pth.tar` | none | n/a (evaluation only) |
| **B** | ViT-H/14 @ 224 | Official `IN1K-vit.h.14-300e.pth.tar` | standard I-JEPA on AgriField | official random multi-block |
| **C** | ViT-H/14 @ 224 | Official `IN1K-vit.h.14-300e.pth.tar` | same budget as B | agricultural structure-aware mask (**not implemented in Stage 2**) |

**Arm B and arm C must each start from the same official checkpoint and train
for the same number of optimizer updates.** They are two independent runs, not
B-then-C chained. Chaining would give C both more total optimizer steps and a
warmed-up encoder, which invalidates a same-budget comparison.

Arm A is evaluated with the official weights untouched. It is the control that
shows what continued agricultural pretraining adds over ImageNet I-JEPA alone.

## Controls held fixed across all arms

| Control | Value | Source |
| --- | --- | --- |
| Backbone | `vit_huge`, ViT-H/14, embed 1280, depth 32, heads 16 | official `configs/in1k_vith14_ep300.yaml` |
| Predictor | depth 12, embed 384 | official |
| Initialization | `IN1K-vit.h.14-300e.pth.tar`, SHA-256 `0382013c…aae22` | official download |
| Resolution / patches | 224×224, 256 patches | official |
| Optimizer | AdamW, β=(0.9, 0.999), no warmup schedule change | official |
| Data version | frozen `agrifield40k_v0` train split | [`data_freeze.md`](data_freeze.md) |
| Data augmentation | RandomResizedCrop(0.3–1.0) + normalize; no blur, no jitter, no flip | official + frozen manifest |
| Mask (B) | official random multi-block: 1 context block (scale 0.85–1.0, AR 1.0), 4 target blocks (scale 0.15–0.2, AR 0.75–1.5), `min_keep=10`, no overlap | official |
| Loss | smooth-L1 over masked target features, target normalized with `layer_norm` | upstream `src/train.py` |
| EMA | 0.996 → 1.0 linear over the run, one update per optimizer step | official |
| Random seeds | at least 3 seeds per arm | this protocol |

## Batch and budget on a single RTX A6000 48GB

The official recipe uses 64 A100-class devices at batch 2048. That is not
reproducible here. Stage 2 measures what one A6000 can actually do, so the
candidate budget below is expressed in optimizer steps and is provisional until
the benchmark report is final.

**Candidate budget (to be confirmed against measured throughput):**

| Setting | Candidate value | Note |
| --- | --- | --- |
| Micro-batch | 1–4, per measured OOM headroom | Stage 2 measured |
| Gradient accumulation | sized so the effective batch is 256–512 | see benchmark report |
| Effective batch | 256 or 512 | proposal |
| Peak LR | 1e-4 – 3e-4, warmup 5% | reduced from official 1e-3 because the effective batch is smaller |
| Weight decay | 0.04 → 0.4 cosine | official |
| Optimizer steps | ~30k–60k (see projected wall-clock in the benchmark report) | proposal |
| Seeds | 3 | protocol |

**This is explicitly not a 300-epoch reproduction.** Arm B is a continued-pretraining
run on 12,458 audited agricultural images, so its step count is set by what one
GPU can complete, not by matching the official ImageNet epoch count. State the
actual completed step count in every Stage 3 report.

## Evaluation protocol

All arms are evaluated identically after pretraining:

1. **Linear probe** on the frozen context encoder over a fixed downstream split,
   with a fixed label budget and a fixed number of epochs.
2. **Few-shot transfer** at the same label budgets, averaged over the same seeds.
3. **k-NN** on frozen features as a cheap, hyperparameter-light second signal.
4. **Attentive probe** as the primary probe where a baseline exists.

Downstream splits come from the frozen data version and never overlap the
pretraining manifest. The Stage 1 val manifest (3,145 images) is the current
candidate; Stage 4 may add held-out benchmarks, but every arm must use the same
one. Report means and standard deviations over seeds, not a single best run.

## Prohibited shortcuts

- Training B first and then adding the new mask on B's weights for C.
- Reporting a short-run loss drop as downstream improvement.
- Changing data, augmentation, or probe between arms.
- Using `pending_audit` images to enlarge the training set mid-protocol.

## Open items before Stage 3 starts

1. Confirm the effective batch / step budget from the Stage 2 benchmark.
2. Confirm the frozen data version clears the source and duplicate gates.
3. Choose and freeze the downstream evaluation split and label budget.
4. Complete human adjudication of split-crossing perceptual duplicate candidates,
   or apply the documented conservative isolation rule and label it as such.

## Reproduction

```bash
# Stage 2 measurements that this budget is derived from
CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python -u tools/bench_vith.py \
    --batch-sizes 1 2 4 --grad-accum 8 --warmup 5 --measure 10 \
    --csv reports/vith_benchmark.csv --json reports/vith_benchmark.json
```
