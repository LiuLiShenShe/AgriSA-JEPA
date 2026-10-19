# Stage 2C resume-consistency verification

**Question this answers:** does an interrupted run, resumed from its own
checkpoint, reproduce the uninterrupted run bit-for-bit? Stage 2C added
mid-epoch stopping (`interrupt_after_optimizer_steps`), an exact optimizer-step
budget (`max_optimizer_steps`), and epoch-start state snapshots to
`src/train.py`. This report is the evidence that those changes work.

Everything below was executed on one RTX A6000 (48 GB), ViT-H/14, 224-px crops,
batch 8, grad_accum 4, bf16, `num_workers: 0`.

## 1. Test design

Three arms on the frozen PROVISIONAL corpus
(`data/frozen/agrifield40k-v0_train.csv`, sha256
`d8a179b81f2052c6d66d2f43d26505652e95b1090560d13c1e9d46efe3b27037`), restricted
to `max_samples: 64`, seed `20261019`.

| Arm | Config | Budget | Where it stops |
| --- | --- | ---: | --- |
| A | `configs/stage2c_resume_baseline.yaml` | 10 updates | end of epoch 5 (epoch boundary) |
| B | `configs/stage2c_resume_interrupted.yaml` | 10, interrupted at 5 | **mid-epoch 3**, 4 micro-steps in |
| C | `configs/stage2c_resume_resumed.yaml` | 10, resumes B | end of epoch 5 |

Geometry: 64 images / batch 8 = 8 micro-steps per epoch; grad_accum 4 → 2
optimizer updates per epoch. Arm A's 10 updates land exactly on an epoch
boundary. Arm B's 5 updates land 2.5 epochs in — deliberately *not* on a
boundary, which is the case that exercises the new replay logic.

`num_workers: 0` is required, not incidental: with worker processes the
augmentation RNG is consumed inside children that cannot be checkpointed, so
exact equality would be unachievable. The comparison is only meaningful because
the whole data pipeline runs in-process.

Both A and B started from the same I-JEPA ViT-H/14 ImageNet-1K checkpoint
(`IN1K-vit.h.14-300e.pth.tar`, sha256
`0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22`).

## 2. Result: the resumed run reproduces the uninterrupted one exactly

Loss traces are in `runs/*/jepa_r0.csv`. Comparing row-by-row on the
`(epoch, itr)` keys each arm actually logged:

| Comparison | Result |
| --- | --- |
| Arm B ⊂ Arm A (prefix) | 19/19 rows present, **max abs loss diff = 0.000e+00** |
| Arm C ⊂ Arm A (tail) | 19/19 rows present, **max abs loss diff = 0.000e+00** |
| Arm A keys not covered by B ∪ C | only `(3, 3)` — see §4 |

Not "close". Identical to the last printed digit, across the interruption
boundary, including the LR/WD/momentum schedule fast-forward and the replayed
data pipeline.

Checkpoint-level agreement corroborates it:

| Field | Arm A | Arm B | Arm C |
| --- | ---: | ---: | ---: |
| `epoch` | 5 | 2 | 5 |
| `epoch_itr` | 0 | 4 | 0 |
| `micro_steps` | 40 | 20 | 40 |
| `optimizer_steps` | 10 | 5 | 10 |
| `epoch_start_state` present | no | **yes** | no |
| `loss` | 0.175817 | 0.190563 | 0.175817 |

Arm B's `epoch_itr=4` is what proves the stop was genuinely mid-epoch (4
micro-steps consumed of 8 in epoch 3), and its `epoch_start_state` snapshot is
what arm C replayed from. Arm C ends with exactly arm A's `micro_steps`,
`optimizer_steps` and `loss`.

## 3. What arm C actually did on resume

From the run log (`INFO` lines, verified on the re-run in §5):

```
INFO:root:loaded encoder from epoch 2: <All keys matched successfully>
INFO:root:Epoch 3
INFO:root:Replaying 4 already-consumed micro-steps of epoch 3
INFO:root:BATCH_FP epoch=2 itr=4 sha=f9e9399bcd33fcf7 sum=155701.359375 maskA=792 maskB=336 rng=ef313aa51f6e33eb
INFO:root:BATCH_FP epoch=3 itr=0 sha=ebb15e544e8bdf46 sum=188396.640625 maskA=552 maskB=384 rng=31df2ea2daeeb12a
INFO:root:BATCH_FP epoch=4 itr=0 sha=403f18da0ae67c4f sum=226048.062500 maskA=664 maskB=384 rng=b33e8904c7e021f3
INFO:root:Reached max_optimizer_steps=10; stopping.
```

The sequence is the point: it restores the epoch-start RNG/loader/mask-collator
snapshot, **replays** the 4 already-consumed micro-steps through the loader
(discarding the batches, keeping the side effects), then resumes training at
`itr=4` with a batch fingerprint identical to the uninterrupted run's.

`BATCH_FP` is emitted by the `AGRISA_DEBUG_BATCH` env var only. It fingerprints
the first real batch of each epoch — sha256 of the image tensor, its sum, mask
element counts, and the torch RNG state — so two runs can be diffed without
trusting the loss column alone. It is off by default and adds no code path to
normal training.

## 4. The one row nobody logged: `(3, 3)`

Arm A logs 39 rows. B logs 19 (epochs 1–3, `itr` 0–2). C logs 19 (epochs 3–5,
`itr` 4–7). The union misses exactly one key: epoch 3, `itr` 3 — the micro-step
that completed arm B's 5th optimizer update and triggered the break.

This is a logging-order artefact, not a training divergence. In `src/train.py`
the budget/interruption `break` is checked immediately after
`optimizer_steps += 1`, and `log_stats()` runs *after* it:

```python
if do_update:
    optimizer_steps += 1
if interrupt_after is not None and optimizer_steps >= interrupt_after:
    logger.info('Interrupted at ...')
    break
# -- Logging
def log_stats(): ...
log_stats()
```

So the micro-step that *causes* the stop is trained and applied but never
logged — in both arms. Arm A's `(3, 3)` row exists only because arm A's budget
(10) does not trigger a break at that point; its budget break fires at `(5, 6)`,
which is likewise the one arm-A row that B∪C would not cover.

The computation itself is covered: arm B's checkpoint records
`micro_steps=20, optimizer_steps=5`, i.e. 20 micro-steps trained — 4 full
windows of 4 plus the 4 micro-steps of the partial 5th window, which includes
`itr=3`. And arm C resumes from that state and reproduces arm A's `(3, 4)`
onward exactly, which would be impossible if `(3, 3)` had diverged.

**Known cosmetic gap, recorded rather than hidden:** if you want every trained
micro-step logged, `log_stats()` must move above the budget break. That is a
deliberate no-op for correctness and was left alone under the minimal-change
rule; it is noted here so nobody mistakes the missing row for a missing update.

## 5. Defect found and fixed during this verification

The three configs were written with `logging.folder` values
`runs/stage2c_resume_P` / `_Q` / `_R`, but the runs were later renamed to
`..._baseline` / `..._interrupted` / `..._resumed`. The configs were not
updated, so **as committed, arm C could not be re-run at all**:
`src/train.py` resolves `read_checkpoint` relative to `logging.folder`
(`os.path.join(folder, r_file)`), giving
`runs/stage2c_resume_R/../stage2c_resume_Q/jepa-latest.pth.tar` — a path that
does not exist.

Fixed in all three configs (`folder`, `read_checkpoint`, and the copy-pasted
header comments that wrongly called every arm "arm A"). Arm C was then re-run
end-to-end with the corrected config; it loaded from
`runs/stage2c_resume_resumed/../stage2c_resume_interrupted/jepa-latest.pth.tar`,
replayed, and produced a loss trace **identical to arm A** (max abs diff
0.000e+00 on the 19 shared keys, same `(3, 3)` logging gap).

The `jepa_r0.csv` in `runs/stage2c_resume_resumed/` now contains that fresh
run's block. The CSV logger appends rather than truncates, so the first re-run
left a duplicated block in the file; the duplicate was removed and the file
holds only the fresh run. Anyone re-running arm C on top of an existing folder
should clear or archive the CSV first.

## 6. What this does and does not prove

**Proven:** a run interrupted mid-epoch, saved, and resumed from that checkpoint
reproduces the uninterrupted run exactly — same data pipeline order, same masks,
same schedule position, same loss values — under `num_workers: 0`.

**Not proven:**
- Multi-worker determinism. With `num_workers > 0` the worker RNG streams are
  not checkpointed, so this guarantee does not extend there. Stage 3's 500-step
  pilot ran with `num_workers: 4`; its resume fidelity was not tested and is
  not claimed.
- Multi-GPU / distributed resume. Single device only.
- That the frozen corpus is fit for reported results — see
  [`leakage_check_v0.md`](leakage_check_v0.md). This report is about engineering
  reproducibility, not data quality.

## 7. Commands

```bash
# arm A — uninterrupted, 10 updates
./.venv/bin/python -u main.py \
  --fname configs/stage2c_resume_baseline.yaml --devices cuda:1

# arm B — same run, interrupted after 5 updates (mid-epoch)
./.venv/bin/python -u main.py \
  --fname configs/stage2c_resume_interrupted.yaml --devices cuda:1

# arm C — resume arm B to the full 10 updates
AGRISA_DEBUG_BATCH=1 ./.venv/bin/python -u main.py \
  --fname configs/stage2c_resume_resumed.yaml --devices cuda:1
```

`AGRISA_DEBUG_BATCH=1` is optional and only adds the `BATCH_FP` lines.
