# I-JEPA

Official PyTorch codebase for I-JEPA (the **Image-based Joint-Embedding Predictive Architecture**) published @ CVPR-23.
[\[arXiv\]](https://arxiv.org/pdf/2301.08243.pdf) [\[JEPAs\]](https://ai.facebook.com/blog/yann-lecun-advances-in-ai-research/) [\[blogpost\]](https://ai.facebook.com/blog/yann-lecun-ai-model-i-jepa/)

## Method
I-JEPA is a method for self-supervised learning. At a high level, I-JEPA predicts the representations of part of an image from the representations of other parts of the same image. Notably, this approach learns semantic image features:
1. without relying on pre-specified invariances to hand-crafted data transformations, which tend to be biased for particular downstream tasks,
2. and without having the model fill in pixel-level details, which tend to result in learning less semantically meaningful representations.

![ijepa](https://github.com/facebookresearch/ijepa/assets/7530871/dbad94ab-ac35-433b-8b4c-ca227886d311)

## Visualizations

As opposed to generative methods that have a pixel decoder, I-JEPA has a predictor that makes predictions in latent space.
The predictor in I-JEPA can be seen as a primitive (and restricted) world-model that is able to model spatial uncertainty in a static image from a partially observable context.
This world model is semantic in the sense that it predicts high level information about unseen regions in the image, rather than pixel-level details.

We trained a stochastic decoder that maps the I-JEPA predicted representations back in pixel space as sketches.
The model correctly captures positional uncertainty and produces high-level object parts with the correct pose (e.g., dog’s head, wolf’s front legs).

![ijepa-predictor-sketch](https://github.com/facebookresearch/ijepa/assets/7530871/9b66e461-fc8b-4b12-9f06-63ec4dfc1452)
<sub>
Caption: Illustrating how the predictor learns to model the semantics of the world. For each image, the portion outside of the blue box is encoded and given to the predictor as context. The predictor outputs a representation for what it expects to be in the region within the blue box. To visualize the prediction, we train a generative model that produces a sketch of the contents represented by the predictor output, and we show a sample output within the blue box. The predictor recognizes the semantics of what parts should be filled in (the top of the dog’s head, the bird’s leg, the wolf’s legs, the other side of the building).
</sub>

## Evaluations

I-JEPA pretraining is also computationally efficient.
It does not involve any overhead associated with applying more computationally intensive data augmentations to produce multiple views.
Only one view of the image needs to be processed by the target encoder, and only the context blocks need to be processed by the context encoder.
Empirically, I-JEPA learns strong off-the-shelf semantic representations without the use of hand-crafted view augmentations.

![1percenteval](https://github.com/facebookresearch/ijepa/assets/7530871/e6e5291f-ca51-43a4-a6cf-069811094ece)
![lineareval](https://github.com/facebookresearch/ijepa/assets/7530871/d8cffa73-5350-444e-987a-7e131a86d767)


## Pretrained models

<table>
  <tr>
    <th colspan="1">arch.</th>
    <th colspan="1">patch size</th>
    <th colspan="1">resolution</th>
    <th colspan="1">epochs</th>
    <th colspan="1">data</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>300</td>
    <td>ImageNet-1K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>16x16</td>
    <td>448x448</td>
    <td>300</td>
    <td>ImageNet-1K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith16-448_ep300.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>66</td>
    <td>ImageNet-22K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vith14_ep66.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g</td>
    <td>16x16</td>
    <td>224x224</td>
    <td>44</td>
    <td>ImageNet-22K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vitg16_ep44.yaml">configs</a></td>
  </tr>
</table>

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── train.py              #   the I-JEPA training loop
│   ├── helper.py             #   helper functions for init of models & opt/loading checkpoint
│   ├── transforms.py         #   pre-train data transforms
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
├── main_distributed.py       # entrypoint for launch distributed I-JEPA pretraining on SLURM cluster
└── main.py                   # entrypoint for launch I-JEPA pretraining locally on your machine
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.

## Launching I-JEPA pretraining

### Single-GPU training
This implementation starts from the [main.py](main.py), which parses the experiment config file and runs the pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run I-JEPA pretraining on GPUs "0","1", and "2" on a local machine using the config [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2
```
*Note: This example is just used for illustrative purposes, as the ViT-H/14 config should be run on 16 A100 80G GPUs for an effective batch-size of 2048, in order to reproduce our results.*

### Multi-GPU training
In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to pre-train on 16 A100 80G GPUs using the pre-training experiment configs specificed inside [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main_distributed.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

---

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.0
* torchvision
* Other dependencies: pyyaml, numpy, opencv, submitit

## AgriSA-JEPA Stage 1

The local Stage 1 work keeps the official I-JEPA implementation and adds a manifest-backed AgriField-40K loader, conservative provenance audit, and small real-image training configs. The upstream checkout is pinned at `52c1ae95d05f743e000e8f10a1f3a79b10cff048`. No agriculture-specific masking method or long training run is included.

### Isolated environment

The tested environment is a project-local Conda prefix. These commands do not install packages into the base Python environment:

```bash
conda create --yes --prefix .venv python=3.13 pip
conda run --prefix .venv python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r environment/requirements-lock.txt
conda run --prefix .venv python -m pip check
```

### Dataset audit and loader check

The verified archive and extracted images live outside Git at `/data/datasets/AgriField-40K/`. The audit expects `AgriField-40K.zip` and `agrifield40k/{train,val}` under that directory. Use `--extract` only when the archive has not already been extracted.

```bash
./.venv/bin/python -u -m tools.audit_data --workers 8 --verify-archive
./.venv/bin/python -u -m tools.validate_loader \
  --manifest pretrain_manifest.csv \
  --root /data/datasets/AgriField-40K \
  --workers 2
```

### Real-image smoke training and resume

The smoke config uses 40 confirmed ACRECropWeed training images, ViT-B/16, 224-pixel input, batch size 4, and the upstream random multi-block masks. The first command trains one epoch and saves a checkpoint; the second restores it and runs the next epoch.

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m tools.synthetic_smoke --device cuda:1
./.venv/bin/python -u main.py \
  --fname configs/original_smoke.yaml --devices cuda:1
./.venv/bin/python -u main.py \
  --fname configs/original_smoke_resume.yaml --devices cuda:1
```

Training logs, verification YAML, and checkpoints are written under `runs/` (ignored by Git). The checked results and limitations are recorded in `reports/`. The dataset card and its individual upstream licenses govern reuse; a manifest entry is not independent legal clearance.

### Stage 2 — ViT-H/14 pilot and the versioned data freeze

Stage 2 moved the backbone to the official ViT-H/14, froze a versioned corpus, and measured single-GPU throughput.

```bash
# re-audit + regenerate every audit product (also re-runs the leakage check)
./.venv/bin/python -u -m tools.stage2_data_audit

# structural leakage checks only, on any pair of manifests
./.venv/bin/python tools/leakage_check.py \
  --train data/frozen/agrifield40k-v0_train.csv \
  --val   data/frozen/agrifield40k-v0_val.csv \
  --md reports/leakage_check_v0.md \
  --json reports/leakage_check_v0.json

# ViT-H/14 single-GPU benchmark sweep
./.venv/bin/python tools/bench_vith.py
```

Frozen corpus: `data_version=agrifield40k-v0`, train 12,447 / val 3,145,
train ∩ val = ∅, manifest SHA-256 in `data/frozen/freeze_manifest.json`. It is
**PROVISIONAL** — see [`reports/stage2c_data_gate.md`](reports/stage2c_data_gate.md).
Two leaks were measured and are *not* fixed: ACRECropWeed frames 2 s apart
across the split, and 31 iNatWeeds observations straddling it. Any number
computed on this split is exploratory until those are re-split.

### Stage 2C — 500-update pilot, resume fidelity, val-loss diagnostic

```bash
# bounded pilot: 500 optimizer updates, batch 16 x grad_accum 8
./.venv/bin/python -u main.py \
  --fname configs/stage3_armB_pilot_500.yaml --devices cuda:1

# resume-consistency: three arms, must agree to the last digit
./.venv/bin/python -u main.py \
  --fname configs/stage2c_resume_baseline.yaml --devices cuda:1
./.venv/bin/python -u main.py \
  --fname configs/stage2c_resume_interrupted.yaml --devices cuda:1
AGRISA_DEBUG_BATCH=1 ./.venv/bin/python -u main.py \
  --fname configs/stage2c_resume_resumed.yaml --devices cuda:1

# offline JEPA val loss for a checkpoint (diagnostic, NOT a result)
./.venv/bin/python tools/eval_jepa_val_loss.py \
  --checkpoint runs/stage3_armB_pilot_500/jepa-latest.pth.tar \
  --manifest data/frozen/agrifield40k-v0_val.csv \
  --out reports/jepa_val_loss_pilot500.json --num-batches 32
```

Results: pilot ran 500 updates in 3,965 micro-steps, loss 0.1908 → 0.1319,
peak 23.4 GB, mean 372 ms/update
([`reports/vith_single_gpu_benchmark.md`](reports/vith_single_gpu_benchmark.md)).
Resume fidelity is exact — max abs loss difference **0.000e+00** across the
interruption boundary
([`reports/stage2c_resume_consistency.md`](reports/stage2c_resume_consistency.md)).
The val-loss number (0.1322 mean) is a self-supervised diagnostic on a manifest
with known leakage; it is not a generalisation result and is not reported as one.

The full-budget Arm B candidate is `configs/stage3_baseline_candidate.yaml`
(30,000 updates). It is a planning artefact and has **not** been executed.

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
