#!/bin/bash
# Stage 8 Phase 3: smoke test — verify the full pipeline (features -> probe
# train -> evaluate) for Official, A_p1 and B_p1 on each new task, with
# minimal data (--limit-images) before any formal run.  Serial on GPU 1.
set -u
cd /fj/jepa-二维图像任务
export CUDA_VISIBLE_DEVICES=1
PY=./.venv-uv/bin/python
OUT=/data/agrisa_jepa_runs/stage8/smoke
CACHE=/data/agrisa_jepa_runs/stage8/smoke_cache
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo "===== SMOKE $name ====="
  "$@" || { echo "!!! SMOKE FAILED: $name (rc=$?)"; exit 1; }
  echo "===== SMOKE OK: $name ====="
}

cropandweed() {
  $PY -m src.downstream.cropandweed_probe \
    --checkpoint "$1" --label "$2" \
    --images-root /data/datasets/cropandweed/images \
    --annotation-root /fj/jepa-二维图像任务/data/holdout_raw/cropandweed_annotations/labelIds/CropAndWeed \
    --cache-dir "$CACHE/cropandweed" --out-dir "$OUT/cropandweed/$2" \
    --seed 0 --limit-images 12 --feature-batch-size 4 --batch-size 8 --epochs 2
}

deepweeds() {
  $PY -m src.downstream.deepweeds_probe \
    --checkpoint "$1" --label "$2" \
    --labels /fj/jepa-二维图像任务/data/holdout_raw/deepweeds_labels \
    --images-root /data/datasets/DeepWeeds/images \
    --cache-dir "$CACHE/deepweeds" --out-dir "$OUT/deepweeds/$2" \
    --seed 0 --limit-images 30 --feature-batch-size 4 --batch-size 32 --epochs 2
}

plantdoc() {
  $PY -m src.downstream.plantdoc_probe \
    --checkpoint "$1" --label "$2" \
    --root /fj/jepa-二维图像任务/data/holdout_raw/plantdoc_extract/PlantDoc-Dataset-master \
    --cache-dir "$CACHE/plantdoc" --out-dir "$OUT/plantdoc/$2" \
    --seed 0 --limit-images 30 --feature-batch-size 4 --batch-size 32 --epochs 2
}

# backbone checkpoints
OFFICIAL=/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar
A_P1=/data/agrisa_jepa_runs/stage7b/stage7b_A_p1/jepa-encoder-step2000.pth
B_P1=/data/agrisa_jepa_runs/stage7b/stage7b_B_p1/jepa-encoder-step2000.pth

run cropandweed-OFFICIAL cropandweed "$OFFICIAL" "smoke_official"
run cropandweed-A_p1    cropandweed "$A_P1" "smoke_A_p1"
run cropandweed-B_p1    cropandweed "$B_P1" "smoke_B_p1"
run deepweeds-OFFICIAL  deepweeds "$OFFICIAL" "smoke_official"
run deepweeds-A_p1      deepweeds "$A_P1" "smoke_A_p1"
run deepweeds-B_p1      deepweeds "$B_P1" "smoke_B_p1"
run plantdoc-OFFICIAL   plantdoc "$OFFICIAL" "smoke_official"
run plantdoc-A_p1       plantdoc "$A_P1" "smoke_A_p1"
run plantdoc-B_p1       plantdoc "$B_P1" "smoke_B_p1"

echo "ALL_SMOKE_OK"