#!/usr/bin/env bash
# Stage 4B: train G0/G1/G2/G3 for 500 updates on one A6000 (GPU1).
#
# Every arm:
#   * starts from the official I-JEPA ViT-H/14 (load_checkpoint: false)
#   * uses the 2,000-update schedule and is INTERRUPTED at update 500
#   * shares the v1 manifest, batch size, augmentation, optimizer and protocol
#
# Logs: /home/test/agrisa_runs/logs/stage4b_<arm>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=.venv-uv
DEVICE=${DEVICE:-cuda:1}
LOG_DIR=${LOG_DIR:-/home/test/agrisa_runs/logs}
mkdir -p "$LOG_DIR"

for ARM in "$@"; do
  case "$ARM" in
    G0) CONFIG=configs/stage4b_geometry_only.yaml ;;
    G1) CONFIG=configs/stage4b_vegetation_controlled.yaml ;;
    G2) CONFIG=configs/stage4b_boundary_controlled.yaml ;;
    G3) CONFIG=configs/stage4b_shifted_boundary.yaml ;;
    *) echo "unknown arm $ARM"; exit 1 ;;
  esac
  echo "=== $ARM : $CONFIG ==="
  uv run --no-active python -u main.py --fname "$CONFIG" --devices "$DEVICE" \
      2>&1 | tee "$LOG_DIR/stage4b_${ARM}.log"
done
