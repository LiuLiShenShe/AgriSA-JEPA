#!/usr/bin/env bash
# Resume the two interrupted Stage 4A arms (C1 vegetation, C3 shuffled) to the
# full 2,000-optimizer-update budget on GPU 1 only, using the uv environment.
#
# Live logs (updated in real time, one PROGRESS line per optimizer update):
#   /home/test/agrisa_runs/logs/C1_vegetation_resume.log
#   /home/test/agrisa_runs/logs/C3_shuffled_resume.log
# Resource samples:
#   /home/test/agrisa_runs/logs/resources_gpu1.csv
#
# Usage: tools/run_stage4a_resume.sh
set -euo pipefail

cd /fj/jepa-二维图像任务
export UV_PROJECT_ENVIRONMENT=.venv-uv
export PYTHONUNBUFFERED=1

LOGDIR=/home/test/agrisa_runs/logs
mkdir -p "$LOGDIR"

tools/monitor_training.sh 1 "$LOGDIR/resources_gpu1.csv" 15 &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

run_one() {
    local name="$1"
    local cfg="$2"
    local log="$LOGDIR/${name}.log"
    {
        echo "===== START ${name} $(date '+%F %T') ====="
        echo "config: ${cfg}"
        echo "python: $(.venv-uv/bin/python -V 2>&1)"
        echo "gpu:    $(nvidia-smi --id=1 --query-gpu=name,memory.total --format=csv,noheader)"
    } | tee -a "$log"

    set +e
    uv run --no-active python -u main.py --fname "$cfg" --devices cuda:1 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e

    echo "===== END ${name} rc=${rc} $(date '+%F %T') =====" | tee -a "$log"
    return "$rc"
}

run_one C1_vegetation_resume configs/stage4a_vegetation_2000_resume.yaml
run_one C3_shuffled_resume configs/stage4a_shuffled_2000_resume.yaml