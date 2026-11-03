#!/usr/bin/env bash
# Lightweight resource monitor for the Stage 3B/4A training runs.
#
# Usage:
#   tools/monitor_training.sh <gpu_index> <log_csv> [interval_seconds]
#
# Writes one CSV row per sample:
#   timestamp,gpu_mem_used_mib,gpu_mem_total_mib,gpu_util_pct,
#   ram_used_gb,ram_avail_gb,swap_used_gb,load1,train_rss_gb
#
# The monitor is read-only: it never touches the training process, its
# checkpoints, or the GPU.  Stop it with Ctrl-C / kill once training ends.
set -u

GPU="${1:-1}"
LOG="${2:-/tmp/agrisa_monitor.csv}"
INTERVAL="${3:-15}"

if [ ! -f "$LOG" ]; then
    echo "timestamp,gpu_mem_used_mib,gpu_mem_total_mib,gpu_util_pct,ram_used_gb,ram_avail_gb,swap_used_gb,load1,train_rss_gb" > "$LOG"
fi

while true; do
    ts="$(date '+%F %T')"
    gpu="$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    mem="$(LC_ALL=C free -m | awk '/^Mem:/{printf "%.1f %.1f", $3/1024, $7/1024}')"
    swap="$(LC_ALL=C free -m | awk '/^Swap:/{printf "%.1f", $3/1024}')"
    load1="$(awk '{print $1}' /proc/loadavg)"
    # -- sum RSS over the training process tree (uv wrapper -> python main.py
    #    -> multiprocessing spawn child).  A plain `ps | grep main.py` misses
    #    the spawn child, which is the process that actually holds the data.
    roots="$(pgrep -f 'main\.py .*--fname' | tr '\n' ' ')"
    pids="$roots"
    frontier="$roots"
    while [ -n "$(echo "$frontier" | tr -d ' ')" ]; do
        next=""
        for p in $frontier; do
            next="$next $(pgrep -P "$p" 2>/dev/null | tr '\n' ' ')"
        done
        pids="$pids $next"
        frontier="$next"
    done
    rss="$(ps -o rss= -p "$(echo "$pids" | tr -s ' ' ',' | sed 's/^,//;s/,$//')" 2>/dev/null \
            | awk '{s+=$1} END {printf "%.1f", s/1048576}')"
    echo "${ts},${gpu},${mem// /,},${swap},${load1},${rss:-0.0}" >> "$LOG"
    sleep "$INTERVAL"
done
