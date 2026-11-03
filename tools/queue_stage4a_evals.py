#!/usr/bin/env python
"""Run the remaining Stage 4A evaluations in a fixed order on one GPU.

The driver is invoked one label at a time so that a label whose checkpoint has
not been produced yet (the arm is still training) simply waits for the
snapshot file to appear instead of failing.

Usage:
    ./.venv/bin/python -m tools.queue_stage4a_evals \
        --device cuda:0 --labels C1_500,C1_2000 --runs-root /home/test/agrisa_runs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

#: label -> (run directory, expected encoder snapshot file)
SNAPSHOTS: Dict[str, Tuple[str, str]] = {
    'B_500': ('stage3b_random_2000', 'jepa-encoder-step500.pth'),
    'B_2000': ('stage3b_random_2000', 'jepa-encoder-step2000.pth'),
    'C1_500': ('stage4a_vegetation_2000', 'jepa-encoder-step500.pth'),
    'C1_2000': ('stage4a_vegetation_2000', 'jepa-encoder-step2000.pth'),
    'C2_500': ('stage4a_boundary_2000', 'jepa-encoder-step500.pth'),
    'C2_2000': ('stage4a_boundary_2000', 'jepa-encoder-step2000.pth'),
    'C3_500': ('stage4a_shuffled_2000', 'jepa-encoder-step500.pth'),
    'C3_2000': ('stage4a_shuffled_2000', 'jepa-encoder-step2000.pth'),
}


def other_evals_running(only: str) -> bool:
    """True when a different evaluation process is still in flight."""
    proc = subprocess.run(['pgrep', '-af', 'tools.eval_stage4a_checkpoints'],
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if 'queue_stage4a_evals' in line:
            continue
        if f'--only {only}' in line:
            continue
        return True
    return False


def evaluation_lock():
    """One evaluation at a time across the whole host.

    The probe loads every cached feature tile for the train and the val split
    into memory (~36 GB fp16 per evaluation).  Two evaluations on two GPUs still
    exhaust the 125 GB of host RAM, so the lock is machine-wide, not per-GPU.
    """
    import fcntl
    lock_dir = Path('runs/stage4a_phenobench')
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = open(lock_dir / 'evaluation.lock', 'w')
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--labels', required=True)
    parser.add_argument('--runs-root', default='/home/test/agrisa_runs')
    parser.add_argument('--wait-timeout', type=float, default=14400.0)
    parser.add_argument('--poll', type=float, default=30.0)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    results = Path('runs/stage4a_phenobench/results')
    lock = evaluation_lock()
    labels: List[str] = [x.strip() for x in args.labels.split(',') if x.strip()]

    for label in labels:
        run_dir, snapshot = SNAPSHOTS[label]
        path = runs_root / run_dir / snapshot
        deadline = time.time() + args.wait_timeout
        while not path.is_file():
            if time.time() > deadline:
                print(f'[{label}] timed out waiting for {path}', flush=True)
                return 1
            print(f'[{label}] waiting for {path} '
                  f'({(deadline - time.time()) / 60:.0f} min left)', flush=True)
            time.sleep(args.poll)
        # never run two evaluations on the same host at the same time (RAM)
        while other_evals_running(label):
            time.sleep(args.poll)
        done = results / label / 'result.json'
        if done.is_file():
            print(f'[{label}] already evaluated', flush=True)
            continue
        cmd = [sys.executable, '-m', 'tools.eval_stage4a_checkpoints',
               '--only', label, '--runs-root', str(runs_root),
               '--device', args.device]
        print(f'[{label}] launching: {" ".join(cmd)}', flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd)
        print(f'[{label}] exit={proc.returncode} in {time.time() - t0:.0f}s',
              flush=True)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
