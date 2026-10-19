# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch


def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):
    """Append-only CSV of per-micro-step training stats.

    A *fresh* run truncates its log file, so a new experiment that reuses a
    previous run's ``logging.folder`` can never silently inherit that run's
    rows.  A *resumed* run (``append=True``) appends, but first asserts the
    existing file's header matches the expected columns; a mismatch raises
    instead of writing rows under the wrong header.  Stage 2D added this to
    close the silent-append footgun in the original upstream logger, which
    opened every file in ``'+a'`` mode.
    """

    def __init__(self, fname, *argv, append: bool = False):
        self.fname = fname
        self.types = []
        header = []
        for i, v in enumerate(argv, 1):
            self.types.append(v[0])
            header.append(v[1])
        expected = ','.join(header)

        if append and os.path.exists(self.fname) and os.path.getsize(self.fname) > 0:
            with open(self.fname, 'r') as f:
                found = f.readline().rstrip('\n')
            if found != expected:
                raise ValueError(
                    f'Refusing to append to {self.fname}: its header\n  {found!r}\n'
                    f'does not match this run\'s columns\n  {expected!r}. '
                    f'Use a different logging.folder or start a fresh run.')
            self.resumed = True
        else:
            with open(self.fname, 'w') as f:
                print(expected, file=f)
            self.resumed = False

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats
