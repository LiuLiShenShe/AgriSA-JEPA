#!/usr/bin/env python
"""Extract a lightweight encoder snapshot from a full training checkpoint.

A ViT-H training checkpoint carries the encoder, the predictor, the target
encoder and the optimizer state: about 10 GB each.  The downstream probe only
needs the encoder, so every evaluation point is turned into a small standalone
file that records exactly which run it came from.

The snapshot keeps the provenance that makes a result auditable:

* the optimizer-step count the weights were taken at,
* the SHA-256 of the training manifest,
* the SHA-256 of the official initialisation the run started from,
* the git commit and the training config.

Usage:
    ./.venv/bin/python -m tools.extract_encoder_snapshot \
        --checkpoint /data/agrisa_jepa_runs/stage3b_random_2000/jepa-latest.pth.tar \
        --out /data/agrisa_jepa_runs/stage3b_random_2000/jepa-encoder-step2000.pth
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expect-steps', type=int, default=None,
                        help='fail if the checkpoint is not at this update count')
    parser.add_argument('--expect-manifest-sha', default=None)
    args = parser.parse_args()

    source = Path(args.checkpoint)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if 'encoder' not in payload:
        raise KeyError(f'{source} has no encoder field')

    steps = int(payload.get('optimizer_steps', -1))
    if args.expect_steps is not None and steps != args.expect_steps:
        raise RuntimeError(
            f'{source} is at optimizer step {steps}, expected {args.expect_steps}')
    manifest_sha = payload.get('manifest_sha256')
    if args.expect_manifest_sha and manifest_sha != args.expect_manifest_sha:
        raise RuntimeError(
            f'{source} was trained on manifest {manifest_sha}, expected '
            f'{args.expect_manifest_sha}')

    pretrained = payload.get('pretrained_init') or {}
    snapshot = {
        'encoder': payload['encoder'],
        'optimizer_steps': steps,
        'epoch': payload.get('epoch'),
        'micro_steps': payload.get('micro_steps'),
        'manifest_sha256': manifest_sha,
        'pretrained_init_sha256': pretrained.get('sha256'),
        'git_commit': payload.get('git_commit'),
        'config': payload.get('config'),
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': sha256_file(source),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, out)

    report = {
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': snapshot['source_checkpoint_sha256'],
        'out': str(out),
        'out_sha256': sha256_file(out),
        'optimizer_steps': steps,
        'manifest_sha256': manifest_sha,
        'pretrained_init_sha256': snapshot['pretrained_init_sha256'],
        'git_commit': snapshot['git_commit'],
    }
    print(json.dumps(report, indent=2))
    report_path = out.with_suffix(out.suffix + '.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
