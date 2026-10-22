#!/usr/bin/env python
"""Evaluate every Stage 3B / Stage 4A checkpoint on PhenoBench with one protocol.

For each evaluation point (arm x optimizer step) this driver

1. extracts the encoder from the full training checkpoint (once),
2. caches the frozen ViT-H features for the PhenoBench train and val splits,
3. trains the identical ``Linear(1280 -> 3)`` probe with the same seeds,
4. scores the official val split with ``phenobench-eval --task semantics`` and
   with the self-implemented IoU, and cross-checks the two.

The protocol (tiling, probe, optimizer, LR, epochs, batch size, seeds) is read
from one arm config, so no arm can accidentally get a different evaluation.

Feature caches are bound to the checkpoint SHA-256 and can be removed after a
checkpoint has been evaluated, which is what keeps the disk usage bounded.

Usage:
    ./.venv/bin/python -m tools.eval_stage4a_checkpoints \
        --runs-root /data/agrisa_jepa_runs \
        --cache-dir runs/stage4a_phenobench/cache \
        --out-dir runs/stage4a_phenobench/results
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.downstream.features import cache_key, sha256_file
from src.downstream.phenobench_probe import (
    cache_split_features,
    train_and_evaluate,
)

#: arm -> (run directory, config file)
ARMS: Dict[str, Tuple[str, str]] = {
    'B': ('stage3b_random_2000', 'configs/stage3b_random_2000.yaml'),
    'C1': ('stage4a_vegetation_2000', 'configs/stage4a_vegetation_2000.yaml'),
    'C2': ('stage4a_boundary_2000', 'configs/stage4a_boundary_2000.yaml'),
    'C3': ('stage4a_shuffled_2000', 'configs/stage4a_shuffled_2000.yaml'),
}

#: The official I-JEPA reference (arm A) is evaluated from the official weights.
ARM_A = {
    'arm': 'A',
    'run_dir': None,
    'config': 'configs/phenobench_armA_linear.yaml',
    'checkpoint': '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
    'label': 'official_i-jepa_vith14',
    'steps': [0],
    'continued_training': 'none',
    'data_version': None,
}


def protocol_from_config(path: str) -> Dict[str, object]:
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    probe = cfg['probe']
    return {
        'epochs': int(probe['epochs']),
        'lr': float(probe['lr']),
        'batch_size': int(probe['batch_size']),
        'weight_decay': float(probe['weight_decay']),
        'seeds': list(probe['seeds']),
    }


def snapshot_for(runs_root: Path, run_dir: str, step: int) -> Optional[Path]:
    """The encoder snapshot for an arm at a given optimizer step."""
    folder = runs_root / run_dir
    direct = folder / f'jepa-encoder-step{step}.pth'
    if direct.is_file():
        return direct
    # the final step lives in the rolling `latest` checkpoint
    if step > 0:
        latest = folder / 'jepa-latest.pth.tar'
        if latest.is_file():
            return None      # the caller extracts it
    return None


def extract_snapshot(runs_root: Path, run_dir: str, step: int) -> Path:
    folder = runs_root / run_dir
    direct = folder / f'jepa-encoder-step{step}.pth'
    if direct.is_file():
        return direct
    if step == 0:
        raise FileNotFoundError(f'no encoder snapshot for step 0 in {folder}')
    source = folder / (f'jepa-step{step}.pth.tar' if (folder / f'jepa-step{step}.pth.tar').is_file()
                       else 'jepa-latest.pth.tar')
    out = folder / f'jepa-encoder-step{step}.pth'
    cmd = [sys.executable, '-m', 'tools.extract_encoder_snapshot',
           '--checkpoint', str(source), '--out', str(out),
           '--expect-steps', str(step)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'encoder extraction failed:\n{proc.stdout}\n{proc.stderr}')
    return out


#: Feature caches are keyed by ``(label, checkpoint sha256)``.  Arm A is the
#: official checkpoint that Stage 3A already cached under its own label; reusing
#: that key keeps the Stage 3A artefacts untouched instead of recomputing 34 GB
#: of identical features.
LEGACY_CACHE_LABELS = {'A_0': 'official_i-jepa_vith14'}

#: every evaluation point this round runs.  ``tools.queue_stage4a_evals`` and
#: ``tools.compare_stage4a_phenobench`` import this table so that "which
#: checkpoints exist" has exactly one answer.
LABELS = {
    'A_0': ('A', 0),
    'B_500': ('B', 500),
    'B_2000': ('B', 2000),
    'C1_500': ('C1', 500),
    'C1_2000': ('C1', 2000),
    'C2_500': ('C2', 500),
    'C2_2000': ('C2', 2000),
    'C3_500': ('C3', 500),
    'C3_2000': ('C3', 2000),
}


def evaluate_one(
    label: str,
    checkpoint: Path,
    root: Path,
    cache_dir: Path,
    out_dir: Path,
    device: str,
    protocol: Dict[str, object],
    continued_training: str,
    data_version: Optional[str],
    keep_cache: bool,
    feature_batch_size: int,
) -> Dict[str, object]:
    result_path = out_dir / f'{label}' / 'result.json'
    if result_path.is_file():
        print(f'[{label}] already evaluated, skipping', flush=True)
        return json.loads(result_path.read_text(encoding='utf-8'))

    cache_label = LEGACY_CACHE_LABELS.get(label, label)
    if cache_label != label:
        print(f'[{label}] reusing the Stage 3A feature cache keyed as '
              f'{cache_label!r} (same checkpoint SHA-256)', flush=True)
    print(f'[{label}] caching features from {checkpoint}', flush=True)
    for split in ('train', 'val'):
        cache_split_features(
            None, root, cache_dir, split, torch_device(device),
            batch_size=feature_batch_size,
            checkpoint=str(checkpoint), arm_label=cache_label)

    payload = train_and_evaluate(
        None, root, cache_dir, out_dir / label, torch_device(device),
        epochs=protocol['epochs'], lr=protocol['lr'],
        batch_size=protocol['batch_size'],
        weight_decay=protocol['weight_decay'], seeds=protocol['seeds'],
        checkpoint=str(checkpoint), arm_label=cache_label,
        continued_training=continued_training, data_version=data_version)

    payload['evaluation_point'] = label
    (out_dir / label / 'result.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    if not keep_cache and cache_label == label:
        sha = sha256_file(checkpoint)
        key = cache_key(label, sha)
        shutil.rmtree(cache_dir / key, ignore_errors=True)
        print(f'[{label}] removed the feature cache {key}', flush=True)
    return payload


def torch_device(spec: str):
    import torch
    return torch.device(spec if torch.cuda.is_available() else 'cpu')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', default='/home/test/agrisa_runs')
    parser.add_argument('--root', default='/data/datasets/PhenoBench/PhenoBench-v110')
    parser.add_argument('--cache-dir', default='runs/stage4a_phenobench/cache')
    parser.add_argument('--out-dir', default='runs/stage4a_phenobench/results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--feature-batch-size', type=int, default=16)
    parser.add_argument('--keep-cache', action='store_true')
    parser.add_argument('--only', default=None,
                        help='comma-separated labels to evaluate, e.g. B_500,C2_2000')
    parser.add_argument('--skip-arm-a', action='store_true')
    parser.add_argument('--phenobench-cache', default='/home/test/agrisa_cache/phenobench-v110',
                        help='memory-mapped copy of the PhenoBench images and labels; '
                             'the bytes are identical to the PNGs, this only avoids '
                             'random reads on the spinning disk that holds them')
    args = parser.parse_args()

    if args.phenobench_cache:
        os.environ['AGRISA_PHENOBENCH_CACHE'] = str(Path(args.phenobench_cache).resolve())

    runs_root = Path(args.runs_root)
    root = Path(args.root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = protocol_from_config('configs/phenobench_armA_linear.yaml')
    print(f'evaluation protocol (identical for every arm): {json.dumps(protocol)}')

    wanted = set(args.only.split(',')) if args.only else None
    summary: List[Dict[str, object]] = []

    if not args.skip_arm_a:
        label = 'A_0'
        if wanted is None or label in wanted:
            t0 = time.time()
            payload = evaluate_one(
                label, Path(ARM_A['checkpoint']), root, cache_dir, out_dir,
                args.device, protocol, ARM_A['continued_training'],
                ARM_A['data_version'], args.keep_cache, args.feature_batch_size)
            summary.append({
                'label': label, 'arm': 'A', 'optimizer_steps': 0,
                'checkpoint': ARM_A['checkpoint'],
                'checkpoint_sha256': payload['backbone_checkpoint_sha256'],
                'aggregate': payload.get('aggregate'),
                'seconds': round(time.time() - t0, 1),
            })

    for arm, (run_dir, config) in ARMS.items():
        cfg = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        steps = [int(s) for s in cfg['optimization']['save_at_optimizer_steps']]
        steps.append(int(cfg['optimization']['max_optimizer_steps']))
        for step in steps:
            label = f'{arm}_{step}'
            if wanted is not None and label not in wanted:
                continue
            if not (runs_root / run_dir).is_dir():
                print(f'[{label}] run directory missing, skipping', flush=True)
                continue
            t0 = time.time()
            snapshot = extract_snapshot(runs_root, run_dir, step)
            payload = evaluate_one(
                label, snapshot, root, cache_dir, out_dir, args.device, protocol,
                f"{cfg['optimization']['max_optimizer_steps']} optimizer updates, "
                f"mask={cfg['mask'].get('mode')}"
                + (f"/{cfg['mask'].get('prior_mode')}"
                   if cfg['mask'].get('prior_mode') else ''),
                f"agrifield40k-v1 (PROVISIONAL), {cfg['optimization']['max_optimizer_steps']} updates",
                args.keep_cache, args.feature_batch_size)
            summary.append({
                'label': label, 'arm': arm, 'optimizer_steps': step,
                'checkpoint': str(snapshot),
                'checkpoint_sha256': payload['backbone_checkpoint_sha256'],
                'aggregate': payload.get('aggregate'),
                'seconds': round(time.time() - t0, 1),
            })

    # The driver is routinely invoked with --only for one arm at a time, so the
    # index is merged with whatever is already on disk instead of overwritten.
    index = out_dir / 'index.json'
    merged: List[Dict[str, object]] = []
    if index.is_file():
        try:
            previous = json.loads(index.read_text(encoding='utf-8'))
            if isinstance(previous, list):
                merged = [entry for entry in previous
                          if isinstance(entry, dict) and 'label' in entry]
        except json.JSONDecodeError:
            merged = []
    labels = {entry['label'] for entry in merged}
    for entry in summary:
        if entry['label'] in labels:
            merged = [e for e in merged if e['label'] != entry['label']]
        merged.append(entry)
    merged.sort(key=lambda e: str(e['label']))
    index.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + '\n',
                     encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
