#!/usr/bin/env python
"""Assert the Stage 4B pilot configs are a mask-only comparison.

Reference arm B is the already-trained 2,000-update random-mask run from Stage
3B.  The four new arms are 500-update interruptions of the SAME 2,000-update
schedule, so the learning-rate / weight-decay / EMA trajectory at update 500 is
identical to B_500.  This tool diffs the YAML field by field and fails on any
difference outside the documented mask-strategy and arm-identity keys.

Usage:
    ./.venv/bin/python -m tools.check_stage4b_protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

REFERENCE = 'configs/stage3b_random_2000.yaml'
CONFIGS = {
    'G0': 'configs/stage4b_geometry_only.yaml',
    'G1': 'configs/stage4b_vegetation_controlled.yaml',
    'G2': 'configs/stage4b_boundary_controlled.yaml',
    'G3': 'configs/stage4b_shifted_boundary.yaml',
}

#: Keys that may differ between B and the G arms (mask strategy + identity).
ALLOWED_DIFFERENT = {
    ('meta', 'stage'),
    ('meta', 'arm'),
    ('meta', 'arm_label'),
    ('mask', 'mode'),
    ('mask', 'prior_mode'),
    ('mask', 'score_mode'),
    ('mask', 'guidance_lambda'),
    ('mask', 'geometry_beta'),
    ('mask', 'geometry_control'),
    ('mask', 'num_candidates'),
    ('mask', 'base_seed'),
    ('mask', 'roll_shift'),
    ('mask', 'boundary_edge_weight'),
    ('logging', 'folder'),
    ('optimization', 'interrupt_after_optimizer_steps'),
}

MUST_MATCH = (
    ('data', 'manifest'),
    ('data', 'manifest_sha256'),
    ('data', 'batch_size'),
    ('data', 'num_workers'),
    ('data', 'augmentation_seed'),
    ('data', 'seed'),
    ('data', 'crop_size'),
    ('data', 'crop_scale'),
    ('data', 'root_path'),
    ('data', 'image_cache_dir'),
    ('meta', 'pretrained_checkpoint'),
    ('meta', 'pretrained_sha256'),
    ('meta', 'model_name'),
    ('meta', 'patch_size'),
    ('meta', 'pred_depth'),
    ('meta', 'pred_emb_dim'),
    ('meta', 'use_bfloat16'),
    ('meta', 'use_sdpa'),
    ('meta', 'activation_checkpointing'),
    ('meta', 'load_checkpoint'),
    ('optimization', 'max_optimizer_steps'),
    ('optimization', 'save_at_optimizer_steps'),
    ('optimization', 'grad_accum'),
    ('optimization', 'lr'),
    ('optimization', 'start_lr'),
    ('optimization', 'final_lr'),
    ('optimization', 'weight_decay'),
    ('optimization', 'final_weight_decay'),
    ('optimization', 'ema'),
    ('optimization', 'warmup'),
    ('optimization', 'epochs'),
    ('optimization', 'ipe_scale'),
    ('mask', 'allow_overlap'),
    ('mask', 'aspect_ratio'),
    ('mask', 'enc_mask_scale'),
    ('mask', 'pred_mask_scale'),
    ('mask', 'num_enc_masks'),
    ('mask', 'num_pred_masks'),
    ('mask', 'min_keep'),
    ('mask', 'patch_size'),
)


def flatten(node, prefix=()) -> Dict[Tuple[str, ...], object]:
    out: Dict[Tuple[str, ...], object] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, prefix + (str(key),)))
    else:
        out[prefix] = node
    return out


def get(flat, key, default=None):
    return flat.get(key, default)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=REFERENCE)
    parser.add_argument('--out', default='reports/stage4b_protocol_check.json')
    args = parser.parse_args()

    paths = {'B': Path(args.reference), **{a: Path(p) for a, p in CONFIGS.items()}}
    flats = {arm: flatten(yaml.safe_load(path.read_text(encoding='utf-8')))
             for arm, path in paths.items()}

    problems: List[str] = []
    all_keys = set()
    for flat in flats.values():
        all_keys |= set(flat)
    for key in sorted(all_keys):
        values = {arm: get(flat, key) for arm, flat in flats.items()}
        present = {arm: v for arm, v in values.items() if v is not None}
        if len(set(map(str, present.values()))) > 1 and key not in ALLOWED_DIFFERENT:
            problems.append(f'disallowed difference at {".".join(key)}: {values}')

    for key in MUST_MATCH:
        values = {arm: get(flat, key) for arm, flat in flats.items()}
        if len(set(map(str, values.values()))) > 1:
            problems.append(f'{".".join(key)} must match across arms, got {values}')

    # -- exact protocol invariants
    for arm, flat in flats.items():
        budget = get(flat, ('optimization', 'max_optimizer_steps'))
        if budget != 2000:
            problems.append(f'{arm}: expected a 2,000-update schedule, got {budget}')
        saves = get(flat, ('optimization', 'save_at_optimizer_steps'))
        if saves is None or 500 not in list(saves):
            problems.append(f'{arm}: must save at optimizer step 500, got {saves}')
        if get(flat, ('meta', 'load_checkpoint')):
            problems.append(f'{arm}: must start from the official checkpoint')
        if get(flat, ('data', 'manifest')) != 'data/frozen/agrifield40k-v1_train.csv':
            problems.append(f'{arm}: must train on agrifield40k-v1')

    for arm in ('G0', 'G1', 'G2', 'G3'):
        flat = flats[arm]
        interrupt = get(flat, ('optimization', 'interrupt_after_optimizer_steps'))
        if interrupt != 500:
            problems.append(f'{arm}: must interrupt after exactly 500 updates, '
                            f'got {interrupt} (this is not a budget; the schedule '
                            f'horizon stays 2,000)')
        if get(flat, ('mask', 'geometry_beta')) != 3.5:
            problems.append(f'{arm}: geometry_beta must be the offline-calibrated 3.5, '
                            f'got {get(flat, ("mask", "geometry_beta"))}')
        if get(flat, ('mask', 'geometry_control')) != 'linear':
            problems.append(f'{arm}: geometry_control must be linear')
        if get(flat, ('mask', 'num_candidates')) != 16:
            problems.append(f'{arm}: num_candidates must be 16')

    # -- G0 reads no structure prior
    if get(flats['G0'], ('mask', 'prior_mode')) is not None:
        problems.append('G0 must set mask.prior_mode: null')
    if get(flats['G0'], ('mask', 'score_mode')) != 'uniform':
        problems.append('G0 must use score_mode: uniform')
    if get(flats['G0'], ('mask', 'guidance_lambda')) != 0.0:
        problems.append('G0 must set guidance_lambda: 0.0')
    # -- G1/G2/G3 keep the Stage 4A priors and strength
    expected_prior = {'G1': 'vegetation', 'G2': 'boundary',
                      'G3': 'shifted_boundary'}
    for arm, prior in expected_prior.items():
        if get(flats[arm], ('mask', 'prior_mode')) != prior:
            problems.append(f'{arm}: expected prior_mode {prior}')
        if get(flats[arm], ('mask', 'guidance_lambda')) != 6.0:
            problems.append(f'{arm}: expected guidance_lambda 6.0')
        if get(flats[arm], ('mask', 'score_mode')) != 'prior':
            problems.append(f'{arm}: expected score_mode prior')
    shift = get(flats['G3'], ('mask', 'roll_shift'))
    if not shift or (int(shift[0]) == 0 and int(shift[1]) == 0):
        problems.append(f'G3: roll_shift must be non-zero, got {shift}')

    # -- the reference B config itself must still be the standard 2,000 run
    if get(flats['B'], ('mask', 'mode')) != 'random':
        problems.append('reference arm B must use the official random mask')
    if get(flats['B'], ('optimization', 'interrupt_after_optimizer_steps')) is not None:
        problems.append('reference arm B must not be an interrupted run')

    manifest = Path(get(flats['B'], ('data', 'manifest')))
    if manifest.is_file():
        actual = sha256_file(manifest)
        recorded = get(flats['B'], ('data', 'manifest_sha256'))
        if actual != recorded:
            problems.append(f'manifest {manifest} hashes to {actual}, '
                            f'but the config records {recorded}')

    report = {
        'reference': str(paths['B']),
        'configs': {arm: str(path) for arm, path in paths.items()},
        'allowed_different': sorted('.'.join(k) for k in ALLOWED_DIFFERENT),
        'must_match': sorted('.'.join(k) for k in MUST_MATCH),
        'problems': problems,
        'ok': not problems,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    if problems:
        print('STAGE 4B PROTOCOL MISMATCH')
        for line in problems:
            print('  ' + line)
        return 1
    print('STAGE 4B PROTOCOL OK')
    print(f'  reference       : {paths["B"]}')
    print(f'  schedule        : 2,000 updates, interrupted at 500')
    print(f'  manifest        : {manifest}')
    print(f'  initialisation  : {get(flats["B"], ("meta", "pretrained_checkpoint"))}')
    for arm, flat in flats.items():
        print(f'  {arm:>2}: mask={get(flat, ("mask", "mode"))} '
              f'prior={get(flat, ("mask", "prior_mode"))} '
              f'lambda={get(flat, ("mask", "guidance_lambda"))} '
              f'beta={get(flat, ("mask", "geometry_beta"))} '
              f'folder={get(flat, ("logging", "folder"))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
