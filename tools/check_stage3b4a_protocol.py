#!/usr/bin/env python
"""Assert the Stage 3B / Stage 4A training configs differ only in the mask.

A mask-only comparison is only interpretable if every other knob is identical.
This tool diffs the four configs field by field and fails on any difference
outside the documented arm-identity and mask-strategy keys.  It also checks the
invariants that the protocol itself requires (same manifest, same budget, same
initialisation, independent training).

Usage:
    ./.venv/bin/python -m tools.check_stage3b4a_protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ARMS = ('B', 'C1', 'C2', 'C3')
DEFAULT_CONFIGS = {
    'B': 'configs/stage3b_random_2000.yaml',
    'C1': 'configs/stage4a_vegetation_2000.yaml',
    'C2': 'configs/stage4a_boundary_2000.yaml',
    'C3': 'configs/stage4a_shuffled_2000.yaml',
}

#: Keys allowed to differ between the four arms.
ALLOWED_DIFFERENT = {
    ('meta', 'arm'),
    ('meta', 'arm_label'),
    ('mask', 'mode'),
    ('mask', 'prior_mode'),
    ('mask', 'guidance_lambda'),
    ('mask', 'num_candidates'),
    ('mask', 'base_seed'),
    ('mask', 'boundary_edge_weight'),
    ('logging', 'folder'),
}

#: Keys that must be identical everywhere and are checked explicitly.
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
    ('meta', 'pretrained_checkpoint'),
    ('meta', 'pretrained_sha256'),
    ('meta', 'model_name'),
    ('meta', 'patch_size'),
    ('meta', 'pred_depth'),
    ('meta', 'pred_emb_dim'),
    ('meta', 'use_bfloat16'),
    ('meta', 'use_sdpa'),
    ('meta', 'activation_checkpointing'),
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


def get(flat: Dict[Tuple[str, ...], object], key: Tuple[str, ...], default=None):
    return flat.get(key, default)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', default=None,
                        help='arm=path (repeatable); defaults to the four configs')
    args = parser.parse_args()

    paths: Dict[str, Path] = {k: Path(v) for k, v in DEFAULT_CONFIGS.items()}
    if args.config:
        paths = {}
        for item in args.config:
            arm, _, path = item.partition('=')
            paths[arm] = Path(path)

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

    # -- protocol invariants
    ref = flats[ARMS[0]]
    budget = get(ref, ('optimization', 'max_optimizer_steps'))
    if budget != 2000:
        problems.append(f'expected a 2000-update budget, got {budget}')
    if get(ref, ('optimization', 'save_at_optimizer_steps')) != [500]:
        problems.append('expected save_at_optimizer_steps == [500]')
    if get(ref, ('data', 'manifest')) != 'data/frozen/agrifield40k-v1_train.csv':
        problems.append('every arm must train on agrifield40k-v1')
    if get(ref, ('meta', 'load_checkpoint')):
        problems.append('every arm must start from the official checkpoint '
                        '(load_checkpoint must be false)')
    for arm, flat in flats.items():
        if get(flat, ('mask', 'mode')) == 'structure_guided':
            lam = get(flat, ('mask', 'guidance_lambda'))
            k = get(flat, ('mask', 'num_candidates'))
            if not lam or lam <= 0:
                problems.append(f'{arm}: guidance_lambda must be > 0')
            if not k or k < 2:
                problems.append(f'{arm}: num_candidates must be >= 2')
        if get(flat, ('mask', 'mode')) == 'random':
            for key in ('prior_mode', 'guidance_lambda'):
                if get(flat, ('mask', key)) is not None:
                    problems.append(f'{arm}: random arm must not set mask.{key}')

    # -- the manifest on disk must still hash to the recorded value
    manifest = Path(get(ref, ('data', 'manifest')))
    if manifest.is_file():
        actual = sha256_file(manifest)
        recorded = get(ref, ('data', 'manifest_sha256'))
        if actual != recorded:
            problems.append(f'manifest {manifest} hashes to {actual}, '
                            f'but the config records {recorded}')

    report = {
        'configs': {arm: str(path) for arm, path in paths.items()},
        'allowed_different': sorted('.'.join(k) for k in ALLOWED_DIFFERENT),
        'must_match': sorted('.'.join(k) for k in MUST_MATCH),
        'problems': problems,
        'ok': not problems,
    }
    Path('reports/stage3b4a_protocol_check.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if problems:
        print('STAGE 3B / 4A PROTOCOL MISMATCH')
        for line in problems:
            print('  ' + line)
        return 1
    print('STAGE 3B / 4A PROTOCOL OK')
    print(f"  arms            : {', '.join(paths)}")
    print(f"  budget          : {budget} optimizer updates, "
          f"save points {get(ref, ('optimization', 'save_at_optimizer_steps'))}")
    print(f"  manifest        : {manifest}")
    print(f"  initialisation  : {get(ref, ('meta', 'pretrained_checkpoint'))}")
    for arm, flat in flats.items():
        mode = get(flat, ('mask', 'mode'))
        prior = get(flat, ('mask', 'prior_mode'))
        lam = get(flat, ('mask', 'guidance_lambda'))
        print(f"  {arm:>2}: mask={mode} prior={prior} lambda={lam} "
              f"folder={get(flat, ('logging', 'folder'))}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
