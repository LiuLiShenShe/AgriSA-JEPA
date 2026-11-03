#!/usr/bin/env python
"""Assert the Stage 6 B1/B2 arms differ only in the sampling policy.

A sampling-only comparison is only interpretable if every other knob is
identical.  This tool diffs the two configs field by field and fails on any
difference outside the documented arm-identity, sampling and output-folder
keys.  It also checks the invariants the Stage 6 protocol requires (same
manifest, same budget, same initialisation, independent training, val
excluded, no image cache).

Usage:
    ./.venv-uv/bin/python -m tools.check_stage6_protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('B1', 'B2')
DEFAULT_CONFIGS = {
    'B1': ROOT / 'configs/stage6_b1_natural_2000.yaml',
    'B2': ROOT / 'configs/stage6_b2_capped_no_backfill_2000.yaml',
}

TRAIN_CSV = ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv'
VAL_CSV = ROOT / 'data/corpus_v4/R3v3_license_verified_val.csv'
PROFILE_JSON = ROOT / 'data/corpus_v4/R3v3_license_verified_sampling_profiles.json'

EXPECTED_TRAIN_SHA = '9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08'
EXPECTED_VAL_SHA = 'f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686'
EXPECTED_PRETRAINED_SHA = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'
EXPECTED_BUDGET = 2000
EXPECTED_SAVE_POINTS = [500, 1000, 1500, 2000]

#: Keys allowed to differ between the two arms.
ALLOWED_DIFFERENT = {
    ('meta', 'arm'),
    ('meta', 'arm_label'),
    ('data', 'sampling', 'mode'),
    ('data', 'sampling', 'profile'),
    ('data', 'sampling', 'profile_sha256'),
    ('data', 'sampling', 'official_profile'),
    ('data', 'sampling', 'cap_unit'),
    ('data', 'sampling', 'epoch_length'),
    ('data', 'sampling', 'cap_share'),
    ('data', 'sampling', 'capped_sources'),
    ('data', 'sampling', 'backfill'),
    ('logging', 'folder'),
}

#: Keys that must be identical everywhere and are checked explicitly.
MUST_MATCH = (
    ('data', 'manifest'),
    ('data', 'manifest_sha256'),
    ('data', 'split'),
    ('data', 'batch_size'),
    ('data', 'num_workers'),
    ('data', 'seed'),
    ('data', 'augmentation_seed'),
    ('data', 'crop_size'),
    ('data', 'crop_scale'),
    ('data', 'root_path'),
    ('data', 'image_cache_dir'),
    ('data', 'pin_mem'),
    ('data', 'max_samples'),
    ('data', 'record_source_exposure'),
    ('data', 'use_color_distortion'),
    ('data', 'use_gaussian_blur'),
    ('data', 'use_horizontal_flip'),
    ('data', 'color_jitter_strength'),
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
    ('meta', 'read_checkpoint'),
    ('optimization', 'max_optimizer_steps'),
    ('optimization', 'save_at_optimizer_steps'),
    ('optimization', 'grad_accum'),
    ('optimization', 'lr'),
    ('optimization', 'start_lr'),
    ('optimization', 'final_lr'),
    ('optimization', 'warmup'),
    ('optimization', 'warmup_steps'),
    ('optimization', 'weight_decay'),
    ('optimization', 'final_weight_decay'),
    ('optimization', 'ema'),
    ('optimization', 'epochs'),
    ('optimization', 'ipe_scale'),
    ('optimization', 'stop_after_epoch'),
    ('optimization', 'interrupt_after_optimizer_steps'),
    ('mask', 'mode'),
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
                        help='arm=path (repeatable); defaults to B1/B2')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'reports/stage6_protocol_check.json',
                        help='where to write the JSON report')
    args = parser.parse_args()

    paths: Dict[str, Path] = dict(DEFAULT_CONFIGS)
    if args.config:
        paths = {}
        for item in args.config:
            arm, _, path = item.partition('=')
            paths[arm] = ROOT / path if not Path(path).is_absolute() else Path(path)

    raw = {arm: yaml.safe_load(path.read_text(encoding='utf-8'))
           for arm, path in paths.items()}
    flats = {arm: flatten(config) for arm, config in raw.items()}

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

    ref = flats[ARMS[0]]

    # -- protocol invariants
    budget = get(ref, ('optimization', 'max_optimizer_steps'))
    if budget != EXPECTED_BUDGET:
        problems.append(f'expected a {EXPECTED_BUDGET}-update budget, got {budget}')
    if get(ref, ('optimization', 'save_at_optimizer_steps')) != EXPECTED_SAVE_POINTS:
        problems.append(f'expected save_at_optimizer_steps == {EXPECTED_SAVE_POINTS}, '
                        f'got {get(ref, ("optimization", "save_at_optimizer_steps"))}')
    if get(ref, ('data', 'split')) != 'train':
        problems.append('Stage 6 trains on the R3v3 train split only')
    if get(ref, ('data', 'num_workers')) != 0:
        problems.append('num_workers must be 0 (only num_workers=0 resume is verified)')
    if get(ref, ('data', 'image_cache_dir')):
        problems.append('no Stage 6 image cache exists; image_cache_dir must be null')
    if get(ref, ('meta', 'load_checkpoint')):
        problems.append('each arm must start from the official checkpoint '
                        '(load_checkpoint must be false)')
    if get(ref, ('optimization', 'interrupt_after_optimizer_steps')):
        problems.append('neither arm may declare an interruption point')
    if get(ref, ('optimization', 'grad_accum')) != 8:
        problems.append(f'expected grad_accum == 8, got {get(ref, ("optimization", "grad_accum"))}')
    if get(ref, ('data', 'batch_size')) != 16:
        problems.append(f'expected batch_size == 16, got {get(ref, ("data", "batch_size"))}')

    # -- frozen inputs must still hash to the recorded values
    frozen = {
        'train_csv': (TRAIN_CSV, EXPECTED_TRAIN_SHA),
        'val_csv': (VAL_CSV, EXPECTED_VAL_SHA),
        'pretrained_checkpoint': (Path(get(ref, ('meta', 'pretrained_checkpoint'))),
                                  EXPECTED_PRETRAINED_SHA),
    }
    for label, (path, expected) in frozen.items():
        if not path.is_file():
            problems.append(f'{label} is missing: {path}')
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f'{label} {path} hashes to {actual}, expected {expected}')

    # -- the manifest on disk must still hash to the recorded value
    manifest = Path(get(ref, ('data', 'manifest')))
    if manifest.is_file():
        actual = sha256_file(manifest)
        recorded = get(ref, ('data', 'manifest_sha256'))
        if actual != recorded:
            problems.append(f'manifest {manifest} hashes to {actual}, '
                            f'but the config records {recorded}')
    else:
        problems.append(f'manifest {manifest} is missing; run '
                        f'python -m tools.stage6_build_train_manifest')

    # -- the capped arm must carry the frozen profile digest
    profile_path = PROFILE_JSON
    if profile_path.is_file():
        profile_sha = sha256_file(profile_path)
    else:
        profile_sha = None
        problems.append(f'sampling profile is missing: {profile_path}')
    for arm, config in raw.items():
        sampling = (config.get('data') or {}).get('sampling') or {'mode': 'natural'}
        mode = sampling.get('mode', 'natural')
        if mode not in ('natural', 'profile_capped'):
            problems.append(f'{arm}: unknown sampling mode {mode!r}')
        if mode == 'profile_capped':
            recorded = sampling.get('profile_sha256')
            if recorded != profile_sha:
                problems.append(f'{arm}: profile_sha256 {recorded} != actual {profile_sha}')
            if sampling.get('backfill') is not False:
                problems.append(f'{arm}: backfill must be False (no-backfill policy)')
            if sampling.get('cap_unit') != 'physical_source_id':
                problems.append(f'{arm}: cap_unit must be physical_source_id')
            if sampling.get('official_profile') != 'by_source':
                problems.append(f'{arm}: official_profile must be by_source')
            if tuple(sampling.get('capped_sources', ())) != ('artemis_imagesafari',):
                problems.append(f'{arm}: capped_sources must be ["artemis_imagesafari"]')
            if sampling.get('epoch_length') != 150000 or sampling.get('cap_share') != 0.5:
                problems.append(f'{arm}: expected epoch_length=150000 and cap_share=0.5')

    # -- the frozen profile must itself reproduce its published numbers
    if profile_path.is_file():
        profile = json.loads(profile_path.read_text(encoding='utf-8'))['by_source']
        capped_info = profile['capped_info']
        for key, expected in (('cap_exposure', 75000.0),
                              ('unfulfilled_exposure', 38828.364238815295),
                              ('cap_share', 0.5)):
            if abs(float(capped_info[key]) - expected) > 1e-6:
                problems.append(f'profile capped_info.{key} = {capped_info[key]}, '
                                f'expected {expected}')
        if capped_info['budget_filled'] is not False:
            problems.append('profile capped_info.budget_filled must be False')
        capped_total = sum(profile['exposure']['capped'].values())
        if abs(capped_total - 111171.6357611847) > 1e-6:
            problems.append(f'profile capped exposure totals {capped_total}, '
                            f'expected 111171.6357611847')

    report = {
        'configs': {arm: str(path) for arm, path in paths.items()},
        'allowed_different': sorted('.'.join(k) for k in ALLOWED_DIFFERENT),
        'must_match': sorted('.'.join(k) for k in MUST_MATCH),
        'frozen_inputs': {label: {'path': str(path), 'sha256': expected}
                          for label, (path, expected) in frozen.items()},
        'problems': problems,
        'ok': not problems,
    }
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    if problems:
        print('STAGE 6 PROTOCOL MISMATCH')
        for line in problems:
            print('  ' + line)
        return 1
    print('STAGE 6 PROTOCOL OK')
    print(f'  arms           : {", ".join(paths)}')
    print(f'  budget         : {budget} optimizer updates, save points '
          f'{get(ref, ("optimization", "save_at_optimizer_steps"))}')
    print(f'  manifest       : {manifest}')
    print(f'  initialisation : {get(ref, ("meta", "pretrained_checkpoint"))}')
    for arm, flat in flats.items():
        sampling = ((raw[arm].get('data') or {}).get('sampling') or {'mode': 'natural'})
        print(f'  {arm:>2}: sampling={sampling.get("mode", "natural"):<15} '
              f'folder={get(flat, ("logging", "folder"))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
