#!/usr/bin/env python
"""Verify the Stage 6D configs against the frozen Stage 6 protocol.

This is the Stage 6D analogue of ``tools/check_stage6_protocol.py``.  It checks
three things:

1. every new config differs from its Stage 6 base only in the documented
   seed / run-identity / output-folder fields;
2. every frozen input (manifest, initialisation checkpoint, budget, mask,
   optimizer, EMA, worker count) is unchanged;
3. within each pair, B1 and B2 differ only in sampling policy and arm identity,
   and use the same paired seed.

Usage::

    ./.venv/bin/python -m tools.check_stage6d_protocol
    ./.venv/bin/python -m tools.check_stage6d_protocol --json reports/stage6d_protocol_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import yaml

from tools.stage6d_build_configs import (
    ALLOWED_CHANGES,
    ARMS,
    BASE_CONFIGS,
    HISTORICAL_SEED,
    PAIRS,
    ROOT,
    RUN_ROOT,
    assert_only_allowed_changes,
    build_plan,
    config_path_for,
    flatten,
    run_id_for,
    sha256_file,
)

#: Fields that must equal the Stage 6 frozen values in every new config.
FROZEN: Dict[Tuple[str, ...], object] = {
    ('data', 'manifest'): 'data/stage6/stage6_r3v3_train_view_nvme.csv',
    ('data', 'manifest_sha256'): '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2',
    ('data', 'split'): 'train',
    ('data', 'batch_size'): 16,
    ('data', 'num_workers'): 0,
    ('data', 'image_cache_dir'): None,
    ('data', 'max_samples'): None,
    ('data', 'crop_size'): 224,
    ('data', 'crop_scale'): [0.3, 1.0],
    ('data', 'use_color_distortion'): False,
    ('data', 'use_gaussian_blur'): False,
    ('data', 'use_horizontal_flip'): False,
    ('data', 'color_jitter_strength'): 0.0,
    ('data', 'pin_mem'): True,
    ('data', 'record_source_exposure'): True,
    ('meta', 'pretrained_checkpoint'): '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
    ('meta', 'pretrained_sha256'): '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22',
    ('meta', 'load_checkpoint'): False,
    ('meta', 'read_checkpoint'): None,
    ('meta', 'model_name'): 'vit_huge',
    ('meta', 'patch_size'): 14,
    ('meta', 'pred_depth'): 12,
    ('meta', 'pred_emb_dim'): 384,
    ('meta', 'use_bfloat16'): True,
    ('meta', 'use_sdpa'): True,
    ('meta', 'activation_checkpointing'): False,
    ('optimization', 'max_optimizer_steps'): 2000,
    ('optimization', 'grad_accum'): 8,
    ('optimization', 'warmup_steps'): 100,
    ('optimization', 'lr'): 0.0002,
    ('optimization', 'start_lr'): 1.0e-05,
    ('optimization', 'final_lr'): 1.0e-06,
    ('optimization', 'weight_decay'): 0.04,
    ('optimization', 'final_weight_decay'): 0.4,
    ('optimization', 'ema'): [0.996, 1.0],
    ('optimization', 'ipe_scale'): 1.0,
    ('optimization', 'save_at_optimizer_steps'): [500, 1000, 1500, 2000],
    ('mask', 'mode'): 'random',
    ('mask', 'allow_overlap'): False,
    ('mask', 'aspect_ratio'): [0.75, 1.5],
    ('mask', 'enc_mask_scale'): [0.85, 1.0],
    ('mask', 'pred_mask_scale'): [0.15, 0.2],
    ('mask', 'num_enc_masks'): 1,
    ('mask', 'num_pred_masks'): 4,
    ('mask', 'min_keep'): 10,
    ('mask', 'patch_size'): 14,
}

#: Keys allowed to differ between B1 and B2 inside one pair.
PAIR_ALLOWED_DIFFERENT = {
    ('meta', 'arm'),
    ('meta', 'arm_label'),
    ('meta', 'run_id'),
    ('data', 'sampling', 'mode'),
    ('data', 'sampling', 'profile'),
    ('data', 'sampling', 'profile_sha256'),
    ('data', 'sampling', 'official_profile'),
    ('data', 'sampling', 'cap_unit'),
    ('data', 'sampling', 'epoch_length'),
    ('data', 'sampling', 'cap_share'),
    ('data', 'sampling', 'capped_sources'),
    ('data', 'sampling', 'backfill'),
    ('data', 'sampling', 'seed'),
    ('logging', 'folder'),
}

FROZEN_FILES = {
    'train_csv': ('data/corpus_v4/R3v3_license_verified_train.csv',
                  '9fa34adb36810a5b0d13109761c60892bd9068f27c8f95a48e728146c1a6be08'),
    'val_csv': ('data/corpus_v4/R3v3_license_verified_val.csv',
                'f4670232237c8017ffb213e739446fce05a0661cbe18315097b1dba214c40686'),
    'sampling_profile': ('data/corpus_v4/R3v3_license_verified_sampling_profiles.json',
                         '91fb23ef5366fa163264a48cdf7c6613769816df751a9695ea9caa284833c505'),
}


def deep_get(flat: Mapping[Tuple[str, ...], object], key: Tuple[str, ...],
             default=None):
    return flat.get(key, default)


def check_frozen(arm: str, cfg: Mapping[str, object]) -> List[str]:
    flat = flatten(cfg)
    problems: List[str] = []
    for key, expected in FROZEN.items():
        actual = deep_get(flat, key, '<missing>')
        if actual != expected:
            problems.append(f'{arm}: frozen field {".".join(key)} = {actual!r}, '
                            f'expected {expected!r}')
    return problems


def check_pair(base_b1: Mapping[str, object], base_b2: Mapping[str, object],
               pair_id: str) -> List[str]:
    flat_b1 = flatten(base_b1)
    flat_b2 = flatten(base_b2)
    problems: List[str] = []
    for key in sorted(set(flat_b1) | set(flat_b2)):
        before = flat_b1.get(key, '<missing>')
        after = flat_b2.get(key, '<missing>')
        if before == after:
            continue
        if key not in PAIR_ALLOWED_DIFFERENT:
            problems.append(f'{pair_id}: {".".join(key)} differs B1={before!r} '
                            f'B2={after!r} outside the sampling-only rule')
    return problems


def check_frozen_files(hash_large_pretrained: bool = False) -> Dict[str, object]:
    report: Dict[str, object] = {'files': {}, 'problems': []}
    for name, (rel, expected) in FROZEN_FILES.items():
        path = ROOT / rel
        if not path.is_file():
            report['problems'].append(f'{name}: missing {rel}')
            continue
        actual = sha256_file(path)
        ok = actual == expected
        report['files'][name] = {'path': rel, 'expected_sha256': expected,
                                 'actual_sha256': actual, 'ok': ok}
        if not ok:
            report['problems'].append(f'{name}: SHA {actual} != {expected}')
    train_view = ROOT / 'data/stage6/stage6_r3v3_train_view_nvme.csv'
    expected_view = '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2'
    if train_view.is_file():
        actual = sha256_file(train_view)
        report['files']['train_view'] = {
            'path': 'data/stage6/stage6_r3v3_train_view_nvme.csv',
            'expected_sha256': expected_view, 'actual_sha256': actual,
            'ok': actual == expected_view}
        if actual != expected_view:
            report['problems'].append(f'train_view: SHA {actual} != {expected_view}')
    else:
        report['problems'].append('train_view: missing')
    if hash_large_pretrained:
        ckpt = Path('/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar')
        expected = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'
        actual = sha256_file(ckpt) if ckpt.is_file() else 'missing'
        report['files']['pretrained'] = {
            'path': str(ckpt), 'expected_sha256': expected,
            'actual_sha256': actual, 'ok': actual == expected}
        if actual != expected:
            report['problems'].append(f'pretrained: SHA {actual} != {expected}')
    return report


def run_checks(hash_large_pretrained: bool = False) -> Dict[str, object]:
    problems: List[str] = []
    plan = build_plan()
    problems.extend(plan['problems'])
    if not plan['ok']:
        return {'ok': False, 'problems': problems, 'runs': []}

    runs_out: List[Dict[str, object]] = []
    seen_run_ids: set = set()
    for run in plan['runs']:
        arm = str(run['arm'])
        seed = int(run['seed'])
        cfg_path = config_path_for(arm, seed)
        if not cfg_path.is_file():
            problems.append(f'{arm}/{seed}: missing {cfg_path.relative_to(ROOT)}')
            continue
        with cfg_path.open('r', encoding='utf-8') as handle:
            cfg = yaml.safe_load(handle)
        problems.extend(check_frozen(arm, cfg))
        if cfg['meta'].get('pretraining_seed') != seed:
            problems.append(f'{arm}/{seed}: meta.pretraining_seed != {seed}')
        if cfg['data'].get('seed') != seed:
            problems.append(f'{arm}/{seed}: data.seed != {seed}')
        if cfg['data'].get('augmentation_seed') != seed:
            problems.append(f'{arm}/{seed}: data.augmentation_seed != {seed}')
        if arm == 'B2' and cfg['data']['sampling'].get('seed') != seed:
            problems.append(f'{arm}/{seed}: data.sampling.seed != {seed}')
        expected_folder = f'{RUN_ROOT}/{run_id_for(arm, seed)}'
        if cfg['logging'].get('folder') != expected_folder:
            problems.append(f'{arm}/{seed}: logging.folder {cfg["logging"].get("folder")} '
                            f'!= {expected_folder}')
        run_id = str(cfg['meta'].get('run_id'))
        if run_id != run['run_id']:
            problems.append(f'{arm}/{seed}: meta.run_id {run_id} != {run["run_id"]}')
        if run_id in seen_run_ids:
            problems.append(f'duplicate run_id {run_id}')
        seen_run_ids.add(run_id)
        runs_out.append({
            'arm': arm, 'seed': seed, 'pair_id': run['pair_id'],
            'run_id': run_id, 'config': str(cfg_path.relative_to(ROOT)),
            'config_sha256': sha256_file(cfg_path),
            'output_dir': cfg['logging']['folder'],
            'sampling_mode': (cfg['data'].get('sampling') or {}).get('mode', 'natural'),
        })

    # B1/B2 within each pair: same seed, only sampling/identity differs
    by_pair: Dict[str, Dict[str, Mapping[str, object]]] = {}
    for run in plan['runs']:
        arm = str(run['arm'])
        cfg_path = config_path_for(arm, int(run['seed']))
        if not cfg_path.is_file():
            continue
        with cfg_path.open('r', encoding='utf-8') as handle:
            by_pair.setdefault(str(run['pair_id']), {})[arm] = yaml.safe_load(handle)
    for pair_id, arms in sorted(by_pair.items()):
        if set(arms) != set(ARMS):
            problems.append(f'{pair_id}: expected B1 and B2, got {sorted(arms)}')
            continue
        if arms['B1']['data']['seed'] != arms['B2']['data']['seed']:
            problems.append(f'{pair_id}: B1/B2 data.seed differ')
        problems.extend(check_pair(arms['B1'], arms['B2'], pair_id))

    frozen = check_frozen_files(hash_large_pretrained=hash_large_pretrained)
    problems.extend(frozen['problems'])

    # seeds must be unique across pairs and distinct from historical
    seeds = [int(pair['seed']) for pair in PAIRS]
    if len(seeds) != len(set(seeds)):
        problems.append(f'duplicate pair seeds: {seeds}')
    if HISTORICAL_SEED in seeds:
        problems.append(f'pair seed collides with historical {HISTORICAL_SEED}')

    return {
        'schema_version': 'stage6d_protocol_check/v1',
        'ok': not problems,
        'problems': problems,
        'historical_seed': HISTORICAL_SEED,
        'pairs': [dict(pair) for pair in PAIRS],
        'runs': runs_out,
        'frozen_inputs': frozen,
        'allowed_changes_from_stage6': sorted(
            '.'.join(key) for arm in ARMS for key in ALLOWED_CHANGES[arm]),
        'pair_allowed_different': sorted('.'.join(key) for key in PAIR_ALLOWED_DIFFERENT),
        'hash_large_pretrained': hash_large_pretrained,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path,
                        default=ROOT / 'reports/stage6d_protocol_check.json')
    parser.add_argument('--hash-large-pretrained', action='store_true',
                        help='also re-hash the 10 GB official checkpoint (slow)')
    args = parser.parse_args(argv)

    report = run_checks(hash_large_pretrained=args.hash_large_pretrained)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'ok': report['ok'], 'problems': report['problems'],
                      'runs': len(report['runs'])}, indent=2, ensure_ascii=False))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
