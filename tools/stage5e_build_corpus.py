#!/usr/bin/env python
"""Stage 5E work package C - build R3v3_license_verified.

Derives a new corpus version from the Stage 5D per-image ledger.  No Stage 5D
quality / modality / domain / dedup / split threshold is re-evaluated or
relaxed; the only new rule is the Stage 5E research-use licence gate.

Outputs (all under data/corpus_v4/, never touching corpus_v2 or corpus_v3)
--------------------------------------------------------------------------
R3v3_license_verified_manifest.json
R3v3_license_verified_train.csv
R3v3_license_verified_val.csv
R3v3_license_verified_quarantine.csv
R3v3_license_verified_exclusions.csv
R3v3_license_verified_sampling_profiles.json
R3v3_license_verified_composition.json

Usage
-----
    .venv-uv/bin/python tools/stage5e_build_corpus.py \
        --ledger data/corpus_v3/images --out data/corpus_v4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.registry import stage5e_licence as lic
from src.registry.stage5e_corpus import (build, conservation, load_ledger,
                                          stage5e_decision)
from src.registry.stage5e_sampling import (build_profiles, entropy, hhi,
                                           reproducibility_check)

EPOCH_LENGTH = 150_000
CAP_SHARE = 0.5
TARGET_IMAGES = 147_285


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


TRAIN_COLUMNS = ['image_id', 'physical_source_id', 'original_dataset_id',
                 'original_image_id', 'relative_path', 'raw_sha256', 'pixel_sha256',
                 'phash', 'dhash', 'width', 'height', 'modality', 'modality_evidence',
                 'acquisition_domain', 'domain_evidence_type', 'crop', 'crop_evidence',
                 'country', 'country_evidence', 'farm_id', 'session_id',
                 'capture_timestamp', 'plot_id', 'group_key', 'group_tier',
                 'group_evidence', 'iqa_musiq', 'basic_status', 'iqa_gate_status',
                 'license_status', 'holdout_status', 'split',
                 'stage5e_research_use', 'stage5e_train_eligible',
                 'stage5e_redistribution_eligible', 'stage5e_primary_reason']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', default='data/corpus_v3/images')
    parser.add_argument('--out', default='data/corpus_v4')
    parser.add_argument('--policy', default='data/corpus_v3/policy.yaml')
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[1/6] loading Stage 5D ledger', flush=True)
    ledger = load_ledger(ROOT / args.ledger)
    print(f'      {len(ledger):,} rows', flush=True)

    print('[2/6] applying Stage 5E licence gate', flush=True)
    decided = stage5e_decision(ledger)
    _, summary = build(decided)

    accepted = decided[decided['stage5e_selected']].copy()
    # ``split`` comes from Stage 5D; the unknown-group train-only rule is inherited.
    train = accepted[accepted['split'].isin(['train', 'TRAIN_ONLY_GROUP_UNKNOWN'])].copy()
    val = accepted[accepted['split'] == 'val'].copy()
    excluded = decided[~decided['stage5e_selected']].copy()

    if len(train) + len(val) != len(accepted):
        raise AssertionError('accepted images with an unexpected split value')

    print('[3/6] writing manifests', flush=True)
    present = [c for c in TRAIN_COLUMNS if c in accepted.columns]
    train_path = out_dir / 'R3v3_license_verified_train.csv'
    val_path = out_dir / 'R3v3_license_verified_val.csv'
    train.to_csv(train_path, index=False, columns=present)
    val.to_csv(val_path, index=False, columns=present)

    quarantine = excluded[excluded['stage5e_primary_reason']
                          .str.contains('QUARANTINE|MISSING', na=False)]
    quarantine_path = out_dir / 'R3v3_license_verified_quarantine.csv'
    quarantine_cols = [c for c in ['image_id', 'physical_source_id', 'relative_path',
                                   'stage5e_primary_reason', 'primary_reason',
                                   'secondary_reasons'] if c in excluded.columns]
    quarantine.to_csv(quarantine_path, index=False, columns=quarantine_cols)

    exclusions_path = out_dir / 'R3v3_license_verified_exclusions.csv'
    exclusion_cols = [c for c in ['image_id', 'physical_source_id', 'relative_path',
                                  'raw_sha256', 'stage5e_primary_reason',
                                  'stage5e_research_use', 'primary_reason',
                                  'secondary_reasons'] if c in excluded.columns]
    excluded.to_csv(exclusions_path, index=False, columns=exclusion_cols)

    print('[4/6] composition + conservation', flush=True)
    by_source = accepted['physical_source_id'].value_counts().to_dict()
    by_upstream = accepted['upstream_dataset'].fillna('').astype(str) \
        .replace('', 'unknown').value_counts().to_dict()
    by_domain = accepted['acquisition_domain'].value_counts().to_dict()
    by_crop = accepted['crop'].fillna('unknown').astype(str).value_counts().to_dict()
    by_country = accepted['country'].fillna('unknown').astype(str).value_counts().to_dict()
    by_modality = accepted['modality'].value_counts().to_dict()
    by_group_tier = accepted['group_tier'].value_counts().to_dict()
    by_research_use = accepted.groupby('physical_source_id')['stage5e_research_use'] \
        .agg(lambda s: s.iloc[0]).to_dict()

    cons = conservation(decided)
    if not cons['conserved']:
        raise AssertionError(f'count conservation failed: {cons}')

    print('[5/6] sampling profiles', flush=True)
    # Sampling applies to the training entity set, so the profiles are built on
    # ``train`` (147,279), not on accepted (which also contains val).  Mixing the
    # two would make every exposure number silently include validation images.
    train_by_source = train['physical_source_id'].value_counts().to_dict()
    train_by_upstream = train['upstream_dataset'].fillna('').astype(str) \
        .replace('', 'unknown').value_counts().to_dict()
    source_profiles = build_profiles(train_by_source, EPOCH_LENGTH, CAP_SHARE)
    upstream_profiles = build_profiles(train_by_upstream, EPOCH_LENGTH, CAP_SHARE)

    # Composition statistics use the *accepted* denominator; the sampling
    # profiles use the *train* denominator.  Reporting one number for both
    # would silently mix 169,326 with 147,279.
    composition_hhi = hhi({source: float(value) for source, value in by_source.items()})
    composition_entropy = entropy({source: float(value) for source, value in by_source.items()})
    composition_upstream_hhi = hhi({source: float(value) for source, value in by_upstream.items()})
    repro = {
        'by_source': reproducibility_check(source_profiles, train, 'physical_source_id'),
        'by_upstream': reproducibility_check(upstream_profiles, train, 'upstream_dataset'),
    }
    if not all(item['entity_count_preserved'] for item in repro.values()):
        raise AssertionError(f'sampling changed the entity set: {repro}')

    sampling_path = out_dir / 'R3v3_license_verified_sampling_profiles.json'
    sampling_path.write_text(json.dumps({
        'note': ('profiles decide exposure frequency only; the entity set, the split and '
                 'the number of independent images are unchanged from Stage 5D'),
        'epoch_length': EPOCH_LENGTH,
        'cap_share': CAP_SHARE,
        'train_total': int(len(train)),
        'accepted_total': int(len(accepted)),
        'by_source': source_profiles,
        'by_upstream_dataset': upstream_profiles,
        'reproducibility': repro,
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[6/6] manifest', flush=True)
    train_sha = sha256_file(train_path)
    val_sha = sha256_file(val_path)
    policy_path = ROOT / args.policy
    policy_sha = sha256_file(policy_path) if policy_path.exists() else ''

    manifest = {
        'version': 'R3v3_license_verified',
        'derived_from': {
            'stage5d_manifest': 'data/corpus_v3/R3v2_strict_rgb_manifest.json',
            'stage5d_ledger_dir': args.ledger,
            'policy': args.policy,
            'policy_sha256': policy_sha,
        },
        'status': 'STAGE5E_AUDITED',
        'status_meaning': ('Stage 5E only re-decided licence research-use. Quality, modality, '
                           'domain, dedup, holdout and split rules are inherited verbatim from '
                           'Stage 5D; nothing was relaxed to reach the target size.'),
        'counts': {
            'ledger_rows': int(len(decided)),
            'accepted': int(len(accepted)),
            'train': int(len(train)),
            'train_only_group_unknown': int((train['split'] == 'TRAIN_ONLY_GROUP_UNKNOWN').sum()),
            'val': int(len(val)),
            'quarantined_or_excluded': int(len(excluded)),
            'licence_excluded_research_use': int(
                (decided['stage5e_primary_reason'] == 'LICENCE_UNRESOLVED_FOR_RESEARCH_USE').sum()),
        },
        'by_source': by_source,
        'by_upstream_dataset': by_upstream,
        'by_domain': by_domain,
        'by_crop': by_crop,
        'by_country': by_country,
        'by_modality': by_modality,
        'by_group_tier': by_group_tier,
        'by_research_use_status': by_research_use,
        'target': {
            'target_images': TARGET_IMAGES,
            'target_exceeded': int(len(accepted)) > TARGET_IMAGES,
            'target_margin': int(len(accepted)) - TARGET_IMAGES,
            'note': ('scale property only; it never overrides licence, QC, holdout or split '
                     'gates'),
        },
        'conservation': cons,
        # Paths are recorded relative to the *output* directory, not to the
        # repository root, so the manifest is identical whether it is built into
        # data/corpus_v4 or into a scratch directory during the rebuild check.
        'train_csv': train_path.name,
        'val_csv': val_path.name,
        'train_csv_sha256': train_sha,
        'val_csv_sha256': val_sha,
        'sampling_profiles': sampling_path.name,
        'concentration': {
            # Composition (denominator = accepted images).
            'composition_denominator': int(len(accepted)),
            'composition_source_hhi': composition_hhi,
            'composition_source_entropy': composition_entropy,
            'composition_upstream_hhi': composition_upstream_hhi,
            'max_source_share': max(by_source.values()) / float(len(accepted)),
            # Sampling (denominator = train images).
            'sampling_denominator': int(len(train)),
            'sampling_source_hhi': source_profiles['entity_hhi'],
            'sampling_source_entropy': source_profiles['entity_entropy'],
            'sampling_upstream_hhi': upstream_profiles['entity_hhi'],
            'sampling_upstream_entropy': upstream_profiles['entity_entropy'],
        },
    }
    manifest_path = out_dir / 'R3v3_license_verified_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding='utf-8')

    composition_path = out_dir / 'R3v3_license_verified_composition.json'
    composition_path.write_text(json.dumps({
        'accepted': int(len(accepted)),
        'train': int(len(train)),
        'val': int(len(val)),
        'by_source': by_source,
        'by_upstream_dataset': by_upstream,
        'by_domain': by_domain,
        'by_crop': by_crop,
        'by_country': by_country,
        'by_modality': by_modality,
        'by_group_tier': by_group_tier,
        'by_research_use_status': by_research_use,
        'unknown_fractions': {
            'crop': by_crop.get('unknown', 0) / float(len(accepted)),
            'country': by_country.get('unknown', 0) / float(len(accepted)),
        },
        'sampling': {
            'natural_hhi': source_profiles['exposure_hhi']['natural'],
            'capped_hhi': source_profiles['exposure_hhi']['capped'],
            'balanced_hhi': source_profiles['exposure_hhi']['balanced'],
            'capped_info': source_profiles['capped_info'],
        },
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps({
        'accepted': int(len(accepted)),
        'train': int(len(train)),
        'val': int(len(val)),
        'by_source': by_source,
        'conservation': cons,
        'target_margin': manifest['target']['target_margin'],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
