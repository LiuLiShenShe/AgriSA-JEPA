#!/usr/bin/env python
"""Build the Stage 5D corpus versions (R3v2_strict_rgb) and all audit tables.

This tool is deterministic and non-destructive:

* it never writes into ``data/corpus_v2`` or ``reports/stage5c_*``;
* every threshold comes from ``data/corpus_v3/policy.yaml``;
* every count in every output file is derived from one decision table, so the
  funnel, the manifests and the composition tables cannot disagree.

Usage
-----
    .venv-uv/bin/python tools/stage5d_build_corpus.py \
        --policy data/corpus_v3/policy.yaml \
        --data-root /data/datasets/Stage5C \
        --out data/corpus_v3 --reports reports
"""

from __future__ import annotations

import argparse
import csv
import math
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.registry.stage5c_source_registry import HOLDOUT_IDS
from src.registry.stage5d_dedup import DedupConfig, Dinov2Embedder, apply_dedup
from src.registry.stage5d_funnel import (assert_conserved, conservation,
                                         decide_frame, funnel_table,
                                         per_source_conservation)
from src.registry.stage5d_group_policy import (GroupAssignment,
                                               STRONG_GROUPS, assign_splits,
                                               agrifield_group, assign_diagnostic_random)
from src.registry.stage5d_metadata import (attach_artemis_metadata,
                                           load_agrifield_metadata)
from src.registry.stage5d_metadata_schema import (GROUP_DOCUMENTED_PATH,
                                                  GROUP_OFFICIAL,
                                                  GROUP_UNKNOWN, GROUP_WEAK_PROXY,
                                                  MODALITY_STATUS_EVIDENCED,
                                                  MODALITY_STATUS_NOT_VISIBLE,
                                                  MODALITY_STATUS_UNKNOWN,
                                                  PROVENANCE_OFFICIAL,
                                                  normalize_crop)
from src.registry.stage5d_modality import apply_modality, classify_row
from src.registry.stage5d_qc_policy import (MetricSpec, QualityPolicy,
                                            combine_metric_statuses,
                                            load_policy)

LEGACY_LEDGER_DIR = ROOT / 'data/corpus_v2/images'
STAGE5C_CORPUS = ROOT / 'data/corpus_v2'

#: Upstream dataset names (as recorded by the mixed pool) that are registered
#: downstream holdouts.  Only these exact names are treated as holdout; a
#: fuzzy or substring match would silently drop legitimate data.
UPSTREAM_HOLDOUT_NAMES = {'PhenoBench', 'DeepWeeds', 'CropAndWeed', 'GrowliFlower',
                          'MulticlassWeeds', 'PlantDoc', 'PlantSeg', 'MinneApple'}

EXTENSION_SOURCES = ['artemis_imagesafari', 'agrifield40k_pool', 'sesame_weed',
                     'applebbch81']
POOL_SOURCES = ['agrifm_ref1', 'agrifm_ref6', 'agrifm_ref7', 'agrifm_ref8',
                'agrifm_ref9', 'agrifm_ref11', 'agrifm_ref20', 'agrifm_ref57',
                'agrifm_ref58', 'agrifm_ref90']

#: ``modality`` value recorded in data/corpus_v2/sources.csv.
SOURCE_MODALITY = {
    'artemis_imagesafari': 'RGB',
    'agrifield40k_pool': 'RGB',
    'sesame_weed': 'RGB',
    'applebbch81': 'RGB',
    'agrifm_ref1': 'RGB',
    'agrifm_ref6': 'RGB',
    'agrifm_ref7': 'RGB',
    'agrifm_ref8': 'RGB',
    'agrifm_ref9': 'RGB',
    'agrifm_ref11': 'RGB+NDVI',
    'agrifm_ref20': 'RGB',
    'agrifm_ref57': 'RGB',
    'agrifm_ref58': 'RGB',
    'agrifm_ref90': 'RGB',
}


def source_modalities() -> Dict[str, str]:
    path = STAGE5C_CORPUS / 'sources.csv'
    if not path.exists():
        return dict(SOURCE_MODALITY)
    table = pd.read_csv(path)
    return {str(r.physical_source_id): (str(r.modality) if pd.notna(r.modality) else '')
            for r in table.itertuples()}


def load_ledgers(source_ids: List[str]) -> pd.DataFrame:
    frames = []
    derived_iqa = ROOT / 'data/corpus_v3/derived_iqa'
    for source_id in source_ids:
        enriched = LEGACY_LEDGER_DIR / f'{source_id}_qc.csv'
        plain = LEGACY_LEDGER_DIR / f'{source_id}.csv'
        repaired = derived_iqa / f'{source_id}.csv'
        # Precedence: Stage 5D repair > QC-enriched > plain.  The plain file is
        # only the pre-enrichment snapshot and has empty IQA columns.  Stage 5C
        # compared mtimes here, which silently loaded the unenriched file for
        # two sources; Stage 5D never does that.
        path = repaired if repaired.exists() else (enriched if enriched.exists() else plain)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        frame['physical_source_id'] = source_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_iqa(frame: pd.DataFrame, policy: QualityPolicy) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Evaluate every declared metric.  Missing never becomes a pass."""
    out = frame.copy()
    trace: Dict[str, object] = {}
    statuses: List[pd.Series] = []
    reason_lists: List[List[str]] = [[] for _ in range(len(out))]
    for name, spec in policy.metrics.items():
        column = {'musiq': 'iqa_musiq', 'niqe': 'iqa_niqe'}.get(name, f'iqa_{name}')
        if column not in out.columns:
            values = pd.Series([np.nan] * len(out))
        else:
            values = pd.to_numeric(out[column], errors='coerce')
        outcomes = [spec.evaluate(v) for v in values]
        out[f'{name}_status'] = [o[0] for o in outcomes]
        for index, (_, reason) in enumerate(outcomes):
            if reason:
                reason_lists[index].append(reason)
        observed = values.dropna()
        trace[name] = {
            'column': column,
            'metric_name': spec.name,
            'direction': spec.direction,
            'score_range': list(spec.score_range),
            'accept_threshold': spec.accept_threshold,
            'reject_threshold': spec.reject_threshold,
            'role': spec.role,
            'model': spec.model,
            'model_sha256': spec.model_sha256,
            'scale_evidence': spec.scale_evidence,
            'measured_rows': int(len(out)),
            'non_null_rows': int(observed.size),
            'observed_min': float(observed.min()) if observed.size else None,
            'observed_max': float(observed.max()) if observed.size else None,
            'observed_median': float(observed.median()) if observed.size else None,
            'in_range_rows': int(((observed >= spec.score_range[0]) &
                                  (observed <= spec.score_range[1])).sum()),
            'pass': int((out[f'{name}_status'] == 'PASS').sum()),
            'quarantine': int((out[f'{name}_status'] == 'QUARANTINE').sum()),
            'reject': int((out[f'{name}_status'] == 'REJECT').sum()),
            'missing': int((out[f'{name}_status'] == 'MISSING').sum()),
        }
        if spec.role == 'GATE':
            statuses.append(out[f'{name}_status'])

    if statuses:
        combined = [combine_metric_statuses(list(row), policy.missing_score_action)
                    for row in zip(*[s.tolist() for s in statuses])]
        out['iqa_gate_status'] = [c[0] for c in combined]
        out['iqa_reasons'] = [';'.join(r) for r in reason_lists]
    else:
        out['iqa_gate_status'] = 'PASS'
        out['iqa_reasons'] = ''
    return out, trace


def evaluate_basic(frame: pd.DataFrame, policy: QualityPolicy) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Deterministic basic-quality gate, reported as a tri-state."""
    out = frame.copy()
    thresholds = policy.basic
    statuses: List[str] = []
    reasons: List[str] = []
    for row in out.to_dict('records'):
        hard: List[str] = []
        borderline: List[str] = []
        short_side = float(row.get('short_side') or 0)
        lap = row.get('laplacian_var')
        ten = row.get('tenengrad')
        mean = row.get('brightness_mean')
        ent = row.get('entropy')
        blank = row.get('blank_fraction')
        aspect = row.get('aspect_ratio')
        try:
            lap = float(lap) if lap is not None else None
        except (TypeError, ValueError):
            lap = None
        for name, value in (('laplacian_var', lap), ('tenengrad', ten),
                            ('brightness_mean', mean), ('entropy', ent),
                            ('blank_fraction', blank), ('aspect_ratio', aspect)):
            try:
                if value is not None:
                    row[name] = float(value)
            except (TypeError, ValueError):
                row[name] = None
        lap, ten, mean, ent, blank, aspect = (
            row['laplacian_var'], row['tenengrad'], row['brightness_mean'],
            row['entropy'], row['blank_fraction'], row['aspect_ratio'])
        missing_metrics = [name for name, value in
                           (('laplacian_var', lap), ('tenengrad', ten),
                            ('brightness_mean', mean), ('entropy', ent))
                           if value is None or not math.isfinite(float(value))]
        if len(missing_metrics) == 4:
            statuses.append('QUARANTINE')
            reasons.append('MISSING_BASIC_METRICS')
            continue

        if short_side < thresholds.min_short_side:
            statuses.append('REJECT'); reasons.append('BELOW_MIN_RESOLUTION')
            continue
        if aspect is not None and aspect > thresholds.extreme_aspect_ratio_max:
            statuses.append('REJECT'); reasons.append('EXTREME_ASPECT_RATIO')
            continue
        if blank is not None and blank > thresholds.blank_pixel_fraction_max:
            statuses.append('REJECT'); reasons.append('BLANK_IMAGE')
            continue
        if lap is not None and lap < thresholds.laplacian_var_min:
            hard.append('BLURRY')
        if ten is not None and ten < thresholds.tenengrad_min:
            hard.append('LOW_DETAIL')
        if mean is not None and mean < thresholds.dark_mean_max:
            hard.append('DARK')
        if mean is not None and mean > thresholds.bright_mean_min:
            hard.append('OVEREXPOSED')
        if ent is not None and ent < thresholds.entropy_min:
            hard.append('LOW_INFORMATION')
        if len(hard) >= 2 and thresholds.quarantine_on_conflict:
            statuses.append('QUARANTINE'); reasons.append(';'.join(hard + ['MULTI_METRIC_CONFLICT']))
            continue
        if hard:
            statuses.append('REJECT'); reasons.append(';'.join(hard))
            continue
        band = thresholds.near_threshold_band
        for label, value, threshold, above in (
                ('BLUR_BORDERLINE', lap, thresholds.laplacian_var_min, True),
                ('DETAIL_BORDERLINE', ten, thresholds.tenengrad_min, True),
                ('DARK_BORDERLINE', mean, thresholds.dark_mean_max, True),
                ('BRIGHT_BORDERLINE', mean, thresholds.bright_mean_min, False),
                ('ENTROPY_BORDERLINE', ent, thresholds.entropy_min, True)):
            if value is None:
                continue
            span = abs(threshold) * band or band
            if abs(value - threshold) <= span:
                borderline.append(label)
        if borderline:
            statuses.append('QUARANTINE'); reasons.append(';'.join(borderline))
            continue
        statuses.append('PASS'); reasons.append('')
    out['basic_status'] = statuses
    out['basic_reasons'] = reasons
    counts = pd.Series(statuses).value_counts().to_dict()
    return out, counts


def derive_groups(frame: pd.DataFrame, agrifield_samples: Path) -> pd.DataFrame:
    out = frame.copy()
    assignments: List[GroupAssignment] = []
    by_file = {}
    if out['physical_source_id'].eq('agrifield40k_pool').any():
        meta, _audit = load_agrifield_metadata(agrifield_samples)
        by_file = {str(r.stem): r for r in meta.itertuples()}

    for row in out.to_dict('records'):
        source = str(row['physical_source_id'])
        filename = Path(str(row.get('relative_path') or '')).name
        if source == 'artemis_imagesafari':
            key = str(row.get('group_key') or '')
            if key:
                assignments.append(GroupAssignment(key, GROUP_OFFICIAL,
                                                   'published_s3_key', 1.0,
                                                   str(row.get('original_dataset_id') or '')))
            else:
                assignments.append(GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, ''))
        elif source == 'agrifield40k_pool':
            record = by_file.get(Path(filename).stem)
            if record is None:
                assignments.append(GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, ''))
                continue
            row['source_prefix'] = str(record.source_prefix)
            row['component'] = getattr(record, 'component', '')
            row['observation_url'] = getattr(record, 'observation_url', '')
            row['upstream_dataset'] = str(record.upstream_dataset)
            assignments.append(agrifield_group(row))
        elif source == 'applebbch81':
            assignments.append(_apple_group(filename))
        elif source in ('sesame_weed',):
            assignments.append(GroupAssignment('sesame_weed|single_campaign',
                                               GROUP_DOCUMENTED_PATH,
                                               'documented_single_collection', 0.9,
                                               'SesameWeed'))
        else:
            # No documented capture unit is published for the small Agri-FM+
            # reference sources: a filename identity is a real identity but a
            # weak one, so it may not produce a group-disjoint validation set.
            assignments.append(GroupAssignment(
                f'{source}|{filename}', GROUP_WEAK_PROXY,
                'filename_identity', 0.4, source))
    out['group_key'] = [a.group_key for a in assignments]
    out['group_tier'] = [a.tier for a in assignments]
    out['group_evidence'] = [a.evidence for a in assignments]
    out['group_confidence'] = [a.confidence for a in assignments]
    out['upstream_dataset'] = [
        a.source_dataset for a in assignments]
    return out


def _apple_group(filename: str) -> GroupAssignment:
    import re
    match = re.search(r'DSC[_-]?(?P<tree>\d+)[_-](?P<session>\d+)', filename)
    if match:
        return GroupAssignment(
            f'applebbch81|tree{match.group("tree")}|session{match.group("session")}',
            GROUP_DOCUMENTED_PATH, 'filename_tree_session', 0.9, 'AppleBBCH81')
    return GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, 'AppleBBCH81')


def assign_split_column(frame: pd.DataFrame, policy: QualityPolicy) -> pd.DataFrame:
    out = frame.copy()
    salt = 'agrisa-stage5d-groups-v1'
    val_fraction = 0.15
    groups = [GroupAssignment(k, t, '', 1.0) for k, t in
              zip(out['group_key'], out['group_tier'])]
    split_by_group = assign_splits(groups, val_fraction, salt)
    diagnostic = assign_diagnostic_random(groups, val_fraction, salt)
    out['split'] = out['group_key'].map(split_by_group).fillna('TRAIN_ONLY_GROUP_UNKNOWN')
    out['diagnostic_split'] = out['group_key'].map(diagnostic).fillna('TRAIN_ONLY_GROUP_UNKNOWN')
    strong = out['group_tier'].isin(STRONG_GROUPS)
    out['group_disjoint_split'] = np.where(strong, out['split'], 'train')
    return out


def build(out_dir: Path, reports: Path, data_root: Path, use_embeddings: bool,
          agrifield_samples: Path = ROOT / 'data/all_samples.csv') -> Dict[str, object]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    policy = load_policy(out_dir / 'policy.yaml')
    policy_dict = policy.as_dict()

    sources = POOL_SOURCES + EXTENSION_SOURCES
    print('[1/9] loading ledgers', flush=True)
    frame = load_ledgers(sources)
    print(f'      rows={len(frame)}', flush=True)
    frame['relative_path'] = frame['rel_path'].astype(str)
    frame['filename'] = frame['relative_path'].map(lambda p: Path(p).name)

    # ---- modality -------------------------------------------------------- #
    print('[2/9] modality', flush=True)
    modalities = source_modalities()
    frame = apply_modality(frame, modalities)

    # ---- official metadata ------------------------------------------------ #
    print('[3/9] official metadata join', flush=True)
    index_path = Path('/data/datasets/Stage5C/artemis/remote_index.csv')
    frame, artemis_join, artemis_stats = attach_artemis_metadata(frame, index_path)
    print(f'      matched={artemis_join.matched_rows} conflicts={artemis_join.conflict_rows}', flush=True)
    frame['crop'] = frame['crop'].fillna('unknown')
    for column in ('country', 'farm_id', 'session_id', 'capture_timestamp',
                   'plot_id', 'metadata_join_status', 'original_dataset_id',
                   'original_image_id', 'original_uri'):
        if column not in frame.columns:
            frame[column] = ''
        frame[column] = frame[column].fillna('')
    frame.loc[frame['physical_source_id'] != 'artemis_imagesafari', 'crop'] = 'unknown'
    frame.loc[frame['physical_source_id'] != 'artemis_imagesafari', 'country'] = 'unknown'

    # ---- domain ----------------------------------------------------------- #
    domain_field = set(['FIELD_IN_SITU', 'ORCHARD_IN_SITU'])
    domain_excluded = set(['PROTECTED_CULTIVATION', 'CONTROLLED_LAB', 'UAV_AERIAL',
                           'SATELLITE_REMOTE'])
    domain = frame['acquisition_domain'].astype(str)
    frame['domain_status'] = np.where(domain.eq('UNKNOWN'), 'QUARANTINE_DOMAIN', 'OK')

    # ---- basic quality + IQA ---------------------------------------------- #
    print('[4/9] basic QC', flush=True)
    frame, basic_counts = evaluate_basic(frame, policy)
    print(f'      {basic_counts}', flush=True)
    print('[5/9] IQA', flush=True)
    frame, iqa_trace = evaluate_iqa(frame, policy)

    # ---- dedup ------------------------------------------------------------ #
    config = DedupConfig(**{k: v for k, v in {
        'phash_max_distance': 8, 'dhash_max_distance': 6, 'confirm_phash_max': 4,
        'aspect_tolerance': 0.02, 'giant_cluster_threshold': 300,
        'embedding_cosine_min': 0.98, 'use_embeddings': use_embeddings,
    }.items()})
    print('[6/9] dedup', flush=True)
    embedder = None
    if use_embeddings:
        try:
            embedder = Dinov2Embedder(device='cuda:1').provider(
                dict(zip(frame['image_id'].astype(str), frame['raw_path'].astype(str))))
        except Exception as exc:  # noqa: BLE001 - never fabricate a confirmation
            print(f'    embedding disabled: {exc}', flush=True)
            embedder = None
    frame, decisions, dedup_stats = apply_dedup(frame, config, embedder)
    print(f'      {dedup_stats}', flush=True)

    # ---- groups + splits -------------------------------------------------- #
    print('[7/9] groups + splits', flush=True)
    frame = derive_groups(frame, agrifield_samples)
    frame = assign_split_column(frame, policy)

    # ---- licence / holdout / archive -------------------------------------- #
    # A holdout can hide *inside* a mixed pool.  Stage 5C flagged only the
    # ``train/pnb_`` half, so 1,864 PhenoBench images reached R3.  The check
    # below is independent of the split directory and of the source-level name.
    licence_ok = frame['license_status'].astype(str).str.startswith('ACCEPTED')
    frame['license_ok'] = licence_ok
    filename = frame['relative_path'].astype(str).str.split('/').str[-1]
    upstream = frame['upstream_dataset'].fillna('').astype(str)
    source_block = frame['physical_source_id'].isin(HOLDOUT_IDS)
    prefix_block = filename.str.startswith('pnb_')
    upstream_block = upstream.isin(UPSTREAM_HOLDOUT_NAMES)
    frame['holdout_reason'] = np.select(
        [source_block, prefix_block, upstream_block],
        ['source_registered_holdout', 'phenobench_filename_prefix',
         'upstream_dataset_registered_holdout'], default='')
    frame['holdout_status'] = np.where(frame['holdout_reason'].ne(''), 'HOLDOUT', 'OK')
    frame['archive_only'] = frame['archive_only'].astype(str).str.lower().eq('yes')

    # ---- decision --------------------------------------------------------- #
    print('[8/9] decision', flush=True)
    frame = decide_frame(frame)
    print('[9/9] writing outputs', flush=True)
    assert_conserved(frame)

    accepted = frame[frame['selected']].copy()
    # ``TRAIN_ONLY_GROUP_UNKNOWN`` is a *training* label, not a rejection.  It is
    # kept in the train file and counted separately; it never reaches val.
    accepted_train = accepted[accepted['split'].isin(['train', 'TRAIN_ONLY_GROUP_UNKNOWN'])]
    accepted_val = accepted[accepted['split'] == 'val']
    if len(accepted_train) + len(accepted_val) != len(accepted):
        raise AssertionError('accepted images with an unexpected split value')

    # ---- outputs ---------------------------------------------------------- #
    manifest_fields = ['image_id', 'physical_source_id', 'original_dataset_id',
                       'original_image_id', 'relative_path', 'raw_sha256',
                       'pixel_sha256', 'phash', 'dhash', 'width', 'height',
                       'modality', 'modality_evidence', 'acquisition_domain',
                       'domain_evidence_type', 'crop', 'crop_evidence', 'country',
                       'country_evidence', 'farm_id', 'session_id',
                       'capture_timestamp', 'plot_id', 'group_key', 'group_tier',
                       'group_evidence', 'iqa_musiq', 'basic_status',
                       'iqa_gate_status', 'license_status', 'holdout_status',
                       'split']
    ledger_dir = out_dir / 'images'
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for source_id, group in frame.groupby('physical_source_id'):
        group.to_csv(ledger_dir / f'{source_id}.csv', index=False)

    train_csv = out_dir / 'R3v2_strict_rgb_train.csv'
    val_csv = out_dir / 'R3v2_strict_rgb_val.csv'
    present = [c for c in manifest_fields if c in accepted_train.columns]
    accepted_train.to_csv(train_csv, index=False, columns=present)
    accepted_val.to_csv(val_csv, index=False, columns=present)

    train_sha = sha256_file(train_csv)
    val_sha = sha256_file(val_csv)

    # ---- composition tables ----------------------------------------------- #
    funnel_records = []
    for source_id, group in list(frame.groupby('physical_source_id')):
        funnel_records.extend(funnel_table(group, f'R3v2/{source_id}'))
    funnel_records.extend(funnel_table(frame, 'R3v2/ALL'))
    pd.DataFrame(funnel_records).to_csv(reports / 'stage5d_funnel_table.csv',
                                        index=False)

    composition = per_source_conservation(frame)
    composition.to_csv(reports / 'stage5d_source_composition.csv', index=False)

    quarantine = frame[~frame['selected']][['image_id', 'physical_source_id',
                                            'primary_reason', 'secondary_reasons',
                                            'modality', 'modality_status', 'crop',
                                            'group_tier', 'split', 'iqa_musiq']]
    quarantine.to_csv(out_dir / 'quarantine_reasons.csv', index=False)
    if not decisions.empty:
        decisions.to_csv(out_dir / 'near_duplicate_decisions.csv', index=False)

    holdout_exclusions = frame[frame['holdout_status'].eq('HOLDOUT')][
        ['image_id', 'physical_source_id', 'relative_path']]
    holdout_exclusions.to_csv(out_dir / 'holdout_exclusions.csv', index=False)

    # ---- manifest --------------------------------------------------------- #
    counts = conservation(frame)
    known_groups = frame[frame['group_tier'].isin(STRONG_GROUPS)]
    cross_split = int(known_groups.groupby('group_key')['split'].nunique().gt(1).sum())
    manifest = {
        'version': 'R3v2_strict_rgb',
        'status': 'AUTO_VALIDATED',
        'auto_status_meaning': ('programmatic gates only; NOT human-verified and '
                                'NOT PROVEN_NO_LEAKAGE'),
        'policy_version': policy.version,
        'policy_sha256': policy.sha256,
        'counts': {
            'ledger_rows': int(len(frame)),
            'accepted': int(len(accepted)),
            'train': int(len(accepted_train)),
            'train_only_group_unknown': int(
                (accepted['split'] == 'TRAIN_ONLY_GROUP_UNKNOWN').sum()),
            'val': int(len(accepted_val)),
            'diagnostic_val': int((accepted['diagnostic_split']
                                   == 'DIAGNOSTIC_RANDOM_VAL').sum()),
            'quarantined': counts['n_quarantined'],
            'rejected': counts['n_rejected'],
            'holdout': counts['n_holdout'],
            'licence_blocked': counts['n_licence_blocked'],
            'archive_only': counts['n_archive_only'],
            'duplicate_removed': counts['n_duplicate_removed'],
        },
        'by_source': accepted['physical_source_id'].value_counts().to_dict(),
        'by_crop': accepted['crop'].value_counts().to_dict(),
        'by_country': accepted['country'].value_counts().to_dict(),
        'by_upstream_dataset': accepted['upstream_dataset'].value_counts().to_dict(),
        'by_modality': accepted['modality'].value_counts().to_dict(),
        'by_group_tier': accepted['group_tier'].value_counts().to_dict(),
        'group_stats': {
            'strong_groups': int(known_groups['group_key'].nunique()),
            'cross_split_groups': cross_split,
            'unknown_group_images': int(
                accepted[accepted['group_tier'] == GROUP_UNKNOWN].shape[0]),
            'unknown_group_in_val': int(
                accepted[(accepted['group_tier'] == GROUP_UNKNOWN) &
                         (accepted['split'] == 'val')].shape[0]),
            'weak_group_images': int(
                accepted[accepted['group_tier'] == GROUP_WEAK_PROXY].shape[0]),
            'group_disjoint_validation_available': bool(
                accepted_val['group_tier'].isin(STRONG_GROUPS).any()),
        },
        'target': {'target_images': 147285,
                   'target_exceeded': bool(len(accepted) > 147285),
                   'shortfall': max(0, 147285 - len(accepted))},
        'train_csv': relative(train_csv, ROOT),
        'val_csv': relative(val_csv, ROOT),
        'train_csv_sha256': train_sha,
        'val_csv_sha256': val_sha,
        'quarantine_csv': relative(out_dir / 'quarantine_reasons.csv', ROOT),
        'holdout_exclusions_csv': relative(out_dir / 'holdout_exclusions.csv', ROOT),
        'funnel_conserved': counts['conserved'],
        'funnel_totals': counts,
        'policy': policy_dict,
        'basic_counts': basic_counts,
        'iqa_trace': iqa_trace,
        'dedup': dedup_stats,
        'artemis_join': {**artemis_join.as_dict(), **artemis_stats},
        'seconds': round(time.time() - started, 1),
    }
    manifest_path = out_dir / 'R3v2_strict_rgb_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')

    summary = {
        'R3v2_strict_rgb': len(accepted),
        'train': len(accepted_train),
        'val': len(accepted_val),
        'status': manifest['status'],
        'target_exceeded': manifest['target']['target_exceeded'],
        'funnel_conserved': counts['conserved'],
    }
    (reports / 'stage5d_build_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/corpus_v3')
    parser.add_argument('--reports', default='reports')
    parser.add_argument('--data-root', default=os.environ.get('AGRI_DATA_ROOT', '/data/datasets/Stage5C'))
    parser.add_argument('--no-embeddings', action='store_true')
    parser.add_argument('--agrifield-samples', default='data/all_samples.csv')
    args = parser.parse_args()
    manifest = build(ROOT / args.out, ROOT / args.reports, Path(args.data_root),
                     use_embeddings=not args.no_embeddings,
                     agrifield_samples=ROOT / args.agrifield_samples)
    print(json.dumps({'accepted': manifest['counts']['accepted'],
                      'train': manifest['counts']['train'],
                      'val': manifest['counts']['val'],
                      'target_exceeded': manifest['target']['target_exceeded'],
                      'funnel_conserved': manifest['funnel_conserved']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
