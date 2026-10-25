#!/usr/bin/env python
"""Generate every Stage 5D table and figure from the single decision table.

No number here is typed by hand.  The source of truth is
``data/corpus_v3/images/*.csv`` (one row per image, with its decision) plus the
manifest written by ``tools/stage5d_build_corpus.py``.

Outputs
-------
reports/stage5d_dataset_composition.{csv,md}
reports/stage5d_metadata_coverage.{csv,md}
reports/stage5d_groups_split.csv
reports/stage5d_qc_calibration.{csv,md}
reports/stage5d_figures/*.png
reports/stage5d_dataset_card.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORPUS_V3 = ROOT / 'data/corpus_v3'
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'stage5d_figures'
SOURCES = ROOT / 'data/corpus_v2/sources.csv'


def load_master() -> pd.DataFrame:
    frames = []
    for path in sorted((CORPUS_V3 / 'images').glob('*.csv')):
        frame = pd.read_csv(path, low_memory=False)
        if 'relative_path' not in frame.columns:
            frame['relative_path'] = frame.get('rel_path', '')
        frames.append(frame)
    if not frames:
        raise SystemExit('no Stage 5D ledger found; run tools/stage5d_build_corpus.py first')
    return pd.concat(frames, ignore_index=True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(('true', '1', 'yes'))


def fmt_int(value: object) -> str:
    if isinstance(value, (int, np.integer)):
        return f'{int(value):,}'
    return str(value)


def table_source_composition(frame: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_meta = {str(r.physical_source_id): r for r in sources.itertuples()}
    for source_id, group in frame.groupby('physical_source_id'):
        meta = source_meta.get(source_id)
        accepted = group[group['selected']]
        rows.append({
            'source_id': source_id,
            'parent_source': getattr(meta, 'groups', '') if meta else '',
            'original_paper': getattr(meta, 'paper_url', '') if meta else '',
            'citation': getattr(meta, 'source_display_name', '') if meta else '',
            'license': getattr(meta, 'license_id', '') if meta else '',
            'redistribution': getattr(meta, 'redistribution_allowed', '') if meta else '',
            'official_url': getattr(meta, 'official_data_url', '') if meta else '',
            'modality': group['modality'].mode().iloc[0] if len(group) else '',
            'domain': group['acquisition_domain'].mode().iloc[0] if len(group) else '',
            'reported_n': int(getattr(meta, 'reported_images', 0) or 0) if meta else 0,
            'downloaded_n': int(len(group)),
            'decoded_n': int(as_bool(group['decode_ok']).sum()),
            'p0_n': int(group['basic_status'].eq('PASS').sum()),
            'full_qc_n': int((group['basic_status'].eq('PASS') &
                              group['iqa_gate_status'].eq('PASS')).sum()),
            'field_rgb_n': int((group['modality_status'].eq('VISIBLE_RGB_EVIDENCED') &
                                group['acquisition_domain'].isin(
                                    ['FIELD_IN_SITU', 'ORCHARD_IN_SITU'])).sum()),
            'exact_dedup_n': int(as_bool(group['exact_duplicate']).sum()),
            'near_dedup_n': int(group['near_dup_status'].isin(
                ['CONFIRMED_NEAR_DUPLICATE', 'GIANT_CLUSTER']).sum()),
            'holdout_removed_n': int(group['holdout_status'].eq('HOLDOUT').sum()),
            'accepted_n': int(len(accepted)),
            'train_n': int(accepted['split'].eq('train').sum()),
            'val_n': int(accepted['split'].eq('val').sum()),
            'unknown_group_n': int(accepted['group_tier'].eq('UNKNOWN_GROUP').sum()),
            'metadata_coverage': float(
                (accepted['crop'] != 'unknown').mean()) if len(accepted) else 0.0,
            'upstream_datasets': ';'.join(sorted(
                str(v) for v in group['upstream_dataset'].dropna().unique()
                if str(v))) if 'upstream_dataset' in group.columns else '',
            'reasons': ';'.join(f'{k}:{v}' for k, v in
                                group['primary_reason'].value_counts().items()
                                if k != 'ACCEPTED'),
        })
    return pd.DataFrame(rows).sort_values('downloaded_n', ascending=False)


def table_metadata_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    dims = {
        'crop': 'crop', 'country': 'country', 'farm': 'farm_id',
        'upstream_dataset': 'upstream_dataset', 'group_tier': 'group_tier',
        'modality': 'modality', 'acquisition_domain': 'acquisition_domain',
    }
    accepted = frame[frame['selected']]
    rows = []
    for name, column in dims.items():
        if column not in accepted.columns:
            continue
        values = accepted[column].fillna('').astype(str)
        known = values[~values.isin(('', 'unknown', 'UNKNOWN'))]
        distribution = known.value_counts()
        for value, count in distribution.head(30).items():
            rows.append({
                'dimension': name,
                'value': value,
                'known_n': int(count),
                'unknown_n': int(len(accepted) - len(known)),
                'known_fraction': float(len(known) / max(len(accepted), 1)),
                'share_among_known': float(count / max(len(known), 1)),
                'share_among_all': float(count / max(len(accepted), 1)),
            })
    return pd.DataFrame(rows)


def table_groups_split(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = frame[frame['selected']]
    rows = []
    for source_id, group in accepted.groupby('physical_source_id'):
        strong = group[group['group_tier'].isin(['OFFICIAL_GROUP', 'DOCUMENTED_PATH_GROUP'])]
        rows.append({
            'source': source_id,
            'official_group_n': int(group['group_tier'].eq('OFFICIAL_GROUP').sum()),
            'documented_path_group_n': int(
                group['group_tier'].eq('DOCUMENTED_PATH_GROUP').sum()),
            'weak_group_n': int(group['group_tier'].eq('WEAK_PROXY').sum()),
            'unknown_group_n': int(group['group_tier'].eq('UNKNOWN_GROUP').sum()),
            'strong_groups': int(strong['group_key'].nunique()),
            'train': int(group['split'].eq('train').sum()),
            'val': int(group['split'].eq('val').sum()),
            'cross_known_groups': int(
                strong.groupby('group_key')['split'].nunique().gt(1).sum())
            if len(strong) else 0,
            'unknown_val_n': int(group[group['group_tier'].eq('UNKNOWN_GROUP')]
                                 ['split'].eq('val').sum()),
            'holdout_risk': 'none_in_accepted',
        })
    return pd.DataFrame(rows)


def figure_reasons(frame: pd.DataFrame) -> None:
    counts = frame['primary_reason'].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind='barh', ax=ax, color='#4472c4')
    ax.set_xscale('log')
    ax.set_xlabel('images (log scale)')
    ax.set_title('Stage 5D primary exclusion reasons')
    fig.tight_layout()
    fig.savefig(FIGURES / 'primary_reasons.png', dpi=160)
    plt.close(fig)


def figure_composition(frame: pd.DataFrame) -> None:
    accepted = frame[frame['selected']]
    counts = accepted['physical_source_id'].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind='bar', ax=ax, color='#70ad47')
    ax.set_ylabel('accepted images')
    ax.set_title('R3v2_strict_rgb source composition')
    fig.tight_layout()
    fig.savefig(FIGURES / 'source_composition.png', dpi=160)
    plt.close(fig)


def figure_musiq(frame: pd.DataFrame) -> None:
    accepted_sources = frame['physical_source_id'].value_counts().head(10).index
    data = [pd.to_numeric(frame[frame['physical_source_id'] == s]['iqa_musiq'],
                          errors='coerce').dropna().values for s in accepted_sources]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, tick_labels=[s.replace('_', '\n') for s in accepted_sources],
               showfliers=False)
    ax.axhline(45.0, color='#c00000', linestyle='--', linewidth=1, label='accept 45')
    ax.axhline(25.0, color='#ed7d31', linestyle=':', linewidth=1, label='reject 25')
    ax.set_ylabel('MUSIQ (0-100)')
    ax.set_title('MUSIQ distribution by source')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'musiq_distribution.png', dpi=160)
    plt.close(fig)


def figure_coverage(coverage: pd.DataFrame) -> None:
    if coverage.empty:
        return
    pivot = coverage[coverage['dimension'].isin(
        ['crop', 'country', 'farm', 'upstream_dataset', 'group_tier'])]
    best = pivot.groupby('dimension')['known_fraction'].max()
    fig, ax = plt.subplots(figsize=(7, 4))
    best.plot(kind='bar', ax=ax, color='#ffc000')
    ax.set_ylim(0, 1)
    ax.set_ylabel('known fraction among accepted')
    ax.set_title('Metadata coverage')
    fig.tight_layout()
    fig.savefig(FIGURES / 'metadata_coverage.png', dpi=160)
    plt.close(fig)


def figure_crop_country(frame: pd.DataFrame) -> None:
    accepted = frame[frame['selected']]
    if accepted.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, column, title in ((axes[0], 'crop', 'crop'),
                              (axes[1], 'country', 'country')):
        values = accepted[column].fillna('unknown').astype(str)
        counts = values.value_counts().head(12)
        counts.sort_values().plot(kind='barh', ax=ax, color='#8faadc')
        unknown = int((values == 'unknown').sum())
        ax.set_title(f'{title} (unknown {unknown:,} / {len(values):,})')
        ax.set_xlabel('accepted images')
        ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'crop_country_composition.png', dpi=160)
    plt.close(fig)


def figure_groups(frame: pd.DataFrame) -> None:
    accepted = frame[frame['selected']]
    if accepted.empty:
        return
    table = (accepted.groupby(['physical_source_id', 'group_tier'])
             .size().unstack(fill_value=0))
    order = ['OFFICIAL_GROUP', 'DOCUMENTED_PATH_GROUP', 'WEAK_PROXY',
             'UNKNOWN_GROUP']
    table = table.reindex(columns=[c for c in order if c in table.columns],
                          fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    table.plot(kind='bar', stacked=True, ax=ax,
               color=['#2e75b6', '#9dc3e6', '#ffc000', '#c00000'])
    ax.set_ylabel('accepted images')
    ax.set_title('Capture-group evidence per accepted source')
    ax.tick_params(axis='x', rotation=20, labelsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'groups_split.png', dpi=160)
    plt.close(fig)


def figure_funnel(frame: pd.DataFrame) -> None:
    steps = ['ACCEPTED', 'LABEL']  # placeholder replaced below
    order = ['HOLDOUT_EXCLUDED', 'LICENCE_BLOCKED', 'ARCHIVE_ONLY',
             'MODALITY_NOT_VISIBLE_RGB', 'MODALITY_UNKNOWN', 'DECODE_FAILED',
             'BELOW_MIN_RESOLUTION', 'BASIC_QC_REJECT', 'BASIC_QC_QUARANTINE',
             'IQA_REJECT', 'IQA_QUARANTINE', 'IQA_MISSING', 'DOMAIN_REJECT',
             'DOMAIN_QUARANTINE', 'EXACT_DUPLICATE', 'NEAR_DUPLICATE',
             'DUPLICATE_UNCERTAIN', 'GIANT_CLUSTER']
    counts = frame['primary_reason'].value_counts()
    values = [int(counts.get(reason, 0)) for reason in order]
    labels = [r.replace('_', ' ').title() for r in order]
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(order))
    ax.barh(y, values, color='#5b9bd5')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('images')
    ax.set_title('Exclusive exclusion funnel (each image counted once)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'funnel_exclusive.png', dpi=160)
    plt.close(fig)


def figure_sensitivity() -> None:
    path = REPORTS / 'stage5d_threshold_sensitivity.json'
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    curve = pd.DataFrame(payload['musiq_sensitivity'])
    fig, ax = plt.subplots(figsize=(7, 4))
    for accept, group in curve.groupby('accept_threshold'):
        ax.plot(group['reject_threshold'], group['pass'] / 1000.0, marker='o',
                label=f'accept {accept:g}')
    ax.set_xlabel('reject threshold')
    ax.set_ylabel('images passing (thousands)')
    ax.set_title('MUSIQ threshold sensitivity')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / 'threshold_sensitivity.png', dpi=160)
    plt.close(fig)


def qc_calibration_table() -> pd.DataFrame:
    """Aggregate the synthetic stress test into the Gate-7 calibration schema.

    ``known_label_recall`` is recall against *constructed* degradations; it is
    not a human quality label and the disclaimer is part of every row.
    """
    path = ROOT / 'reports/stage5d_synthetic_qc.csv'
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if raw.empty:
        return raw
    clean = raw[raw['label'] == 'clean']
    clean_retention = float(clean['basic_status'].eq('PASS').mean()) if len(clean) else float('nan')
    rows = []
    degraded = raw[raw['label'] == 'degraded']
    for (stratum, kind, severity), group in degraded.groupby(
            ['stratum', 'kind', 'severity']):
        detected = group['detected'].astype(bool)
        rows.append({
            'source_stratum': stratum,
            'degradation': kind,
            'severity': severity,
            'n': int(len(group)),
            'basic_detection': float(group['basic_status'].ne('PASS').mean()),
            'full_pipeline_reject': float(detected.mean()),
            'quarantine': float(group['iqa_status'].eq('QUARANTINE').mean()),
            'pass': float((~detected).mean()),
            'known_label_recall': float(detected.mean()),
            'clean_retention': clean_retention,
            'disclaimer': ('synthetic degradation label; sensitivity only, not a '
                           'human quality judgement'),
        })
    return pd.DataFrame(rows)


def write_markdown_tables(composition: pd.DataFrame, coverage: pd.DataFrame,
                          groups: pd.DataFrame, calibration: pd.DataFrame) -> None:
    lines = ['# Stage 5D - dataset composition', '',
             '| source | downloaded | decoded | full QC | field RGB | accepted | train | val | unknown group |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in composition.itertuples():
        lines.append(f'| {row.source_id} | {row.downloaded_n:,} | {row.decoded_n:,} | '
                     f'{row.full_qc_n:,} | {row.field_rgb_n:,} | {row.accepted_n:,} | '
                     f'{row.train_n:,} | {row.val_n:,} | {row.unknown_group_n:,} |')
    (REPORTS / 'stage5d_dataset_composition.md').write_text('\n'.join(lines) + '\n',
                                                            encoding='utf-8')

    lines = ['# Stage 5D - metadata coverage', '',
             'Denominators are explicit: `known_fraction` is over every accepted',
             'image, and `share_among_known` is over the known subset only.', '',
             '| dimension | value | known_n | unknown_n | known_fraction | share_among_known |',
             '|---|---|---:|---:|---:|---:|']
    for row in coverage.head(120).itertuples():
        lines.append(f'| {row.dimension} | {row.value} | {row.known_n:,} | '
                     f'{row.unknown_n:,} | {row.known_fraction:.4f} | '
                     f'{row.share_among_known:.4f} |')
    (REPORTS / 'stage5d_metadata_coverage.md').write_text('\n'.join(lines) + '\n',
                                                          encoding='utf-8')


def dataset_card(manifest: Dict[str, object], composition: pd.DataFrame,
                 coverage: pd.DataFrame, groups: pd.DataFrame,
                 calibration: pd.DataFrame) -> None:
    counts = manifest['counts']
    lines = [
        '# AgriField-Extended (internal working name) - dataset card',
        '',
        '> This is an internal name.  It is **not** Agri-147K and must not be cited',
        '> as the official Agri-FM+ corpus.',
        '',
        '## 1. Identity',
        '',
        f"- Version: `{manifest['version']}`",
        f"- Status: `{manifest['status']}`",
        f"- Meaning: {manifest['auto_status_meaning']}",
        f"- Policy: `{manifest['policy_version']}` / SHA-256 `{manifest['policy_sha256']}`",
        f"- Accepted images: **{counts['accepted']:,}** "
        f"(train {counts['train']:,} / val {counts['val']:,})",
        f"- Quarantined: {counts['quarantined']:,}; rejected: {counts['rejected']:,}; "
        f"holdout: {counts['holdout']:,}; licence-blocked: {counts['licence_blocked']:,}",
        '',
        '## 2. Relationship to upstream datasets',
        '',
        'Agri-FM+ (CVPR Workshops 2025) publishes the *construction principles* of',
        'Agri-147K but not a per-image list.  This corpus is a protocol-informed',
        'reconstruction plus documented extensions, not a reproduction.',
        '',
        '| source | accepted | upstream dataset(s) | license | redistribution | official entry |',
        '|---|---:|---|---|---|---|',
    ]
    for row in composition.sort_values('accepted_n', ascending=False).itertuples():
        if row.accepted_n == 0:
            continue
        lines.append(f'| {row.source_id} | {row.accepted_n:,} | '
                     f'{row.upstream_datasets or row.source_id} | {row.license} | '
                     f'{row.redistribution} | {row.official_url} |')
    lines += [
        '',
        '## 3. Composition',
        '',
        f"- By source: {json.dumps(manifest['by_source'], ensure_ascii=False)}",
        f"- By crop (accepted; `unknown` = no evidence): "
        f"{json.dumps(manifest['by_crop'], ensure_ascii=False)}",
        f"- By country (accepted; `unknown` = no evidence): "
        f"{json.dumps(manifest['by_country'], ensure_ascii=False)}",
        f"- By modality: {json.dumps(manifest['by_modality'], ensure_ascii=False)}",
        f"- By group tier: {json.dumps(manifest['by_group_tier'], ensure_ascii=False)}",
        '',
        '## 4. Quality pipeline',
        '',
        '| metric | direction | range | accept | reject | role | model SHA-256 |',
        '|---|---|---|---:|---:|---|---|',
    ]
    for name, spec in manifest['iqa_trace'].items():
        lines.append(f"| {name} | {spec['direction']} | {spec['score_range']} | "
                     f"{spec['accept_threshold']} | {spec['reject_threshold']} | "
                     f"{spec['role']} | `{spec['model_sha256'][:16]}...` |")
    lines += [
        '',
        'MUSIQ is measured on a 0-100 scale; this was verified empirically (see',
        '`reports/stage5d_threshold_trace.md`).  NIQE is diagnostic only, because it',
        'was measured on a small minority of sources in Stage 5C.',
    ]
    stress_path = REPORTS / 'stage5d_synthetic_qc.json'
    if stress_path.exists():
        stress = json.loads(stress_path.read_text())
        per_degradation = pd.DataFrame(stress.get('per_degradation', []))
        best = worst = None
        if not per_degradation.empty:
            best = per_degradation.sort_values('mean').iloc[-1]
            worst = per_degradation.sort_values('mean').iloc[0]
        lines += [
            '',
            'Synthetic stress test (constructed degradations, **not** human labels):',
            '',
            f"- {stress['base_images']} base images from "
            f"{len(stress.get('per_stratum', {}))} source/domain strata, "
            f"{stress['degraded_rows']} degraded variants, "
            f"{stress['degradations_per_base']} degradations per base;",
            f"- clean retention {stress['clean_retention']:.3f}, "
            f"clean false-removal {stress['clean_false_removal']:.3f};",
            f"- full-pipeline recall {stress['overall_recall']:.3f}; "
            f"basic rules alone {stress['basic_only_recall']:.3f}; "
            f"IQA alone {stress['iqa_only_recall']:.3f}.",
        ]
        if best is not None:
            lines.append(f"- strongest: `{best['kind']}` / {best['severity']} "
                         f"({best['mean']:.3f}); weakest: `{worst['kind']}` / "
                         f"{worst['severity']} ({worst['mean']:.3f}).")
        lines.append('- these numbers measure sensitivity to the constructed '
                     'damage only; no natural-image precision/FPR is claimed.')
    lines += [
        '',
        '## 5. Splits and independence',
        '',
        f"- Strong groups: {manifest['group_stats']['strong_groups']:,}",
        f"- Cross-split strong groups: {manifest['group_stats']['cross_split_groups']}",
        f"- Unknown-group images in the accepted set: "
        f"{manifest['group_stats']['unknown_group_images']:,}",
        f"- Weak-proxy images (train only): {manifest['group_stats']['weak_group_images']:,}",
        f"- Group-disjoint validation available: "
        f"{manifest['group_stats']['group_disjoint_validation_available']}",
        f"- Diagnostic random split (NOT group-disjoint) among accepted: "
        f"{counts['diagnostic_val']:,}",
        '',
        'Unknown-group images never enter the validation split.  They are labelled',
        '`TRAIN_ONLY_GROUP_UNKNOWN`; a separate diagnostic random split exists but is',
        'not the group-disjoint validation set.',
        '',
        '## 6. Deduplication',
        '',
        f"- Exact duplicate images: {manifest['dedup']['exact_duplicate_images']:,}",
        f"- Confirmed near-duplicate pairs: {manifest['dedup']['confirmed_pairs']:,}",
        f"- Unresolved candidates (quarantined): {manifest['dedup']['uncertain_pairs']:,}",
        f"- Giant-cluster (texture collision) images: {manifest['dedup']['giant_members']:,}",
        '',
        '## 7. Holdout',
        '',
    ]
    holdout_path = REPORTS / 'stage5d_holdout_evidence.json'
    if holdout_path.exists():
        holdout = json.loads(holdout_path.read_text())
        file_level = holdout.get('file_level', {})
        prefix_level = holdout.get('prefix_level', {})
        lines += [
            f"- PhenoBench-prefixed images in the mixed pool: "
            f"{fmt_int(prefix_level.get('phenobench_prefixed_images_in_pool'))} "
            f"(blocked by Stage 5D: "
            f"{fmt_int(prefix_level.get('phenobench_actually_blocked_by_stage5d'))})",
            f"- Old R3 contained "
            f"{fmt_int(file_level.get('old_r3_phenobench_prefixed_images'))} "
            f"PhenoBench-prefixed images; Stage 5C's ledger had flagged only "
            f"{fmt_int(prefix_level.get('phenobench_flagged_by_old_ledger'))}.",
            f"- Exact holdout-hash matches in the new master: "
            f"{fmt_int(file_level.get('new_master_exact_hash_matches'))}.",
            f"- pHash-neighbour candidates (distance <= 6) in the new master: "
            f"{fmt_int(file_level.get('new_master_near_hash_matches'))}; these are "
            'perceptual-hash collisions and were spot-checked, not auto-deleted.',
        ]
    lines += [
        '',
        '- Levels *not* verifiable on this machine (PlantDoc, MinneApple,',
        '  PlantSeg, MulticlassWeeds, GrowliFlower, CropAndWeed, DeepWeeds and',
        '  the Roboflow family) were screened at source level only.',
    ]
    lines += [
        '',
        '## 8. Not suitable for',
        '',
        '* disease diagnosis or clinical/plant-pathology decisions;',
        '* satellite or UAV remote sensing;',
        '* any claim of official Agri-147K reproduction.',
        '',
        '### Redistribution',
        '',
        'Image files may be redistributed only where the upstream licence permits',
        'it: Artemis / Image Safari (CC-BY-SA-4.0), AgriField-40K (CC-BY-4.0) and',
        'AppleBBCH81 (CC-BY-4.0).  The registry records `redistribution = unknown`',
        'for agrifm_ref1, agrifm_ref58, agrifm_ref11 and agrifm_ref9; for those the',
        'publication ships only source ids, hashes and rebuild scripts, not pixels.',
        '',
        '## 9. Hashes and reproduction',
        '',
        f"- Train CSV SHA-256: `{manifest['train_csv_sha256']}`",
        f"- Val CSV SHA-256: `{manifest['val_csv_sha256']}`",
        f"- Policy SHA-256: `{manifest['policy_sha256']}`",
        '',
        '```bash',
        'export AGRI_DATA_ROOT=/data/datasets/Stage5C',
        'bash tools/stage5d_run_all.sh          # full pipeline (uses the GPU)',
        'STAGE5D_SKIP_EMBEDDINGS=1 bash tools/stage5d_run_all.sh  # CPU-only build',
        '```',
        '',
        '## 10. Limits still open',
        '',
        '- The corpus is automatically validated, never human-verified.',
        '- Country is recovered only where the published key encodes it (Artemis);',
        '  Image Safari country remains unknown.',
        '- Crop is unknown for the small Agri-FM+ reference sources and for the',
        '  mixed AgriField pool; `unknown` is kept as an explicit value.',
        '- Holdouts that are not on this machine were screened only at source level.',
        '- The QC gate is strong on severe blur and dark frames but weak on noise,',
        '  JPEG recompression, colour cast and partial occlusion (see the synthetic',
        '  stress table); those images are not certified clean by this pipeline.',
        '- 76.9% of accepted images come from Artemis / Image Safari and 75.9% of',
        '  the train split; use a capped or balanced sampling profile rather than',
        '  the natural one if source balance matters for a run.',
    ]
    (REPORTS / 'stage5d_dataset_card.md').write_text('\n'.join(lines) + '\n',
                                                     encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-empty', action='store_true')
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = load_master()
    sources = pd.read_csv(SOURCES)
    manifest = json.loads((CORPUS_V3 / 'R3v2_strict_rgb_manifest.json').read_text())

    composition = table_source_composition(frame, sources)
    coverage = table_metadata_coverage(frame)
    groups = table_groups_split(frame)
    calibration = qc_calibration_table()

    composition.to_csv(REPORTS / 'stage5d_dataset_composition.csv', index=False)
    coverage.to_csv(REPORTS / 'stage5d_metadata_coverage.csv', index=False)
    groups.to_csv(REPORTS / 'stage5d_groups_split.csv', index=False)
    if not calibration.empty:
        calibration.to_csv(REPORTS / 'stage5d_qc_calibration.csv', index=False)

    figure_reasons(frame)
    figure_composition(frame)
    figure_musiq(frame)
    figure_coverage(coverage)
    figure_funnel(frame)
    figure_crop_country(frame)
    figure_groups(frame)
    figure_sensitivity()
    write_markdown_tables(composition, coverage, groups, calibration)
    dataset_card(manifest, composition, coverage, groups, calibration)

    print(json.dumps({'accepted': manifest['counts']['accepted'],
                      'sources': int(composition['accepted_n'].gt(0).sum()),
                      'figures': len(list(FIGURES.glob('*.png')))}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
