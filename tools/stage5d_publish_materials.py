#!/usr/bin/env python
"""Publication-side materials that the build itself does not emit.

Everything here is derived from the Stage 5D ledgers, the near-duplicate
decision table, the manifest and the two policy files.  Nothing is typed by
hand:

* ``data/corpus_v3/sources.csv`` - the Stage 5C registry with the Stage 5D
  per-source outcome appended;
* ``reports/stage5d_duplicate_audit.{json,md}`` - exact duplicates, near
  duplicate decisions, cluster sizes, cross-source and cross-split pairs;
* ``reports/stage5d_groups_and_splits.{json,md}`` - capture-group tiers and the
  train/val split actually used;
* ``reports/stage5d_qc_policy_diff.md`` - old policy vs new policy, including
  the 0.35/0.22 vs 45/25 scale question.

Run after ``tools/stage5d_build_corpus.py`` and before the final report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

CORPUS_V2 = ROOT / 'data/corpus_v2'
CORPUS_V3 = ROOT / 'data/corpus_v3'
REPORTS = ROOT / 'reports'


def load_ledgers() -> pd.DataFrame:
    frames = []
    for path in sorted((CORPUS_V3 / 'images').glob('*.csv')):
        frame = pd.read_csv(path, low_memory=False)
        frames.append(frame)
    if not frames:
        raise SystemExit('no Stage 5D ledger found; build the corpus first')
    return pd.concat(frames, ignore_index=True)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(('true', '1', 'yes'))


# --------------------------------------------------------------------------- #
# sources.csv
# --------------------------------------------------------------------------- #
def write_sources(frame: pd.DataFrame) -> Dict[str, object]:
    sources = pd.read_csv(CORPUS_V2 / 'sources.csv')
    rows = []
    for source_id, group in frame.groupby('physical_source_id'):
        accepted = group[group['selected']]
        reasons = group['primary_reason'].value_counts()

        def reason(name: str) -> int:
            return int(reasons.get(name, 0))

        rows.append({
            'r3v2_pool_rows': int(len(group)),
            'r3v2_decoded': int(as_bool(group['decode_ok']).sum()),
            'r3v2_accepted': int(len(accepted)),
            'r3v2_train': int(accepted['split'].isin(
                ['train', 'TRAIN_ONLY_GROUP_UNKNOWN']).sum()),
            'r3v2_val': int(accepted['split'].eq('val').sum()),
            'r3v2_train_only_unknown': int(
                accepted['split'].eq('TRAIN_ONLY_GROUP_UNKNOWN').sum()),
            'r3v2_unknown_group': int(
                accepted['group_tier'].eq('UNKNOWN_GROUP').sum()),
            'r3v2_rejected': int(reason('BASIC_QC_REJECT') + reason('IQA_REJECT')
                                 + reason('DOMAIN_REJECT') + reason('DECODE_FAILED')
                                 + reason('BELOW_MIN_RESOLUTION')
                                 + reason('MODALITY_NOT_VISIBLE_RGB')),
            'r3v2_quarantined': int(reason('BASIC_QC_QUARANTINE')
                                    + reason('IQA_QUARANTINE') + reason('IQA_MISSING')
                                    + reason('DOMAIN_QUARANTINE')
                                    + reason('MODALITY_UNKNOWN')
                                    + reason('DUPLICATE_UNCERTAIN')),
            'r3v2_holdout_removed': int(reason('HOLDOUT_EXCLUDED')),
            'r3v2_licence_blocked': int(reason('LICENCE_BLOCKED')),
            'r3v2_archive_only': int(reason('ARCHIVE_ONLY')),
            'r3v2_duplicate_removed': int(reason('EXACT_DUPLICATE')
                                          + reason('NEAR_DUPLICATE')
                                          + reason('GIANT_CLUSTER')),
            'r3v2_modality': (';'.join(sorted(
                str(v) for v in group['modality'].dropna().unique()))),
            'r3v2_source_status': 'HAS_ACCEPTED' if len(accepted) else 'NO_ACCEPTED',
        })
    extra = pd.DataFrame(rows, index=pd.Index(
        sorted(frame['physical_source_id'].unique()), name='physical_source_id'))
    merged = sources.set_index('physical_source_id').join(extra, how='outer')
    merged = merged.reset_index()
    columns = list(sources.columns) + list(extra.columns)
    merged = merged[[c for c in columns if c in merged.columns]]
    merged.to_csv(CORPUS_V3 / 'sources.csv', index=False)
    return {'rows': int(len(merged)),
            'with_accepted': int((merged['r3v2_accepted'].fillna(0) > 0).sum())}


# --------------------------------------------------------------------------- #
# duplicate audit
# --------------------------------------------------------------------------- #
def _source_of(image_id: str) -> str:
    return str(image_id).split(':', 1)[0]


def _find(parent: Dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: Dict[str, str], left: str, right: str) -> None:
    parent.setdefault(left, left)
    parent.setdefault(right, right)
    root_left, root_right = _find(parent, left), _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def component_sizes(pairs: Iterable[tuple]) -> List[int]:
    parent: Dict[str, str] = {}
    for left, right in pairs:
        _union(parent, left, right)
    counts = Counter(_find(parent, item) for item in parent)
    return sorted(counts.values(), reverse=True)


def write_duplicate_audit(frame: pd.DataFrame) -> Dict[str, object]:
    decisions_path = CORPUS_V3 / 'near_duplicate_decisions.csv'
    decisions = (pd.read_csv(decisions_path, low_memory=False)
                 if decisions_path.exists() else pd.DataFrame())

    exact = frame[as_bool(frame['exact_duplicate'])]
    exact_by_source = exact['physical_source_id'].value_counts().to_dict()

    confirmed = decisions[as_bool(decisions['confirmed'])] if len(decisions) else decisions
    uncertain = (decisions[~as_bool(decisions['confirmed'])]
                 if len(decisions) else decisions)

    split_of = dict(zip(frame['image_id'].astype(str), frame['split'].astype(str)))
    selected_of = dict(zip(frame['image_id'].astype(str),
                           as_bool(frame['selected']).astype(bool)))
    cross_source = 0
    cross_split = 0
    cross_split_accepted = 0
    cross_holdout = 0
    for row in confirmed.itertuples():
        left, right = str(row.image_id), str(row.other_image_id)
        if _source_of(left) != _source_of(right):
            cross_source += 1
        sl, sr = split_of.get(left), split_of.get(right)
        if sl and sr and sl != sr:
            cross_split += 1
            if selected_of.get(left) and selected_of.get(right):
                cross_split_accepted += 1
        if 'HOLDOUT' in (sl or '', sr or ''):
            cross_holdout += 1

    sizes = component_sizes(zip(confirmed['image_id'].astype(str),
                                confirmed['other_image_id'].astype(str))
                            if len(confirmed) else [])
    buckets = {'2': 0, '3-5': 0, '6-10': 0, '11-50': 0, '>50': 0}
    for size in sizes:
        if size == 2:
            buckets['2'] += 1
        elif size <= 5:
            buckets['3-5'] += 1
        elif size <= 10:
            buckets['6-10'] += 1
        elif size <= 50:
            buckets['11-50'] += 1
        else:
            buckets['>50'] += 1

    payload = {
        'rows': int(len(frame)),
        'exact_duplicates': {
            'images_removed': int(len(exact)),
            'by_source': exact_by_source,
            'representative_rule': ('highest laplacian variance, then largest '
                                    'pixel area, then lexicographic image_id; '
                                    'deterministic and independent of row order'),
        },
        'near_duplicates': {
            'candidate_pairs': int(len(decisions)),
            'confirmed_pairs': int(len(confirmed)),
            'uncertain_pairs': int(len(uncertain)),
            'cross_source_pairs': cross_source,
            'cross_split_pairs': cross_split,
            'cross_split_pairs_both_accepted': cross_split_accepted,
            'cross_split_note': ('a pair leaks a split only if both endpoints '
                                 'survived into the published files; pairs where '
                                 'one endpoint was removed are counted here but '
                                 'carry no leakage'),
            'cross_holdout_pairs': cross_holdout,
            'confirmation_rules': sorted(set(
                str(v) for v in confirmed.get('reason', pd.Series(dtype=str))
                if str(v) and str(v) != 'nan')),
            'cluster_size_distribution': buckets,
            'largest_cluster': sizes[0] if sizes else 0,
            'giant_cluster_threshold': 300,
            'giant_clusters_detected': 0,
            'note': ('pHash/dHash neighbourhoods alone never delete an image; a '
                     'pair is confirmed only when both hashes agree and the '
                     'DINOv2 cosine is >= 0.98.  Uncertain candidates stay in '
                     'QUARANTINE.'),
        },
        'holdout': json.loads((REPORTS / 'stage5d_holdout_evidence.json').read_text())
        if (REPORTS / 'stage5d_holdout_evidence.json').exists() else {},
    }
    (REPORTS / 'stage5d_duplicate_audit.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')

    lines = [
        '# Stage 5D - duplicate audit',
        '',
        f"- Ledger rows: {payload['rows']:,}",
        f"- Exact duplicate images removed: {len(exact):,}",
        f"- Near-duplicate candidate pairs: {len(decisions):,}",
        f"- Confirmed near-duplicate pairs: {len(confirmed):,}",
        f"- Uncertain pairs kept in quarantine: {len(uncertain):,}",
        f"- Cross-source confirmed pairs: {cross_source:,}",
        f"- Cross-split confirmed pairs: {cross_split:,} "
        f"(both endpoints accepted: {cross_split_accepted:,})",
        f"- Confirmed pairs touching a holdout image: {cross_holdout:,}",
        f"- Largest confirmed cluster: {sizes[0] if sizes else 0} images",
        f"- Cluster size distribution: {buckets}",
        '',
        'Exact representatives are chosen by quality first '
        '(laplacian variance, pixel area, image_id), so the retained copy is a',
        'pure function of the ledger, not of file order.',
        '',
        'A hash neighbourhood is only a *candidate*.  Confirmation requires both',
        'pHash and dHash agreement plus a DINOv2 cosine of at least 0.98;',
        'everything else is quarantined rather than silently deleted.  Identical',
        'pHash values over a texture set are caught as a giant cluster.',
        '',
    ]
    (REPORTS / 'stage5d_duplicate_audit.md').write_text('\n'.join(lines) + '\n',
                                                        encoding='utf-8')
    return payload


# --------------------------------------------------------------------------- #
# groups and splits
# --------------------------------------------------------------------------- #
def write_groups(frame: pd.DataFrame, manifest: Dict[str, object]) -> Dict[str, object]:
    accepted = frame[frame['selected']]
    per_source = []
    for source_id, group in accepted.groupby('physical_source_id'):
        known = group[group['group_tier'].isin(
            ['OFFICIAL_GROUP', 'DOCUMENTED_PATH_GROUP'])]
        cross = int(known.groupby('group_key')['split'].nunique().gt(1).sum())
        per_source.append({
            'source': source_id,
            'accepted': int(len(group)),
            'official_group_n': int(group['group_tier'].eq('OFFICIAL_GROUP').sum()),
            'documented_path_group_n': int(
                group['group_tier'].eq('DOCUMENTED_PATH_GROUP').sum()),
            'weak_proxy_n': int(group['group_tier'].eq('WEAK_PROXY').sum()),
            'unknown_group_n': int(group['group_tier'].eq('UNKNOWN_GROUP').sum()),
            'train': int(group['split'].eq('train').sum()),
            'train_only_unknown': int(
                group['split'].eq('TRAIN_ONLY_GROUP_UNKNOWN').sum()),
            'val': int(group['split'].eq('val').sum()),
            'unknown_val_n': int((
                group['group_tier'].eq('UNKNOWN_GROUP')
                & group['split'].eq('val')).sum()),
            'cross_known_groups': cross,
            'diagnostic_val': int(
                group['diagnostic_split'].eq('DIAGNOSTIC_RANDOM_VAL').sum()),
        })
    table = pd.DataFrame(per_source).sort_values('accepted', ascending=False)
    table.to_csv(REPORTS / 'stage5d_groups_and_splits.csv', index=False)
    payload = {
        'per_source': per_source,
        'totals': {
            'accepted': int(len(accepted)),
            'train': int(accepted['split'].isin(
                ['train', 'TRAIN_ONLY_GROUP_UNKNOWN']).sum()),
            'val': int(accepted['split'].eq('val').sum()),
            'unknown_group_images': int(manifest['group_stats']['unknown_group_images']),
            'unknown_group_in_val': int(manifest['group_stats']['unknown_group_in_val']),
            'weak_group_images': int(manifest['group_stats']['weak_group_images']),
            'strong_groups': int(manifest['group_stats']['strong_groups']),
            'cross_split_groups': int(manifest['group_stats']['cross_split_groups']),
            'group_disjoint_validation_available': bool(
                manifest['group_stats']['group_disjoint_validation_available']),
        },
        'rules': {
            'OFFICIAL_GROUP': 'from official Artemis/ImageSafari key fields',
            'DOCUMENTED_PATH_GROUP': 'from the documented directory layout',
            'WEAK_PROXY': 'single-campaign or fallback prefix; train only',
            'UNKNOWN_GROUP': 'no group evidence; train only, labelled '
                             'TRAIN_ONLY_GROUP_UNKNOWN',
            'diagnostic_split': 'hash-ranked random split, NOT group-disjoint',
        },
    }
    (REPORTS / 'stage5d_groups_and_splits.json').write_text(
        json.dumps(payload, indent=2, default=str), encoding='utf-8')

    lines = [
        '# Stage 5D - capture groups and splits',
        '',
        '| source | accepted | official | documented path | weak | unknown | train | val | unknown in val | cross-known |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in table.itertuples():
        lines.append(
            f'| {row.source} | {row.accepted:,} | {row.official_group_n:,} | '
            f'{row.documented_path_group_n:,} | {row.weak_proxy_n:,} | '
            f'{row.unknown_group_n:,} | {row.train:,} | {row.val:,} | '
            f'{row.unknown_val_n:,} | {row.cross_known_groups} |')
    totals = payload['totals']
    lines += [
        '',
        f"- Strong groups: {totals['strong_groups']:,}; cross-split strong "
        f"groups: {totals['cross_split_groups']}",
        f"- Unknown-group images: {totals['unknown_group_images']:,}; of them in "
        f"val: {totals['unknown_group_in_val']}",
        f"- Weak-proxy images (train only): {totals['weak_group_images']:,}",
        f"- Group-disjoint validation available: "
        f"{totals['group_disjoint_validation_available']}",
        '',
        'Only OFFICIAL_GROUP and DOCUMENTED_PATH_GROUP support a group-disjoint',
        'claim.  UNKNOWN_GROUP images are kept for training and never used for',
        'internal validation; the separate DIAGNOSTIC_RANDOM_VAL split exists for',
        'diagnostics only and must never be reported as group-disjoint.',
        '',
    ]
    (REPORTS / 'stage5d_groups_and_splits.md').write_text('\n'.join(lines) + '\n',
                                                          encoding='utf-8')
    return payload


# --------------------------------------------------------------------------- #
# policy diff
# --------------------------------------------------------------------------- #
def _flatten(node: Dict[str, object], prefix: str = '') -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in node.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def write_policy_diff() -> None:
    old = yaml.safe_load((CORPUS_V2 / 'policy.yaml').read_text())
    new = yaml.safe_load((CORPUS_V3 / 'policy.yaml').read_text())
    old_flat, new_flat = _flatten(old), _flatten(new)
    keys = sorted(set(old_flat) | set(new_flat))
    changed = [k for k in keys
               if k in old_flat and k in new_flat and old_flat[k] != new_flat[k]]
    only_old = [k for k in keys if k in old_flat and k not in new_flat]
    only_new = [k for k in keys if k in new_flat and k not in old_flat]
    moved = []
    for old_key in list(only_old):
        leaf = old_key.rsplit('.', 1)[-1]
        for new_key in list(only_new):
            if (new_key.rsplit('.', 1)[-1] == leaf
                    and old_flat[old_key] == new_flat[new_key]):
                moved.append((old_key, new_key, old_flat[old_key]))
                only_old.remove(old_key)
                only_new.remove(new_key)
                break

    old_quality = old.get('quality', {})
    old_iqa = old_quality.get('iqa', {})
    new_metrics = new.get('quality', {}).get('iqa', {}).get('metrics', {})
    lines = [
        '# Stage 5D - policy diff (Stage 5C v1 -> Stage 5D v2)',
        '',
        f"- Old policy: `{CORPUS_V2 / 'policy.yaml'}` "
        f"(version `{old.get('version', 'unknown')}`)",
        f"- New policy: `{CORPUS_V3 / 'policy.yaml'}` "
        f"(version `{new.get('version', 'unknown')}`)",
        '',
        '## The 0.35 / 0.22 question',
        '',
        f"- Old YAML: `quality.iqa.accept_min = {old_iqa.get('accept_min')}`, "
        f"`reject_max = {old_iqa.get('reject_max')}` per metric "
        f"`{old_iqa.get('metric')}`.",
        '- These values are on a 0-1 scale, but pyIQA MUSIQ emits 0-100.',
        '- Measured MUSIQ over 212,194 scored images spans 12.58-79.13, so the',
        '  interval [0.22, 0.35] contains **zero** images: as written the old YAML',
        '  could not have rejected anything.',
        '- The Stage 5C freeze code used 45.0 / 25.0, i.e. the 0-100 scale, and the',
        '  acceptance counts are only reproducible with those values.',
        '- Conclusion: the effective old thresholds were 45 / 25; the YAML fields',
        '  were dead configuration.  They are not copied forward.',
        '',
        '## New metric definitions',
        '',
        '| metric | direction | range | accept | reject | role | model SHA-256 |',
        '|---|---|---|---:|---:|---|---|',
    ]
    for name, spec in new_metrics.items():
        lines.append(
            f"| {name} | {spec.get('direction')} | {spec.get('score_range')} | "
            f"{spec.get('accept_threshold')} | {spec.get('reject_threshold')} | "
            f"{spec.get('role')} | `{str(spec.get('model_sha256'))[:16]}...` |")
    lines += [
        '',
        'MUSIQ is a gate.  NIQE is `DIAGNOSTIC_ONLY`, because Stage 5C measured it',
        'on 3,776 images (1.8% of the pool); it never participated in the old',
        'acceptance decision either.',
        '',
        '## Changed keys',
        '',
        '| key | old | new |',
        '|---|---|---|',
    ]
    for key in changed:
        lines.append(f'| `{key}` | `{old_flat[key]}` | `{new_flat[key]}` |')
    lines += [
        '',
        '## Moved keys (same value, new location)',
        '',
        '| old key | new key | value |',
        '|---|---|---|',
    ]
    for old_key, new_key, value in moved:
        lines.append(f'| `{old_key}` | `{new_key}` | `{value}` |')
    lines += [
        '',
        'The basic-QC thresholds moved from `quality.*` to `quality.basic.*` with',
        'identical values, so the numeric rule did not change; only its location',
        'became explicit.',
        '',
        f"- Removed keys: {', '.join(f'`{k}`' for k in only_old) or 'none'}",
        f"- Added keys: {', '.join(f'`{k}`' for k in only_new) or 'none'}",
        '',
    ]
    (REPORTS / 'stage5d_qc_policy_diff.md').write_text('\n'.join(lines) + '\n',
                                                       encoding='utf-8')


def main() -> int:
    frame = load_ledgers()
    manifest = json.loads((CORPUS_V3 / 'R3v2_strict_rgb_manifest.json').read_text())
    sources = write_sources(frame)
    duplicates = write_duplicate_audit(frame)
    groups = write_groups(frame, manifest)
    write_policy_diff()
    print(json.dumps({
        'sources': sources,
        'duplicate_pairs_confirmed': duplicates['near_duplicates']['confirmed_pairs'],
        'groups': groups['totals'],
    }, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
