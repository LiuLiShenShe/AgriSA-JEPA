#!/usr/bin/env python
"""Build R0 / R1 / R2 / R3 and their manifests.

Usage::

    python tools/stage5c_build_versions.py --sources agrifm_ref1,agrifm_ref6

Reads the per-source CSVs in ``data/corpus_v2/images/`` (preferring the
CleanVision/IQA-enriched ``*_qc.csv`` when present), runs global dedup, writes
the version manifests and the filtering funnel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.registry.stage5c_download import DownloadLedger
from src.registry.stage5c_freeze import (FIELD_DOMAINS, HOLDOUT_IDS,
                                         by_source_composition, classify_rows,
                                         funnel, freeze_gates, manifest_payload)
from src.registry.stage5c_near_duplicate import (EmbeddingConfirmer,
                                                 choose_representatives,
                                                 connected_components, exact_groups,
                                                 near_duplicate_pairs)

IMAGE_DIR = ROOT / 'data/corpus_v2/images'
CORPUS = ROOT / 'data/corpus_v2'
REPORTS = ROOT / 'reports'
POLICY_PATH = CORPUS / 'policy.yaml'
DATA_ROOT = Path('/data/datasets/Stage5C')

EXTENSION_IDS = ['artemis_imagesafari', 'agrifield40k_pool',
                 'sesame_weed', 'applebbch81']


def load_source(source_id: str) -> pd.DataFrame:
    enriched = IMAGE_DIR / f'{source_id}_qc.csv'
    plain = IMAGE_DIR / f'{source_id}.csv'
    if enriched.exists() and plain.exists() and plain.stat().st_mtime > enriched.stat().st_mtime:
        path = plain
    else:
        path = enriched if enriched.exists() else plain
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    frame['physical_source_id'] = source_id
    return frame


def holdout_isolation(frame: pd.DataFrame) -> pd.DataFrame:
    """Source-level and file-level holdout blocking."""
    source_blocked = frame['physical_source_id'].isin(HOLDOUT_IDS)
    file_blocked = frame['physical_source_id'].eq('agrifield40k_pool') & \
        frame['rel_path'].astype(str).str.split('/').str[-1].str.startswith('pnb_')
    frame = frame.copy()
    frame['holdout_b'] = source_blocked | file_blocked
    return frame


def add_dedup_columns(frame: pd.DataFrame, do_embeddings: bool = True) -> pd.DataFrame:
    frame = frame.copy()
    for column, default in (('dup_representative', True),
                            ('near_dup_unconfirmed', False),
                            ('giant_cluster_suspect', False),
                            ('duplicate_cluster_id', ''),
                            ('exact_duplicate_removed', 0),
                            ('near_duplicate_removed', 0)):
        if column not in frame.columns:
            frame[column] = default

    decodable = frame[frame['decode_ok'].astype(str).str.lower().isin(['true', '1'])]
    # ---- exact duplicates -------------------------------------------------- #
    canonical = exact_groups(decodable.to_dict('records'))
    frame['duplicate_cluster_id'] = frame['image_id'].map(
        lambda i: canonical.get(i, ''))
    exact_groups_map: Dict[str, List[str]] = {}
    for image_id, group in canonical.items():
        exact_groups_map.setdefault(group, []).append(image_id)
    reps = set(canonical.values())
    frame['dup_representative'] = frame['image_id'].isin(reps)
    frame['exact_duplicate_removed'] = frame['image_id'].map(
        lambda i: 0 if canonical.get(i, i) == i else 1)
    frame['dup_representative'] = frame['dup_representative'].fillna(False).infer_objects(copy=False)

    # ---- near duplicates --------------------------------------------------- #
    pairs = near_duplicate_pairs(decodable.to_dict('records'))
    confirmed = [p for p in pairs if p.confirmed]
    unconfirmed = [p for p in pairs if not p.confirmed]
    suspicious_ids = {p.left for p in unconfirmed} | {p.right for p in unconfirmed}
    frame['near_dup_unconfirmed'] = frame['image_id'].isin(suspicious_ids)

    components = connected_components(confirmed, confirmed_only=True)
    sizes: Dict[str, int] = {}
    for group in components.values():
        sizes[group] = sizes.get(group, 0) + 1
    giant = {group for group, size in sizes.items() if size > 300}
    frame['giant_cluster_suspect'] = frame['image_id'].isin(
        {i for i, g in components.items() if g in giant})
    frame['near_duplicate_removed'] = frame['image_id'].map(
        lambda i: 1 if components.get(i, i) != i else 0)

    # ---- embedding confirmation for ambiguous pairs ------------------------ #
    if do_embeddings and unconfirmed:
        ids = sorted(suspicious_ids)
        subset = frame[frame['image_id'].isin(ids)]
        if len(subset):
            try:
                confirmer = EmbeddingConfirmer(device='cuda:1')
                vectors = confirmer.embed(subset['raw_path'].tolist())
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = vectors / norms
                id_to_row = {image_id: i for i, image_id in enumerate(subset['image_id'])}
                still_unconfirmed = set()
                for pair in unconfirmed:
                    left = id_to_row.get(pair.left)
                    right = id_to_row.get(pair.right)
                    if left is None or right is None:
                        still_unconfirmed.update({pair.left, pair.right})
                        continue
                    cosine = float(vectors[left] @ vectors[right])
                    if cosine >= 0.97 and pair.phash_distance <= 6:
                        # Confirmed by two independent modalities.
                        still_unconfirmed.discard(pair.left)
                        still_unconfirmed.discard(pair.right)
                    else:
                        still_unconfirmed.update({pair.left, pair.right})
                frame['near_dup_unconfirmed'] = frame['image_id'].isin(still_unconfirmed)
            except Exception as exc:  # noqa: BLE001 - never fabricate acceptance
                frame['embedding_error'] = str(exc)
    return frame


def write_version(frame: pd.DataFrame, version: str, mask,
                  status: str, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    selected = frame[mask].copy()
    payload = manifest_payload(version, selected, POLICY_PATH, status, extra)
    out_csv = CORPUS / f'{version}_manifest.csv'
    fields = ['image_id', 'physical_source_id', 'origin_ref', 'raw_path',
              'raw_sha256', 'pixel_sha256', 'phash', 'dhash', 'width', 'height',
              'acquisition_domain', 'domain_evidence_type', 'domain_confidence',
              'group_key', 'group_known', 'qc_status', 'qc_reason_codes',
              'iqa_musiq', 'iqa_niqe', 'cleanvision_flags', 'license_status',
              'holdout_flag', 'train_split']
    selected.to_csv(out_csv, index=False, columns=[c for c in fields if c in selected.columns])
    payload['manifest_csv'] = str(out_csv)
    payload['manifest_csv_sha256'] = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    payload['gates'] = freeze_gates(selected) if len(selected) else {}
    if status == 'AUTO_FROZEN_CANDIDATE':
        payload['status'] = 'AUTO_FROZEN' if payload['gates'] and all(payload['gates'].values()) else 'GATES_INCOMPLETE'
        payload['auto_status_meaning'] = (
            'programmatic gates satisfied; NOT human-verified'
            if payload['status'] == 'AUTO_FROZEN' else 'programmatic gates incomplete')
    json_path = CORPUS / f'{version}_manifest.json'
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', required=True,
                        help='comma separated source ids or ALL')
    parser.add_argument('--no-embeddings', action='store_true')
    parser.add_argument('--r3', action='store_true', help='also build R3 from extension sources')
    args = parser.parse_args()

    started = time.time()
    if args.sources.strip().upper() == 'ALL':
        # ``X.csv`` and ``X_qc.csv`` collapse to the same source id; loading
        # both would duplicate every row and corrupt the dedup stage.
        sources = sorted({p.stem.replace('_qc', '') for p in IMAGE_DIR.glob('*.csv')
                          if p.stem not in ('cleanvision_issues',)})
    else:
        sources = [s.strip() for s in args.sources.split(',') if s.strip()]
    frames = []
    for source in sources:
        print(f'== load {source}', flush=True)
        frames.append(load_source(source))
    frame = pd.concat(frames, ignore_index=True)
    frame = holdout_isolation(frame)
    # Extension sources are R3-only.  Flagging them here (instead of loading
    # them a second time) keeps one dedup pass over one frame.
    frame['in_pool'] = ~frame['physical_source_id'].isin(EXTENSION_IDS)
    for column, default in (('near_dup_unconfirmed', False),
                            ('dup_representative', True),
                            ('giant_cluster_suspect', False)):
        if column not in frame.columns:
            frame[column] = default
    frame = classify_rows(frame)
    frame = add_dedup_columns(frame, do_embeddings=not args.no_embeddings)
    frame = classify_rows(frame)

    # ---- R0 -------------------------------------------------------------- #
    r0 = write_version(frame, 'R0_agrifm35_source_pool', pd.Series(True, index=frame.index),
                       'AUDIT_ONLY', {'note': 'everything actually on disk; audit layer, never pretraining data'})

    pool = frame[frame['in_pool']]

    # ---- R1 -------------------------------------------------------------- #
    r1_mask = pool['selected_for_R1']
    r1 = write_version(pool, 'R1_agrifm147_rebuild', r1_mask, 'PROTOCOL_INFORMED_RECONSTRUCTION',
                       {'note': 'rules the paper discloses; P1 gates are NOT applied here'})

    # ---- R2 -------------------------------------------------------------- #
    r2_mask = pool['selected_for_R2']
    r2 = write_version(pool, 'R2_field_strict', r2_mask, 'AUTO_FROZEN_CANDIDATE',
                       {'note': 'programmatic gates only; not human-verified'})
    if not all(r2['gates'].values()):
        r2['status'] = 'GATES_INCOMPLETE'
        (CORPUS / 'R2_field_strict_manifest.json').write_text(
            json.dumps(r2, indent=2, default=str), encoding='utf-8')

    # ---- R3 -------------------------------------------------------------- #
    r3_payload: Optional[Dict[str, object]] = None
    if args.r3:
        loaded = set(frame['physical_source_id'])
        missing = [s for s in EXTENSION_IDS if s not in loaded]
        extra_frames = []
        for source in missing:
            path = IMAGE_DIR / f'{source}_qc.csv'
            if not path.exists():
                path = IMAGE_DIR / f'{source}.csv'
            if not path.exists():
                print(f'!! extension ledger missing: {source}', flush=True)
                continue
            extra_frames.append(load_source(source))
        if extra_frames:
            extension = pd.concat(extra_frames, ignore_index=True)
            extension = holdout_isolation(extension)
            extension['in_pool'] = False
            for column, default in (('near_dup_unconfirmed', False),
                                    ('dup_representative', True),
                                    ('giant_cluster_suspect', False)):
                if column not in extension.columns:
                    extension[column] = default
            combined = pd.concat([frame, extension], ignore_index=True)
            combined = add_dedup_columns(combined, do_embeddings=not args.no_embeddings)
            combined = classify_rows(combined)
        else:
            combined = frame
        r3_mask = combined['selected_for_R3']
        if r3_mask.any() or len(EXTENSION_IDS):
            r3_payload = write_version(combined, 'R3_field_extended', r3_mask,
                                       'AUTO_FROZEN_CANDIDATE',
                                       {'note': 'R2 plus verified extensions; programmatic gates only'})
            if r3_payload['counts']['images'] <= 147_285:
                # Programmatic gates can all pass and the corpus can still be
                # too small.  Say so in the manifest instead of implying scale.
                r3_payload['status'] = 'BELOW_SCALE_TARGET'
                r3_payload['scale_target'] = 147_285
                r3_payload['scale_shortfall'] = 147_285 - r3_payload['counts']['images']
                (CORPUS / 'R3_field_extended_manifest.json').write_text(
                    json.dumps(r3_payload, indent=2, default=str), encoding='utf-8')
            frame = combined
        else:
            print('!! no extension ledgers available; R3 not built', flush=True)

    # ---- funnel + composition ------------------------------------------- #
    REPORTS.mkdir(parents=True, exist_ok=True)
    records = []
    pool_frame = frame[frame['in_pool']] if 'in_pool' in frame.columns else frame
    records.extend(funnel(pool_frame, 'R1', strict=False))
    records.extend(funnel(pool_frame, 'R2', strict=True))
    if r3_payload is not None:
        records.extend(funnel(frame, 'R3', strict=True))
    with open(REPORTS / 'stage5c_filtering_funnel.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    composition = by_source_composition(frame)
    composition.to_csv(REPORTS / 'stage5c_dataset_composition.csv', index=False)
    composition.to_json(REPORTS / 'stage5c_dataset_composition.json', orient='records', indent=2)
    try:
        composition.to_parquet(REPORTS / 'stage5c_dataset_composition.parquet', index=False)
    except Exception as exc:  # noqa: BLE001
        print('parquet export skipped:', exc)

    summary = {
        'r0': r0, 'r1': r1, 'r2': r2, 'r3': r3_payload,
        'seconds': round(time.time() - started, 1),
        'sources_loaded': sources,
    }
    (REPORTS / 'stage5c_versions_summary.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({
        'R0_images': r0['counts']['images'],
        'R1_images': r1['counts']['images'],
        'R2_images': r2['counts']['images'],
        'R2_status': r2['status'],
        'R2_gates': r2['gates'],
        'R3_images': (r3_payload or {}).get('counts', {}).get('images'),
        'R3_status': (r3_payload or {}).get('status'),
        'seconds': summary['seconds'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
