#!/usr/bin/env python
"""Build the per-image Stage 5C ledger for one or more sources.

Stage 1 (this tool): decode, hashes, hand-crafted quality metrics, metadata
domain, group/split, licence and holdout flags.
Stage 2 (``stage5c_run_qc.py``): CleanVision + pyIQA enrichment.
Stage 3 (``stage5c_build_versions.py``): global exact/near dedup and R1/R2/R3.

The split keeps memory bounded: one CSV per physical source under
``data/corpus_v2/images/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.stage5c_qc_basic import (IMAGE_SUFFIXES, QualityPolicy,
                                           basic_decision, decode_and_measure,
                                           paper_decision)
from src.registry.stage5c_group_policy import (GROUP_UNKNOWN, assign_split,
                                               derive_group)
from src.registry.stage5c_source_registry import HOLDOUT_IDS, all_rows

DATA_ROOT = Path('/data/datasets/Stage5C')
OUT_DIR = ROOT / 'data/corpus_v2/images'
POLICY_PATH = ROOT / 'data/corpus_v2/policy.yaml'

ROOTS: Dict[str, Path] = {
    'agrifm_ref1': DATA_ROOT / 'raw/agrifm_ref1',
    'perennial_plants': DATA_ROOT / 'raw/perennial_plants',
    'agrifm_ref6': DATA_ROOT / 'raw/agrifm_ref6',
    'agrifm_ref7': DATA_ROOT / 'raw/agrifm_ref7',
    'agrifm_ref8': DATA_ROOT / 'raw/agrifm_ref8',
    'agrifm_ref9': DATA_ROOT / 'raw/agrifm_ref9',
    'agrifm_ref10': DATA_ROOT / 'raw/agrifm_ref10',
    'agrifm_ref11': DATA_ROOT / 'raw/agrifm_ref11',
    'agrifm_ref20': DATA_ROOT / 'raw/agrifm_ref20',
    'agrifm_ref57': DATA_ROOT / 'raw/agrifm_ref57',
    'agrifm_ref58': DATA_ROOT / 'raw/agrifm_ref58',
    'agrifm_ref90': DATA_ROOT / 'raw/agrifm_ref90',
    'plantvillage': Path('/data/datasets/PlantVillage/extracted'),
    'gwhd_2021': Path('/data/datasets/GWHD2021/extracted/gwhd_2021/images'),
    'phenobench': Path('/data/datasets/PhenoBench/PhenoBench-v110'),
    'deepweeds': Path('/data/datasets/DeepWeeds'),
    'ronin': Path('/data/datasets/Stage5B/weedcrop/extracted'),
    'sesame_weed': Path('/data/datasets/Stage5B/sesame_weed/extracted/agri_data/data'),
    'applebbch81': Path('/data/datasets/Stage5B/applebbch81/extracted'),
    'agrifield40k_pool': Path('/data/datasets/AgriField-40K/agrifield40k'),
    'artemis_imagesafari': DATA_ROOT / 'raw/artemis_imagesafari',
}

#: Licence conflicts are recorded, never auto-resolved.
LICENCE_OVERRIDE: Dict[str, str] = {
    'perennial_plants': 'LICENSE_BLOCKED: Kaggle says "Other", ledger says MIT',
    'plantvillage': 'LICENSE_BLOCKED: Zenodo says CC-BY-4.0, dataset description says CC-BY-SA-3.0-US',
    'deepweeds': 'HOLDOUT',
}

LICENCE_ACCEPTED = ('CC0', 'CC-BY', 'MIT', 'APACHE', 'LGPL', 'ODBL', 'PUBLIC-DOMAIN')

FIELDS = [
    'image_id', 'physical_source_id', 'origin_ref',
    'raw_path', 'archive_id', 'raw_sha256', 'pixel_sha256', 'phash', 'dhash',
    'width', 'height', 'channels', 'bytes', 'format', 'mode', 'exif_orientation',
    'decode_ok', 'decode_error',
    'laplacian_var', 'tenengrad', 'brightness_mean', 'dark_pixel_fraction',
    'bright_pixel_fraction', 'entropy', 'colourfulness', 'blank_fraction',
    'aspect_ratio', 'short_side',
    'cleanvision_flags', 'iqa_musiq', 'iqa_niqe',
    'acquisition_domain', 'domain_evidence_type', 'domain_model_sha',
    'domain_confidence', 'domain_status',
    'crop_label_if_metadata', 'farm_id', 'plot_id', 'plant_id', 'session_id',
    'group_key', 'group_known', 'duplicate_cluster_id', 'nearest_sim',
    'qc_policy_version', 'qc_status', 'qc_reason_codes',
    'p0_status', 'p0_reason_codes',
    'holdout_flag', 'license_status', 'archive_only',
    'train_split', 'selected_for_R1', 'selected_for_R2', 'selected_for_R3',
    'rel_path', 'source_priority',
]


def load_policy() -> Dict[str, object]:
    import yaml
    return yaml.safe_load(POLICY_PATH.read_text(encoding='utf-8'))


def phash_from_metrics(metrics: Dict[str, object]) -> tuple[str, str]:
    # Placeholder: computed from the decoded file in `_measure`, see below.
    return '', ''


def enumerate_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _measure_one(args):
    path_str, = args
    result = decode_and_measure(Path(path_str))
    if result.get('decode_error', '').startswith('read:'):
        # A single read failure is usually a transient NFS/FUSE hiccup; retry
        # once before recording a permanent decode failure.
        result = decode_and_measure(Path(path_str))
    if result['decode_ok']:
        from PIL import Image
        from src.registry.ingest import dhash64, phash64
        try:
            with Image.open(path_str) as image:
                image.load()
                result['phash64'] = phash64(image)
                result['dhash64'] = dhash64(image)
        except Exception as exc:  # noqa: BLE001
            result['decode_error'] = f'hash: {exc}'
    return result


def licence_status(source_id: str, licence: str) -> str:
    if source_id in LICENCE_OVERRIDE:
        return LICENCE_OVERRIDE[source_id]
    value = (licence or '').upper()
    if not any(value.startswith(prefix) for prefix in LICENCE_ACCEPTED):
        if value in ('', 'UNVERIFIED', 'UNKNOWN', 'NAN'):
            return 'RESEARCH_ONLY_UNVERIFIED'
        return f'LICENSE_BLOCKED: {licence}'
    if 'NC' in value:
        return 'ACCEPTED_NONCOMMERCIAL'
    return 'ACCEPTED'


def build_source(source_id: str, workers: int = 16, limit: Optional[int] = None,
                 skip_visual: bool = False) -> Dict[str, object]:
    policy = load_policy()
    qc_policy = QualityPolicy.from_dict(policy)
    row_meta = {r.physical_source_id: r for r in all_rows()}
    meta = row_meta.get(source_id)
    if meta is None:
        raise KeyError(f'unknown source {source_id}')
    root = ROOTS.get(source_id)
    paths = enumerate_images(root) if root else []
    if limit:
        paths = paths[:limit]
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    if paths:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_measure_one, (str(p),)): p for p in paths}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r: r['path'])

    # ---- metadata domain + vision for ambiguous images -------------------- #
    from src.registry.stage5c_domain_classifier import (VisionDomainClassifier,
                                                        decide)
    classifier = None
    ambiguous: List[Dict[str, object]] = []
    for result in results:
        rel = str(Path(result['path']).relative_to(root)) if root else Path(result['path']).name
        result['rel_path'] = rel
        decision = decide(source_id, rel, None, None)
        result['_meta_domain'] = decision
        if decision is None or decision.domain == 'UNKNOWN':
            ambiguous.append(result)
        else:
            _apply_domain(result, decision, source_id, rel)

    if ambiguous and not skip_visual:
        classifier = VisionDomainClassifier(device=policy['quality']['iqa']['device'])
        import torch
        from PIL import Image
        batch = 64
        for start in range(0, len(ambiguous), batch):
            chunk = ambiguous[start:start + batch]
            paths_chunk = [Path(r['path']) for r in chunk if r['decode_ok']]
            if not paths_chunk:
                for result in chunk:
                    _apply_domain(result, decide(source_id, result['rel_path'], None, None),
                                  source_id, result['rel_path'])
                continue
            predictions = classifier.classify(paths_chunk, batch_size=len(paths_chunk))
            cursor = 0
            for result in chunk:
                if not result['decode_ok']:
                    _apply_domain(result, decide(source_id, result['rel_path'], None, None),
                                  source_id, result['rel_path'])
                    continue
                domain, confidence, scores = predictions[cursor]
                cursor += 1
                # Use the batched prediction directly; do not re-run the model.
                from src.registry.stage5c_domain_classifier import DomainDecision
                ordered = sorted(scores.values(), reverse=True)
                margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
                if margin < 0.08 or confidence < 0.55:
                    decision = DomainDecision('UNKNOWN', 'vision_model_inferred', confidence,
                                              'QUARANTINE_DOMAIN', 'low-confidence visual domain',
                                              classifier.model_sha, scores)
                else:
                    decision = DomainDecision(domain, 'vision_model_inferred', confidence,
                                              'OK', 'visual model accepted',
                                              classifier.model_sha, scores)
                _apply_domain(result, decision, source_id, result['rel_path'])
            if start % (batch * 10) == 0:
                print(f'  vision {source_id}: {start + len(chunk)}/{len(ambiguous)}', flush=True)
    elif ambiguous:
        # ``--skip-visual``: the model is deliberately not run.  These rows stay
        # UNKNOWN and are therefore excluded from R2/R3; the flag is recorded so
        # nobody mistakes "not classified" for "classified as not-field".
        for result in ambiguous:
            result['domain'] = 'UNKNOWN'
            result['domain_evidence'] = 'vision_skipped'
            result['domain_confidence'] = 0.0
            result['domain_status'] = 'UNCLASSIFIED'

    # ---- remaining fields -------------------------------------------------- #
    licence = meta.license_id
    lic = licence_status(source_id, licence)
    is_holdout = source_id in HOLDOUT_IDS or meta.holdout
    rows_out = []
    for result in results:
        image_id = f'{source_id}:' + hashlib.sha1(result['rel_path'].encode()).hexdigest()[:16]
        group_key, group_known, _ = derive_group(source_id, result['rel_path'])
        split = assign_split(group_key, group_known, image_id,
                             salt=policy['groups']['hash_salt'],
                             val_fraction=policy['groups']['val_fraction'])
        status, reasons = basic_decision(result, qc_policy)
        p0, p0_reasons = paper_decision(result, qc_policy, 'ACCEPTED' in lic)
        if source_id == 'agrifield40k_pool' and result['rel_path'].split('/')[0] == 'train' and Path(result['rel_path']).name.startswith('pnb_'):
            is_row_holdout = True
        else:
            is_row_holdout = False
        row = {field: '' for field in FIELDS}
        row.update({
            'image_id': image_id,
            'physical_source_id': source_id,
            'origin_ref': ';'.join(meta.original_ref_ids),
            'raw_path': result['path'],
            'raw_sha256': result['raw_sha256'],
            'pixel_sha256': result['pixel_sha256'],
            'phash': result.get('phash64', ''),
            'dhash': result.get('dhash64', ''),
            'width': result['width'], 'height': result['height'],
            'channels': result['channels'], 'bytes': result['bytes'],
            'format': result['format'], 'mode': result['mode'],
            'exif_orientation': result['exif_orientation'],
            'decode_ok': result['decode_ok'], 'decode_error': result['decode_error'],
            'laplacian_var': result['laplacian_var'], 'tenengrad': result['tenengrad'],
            'brightness_mean': result['brightness_mean'],
            'dark_pixel_fraction': result['dark_pixel_fraction'],
            'bright_pixel_fraction': result['bright_pixel_fraction'],
            'entropy': result['entropy'], 'colourfulness': result['colourfulness'],
            'blank_fraction': result['blank_fraction'], 'aspect_ratio': result['aspect_ratio'],
            'short_side': result['short_side'],
            'acquisition_domain': result.get('domain', 'UNKNOWN'),
            'domain_evidence_type': result.get('domain_evidence', 'none'),
            'domain_model_sha': result.get('domain_model_sha', ''),
            'domain_confidence': result.get('domain_confidence', ''),
            'domain_status': result.get('domain_status', 'OK'),
            'group_key': group_key, 'group_known': group_known,
            'qc_policy_version': policy['version'],
            'qc_status': status, 'qc_reason_codes': ';'.join(reasons),
            'p0_status': p0, 'p0_reason_codes': ';'.join(p0_reasons),
            'holdout_flag': 'yes' if (is_holdout or is_row_holdout) else 'no',
            'license_status': lic,
            'archive_only': 'yes' if meta.archive_only else 'no',
            'train_split': split,
            'rel_path': result['rel_path'],
            'source_priority': meta.groups[0] if meta.groups else '',
        })
        row['selected_for_R1'] = 'yes' if p0 == 'AUTO_ACCEPT' and not is_holdout else 'no'
        row['selected_for_R2'] = 'no'
        row['selected_for_R3'] = 'no'
        rows_out.append(row)

    out_path = OUT_DIR / f'{source_id}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        'source_id': source_id,
        'root': str(root),
        'files_found': len(paths),
        'decoded': sum(1 for r in rows_out if str(r['decode_ok']) in ('True', '1', 'true')),
        'qc_accept': sum(1 for r in rows_out if r['qc_status'] == 'AUTO_ACCEPT'),
        'qc_reject': sum(1 for r in rows_out if r['qc_status'] == 'AUTO_REJECT'),
        'qc_quarantine': sum(1 for r in rows_out if r['qc_status'] == 'QUARANTINE'),
        'domain_field': sum(1 for r in rows_out if r['acquisition_domain'] in ('FIELD_IN_SITU', 'ORCHARD_IN_SITU')),
        'domain_unknown': sum(1 for r in rows_out if r['acquisition_domain'] == 'UNKNOWN'),
        'holdout': sum(1 for r in rows_out if r['holdout_flag'] == 'yes'),
        'seconds': round(time.time() - started, 1),
        'csv': str(out_path),
    }
    (OUT_DIR / f'{source_id}.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def _apply_domain(result: Dict[str, object], decision, source_id: str, rel: str) -> None:
    result['domain'] = decision.domain
    result['domain_evidence'] = decision.evidence_type
    result['domain_confidence'] = decision.confidence
    result['domain_status'] = decision.status
    result['domain_model_sha'] = decision.model_sha
    if decision.status == 'QUARANTINE_DOMAIN':
        result['domain'] = 'UNKNOWN'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--skip-visual', action='store_true',
                        help='do not run the CLIP vision classifier; ambiguous '
                             'images stay UNKNOWN and are excluded from R2/R3')
    args = parser.parse_args()
    targets = args.source or (list(ROOTS) if args.all else [])
    if not targets:
        parser.error('pass --source or --all')
    summaries = []
    for source_id in targets:
        print(f'==> {source_id}', flush=True)
        summary = build_source(source_id, workers=args.workers, limit=args.limit,
                               skip_visual=args.skip_visual)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    path = ROOT / 'reports/stage5c_ledger_run.json'
    prior = []
    if path.exists():
        try:
            prior = json.loads(path.read_text())
        except json.JSONDecodeError:
            prior = []
    path.write_text(json.dumps(prior + summaries, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
