#!/usr/bin/env python
"""Calibrate the automated QC thresholds with deterministic synthetic faults.

This is a *sensitivity* calibration, not a human ground truth: it measures
whether the detectors respond to known degradations, and how often clean
images are flagged.  The report says so explicitly.  No threshold is changed
to make a corpus larger.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.registry.stage5c_qc_basic import (QualityPolicy, basic_decision,
                                           decode_and_measure)

WORK = ROOT / 'data/corpus_v2/work/qc_calibration'
IMAGE_DIR = ROOT / 'data/corpus_v2/images'
REPORTS = ROOT / 'reports'


def distort(image: Image.Image, kind: str) -> Image.Image:
    if kind == 'blur_sigma2':
        return image.filter(ImageFilter.GaussianBlur(2))
    if kind == 'blur_sigma4':
        return image.filter(ImageFilter.GaussianBlur(4))
    if kind == 'dark_0.25':
        return ImageEnhance.Brightness(image).enhance(0.25)
    if kind == 'bright_1.8':
        return ImageEnhance.Brightness(image).enhance(1.8)
    if kind == 'jpeg_q10':
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='JPEG', quality=10)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    if kind == 'noise_sigma25':
        array = np.asarray(image.convert('RGB'), dtype=np.float32)
        rng = np.random.default_rng(20261024)
        array = np.clip(array + rng.normal(0, 25, array.shape), 0, 255)
        return Image.fromarray(array.astype(np.uint8))
    if kind == 'downscale_0.25':
        size = (max(8, image.width // 4), max(8, image.height // 4))
        return image.resize(size, Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.BILINEAR)
    raise ValueError(kind)


FAULTS = ['blur_sigma2', 'blur_sigma4', 'dark_0.25', 'bright_1.8',
          'jpeg_q10', 'noise_sigma25', 'downscale_0.25']


def _metrics(path: Path) -> Dict[str, object]:
    result = decode_and_measure(path)
    return result


def detection(metrics: Dict[str, object], fault: str, policy: QualityPolicy) -> bool:
    if fault.startswith('blur'):
        return (metrics.get('laplacian_var') or 0) < policy.laplacian_var_min
    if fault == 'dark_0.25':
        return (metrics.get('brightness_mean') or 255) < policy.dark_mean_max
    if fault == 'bright_1.8':
        return (metrics.get('brightness_mean') or 0) > policy.bright_mean_min
    if fault in ('jpeg_q10', 'noise_sigma25', 'downscale_0.25'):
        return (metrics.get('laplacian_var') or 0) < policy.laplacian_var_min or \
               (metrics.get('entropy') or 99) < policy.entropy_min
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', default='agrifm_ref1,agrifm_ref6')
    parser.add_argument('--per-source', type=int, default=12)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--skip-iqa', action='store_true')
    args = parser.parse_args()
    import yaml
    policy = QualityPolicy.from_dict(yaml.safe_load(
        (ROOT / 'data/corpus_v2/policy.yaml').read_text()))
    WORK.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for source in [s.strip() for s in args.sources.split(',') if s.strip()]:
        path = IMAGE_DIR / f'{source}.csv'
        if not path.exists():
            print('missing', path)
            continue
        with open(path, newline='', encoding='utf-8') as handle:
            ledger = [r for r in csv.DictReader(handle)
                      if str(r.get('decode_ok')).lower() in ('true', '1')]
        for index, record in enumerate(ledger[:args.per_source]):
            original_path = Path(record['raw_path'])
            try:
                with Image.open(original_path) as image:
                    image = image.convert('RGB')
                    for fault in FAULTS:
                        broken = distort(image, fault)
                        out = WORK / f'{source}_{index}_{fault}.jpg'
                        broken.save(out, quality=92)
                        rows.append({'source': source, 'case': fault,
                                     'path': str(out), 'clean_path': str(original_path),
                                     'is_clean': False})
                    clean = WORK / f'{source}_{index}_clean.jpg'
                    image.save(clean, quality=92)
                    rows.append({'source': source, 'case': 'clean', 'path': str(clean),
                                 'clean_path': str(original_path), 'is_clean': True})
            except Exception as exc:  # noqa: BLE001
                print('skip', original_path, exc)

    # Metrics for every generated file.
    for row in rows:
        metrics = _metrics(Path(row['path']))
        row.update({key: metrics.get(key) for key in
                    ('laplacian_var', 'tenengrad', 'brightness_mean', 'entropy',
                     'blank_fraction', 'short_side')})

    if not args.skip_iqa:
        from tools.stage5c_run_qc import score_pyiqa
        scores = score_pyiqa([r['path'] for r in rows], ['musiq'], args.device)
        for row in rows:
            row['iqa_musiq'] = scores.get(row['path'], {}).get('musiq', '')

    # Detection matrices.
    clean = [r for r in rows if r['is_clean']]
    report = []
    for fault in FAULTS:
        positives = [r for r in rows if r['case'] == fault]
        tp = sum(1 for r in positives if detection(r, fault, policy))
        fp = sum(1 for r in clean if detection(r, 'blur' if fault.startswith('blur') else fault, policy))
        report.append({
            'fault': fault,
            'positives': len(positives),
            'clean_negatives': len(clean),
            'true_positive': tp,
            'false_positive': fp,
            'recall': round(tp / max(1, len(positives)), 4),
            'false_positive_rate': round(fp / max(1, len(clean)), 4),
            'precision': round(tp / max(1, tp + fp), 4),
        })

    # IQA separation between clean and degraded.
    iqa_rows = []
    if 'iqa_musiq' in rows[0]:
        for fault in FAULTS:
            values = [float(r['iqa_musiq']) for r in rows
                      if r['case'] == fault and r.get('iqa_musiq') not in ('', None)]
            clean_values = [float(r['iqa_musiq']) for r in clean if r.get('iqa_musiq') not in ('', None)]
            if values and clean_values:
                iqa_rows.append({
                    'fault': fault,
                    'clean_median': round(float(np.median(clean_values)), 3),
                    'fault_median': round(float(np.median(values)), 3),
                    'delta': round(float(np.median(clean_values) - np.median(values)), 3),
                    'share_below_35': round(float(np.mean(np.array(values) < 35)), 3),
                })

    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / 'stage5c_qc_calibration.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    payload = {
        'method': 'deterministic synthetic degradations; sensitivity only, NOT human ground truth',
        'policy': {
            'laplacian_var_min': policy.laplacian_var_min,
            'dark_mean_max': policy.dark_mean_max,
            'bright_mean_min': policy.bright_mean_min,
            'entropy_min': policy.entropy_min,
        },
        'detectors': report,
        'iqa_separation': iqa_rows,
        'images_evaluated': len(rows),
        'clean_images': len(clean),
    }
    (REPORTS / 'stage5c_qc_calibration.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2)[:3000])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
