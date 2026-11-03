"""Deterministic synthetic degradation harness for the Stage 5D QC pipeline.

The harness measures *sensitivity to constructed damage*, nothing more.  No
human quality label exists for the natural images, so no claim about natural
image accuracy is made.

Each base image is degraded with a fixed seed and the full pipeline decision is
recomputed for every variant:

* basic deterministic metrics (the P1 gate);
* MUSIQ (the reference-free IQA gate) when the model weights are available.

``--offline`` skips the model and reports the deterministic gate only, which is
enough to quantify the recall that the *basic* rules alone achieve.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.registry.stage5d_qc_validation import DEGRADATIONS, degrade, measure_pil
from src.registry.stage5d_qc_policy import load_policy, MetricSpec


def stratify(master: pd.DataFrame, strata: int, per_stratum: int,
             seed: int) -> List[Tuple[str, str]]:
    """Deterministically pick base images across sources and domains."""
    candidates = master[master['decode_ok'].astype(str).str.lower().isin(('true', '1'))]
    candidates = candidates[candidates['raw_path'].astype(str).str.len() > 0]
    rng = np.random.default_rng(seed)
    chosen: List[Tuple[str, str]] = []
    for source, group in candidates.groupby('physical_source_id'):
        domains = group['acquisition_domain'].astype(str).unique().tolist()
        for domain in domains:
            subset = group[group['acquisition_domain'].astype(str) == domain]
            take = min(per_stratum, len(subset))
            if take == 0:
                continue
            rows = subset.iloc[sorted(rng.choice(len(subset), take, replace=False))]
            for row in rows.itertuples():
                chosen.append((str(row.raw_path), f'{source}|{domain}'))
    if not chosen:
        raise SystemExit('no decodable images found for the synthetic harness')
    return chosen


def run(policy_path: Path, master: pd.DataFrame, strata: int, per_stratum: int,
        seed: int, offline: bool) -> Tuple[pd.DataFrame, Dict[str, object]]:
    policy = load_policy(policy_path)
    musiq_spec = policy.metrics.get('musiq')
    musiq = None
    if not offline and musiq_spec is not None:
        import torch
        import pyiqa
        musiq = pyiqa.create_metric('musiq', device='cuda:1')
        torch.set_num_threads(1)

    bases = stratify(master, strata, per_stratum, seed)
    records: List[Dict[str, object]] = []
    for index, (path, stratum) in enumerate(bases):
        try:
            with Image.open(path) as image:
                image.load()
                base = image.convert('RGB').copy()
        except Exception:  # noqa: BLE001
            continue
        for variant, base_id in (('clean', f'clean{index}'),):
            metrics = measure_pil(base)
            status, reasons = _basic(metrics, policy)
            score = _score_musiq(musiq, base) if musiq is not None else None
            iqa_status, iqa_reason = ('SKIPPED', 'MODEL_NOT_RUN') if score is None \
                else musiq_spec.evaluate(score)
            records.append({
                'stratum': stratum, 'variant': variant, 'kind': 'none',
                'severity': 'clean', 'base_id': base_id,
                'short_side': metrics['short_side'],
                'laplacian_var': round(metrics['laplacian_var'], 3),
                'musiq': score, 'basic_status': status, 'basic_reasons': ';'.join(reasons),
                'iqa_status': iqa_status, 'iqa_reason': iqa_reason,
                'label': 'clean', 'detected': 0,
            })
        for kind, value, severity in DEGRADATIONS:
            damaged = degrade(base, kind, value, seed_key=f'{index}')
            if damaged is None:
                continue
            metrics = measure_pil(damaged)
            status, reasons = _basic(metrics, policy)
            score = _score_musiq(musiq, damaged) if musiq is not None else None
            iqa_status, iqa_reason = ('SKIPPED', 'MODEL_NOT_RUN') if score is None \
                else musiq_spec.evaluate(score)
            detected = int(status in ('REJECT', 'QUARANTINE') or
                           iqa_status in ('REJECT', 'QUARANTINE'))
            records.append({
                'stratum': stratum, 'variant': f'{kind}_{value}',
                'kind': kind, 'severity': severity,
                'base_id': f'clean{index}',
                'short_side': metrics['short_side'],
                'laplacian_var': round(metrics['laplacian_var'], 3),
                'musiq': score, 'basic_status': status,
                'basic_reasons': ';'.join(reasons),
                'iqa_status': iqa_status, 'iqa_reason': iqa_reason,
                'label': 'degraded', 'detected': detected,
            })
        if (index + 1) % 25 == 0:
            print(f'  {index + 1}/{len(bases)} base images', flush=True)
    return pd.DataFrame(records), {'base_images': len(bases),
                                   'degradations_per_base': len(DEGRADATIONS)}


def _basic(metrics: Dict[str, object], policy) -> Tuple[str, List[str]]:
    thresholds = policy.basic
    hard: List[str] = []
    if metrics['short_side'] < thresholds.min_short_side:
        return 'REJECT', ['BELOW_MIN_RESOLUTION']
    if metrics['blank_fraction'] > thresholds.blank_pixel_fraction_max:
        return 'REJECT', ['BLANK_IMAGE']
    if metrics['laplacian_var'] < thresholds.laplacian_var_min:
        hard.append('BLURRY')
    if metrics['tenengrad'] < thresholds.tenengrad_min:
        hard.append('LOW_DETAIL')
    if metrics['brightness_mean'] < thresholds.dark_mean_max:
        hard.append('DARK')
    if metrics['brightness_mean'] > thresholds.bright_mean_min:
        hard.append('OVEREXPOSED')
    if metrics['entropy'] < thresholds.entropy_min:
        hard.append('LOW_INFORMATION')
    if len(hard) >= 2 and thresholds.quarantine_on_conflict:
        return 'QUARANTINE', hard + ['MULTI_METRIC_CONFLICT']
    if hard:
        return 'REJECT', hard
    return 'PASS', []


def _score_musiq(metric, image: Image.Image) -> Optional[float]:
    import torch
    from torchvision import transforms
    tensor = transforms.ToTensor()(image.resize((224, 224))).unsqueeze(0).to('cuda:1')
    with torch.inference_mode():
        return float(metric(tensor))


def load_master() -> pd.DataFrame:
    """Accepted Stage 5D images, read from the per-source ledgers.

    The published train/val CSV deliberately omits ``raw_path`` and
    ``decode_ok``; the per-source ledger keeps them, so the harness reads the
    ledger and keeps only the accepted rows.
    """
    frames = []
    for path in sorted((ROOT / 'data/corpus_v3/images').glob('*.csv')):
        frame = pd.read_csv(
            path, low_memory=False,
            usecols=['image_id', 'physical_source_id', 'raw_path',
                     'acquisition_domain', 'decode_ok', 'selected'])
        frame = frame[frame['selected'].astype(str).str.lower().isin(
            ('true', '1'))]
        frames.append(frame)
    if not frames:
        raise SystemExit('no Stage 5D ledger found; build the corpus first')
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-stratum', type=int, default=6)
    parser.add_argument('--strata', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    master = load_master()
    frame, info = run(ROOT / 'data/corpus_v3/policy.yaml', master,
                      args.strata, args.per_stratum, args.seed, args.offline)

    clean = frame[frame['label'] == 'clean']
    degraded = frame[frame['label'] == 'degraded']
    summary = {
        **info,
        'clean_rows': int(len(clean)),
        'degraded_rows': int(len(degraded)),
        'clean_retention': float((clean['basic_status'] == 'PASS').mean()),
        'clean_false_removal': float((clean['basic_status'] != 'PASS').mean()),
        'overall_recall': float(degraded['detected'].mean()),
        'basic_only_recall': float((degraded['basic_status'] != 'PASS').mean()),
        'iqa_only_recall': float((degraded['iqa_status'].isin(
            ['REJECT', 'QUARANTINE'])).mean()) if not args.offline else None,
        'per_degradation': degraded.groupby(['kind', 'severity'])['detected']
        .agg(['count', 'mean']).reset_index().to_dict('records'),
        'per_stratum': degraded.groupby('stratum')['detected'].mean().to_dict(),
        'disclaimer': ('labels are synthetic degradations, not human quality '
                       'judgements; recall measures sensitivity to the constructed '
                       'damage only'),
    }
    REPORTS = ROOT / 'reports'
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS / 'stage5d_synthetic_qc.csv', index=False)
    (REPORTS / 'stage5d_synthetic_qc.json').write_text(
        json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('per_degradation', 'per_stratum')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
