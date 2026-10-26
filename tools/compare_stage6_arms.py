#!/usr/bin/env python
"""Compare the Stage 6 B1/B2 runs from their own artefacts.

Produces ``reports/stage6_b1_b2_comparison.{md,csv,json}``.  Realised source
exposure is always taken from the per-run ledger (or recomputed from the
per-sample ledger CSV), never from the config.

Usage:
    ./.venv-uv/bin/python -m tools.compare_stage6_arms
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

from src.datasets.stage6_sampling import concentration

ROOT = Path(__file__).resolve().parents[1]
PROFILE_JSON = ROOT / 'data/corpus_v4/R3v3_license_verified_sampling_profiles.json'

DEFAULT = {
    'B1': ROOT / 'runs/stage6_b1_natural_2000',
    'B2': ROOT / 'runs/stage6_b2_capped_no_backfill_2000',
}

#: fields that must be identical between the two runs (everything except
#: sampling policy, arm identity and output folder)
NON_SAMPLING_KEYS = (
    'meta.model_name', 'meta.patch_size', 'meta.pred_depth', 'meta.pred_emb_dim',
    'meta.use_bfloat16', 'meta.use_sdpa', 'meta.activation_checkpointing',
    'meta.pretrained_checkpoint', 'meta.pretrained_sha256',
    'data.manifest', 'data.manifest_sha256', 'data.split', 'data.seed',
    'data.augmentation_seed', 'data.batch_size',
    'data.num_workers', 'data.crop_size', 'data.max_samples',
    'optimization.max_optimizer_steps', 'optimization.save_at_optimizer_steps',
    'optimization.grad_accum', 'optimization.lr', 'optimization.start_lr',
    'optimization.final_lr', 'optimization.warmup', 'optimization.warmup_steps',
    'optimization.weight_decay', 'optimization.final_weight_decay',
    'optimization.ema', 'optimization.epochs', 'optimization.ipe_scale',
    'mask.mode', 'mask.allow_overlap', 'mask.aspect_ratio',
    'mask.enc_mask_scale', 'mask.pred_mask_scale', 'mask.num_enc_masks',
    'mask.num_pred_masks', 'mask.min_keep', 'mask.patch_size',
)


def dig(node, dotted: str):
    current = node
    for part in dotted.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def read_csv_rows(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def selected_by_source(run_dir: Path) -> Dict[str, object]:
    """Recompute source counts from the per-sample ledger CSV (full run)."""
    ledger_path = run_dir / 'source_exposure_ledger.csv'
    if not ledger_path.is_file():
        summary = read_json(run_dir / 'source_exposure_ledger.summary.json')
        return {'per_sample_rows': 0, 'hhi': summary.get('hhi'),
                'entropy': summary.get('entropy'),
                'max_share': summary.get('max_share'),
                'max_source': summary.get('max_source'),
                'total': summary.get('total')}
    counts: Dict[str, int] = {}
    n = 0
    with ledger_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            source = row.get('physical_source_id', '')
            counts[source] = counts.get(source, 0) + 1
            n += 1
    summary = concentration(counts)
    summary['per_sample_rows'] = n
    summary['sources'] = len(counts)
    return summary


def summarise_arm(name: str, run_dir: Path) -> Dict[str, object]:
    params = {}
    if (run_dir / 'params-ijepa.yaml').is_file():
        params = yaml.safe_load((run_dir / 'params-ijepa.yaml').read_text(encoding='utf-8'))
    verification = {}
    if (run_dir / 'training_verification.yaml').is_file():
        verification = yaml.safe_load(
            (run_dir / 'training_verification.yaml').read_text(encoding='utf-8'))
    csv_rows = read_csv_rows(run_dir / 'jepa_r0.csv')
    losses = [float(row['loss']) for row in csv_rows]
    times = [float(row['time (ms)']) for row in csv_rows]
    progress = []
    console = run_dir / 'console.log'
    if console.is_file():
        for line in console.read_text(encoding='utf-8', errors='replace').splitlines():
            if 'PROGRESS step=' in line:
                progress.append(line)
    return {
        'arm': name,
        'run_dir': str(run_dir),
        'params': params,
        'verification': verification,
        'csv_rows': len(csv_rows),
        'loss_first': losses[0] if losses else None,
        'loss_last': losses[-1] if losses else None,
        'loss_min': min(losses) if losses else None,
        'loss_max': max(losses) if losses else None,
        'csv_time_ms_mean': (sum(times) / len(times)) if times else None,
        'progress_lines': len(progress),
        'ledger': selected_by_source(run_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1-run', type=Path, default=DEFAULT['B1'])
    parser.add_argument('--b2-run', type=Path, default=DEFAULT['B2'])
    parser.add_argument('--out-prefix', type=Path,
                        default=ROOT / 'reports/stage6_b1_b2_comparison')
    args = parser.parse_args()

    arms = {
        'B1': summarise_arm('B1', args.b1_run),
        'B2': summarise_arm('B2', args.b2_run),
    }

    problems: List[str] = []
    b1_params, b2_params = arms['B1']['params'], arms['B2']['params']
    for key in NON_SAMPLING_KEYS:
        left, right = dig(b1_params, key), dig(b2_params, key)
        if left != right:
            problems.append(f'{key}: B1={left!r} B2={right!r}')
    if arms['B1']['params'] and arms['B2']['params']:
        for arm in ('B1', 'B2'):
            if arms[arm]['verification'].get('optimizer_steps_total') != 2000:
                problems.append(f"{arm} optimizer_steps_total="
                                f"{arms[arm]['verification'].get('optimizer_steps_total')} != 2000")
        if arms['B1']['verification'].get('manifest_sha256') != \
                arms['B2']['verification'].get('manifest_sha256'):
            problems.append('manifest_sha256 differs between arms')
        for arm in ('B1', 'B2'):
            init = arms[arm]['verification'].get('pretrained_init') or {}
            if init.get('sha256') != arms['B1']['verification'].get(
                    'pretrained_init', {}).get('sha256'):
                problems.append(f'{arm} pretrained SHA differs')

    profile = read_json(PROFILE_JSON).get('by_source', {})
    comparison = {
        'arms': arms,
        'profile_nominal_capped_exposure': profile.get('exposure', {}).get('capped'),
        'profile_integer_allocation': (
            arms['B2'].get('verification', {}).get('sampling_profile', {})
            .get('integer_allocation')),
        'profile_info': {
            'cap_exposure': profile.get('capped_info', {}).get('cap_exposure'),
            'unfulfilled_exposure': profile.get('capped_info', {}).get('unfulfilled_exposure'),
            'budget_filled': profile.get('capped_info', {}).get('budget_filled'),
            'backfill': False,
        },
        'non_sampling_mismatches': problems,
        'protocol_ok': not problems,
    }

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (args.out_prefix.with_suffix('.json')).write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    rows: List[Dict[str, object]] = []
    for arm in ('B1', 'B2'):
        data = arms[arm]
        rows.append({
            'arm': arm,
            'run_dir': data['run_dir'],
            'sampling_mode': (data['params'].get('data', {}).get('sampling') or {})
            .get('mode', 'natural'),
            'optimizer_steps_total': data['verification'].get('optimizer_steps_total'),
            'micro_steps_total': data['verification'].get('micro_steps_total'),
            'csv_rows': data['csv_rows'],
            'loss_first': data['loss_first'],
            'loss_last': data['loss_last'],
            'loss_min': data['loss_min'],
            'loss_max': data['loss_max'],
            'peak_memory_mib': data['verification'].get('peak_memory_mib'),
            'step_time_ms_mean': data['verification'].get('step_time_ms_mean'),
            'emitted_samples': data['ledger'].get('total'),
            'hhi': data['ledger'].get('hhi'),
            'entropy': data['ledger'].get('entropy'),
            'max_share': data['ledger'].get('max_share'),
            'max_source': data['ledger'].get('max_source'),
            'optimizer_steps_per_epoch': data['verification'].get('optimizer_steps_per_epoch'),
        })
    with args.out_prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        '# Stage 6 — B1 / B2 对照汇总',
        '',
        '| 指标 | B1 natural | B2 capped (no backfill) |',
        '|---|---:|---:|',
    ]
    for key in ('optimizer_steps_total', 'micro_steps_total', 'csv_rows',
                'emitted_samples', 'loss_first', 'loss_last', 'loss_min',
                'loss_max', 'peak_memory_mib', 'step_time_ms_mean',
                'hhi', 'entropy', 'max_share', 'max_source',
                'optimizer_steps_per_epoch'):
        lines.append(f"| {key} | {rows[0].get(key)} | {rows[1].get(key)} |")
    lines += ['', '## 非采样协议字段一致性', '']
    if problems:
        lines.append('**不一致项：**')
        lines += [f'- `{problem}`' for problem in problems]
    else:
        lines.append('除采样策略、arm 标识与输出目录外，所有协议字段一致。')
    args.out_prefix.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'protocol_ok': not problems, 'problems': problems}, indent=2))
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
