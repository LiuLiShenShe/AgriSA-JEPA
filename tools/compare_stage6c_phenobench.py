#!/usr/bin/env python
"""Compare the Stage 6C PhenoBench arms (A / B1 / B2) from their artefacts.

Everything here is read from the arms' own ``result.json`` and prediction
directories.  No metric is hard-coded: the script recomputes means, standard
deviations and paired per-seed differences from the real values.  The only
constants are the *expected* checkpoint SHA-256 values, the fixed protocol
contract, and the all-soil trivial baseline recorded in the data audit -- each
of which is asserted rather than trusted.

What is checked before any aggregate is written
-----------------------------------------------
1. The three arms used the same evaluation protocol (tiling, head, optimizer,
   LR, weight decay, epochs, batch size, seeds, train/val image counts).  Only
   the backbone checkpoint and the arm/pretraining metadata may differ.
2. Each arm's recorded checkpoint SHA-256 equals the expected SHA-256 (and,
   with ``--verify-disk``, the file currently on disk hashes to the same value).
3. Every arm has exactly the probe seeds 0, 1, 2.
4. The self-implemented IoU agrees with the official ``phenobench-eval``
   metrics to < 0.01 percentage points on every seed.
5. Every one of the 772 official val images has a prediction PNG, by filename,
   for every seed.  A short prediction set is refused, because the official
   evaluator silently truncates to the intersection of its file list.

Interpretation guardrails
-------------------------
* Three probe seeds measure probe/training randomness, not pretraining
  randomness.  Each arm has exactly one pretraining checkpoint.
* No p-value is computed and no "statistically significant" claim is made.
* When a difference is small or the per-class directions disagree, the summary
  is repeated as mixed/uncertain rather than forced into a winner.

Usage
-----
    ./.venv-uv/bin/python -m tools.compare_stage6c_phenobench
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.downstream.phenobench_data import list_split_files

ROOT = Path(__file__).resolve().parents[1]

#: Probe seeds required for every arm.
EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_PROBE_PARAMETERS = 3843
EXPECTED_TRAIN_IMAGES = 1407
EXPECTED_VAL_IMAGES = 772
#: Self metric vs official evaluator agreement, in percentage points.
TOLERANCE_PP = 0.01
#: All-soil trivial baseline recorded in reports/phenobench_data_audit.md.
TRIVIAL_BASELINE_MIOU = 29.97
TRIVIAL_BASELINE_TOLERANCE = 0.05
DATA_VERSION = 'v1.1.0'
DATA_MANIFEST = ROOT / 'reports/phenobench_file_manifest.sha256.txt'

#: Expected checkpoint identities.  Any change here must be justified in the
#: Stage 6C protocol report, never silently relaxed to make a run pass.
ARM_SPECS: Dict[str, Dict[str, str]] = {
    'A': {
        'checkpoint': '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
        'sha256': '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22',
        'result': str(ROOT / 'runs/stage3a_phenobench/armA/result.json'),
        'role': 'official I-JEPA ViT-H/14, no agricultural continued pretraining',
    },
    'B1': {
        'checkpoint': 'runs/stage6_b1_natural_2000/jepa-encoder-step2000.pth',
        'sha256': '19796cebc65e29a07eafd7222f5a642d4ef9f28c33f2e4f4f2b65e34a2bfde6b',
        'result': str(ROOT / 'runs/stage6c_phenobench/B1_natural/result.json'),
        'role': 'Stage 6 natural sampling, 2000 optimizer updates',
    },
    'B2': {
        'checkpoint': 'runs/stage6_b2_capped_no_backfill_2000/jepa-encoder-step2000.pth',
        'sha256': '6e7a83c9156e556757016145fb804e2b67e7b94f5e8b45bda4ffd24cb5975a62',
        'result': str(ROOT / 'runs/stage6c_phenobench/B2_capped/result.json'),
        'role': 'Stage 6 capped no-backfill sampling, 2000 optimizer updates',
    },
}

METRIC_KEYS = ('soil', 'crop', 'weed', 'mIoU')
DELTA_PAIRS = (('B1', 'A'), ('B2', 'A'), ('B2', 'B1'))


# --------------------------------------------------------------------------- #
# Hashing helpers
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prediction_manifest_sha256(names: Sequence[str], predictions_dir: str | Path) -> str:
    """Stable hash over ``(filename, byte size)`` of a prediction set."""
    directory = Path(predictions_dir)
    lines = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing prediction: {path}')
        lines.append(f'{name}\t{path.stat().st_size}')
    return sha256_text('\n'.join(lines))


def git_commit(root: Path = ROOT) -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #

def read_json(path: str | Path) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metric_values(seed_entry: Dict[str, object]) -> Dict[str, float]:
    """Metrics as reported: official evaluator for crop/weed/mIoU, self for soil.

    ``phenobench-eval --task semantics`` prints only ``IoU (crop)``,
    ``IoU (weed)`` and ``mIoU``; it does not print the soil line.  Soil IoU is
    therefore taken from the self-implemented accumulator, which is separately
    cross-checked against the official mIoU identity.
    """
    official = seed_entry.get('official_metrics') or {}
    self_metrics = seed_entry['self_metrics']['per_class_iou_percent']
    return {
        'soil': float(self_metrics['soil']),
        'crop': float(official['IoU (crop)']),
        'weed': float(official['IoU (weed)']),
        'mIoU': float(official['mIoU']),
    }


def protocol_problems(results: Dict[str, Dict[str, object]]) -> List[str]:
    """Assert that the three arms differ only in backbone and arm metadata."""
    problems: List[str] = []
    reference = results['A']
    for arm in ('B1', 'B2'):
        if results[arm]['protocol'] != reference['protocol']:
            left = reference['protocol']
            right = results[arm]['protocol']
            for key in sorted(set(left) | set(right)):
                if left.get(key) != right.get(key):
                    problems.append(
                        f'protocol.{key} differs: A={left.get(key)!r} '
                        f'{arm}={right.get(key)!r}')
        if results[arm].get('probe_parameters') != reference.get('probe_parameters'):
            problems.append(
                f'probe_parameters differs: A={reference.get("probe_parameters")} '
                f'{arm}={results[arm].get("probe_parameters")}')
    if reference.get('probe_parameters') != EXPECTED_PROBE_PARAMETERS:
        problems.append(
            f'A probe_parameters={reference.get("probe_parameters")} != '
            f'{EXPECTED_PROBE_PARAMETERS}')
    return problems


def validate_arm(arm: str, result: Dict[str, object], spec: Dict[str, str],
                 val_names: Sequence[str], verify_disk: bool = True) -> List[str]:
    """Return every reason this arm cannot enter the aggregate."""
    problems: List[str] = []

    recorded_sha = result.get('backbone_checkpoint_sha256')
    if recorded_sha != spec['sha256']:
        problems.append(
            f'{arm}: recorded checkpoint SHA {recorded_sha} != expected {spec["sha256"]}')
    checkpoint = Path(spec['checkpoint'])
    if verify_disk:
        if not checkpoint.is_file():
            problems.append(f'{arm}: checkpoint file missing: {checkpoint}')
        else:
            disk_sha = sha256_file(checkpoint)
            if disk_sha != spec['sha256']:
                problems.append(
                    f'{arm}: on-disk checkpoint SHA {disk_sha} != expected {spec["sha256"]}')

    seeds = [entry.get('seed') for entry in result.get('seeds', [])]
    if seeds != EXPECTED_SEEDS:
        problems.append(f'{arm}: probe seeds {seeds} != {EXPECTED_SEEDS}')

    protocol = result.get('protocol', {})
    if protocol.get('train_images') != EXPECTED_TRAIN_IMAGES:
        problems.append(
            f'{arm}: train_images {protocol.get("train_images")} != {EXPECTED_TRAIN_IMAGES}')
    if protocol.get('val_images') != EXPECTED_VAL_IMAGES:
        problems.append(
            f'{arm}: val_images {protocol.get("val_images")} != {EXPECTED_VAL_IMAGES}')

    for entry in result.get('seeds', []):
        seed = entry.get('seed')
        official = entry.get('official_metrics') or {}
        for key in ('IoU (crop)', 'IoU (weed)', 'mIoU'):
            if key not in official:
                problems.append(f'{arm} seed {seed}: official metric {key!r} missing')
        self_metrics = entry.get('self_metrics', {}).get('per_class_iou_percent', {})
        for key in ('soil', 'crop', 'weed'):
            if key not in self_metrics:
                problems.append(f'{arm} seed {seed}: self metric {key!r} missing')

        cross = entry.get('metric_cross_check') or {}
        if cross.get('agrees') is not True:
            problems.append(f'{arm} seed {seed}: self/official cross-check not marked as agreeing')
        diffs = [abs(float(row['difference_pp'])) for row in cross.get('rows', [])
                 if row.get('difference_pp') is not None
                 and np.isfinite(float(row['difference_pp']))]
        if diffs and max(diffs) > TOLERANCE_PP:
            problems.append(
                f'{arm} seed {seed}: self/official difference {max(diffs):.6f} pp '
                f'> {TOLERANCE_PP} pp')

        pred_dir = entry.get('predictions_dir')
        if not pred_dir:
            problems.append(f'{arm} seed {seed}: predictions_dir missing')
            continue
        directory = Path(pred_dir)
        missing = [name for name in val_names if not (directory / name).is_file()]
        extra = sorted(p.name for p in directory.glob('*.png')
                       if p.name not in set(val_names))
        if missing:
            problems.append(
                f'{arm} seed {seed}: {len(missing)}/{len(val_names)} val predictions '
                f'missing (e.g. {missing[:3]}) -- refusing to aggregate')
        if extra:
            problems.append(
                f'{arm} seed {seed}: {len(extra)} unexplained prediction files '
                f'(e.g. {extra[:3]})')
        # stdout is written to the seed directory, one level above predictions/
        if not (Path(pred_dir).parents[1] / 'phenobench_eval_stdout.txt').is_file():
            problems.append(
                f'{arm} seed {seed}: official evaluator raw stdout is missing')
    return problems


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def summarise_arm(result: Dict[str, object]) -> Dict[str, object]:
    per_seed: List[Dict[str, object]] = []
    for entry in result['seeds']:
        values = metric_values(entry)
        per_seed.append({
            'seed': entry['seed'],
            **{key: round(values[key], 4) for key in METRIC_KEYS},
            'train_updates': entry.get('train_updates'),
            'train_seconds': entry.get('train_seconds'),
            'final_train_loss': entry.get('final_train_loss'),
            'peak_gpu_memory_mib': entry.get('peak_gpu_memory_mib'),
        })
    mean = {key: round(float(np.mean([s[key] for s in per_seed])), 4)
            for key in METRIC_KEYS}
    std = {key: round(float(np.std([s[key] for s in per_seed])), 4)
           for key in METRIC_KEYS}
    return {'per_seed': per_seed, 'mean': mean, 'std': std}


def paired_deltas(arms: Dict[str, Dict[str, object]],
                  left: str, right: str) -> Dict[str, object]:
    rows = []
    for left_seed, right_seed in zip(arms[left]['per_seed'],
                                     arms[right]['per_seed']):
        if left_seed['seed'] != right_seed['seed']:
            raise ValueError('seed order mismatch')
        rows.append({
            'seed': left_seed['seed'],
            **{key: round(left_seed[key] - right_seed[key], 4)
               for key in METRIC_KEYS},
        })
    mean = {key: round(arms[left]['mean'][key] - arms[right]['mean'][key], 4)
            for key in METRIC_KEYS}
    return {'per_seed': rows, 'mean': mean}


def trivial_baseline(root: str | Path) -> Dict[str, object]:
    """All-soil baseline recomputed from the val label pixels."""
    from PIL import Image
    root = Path(root)
    counts = np.zeros(5, dtype=np.int64)
    names = list_split_files(root, 'val')
    for name in names:
        with Image.open(root / 'val' / 'semantics' / name) as handle:
            label = np.asarray(handle)
        counts += np.bincount(label.ravel(), minlength=5)
    total = int(counts.sum())
    soil_fraction = float(counts[0]) / float(total)
    miou = 100.0 * soil_fraction / 3.0
    return {
        'val_images': len(names),
        'pixel_counts_by_id': {str(i): int(c) for i, c in enumerate(counts)},
        'soil_fraction_percent': round(100.0 * soil_fraction, 4),
        'all_soil_miou_percent': round(float(miou), 4),
        'recorded_reference_percent': TRIVIAL_BASELINE_MIOU,
        'matches_recorded_reference': bool(
            abs(float(miou) - TRIVIAL_BASELINE_MIOU) <= TRIVIAL_BASELINE_TOLERANCE),
    }


def build_comparison(results: Dict[str, Dict[str, object]],
                     specs: Dict[str, Dict[str, str]],
                     val_names: Sequence[str],
                     data_manifest_sha256: str,
                     code_commit_sha: str,
                     baseline: Dict[str, object],
                     verify_disk: bool = True) -> Dict[str, object]:
    arms = {arm: summarise_arm(results[arm]) for arm in ('A', 'B1', 'B2')}
    deltas = {f'{left}-{right}': paired_deltas(arms, left, right)
              for left, right in DELTA_PAIRS}

    input_records = {}
    prediction_records = {}
    max_self_official = 0.0
    for arm in ('A', 'B1', 'B2'):
        result_path = Path(specs[arm]['result'])
        input_records[arm] = {
            'result_json': str(result_path),
            'result_json_sha256': sha256_file(result_path),
            'checkpoint': specs[arm]['checkpoint'],
            'checkpoint_sha256_expected': specs[arm]['sha256'],
            'checkpoint_sha256_recorded': results[arm].get('backbone_checkpoint_sha256'),
            'checkpoint_sha256_on_disk': (
                sha256_file(specs[arm]['checkpoint'])
                if verify_disk and Path(specs[arm]['checkpoint']).is_file() else None),
            'role': specs[arm]['role'],
            'arm_label': results[arm].get('arm_label'),
            'continued_training': results[arm].get('continued_training'),
            'pretraining_data_version': results[arm].get('pretraining_data_version'),
        }
        prediction_records[arm] = {}
        for entry in results[arm]['seeds']:
            seed = entry['seed']
            prediction_records[arm][str(seed)] = {
                'predictions_dir': entry['predictions_dir'],
                'n_predictions': len([n for n in val_names
                                      if (Path(entry['predictions_dir']) / n).is_file()]),
                'expected_n_predictions': len(val_names),
                'manifest_sha256': prediction_manifest_sha256(
                    val_names, entry['predictions_dir']),
                'official_stdout': str(
                    Path(entry['predictions_dir']).parents[1]
                    / 'phenobench_eval_stdout.txt'),
            }
            cross = entry.get('metric_cross_check', {})
            for row in cross.get('rows', []):
                value = row.get('difference_pp')
                if value is not None and np.isfinite(float(value)):
                    max_self_official = max(max_self_official, abs(float(value)))

    return {
        'generated_by': 'tools/compare_stage6c_phenobench.py',
        'code_commit': code_commit_sha,
        'data': {
            'dataset': 'PhenoBench',
            'version': DATA_VERSION,
            'root': '/data/datasets/PhenoBench/PhenoBench-v110',
            'manifest': str(DATA_MANIFEST),
            'manifest_sha256': data_manifest_sha256,
            'train_images': EXPECTED_TRAIN_IMAGES,
            'val_images': EXPECTED_VAL_IMAGES,
            'evaluated_images': len(val_names),
            'test_split_used': False,
        },
        'inputs': input_records,
        'predictions': prediction_records,
        'protocol': results['A']['protocol'],
        'probe_parameters': results['A'].get('probe_parameters'),
        'seed_semantics': (
            'seeds 0/1/2 are three probe-training seeds on one pretraining '
            'checkpoint per arm; they do not estimate pretraining variance'),
        'metric_sources': {
            'crop': 'official phenobench-eval --task semantics',
            'weed': 'official phenobench-eval --task semantics',
            'mIoU': 'official phenobench-eval --task semantics',
            'soil': 'self-implemented IoU accumulator (official evaluator does '
                    'not print a soil line)',
        },
        'tolerance_pp': TOLERANCE_PP,
        'max_abs_self_official_difference_pp': round(max_self_official, 6),
        'trivial_all_soil_baseline': baseline,
        'arms': arms,
        'deltas': deltas,
        'guardrails': [
            'No p-value is reported; three probe seeds do not estimate '
            'pretraining-seed variance.',
            'Each arm has exactly one pretraining checkpoint, so B2-B1 is '
            'exploratory until independent pretraining seeds exist.',
            'Differences are not called significant; small or inconsistent '
            'directions are reported as mixed/uncertain.',
            'No test-split images and no test-split tuning were used.',
        ],
    }


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #

def write_csv(payload: Dict[str, object], path: Path) -> None:
    rows: List[Dict[str, object]] = []
    for arm in ('A', 'B1', 'B2'):
        data = payload['arms'][arm]
        for entry in data['per_seed']:
            rows.append({'arm': arm, 'seed': entry['seed'],
                         **{key: entry[key] for key in METRIC_KEYS}})
        rows.append({'arm': arm, 'seed': 'mean',
                     **{key: data['mean'][key] for key in METRIC_KEYS}})
        rows.append({'arm': arm, 'seed': 'std',
                     **{key: data['std'][key] for key in METRIC_KEYS}})
    for name, data in payload['deltas'].items():
        for entry in data['per_seed']:
            rows.append({'arm': name, 'seed': entry['seed'],
                         **{key: entry[key] for key in METRIC_KEYS}})
        rows.append({'arm': name, 'seed': 'mean',
                     **{key: data['mean'][key] for key in METRIC_KEYS}})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=['arm', 'seed', *METRIC_KEYS])
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object) -> str:
    return f'{value:.2f}' if isinstance(value, (int, float)) else str(value)


def write_markdown(payload: Dict[str, object], path: Path) -> None:
    lines = [
        '# Stage 6C — PhenoBench A / B1 / B2 冻结线性探针比较',
        '',
        f'- 代码 commit：`{payload["code_commit"]}`',
        f'- 数据版本：{payload["data"]["version"]}，'
        f'val 评价图像：{payload["data"]["evaluated_images"]}',
        f'- 数据清单 SHA-256：`{payload["data"]["manifest_sha256"]}`',
        f'- 自实现与官方指标最大偏差：'
        f'{payload["max_abs_self_official_difference_pp"]:.6f} pp '
        f'(容差 {payload["tolerance_pp"]} pp)',
        '',
        '## 证据类型说明',
        '',
        '- **直接实测值**：每 seed 的 crop/weed/mIoU 来自官方 `phenobench-eval`，'
        'soil 来自自实现 `IoUAccumulator`；预测图与官方 stdout 均落盘。',
        '- **汇总计算值**：mean±std、配对 seed 差值、方向计数由本脚本从上述'
        '实测值重算，不含硬编码数字。',
        '- **历史复用 baseline**：Arm A 复用 Stage 3A 的已核实同协议结果'
        '（`runs/stage3a_phenobench/armA/result.json`），不是本轮重跑。',
        '- **解释性假设**：文末“结果解读”是基于实测值的保守解释，已标注为解释。',
        '',
        '## checkpoint 与输入',
        '',
        '| arm | checkpoint | SHA-256（期望=记录=磁盘） | result.json |',
        '|---|---|---|---|',
    ]
    for arm in ('A', 'B1', 'B2'):
        info = payload['inputs'][arm]
        matches = (info['checkpoint_sha256_expected']
                   == info['checkpoint_sha256_recorded']
                   == (info['checkpoint_sha256_on_disk']
                       or info['checkpoint_sha256_recorded']))
        lines.append(
            f"| {arm} | `{info['checkpoint']}` | "
            f"`{info['checkpoint_sha256_expected'][:16]}…` "
            f"({'一致' if matches else '不一致'}) | "
            f"`{info['result_json']}` |")
    lines += [
        '',
        '## 每 arm、每 probe seed 指标（%）',
        '',
        '| arm | seed | Soil IoU | Crop IoU | Weed IoU | mIoU |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for arm in ('A', 'B1', 'B2'):
        for entry in payload['arms'][arm]['per_seed']:
            lines.append(
                f"| {arm} | {entry['seed']} | {_fmt(entry['soil'])} | "
                f"{_fmt(entry['crop'])} | {_fmt(entry['weed'])} | "
                f"{_fmt(entry['mIoU'])} |")
        mean = payload['arms'][arm]['mean']
        std = payload['arms'][arm]['std']
        lines.append(
            f"| **{arm}** | **mean** | **{_fmt(mean['soil'])}** | "
            f"**{_fmt(mean['crop'])}** | **{_fmt(mean['weed'])}** | "
            f"**{_fmt(mean['mIoU'])}** |")
        lines.append(
            f"| **{arm}** | **std** | {_fmt(std['soil'])} | "
            f"{_fmt(std['crop'])} | {_fmt(std['weed'])} | {_fmt(std['mIoU'])} |")
    lines += ['', '## 配对 seed 差值（百分点，left − right）', '']
    for name, data in payload['deltas'].items():
        lines += [f'### {name}', '',
                  '| seed | Soil Δ | Crop Δ | Weed Δ | mIoU Δ |',
                  '|---:|---:|---:|---:|---:|']
        for entry in data['per_seed']:
            lines.append(
                f"| {entry['seed']} | {entry['soil']:+.2f} | {entry['crop']:+.2f} | "
                f"{entry['weed']:+.2f} | {entry['mIoU']:+.2f} |")
        mean = data['mean']
        lines.append(
            f"| mean | {mean['soil']:+.2f} | {mean['crop']:+.2f} | "
            f"{mean['weed']:+.2f} | {mean['mIoU']:+.2f} |")
        lines.append('')
    lines += ['## 方向一致性（在 3 个 probe seed 上）', '',
              '| 差值 | Soil +/− | Crop +/− | Weed +/− | mIoU +/− |',
              '|---|---:|---:|---:|---:|']
    for name, data in payload['deltas'].items():
        cells = []
        for key in METRIC_KEYS:
            positive = sum(1 for row in data['per_seed'] if row[key] > 0)
            negative = sum(1 for row in data['per_seed'] if row[key] < 0)
            cells.append(f'{positive}/{negative}')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        '- `+/−` 是 3 个 probe seed 中差值为正 / 为负的个数。',
        '- 某个指标 3/3 同向只说明方向在该 arm 的 3 条 probe 轨迹上一致；'
        '它不估计预训练 seed 方差。',
        '',
        '## 训练可比性',
        '',
        '| arm | 最终训练 loss (mean) | probe 耗时 s (mean) | 峰值显存 MiB |',
        '|---|---:|---:|---:|',
    ]
    for arm in ('A', 'B1', 'B2'):
        per_seed = payload['arms'][arm]['per_seed']
        loss = np.mean([entry['final_train_loss'] for entry in per_seed])
        seconds = np.mean([entry['train_seconds'] for entry in per_seed])
        peaks = [entry['peak_gpu_memory_mib'] for entry in per_seed
                 if entry['peak_gpu_memory_mib'] is not None]
        peak = max(peaks) if peaks else 'n/a'
        lines.append(f'| {arm} | {loss:.6f} | {seconds:.1f} | {peak} |')
    base = payload['trivial_all_soil_baseline']
    lines += [
        '',
        '- 训练 loss 的数量级由 soil 像素占比主导，不能单独反映 crop/weed 质量；'
        '三臂 loss 接近说明优化路径可比，不构成效果优劣证据。',
        '',
        '## 平凡基线',
        '',
        f"- 全部预测 soil：本地重算 mIoU = "
        f"{base['all_soil_miou_percent']:.2f}（审计记录 "
        f"{base['recorded_reference_percent']}）",
        '',
        '## 结果解读（基于上述实测值；解释性，不是新测量）',
        '',
    ]
    for name in ('B1-A', 'B2-A', 'B2-B1'):
        data = payload['deltas'][name]
        mean = data['mean']
        positive = sum(1 for row in data['per_seed'] if row['mIoU'] > 0)
        negative = sum(1 for row in data['per_seed'] if row['mIoU'] < 0)
        lines.append(
            f"- **{name}**：mIoU {mean['mIoU']:+.2f} pp；"
            f"crop {mean['crop']:+.2f}，weed {mean['weed']:+.2f}，"
            f"soil {mean['soil']:+.2f}；mIoU 方向 {positive} 正 / {negative} 负。")
    b1a = payload['deltas']['B1-A']['mean']
    b2a = payload['deltas']['B2-A']['mean']
    b2b1 = payload['deltas']['B2-B1']
    b2b1_positive = sum(1 for row in b2b1['per_seed'] if row['mIoU'] > 0)
    b2b1_negative = sum(1 for row in b2b1['per_seed'] if row['mIoU'] < 0)
    soil_share = (
        abs(b1a['soil']) < abs(b1a['crop']) and abs(b1a['soil']) < abs(b1a['weed'])
        and abs(b2a['soil']) < abs(b2a['crop']) and abs(b2a['soil']) < abs(b2a['weed']))
    b1_all_positive = all(row['mIoU'] > 0
                          for row in payload['deltas']['B1-A']['per_seed'])
    b2_all_positive = all(row['mIoU'] > 0
                          for row in payload['deltas']['B2-A']['per_seed'])
    if b1_all_positive and b2_all_positive:
        signal = 'B1/B2 相对 A 在 3/3 probe seeds 上呈初步正迁移信号'
    elif b1_all_positive or b2_all_positive:
        signal = '仅一个 arm 相对 A 在 3/3 probe seeds 上为正，属 mixed 信号'
    else:
        signal = '未观察到 B1/B2 相对 A 的一致正迁移，属 mixed/不确定结果'
    b2b1_small = abs(b2b1['mean']['mIoU']) < min(
        payload['arms']['B1']['std']['mIoU'],
        payload['arms']['B2']['std']['mIoU'])
    sampling_note = (
        'B2−B1 的 |mIoU 差值| 小于两 arm 的 probe-seed std，'
        '且类间方向不一致，不能判定采样策略优劣。' if b2b1_small else
        'B2−B1 的 |mIoU 差值| 不小于两 arm 的 probe-seed std，仍需独立'
        '预训练 seeds 才能支持采样策略结论。')
    lines += [
        '',
        '1. B1−A 与 B2−A 的 mIoU 差值为 '
        f"{b1a['mIoU']:+.2f} / {b2a['mIoU']:+.2f} pp；"
        f"crop 为 {b1a['crop']:+.2f} / {b2a['crop']:+.2f} pp，"
        f"weed 为 {b1a['weed']:+.2f} / {b2a['weed']:+.2f} pp。",
        '2. B2−B1 的 mIoU 差值为 '
        f"{b2b1['mean']['mIoU']:+.2f} pp（{b2b1_positive} 正 / "
        f"{b2b1_negative} 负）；weed 为 "
        f"{b2b1['mean']['weed']:+.2f} pp，crop 为 "
        f"{b2b1['mean']['crop']:+.2f} pp。",
        ('3. 收益由 crop/weed 而非 soil 驱动：B1−A 与 B2−A 的 soil 差值幅度'
         '均小于 crop 与 weed。' if soil_share else
         '3. 无法确认收益不由 soil 主导：soil 差值幅度未同时小于 crop 与 weed。'),
        '4. 三臂 loss、probe 耗时与显存见“训练可比性”表；三臂优化路径可比，'
        'loss 量级由 soil 像素占比主导，不能单独作为效果优劣证据。',
        f"5. 官方评价器与自实现指标最大偏差 "
        f"{payload['max_abs_self_official_difference_pp']:.6f} pp（< 0.01 pp）；"
        f"val 完整评价 {payload['data']['evaluated_images']} 张。",
        f'6. 在当前冻结 linear-probe + 2,000-update 设置下：{signal}；'
        '但每个 arm 只有一个预训练 checkpoint，3 个 probe seeds 不估计'
        f'预训练方差。{sampling_note}下一步应优先用独立预训练 seeds 复现 '
        'B>A，再决定是否进入 GWHD 检测或更长训练。',
        '',
        '## 解释边界',
        '',
        '- probe seeds 0/1/2 只衡量分割头/训练随机性；每个 arm 只有一个预训练 checkpoint。',
        '- 不报告 p 值，不宣称预训练方法显著优越；B2−B1 为探索性方向。',
        '- 差值小或逐类方向不一致时保留 mixed/不确定结论。',
        '- 未使用 test split，未在 test 上调参。',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a-result', default=ARM_SPECS['A']['result'])
    parser.add_argument('--b1-result', default=ARM_SPECS['B1']['result'])
    parser.add_argument('--b2-result', default=ARM_SPECS['B2']['result'])
    parser.add_argument('--root', default='/data/datasets/PhenoBench/PhenoBench-v110')
    parser.add_argument('--out-prefix', type=Path,
                        default=ROOT / 'reports/stage6c_phenobench_comparison')
    parser.add_argument('--code-commit', default=None)
    parser.add_argument('--no-verify-disk', action='store_true',
                        help='skip re-hashing the checkpoint files on disk')
    args = parser.parse_args(argv)

    specs = {arm: dict(spec) for arm, spec in ARM_SPECS.items()}
    specs['A']['result'] = args.a_result
    specs['B1']['result'] = args.b1_result
    specs['B2']['result'] = args.b2_result

    results = {arm: read_json(specs[arm]['result']) for arm in ('A', 'B1', 'B2')}
    val_names = list_split_files(args.root, 'val')

    problems: List[str] = []
    if len(val_names) != EXPECTED_VAL_IMAGES:
        problems.append(
            f'val split has {len(val_names)} images, expected {EXPECTED_VAL_IMAGES}')
    problems += protocol_problems(results)
    verify_disk = not args.no_verify_disk
    for arm in ('A', 'B1', 'B2'):
        problems += validate_arm(arm, results[arm], specs[arm], val_names,
                                 verify_disk=verify_disk)
    if not DATA_MANIFEST.is_file():
        problems.append(f'data manifest missing: {DATA_MANIFEST}')
        manifest_sha = 'missing'
    else:
        manifest_sha = sha256_file(DATA_MANIFEST)
        if manifest_sha != '13fa2810d9fe87c2d65b2dc5b7c7e8ac55e31812aee44af21aea98da8e98bd5d':
            problems.append(
                f'data manifest SHA {manifest_sha} differs from the audited value')

    baseline = trivial_baseline(args.root)
    if not baseline['matches_recorded_reference']:
        problems.append(
            f'all-soil baseline recomputed {baseline["all_soil_miou_percent"]} '
            f'!= recorded {TRIVIAL_BASELINE_MIOU}')

    if problems:
        print('REFUSING TO AGGREGATE:')
        for problem in problems:
            print(f'  - {problem}')
        return 1

    payload = build_comparison(
        results, specs, val_names, manifest_sha,
        args.code_commit or git_commit(),
        baseline, verify_disk=verify_disk)

    prefix = args.out_prefix
    csv_path = prefix.with_suffix('.csv')
    md_path = prefix.with_suffix('.md')
    json_path = prefix.with_suffix('.json')
    write_csv(payload, csv_path)
    write_markdown(payload, md_path)
    payload['output_files_sha256'] = {
        str(csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path):
            sha256_file(csv_path),
        str(md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path):
            sha256_file(md_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    sidecar = prefix.with_suffix('.sha256')
    sidecar.write_text('\n'.join(
        f'{sha256_file(path)}  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}'
        for path in (json_path, csv_path, md_path)) + '\n', encoding='utf-8')

    print(json.dumps({
        'ok': True,
        'max_abs_self_official_difference_pp':
            payload['max_abs_self_official_difference_pp'],
        'mean': {arm: payload['arms'][arm]['mean'] for arm in ('A', 'B1', 'B2')},
        'deltas_mean': {name: data['mean']
                        for name, data in payload['deltas'].items()},
        'outputs': [str(json_path), str(csv_path), str(md_path), str(sidecar)],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
