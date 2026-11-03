"""Stage 6C PhenoBench comparison tests.

These tests exercise the comparison tool on synthetic ``result.json`` fixtures.
They do not read the real runs, so they are fast and deterministic; the real
artefacts are validated by running the tool itself.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.compare_stage6c_phenobench import (
    EXPECTED_SEEDS,
    METRIC_KEYS,
    TOLERANCE_PP,
    build_comparison,
    metric_values,
    paired_deltas,
    protocol_problems,
    summarise_arm,
    trivial_baseline,
    validate_arm,
    write_csv,
    write_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


def make_protocol(**overrides):
    protocol = {
        'backbone': 'ViT-H/14, frozen, 224x224 tiles, 16x16 tokens, embed 1280',
        'head': 'Linear(1280 -> 3) over patch tokens, nearest upsample x14',
        'tiling': ('overlapping 224x224, stride 200, 5x5 per 1024 image, '
                   'no padding, every pixel covered, overlaps logit-averaged'),
        'optimizer': 'AdamW',
        'lr': 0.001,
        'weight_decay': 0.0,
        'epochs': 10,
        'batch_size': 32,
        'seeds': [0, 1, 2],
        'loss': 'patch cross-entropy, exact equivalent of nearest-upsampled pixel CE',
        'train_images': 1407,
        'val_images': 772,
    }
    protocol.update(overrides)
    return protocol


def make_result(arm: str, metrics, out_dir: Path, val_names, *,
                protocol=None, checkpoint_sha='deadbeef',
                cross_agrees=True, cross_diff=0.004,
                seeds=(0, 1, 2), pred_override=None):
    """Build a result.json plus prediction folders that validate_arm accepts."""
    protocol = protocol or make_protocol()
    per_seed = []
    for position, seed in enumerate(seeds):
        soil, crop, weed, miou = metrics[position]
        pred_dir = out_dir / f'seed{seed}' / 'predictions' / 'semantics'
        pred_dir.mkdir(parents=True, exist_ok=True)
        for name in val_names:
            (pred_dir / name).write_bytes(b'\x00')
        if pred_override:
            pred_override(pred_dir, val_names)
        (pred_dir.parents[1] / 'phenobench_eval_stdout.txt').write_text(
            'IoU (crop): 0\n', encoding='utf-8')
        per_seed.append({
            'seed': seed,
            'train_updates': 11000,
            'final_train_loss': 0.0644,
            'train_seconds': 70.0,
            'peak_gpu_memory_mib': 120.0,
            'self_metrics': {
                'per_class_iou_percent': {
                    'soil': soil, 'crop': crop, 'weed': weed},
                'mIoU_percent': (soil + crop + weed) / 3.0,
            },
            'official_metrics': {
                'IoU (crop)': crop, 'IoU (weed)': weed, 'mIoU': miou},
            'metric_cross_check': {
                'rows': [
                    {'metric': 'crop', 'self_percent': crop,
                     'official_percent': crop, 'difference_pp': cross_diff},
                    {'metric': 'weed', 'self_percent': weed,
                     'official_percent': weed, 'difference_pp': -cross_diff},
                    {'metric': 'mIoU', 'self_percent': miou,
                     'official_percent': miou, 'difference_pp': cross_diff},
                ],
                'max_abs_difference_pp': cross_diff,
                'tolerance_pp': TOLERANCE_PP,
                'agrees': cross_agrees,
            },
            'predictions_dir': str(pred_dir),
        })
    payload = {
        'arm': arm,
        'arm_label': f'stage6c_{arm.lower()}',
        'cache_key': f'{arm}__deadbeef',
        'backbone_checkpoint': f'/tmp/{arm}.pth',
        'backbone_checkpoint_sha256': checkpoint_sha,
        'continued_training': 'test',
        'pretraining_data_version': 'test',
        'protocol': protocol,
        'probe_parameters': 3843,
        'seeds': per_seed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'result.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    return payload


VAL_NAMES = ['val_0000.png', 'val_0001.png', 'val_0002.png', 'val_0003.png']


def make_spec(arm: str, result_path: Path, sha: str) -> dict:
    return {
        'checkpoint': f'/tmp/{arm}.pth',
        'sha256': sha,
        'result': str(result_path),
        'role': 'test',
    }


def test_metric_values_prefers_official_for_three_classes_and_self_soil():
    entry = {
        'self_metrics': {'per_class_iou_percent': {
            'soil': 97.6, 'crop': 81.0, 'weed': 35.0}},
        'official_metrics': {
            'IoU (crop)': 81.5, 'IoU (weed)': 35.5, 'mIoU': 71.7},
    }
    values = metric_values(entry)
    assert values == {'soil': 97.6, 'crop': 81.5, 'weed': 35.5, 'mIoU': 71.7}


def test_protocol_problems_accepts_identical_and_flags_lr():
    base = make_result('A', [(97, 81, 35, 71)] * 3, Path('/tmp/none'), VAL_NAMES)
    same = make_result('B1', [(97, 81, 35, 71)] * 3, Path('/tmp/none'), VAL_NAMES)
    assert protocol_problems({'A': base, 'B1': same, 'B2': same}) == []

    altered = make_result('B2', [(97, 81, 35, 71)] * 3, Path('/tmp/none'),
                          VAL_NAMES, protocol=make_protocol(lr=0.0003))
    problems = protocol_problems({'A': base, 'B1': same, 'B2': altered})
    assert any('protocol.lr differs' in p for p in problems)


def test_validate_arm_accepts_good_fixture(tmp_path):
    payload = make_result('A', [(97, 81, 35, 71)] * 3, tmp_path, VAL_NAMES,
                          checkpoint_sha='abc123')
    spec = make_spec('A', tmp_path / 'result.json', 'abc123')
    assert validate_arm('A', payload, spec, VAL_NAMES, verify_disk=False) == []


def test_validate_arm_rejects_wrong_checkpoint_sha(tmp_path):
    payload = make_result('A', [(97, 81, 35, 71)] * 3, tmp_path, VAL_NAMES,
                          checkpoint_sha='wrong')
    spec = make_spec('A', tmp_path / 'result.json', 'expected')
    problems = validate_arm('A', payload, spec, VAL_NAMES, verify_disk=False)
    assert any('checkpoint SHA' in p for p in problems)


def test_validate_arm_rejects_missing_seed(tmp_path):
    payload = make_result('A', [(97, 81, 35, 71)] * 2, tmp_path, VAL_NAMES,
                          checkpoint_sha='abc123', seeds=(0, 1))
    spec = make_spec('A', tmp_path / 'result.json', 'abc123')
    problems = validate_arm('A', payload, spec, VAL_NAMES, verify_disk=False)
    assert any('probe seeds' in p for p in problems)


def test_validate_arm_rejects_missing_prediction(tmp_path):
    def drop_one(pred_dir, names):
        (pred_dir / names[0]).unlink()

    payload = make_result('B1', [(97, 81, 35, 71)] * 3, tmp_path, VAL_NAMES,
                          checkpoint_sha='abc123', pred_override=drop_one)
    spec = make_spec('B1', tmp_path / 'result.json', 'abc123')
    problems = validate_arm('B1', payload, spec, VAL_NAMES, verify_disk=False)
    assert any('refusing to aggregate' in p for p in problems)


def test_validate_arm_rejects_self_official_disagreement(tmp_path):
    payload = make_result('B2', [(97, 81, 35, 71)] * 3, tmp_path, VAL_NAMES,
                          checkpoint_sha='abc123', cross_agrees=False,
                          cross_diff=0.5)
    spec = make_spec('B2', tmp_path / 'result.json', 'abc123')
    problems = validate_arm('B2', payload, spec, VAL_NAMES, verify_disk=False)
    assert any('cross-check' in p for p in problems)
    assert any('difference' in p for p in problems)


def test_summarise_and_paired_deltas():
    arms = {}
    for arm, offset in (('A', 0.0), ('B1', 1.0), ('B2', 0.5)):
        payload = {
            'seeds': [
                {
                    'seed': seed,
                    'self_metrics': {'per_class_iou_percent': {
                        'soil': 90 + offset, 'crop': 80 + offset,
                        'weed': 30 + offset}},
                    'official_metrics': {
                        'IoU (crop)': 80 + offset,
                        'IoU (weed)': 30 + offset,
                        'mIoU': 70 + offset},
                }
                for seed in EXPECTED_SEEDS
            ]
        }
        arms[arm] = summarise_arm(payload)
    assert arms['B1']['mean']['mIoU'] == pytest.approx(71.0)
    assert arms['A']['std']['mIoU'] == 0.0

    delta = paired_deltas(arms, 'B1', 'A')
    assert delta['mean']['mIoU'] == pytest.approx(1.0)
    assert [row['mIoU'] for row in delta['per_seed']] == [1.0, 1.0, 1.0]
    delta2 = paired_deltas(arms, 'B2', 'B1')
    assert delta2['mean']['mIoU'] == pytest.approx(-0.5)


def test_build_comparison_and_writers(tmp_path):
    val_names = VAL_NAMES
    metrics = {
        'A': [(97.6, 81.6, 38.7, 72.6)] * 3,
        'B1': [(97.7, 81.9, 39.0, 72.9)] * 3,
        'B2': [(97.5, 81.2, 38.0, 72.2)] * 3,
    }
    results, specs = {}, {}
    for arm in ('A', 'B1', 'B2'):
        out = tmp_path / arm
        payload = make_result(arm, metrics[arm], out, val_names,
                              checkpoint_sha=f'sha_{arm}')
        results[arm] = payload
        specs[arm] = make_spec(arm, out / 'result.json', f'sha_{arm}')

    baseline = {
        'all_soil_miou_percent': 29.97,
        'recorded_reference_percent': 29.97,
        'matches_recorded_reference': True,
    }
    payload = build_comparison(
        results, specs, val_names, 'manifest_sha', 'commit_sha', baseline,
        verify_disk=False)

    assert payload['deltas']['B1-A']['mean']['mIoU'] == pytest.approx(0.3)
    assert payload['deltas']['B2-B1']['mean']['mIoU'] == pytest.approx(-0.7)
    assert payload['max_abs_self_official_difference_pp'] <= TOLERANCE_PP
    assert payload['predictions']['A']['0']['n_predictions'] == len(val_names)
    assert payload['data']['train_images'] == 1407
    assert payload['data']['val_images'] == 772
    assert 'No p-value' in payload['guardrails'][0]
    assert 'do not estimate pretraining variance' in payload['seed_semantics']

    csv_path = tmp_path / 'out.csv'
    md_path = tmp_path / 'out.md'
    write_csv(payload, csv_path)
    write_markdown(payload, md_path)

    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ['arm', 'seed', *METRIC_KEYS]
    assert any(row['arm'] == 'B2-B1' and row['seed'] == 'mean' for row in rows)
    assert md_path.read_text(encoding='utf-8').count('B2-B1') >= 1


def test_explicit_checkpoint_never_falls_back_to_legacy_arm(tmp_path):
    from src.downstream.features import cache_key
    from src.downstream.phenobench_probe import resolve_arm_config

    checkpoint = tmp_path / 'jepa-encoder-step2000.pth'
    checkpoint.write_bytes(b'not a real checkpoint, only a hash target')
    cfg = resolve_arm_config(
        None, str(checkpoint), 'stage6c_b1_natural',
        continued_training='2000 updates', data_version='R3v3')
    assert cfg['arm'] == 'stage6c_b1_natural'
    assert cfg['label'] == 'stage6c_b1_natural'
    assert cfg['checkpoint'] == str(checkpoint)
    assert cfg['checkpoint_sha256'] != ''

    # A cache key is bound to both the label and the weight hash.
    first = cache_key('stage6c_b1_natural', 'a' * 64)
    second = cache_key('stage6c_b1_natural', 'b' * 64)
    third = cache_key('stage6c_b2_capped_no_backfill', 'a' * 64)
    assert len({first, second, third}) == 3


def test_trivial_baseline_recomputes_from_labels(tmp_path):
    root = tmp_path / 'PhenoBench'
    (root / 'val' / 'images').mkdir(parents=True)
    (root / 'val' / 'semantics').mkdir(parents=True)
    name = 'val_0000.png'
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
        root / 'val' / 'images' / name)
    label = np.zeros((10, 10), dtype=np.uint8)
    label[0, :] = 1  # 10 / 100 crop pixels -> soil fraction 0.9
    Image.fromarray(label).save(root / 'val' / 'semantics' / name)
    baseline = trivial_baseline(root)
    assert baseline['val_images'] == 1
    assert baseline['soil_fraction_percent'] == pytest.approx(90.0)
    assert baseline['all_soil_miou_percent'] == pytest.approx(30.0)
