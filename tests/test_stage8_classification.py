"""Stage 8 tests: classification probe and its data adapters.

Covers the shared classifier head arithmetic (mean pooling, shapes, metric
computation), and the DeepWeeds fold-0 / PlantDoc 28-class split logic using
small in-repo fixtures rather than the multi-GB image sets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.downstream.classifier_probe import (LinearClassificationProbe,
                                              evaluate_classifier,
                                              train_classifier)
from src.downstream.tiling import TILE_SIZE

ROOT = Path(__file__).resolve().parent.parent


def test_probe_pools_and_classifies():
    probe = LinearClassificationProbe(num_classes=9)
    tokens = torch.randn(7, 256, 1280)
    out = probe(tokens)
    assert out.shape == (7, 9)
    with pytest.raises(ValueError):
        probe(torch.randn(7, 64, 1280))


def test_probe_rejects_2d_input():
    probe = LinearClassificationProbe(num_classes=2)
    with pytest.raises(ValueError):
        probe(torch.randn(7, 1280))


def test_train_classifier_reduces_loss_on_separable_data():
    torch.manual_seed(0)
    base = torch.randn(4, 256, 1280)
    # class 0 tokens shifted +1, class 1 shifted -1 -> separable by mean pool
    tokens = torch.cat([base + 1.0, base - 1.0], dim=0)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    probe = LinearClassificationProbe(num_classes=2)
    res = train_classifier(probe, tokens, labels, epochs=30, batch_size=4,
                           lr=1e-2, seed=0, device=torch.device('cpu'))
    assert res['loss_history'][-1] < res['loss_history'][0]
    metrics = evaluate_classifier(probe, tokens, labels, torch.device('cpu'))
    assert metrics['accuracy'] == 1.0
    assert metrics['macro_f1'] == 1.0


def test_evaluate_classifier_confusion_counts():
    probe = LinearClassificationProbe(num_classes=2)
    with torch.no_grad():
        probe.head.weight.zero_()
        probe.head.bias.zero_()          # logits all 0 -> argmax = class 0
    tokens = torch.randn(10, 256, 1280)
    labels = torch.tensor([0] * 5 + [1] * 5)
    metrics = evaluate_classifier(probe, tokens, labels, torch.device('cpu'))
    # all predictions class 0 -> recall(0)=1, recall(1)=0, acc=0.5
    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['per_class_recall'][0] == 1.0
    assert metrics['per_class_recall'][1] == 0.0
    assert metrics['per_class_f1'][1] == 0.0


def test_deepweeds_split_from_fixture(tmp_path: Path):
    from src.downstream.deepweeds_probe import build_split, read_names
    root = tmp_path
    names = [f'im{i}.jpg' for i in range(20)]
    for part, idx in (('train_subset0.csv', range(10)),
                      ('val_subset0.csv', range(10, 15)),
                      ('test_subset0.csv', range(15, 20))):
        with (root / part).open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['Filename', 'Label'])
            for i in idx:
                w.writerow([names[i], i % 9])
    with (root / 'labels.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Filename', 'Label', 'Species'])
        for i, n in enumerate(names):
            w.writerow([n, i % 9, 's'])
    split = build_split(root)
    assert len(split['train']) == 10 and len(split['val']) == 5
    assert len(split['test']) == 5
    # duplicate coverage error path
    with (root / 'labels.csv').open('a', newline='') as fh:
        fh.write('extra.jpg,0,x\n')
    with pytest.raises(ValueError):
        build_split(root)


def test_plantdoc_split_from_fixture(tmp_path: Path):
    from src.downstream.plantdoc_probe import build_split
    root = tmp_path
    for split_name in ('train', 'test'):
        for c in ('classA', 'classB'):
            d = root / split_name / c
            d.mkdir(parents=True)
            for i in range(10 if split_name == 'train' else 2):
                (d / f'img{i}.jpg').write_bytes(b'x')
    split = build_split(root)
    assert len(split['test']) == 4
    # 15% of 10 per class, min 1 -> 1 or 2 per class to val
    assert 2 <= len(split['val']) <= 4
    assert len(split['train']) + len(split['val']) == 20
    # no image in two splits
    assert not (set(split['train']) & set(split['val']))
    assert not (set(split['train']) & set(split['test']))


def test_plantdoc_split_test_subset_of_train(tmp_path: Path):
    from src.downstream.plantdoc_probe import build_split
    root = tmp_path
    (root / 'train' / 'classA').mkdir(parents=True)
    (root / 'train' / 'classB').mkdir(parents=True)   # B only in train
    (root / 'test' / 'classA').mkdir(parents=True)
    for d in (root / 'train' / 'classA', root / 'train' / 'classB',
              root / 'test' / 'classA'):
        for i in range(4):
            (d / f'img{i}.jpg').write_bytes(b'x')
    split = build_split(root)          # test covers only classA -> permitted
    assert any('classA' in n for n in split['test'])


def test_plantdoc_split_rejects_test_only_class(tmp_path: Path):
    from src.downstream.plantdoc_probe import build_split
    root = tmp_path
    (root / 'train' / 'classA').mkdir(parents=True)
    (root / 'test' / 'classA').mkdir(parents=True)
    (root / 'test' / 'classZ').mkdir(parents=True)   # only in test -> error
    for d in (root / 'train' / 'classA', root / 'test' / 'classA',
              root / 'test' / 'classZ'):
        for i in range(4):
            (d / f'img{i}.jpg').write_bytes(b'x')
    with pytest.raises(ValueError):
        build_split(root)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))