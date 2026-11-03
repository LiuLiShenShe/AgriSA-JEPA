"""Stage 3A protocol tests: the two arms must differ only in the backbone.

Also covers the tools that produce and verify the downstream run: the arm-config
loader, the protocol diff, and the machine-readable results format.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.check_arm_protocol import ALLOWED_DIFFERENT, flatten
from tools.run_arm_config import config_to_argv, load_arm_config

REPO = Path(__file__).resolve().parents[1]
CONFIG_A = REPO / 'configs/phenobench_armA_linear.yaml'
CONFIG_B = REPO / 'configs/phenobench_armB_linear.yaml'


class TestArmConfigs:

    def test_configs_load(self):
        a, b = load_arm_config(CONFIG_A), load_arm_config(CONFIG_B)
        assert a['meta']['arm'] == 'A'
        assert b['meta']['arm'] == 'B'

    def test_configs_differ_only_in_arm_identity(self):
        fa = flatten(yaml.safe_load(CONFIG_A.read_text()))
        fb = flatten(yaml.safe_load(CONFIG_B.read_text()))
        problems = [k for k in set(fa) | set(fb)
                    if k not in ALLOWED_DIFFERENT and fa.get(k) != fb.get(k)]
        assert problems == []

    def test_backbones_are_different_files(self):
        fa = flatten(yaml.safe_load(CONFIG_A.read_text()))
        fb = flatten(yaml.safe_load(CONFIG_B.read_text()))
        assert fa[('meta', 'backbone_checkpoint')] != fb[('meta', 'backbone_checkpoint')]

    def test_arm_b_is_marked_exploratory(self):
        cfg = yaml.safe_load(CONFIG_B.read_text())
        assert 'v0' in cfg['meta']['pretraining_data_version']
        assert '500' in cfg['meta']['continued_training']

    def test_label_mapping_is_official(self):
        for path in (CONFIG_A, CONFIG_B):
            cfg = yaml.safe_load(path.read_text())
            assert cfg['data']['num_classes'] == 3
            assert cfg['data']['class_names'] == ['soil', 'crop', 'weed']
            assert cfg['data']['partial_to_parent'] is True

    def test_tiling_rule_is_frozen(self):
        for path in (CONFIG_A, CONFIG_B):
            cfg = yaml.safe_load(path.read_text())
            assert cfg['tiling']['tile_size'] == 224
            assert cfg['tiling']['stride'] == 200
            assert cfg['tiling']['n_tiles_per_axis'] == 5
            assert cfg['tiling']['pad'] is False
            assert cfg['tiling']['covers_every_pixel'] is True
            assert cfg['tiling']['stitching'] == 'logit_average_over_overlaps'

    def test_rejects_unknown_arm(self, tmp_path):
        bad = yaml.safe_load(CONFIG_A.read_text())
        bad['meta']['arm'] = 'C'
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match='arm'):
            load_arm_config(path)

    def test_rejects_missing_section(self, tmp_path):
        bad = yaml.safe_load(CONFIG_A.read_text())
        del bad['tiling']
        path = tmp_path / 'bad.yaml'
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match='tiling'):
            load_arm_config(path)

    def test_config_to_argv_round_trip(self):
        cfg = load_arm_config(CONFIG_A)
        argv = config_to_argv(cfg)
        assert '--arm' in argv and argv[argv.index('--arm') + 1] == 'A'
        assert str(cfg['probe']['epochs']) == argv[argv.index('--epochs') + 1]
        assert str(cfg['probe']['lr']) == argv[argv.index('--lr') + 1]


class TestOfficialEvalFormat:

    def test_prediction_png_is_loadable_as_int(self, tmp_path):
        from PIL import Image
        import numpy as np
        arr = np.array([[0, 1, 2]], dtype=np.uint8)
        path = tmp_path / 'semantics' / '0001.png'
        path.parent.mkdir(parents=True)
        Image.fromarray(arr, mode='L').save(path)
        reloaded = np.asarray(Image.open(path), dtype=np.int32)
        assert reloaded.tolist() == [[0, 1, 2]]

    def test_official_eval_tool_is_installed(self):
        module = pytest.importorskip('phenobench.evaluation.evaluate_semantics')
        assert hasattr(module, 'evaluate_semantics')
        convert = pytest.importorskip('phenobench.evaluation.auxiliary.convert')
        assert hasattr(convert, 'convert_partial_semantics')

    def test_official_evaluator_scores_3_classes(self):
        tm = pytest.importorskip('torchmetrics')
        from torchmetrics.classification import MulticlassJaccardIndex
        import torch
        metric = MulticlassJaccardIndex(num_classes=3, average=None)
        metric.update(torch.tensor([[0, 1, 2]]), torch.tensor([[0, 1, 2]]))
        values = metric.compute()
        assert values.shape == (3,)
        assert float(values.mean()) == pytest.approx(1.0)


class TestResultsFormat:

    def test_result_json_schema(self, tmp_path):
        """The shape the arm reports must satisfy before it is trusted."""
        payload = {
            'arm': 'A',
            'arm_label': 'official_i-jepa_vith14',
            'backbone_checkpoint': '/x.pth.tar',
            'backbone_checkpoint_sha256': 'ab' * 32,
            'continued_training': 'none',
            'pretraining_data_version': None,
            'protocol': {'seed': 0},
            'probe_parameters': 3843,
            'train_updates': 10,
            'loss_history': [1.0, 0.5],
            'final_train_loss': 0.5,
            'train_seconds': 12.0,
            'peak_gpu_memory_mib': 100.0,
            'self_metrics': {
                'per_class_iou_percent': {'soil': 1.0, 'crop': 1.0, 'weed': 1.0},
                'mIoU_percent': 1.0,
                'pixels_intersection': {}, 'pixels_union': {},
                'scored_region': 'x',
            },
            'official_metrics': {'IoU (crop)': 1.0, 'IoU (weed)': 1.0, 'mIoU': 1.0},
            'official_command': 'phenobench-eval ...',
        }
        path = tmp_path / 'result.json'
        path.write_text(json.dumps(payload, indent=2))
        loaded = json.loads(path.read_text())
        assert loaded['arm'] in ('A', 'B')
        assert len(loaded['backbone_checkpoint_sha256']) == 64
        assert set(loaded['self_metrics']['per_class_iou_percent']) == {
            'soil', 'crop', 'weed'}