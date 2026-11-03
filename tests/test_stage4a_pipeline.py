"""Stage 3B / 4A pipeline tests: image cache, reproducibility, snapshots, caches.

These complement ``test_stage4a_masks.py`` (the mask algorithm) and
``test_stage3a_downstream.py`` (the evaluation protocol).  Here we cover the
plumbing the protocol adds in this round:

* the NVMe image cache is pixel-equivalent to decoding the source files and
  refuses a manifest it was not built from (protocol §12, §13.2 #25),
* single-process loading is bit-reproducible (protocol §6.3),
* encoder snapshots record the provenance needed to audit an evaluation point
  (protocol §4.2),
* feature caches are bound to the checkpoint SHA-256, so one arm cannot silently
  reuse another arm's features (protocol §10, §12, §13.3 #27/#28).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CONFIGS = [
    'configs/stage3b_random_2000.yaml',
    'configs/stage4a_vegetation_2000.yaml',
    'configs/stage4a_boundary_2000.yaml',
    'configs/stage4a_shuffled_2000.yaml',
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# synthetic dataset with a real on-disk manifest
# --------------------------------------------------------------------------- #

def make_synthetic_dataset(tmp_path: Path, n: int = 4, side: int = 64):
    """Write ``n`` tiny RGB images plus the manifest columns the loader needs."""
    root = tmp_path / 'images'
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for index in range(n):
        array = rng.integers(0, 256, size=(side, side, 3), dtype=np.uint8)
        relative = f'img_{index:03d}.png'
        Image.fromarray(array, mode='RGB').save(root / relative)
        rows.append({
            'sample_id': f'sample_{index:03d}',
            'path': relative,
            'split': 'train',
            'status': 'include',
            'sha256': sha256(root / relative),
        })
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest, root


def build_cache(manifest: Path, root: Path, out: Path, side: int = 64) -> None:
    command = [sys.executable, '-m', 'tools.build_image_cache',
               '--manifest', str(manifest), '--root', str(root),
               '--out', str(out), '--size', str(side)]
    proc = subprocess.run(command, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- #
# image cache
# --------------------------------------------------------------------------- #

class TestImageCache:

    def test_cache_is_pixel_equivalent_to_disk(self, tmp_path):
        manifest, root = make_synthetic_dataset(tmp_path)
        cache = tmp_path / 'cache'
        build_cache(manifest, root, cache)

        from src.datasets.agri_dataset import make_agrifield40k
        from src.transforms import make_transforms

        def loader(cache_dir):
            transform = make_transforms(crop_size=side, crop_scale=(0.5, 1.0),
                                        color_jitter=0.0, horizontal_flip=False,
                                        color_distortion=False, gaussian_blur=False)
            return make_agrifield40k(
                transform, 2, None, False, 0, 1, 0,
                manifest_path=str(manifest), root_path=str(root), split='train',
                drop_last=False, seed=7, augmentation_seed=11,
                image_cache_dir=cache_dir)[1]

        side = 64
        from_cache = [batch for batch in loader(str(cache))]
        from_disk = [batch for batch in loader(None)]
        assert len(from_cache) == len(from_disk)
        for cached, disk in zip(from_cache, from_disk):
            assert torch.equal(cached[0], disk[0])
            assert list(cached[1]) == list(disk[1])

    def test_cache_index_records_manifest_sha(self, tmp_path):
        manifest, root = make_synthetic_dataset(tmp_path)
        cache = tmp_path / 'cache'
        build_cache(manifest, root, cache)
        index = json.loads((cache / 'index.json').read_text(encoding='utf-8'))
        assert index['manifest_sha256'] == sha256(manifest)
        assert index['count'] == index['requested'] == 4
        assert index['sha256_verified'] is True
        assert index['failures'] == []

    def test_cache_rejects_other_manifest(self, tmp_path):
        manifest, root = make_synthetic_dataset(tmp_path)
        cache = tmp_path / 'cache'
        build_cache(manifest, root, cache)
        index_path = cache / 'index.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        index['manifest_sha256'] = '0' * 64
        index_path.write_text(json.dumps(index), encoding='utf-8')

        from src.datasets.agri_dataset import AgriFieldDataset
        with pytest.raises(ValueError, match='different manifest'):
            AgriFieldDataset(manifest_path=str(manifest), split='train',
                             root_path=str(root), image_cache_dir=str(cache))

    def test_cache_rejects_diverged_sample_order(self, tmp_path):
        manifest, root = make_synthetic_dataset(tmp_path)
        cache = tmp_path / 'cache'
        build_cache(manifest, root, cache)
        index_path = cache / 'index.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        index['sample_ids'] = list(reversed(index['sample_ids']))
        index_path.write_text(json.dumps(index), encoding='utf-8')

        from src.datasets.agri_dataset import AgriFieldDataset
        with pytest.raises(ValueError, match='sample order diverges'):
            AgriFieldDataset(manifest_path=str(manifest), split='train',
                             root_path=str(root), image_cache_dir=str(cache))

    def test_cache_refuses_unverified_build(self, tmp_path):
        manifest, root = make_synthetic_dataset(tmp_path)
        cache = tmp_path / 'cache'
        build_cache(manifest, root, cache)
        index_path = cache / 'index.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        index['sha256_verified'] = False
        index_path.write_text(json.dumps(index), encoding='utf-8')

        from src.datasets.agri_dataset import AgriFieldDataset
        with pytest.raises(ValueError, match='without per-file sha256'):
            AgriFieldDataset(manifest_path=str(manifest), split='train',
                             root_path=str(root), image_cache_dir=str(cache))

    def test_incomplete_cache_directory_rejected(self, tmp_path):
        from src.datasets.agri_dataset import AgriFieldDataset
        empty = tmp_path / 'empty'
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match='incomplete'):
            AgriFieldDataset(manifest_path=str(REPO / 'data/frozen/agrifield40k-v1_train.csv'),
                             split='train', image_cache_dir=str(empty))

    def test_cache_serves_a_manifest_prefix(self, tmp_path):
        manifest, root = make_synthetic_dataset(tmp_path)
        cache = tmp_path / 'cache'
        build_cache(manifest, root, cache)
        from src.datasets.agri_dataset import AgriFieldDataset
        dataset = AgriFieldDataset(manifest_path=str(manifest), split='train',
                                   root_path=str(root), image_cache_dir=str(cache),
                                   max_samples=2)
        assert len(dataset) == 2


# --------------------------------------------------------------------------- #
# reproducibility of the data / mask streams
# --------------------------------------------------------------------------- #

class TestReproducibility:

    @staticmethod
    def _collate_pass(seed: int, batches: int = 3):
        from src.masks.multiblock import MaskCollator
        collator = MaskCollator(input_size=224, patch_size=14,
                                pred_mask_scale=(0.15, 0.2),
                                enc_mask_scale=(0.85, 1.0),
                                aspect_ratio=(0.75, 1.5),
                                nenc=1, npred=4, allow_overlap=False, min_keep=10)
        torch.manual_seed(seed)
        out = []
        for _ in range(batches):
            batch = [(torch.zeros(3, 224, 224), str(i)) for i in range(4)]
            _, enc, pred = collator(batch)
            out.append((enc[0].clone(), [p.clone() for p in pred]))
        return out

    def test_single_process_collation_is_reproducible(self):
        first = self._collate_pass(1234)
        second = self._collate_pass(1234)
        for (enc_a, pred_a), (enc_b, pred_b) in zip(first, second):
            assert torch.equal(enc_a, enc_b)
            for pa, pb in zip(pred_a, pred_b):
                assert torch.equal(pa, pb)

    def test_different_seed_changes_the_masks(self):
        first = self._collate_pass(1234)
        second = self._collate_pass(4321)
        assert not all(torch.equal(a[0], b[0]) for a, b in zip(first, second))

    def test_augmentation_seed_is_per_sample_and_epoch(self, tmp_path):
        """The same (sample, epoch) must give the same crop in any order."""
        manifest, root = make_synthetic_dataset(tmp_path)
        from src.datasets.agri_dataset import AgriFieldDataset
        from src.transforms import make_transforms

        transform = make_transforms(crop_size=32, crop_scale=(0.4, 1.0),
                                    color_jitter=0.0, horizontal_flip=False,
                                    color_distortion=False, gaussian_blur=False)
        dataset = AgriFieldDataset(manifest_path=str(manifest), transform=transform,
                                   split='train', root_path=str(root),
                                   augmentation_seed=99)
        dataset.set_epoch(3)
        first = dataset[1][0]
        _ = dataset[0][0]          # touching another sample must not shift it
        _ = dataset[2][0]
        second = dataset[1][0]
        assert torch.equal(first, second)

        dataset.set_epoch(4)
        third = dataset[1][0]
        assert not torch.equal(first, third)


# --------------------------------------------------------------------------- #
# training configuration guarantees
# --------------------------------------------------------------------------- #

class TestTrainingConfiguration:

    def test_all_arms_load_in_a_single_process(self):
        for name in CONFIGS:
            cfg = yaml.safe_load((REPO / name).read_text(encoding='utf-8'))
            assert cfg['data']['num_workers'] == 0, name

    def test_all_arms_share_the_same_image_cache(self):
        values = set()
        for name in CONFIGS:
            cfg = yaml.safe_load((REPO / name).read_text(encoding='utf-8'))
            values.add(cfg['data'].get('image_cache_dir'))
        assert len(values) == 1
        assert None not in values

    def test_save_points_lie_inside_the_budget(self):
        for name in CONFIGS:
            cfg = yaml.safe_load((REPO / name).read_text(encoding='utf-8'))
            opt = cfg['optimization']
            budget = opt['max_optimizer_steps']
            saves = opt['save_at_optimizer_steps']
            assert saves, name
            assert all(0 < step < budget for step in saves), name

    def test_every_arm_starts_from_the_official_checkpoint(self):
        shas = set()
        for name in CONFIGS:
            cfg = yaml.safe_load((REPO / name).read_text(encoding='utf-8'))
            assert cfg['meta']['load_checkpoint'] is False
            shas.add(cfg['meta']['pretrained_sha256'])
            assert cfg['meta']['pretrained_checkpoint'].endswith('IN1K-vit.h.14-300e.pth.tar')
        assert len(shas) == 1

    def test_pretrained_sha_matches_the_file_on_disk(self):
        cfg = yaml.safe_load((REPO / CONFIGS[0]).read_text(encoding='utf-8'))
        checkpoint = Path(cfg['meta']['pretrained_checkpoint'])
        if not checkpoint.is_file():
            pytest.skip(f'official checkpoint not present: {checkpoint}')
        assert sha256(checkpoint) == cfg['meta']['pretrained_sha256']

    def test_protocol_checker_passes(self):
        proc = subprocess.run([sys.executable, '-m', 'tools.check_stage3b4a_protocol'],
                              capture_output=True, text=True, cwd=REPO)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert 'PROTOCOL OK' in proc.stdout


# --------------------------------------------------------------------------- #
# encoder snapshot provenance
# --------------------------------------------------------------------------- #

class TestEncoderSnapshot:

    @staticmethod
    def _fake_checkpoint(path: Path, steps: int, manifest_sha: str) -> None:
        torch.save({
            'encoder': {'layer.weight': torch.zeros(2, 2)},
            'optimizer_steps': steps,
            'epoch': 2,
            'micro_steps': steps * 8,
            'manifest_sha256': manifest_sha,
            'pretrained_init': {'sha256': 'a' * 64},
            'git_commit': 'deadbeef',
            'config': {'meta': {'arm': 'B'}},
        }, path)

    def test_snapshot_records_steps_and_hashes(self, tmp_path):
        source = tmp_path / 'jepa-latest.pth.tar'
        self._fake_checkpoint(source, steps=2000, manifest_sha='b' * 64)
        out = tmp_path / 'encoder.pth'
        proc = subprocess.run(
            [sys.executable, '-m', 'tools.extract_encoder_snapshot',
             '--checkpoint', str(source), '--out', str(out),
             '--expect-steps', '2000', '--expect-manifest-sha', 'b' * 64],
            capture_output=True, text=True, cwd=REPO)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        payload = torch.load(out, map_location='cpu', weights_only=False)
        assert payload['optimizer_steps'] == 2000
        assert payload['manifest_sha256'] == 'b' * 64
        assert payload['source_checkpoint_sha256'] == sha256(source)
        assert 'encoder' in payload

    def test_snapshot_refuses_the_wrong_step(self, tmp_path):
        source = tmp_path / 'jepa-latest.pth.tar'
        self._fake_checkpoint(source, steps=2000, manifest_sha='b' * 64)
        proc = subprocess.run(
            [sys.executable, '-m', 'tools.extract_encoder_snapshot',
             '--checkpoint', str(source), '--out', str(tmp_path / 'x.pth'),
             '--expect-steps', '500'],
            capture_output=True, text=True, cwd=REPO)
        assert proc.returncode != 0
        assert 'expected 500' in (proc.stderr + proc.stdout)

    def test_snapshot_refuses_the_wrong_manifest(self, tmp_path):
        source = tmp_path / 'jepa-latest.pth.tar'
        self._fake_checkpoint(source, steps=2000, manifest_sha='b' * 64)
        proc = subprocess.run(
            [sys.executable, '-m', 'tools.extract_encoder_snapshot',
             '--checkpoint', str(source), '--out', str(tmp_path / 'x.pth'),
             '--expect-manifest-sha', 'c' * 64],
            capture_output=True, text=True, cwd=REPO)
        assert proc.returncode != 0
        assert 'trained on manifest' in (proc.stderr + proc.stdout)


# --------------------------------------------------------------------------- #
# downstream: checkpoint-bound feature caches
# --------------------------------------------------------------------------- #

class TestDownstreamCacheBinding:

    def test_cache_key_depends_on_the_weights_not_only_the_label(self):
        from src.downstream.features import cache_key
        assert cache_key('B_2000', 'a' * 64) != cache_key('B_2000', 'b' * 64)
        assert cache_key('B_2000', 'a' * 64) != cache_key('C1_2000', 'a' * 64)
        assert cache_key('B_2000', 'a' * 64) == cache_key('B_2000', 'a' * 64)

    def test_cache_manifest_is_refused_for_other_weights(self, tmp_path):
        from src.downstream.phenobench_probe import (
            verify_cache_manifest, write_cache_manifest)
        first = {'label': 'B_2000', 'checkpoint': '/x/b.pth',
                 'checkpoint_sha256': 'a' * 64}
        write_cache_manifest(tmp_path, first)
        verify_cache_manifest(tmp_path, first)          # same weights: fine
        other = {'label': 'C1_2000', 'checkpoint': '/x/c1.pth',
                 'checkpoint_sha256': 'b' * 64}
        with pytest.raises(RuntimeError, match='Refusing to reuse'):
            verify_cache_manifest(tmp_path, other)

    def test_writing_a_cache_manifest_with_other_weights_is_refused(self, tmp_path):
        from src.downstream.phenobench_probe import write_cache_manifest
        write_cache_manifest(tmp_path, {'label': 'B_2000', 'checkpoint': '/x/b.pth',
                                        'checkpoint_sha256': 'a' * 64})
        with pytest.raises(RuntimeError, match='Refusing to reuse'):
            write_cache_manifest(tmp_path, {'label': 'C2_2000', 'checkpoint': '/x/c2.pth',
                                            'checkpoint_sha256': 'c' * 64})

    def test_missing_cache_manifest_is_refused(self, tmp_path):
        from src.downstream.phenobench_probe import verify_cache_manifest
        with pytest.raises(FileNotFoundError, match='cache_manifest.json'):
            verify_cache_manifest(tmp_path, {'label': 'B', 'checkpoint': '/x',
                                             'checkpoint_sha256': 'a' * 64})

    def test_checkpoint_config_records_the_sha(self, tmp_path):
        from src.downstream.features import checkpoint_config
        weights = tmp_path / 'encoder.pth'
        torch.save({'encoder': {}}, weights)
        cfg = checkpoint_config(weights, 'B_2000', continued_training='2000 updates')
        assert cfg['checkpoint_sha256'] == sha256(weights)
        assert cfg['label'] == 'B_2000'

    def test_missing_checkpoint_raises(self, tmp_path):
        from src.downstream.features import checkpoint_config
        with pytest.raises(FileNotFoundError):
            checkpoint_config(tmp_path / 'nope.pth', 'x')


# --------------------------------------------------------------------------- #
# evaluation driver wiring
# --------------------------------------------------------------------------- #

class TestEvaluationDriver:

    def test_arms_map_to_the_configured_run_folders(self):
        from tools.eval_stage4a_checkpoints import ARMS
        assert set(ARMS) == {'B', 'C1', 'C2', 'C3'}
        for arm, (run_dir, config) in ARMS.items():
            cfg = yaml.safe_load((REPO / config).read_text(encoding='utf-8'))
            assert cfg['meta']['arm'] == arm
            assert Path(cfg['logging']['folder']).name == run_dir

    def test_one_probe_protocol_for_every_arm(self):
        from tools.eval_stage4a_checkpoints import protocol_from_config
        protocol = protocol_from_config('configs/phenobench_armA_linear.yaml')
        assert protocol['seeds'] == [0, 1, 2]
        assert protocol['epochs'] > 0 and protocol['lr'] > 0

    def test_evaluation_points_cover_both_snapshot_steps(self):
        from tools.eval_stage4a_checkpoints import ARMS
        for arm, (_, config) in ARMS.items():
            cfg = yaml.safe_load((REPO / config).read_text(encoding='utf-8'))
            steps = list(cfg['optimization']['save_at_optimizer_steps'])
            steps.append(cfg['optimization']['max_optimizer_steps'])
            assert steps == [500, 2000], arm

    def test_snapshot_lookup_prefers_the_direct_file(self, tmp_path):
        from tools.eval_stage4a_checkpoints import extract_snapshot
        run = tmp_path / 'run'
        run.mkdir()
        direct = run / 'jepa-encoder-step500.pth'
        torch.save({'encoder': {}}, direct)
        assert extract_snapshot(tmp_path, 'run', 500) == direct


# --------------------------------------------------------------------------- #
# the prior must not touch any label
# --------------------------------------------------------------------------- #

class TestPriorUsesNoLabels:

    FORBIDDEN = ('phenobench', 'label_png', 'semantic_label', 'ground_truth',
                 'gt_mask', 'annotation_path', 'annotations_path')
    #: ``from __future__ import annotations`` is not a label access.
    ALLOWED_LINES = ('from __future__ import annotations',)

    @classmethod
    def _label_free(cls, relative: str) -> bool:
        for line in (REPO / relative).read_text(encoding='utf-8').splitlines():
            lowered = line.lower()
            if any(allowed in lowered for allowed in cls.ALLOWED_LINES):
                continue
            if any(token in lowered for token in cls.FORBIDDEN):
                return False
        return True

    def test_structure_prior_module_has_no_label_access(self):
        assert self._label_free('src/masks/agri_structure.py')

    def test_guided_collator_module_has_no_label_access(self):
        assert self._label_free('src/masks/structure_guided.py')

    def test_prior_api_takes_only_images_and_ids(self):
        import inspect
        from src.masks.agri_structure import StructurePrior
        signature = inspect.signature(StructurePrior.compute)
        assert list(signature.parameters) == ['self', 'images', 'sample_ids', 'base_seed']


# --------------------------------------------------------------------------- #
# machine-readable outputs
# --------------------------------------------------------------------------- #

class TestMachineReadableOutputs:

    def test_protocol_report_is_json(self):
        path = REPO / 'reports/stage3b4a_protocol_check.json'
        if not path.is_file():
            pytest.skip('protocol report not generated yet')
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert list(payload['configs']) == ['B', 'C1', 'C2', 'C3']
        assert payload['ok'] is True and payload['problems'] == []
        # only the mask strategy and the run folder are allowed to differ
        assert payload['allowed_different'] == sorted([
            'logging.folder', 'meta.arm', 'meta.arm_label',
            'mask.base_seed', 'mask.boundary_edge_weight', 'mask.guidance_lambda',
            'mask.mode', 'mask.num_candidates', 'mask.prior_mode'])

    def test_mask_statistics_are_json(self):
        path = REPO / 'runs/stage4a_masks/mask_statistics.json'
        if not path.is_file():
            pytest.skip('mask statistics not generated yet')
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert set(payload['method_statistics']) >= {'B', 'C1', 'C2', 'C3'}
        for arm in ('B', 'C1', 'C2', 'C3'):
            entry = payload['method_statistics'][arm]
            assert entry['total_target_area']['mean'] > 0
            assert entry['target_context_overlap']['max'] == 0

    def test_reproducibility_report_is_json(self):
        path = REPO / 'reports/multworker_reproducibility.json'
        if not path.is_file():
            pytest.skip('reproducibility report not generated yet')
        payload = json.loads(path.read_text(encoding='utf-8'))
        assert payload['num_workers_0']['identical_across_repeated_runs'] is True
        assert payload['image_cache'] is not None

# --------------------------------------------------------------------------- #
# comparison / ablation tables (protocol §10.2, §11, §17 D)
# --------------------------------------------------------------------------- #

class TestComparisonTable:

    def _fake_result(self, values):
        return {
            'backbone_checkpoint': '/x.pth',
            'backbone_checkpoint_sha256': 'ab' * 32,
            'continued_training': 'none',
            'protocol': {'seeds': [0, 1, 2]},
            'aggregate': {
                'n_seeds': len(values),
                'official': {
                    metric: {'mean': sum(v) / len(v), 'std': 0.0, 'values': list(v)}
                    for metric, v in values.items()
                },
                'self': {
                    metric: {'mean': sum(v) / len(v), 'std': 0.0, 'values': list(v)}
                    for metric, v in values.items()
                },
            },
            'seeds': [{'metric_cross_check': {'max_abs_difference_pp': 0.0,
                                              'agrees': True}}],
        }

    def test_paired_delta_is_in_percentage_points(self):
        from tools.compare_stage4a_phenobench import paired_delta
        a = {'mIoU': {'mean': 73.0, 'std': 0.1, 'n': 3, 'values': [73.0, 73.0, 73.0]}}
        b = {'mIoU': {'mean': 72.0, 'std': 0.1, 'n': 3, 'values': [72.0, 72.0, 72.0]}}
        delta = paired_delta(a, b, 'mIoU')
        assert delta['delta_pp'] == pytest.approx(1.0)
        assert delta['relative_percent'] == pytest.approx(100.0 / 72.0)
        assert delta['wins'] == 3 and delta['losses'] == 0

    def test_paired_delta_reports_a_sign_test_p_value(self):
        from tools.compare_stage4a_phenobench import paired_delta
        a = {'weed': {'mean': 40.0, 'std': 1.0, 'n': 3, 'values': [39.0, 40.0, 41.0]}}
        b = {'weed': {'mean': 36.0, 'std': 1.0, 'n': 3, 'values': [35.0, 36.0, 37.0]}}
        delta = paired_delta(a, b, 'weed')
        assert 0.0 <= delta['sign_test_p_approx'] <= 1.0
        assert delta['se_of_delta_pp'] >= 0.0

    def test_every_arm_step_pair_is_covered_by_the_labels(self):
        from tools.compare_stage4a_phenobench import LABELS
        expected = {'A_0'}
        for arm in ('B', 'C1', 'C2', 'C3'):
            for step in (500, 2000):
                expected.add(f'{arm}_{step}')
        assert set(LABELS) == expected

    def test_every_label_is_evaluated_relative_to_b_at_the_same_step(self):
        from tools.compare_stage4a_phenobench import LABELS
        # the protocol requires B/C1/C2/C3 at both 500 and 2,000 updates, so
        # the baseline for a difference is always the same-budget B point
        for arm in ('C1', 'C2', 'C3'):
            for step in (500, 2000):
                assert LABELS[f'{arm}_{step}'][1] == LABELS[f'B_{step}'][1]


# --------------------------------------------------------------------------- #
# the per-GPU evaluation queue
# --------------------------------------------------------------------------- #

class TestEvaluationQueue:

    def test_queue_knows_every_evaluation_point(self):
        from tools.eval_stage4a_checkpoints import LABELS as EVAL_LABELS
        from tools.queue_stage4a_evals import SNAPSHOTS
        # A_0 is the official checkpoint and needs no training snapshot
        assert set(SNAPSHOTS) == set(EVAL_LABELS) - {'A_0'}

    def test_snapshot_paths_match_the_configured_run_folders(self):
        from tools.eval_stage4a_checkpoints import ARMS
        from tools.queue_stage4a_evals import SNAPSHOTS
        for label, (run_dir, snapshot) in SNAPSHOTS.items():
            arm = label.split('_')[0]
            assert ARMS[arm][0] == run_dir, label
            step = int(label.split('_')[1])
            assert snapshot == f'jepa-encoder-step{step}.pth'

    def test_queue_waits_instead_of_failing_on_a_missing_checkpoint(self):
        import inspect
        from tools import queue_stage4a_evals
        source = inspect.getsource(queue_stage4a_evals.main)
        # it must poll for the snapshot rather than evaluate a missing file
        assert 'while not path.is_file()' in source
        assert 'wait-timeout' in inspect.getsource(queue_stage4a_evals)
