"""Tests for the Stage 2D training-log fixes.

Two bugs are covered:

1. **Budget-end log loss.**  ``src/train.py`` used to ``break`` out of the
   micro-step loop *before* calling ``log_stats()``, so the micro-step that
   performed the final optimizer update of a bounded run never reached the CSV.
   The fix moves logging ahead of the break.  These tests assert the ordering
   property directly on the source, and assert the observable consequence on a
   real (tiny, CPU) training run.

2. **Silent CSV append.**  ``CSVLogger`` opened every file in ``'+a'`` mode, so
   a new experiment reusing a previous run's ``logging.folder`` silently
   inherited that run's rows.  The fix truncates on a fresh run and refuses a
   header mismatch on resume.

CPU-only.  The real training run uses a tiny synthetic dataset and a ViT-Base so
it finishes in seconds.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest
import torch

from src.utils.logging import CSVLogger

REPO = Path(__file__).resolve().parents[1]
TRAIN_SRC = (REPO / 'src' / 'train.py').read_text(encoding='utf-8')


# --------------------------------------------------------------------------- #
# 1. Ordering: logging must precede the budget/interrupt break
# --------------------------------------------------------------------------- #

class TestBudgetEndLoggingOrder:

    def test_log_stats_precedes_budget_break(self):
        """In the micro-step loop, `log_stats()` must come before the break."""
        loop = TRAIN_SRC[TRAIN_SRC.index('for itr, (udata, masks_enc, masks_pred)'):
                         TRAIN_SRC.index('budget_reached = (')]
        log_at = loop.index('log_stats()')
        break_at = loop.index("logger.info('Reached max_optimizer_steps")
        assert log_at < break_at, (
            'log_stats() must run before the budget break, otherwise the final '
            'optimizer update of a bounded run is never written to the CSV')

    def test_log_stats_precedes_interrupt_break(self):
        loop = TRAIN_SRC[TRAIN_SRC.index('for itr, (udata, masks_enc, masks_pred)'):
                         TRAIN_SRC.index('budget_reached = (')]
        log_at = loop.index('log_stats()')
        break_at = loop.index("logger.info('Interrupted at interrupt_after_optimizer_steps")
        assert log_at < break_at

    def test_nonfinite_check_still_runs_after_logging(self):
        loop = TRAIN_SRC[TRAIN_SRC.index('for itr, (udata, masks_enc, masks_pred)'):
                         TRAIN_SRC.index('budget_reached = (')]
        log_at = loop.index('log_stats()')
        # there is an earlier finite-loss guard inside train_step(); the one that
        # matters is the last occurrence, after logging
        assert log_at < loop.rindex('Non-finite loss at epoch')

    def test_update_count_still_increments_before_break(self):
        loop = TRAIN_SRC[TRAIN_SRC.index('for itr, (udata, masks_enc, masks_pred)'):
                         TRAIN_SRC.index('budget_reached = (')]
        assert loop.index('optimizer_steps += 1') < loop.index('log_stats()')


# --------------------------------------------------------------------------- #
# 2. CSVLogger append semantics
# --------------------------------------------------------------------------- #

class TestCSVLogger:

    def _logger(self, path):
        return CSVLogger(path, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'))

    def test_fresh_run_truncates_existing_file(self, tmp_path):
        path = tmp_path / 'jepa_r0.csv'
        path.write_text('epoch,itr,loss\n99,99,9.9\n', encoding='utf-8')
        logger = self._logger(path)
        logger.log(1, 0, 0.5)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1, 'a fresh run must not inherit an older run\'s rows'
        assert rows[0]['epoch'] == '1'

    def test_fresh_run_writes_header(self, tmp_path):
        path = tmp_path / 'jepa_r0.csv'
        self._logger(path)
        assert path.read_text().splitlines()[0] == 'epoch,itr,loss'

    def test_resume_appends_and_keeps_prior_rows(self, tmp_path):
        path = tmp_path / 'jepa_r0.csv'
        logger = self._logger(path)
        logger.log(1, 0, 0.5)
        logger.log(1, 1, 0.4)
        resumed = CSVLogger(path, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'),
                            append=True)
        resumed.log(2, 0, 0.3)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 3
        assert rows[-1]['epoch'] == '2'

    def test_resume_refuses_header_mismatch(self, tmp_path):
        path = tmp_path / 'jepa_r0.csv'
        path.write_text('epoch,itr,loss,extra\n1,0,0.5,x\n', encoding='utf-8')
        with pytest.raises(ValueError, match='Refusing to append'):
            self._logger(path) if False else CSVLogger(
                path, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'), append=True)

    def test_resume_of_empty_file_starts_fresh(self, tmp_path):
        path = tmp_path / 'jepa_r0.csv'
        path.write_text('', encoding='utf-8')
        logger = CSVLogger(path, ('%d', 'epoch'), ('%d', 'itr'), ('%.5f', 'loss'),
                           append=True)
        logger.log(1, 0, 0.5)
        assert len(list(csv.DictReader(path.open()))) == 1

    def test_train_uses_append_only_when_resuming(self):
        assert 'append=bool(load_model)' in TRAIN_SRC


# --------------------------------------------------------------------------- #
# 3. End-to-end: a real bounded run must log its final update
# --------------------------------------------------------------------------- #

def _write_smoke_config(tmp_path: Path, folder: Path, budget: int,
                        interrupt_after=None, load_checkpoint: bool = False,
                        read_checkpoint=None) -> Path:
    import yaml
    cfg = {
        'meta': {
            'model_name': 'vit_base', 'patch_size': 16, 'pred_depth': 6,
            'pred_emb_dim': 384, 'load_checkpoint': load_checkpoint,
            'read_checkpoint': read_checkpoint, 'pretrained_checkpoint': None,
            'copy_data': False, 'use_bfloat16': False, 'use_sdpa': True,
            'activation_checkpointing': False,
        },
        'data': {
            'dataset': 'agrifield40k', 'manifest': str(tmp_path / 'manifest.csv'),
            'split': 'train', 'seed': 0, 'batch_size': 2, 'crop_size': 224,
            'crop_scale': [0.3, 1.0], 'image_folder': '.', 'num_workers': 0,
            'pin_mem': False, 'root_path': str(tmp_path), 'max_samples': 8,
            'use_color_distortion': False, 'use_gaussian_blur': False,
            'use_horizontal_flip': False, 'color_jitter_strength': 0.0,
        },
        'mask': {
            'allow_overlap': False, 'aspect_ratio': [0.75, 1.5],
            'enc_mask_scale': [0.85, 1.0], 'min_keep': 10, 'num_enc_masks': 1,
            'num_pred_masks': 4, 'patch_size': 16, 'pred_mask_scale': [0.15, 0.2],
        },
        'optimization': {
            'ema': [0.996, 1.0], 'epochs': 50, 'stop_after_epoch': None,
            'max_optimizer_steps': budget, 'final_lr': 1e-6,
            'final_weight_decay': 0.04, 'ipe_scale': 1.0, 'lr': 1e-4,
            'start_lr': 1e-4, 'warmup': 0.0, 'weight_decay': 0.04,
            'grad_accum': 2,
        },
        'logging': {'folder': str(folder), 'write_tag': 'jepa'},
    }
    if interrupt_after is not None:
        cfg['optimization']['interrupt_after_optimizer_steps'] = interrupt_after
    path = tmp_path / 'smoke.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return path


def _make_manifest_and_images(tmp_path: Path, n: int = 8) -> Path:
    """A manifest plus real JPEGs so the loader can decode them."""
    import csv as _csv
    from PIL import Image
    import hashlib

    img_dir = tmp_path / 'imgs'
    img_dir.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        p = img_dir / f'img_{i:03d}.jpg'
        Image.fromarray(
            (torch.rand(64, 64, 3) * 255).to(torch.uint8).numpy()).save(p)
        data = p.read_bytes()
        rows.append({
            'sample_id': f'img_{i:03d}', 'path': f'imgs/img_{i:03d}.jpg',
            'source_prefix': 'syn', 'source': 'Synthetic', 'original_id': f'img_{i:03d}',
            'split': 'train', 'original_split': 'train', 'width': 64, 'height': 64,
            'mode': 'RGB', 'format': 'JPEG', 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
            'pixel_sha256': hashlib.sha256(data).hexdigest(),
            'phash64': '0', 'dhash64': '0', 'decode_ok': 'true',
            'license': 'CC0-1.0', 'compilation_license': 'CC0-1.0',
            'source_url': '', 'source_revision': '', 'attribution': '',
            'observation_url': '', 'status': 'include', 'reason': '', 'component': '',
        })
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as handle:
        writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _run_training(config_path: Path, devices=('cpu',)) -> None:
    """Run one process of the training loop in-process.

    The process group is destroyed afterwards, mirroring ``main.py``'s
    ``process_main``.  Leaving it initialized would leak into later tests (e.g.
    torchmetrics would try to all-reduce over it).
    """
    import os

    import yaml
    import torch.distributed as dist
    from src.utils.distributed import init_distributed
    from src.train import main as app_main
    params = yaml.safe_load(Path(config_path).read_text())
    # -- a real training run may already own the default rendezvous port, so the
    #    in-process test uses its own and restores the previous value afterwards.
    previous = os.environ.get('AGRISA_MASTER_PORT')
    os.environ['AGRISA_MASTER_PORT'] = '40199'
    try:
        init_distributed(rank_and_world_size=(0, 1))
        try:
            app_main(args=params)
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
    finally:
        if previous is None:
            os.environ.pop('AGRISA_MASTER_PORT', None)
        else:
            os.environ['AGRISA_MASTER_PORT'] = previous


@pytest.mark.slow
class TestBudgetEndLoggingEndToEnd:
    """A real bounded run must write its final optimizer update to the CSV.

    Marked ``slow`` because it runs the actual training loop (tiny, CPU, ViT-Base).
    """

    def test_final_update_is_logged(self, tmp_path):
        _make_manifest_and_images(tmp_path)
        folder = tmp_path / 'run'
        cfg = _write_smoke_config(tmp_path, folder, budget=3)
        _run_training(cfg)
        rows = list(csv.DictReader((folder / 'jepa_r0.csv').open()))
        # 8 images, batch 2 -> 4 micro-steps/epoch, grad_accum 2 -> 2 updates/epoch
        # budget 3 lands mid-epoch 2.  Every micro-step must be logged, including
        # the one that performed update #3.
        assert len(rows) == 6, f'expected 6 logged micro-steps, got {len(rows)}'
        assert rows[-1]['epoch'] == '2'

    def test_interrupted_run_logs_its_final_update(self, tmp_path):
        _make_manifest_and_images(tmp_path)
        folder = tmp_path / 'run'
        cfg = _write_smoke_config(tmp_path, folder, budget=10, interrupt_after=3)
        _run_training(cfg)
        rows = list(csv.DictReader((folder / 'jepa_r0.csv').open()))
        assert len(rows) == 6

    def test_resume_does_not_duplicate_or_drop_rows(self, tmp_path):
        _make_manifest_and_images(tmp_path)
        folder = tmp_path / 'run'
        _run_training(_write_smoke_config(tmp_path, folder, budget=10,
                                          interrupt_after=2))
        first = list(csv.DictReader((folder / 'jepa_r0.csv').open()))
        assert len(first) == 4
        # resume from the interruption point, same folder -> append
        _run_training(_write_smoke_config(
            tmp_path, folder, budget=10, load_checkpoint=True,
            read_checkpoint='jepa-latest.pth.tar'))
        rows = list(csv.DictReader((folder / 'jepa_r0.csv').open()))
        assert len(rows) > len(first), 'resume must append, not truncate'
        # 8 images / batch 2 = 4 micro-steps/epoch; budget 10 updates at grad_accum
        # 2 = 20 micro-steps across the whole run, logged once each across the
        # interrupted and the resumed process.
        assert len(rows) == 20, f'expected 20 rows total, got {len(rows)}'
        positions = [(r['epoch'], r['itr']) for r in rows]
        assert len(set(positions)) == len(positions), 'duplicate micro-steps logged'
        # and the interrupted process really stopped mid-epoch-1
        assert (first[-1]['epoch'], first[-1]['itr']) == ('1', '3')