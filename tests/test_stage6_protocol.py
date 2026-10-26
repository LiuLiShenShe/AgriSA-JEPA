"""Stage 6 protocol, manifest, ledger and checkpoint tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.helper import init_opt
from src.models import vision_transformer as vit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_B1 = ROOT / 'configs/stage6_b1_natural_2000.yaml'
CONFIG_B2 = ROOT / 'configs/stage6_b2_capped_no_backfill_2000.yaml'
CONFIG_SMOKE = ROOT / 'configs/stage6_smoke_resume.yaml'
TRAIN_CSV = ROOT / 'data/corpus_v4/R3v3_license_verified_train.csv'
VAL_CSV = ROOT / 'data/corpus_v4/R3v3_license_verified_val.csv'
VIEW_CSV = ROOT / 'data/stage6/stage6_r3v3_train_view.csv'
VIEW_JSON = ROOT / 'data/stage6/stage6_r3v3_train_view.json'


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


# --------------------------------------------------------------------------- #
# protocol checker
# --------------------------------------------------------------------------- #

def test_stage6_protocol_passes(tmp_path):
    result = subprocess.run(
        [sys.executable, '-m', 'tools.check_stage6_protocol',
         '--out', str(tmp_path / 'check.json')],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'STAGE 6 PROTOCOL OK' in result.stdout
    payload = json.loads((tmp_path / 'check.json').read_text())
    assert payload['ok'] is True
    assert payload['problems'] == []


def test_protocol_checker_detects_a_disallowed_difference(tmp_path):
    altered = tmp_path / 'b2_bad_lr.yaml'
    config = load_yaml(CONFIG_B2)
    config['optimization']['lr'] = 0.0003
    altered.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'tools.check_stage6_protocol',
         '--config', f'B1={CONFIG_B1}',
         '--config', f'B2={altered}',
         '--out', str(tmp_path / 'check.json')],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'optimization.lr' in result.stdout


def test_protocol_checker_detects_backfill(tmp_path):
    altered = tmp_path / 'b2_backfill.yaml'
    config = load_yaml(CONFIG_B2)
    config['data']['sampling']['backfill'] = True
    altered.write_text(yaml.safe_dump(config), encoding='utf-8')
    result = subprocess.run(
        [sys.executable, '-m', 'tools.check_stage6_protocol',
         '--config', f'B1={CONFIG_B1}',
         '--config', f'B2={altered}',
         '--out', str(tmp_path / 'check.json')],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'backfill' in result.stdout


# --------------------------------------------------------------------------- #
# config invariants
# --------------------------------------------------------------------------- #

def test_arms_share_budget_and_save_points():
    for path in (CONFIG_B1, CONFIG_B2):
        optimization = load_yaml(path)['optimization']
        assert optimization['max_optimizer_steps'] == 2000
        assert optimization['save_at_optimizer_steps'] == [500, 1000, 1500, 2000]
        assert optimization['grad_accum'] == 8
        assert optimization['warmup_steps'] == 100


def test_arms_only_differ_in_sampling_and_identity():
    b1 = load_yaml(CONFIG_B1)
    b2 = load_yaml(CONFIG_B2)
    assert b1['data']['sampling'] == {'mode': 'natural'}
    assert b2['data']['sampling']['mode'] == 'profile_capped'
    assert b1['meta']['arm'] != b2['meta']['arm']
    assert b1['logging']['folder'] != b2['logging']['folder']
    for section in ('meta', 'mask', 'optimization', 'data'):
        for key, value in b1[section].items():
            if section == 'data' and key == 'sampling':
                continue
            if section == 'meta' and key in ('arm', 'arm_label'):
                continue
            assert b2[section][key] == value, f'{section}.{key}'


def test_arms_use_the_r3v3_train_split():
    for path in (CONFIG_B1, CONFIG_B2):
        data = load_yaml(path)['data']
        assert data['split'] == 'train'
        assert 'stage6_r3v3_train_view' in data['manifest']
        assert data['image_cache_dir'] is None
        assert data['num_workers'] == 0


def test_manifest_sha_matches_config():
    import hashlib
    view = ROOT / 'data/stage6/stage6_r3v3_train_view_nvme.csv'
    if not view.is_file():
        view = VIEW_CSV
    actual = hashlib.sha256(view.read_bytes()).hexdigest()
    for path in (CONFIG_B1, CONFIG_B2):
        assert load_yaml(path)['data']['manifest_sha256'] == actual


def test_warmup_override_makes_lr_trajectory_arm_independent():
    """Different samples-per-epoch must not change the LR/WD trajectory."""

    def schedule(iterations_per_epoch):
        encoder = vit.vit_huge(img_size=[224], patch_size=14)
        predictor = vit.vit_predictor(
            num_patches=encoder.patch_embed.num_patches,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=384, depth=12, num_heads=encoder.num_heads)
        opt, _, sched, wd = init_opt(
            encoder=encoder, predictor=predictor,
            iterations_per_epoch=iterations_per_epoch,
            start_lr=1e-5, ref_lr=2e-4, final_lr=1e-6,
            warmup=0.05, num_epochs=24, wd=0.04, final_wd=0.4,
            total_steps=2000, warmup_steps_override=100)
        lrs, wds = [], []
        for _ in range(2000):
            lrs.append(sched.step())
            wds.append(wd.step())
        return lrs, wds

    lrs_a, wds_a = schedule(1151)
    lrs_b, wds_b = schedule(869)
    assert lrs_a == lrs_b
    assert wds_a == wds_b


def test_warmup_override_rejects_out_of_range():
    encoder = vit.vit_huge(img_size=[224], patch_size=14)
    predictor = vit.vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim, predictor_embed_dim=384,
        depth=12, num_heads=encoder.num_heads)
    with pytest.raises(ValueError, match='warmup_steps'):
        init_opt(encoder=encoder, predictor=predictor,
                 iterations_per_epoch=10, start_lr=1e-5, ref_lr=2e-4,
                 final_lr=1e-6, warmup=0.05, num_epochs=1,
                 total_steps=20, warmup_steps_override=100)


# --------------------------------------------------------------------------- #
# derived training view
# --------------------------------------------------------------------------- #

def test_view_membership_matches_frozen_train_split():
    with TRAIN_CSV.open(newline='', encoding='utf-8') as handle:
        train_ids = [row['image_id'] for row in csv.DictReader(handle)]
    with VIEW_CSV.open(newline='', encoding='utf-8') as handle:
        view = list(csv.DictReader(handle))
    assert len(view) == len(train_ids) == 147_277
    # -- row order is unchanged, so index i is the same image in both files
    assert [row['sample_id'] for row in view] == train_ids


def test_view_excludes_the_val_split():
    with VAL_CSV.open(newline='', encoding='utf-8') as handle:
        val_ids = {row['image_id'] for row in csv.DictReader(handle)}
    with VIEW_CSV.open(newline='', encoding='utf-8') as handle:
        view_ids = {row['sample_id'] for row in csv.DictReader(handle)}
    assert view_ids.isdisjoint(val_ids)
    assert len(view_ids) == 147_277


def test_view_marks_all_rows_include_and_train():
    with VIEW_CSV.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            assert row['status'] == 'include'
            assert row['split'] == 'train'
            assert row['physical_source_id']


def test_view_provenance_records_counts():
    payload = json.loads(VIEW_JSON.read_text(encoding='utf-8'))
    assert payload['membership']['rows'] == 147_277
    assert payload['membership']['train_val_overlap'] == 0
    assert payload['membership']['all_files_exist'] is True
    assert sum(payload['by_source'].values()) == 147_277


# --------------------------------------------------------------------------- #
# source exposure ledger
# --------------------------------------------------------------------------- #

def test_ledger_counts_match_the_emitted_stream(tmp_path):
    from src.datasets.source_ledger import SourceExposureLedger
    mapping = {'s1': 'a', 's2': 'a', 's3': 'b'}
    ledger = SourceExposureLedger(tmp_path / 'ledger.csv', mapping, part='0')
    ledger.record(epoch=0, itr=0, micro_step=1, optimizer_step=0,
                  sample_ids=['s1', 's2', 's3'])
    ledger.record(epoch=0, itr=1, micro_step=2, optimizer_step=0,
                  sample_ids=['s1', 's1'])
    ledger.flush()
    summary = json.loads((tmp_path / 'ledger.summary.json').read_text())
    assert summary['counts'] == {'a': 4, 'b': 1}
    assert summary['total'] == 5
    assert summary['samples'] == 5
    with (tmp_path / 'ledger.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {row['physical_source_id'] for row in rows} == {'a', 'b'}


def test_ledger_appends_parts_without_losing_counts(tmp_path):
    from src.datasets.source_ledger import SourceExposureLedger
    mapping = {'s1': 'a', 's2': 'b'}
    first = SourceExposureLedger(tmp_path / 'ledger.csv', mapping, part='0')
    first.record(0, 0, 1, 0, ['s1'])
    first.flush()
    second = SourceExposureLedger(tmp_path / 'ledger.csv', mapping, part='10')
    second.record(0, 1, 2, 10, ['s2'])
    second.flush()
    summary = json.loads((tmp_path / 'ledger.summary.json').read_text())
    assert summary['counts'] == {'a': 1, 'b': 1}
    assert set(summary['parts']) == {'0', '10'}


def test_ledger_rejects_unknown_sample(tmp_path):
    from src.datasets.source_ledger import SourceExposureLedger
    ledger = SourceExposureLedger(tmp_path / 'ledger.csv', {'s1': 'a'})
    ledger.record(0, 0, 1, 0, ['missing'])
    with pytest.raises(RuntimeError, match='absent from the manifest'):
        ledger.flush()


def test_ledger_flush_does_not_duplicate_rows(tmp_path):
    """flush() must append only the newly accumulated rows, not rewrite the
    whole buffer, otherwise every epoch-end flush would duplicate earlier rows.
    """
    from src.datasets.source_ledger import SourceExposureLedger
    import csv as _csv
    mapping = {'s1': 'a', 's2': 'b', 's3': 'a'}
    ledger = SourceExposureLedger(tmp_path / 'ledger.csv', mapping, part='0')
    ledger.record(0, 0, 1, 0, ['s1', 's2', 's3'])
    ledger.flush()
    # -- a second flush must not rewrite the same rows
    ledger.record(0, 1, 2, 0, ['s1', 's2', 's3'])
    ledger.flush()
    rows = list(_csv.DictReader((tmp_path / 'ledger.csv').open(newline='')))
    assert len(rows) == 6, rows
    assert len({(r['micro_step'], r['sample_id']) for r in rows}) == 6


def test_ledger_summary_counts_match_unique_rows(tmp_path):
    from src.datasets.source_ledger import SourceExposureLedger
    import csv as _csv
    mapping = {'s1': 'a', 's2': 'b', 's3': 'a'}
    ledger = SourceExposureLedger(tmp_path / 'ledger.csv', mapping, part='0')
    ledger.record(0, 0, 1, 0, ['s1', 's2', 's3'])
    ledger.flush()
    ledger.record(0, 1, 2, 0, ['s1', 's2', 's3'])
    ledger.flush()
    with (tmp_path / 'ledger.csv').open(newline='') as handle:
        unique = {(r['micro_step'], r['sample_id']) for r in _csv.DictReader(handle)}
    summary = json.loads((tmp_path / 'ledger.summary.json').read_text())
    assert summary['total'] == len(unique) == 6
    assert sum(summary['counts'].values()) == 6


# --------------------------------------------------------------------------- #
# smoke checkpoint contents (skipped when the artefact is absent)
# --------------------------------------------------------------------------- #

SMOKE_CHECKPOINT = ROOT / 'runs/stage6_smoke_resume/jepa-latest.pth.tar'


@pytest.mark.skipif(not SMOKE_CHECKPOINT.is_file(),
                    reason='Stage 6 smoke checkpoint not present on this machine')
def test_smoke_checkpoint_holds_every_resume_field():
    import torch
    checkpoint = torch.load(SMOKE_CHECKPOINT, map_location='cpu', weights_only=False)
    for field in ('encoder', 'predictor', 'target_encoder', 'opt', 'epoch',
                  'epoch_itr', 'epoch_start_state', 'optimizer_steps',
                  'micro_steps', 'grad_accum', 'config', 'hardware',
                  'git_commit', 'pretrained_init', 'rng_state',
                  'loader_generator_state', 'mask_collator_step',
                  'manifest_sha256'):
        assert field in checkpoint, field
    assert checkpoint['optimizer_steps'] == 10
    assert checkpoint['micro_steps'] == 80
    assert checkpoint['epoch_itr'] == 80
    assert checkpoint['grad_accum'] == 8
    assert checkpoint['manifest_sha256'] == json.loads(
        VIEW_JSON.read_text())['out_csv_sha256']
    assert checkpoint['hardware']['name'] == 'NVIDIA RTX A6000'


@pytest.mark.skipif(not (ROOT / 'runs/stage6_smoke_resume_continued/training_verification.yaml').is_file(),
                    reason='Stage 6 smoke resume verification not present')
def test_smoke_resume_has_no_duplicate_or_missing_updates():
    first = ROOT / 'runs/stage6_smoke_resume/jepa_r0.csv'
    second = ROOT / 'runs/stage6_smoke_resume_continued/jepa_r0.csv'
    def itrs(path):
        with path.open(newline='') as handle:
            return [int(row['itr']) for row in csv.DictReader(handle)]
    first_itrs, second_itrs = itrs(first), itrs(second)
    assert first_itrs == list(range(0, 80))
    assert second_itrs == list(range(80, 160))
    assert first_itrs + second_itrs == list(range(0, 160))
    verification = yaml.safe_load(
        (ROOT / 'runs/stage6_smoke_resume_continued/training_verification.yaml')
        .read_text())
    assert verification['optimizer_steps_total'] == 20
    assert verification['micro_steps_total'] == 160
    assert verification['ema_formula_verified'] is True
    assert verification['loss_finite'] is True
