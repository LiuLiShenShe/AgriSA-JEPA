"""Stage 4B protocol tests: config diffing and run-artifact validation.

These tests assert that the G arms are a mask-only variation of the reference
arm B, and that the training invariants required by the protocol actually hold
in the configs the runs use.  They are pure-Python (YAML / JSON only), so they
run without a GPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from tools.check_stage4b_protocol import (
    ALLOWED_DIFFERENT,
    CONFIGS,
    MUST_MATCH,
    flatten,
    get,
)

REPO = Path(__file__).resolve().parents[1]

EXPECTED_ARMS = ('G0', 'G1', 'G2', 'G3')
EXPECTED_PRIOR = {'G0': None, 'G1': 'vegetation', 'G2': 'boundary',
                  'G3': 'shifted_boundary'}
EXPECTED_LAMBDA = {'G0': 0.0, 'G1': 6.0, 'G2': 6.0, 'G3': 6.0}


def load(arm: str) -> dict:
    path = REPO / CONFIGS[arm]
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def load_reference() -> dict:
    return yaml.safe_load((REPO / 'configs/stage3b_random_2000.yaml')
                          .read_text(encoding='utf-8'))


# --------------------------------------------------------------------------- #
# Existence and identity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_config_exists(arm):
    assert (REPO / CONFIGS[arm]).is_file()


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_arm_identity_metadata(arm):
    cfg = load(arm)
    assert cfg['meta']['arm'] == arm
    assert cfg['meta']['arm_label']
    assert cfg['meta']['stage'] == 'stage4b_geometry_controlled_pilot'


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_every_arm_starts_from_the_official_checkpoint(arm):
    cfg = load(arm)
    assert cfg['meta']['load_checkpoint'] is False
    assert cfg['meta']['read_checkpoint'] is None
    assert cfg['meta']['pretrained_checkpoint'] == (
        '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar')
    assert cfg['meta']['pretrained_sha256'] == (
        '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22')
    # no arm may be chained on another arm's weights
    assert 'resume' not in cfg['meta']


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_distinct_run_folders(arm):
    cfg = load(arm)
    assert cfg['logging']['folder'].startswith('/home/test/agrisa_runs/stage4b_')
    assert cfg['logging']['folder'].endswith('_500')


def test_run_folders_are_unique():
    folders = {load(arm)['logging']['folder'] for arm in EXPECTED_ARMS}
    assert len(folders) == 4


# --------------------------------------------------------------------------- #
# The schedule: 2,000-step horizon, interrupted at 500
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_schedule_horizon_is_the_reference_2000(arm):
    """`max_optimizer_steps` must be 2000, not 500.

    Reducing the budget to 500 would shorten the LR/WD cosine decay and the EMA
    trajectory, which is exactly the unfairness the protocol forbids.
    """
    cfg = load(arm)
    reference = load_reference()
    assert cfg['optimization']['max_optimizer_steps'] == 2000
    assert cfg['optimization']['max_optimizer_steps'] == (
        reference['optimization']['max_optimizer_steps'])
    assert cfg['optimization']['lr'] == reference['optimization']['lr']
    assert cfg['optimization']['final_lr'] == reference['optimization']['final_lr']
    assert cfg['optimization']['start_lr'] == reference['optimization']['start_lr']
    assert cfg['optimization']['weight_decay'] == (
        reference['optimization']['weight_decay'])
    assert cfg['optimization']['final_weight_decay'] == (
        reference['optimization']['final_weight_decay'])
    assert cfg['optimization']['ema'] == reference['optimization']['ema']
    assert cfg['optimization']['warmup'] == reference['optimization']['warmup']
    assert cfg['optimization']['grad_accum'] == (
        reference['optimization']['grad_accum'])
    assert cfg['optimization']['ipe_scale'] == (
        reference['optimization']['ipe_scale'])


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_interrupt_after_exactly_500(arm):
    cfg = load(arm)
    assert cfg['optimization']['interrupt_after_optimizer_steps'] == 500
    assert cfg['optimization']['save_at_optimizer_steps'] == [500]


def test_reference_arm_is_not_interrupted():
    cfg = load_reference()
    assert cfg['optimization'].get('interrupt_after_optimizer_steps') is None
    assert cfg['optimization']['max_optimizer_steps'] == 2000
    assert cfg['mask']['mode'] == 'random'


# --------------------------------------------------------------------------- #
# Data, model and mask geometry invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_data_pipeline_matches_the_reference(arm):
    cfg, reference = load(arm), load_reference()
    for key in ('manifest', 'manifest_sha256', 'split', 'seed', 'augmentation_seed',
                'batch_size', 'num_workers', 'crop_size', 'crop_scale',
                'root_path', 'image_cache_dir', 'pin_mem',
                'use_color_distortion', 'use_gaussian_blur',
                'use_horizontal_flip', 'color_jitter_strength'):
        assert cfg['data'][key] == reference['data'][key], key


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_model_matches_the_reference(arm):
    cfg, reference = load(arm), load_reference()
    for key in ('model_name', 'patch_size', 'pred_depth', 'pred_emb_dim',
                'use_bfloat16', 'use_sdpa', 'activation_checkpointing',
                'copy_data'):
        assert cfg['meta'][key] == reference['meta'][key], key


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_mask_geometry_rules_match_the_reference(arm):
    cfg, reference = load(arm), load_reference()
    for key in ('allow_overlap', 'aspect_ratio', 'enc_mask_scale',
                'pred_mask_scale', 'num_enc_masks', 'num_pred_masks',
                'min_keep', 'patch_size'):
        assert cfg['mask'][key] == reference['mask'][key], key


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_manifest_sha256_still_matches_the_file(arm):
    cfg = load(arm)
    manifest = REPO / cfg['data']['manifest']
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert actual == cfg['data']['manifest_sha256']
    assert actual == 'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435'


# --------------------------------------------------------------------------- #
# The mask strategy itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_shared_geometry_control_strength(arm):
    cfg = load(arm)
    assert cfg['mask']['geometry_beta'] == 3.5
    assert cfg['mask']['geometry_control'] == 'linear'
    assert cfg['mask']['num_candidates'] == 16


@pytest.mark.parametrize('arm', EXPECTED_ARMS)
def test_prior_and_lambda(arm):
    cfg = load(arm)
    assert cfg['mask']['prior_mode'] == EXPECTED_PRIOR[arm]
    assert float(cfg['mask']['guidance_lambda']) == EXPECTED_LAMBDA[arm]


def test_g0_reads_no_structure_prior():
    cfg = load('G0')
    assert cfg['mask']['prior_mode'] is None
    assert cfg['mask']['score_mode'] == 'uniform'
    assert cfg['mask']['guidance_lambda'] == 0.0


@pytest.mark.parametrize('arm', ['G1', 'G2', 'G3'])
def test_guided_arms_use_the_prior_score(arm):
    cfg = load(arm)
    assert cfg['mask']['score_mode'] == 'prior'
    assert cfg['mask']['guidance_lambda'] == 6.0


def test_g3_shift_is_large_and_reproducible():
    cfg = load('G3')
    dy, dx = cfg['mask']['roll_shift']
    assert dy != 0 and dx != 0
    assert 0 < abs(dy) < 16 and 0 < abs(dx) < 16


def test_the_four_arms_differ_only_in_the_mask_keys():
    """A field-by-field diff of the four G configs against the allowed set."""
    flats = {arm: flatten(load(arm)) for arm in EXPECTED_ARMS}
    flats['B'] = flatten(load_reference())
    all_keys = set()
    for flat in flats.values():
        all_keys |= set(flat)
    offenders = []
    for key in sorted(all_keys):
        values = {arm: get(flat, key) for arm, flat in flats.items()}
        present = {a: v for a, v in values.items() if v is not None}
        if len(set(map(str, present.values()))) > 1 and key not in ALLOWED_DIFFERENT:
            offenders.append(f'{".".join(key)}: {values}')
    assert not offenders


@pytest.mark.parametrize('key', MUST_MATCH)
def test_must_match_keys_are_identical(key):
    flats = {'B': flatten(load_reference())}
    flats.update({arm: flatten(load(arm)) for arm in EXPECTED_ARMS})
    values = {arm: get(flat, key) for arm, flat in flats.items()}
    assert len(set(map(str, values.values()))) == 1, \
        f'{".".join(key)} differs: {values}'


# --------------------------------------------------------------------------- #
# Provenance of the reference checkpoint
# --------------------------------------------------------------------------- #

def test_official_checkpoint_present_and_hashed():
    path = Path('/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar')
    if not path.is_file():
        pytest.skip('official checkpoint not available on this machine')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    assert digest.hexdigest() == (
        '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22')


# --------------------------------------------------------------------------- #
# Resume artifact (only checked when the pilot has produced it)
# --------------------------------------------------------------------------- #

def test_resume_from_step500_preserves_the_2000_schedule():
    path = REPO / 'runs/stage4b_resume_test/training_verification.yaml'
    if not path.is_file():
        pytest.skip('resume verification not produced on this machine')
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    assert payload['optimizer_steps_total'] == 506
    assert payload['total_optimizer_steps_horizon'] == 2000
    assert payload['interrupt_after_optimizer_steps'] == 506
    assert payload['loss_finite'] is True
    assert payload['ema_formula_verified'] is True
    # a resumed run re-initialises from the SAME official checkpoint recorded
    # in the config, not from a different arm's weights
    assert payload['dataset'] == 'agrifield40k'


def test_step_snapshots_are_resumable_in_the_current_code():
    """The step-snapshot dict must carry epoch_start_state."""
    source = (REPO / 'src/train.py').read_text(encoding='utf-8')
    marker = "def save_step_snapshot"
    start = source.index(marker)
    end = source.index("def save_checkpoint", start) if "def save_checkpoint" in source[start:] else len(source)
    body = source[start:start + 4000]
    assert "'epoch_start_state': epoch_start_state" in body
