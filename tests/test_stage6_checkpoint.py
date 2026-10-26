"""Stage 6 checkpoint reload / encoder export tests.

These run on the smoke artefacts (10-step interruption + resume to 20), which
are the cheapest run that exercises every checkpoint field.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'runs/stage6_smoke_resume/jepa-latest.pth.tar'
SMOKE_CONT = ROOT / 'runs/stage6_smoke_resume_continued/jepa-latest.pth.tar'


def load(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


@pytest.mark.skipif(not SMOKE.is_file(), reason='Stage 6 smoke checkpoint not present')
def test_interrupted_checkpoint_stopped_mid_epoch():
    checkpoint = load(SMOKE)
    assert checkpoint['epoch_itr'] > 0, 'the smoke stop must be mid-epoch'
    assert checkpoint['epoch_start_state'] is not None
    for field in ('rng_state', 'loader_generator_state', 'mask_collator_step'):
        assert field in checkpoint['epoch_start_state']


@pytest.mark.skipif(not SMOKE.is_file(), reason='Stage 6 smoke checkpoint not present')
def test_checkpoint_manifest_binds_to_the_train_view():
    checkpoint = load(SMOKE)
    assert checkpoint['manifest_sha256']
    assert checkpoint['config']['data']['split'] == 'train'


@pytest.mark.skipif(not SMOKE_CONT.is_file(), reason='Stage 6 smoke resume checkpoint not present')
def test_resumed_checkpoint_reached_twenty_updates():
    checkpoint = load(SMOKE_CONT)
    assert checkpoint['optimizer_steps'] == 20
    assert checkpoint['micro_steps'] == 160
    assert checkpoint['config']['optimization']['max_optimizer_steps'] == 20
    assert checkpoint['config']['meta']['load_checkpoint'] is True


@pytest.mark.skipif(not SMOKE_CONT.is_file(), reason='Stage 6 smoke resume checkpoint not present')
def test_resume_reproduces_the_same_model_state():
    """The resumed run must be a continuation, not a re-init.

    The interruption happened at step 10, so the resumed encoder cannot equal
    the official checkpoint (which is what a re-init would produce) and cannot
    equal the step-10 snapshot either (which is what a no-op resume would
    produce).
    """
    from src.pretrained import strip_module_prefix

    checkpoint = load(SMOKE_CONT)
    official = Path(checkpoint['config']['meta']['pretrained_checkpoint'])
    payload = torch.load(official, map_location='cpu', weights_only=False)
    official_encoder = strip_module_prefix(payload['encoder'])

    # -- pick a real weight matrix the optimizer must have moved. Positional /
    #    class-token buffers are not trainable, so they would trivially match.
    weight_key = 'module.blocks.0.mlp.fc1.weight'
    assert weight_key in checkpoint['encoder'], sorted(checkpoint['encoder'])[:5]
    current = checkpoint['encoder'][weight_key]
    official_tensor = official_encoder[weight_key.removeprefix('module.')]
    assert not torch.equal(current, official_tensor), (
        'resumed encoder equals the official checkpoint: the resume did nothing')

    interrupted = load(SMOKE)['encoder']
    assert not torch.equal(current, interrupted[weight_key]), (
        'resumed encoder equals the step-10 snapshot: no training happened after resume')


@pytest.mark.skipif(not SMOKE_CONT.is_file(), reason='Stage 6 smoke resume checkpoint not present')
def test_encoder_export_round_trip():
    """The encoder snapshot must be loadable into a fresh model."""
    from src.helper import init_model

    checkpoint = load(SMOKE_CONT)
    encoder, _ = init_model(
        device=torch.device('cpu'), patch_size=14, crop_size=224,
        pred_depth=12, pred_emb_dim=384, model_name='vit_huge',
        use_sdpa=True, activation_checkpointing=False)
    state = {key.removeprefix('module.'): value
             for key, value in checkpoint['encoder'].items()}
    message = encoder.load_state_dict(state, strict=True)
    assert message.missing_keys == []
    assert message.unexpected_keys == []


@pytest.mark.skipif(not SMOKE_CONT.is_file(), reason='Stage 6 smoke resume checkpoint not present')
def test_optimizer_and_ema_state_survived_the_resume():
    checkpoint = load(SMOKE_CONT)
    assert checkpoint['opt']['state'], 'optimizer state must be non-empty'
    assert len(checkpoint['opt']['param_groups']) == 4
    target = {key.removeprefix('module.'): value
              for key, value in checkpoint['target_encoder'].items()}
    encoder = {key.removeprefix('module.'): value
               for key, value in checkpoint['encoder'].items()}
    assert set(target) == set(encoder)
    # -- EMA has moved the target encoder away from the encoder
    differences = [key for key in encoder
                   if not torch.equal(target[key], encoder[key])]
    assert differences, 'target encoder must differ from the encoder after EMA'
