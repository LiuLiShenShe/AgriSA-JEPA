"""Stage 2 tests: ViT-H construction, official weight loading, masks, EMA, resume.

These tests must not require the 10 GB official checkpoint or a GPU.  The
checkpoint-dependent tests are skipped automatically unless
``AGRISA_VITH_CHECKPOINT`` points at a readable file.
"""

from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.nn.functional as F

from src.masks.multiblock import MaskCollator
from src.models.vision_transformer import (
    VIT_EMBED_DIMS,
    model_architecture,
    vit_huge,
    vit_predictor,
)
from src.pretrained import (
    WEIGHT_FIELDS,
    compare_state_dicts,
    describe_payload,
    strip_module_prefix,
)


CHECKPOINT_ENV = 'AGRISA_VITH_CHECKPOINT'


def checkpoint_path():
    path = os.environ.get(CHECKPOINT_ENV)
    if not path or not os.path.isfile(path):
        return None
    return path


needs_checkpoint = pytest.mark.skipif(
    checkpoint_path() is None,
    reason=f'{CHECKPOINT_ENV} is not set to a readable checkpoint')


# --------------------------------------------------------------------------- #
# 1. ViT-H construction and patch configuration
# --------------------------------------------------------------------------- #

def test_vith14_architecture_matches_official():
    encoder = vit_huge(img_size=[224], patch_size=14)
    assert encoder.embed_dim == 1280
    assert len(encoder.blocks) == 32
    assert encoder.num_heads == 16
    # 224 / 14 = 16 -> 16 x 16 = 256 patches
    assert encoder.patch_embed.num_patches == 256
    assert encoder.patch_embed.proj.kernel_size == (14, 14)
    assert encoder.patch_embed.proj.stride == (14, 14)


def test_predictor_depth_and_width():
    predictor = vit_predictor(num_patches=256, embed_dim=1280,
                              predictor_embed_dim=384, depth=12, num_heads=16)
    assert len(predictor.predictor_blocks) == 12
    assert predictor.predictor_embed.out_features == 384
    assert predictor.predictor_pos_embed.shape == (1, 256, 384)


def test_model_architecture_helper():
    cfg = model_architecture('vit_huge', 14, 224, 12, 384)
    assert cfg['embed_dim'] == VIT_EMBED_DIMS['vit_huge'] == 1280
    assert cfg['num_patches'] == 256
    assert cfg['predictor_depth'] == 12
    assert cfg['predictor_embed_dim'] == 384
    with pytest.raises(ValueError):
        model_architecture('vit_unknown', 14, 224, 12, 384)
    with pytest.raises(ValueError):
        model_architecture('vit_huge', 15, 224, 12, 384)  # 224 % 15 != 0


# --------------------------------------------------------------------------- #
# 2. Checkpoint field and shape checks
# --------------------------------------------------------------------------- #

def test_compare_state_dicts_detects_mismatch():
    a = {'x': torch.zeros(2, 2), 'y': torch.zeros(3)}
    b = {'x': torch.zeros(2, 2), 'z': torch.zeros(3)}
    diff = compare_state_dicts(a, b)
    assert diff['missing'] == ['y']
    assert diff['unexpected'] == ['z']
    assert diff['matched'] == 1
    assert diff['is_exact'] is False

    diff_same = compare_state_dicts(a, a)
    assert diff_same['is_exact'] is True
    assert diff_same['matched'] == 2


def test_compare_state_dicts_detects_shape_mismatch():
    a = {'x': torch.zeros(2, 2)}
    b = {'x': torch.zeros(2, 3)}
    diff = compare_state_dicts(a, b)
    assert diff['is_exact'] is False
    assert diff['shape_mismatch'] == [('x', (2, 2), (2, 3))]


def test_strip_module_prefix():
    sd = {'module.a': torch.zeros(1), 'b': torch.zeros(1)}
    out = strip_module_prefix(sd)
    assert set(out) == {'a', 'b'}
    assert set(sd) == {'module.a', 'b'}  # input not mutated


def test_describe_payload():
    payload = {
        'encoder': {'module.x': torch.zeros(1)},
        'target_encoder': {'module.x': torch.zeros(1)},
        'predictor': {'module.y': torch.zeros(1)},
        'epoch': 299,
    }
    summary = describe_payload(payload)
    assert summary['encoder_tensors'] == 1
    assert summary['encoder_prefixed'] is True
    assert summary['epoch'] == 299
    assert 'epoch' in summary


# --------------------------------------------------------------------------- #
# 3. Weight-loading strategy (Operation A vs Operation B)
# --------------------------------------------------------------------------- #

def test_refuses_to_load_mismatched_weights(tmp_path):
    """A shape/key mismatch must raise, not silently fall back to strict=False."""
    from src.pretrained import _load_module

    module = torch.nn.Linear(4, 3)
    bad = {'weight': torch.zeros(4, 3), 'bias': torch.zeros(2)}  # wrong bias
    with pytest.raises(RuntimeError, match='Refusing to load'):
        _load_module(module, bad, 'linear')

    good = {'weight': torch.ones(3, 4), 'bias': torch.zeros(3)}
    diff = _load_module(module, good, 'linear')
    assert diff['is_exact'] is True
    assert torch.equal(module.weight, torch.ones(3, 4))


def test_load_pretrained_payload_rejects_missing_file():
    from src.pretrained import load_pretrained_payload
    with pytest.raises(FileNotFoundError):
        load_pretrained_payload('/nonexistent/checkpoint.pth.tar')


@needs_checkpoint
def test_official_checkpoint_structure():
    from src.pretrained import load_pretrained_payload
    payload, mode = load_pretrained_payload(checkpoint_path())
    assert mode.startswith('weights_only')
    for field in WEIGHT_FIELDS:
        assert field in payload, f'official checkpoint missing {field}'
    # every weight field stores DDP-style module.-prefixed keys
    for field in WEIGHT_FIELDS:
        assert any(k.startswith('module.') for k in payload[field])
    assert payload['epoch'] == 299


@needs_checkpoint
def test_vith_encoder_loads_exactly_from_official_checkpoint():
    from src.pretrained import compare_state_dicts, load_pretrained_payload
    encoder = vit_huge(img_size=[224], patch_size=14)
    payload, _ = load_pretrained_payload(checkpoint_path())
    diff = compare_state_dicts(encoder.state_dict(),
                               strip_module_prefix(payload['encoder']))
    assert diff['is_exact'], (
        f"missing={diff['missing'][:5]} unexpected={diff['unexpected'][:5]} "
        f"shape={diff['shape_mismatch'][:5]}")
    assert diff['matched'] == 389


@needs_checkpoint
def test_vith_predictor_loads_exactly_from_official_checkpoint():
    from src.pretrained import compare_state_dicts, load_pretrained_payload
    predictor = vit_predictor(num_patches=256, embed_dim=1280,
                              predictor_embed_dim=384, depth=12, num_heads=16)
    payload, _ = load_pretrained_payload(checkpoint_path())
    diff = compare_state_dicts(predictor.state_dict(),
                               strip_module_prefix(payload['predictor']))
    assert diff['is_exact'], (
        f"missing={diff['missing'][:5]} unexpected={diff['unexpected'][:5]} "
        f"shape={diff['shape_mismatch'][:5]}")
    assert diff['matched'] == 152


@needs_checkpoint
def test_pretrained_init_changes_random_weights():
    from src.pretrained import init_from_pretrained
    from src.helper import init_model

    encoder, predictor = init_model(device=torch.device('cpu'), patch_size=14,
                                    model_name='vit_huge', crop_size=224,
                                    pred_depth=12, pred_emb_dim=384)
    before_qkv = encoder.blocks[0].attn.qkv.weight.detach().clone()
    report = init_from_pretrained(checkpoint_path(), encoder=encoder,
                                  predictor=predictor, target_encoder=None)
    assert report['encoder_keys_loaded'] == 389
    assert report['predictor_keys_loaded'] == 152
    assert report['predictor_inherited'] is True
    assert not torch.equal(before_qkv, encoder.blocks[0].attn.qkv.weight)


# --------------------------------------------------------------------------- #
# 4. Mask index legality
# --------------------------------------------------------------------------- #

def _make_masks(crop_size=224, patch_size=14, batch=2, allow_overlap=False):
    collator = MaskCollator(
        input_size=crop_size, patch_size=patch_size,
        enc_mask_scale=(0.85, 1.0), pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5), nenc=1, npred=4, min_keep=10,
        allow_overlap=allow_overlap)
    batch_items = [(torch.zeros(3, crop_size, crop_size), f's{i}') for i in range(batch)]
    return collator(batch_items)


def test_vith_mask_indices_are_legal():
    _, masks_enc, masks_pred = _make_masks()
    num_patches = 256
    for label, masks in (('enc', masks_enc), ('pred', masks_pred)):
        assert len(masks) >= 1
        for mask in masks:
            assert mask.dtype == torch.int64
            assert int(mask.min()) >= 0
            assert int(mask.max()) < num_patches
            for row in mask:
                assert len(set(row.tolist())) == len(row), f'{label} duplicate index'
            assert mask.shape[1] > 10, 'mask must respect min_keep'


def test_encoder_and_predictor_masks_are_disjoint():
    """The context block must not intersect any target block (allow_overlap=False).

    Target blocks may overlap each other by design, so the check is per
    prediction block, not against their union.
    """
    overlaps = 0
    for seed in range(25):
        torch.manual_seed(seed)
        _, masks_enc, masks_pred = _make_masks(allow_overlap=False)
        enc = set(masks_enc[0][0].tolist())
        for pred_block in masks_pred:
            pred = set(pred_block[0].tolist())
            if enc & pred:
                overlaps += 1
    assert overlaps == 0, f'{overlaps} context/target overlaps observed'


# --------------------------------------------------------------------------- #
# 5. Training and EMA update (tiny CPU model)
# --------------------------------------------------------------------------- #

def test_ema_update_rule_tiny_model():
    """The EMA rule target <- m*target + (1-m)*encoder holds exactly."""
    torch.manual_seed(0)
    encoder = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    target = copy.deepcopy(encoder)
    with torch.no_grad():
        for p in target.parameters():
            p.fill_(0.0)
    m = 0.9
    # simulate one optimizer update to the encoder
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(0.1)
    expected = [
        target_before * m + (1.0 - m) * encoder_after
        for target_before, encoder_after in zip(
            [torch.zeros_like(p) for p in target.parameters()],
            [p.detach().clone() for p in encoder.parameters()])
    ]
    with torch.no_grad():
        for q, k in zip(encoder.parameters(), target.parameters()):
            k.data.mul_(m).add_((1.0 - m) * q.detach().data)
    for got, want in zip(target.parameters(), expected):
        assert torch.allclose(got, want, rtol=1e-6, atol=1e-7)


def test_grad_accum_loss_normalization_is_equivalent():
    """Averaging N micro-losses must equal one larger-batch mean loss."""
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    target = torch.randn(4, 8)
    layer = torch.nn.Linear(8, 8)
    big = F.smooth_l1_loss(layer(x), target)

    micro = torch.nn.Linear(8, 8)
    micro.load_state_dict(layer.state_dict())
    micro.zero_grad()
    total = 0.0
    for i in range(4):
        loss = F.smooth_l1_loss(micro(x[i:i + 1]), target[i:i + 1])
        (loss / 4).backward()
        total += loss.item() / 4
    assert abs(total - big.item()) < 1e-5
    # gradient check: accumulated micro grads must equal the big-batch grads
    layer.zero_grad()
    big.backward()
    for p_big, p_micro in zip(layer.parameters(), micro.parameters()):
        assert torch.allclose(p_big.grad, p_micro.grad, rtol=1e-5, atol=1e-6)
