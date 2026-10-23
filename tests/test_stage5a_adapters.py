"""Stage 5A tests: non-invasive zero-init adapters on the frozen encoder."""

from __future__ import annotations

import pytest
import torch

from src.downstream.adapters import (adapter_parameter_count, attach_adapters,
                                     assert_backbone_frozen_after_backward,
                                     assert_identity_at_init)
from src.models.vision_transformer import vit_tiny


def _encoder():
    torch.manual_seed(0)
    encoder = vit_tiny(img_size=[224], patch_size=14)
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def test_zero_init_adapter_is_identity():
    encoder = _encoder().eval()
    registry = attach_adapters(encoder, dim=192, bottleneck=8, layers=[0, 1])
    x = torch.randn(1, 3, 224, 224)
    assert assert_identity_at_init(encoder, x, registry)


def test_adapter_disabled_restores_original_forward():
    encoder = _encoder().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        original = encoder(x)
    registry = attach_adapters(encoder, dim=192, bottleneck=8)
    registry.set_enabled(False)
    with torch.no_grad():
        disabled = encoder(x)
    assert torch.equal(original, disabled)
    registry.remove()


def test_only_adapter_parameters_are_trainable():
    encoder = _encoder()
    registry = attach_adapters(encoder, dim=192, bottleneck=8, layers=[0, 1])
    assert adapter_parameter_count(registry) == 2 * (192 * 8 + 8 + 8 * 192 + 192)
    for name, param in encoder.named_parameters():
        assert not param.requires_grad, name
    assert all(p.requires_grad for p in registry.parameters())


def test_backbone_unchanged_after_backward():
    encoder = _encoder().train()
    registry = attach_adapters(encoder, dim=192, bottleneck=8, layers=[0, 1])
    assert assert_backbone_frozen_after_backward(
        encoder, registry, torch.randn(1, 3, 224, 224))
    registry.remove()


def test_adapter_teaches_something_after_step():
    encoder = _encoder().train()
    registry = attach_adapters(encoder, dim=192, bottleneck=8, layers=[0])
    x = torch.randn(1, 3, 224, 224)
    optimizer = torch.optim.SGD(registry.parameters(), lr=0.1)
    before = [p.detach().clone() for p in registry.parameters()]
    out = encoder(x)
    out.pow(2).mean().backward()
    optimizer.step()
    after = list(registry.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    registry.remove()


def test_bad_layer_index_is_rejected():
    encoder = _encoder()
    with pytest.raises(IndexError):
        attach_adapters(encoder, dim=192, layers=[999])
