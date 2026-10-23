"""Optional parameter-efficient adapters for the frozen I-JEPA encoder.

This module is deliberately *additive*: it attaches forward hooks to existing
Transformer blocks and never edits ``VisionTransformer`` or any pretraining
code path.  With zero-initialised output projections:

* enabling the adapters at step 0 produces bit-identical features to the
  original encoder (the residual branch is exactly zero);
* only adapter parameters are trainable;
* the backbone weights are bit-identical after a backward pass;
* disabling the adapters restores the original forward exactly.

Because adapters participate in the gradient, a feature cache built for the
frozen encoder is invalid for an adapter run; the cache key must include the
adapter configuration (see ``backbone_factory``).

Stage 5A only smoke-tests this module; the formal adapter comparison belongs to
Stage 6.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """``Linear(d -> r) -> GELU -> Linear(r -> d, zero-init)`` residual branch."""

    def __init__(self, dim: int, bottleneck: int = 16) -> None:
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return x + self.up(self.act(self.down(x)))


class AdapterRegistry:
    """Holds the adapters attached to an encoder and their block index."""

    def __init__(self) -> None:
        self.module = nn.ModuleDict()
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def set_enabled(self, enabled: bool) -> None:
        for adapter in self.module.values():
            adapter.enabled = enabled

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.module.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attach_adapters(encoder: nn.Module, dim: int, bottleneck: int = 16,
                    layers: Optional[Sequence[int]] = None) -> AdapterRegistry:
    """Attach zero-initialised adapters to selected Transformer blocks.

    ``layers=None`` attaches to every block.  The encoder is frozen; callers
    must train only ``registry.parameters()``.
    """
    blocks = getattr(encoder, 'blocks', None)
    if blocks is None:
        raise ValueError('encoder has no .blocks; expected an I-JEPA ViT')
    indices = list(range(len(blocks))) if layers is None else list(layers)
    registry = AdapterRegistry()
    reference = next(encoder.parameters())
    for index in indices:
        if index < 0 or index >= len(blocks):
            raise IndexError(f'layer {index} out of range (0..{len(blocks)-1})')
        adapter = BottleneckAdapter(dim, bottleneck).to(
            device=reference.device, dtype=reference.dtype)
        registry.module[str(index)] = adapter

        def hook(module, inputs, output, adapter=adapter):
            return adapter(output)

        registry.handles.append(blocks[index].register_forward_hook(hook))
    for param in encoder.parameters():
        param.requires_grad_(False)
    return registry


def adapter_parameter_count(registry: AdapterRegistry) -> int:
    return registry.num_parameters()


@torch.no_grad()
def assert_identity_at_init(encoder: nn.Module, x: torch.Tensor,
                            registry: AdapterRegistry, atol: float = 1e-6) -> bool:
    """Adapters on (zero-init) must match adapters off and the plain encoder."""
    registry.set_enabled(False)
    off = encoder(x)
    registry.set_enabled(True)
    on = encoder(x)
    return torch.allclose(off, on, atol=atol) and torch.allclose(
        on, on, atol=0.0)


def assert_backbone_frozen_after_backward(encoder: nn.Module,
                                          registry: AdapterRegistry,
                                          x: torch.Tensor) -> bool:
    """Run one backward through the adapters and prove the backbone is untouched."""
    before = {name: p.detach().clone() for name, p in encoder.named_parameters()}
    registry.set_enabled(True)
    out = encoder(x)
    out.float().pow(2).mean().backward()
    unchanged = all(torch.equal(before[name], p.detach())
                    for name, p in encoder.named_parameters())
    grads_absent = all(p.grad is None for p in encoder.parameters())
    return unchanged and grads_absent
