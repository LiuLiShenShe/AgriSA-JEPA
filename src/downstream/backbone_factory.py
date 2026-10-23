"""Backbone factory for every downstream evaluation.

One place decides which weights a backbone name maps to, what its SHA-256 is,
and whether it is frozen.  A cache key derived from the hash makes it
impossible for two backbones to share cached features.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

from src.downstream.features import PATCH_SIZE, build_encoder, sha256_file

OFFICIAL_IJEPA = '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar'
AGRI_JEPA_2000 = '/home/test/agrisa_runs/stage3b_random_2000/jepa-encoder-step2000.pth'


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    checkpoint: str
    label: str
    continued_training: str
    data_version: Optional[str]
    frozen: bool = True
    input_size: int = 518

    @property
    def checkpoint_sha256(self) -> str:
        return sha256_file(self.checkpoint)

    @property
    def cache_key(self) -> str:
        payload = (f'{self.label}|{self.checkpoint_sha256}|'
                   f'input={self.input_size}|frozen={self.frozen}')
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


#: The two arms of every Stage 5A A/B comparison.  ``B`` is the 2,000-update
#: agricultural continued-pretraining encoder, not a 500-step pilot.
STAGE5A_ARMS: Dict[str, BackboneSpec] = {
    'A': BackboneSpec(
        name='A',
        checkpoint=OFFICIAL_IJEPA,
        label='official_i-jepa_vith14_300e',
        continued_training='none (ImageNet-1K, 300 epochs)',
        data_version=None,
    ),
    'B': BackboneSpec(
        name='B',
        checkpoint=AGRI_JEPA_2000,
        label='agricultural_jepa_vith14_2000step',
        continued_training='2,000 optimizer updates, official I-JEPA objective, '
                           'agrifield40k-v1 train split',
        data_version='agrifield40k-v1 (PROVISIONAL)',
        input_size=224,   # encoder snapshot itself is resolution-agnostic
    ),
}

LEGACY_ARMS: Dict[str, BackboneSpec] = {
    'A': BackboneSpec(
        name='A', checkpoint=OFFICIAL_IJEPA,
        label='official_i-jepa_vith14',
        continued_training='none', data_version=None, input_size=224),
    'B500': BackboneSpec(
        name='B500',
        checkpoint='runs/stage3_armB_pilot_500/jepa-latest.pth.tar',
        label='agricultural_jepa_vith14_500step',
        continued_training='500 optimizer updates',
        data_version='agrifield40k-v0 (PROVISIONAL)', input_size=224),
}


def resolve(name: str) -> BackboneSpec:
    if name in STAGE5A_ARMS:
        return STAGE5A_ARMS[name]
    if name in LEGACY_ARMS:
        return LEGACY_ARMS[name]
    raise KeyError(f'unknown backbone {name!r}; '
                   f'expected {sorted(STAGE5A_ARMS) + sorted(LEGACY_ARMS)}')


def build_frozen_encoder(spec: BackboneSpec, device: torch.device,
                         input_size: Optional[int] = None):
    """Strictly load a ViT-H/14 and freeze it; never silently partially loads."""
    encoder = build_encoder(spec.checkpoint, device)
    desired = input_size or spec.input_size
    if desired != 224:
        from src.downstream.vit_detection import resize_pos_embed
        grid = desired // PATCH_SIZE
        encoder.pos_embed = torch.nn.Parameter(
            resize_pos_embed(encoder.pos_embed.detach(), grid),
            requires_grad=False)
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval()
    return encoder
