"""Stage 8 backbone registry: the five primary backbones plus exploratory C.

Every backbone is a *frozen* ViT-H/14 encoder.  We reuse
``backbone_factory.BackboneSpec`` so cache keys stay a function of
(label, checkpoint SHA, input size, frozen) and two backbones can never share a
feature cache.  All Stage 7B checkpoints are the *encoder snapshots*
(``jepa-encoder-step2000.pth``) whose SHA-256 was recorded at probe time; the
registry asserts each one matches the recorded value before it is allowed to
run a single experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import hashlib

from src.downstream.backbone_factory import BackboneSpec
from src.downstream.features import sha256_file

STAGE7B_ROOT = Path('/data/agrisa_jepa_runs/stage7b')

OFFICIAL_IJEPA = '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar'
OFFICIAL_SHA = '0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22'


@dataclass(frozen=True)
class Stage8Backbone:
    key: str
    arm: str            # 'OFFICIAL' | 'A' | 'B' | 'C'
    pretraining_seed: Optional[int]
    checkpoint: str
    label: str
    horizon: Optional[int]
    warmup: Optional[int]
    updates: int
    role: str           # 'baseline' | 'primary' | 'exploratory'
    continued_training: str
    data_version: str
    input_size: int = 518

    @property
    def checkpoint_sha256(self) -> str:
        return sha256_file(self.checkpoint)

    def to_spec(self, input_size: Optional[int] = None) -> BackboneSpec:
        return BackboneSpec(
            name=self.key, checkpoint=self.checkpoint, label=self.label,
            continued_training=self.continued_training,
            data_version=self.data_version,
            input_size=input_size or self.input_size)


# The five primary backbones.  Order matters: Official first (baseline), then
# the paired A/B arms in p1, p2 order.
PRIMARY = ('OFFICIAL', 'A_p1', 'A_p2', 'B_p1', 'B_p2')
EXPLORATORY = ('C_p1', 'C_p2')
PRETRAIN_SEEDS = {'p1': 20261104, 'p2': 20261105}
DOWNSTREAM_SEEDS = (0, 1, 2)


def _stage7b(name: str, seed_tag: str, arm: str, horizon: int, warmup: int,
             role: str) -> Stage8Backbone:
    tag = seed_tag.upper()
    label = (f'stage7b_{name}__arm{arm}_h{horizon}_w{warmup}_'
             f'pseed{PRETRAIN_SEEDS[seed_tag]}')
    sched = (f'{horizon}-step horizon, {warmup}-step warmup, 2,000 updates'
             if horizon else '')
    return Stage8Backbone(
        key=name, arm=arm, pretraining_seed=PRETRAIN_SEEDS[seed_tag],
        checkpoint=str(STAGE7B_ROOT / f'stage7b_{name}' /
                       'jepa-encoder-step2000.pth'),
        label=label, horizon=horizon, warmup=warmup, updates=2000,
        role=role,
        continued_training=f'agricultural continued pretraining ({sched})',
        data_version='agrifield40k-v1 R3v3 (147,277 images)')


def build_registry(include_exploratory: bool = True) -> Dict[str, Stage8Backbone]:
    reg: Dict[str, Stage8Backbone] = {
        'OFFICIAL': Stage8Backbone(
            key='OFFICIAL', arm='OFFICIAL', pretraining_seed=None,
            checkpoint=OFFICIAL_IJEPA,
            label='official_i-jepa_vith14_300e', horizon=None, warmup=None,
            updates=0, role='baseline',
            continued_training='none (ImageNet-1K, 300 epochs)',
            data_version=None),
        'A_p1': _stage7b('A_p1', 'p1', 'A', 2000, 100, 'primary'),
        'A_p2': _stage7b('A_p2', 'p2', 'A', 2000, 100, 'primary'),
        'B_p1': _stage7b('B_p1', 'p1', 'B', 50000, 100, 'primary'),
        'B_p2': _stage7b('B_p2', 'p2', 'B', 50000, 100, 'primary'),
    }
    if include_exploratory:
        reg['C_p1'] = _stage7b('C_p1', 'p1', 'C', 50000, 2500, 'exploratory')
        reg['C_p2'] = _stage7b('C_p2', 'p2', 'C', 50000, 2500, 'exploratory')
    return reg


# Recorded encoder SHA-256 values, read from the authoritative Stage 7B probe
# manifest.  Never hard-code hashes by hand: the registry re-computes each
# checkpoint's SHA-256 and asserts equality against the manifest before any
# experiment is allowed to run.
_STAGE7B_MANIFEST_PATH = (Path(__file__).resolve().parent.parent
                          / 'reports' / 'stage7b_phenobench_manifest.json')


def recorded_checkpoint_sha256() -> Dict[str, str]:
    import json
    manifest = json.loads(_STAGE7B_MANIFEST_PATH.read_text(encoding='utf-8'))
    return {e['run_id']: e['checkpoint_sha256'] for e in manifest['entries']}


def verify_checkpoint_sha256(key: str,
                             recorded: Optional[Dict[str, str]] = None) -> str:
    """SHA-256 of a registered checkpoint, asserted equal to the recorded value.

    A mismatch is a hard error: a wrong-weight backbone would silently invalidate
    every downstream comparison, so nothing is allowed to proceed on one.
    """
    reg = build_registry(include_exploratory=True)
    spec = reg[key]
    actual = spec.checkpoint_sha256
    expected = (recorded if recorded is not None
                else recorded_checkpoint_sha256()).get(key)
    if expected is None:
        raise KeyError(f'{key}: no recorded SHA-256 in Stage 7B manifest')
    if actual != expected:
        raise AssertionError(
            f'{key}: checkpoint SHA-256 mismatch\n  actual   {actual}\n'
            f'  expected {expected}')
    return actual


__all__ = ['Stage8Backbone', 'build_registry', 'verify_checkpoint_sha256',
           'recorded_checkpoint_sha256',
           'PRIMARY', 'EXPLORATORY', 'PRETRAIN_SEEDS', 'DOWNSTREAM_SEEDS',
           'OFFICIAL_IJEPA', 'OFFICIAL_SHA']
