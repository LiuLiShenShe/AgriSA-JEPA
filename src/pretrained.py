"""Official I-JEPA pretrained-weight initialization for AgriSA-JEPA Stage 2.

This module implements **Operation A: Pretrained Initialization**.  It reads an
official facebookresearch/ijepa checkpoint (e.g. ``IN1K-vit.h.14-300e.pth.tar``)
and transfers ONLY the model weights into a freshly constructed agricultural
model.  The optimizer, LR/weight-decay schedules, scaler and epoch counters are
deliberately NOT transferred: those belong to a new agricultural experiment.

Operation A must never be confused with **Operation B: Training Resume**, which
restores ``encoder``/``predictor``/``target_encoder``/``opt``/``scaler`` plus the
schedules, EMA progress and RNG state of *this* project.  Operation B lives in
``src/helper.py: load_checkpoint`` and ``src/train.py``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

#: Top-level fields that the official I-JEPA pretraining checkpoint contains.
EXPECTED_TOP_LEVEL_FIELDS = (
    'encoder',
    'predictor',
    'target_encoder',
    'opt',
    'scaler',
    'epoch',
)

#: Fields that carry model weights and may be transferred by Operation A.
WEIGHT_FIELDS = ('encoder', 'predictor', 'target_encoder')


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file and return its SHA-256 hex digest."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop the ``module.`` prefix that DistributedDataParallel adds.

    Returns a new dict; the input is not mutated.
    """
    prefix = 'module.'
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def load_pretrained_payload(
    path: str | Path,
    weights_only: bool = True,
    trusted: bool = False,
) -> Tuple[Mapping, str]:
    """Safely deserialize an official I-JEPA checkpoint.

    ``weights_only=True`` is attempted first so that no arbitrary Python objects
    are unpickled.  Official Meta checkpoints contain only plain containers and
    tensors for the weight fields, so this normally succeeds.  If it does not,
    the caller is told explicitly which mode was used instead of silently
    falling back.

    ``trusted=True`` skips the ``weights_only=True`` attempt and deserializes
    directly with ``weights_only=False``.  The caller must only pass it after
    verifying the file's SHA-256 (as ``src/train.py`` does for the configured
    official checkpoint): the official payload contains optimizer state that
    ``weights_only=True`` cannot represent, so the safe-first path otherwise
    reads the whole multi-gigabyte file twice on every start.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Pretrained checkpoint not found: {path}')
    if trusted:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f'Unexpected checkpoint payload type: {type(payload).__name__}')
        return payload, 'weights_only=False (trusted, SHA-256 verified)'
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        mode = 'weights_only=True'
    except Exception as weights_only_error:  # pragma: no cover - depends on file
        logger.warning(
            'weights_only=True failed for %s (%s); falling back to '
            'weights_only=False. Only do this for checkpoints whose provenance '
            'you trust.',
            path,
            type(weights_only_error).__name__,
        )
        payload = torch.load(path, map_location='cpu', weights_only=False)
        mode = 'weights_only=False'
    if not isinstance(payload, Mapping):
        raise TypeError(f'Unexpected checkpoint payload type: {type(payload).__name__}')
    return payload, mode


def describe_payload(payload: Mapping) -> Dict[str, object]:
    """Summarise the top-level structure of an official checkpoint payload."""
    summary: Dict[str, object] = {'top_level_keys': sorted(map(str, payload.keys()))}
    for field in WEIGHT_FIELDS:
        state = payload.get(field)
        if isinstance(state, Mapping):
            stripped = strip_module_prefix(state)
            summary[f'{field}_tensors'] = len(stripped)
            summary[f'{field}_prefixed'] = any(str(k).startswith('module.') for k in state)
    for scalar in ('epoch', 'loss', 'batch_size', 'world_size', 'lr'):
        if scalar in payload:
            summary[scalar] = payload[scalar]
    return summary


def compare_state_dicts(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Return an exact, human-readable diff between two state dicts.

    Both inputs must already be prefix-normalized.  Nothing is hidden behind
    ``strict=False``: any missing, unexpected or shape-mismatched tensor is
    listed so the caller can decide explicitly what to do.
    """
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatch = sorted(
        (key, tuple(model_state[key].shape), tuple(checkpoint_state[key].shape))
        for key in model_keys & checkpoint_keys
        if tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)
    )
    matched = len(model_keys & checkpoint_keys) - len(shape_mismatch)
    return {
        'matched': matched,
        'model_tensors': len(model_keys),
        'checkpoint_tensors': len(checkpoint_keys),
        'missing': missing,
        'unexpected': unexpected,
        'shape_mismatch': shape_mismatch,
        'is_exact': not missing and not unexpected and not shape_mismatch,
    }


def _load_module(module: nn.Module, checkpoint_state: Mapping[str, torch.Tensor], label: str):
    """Load one weight field with strict key/shape validation."""
    model_state = {key: value for key, value in module.state_dict().items()}
    diff = compare_state_dicts(model_state, strip_module_prefix(checkpoint_state))
    if not diff['is_exact']:
        raise RuntimeError(
            f'Refusing to load {label}: checkpoint does not match the model exactly.\n'
            f"  matched tensors : {diff['matched']}/{diff['model_tensors']}\n"
            f"  missing         : {diff['missing'][:10]}{' ...' if len(diff['missing']) > 10 else ''}\n"
            f"  unexpected      : {diff['unexpected'][:10]}{' ...' if len(diff['unexpected']) > 10 else ''}\n"
            f"  shape mismatch  : {diff['shape_mismatch'][:5]}"
        )
    # strict=True after an explicit, verified comparison: no hidden tolerance.
    module.load_state_dict(strip_module_prefix(checkpoint_state), strict=True)
    return diff


def init_from_pretrained(
    path: str | Path,
    encoder: nn.Module,
    predictor: nn.Module,
    target_encoder: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    sha256: Optional[str] = None,
    trusted: bool = False,
) -> Dict[str, object]:
    """Operation A: initialize the model weights from an official I-JEPA run.

    Encoder, target encoder and the shape-compatible predictor are all loaded
    from the checkpoint.  Returns a report dict describing exactly what was
    transferred.  Optimizer/scheduler/scaler/epoch are intentionally untouched.

    ``sha256`` may be supplied by a caller that has already verified the file,
    which avoids re-hashing multiple gigabytes here.  ``trusted`` is forwarded
    to :func:`load_pretrained_payload` and must only be set together with a
    verified digest.
    """
    payload, load_mode = load_pretrained_payload(path, trusted=trusted)
    missing_fields = [f for f in WEIGHT_FIELDS if f not in payload]
    if missing_fields:
        raise KeyError(
            f'Official checkpoint is missing weight fields {missing_fields}; '
            f'found {sorted(map(str, payload.keys()))}'
        )

    report: Dict[str, object] = {
        'path': str(path),
        'sha256': sha256 if sha256 is not None else sha256_file(path),
        'load_mode': load_mode,
        'payload': describe_payload(payload),
        'transferred': {},
        'predictor_inherited': None,
    }

    report['transferred']['encoder'] = _load_module(
        encoder, payload['encoder'], 'encoder')
    report['transferred']['predictor'] = _load_module(
        predictor, payload['predictor'], 'predictor')
    if target_encoder is not None:
        report['transferred']['target_encoder'] = _load_module(
            target_encoder, payload['target_encoder'], 'target_encoder')

    if device is not None:
        encoder.to(device)
        predictor.to(device)
        if target_encoder is not None:
            target_encoder.to(device)

    # Record whether the predictor was inherited completely (it always is when
    # the strict comparison above passed, but the flag documents the contract).
    report['predictor_inherited'] = bool(report['transferred']['predictor']['is_exact'])
    report['encoder_keys_loaded'] = report['transferred']['encoder']['matched']
    report['predictor_keys_loaded'] = report['transferred']['predictor']['matched']
    if target_encoder is not None:
        report['target_encoder_keys_loaded'] = report['transferred']['target_encoder']['matched']
    report['epoch_in_source_checkpoint'] = payload.get('epoch')

    logger.info(
        'Operation A complete: encoder=%d predictor=%d target_encoder=%s tensors from %s (%s)',
        report['encoder_keys_loaded'],
        report['predictor_keys_loaded'],
        report.get('target_encoder_keys_loaded'),
        path,
        load_mode,
    )
    return report


__all__ = [
    'EXPECTED_TOP_LEVEL_FIELDS',
    'WEIGHT_FIELDS',
    'sha256_file',
    'strip_module_prefix',
    'load_pretrained_payload',
    'describe_payload',
    'compare_state_dicts',
    'init_from_pretrained',
]
