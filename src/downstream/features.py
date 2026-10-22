"""Frozen ViT-H/14 feature extraction for downstream segmentation probing.

Two arms, one encoder architecture, one extraction path:

* **Arm A** — official ``IN1K-vit.h.14-300e.pth.tar``, untouched.
* **Arm B** — ``runs/stage3_armB_pilot_500/jepa-latest.pth.tar``, the same
  architecture after 500 agricultural continued-pretraining updates.

The encoder is always frozen (``requires_grad=False``, ``eval()``).  This module
never trains it; it only turns 224x224 tiles into 16x16x1280 patch features.

Which weights are read
----------------------
Arm A's official checkpoint stores ``encoder`` / ``predictor`` /
``target_encoder``.  Arm B's training checkpoint stores the same three fields
plus optimizer/schedule state, and its tensors carry a ``module.`` prefix from
DDP.  Both are handled by :func:`load_backbone_state`, which strips the prefix
and refuses any key/shape mismatch rather than loading partially.

Only the *context encoder* is used for downstream evaluation.  The target
encoder is an EMA copy used to build JEPA targets during pretraining; it is not
a feature extractor and is never used here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from src.models.vision_transformer import vit_huge
from src.pretrained import compare_state_dicts, strip_module_prefix

PATCH_SIZE = 14
TILE_SIZE = 224
TOKEN_GRID = TILE_SIZE // PATCH_SIZE   # 16
TOKENS_PER_TILE = TOKEN_GRID * TOKEN_GRID   # 256
EMBED_DIM = 1280

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_backbone_state(checkpoint_path: str | Path,
                        field: str = 'encoder') -> Dict[str, torch.Tensor]:
    """Load and prefix-normalize one weight field from either checkpoint type."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Backbone checkpoint not found: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f'Unexpected checkpoint payload: {type(payload).__name__}')
    if field not in payload:
        raise KeyError(
            f'{path} has no {field!r} field; found {sorted(map(str, payload.keys()))[:10]}')
    state = strip_module_prefix(payload[field])
    return {k: v for k, v in state.items()}


def build_encoder(checkpoint_path: str | Path, device: torch.device,
                  field: str = 'encoder', use_sdpa: bool = True) -> torch.nn.Module:
    """Construct ViT-H/14 @224 and load weights strictly; return a frozen encoder."""
    encoder = vit_huge(img_size=[TILE_SIZE], patch_size=PATCH_SIZE, use_sdpa=use_sdpa)
    model_state = encoder.state_dict()
    state = load_backbone_state(checkpoint_path, field=field)
    diff = compare_state_dicts(model_state, state)
    if not diff['is_exact']:
        raise RuntimeError(
            f'Refusing to load {checkpoint_path}: backbone does not match exactly.\n'
            f"  matched        : {diff['matched']}/{diff['model_tensors']}\n"
            f"  missing        : {diff['missing'][:5]}\n"
            f"  unexpected     : {diff['unexpected'][:5]}\n"
            f"  shape mismatch : {diff['shape_mismatch'][:3]}")
    encoder.load_state_dict(state, strict=True)
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def normalize_tiles(tiles: np.ndarray, mean: Sequence[float] = IMAGENET_MEAN,
                    std: Sequence[float] = IMAGENET_STD) -> torch.Tensor:
    """uint8 ``[N,H,W,3]`` tiles -> normalized float32 ``[N,3,H,W]`` tensor."""
    x = torch.from_numpy(np.ascontiguousarray(tiles)).float().div_(255.0)
    x = x.permute(0, 3, 1, 2)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


@torch.no_grad()
def extract_tile_features(encoder: torch.nn.Module, tiles: np.ndarray,
                          device: torch.device, batch_size: int = 16,
                          use_bfloat16: bool = True,
                          return_grid: bool = False):
    """Run the frozen encoder over ``[N,224,224,3]`` tiles.

    Returns ``[N, 256, 1280]`` patch tokens in row-major patch order, or
    ``[N, 16, 16, 1280]`` when ``return_grid=True``.
    """
    if tiles.ndim != 4 or tiles.shape[1:] != (TILE_SIZE, TILE_SIZE, 3):
        raise ValueError(f'Expected [N,{TILE_SIZE},{TILE_SIZE},3] tiles, got {tiles.shape}')
    feats = []
    for start in range(0, len(tiles), batch_size):
        batch = normalize_tiles(tiles[start:start + batch_size]).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=use_bfloat16 and device.type == 'cuda'):
            out = encoder(batch)
        feats.append(out.float().cpu())
    tokens = torch.cat(feats, dim=0)
    if tokens.shape[1:] != (TOKENS_PER_TILE, EMBED_DIM):
        raise RuntimeError(
            f'Encoder produced {tuple(tokens.shape[1:])}, expected '
            f'({TOKENS_PER_TILE}, {EMBED_DIM})')
    if return_grid:
        return tokens.view(len(tokens), TOKEN_GRID, TOKEN_GRID, EMBED_DIM)
    return tokens


def arm_config(arm: str) -> Dict[str, object]:
    """The two arms of the downstream comparison.  No other arm exists here."""
    if arm == 'A':
        return {
            'arm': 'A',
            'label': 'official_i-jepa_vith14',
            'checkpoint': '/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar',
            'continued_training': 'none',
            'data_version': None,
        }
    if arm == 'B':
        return {
            'arm': 'B',
            'label': 'agricultural_jepa_vith14_500step',
            'checkpoint': 'runs/stage3_armB_pilot_500/jepa-latest.pth.tar',
            'continued_training': '500 optimizer updates, standard I-JEPA objective',
            'data_version': 'agrifield40k-v0 (PROVISIONAL)',
        }
    raise ValueError(f"Unknown arm {arm!r}; expected 'A' or 'B'")


def checkpoint_config(checkpoint_path: str | Path, label: str,
                      continued_training: str = 'unknown',
                      data_version: Optional[str] = None) -> Dict[str, object]:
    """Describe an arbitrary backbone checkpoint for the downstream evaluation.

    Used by the Stage 3B / Stage 4A arms, whose backbones are mid-run encoder
    snapshots rather than the two legacy arms.  The checkpoint's SHA-256 is
    recorded so a feature cache can never be reused across different weights.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Backbone checkpoint not found: {path}')
    return {
        'arm': label,
        'label': label,
        'checkpoint': str(path),
        'checkpoint_sha256': sha256_file(path),
        'continued_training': continued_training,
        'data_version': data_version,
    }


def cache_key(label: str, checkpoint_sha256: str) -> str:
    """Feature-cache key.

    The key binds the cache to the *weights*, not just to an arm name, so a
    re-trained arm can never silently reuse another arm's features.
    """
    return f'{label}__{checkpoint_sha256[:16]}'