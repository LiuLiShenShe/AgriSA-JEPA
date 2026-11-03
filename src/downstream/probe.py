"""Frozen-backbone linear segmentation probe over ViT-H patch tokens.

The probe is deliberately the simplest head that can be wrong in a diagnosable
way: one ``Linear(1280, 3)`` applied to every patch token, then a 16x16 class
grid per tile.  Both arms use this identical head; no arm gets a different
decoder.

Exact patch-level loss
----------------------
The natural loss is per-pixel cross-entropy after nearest-upsampling the 16x16
class grid by 14 to 224x224.  Because nearest upsampling assigns the same logits
to every pixel of a 14x14 patch, that loss is exactly::

    loss = sum_{patch, pixel in patch} -log softmax(logits_patch)[y_pixel]
           / total_pixels
         = sum_{patch} sum_c count[patch, c] * -log_softmax(logits_patch)[c]
           / total_pixels

where ``count[patch, c]`` is the number of pixels of class ``c`` in that patch.
Computing it this way is mathematically identical and avoids materialising a
``[N, 224, 224]`` label tensor (about 10 GB for the training split).  The
prediction path still stitches at full resolution, so the reported metric is a
full-resolution score.

Masking
-------
* Uncovered border pixels (no tile predicts them) are excluded from both the loss
  and the self-implemented metric.
* Every covered pixel is supervised, including partial crop/weed, because the
  official metric maps ``3 -> 1`` and ``4 -> 2`` before scoring.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.downstream.features import EMBED_DIM, PATCH_SIZE, TOKEN_GRID
from src.downstream.tiling import TILE_SIZE


class LinearSegmentationProbe(nn.Module):
    """``Linear(embed_dim -> num_classes)`` over patch tokens."""

    def __init__(self, embed_dim: int = EMBED_DIM, num_classes: int = 3) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[N, 256, D]`` -> logits ``[N, C, 16, 16]``."""
        if tokens.ndim != 3:
            raise ValueError(f'Expected [N, tokens, D] tokens, got {tuple(tokens.shape)}')
        logits = self.head(tokens)                       # [N, 256, C]
        n = logits.shape[0]
        logits = logits.reshape(n, TOKEN_GRID, TOKEN_GRID, -1).permute(0, 3, 1, 2)
        return logits                                    # [N, C, 16, 16]

    @torch.no_grad()
    def predict_tiles(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-pixel tile predictions ``[N, 224, 224]`` int64, nearest upsample."""
        logits = self.forward(tokens)
        up = F.interpolate(logits, size=(TILE_SIZE, TILE_SIZE), mode='nearest')
        return up.argmax(dim=1)

    @torch.no_grad()
    def predict_tile_logits(self, tokens: torch.Tensor) -> np.ndarray:
        """Per-pixel tile logits ``[N, C, 224, 224]`` float32 (for overlap averaging)."""
        logits = self.forward(tokens)
        up = F.interpolate(logits, size=(TILE_SIZE, TILE_SIZE), mode='nearest')
        return up.float().cpu().numpy()


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def patch_class_counts(label_tiles: np.ndarray, num_classes: int = 3,
                       patch: int = PATCH_SIZE) -> np.ndarray:
    """Per-patch class histograms ``[N, 16, 16, C]`` int32.

    ``label_tiles`` is ``[N, 224, 224]`` with labels already mapped to the
    official 3-class space.  Summing the 14x14 block of each patch gives the
    exact pixel counts the upsampled loss would see.
    """
    n, h, w = label_tiles.shape
    if h % patch or w % patch:
        raise ValueError(f'label tile {h}x{w} not divisible by patch {patch}')
    gh, gw = h // patch, w // patch
    flat = label_tiles.reshape(n, gh, patch, gw, patch).transpose(0, 1, 3, 2, 4)
    flat = flat.reshape(n, gh, gw, patch * patch)
    counts = np.zeros((n, gh, gw, num_classes), dtype=np.int32)
    for c in range(num_classes):
        counts[..., c] = (flat == c).sum(axis=-1)
    return counts


def patch_ce_loss(logits: torch.Tensor, counts: torch.Tensor,
                  coverage: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Exact nearest-upsample pixel cross-entropy from patch class counts.

    ``logits`` is ``[B, C, 16, 16]``, ``counts`` is ``[B, 16, 16, C]`` (float or
    int), ``coverage`` is an optional ``[B, 16, 16]`` weight (1 = supervised).
    """
    b, c, gh, gw = logits.shape
    log_probs = F.log_softmax(logits, dim=1)                 # [B,C,16,16]
    log_probs = log_probs.permute(0, 2, 3, 1)                # [B,16,16,C]
    per_patch = -(counts * log_probs).sum(dim=-1)            # [B,16,16]
    if coverage is not None:
        per_patch = per_patch * coverage
        denominator = (counts.sum(dim=-1) * coverage).sum().clamp_min(1.0)
    else:
        denominator = counts.sum().clamp_min(1.0)
    return per_patch.sum() / denominator


def train_probe(
    probe: LinearSegmentationProbe,
    train_tokens: torch.Tensor,
    train_counts: torch.Tensor,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: torch.device = torch.device('cpu'),
    verbose: bool = False,
    log_every: int = 0,
) -> Dict[str, object]:
    """Train the probe on cached tile features and per-patch class counts.

    ``train_tokens`` is ``[N, 256, 1280]`` (fp16 is fine and is cast per batch),
    ``train_counts`` is ``[N, 16, 16, 3]`` int32 from :func:`patch_class_counts`.

    Features stay on CPU; each mini-batch is moved to the device, so memory does
    not scale with the size of the training split.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    probe.to(device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    n = train_tokens.shape[0]
    generator = torch.Generator().manual_seed(seed)
    history, updates = [], 0
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total_loss, total_pixels = 0.0, 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            tokens = train_tokens[idx].to(device).float()
            counts = train_counts[idx].to(device).float()
            logits = probe(tokens)                              # [B,C,16,16]
            loss = patch_ce_loss(logits, counts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            pixels = float(counts.sum().item())
            total_loss += float(loss.item()) * pixels
            total_pixels += pixels
            if log_every and updates % log_every == 0:
                print(f'    update {updates} loss {loss.item():.4f}', flush=True)
        epoch_loss = total_loss / max(total_pixels, 1.0)
        history.append(epoch_loss)
        if verbose:
            print(f'  epoch {epoch + 1}/{epochs} loss {epoch_loss:.4f}', flush=True)
    probe.eval()
    return {'loss_history': history, 'updates': updates,
            'final_loss': history[-1] if history else None}


class IoUAccumulator:
    """Per-class IoU over full-resolution images, matching the official metric.

    ``update`` takes a ``[H,W]`` prediction and label and an optional validity
    mask; only valid pixels contribute.  ``compute`` returns per-class IoU in
    ``[0, 1]``.  A class with zero union contributes 1.0, which is the
    torchmetrics ``MulticlassJaccardIndex`` convention for an absent class.
    """

    def __init__(self, num_classes: int = 3) -> None:
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes, dtype=np.int64)
        self.union = np.zeros(num_classes, dtype=np.int64)

    def update(self, prediction: np.ndarray, label: np.ndarray,
               valid: Optional[np.ndarray] = None) -> None:
        pred = prediction.ravel()
        true = label.ravel()
        if valid is not None:
            keep = valid.ravel()
            pred = pred[keep]
            true = true[keep]
        for c in range(self.num_classes):
            p = pred == c
            t = true == c
            self.intersection[c] += int(np.count_nonzero(p & t))
            self.union[c] += int(np.count_nonzero(p | t))

    def compute(self) -> Dict[int, float]:
        # plain Python floats so the result is JSON-serializable
        return {c: (1.0 if self.union[c] == 0
                    else float(self.intersection[c]) / float(self.union[c]))
                for c in range(self.num_classes)}