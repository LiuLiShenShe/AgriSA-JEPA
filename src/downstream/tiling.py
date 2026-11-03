"""224x224 tiling and stitching back to the source resolution.

The frozen ViT-H/14 was pretrained at 224x224 (16x16 patch tokens).  PhenoBench
images are 1024x1024.  Resizing to 224 throws away the fine plant boundaries this
task is about, so the encoder runs over a regular grid of 224x224 tiles and the
patch predictions are stitched back to the original grid.

Tile rule (identical for every arm, every image, every run)
-----------------------------------------------------------
For a ``size``-pixel axis and ``tile``-pixel window with ``n`` tiles along that
axis::

    n      = ceil(size / tile)
    stride = ceil((size - tile) / (n - 1))
    offsets = [min(i * stride, size - tile) for i in range(n)]

With ``size=1024``, ``tile=224`` this is ``n=5``, ``stride=200``, offsets
``0, 200, 400, 600, 800``: every pixel is covered, the last tile ends exactly at
1024, and there is **no padding** and **no cropping of image content**.  Adjacent
tiles overlap by 24 pixels.

Stitching
---------
Each tile yields 16x16 patch logits, upsampled by 14 to 224x224.  Overlapping
pixels receive the mean of the per-class logits of every tile covering them, and
the class is the argmax of that mean.  So a pixel's prediction never depends on
which single tile happened to cover it.

Past version
------------
An earlier revision used non-overlapping fully-inside tiles (4x4, covering only
0..895).  That left the 128-pixel border unpredicted, which made the self-metric
(interior only) disagree with the official full-image metric.  This revision
covers the whole image so the two agree by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

TILE_SIZE = 224


@dataclass(frozen=True)
class Tile:
    """One 224x224 window in source-image pixel coordinates."""
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


def tile_offsets(size: int, tile: int = TILE_SIZE) -> List[int]:
    """Uniform tile offsets covering ``[0, size)`` with no padding and no gaps."""
    if size < tile:
        raise ValueError(f'Axis {size} is smaller than one {tile}x{tile} tile')
    n = math.ceil(size / tile)
    if n == 1:
        return [0]
    stride = math.ceil((size - tile) / (n - 1))
    offsets = [min(i * stride, size - tile) for i in range(n)]
    if offsets[-1] + tile != size:
        raise AssertionError(f'last tile {offsets[-1]} + {tile} != {size}')
    if any(offsets[i + 1] <= offsets[i] for i in range(len(offsets) - 1)):
        raise AssertionError('tile offsets not strictly increasing')
    return offsets


def tile_coords(height: int, width: int, tile: int = TILE_SIZE) -> List[Tile]:
    """Row-major tile grid covering every pixel of a ``height x width`` image."""
    offsets_y = tile_offsets(height, tile)
    offsets_x = tile_offsets(width, tile)
    return [Tile(top=top, left=left, bottom=top + tile, right=left + tile)
            for top in offsets_y for left in offsets_x]


def n_tiles_per_axis(size: int, tile: int = TILE_SIZE) -> int:
    return len(tile_offsets(size, tile))


def covered_region(height: int, width: int, tile: int = TILE_SIZE) -> Tuple[int, int]:
    """Pixel extent covered by the tile grid (the whole image under this rule)."""
    return (height, width)


def covered_mask(height: int, width: int, tile: int = TILE_SIZE) -> np.ndarray:
    """Boolean mask of every pixel some tile predicts (all of them)."""
    offsets_y = tile_offsets(height, tile)
    offsets_x = tile_offsets(width, tile)
    mask = np.zeros((height, width), dtype=bool)
    for top in offsets_y:
        for left in offsets_x:
            mask[top:top + tile, left:left + tile] = True
    return mask


def tile_image(image: np.ndarray, tiles: Sequence[Tile]) -> np.ndarray:
    """Stack tiles into an ``[N, H, W, C]`` array."""
    out = np.empty((len(tiles), tiles[0].height, tiles[0].width, image.shape[2]),
                   dtype=image.dtype)
    for i, t in enumerate(tiles):
        out[i] = image[t.top:t.bottom, t.left:t.right]
    return out


def tile_label(label: np.ndarray, tiles: Sequence[Tile]) -> np.ndarray:
    """Stack label tiles into an ``[N, H, W]`` array."""
    out = np.empty((len(tiles), tiles[0].height, tiles[0].width), dtype=label.dtype)
    for i, t in enumerate(tiles):
        out[i] = label[t.top:t.bottom, t.left:t.right]
    return out


def crop_fraction(height: int, width: int, tile: int = TILE_SIZE) -> float:
    """Fraction of pixels covered by more than one tile (redundant compute)."""
    offsets_y = tile_offsets(height, tile)
    offsets_x = tile_offsets(width, tile)
    counts = np.zeros((height, width), dtype=np.int32)
    for top in offsets_y:
        for left in offsets_x:
            counts[top:top + tile, left:left + tile] += 1
    return float((counts > 1).mean())


def stitch(predictions: np.ndarray, height: int, width: int,
           tiles: Sequence[Tile], fill_value: int = -1) -> np.ndarray:
    """Hard-stitch per-tile label maps; later tiles in row-major order win.

    Provided for tests and for callers that have already reduced tiles to hard
    labels.  The probe itself uses :func:`stitch_logits` so overlaps are averaged
    instead of resolved by tile order.
    """
    out = np.full((height, width), fill_value, dtype=np.int16)
    for i, t in enumerate(tiles):
        pred = np.asarray(predictions[i])
        if pred.shape != (t.height, t.width):
            raise ValueError(
                f'prediction {i} has shape {pred.shape}, expected '
                f'({t.height}, {t.width})')
        out[t.top:t.bottom, t.left:t.right] = pred
    return out


def stitch_logits(tile_logits: np.ndarray, height: int, width: int,
                  tiles: Sequence[Tile]) -> np.ndarray:
    """Average per-class logits over overlapping tiles, then argmax.

    ``tile_logits`` is ``[N, C, 224, 224]`` in the same order as ``tiles``.
    Returns ``[H, W]`` int16 label map.
    """
    n, c, th, tw = tile_logits.shape
    acc = np.zeros((c, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for i, t in enumerate(tiles):
        acc[:, t.top:t.bottom, t.left:t.right] += tile_logits[i]
        counts[t.top:t.bottom, t.left:t.right] += 1.0
    missing = counts == 0
    if missing.any():
        raise ValueError(f'{int(missing.sum())} pixels are covered by no tile')
    acc /= counts[None, :, :]
    return acc.argmax(axis=0).astype(np.int16)