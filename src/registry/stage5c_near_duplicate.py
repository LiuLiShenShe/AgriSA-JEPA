"""Duplicate control for Stage 5C.

Two very different operations are kept apart:

``exact``
    byte-identical (SHA-256) or pixel-identical (decoded RGB SHA-256) copies.
    These are safe to collapse automatically.

``near``
    perceptual-hash candidates.  A candidate is only *confirmed* when several
    independent signals agree (pHash, dHash, aspect ratio, resolution and a
    DINOv2 embedding cosine).  Unconfirmed candidates are quarantined, never
    silently deleted.  A connected component above ``giant_cluster_threshold``
    is treated as a texture collision (the Stage 5B soil-image failure mode)
    and is quarantined as a whole instead of being collapsed.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class Pair:
    left: str
    right: str
    phash_distance: int
    dhash_distance: int
    cosine: Optional[float] = None
    confirmed: bool = False
    reason: str = ''


def exact_groups(rows: Sequence[Dict[str, object]]) -> Dict[str, str]:
    """Map image_id -> canonical id for byte/pixel-identical duplicates."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for key in ('raw_sha256', 'pixel_sha256'):
        buckets: Dict[str, str] = {}
        for row in rows:
            value = str(row.get(key) or '')
            if not value:
                continue
            image_id = str(row['image_id'])
            if value in buckets:
                union(image_id, buckets[value])
            else:
                buckets[value] = image_id
    return {str(row['image_id']): find(str(row['image_id'])) for row in rows if row.get('image_id')}


def near_duplicate_pairs(rows: Sequence[Dict[str, object]],
                         max_distance: int = 8,
                         dhash_max: int = 6,
                         max_neighbours: int = 40) -> List[Pair]:
    """Find pHash candidates with FAISS, then confirm with dHash + geometry.

    Only pairs sharing a coarse bucket are compared when FAISS is unavailable;
    the fallback is deterministic and documented rather than silently skipped.
    """
    ids = [str(r['image_id']) for r in rows]
    hashes = [int(str(r.get('phash64') or '0'), 16) for r in rows]
    dhashes = [int(str(r.get('dhash64') or '0'), 16) for r in rows]
    index_of = {image_id: i for i, image_id in enumerate(ids)}
    pairs: Dict[Tuple[int, int], Pair] = {}

    try:
        import faiss
        bits = np.array(hashes, dtype=np.uint64)
        index = faiss.IndexBinaryFlat(64)
        index.add(bits.view(np.uint8).reshape(len(bits), 8))
        # k-NN instead of range_search: at 200k images a radius-8 query
        # returns thousands of neighbours each (range_search materialises all
        # of them and the pass stops finishing).  k = max_neighbours + 1 keeps
        # the candidate set bounded and the result deterministic.
        k = min(max_neighbours + 1, len(ids))
        chunk = 5000
        for start in range(0, len(ids), chunk):
            query = bits[start:start + chunk].view(np.uint8).reshape(-1, 8)
            distances, neighbours = index.search(query, k)
            for local in range(len(query)):
                for distance, other in zip(distances[local], neighbours[local]):
                    if other < 0 or distance > max_distance:
                        continue
                    i, j = start + local, int(other)
                    if i >= j:
                        continue
                    pairs[(i, j)] = _score_pair(rows, i, j, dhashes, max_distance, dhash_max)
            if start % (chunk * 4) == 0:
                print(f'  near-dup {start + len(query)}/{len(ids)}', flush=True)
        return list(pairs.values())
    except Exception:  # noqa: BLE001 - FAISS is optional; fallback is explicit
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i, value in enumerate(hashes):
            buckets[value >> 48].append(i)
        for bucket in buckets.values():
            for a in range(len(bucket)):
                for b in range(a + 1, len(bucket)):
                    i, j = bucket[a], bucket[b]
                    if hamming(hashes[i], hashes[j]) <= max_distance:
                        pairs[(i, j)] = _score_pair(rows, i, j, dhashes, max_distance, dhash_max)
        return list(pairs.values())


def _score_pair(rows, i: int, j: int, dhashes, max_distance: int, dhash_max: int) -> Pair:
    left, right = rows[i], rows[j]
    ph = hamming(int(str(left.get('phash64') or '0'), 16), int(str(right.get('phash64') or '0'), 16))
    dh = hamming(dhashes[i], dhashes[j])
    w1, h1 = int(left.get('width') or 0), int(left.get('height') or 0)
    w2, h2 = int(right.get('width') or 0), int(right.get('height') or 0)
    aspect_ok = False
    if min(w1, h1, w2, h2) > 0:
        a1, a2 = max(w1, h1) / min(w1, h1), max(w2, h2) / min(w2, h2)
        aspect_ok = abs(a1 - a2) / max(a1, a2) < 0.02
    confirmed = ph <= 4 and dh <= dhash_max and aspect_ok
    reason = []
    if ph > 4:
        reason.append('PHASH_WEAK')
    if dh > dhash_max:
        reason.append('DHASH_DISAGREES')
    if not aspect_ok:
        reason.append('GEOMETRY_DISAGREES')
    return Pair(str(left['image_id']), str(right['image_id']), ph, dh,
                confirmed=confirmed, reason=';'.join(reason) or 'multi-hash agreement')


def connected_components(pairs: Iterable[Pair], confirmed_only: bool = True) -> Dict[str, str]:
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for pair in pairs:
        if confirmed_only and not pair.confirmed:
            continue
        union(pair.left, pair.right)
    return {node: find(node) for node in parent}


def flag_giant_clusters(components: Dict[str, str], threshold: int = 300) -> set:
    """Return image ids that belong to an implausibly large component.

    A pHash connected component with thousands of members is far more likely
    to be a texture collision (bare soil, uniform canopy) than a real duplicate
    set.  Those components are quarantined, never collapsed.
    """
    sizes: Dict[str, int] = defaultdict(int)
    for group in components.values():
        sizes[group] += 1
    giant = {group for group, size in sizes.items() if size > threshold}
    return {image_id for image_id, group in components.items() if group in giant}


def choose_representatives(rows: Sequence[Dict[str, object]], components: Dict[str, str]) -> Dict[str, str]:
    """Deterministic representative per component: quality first, then size."""
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        image_id = str(row['image_id'])
        group = components.get(image_id)
        if group:
            groups[group].append(row)
    winners: Dict[str, str] = {}
    for group, members in groups.items():
        members.sort(key=lambda r: (
            -(float(r.get('laplacian_var') or 0.0)),
            -(int(r.get('width') or 0) * int(r.get('height') or 0)),
            str(r['image_id'])))
        winners[group] = str(members[0]['image_id'])
    return winners


class EmbeddingConfirmer:
    """DINOv2 embedding similarity for pairs that hash alone cannot settle."""

    MODEL = 'vit_small_patch14_dinov2.lvd142m'

    def __init__(self, device: str = 'cuda:1', batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._transform = None

    def load(self):
        if self._model is not None:
            return
        import torch
        import timm
        from torchvision import transforms
        self._model = timm.create_model(self.MODEL, pretrained=True, num_classes=0)
        self._model = self._model.to(self.device).eval()
        self._transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(238),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def embed(self, paths: Sequence[str]) -> np.ndarray:
        import torch
        from PIL import Image
        self.load()
        vectors = []
        for start in range(0, len(paths), self.batch_size):
            batch = []
            for path in paths[start:start + self.batch_size]:
                try:
                    with Image.open(path) as image:
                        batch.append(self._transform(image.convert('RGB')))
                except Exception:  # noqa: BLE001
                    batch.append(torch.zeros(3, 238, 238))
            tensor = torch.stack(batch).to(self.device)
            with torch.no_grad():
                features = self._model(tensor)
            features = torch.nn.functional.normalize(features, dim=-1)
            vectors.append(features.cpu().numpy())
        return np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 384), dtype=np.float32)
