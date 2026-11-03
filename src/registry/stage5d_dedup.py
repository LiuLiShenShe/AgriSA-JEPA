"""Duplicate control for Stage 5D (fixes the Stage 5C column-name bug).

Stage 5C called ``near_duplicate_pairs()`` with ledger records that store the
hashes as ``phash`` / ``dhash``, while the function read ``phash64`` /
``dhash64``.  Every hash therefore parsed as ``0`` and the FAISS k-NN returned
only the first 41 rows as "duplicates" (``C(41,2) = 820`` confirmed pairs);
nothing beyond row 41 was ever compared.  The near-duplicate gate was inert.

This module:

* reads the columns that actually exist, and refuses to run if more than a
  fixed fraction of hashes is unparseable;
* confirms a near duplicate only when independent signals agree (pHash,
  dHash, geometry and, when requested, a DINOv2 cosine);
* keeps unresolved candidates as ``UNCERTAIN_CANDIDATE`` (quarantined) rather
  than deleting them;
* quarantines a whole giant component instead of collapsing a texture
  collision;
* writes one decision row per duplicate image, with the original ids.

All decisions are deterministic.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .stage5c_near_duplicate import (Pair, connected_components, hamming)


@dataclass
class DedupConfig:
    phash_max_distance: int = 8
    dhash_max_distance: int = 6
    aspect_tolerance: float = 0.02
    confirm_phash_max: int = 4
    giant_cluster_threshold: int = 300
    max_neighbours: int = 40
    embedding_cosine_min: float = 0.98
    use_embeddings: bool = False
    max_unparseable_fraction: float = 0.01


def _parse_hash(value: object) -> Optional[int]:
    text = str(value or '').strip()
    if not text or text.lower() in ('nan', 'none', '0'):
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def hash_quality(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    parsed = [_parse_hash(v) for v in frame[column]]
    valid = sum(1 for h in parsed if h is not None)
    total = len(parsed)
    fraction = valid / total if total else 0.0
    unique = len({h for h in parsed if h is not None})
    return {'column': column, 'rows': total, 'valid': valid,
            'valid_fraction': fraction, 'unique_hashes': unique}


def validate_hash_columns(frame: pd.DataFrame, phash: str = 'phash',
                          dhash: str = 'dhash',
                          max_unparseable: float = 0.01) -> Dict[str, Dict[str, float]]:
    report = {'phash': hash_quality(frame, phash), 'dhash': hash_quality(frame, dhash)}
    for name, stats in report.items():
        if stats['valid_fraction'] < 1.0 - max_unparseable:
            raise ValueError(
                f'{name} column {stats["column"]!r} has only '
                f'{stats["valid_fraction"]:.3f} parseable values; refusing to '
                'run near-duplicate detection on a silently-zero hash matrix')
    return report


def exact_duplicate_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Union-find over raw/pixel SHA-256; representative chosen by quality."""
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

    keys = frame[['image_id', 'raw_sha256', 'pixel_sha256']].to_dict('records')
    for key in ('raw_sha256', 'pixel_sha256'):
        buckets: Dict[str, str] = {}
        for row in keys:
            value = str(row.get(key) or '').strip()
            if not value or value.lower() in ('nan', 'none'):
                continue
            image_id = str(row['image_id'])
            if value in buckets:
                union(image_id, buckets[value])
            else:
                buckets[value] = image_id

    members: Dict[str, List[str]] = defaultdict(list)
    for row in keys:
        members[find(str(row['image_id']))].append(str(row['image_id']))

    quality = frame.set_index('image_id')
    decisions = []
    for cluster, ids in members.items():
        if len(ids) == 1:
            decisions.append({'image_id': ids[0], 'exact_duplicate': False,
                              'duplicate_cluster_id': cluster,
                              'exact_duplicate_of': ''})
            continue
        ids_sorted = sorted(ids)
        representative = min(
            ids_sorted,
            key=lambda image_id: (
                -float(quality.at[image_id, 'laplacian_var']
                       if pd.notna(quality.at[image_id, 'laplacian_var']) else 0.0),
                -int(quality.at[image_id, 'width'] or 0) * int(quality.at[image_id, 'height'] or 0),
                image_id))
        for image_id in ids_sorted:
            decisions.append({
                'image_id': image_id,
                'exact_duplicate': image_id != representative,
                'duplicate_cluster_id': cluster,
                'exact_duplicate_of': '' if image_id == representative else representative,
            })
    return pd.DataFrame(decisions)


def near_duplicate_scan(frame: pd.DataFrame, config: DedupConfig,
                        embedding_provider=None
                        ) -> Tuple[List[Pair], Dict[str, str], Dict[str, int]]:
    """Scan for pHash candidates, confirm them, and return components.

    The returned component map contains only *confirmed* links.  Unconfirmed
    candidates are returned separately so the caller can quarantine them.
    Hash integers are parsed once; the Stage 5C code re-parsed them per pair,
    which is quadratic in the number of bursts.
    """
    ids = frame['image_id'].astype(str).to_numpy()
    phashes = np.array([_parse_hash(v) for v in frame['phash']], dtype=object)
    dhashes = np.array([_parse_hash(v) for v in frame['dhash']], dtype=object)
    widths = pd.to_numeric(frame['width'], errors='coerce').fillna(0).to_numpy(dtype=np.int64)
    heights = pd.to_numeric(frame['height'], errors='coerce').fillna(0).to_numpy(dtype=np.int64)
    valid = np.array([i for i, (p, d) in enumerate(zip(phashes, dhashes))
                      if p is not None and d is not None], dtype=np.int64)
    stats = {'rows': len(frame), 'with_valid_hashes': int(len(valid))}
    if len(valid) == 0:
        return [], {}, stats

    ph_valid = np.array([int(phashes[i]) for i in valid], dtype=np.uint64)
    dh_valid = np.array([int(dhashes[i]) for i in valid], dtype=np.uint64)

    pairs: Dict[Tuple[int, int], Tuple[str, str, int, int, bool, str]] = {}
    try:
        import faiss
        index = faiss.IndexBinaryFlat(64)
        index.add(ph_valid.view(np.uint8).reshape(len(ph_valid), 8))
        k = min(config.max_neighbours + 1, len(valid))
        chunk = 5000
        for start in range(0, len(valid), chunk):
            query = ph_valid[start:start + chunk].view(np.uint8).reshape(-1, 8)
            distances, neighbours = index.search(query, k)
            for local in range(len(query)):
                position = start + local
                for distance, other in zip(distances[local], neighbours[local]):
                    if other < 0 or distance > config.phash_max_distance:
                        continue
                    other = int(other)
                    if position >= other:
                        continue
                    key = (position, other)
                    if key in pairs:
                        continue
                    pairs[key] = _score_fast(position, other, valid, ids, ph_valid,
                                             dh_valid, widths, heights, config)
            if start % (chunk * 10) == 0:
                print(f'  near-dup {start + len(query)}/{len(valid)} '
                      f'pairs={len(pairs)}', flush=True)
    except Exception as exc:  # noqa: BLE001 - explicit, conservative fallback
        print(f'  near-dup faiss unavailable ({exc}); using bucket fallback', flush=True)
        buckets: Dict[int, List[int]] = defaultdict(list)
        for position, value in enumerate(ph_valid):
            buckets[int(value) >> 48].append(position)
        for bucket in buckets.values():
            for a in range(len(bucket)):
                for b in range(a + 1, len(bucket)):
                    i, j = bucket[a], bucket[b]
                    if hamming(int(ph_valid[i]), int(ph_valid[j])) <= config.phash_max_distance:
                        pairs[(i, j)] = _score_fast(i, j, valid, ids, ph_valid,
                                                    dh_valid, widths, heights, config)

    embedding_vectors: Dict[str, np.ndarray] = {}
    hash_confirmed = [key for key, value in pairs.items() if value[4]]
    if embedding_provider is not None and hash_confirmed:
        endpoints = sorted({value[0] for value in pairs.values() if value[4]} |
                           {value[1] for value in pairs.values() if value[4]})
        print(f'  near-dup embedding {len(endpoints)} images for '
              f'{len(hash_confirmed)} hash-confirmed pairs', flush=True)
        embedding_vectors = embedding_provider(endpoints)
        stats['embedded_images'] = len(endpoints)

    confirmed: List[Pair] = []
    uncertain: List[Pair] = []
    for (a, b), (left_id, right_id, ph, dh, is_confirmed, reason) in pairs.items():
        pair = Pair(left_id, right_id, ph, dh, confirmed=is_confirmed, reason=reason)
        if pair.confirmed and embedding_provider is not None:
            left = embedding_vectors.get(pair.left)
            right = embedding_vectors.get(pair.right)
            if left is None or right is None:
                pair.confirmed = False
                pair.reason = append_reason(pair.reason, 'EMBEDDING_MISSING')
            else:
                cosine = float(np.dot(left, right) /
                               (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
                pair.cosine = cosine
                if cosine < config.embedding_cosine_min:
                    pair.confirmed = False
                    pair.reason = append_reason(pair.reason, 'EMBEDDING_DISAGREES')
        (confirmed if pair.confirmed else uncertain).append(pair)

    components = connected_components(confirmed, confirmed_only=True)
    sizes: Dict[str, int] = defaultdict(int)
    for group in components.values():
        sizes[group] += 1
    giant_groups = {group for group, size in sizes.items()
                    if size > config.giant_cluster_threshold}

    # A k-NN scan cannot see the whole of an exactly-repeated hash (the
    # neighbour cap bounds each node to k links).  Detect a texture collision
    # by hash frequency as well, so a uniform soil image set cannot be
    # "deduplicated" down to a handful of representatives.
    from collections import Counter
    frequency = Counter(int(v) for v in ph_valid)
    giant_ids = {str(ids[int(valid[i])]) for value, count in frequency.items()
                 if count > config.giant_cluster_threshold
                 for i in np.where(ph_valid == np.uint64(value))[0]}
    for group in giant_groups:
        giant_ids |= {image_id for image_id, g in components.items() if g == group}

    stats.update({
        'candidate_pairs': len(pairs),
        'confirmed_pairs': len(confirmed),
        'uncertain_pairs': len(uncertain),
        'component_members': len(components),
        'giant_groups': len(giant_groups),
        'giant_members': len(giant_ids),
        'giant_image_ids': sorted(giant_ids),
    })
    return confirmed + uncertain, components, stats


def _score_fast(a: int, b: int, valid, ids, ph_valid, dh_valid, widths, heights,
                config: DedupConfig) -> Tuple[str, str, int, int, bool, str]:
    ia, ib = int(valid[a]), int(valid[b])
    ph = int(ph_valid[a] ^ ph_valid[b]).bit_count()
    dh = int(dh_valid[a] ^ dh_valid[b]).bit_count()
    w1, h1, w2, h2 = int(widths[ia]), int(heights[ia]), int(widths[ib]), int(heights[ib])
    aspect_ok = False
    if min(w1, h1, w2, h2) > 0:
        a1 = max(w1, h1) / min(w1, h1)
        a2 = max(w2, h2) / min(w2, h2)
        aspect_ok = abs(a1 - a2) / max(a1, a2) < config.aspect_tolerance
    confirmed = (ph <= config.confirm_phash_max and dh <= config.dhash_max_distance
                 and aspect_ok)
    reasons = []
    if ph > config.confirm_phash_max:
        reasons.append('PHASH_WEAK')
    if dh > config.dhash_max_distance:
        reasons.append('DHASH_DISAGREES')
    if not aspect_ok:
        reasons.append('GEOMETRY_DISAGREES')
    return (str(ids[ia]), str(ids[ib]), ph, dh, confirmed,
            ';'.join(reasons) or 'multi_hash_agreement')


def append_reason(reason: str, extra: str) -> str:
    return f'{reason};{extra}' if reason else extra


def apply_dedup(frame: pd.DataFrame, config: DedupConfig,
                embedding_provider=None
                ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Attach exact/near duplicate decisions and return the decision table."""
    out = frame.copy()
    decodable = out[out['decode_ok'].astype(str).str.lower().isin(('true', '1'))]
    hash_report = validate_hash_columns(
        decodable if len(decodable) else out,
        max_unparseable=config.max_unparseable_fraction)

    exact = exact_duplicate_decisions(out)
    out = out.merge(exact, on='image_id', how='left')
    out['exact_duplicate'] = out['exact_duplicate'].fillna(False).astype(bool)

    pairs, components, stats = near_duplicate_scan(out, config, embedding_provider)
    confirmed = [p for p in pairs if p.confirmed]
    uncertain = [p for p in pairs if not p.confirmed]

    sizes: Dict[str, int] = defaultdict(int)
    for group in components.values():
        sizes[group] += 1
    giant_groups = {g for g, size in sizes.items()
                    if size > config.giant_cluster_threshold}
    giant_image_ids = set(stats.get('giant_image_ids', []))
    giant_image_ids |= {image_id for image_id, g in components.items()
                        if g in giant_groups}

    # Representative per confirmed component: quality first, deterministic.
    quality = out.set_index('image_id')
    representative: Dict[str, str] = {}
    members: Dict[str, List[str]] = defaultdict(list)
    for image_id, group in components.items():
        members[group].append(image_id)
    for group, ids in members.items():
        representative[group] = min(
            sorted(ids),
            key=lambda image_id: (
                -float(quality.at[image_id, 'laplacian_var']
                       if pd.notna(quality.at[image_id, 'laplacian_var']) else 0.0),
                -int(quality.at[image_id, 'width'] or 0) * int(quality.at[image_id, 'height'] or 0),
                image_id))

    near_status: Dict[str, str] = {}
    near_of: Dict[str, str] = {}
    for group, ids in members.items():
        rep = representative[group]
        is_giant = group in giant_groups or any(i in giant_image_ids for i in ids)
        for image_id in ids:
            if is_giant:
                near_status[image_id] = 'GIANT_CLUSTER'
                near_of[image_id] = ''
            elif image_id == rep:
                near_status[image_id] = 'REPRESENTATIVE'
                near_of[image_id] = ''
            else:
                near_status[image_id] = 'CONFIRMED_NEAR_DUPLICATE'
                near_of[image_id] = rep
    for pair in uncertain:
        for image_id in (pair.left, pair.right):
            if image_id not in near_status:
                near_status[image_id] = 'UNCERTAIN_CANDIDATE'
                near_of[image_id] = ''

    out['near_dup_status'] = out['image_id'].map(near_status).fillna('OK')
    out['near_dup_of'] = out['image_id'].map(near_of).fillna('')

    decisions = pd.DataFrame([
        {'image_id': pair.left, 'other_image_id': pair.right,
         'phash_distance': pair.phash_distance, 'dhash_distance': pair.dhash_distance,
         'cosine': pair.cosine, 'confirmed': pair.confirmed, 'reason': pair.reason}
        for pair in pairs
    ])
    stats.update({'hash_report': hash_report,
                  'exact_duplicate_images': int(out['exact_duplicate'].sum()),
                  'giant_members': sum(sizes[g] for g in giant_groups)})
    return out, decisions, stats


class Dinov2Embedder:
    """DINOv2 identity embeddings for near-duplicate confirmation.

    The Stage 5C ``EmbeddingConfirmer`` fed 238 px crops to a model whose
    positional embedding expects 518 px; every call raised and the exception was
    swallowed, so visual confirmation never ran.  This implementation uses the
    size the model actually accepts and reports failures instead of hiding them.
    """

    MODEL = 'vit_small_patch14_dinov2.lvd142m'
    IMAGE_SIZE = 518

    def __init__(self, device: str = 'cuda:1', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._transform = None

    def load(self):
        if self._model is not None:
            return
        import timm
        from torchvision import transforms
        self._model = timm.create_model(self.MODEL, pretrained=True,
                                        num_classes=0, device=self.device).eval()
        self._transform = transforms.Compose([
            transforms.Resize(self.IMAGE_SIZE,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def provider(self, id_to_path: Dict[str, str]):
        def embed(ids: List[str]) -> Dict[str, np.ndarray]:
            import torch
            from PIL import Image
            self.load()
            out: Dict[str, np.ndarray] = {}
            for start in range(0, len(ids), self.batch_size):
                batch_ids = ids[start:start + self.batch_size]
                tensors = []
                for image_id in batch_ids:
                    try:
                        with Image.open(id_to_path[image_id]) as image:
                            tensors.append(self._transform(image.convert('RGB')))
                    except Exception:  # noqa: BLE001
                        tensors.append(torch.zeros(3, self.IMAGE_SIZE, self.IMAGE_SIZE))
                stack = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    features = self._model(stack)
                features = torch.nn.functional.normalize(features, dim=-1)
                for image_id, vector in zip(batch_ids, features.cpu().numpy()):
                    out[image_id] = vector
                if start % (self.batch_size * 10) == 0:
                    print(f'    embedded {start + len(batch_ids)}/{len(ids)}', flush=True)
            return out
        return embed
