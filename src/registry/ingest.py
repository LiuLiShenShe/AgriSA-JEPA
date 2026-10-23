"""Incremental source ingestion: decode, hash, group, and screen duplicates.

This is the Gate-3 machinery used for every new source.  It never decides that
an image is *approved*; it produces the evidence that a later human/curator
decision needs:

* ``sha256``          -- byte identity (exact duplicates)
* ``pixel_sha256``    -- decoded RGB identity (re-encoded duplicates)
* ``phash64``/``dhash64`` -- perceptual screening (candidates, not verdicts)
* ``group_key``       -- class folder / session proxy used for isolation checks

Near-duplicate pairs are always emitted as *candidates*.  A stepped image and
its neighbour are not the same observation, and a perceptual hash cannot tell
the difference, so nothing is auto-deleted here.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

MANIFEST_FIELDS: List[str] = [
    'sample_id', 'source_id', 'source_name', 'rel_path', 'path', 'class_label',
    'group_key', 'width', 'height', 'mode', 'format', 'bytes', 'sha256',
    'pixel_sha256', 'phash64', 'dhash64', 'decode_ok', 'decode_error',
    'duplicate_status', 'duplicate_of', 'license', 'attribution',
    'source_url', 'source_archive_sha256',
]


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n x n) without scipy."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    mat = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    mat[0, :] *= np.sqrt(0.5)
    return mat


_DCT32 = _dct_matrix(32)


def phash64(image: Image.Image) -> str:
    """64-bit DCT perceptual hash (low-frequency 8x8 block, median threshold)."""
    gray = image.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    coeffs = _DCT32 @ pixels @ _DCT32.T
    low = coeffs[:8, :8].flatten()
    bits = (low > np.median(low)).astype(np.uint64)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def dhash64(image: Image.Image) -> str:
    """64-bit difference hash (9x8 gradient signs)."""
    gray = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert('RGB')
    return hashlib.sha256(np.asarray(rgb, dtype=np.uint8).tobytes()).hexdigest()


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _class_label(rel: Path, root: Path) -> str:
    parts = rel.parts
    return parts[-2] if len(parts) >= 2 else ''


def _group_key(rel: Path) -> str:
    """Filename source token after ``___`` if present, else the class folder."""
    stem = rel.stem
    if '___' in stem:
        tail = stem.split('___', 1)[1]
        token = tail.split()[0] if tail.split() else tail
        return f'{rel.parent.name}|{token}'
    return rel.parent.name


@dataclass
class IngestConfig:
    source_id: str
    source_name: str
    root: Path
    out_csv: Path
    license: str = ''
    attribution: str = ''
    source_url: str = ''
    source_archive_sha256: str = ''
    phash_threshold: int = 6
    dhash_threshold: int = 6
    existing_manifest: Optional[Path] = None
    max_examples: int = 50


@dataclass
class IngestResult:
    rows: List[Dict[str, object]] = field(default_factory=list)
    audit: Dict[str, object] = field(default_factory=dict)


def _bits(hex_hashes: Sequence[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def _pairs_within(hashes: np.ndarray, threshold: int,
                  chunk: int = 512) -> List[Tuple[int, int, int]]:
    """All index pairs (i<j) with Hamming distance <= threshold."""
    out: List[Tuple[int, int, int]] = []
    n = len(hashes)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = hashes[start:stop][:, None] ^ hashes[None, :]
        dist = np.bitwise_count(block)
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i, j = start + r, c
            if i < j:
                out.append((i, j, int(dist[r, c])))
    return out


def ingest(config: IngestConfig,
           reference_rows: Optional[Sequence[Dict[str, object]]] = None) -> IngestResult:
    """Walk ``config.root``, hash every image, and screen duplicates.

    ``reference_rows`` are existing corpus rows (``sha256``/``pixel_sha256``/
    ``phash64``) used for *cross-source* screening.  A match is reported as a
    candidate/duplicate; it is never silently dropped.
    """
    result = IngestResult()
    paths = list(iter_images(config.root))
    rows: List[Dict[str, object]] = []
    decode_failures: List[Dict[str, str]] = []
    for path in paths:
        rel = path.relative_to(config.root)
        row: Dict[str, object] = {
            'sample_id': f"{config.source_id}:{rel.as_posix()}",
            'source_id': config.source_id,
            'source_name': config.source_name,
            'rel_path': rel.as_posix(),
            'path': str(path),
            'class_label': _class_label(rel, config.root),
            'group_key': _group_key(rel),
            'width': None, 'height': None, 'mode': '', 'format': '',
            'bytes': path.stat().st_size, 'sha256': file_sha256(path),
            'pixel_sha256': '', 'phash64': '', 'dhash64': '',
            'decode_ok': False, 'decode_error': '',
            'duplicate_status': 'unique', 'duplicate_of': '',
            'license': config.license, 'attribution': config.attribution,
            'source_url': config.source_url,
            'source_archive_sha256': config.source_archive_sha256,
        }
        try:
            with Image.open(path) as image:
                image.load()
                row['width'], row['height'] = image.size
                row['mode'] = image.mode
                row['format'] = image.format or ''
                row['pixel_sha256'] = pixel_sha256(image)
                row['phash64'] = phash64(image)
                row['dhash64'] = dhash64(image)
                row['decode_ok'] = True
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            row['decode_error'] = f'{type(exc).__name__}: {exc}'
            decode_failures.append({'path': rel.as_posix(), 'error': row['decode_error']})
        rows.append(row)

    decoded = [r for r in rows if r['decode_ok']]

    # ---- exact duplicates (byte and decoded pixel) ----
    by_sha: Dict[str, List[int]] = {}
    by_pixel: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        by_sha.setdefault(str(row['sha256']), []).append(idx)
        if row['decode_ok']:
            by_pixel.setdefault(str(row['pixel_sha256']), []).append(idx)
    exact_pairs: List[Dict[str, object]] = []
    for kind, mapping in (('sha256', by_sha), ('pixel_sha256', by_pixel)):
        for hash_value, members in mapping.items():
            if len(members) > 1:
                keep = members[0]
                for other in members[1:]:
                    rows[other]['duplicate_status'] = f'exact_{kind}'
                    rows[other]['duplicate_of'] = str(rows[keep]['sample_id'])
                    exact_pairs.append({
                        'kind': kind, 'kept': rows[keep]['sample_id'],
                        'duplicate': rows[other]['sample_id']})

    # ---- perceptual candidates (pHash and dHash) ----
    ph = _bits([str(r['phash64']) for r in decoded])
    dh = _bits([str(r['dhash64']) for r in decoded])
    decoded_ids = [str(r['sample_id']) for r in decoded]

    near_pairs: List[Dict[str, object]] = []
    seen = set()
    for kind, hashes, threshold in (('phash', ph, config.phash_threshold),
                                    ('dhash', dh, config.dhash_threshold)):
        for i, j, dist in _pairs_within(hashes, threshold):
            key = (kind, i, j)
            if key in seen:
                continue
            seen.add(key)
            near_pairs.append({'kind': kind, 'distance': dist,
                               'a': decoded_ids[i], 'b': decoded_ids[j],
                               'review_status': 'pending_manual_review'})

    # ---- cross-source overlap ----
    cross_matches: List[Dict[str, object]] = []
    if reference_rows:
        ref_by_sha = {str(r.get('sha256', '')): r for r in reference_rows}
        ref_by_pixel = {str(r.get('pixel_sha256', '')): r
                        for r in reference_rows if r.get('pixel_sha256')}
        for row in rows:
            hit = ref_by_sha.get(str(row['sha256'])) or ref_by_pixel.get(
                str(row['pixel_sha256']))
            if hit is not None:
                row['duplicate_status'] = 'cross_source_duplicate'
                row['duplicate_of'] = f"existing:{hit.get('sample_id')}"
                cross_matches.append({
                    'sample_id': row['sample_id'],
                    'existing_sample_id': hit.get('sample_id'),
                    'existing_source': hit.get('source'),
                    'kind': 'sha256/pixel_sha256'})
        ref_ph = [str(r.get('phash64')) for r in reference_rows
                  if r.get('phash64')]
        if ref_ph:
            ref_hashes = _bits(ref_ph)
            ref_ids = [r.get('sample_id') for r in reference_rows if r.get('phash64')]
            # compare in query chunks to bound memory
            for start in range(0, len(ph), 256):
                block = ph[start:start + 256][:, None] ^ ref_hashes[None, :]
                dist = np.bitwise_count(block)
                hits = np.argwhere(dist <= config.phash_threshold)
                for r, c in hits[:200]:
                    cross_matches.append({
                        'sample_id': decoded_ids[start + int(r)],
                        'existing_sample_id': ref_ids[int(c)],
                        'kind': 'phash_candidate',
                        'distance': int(dist[r, c]),
                        'review_status': 'pending_manual_review'})

    duplicate_free = sum(1 for r in decoded
                         if str(r['duplicate_status']) == 'unique')
    audit = {
        'source_id': config.source_id,
        'source_name': config.source_name,
        'root': str(config.root),
        'license': config.license,
        'source_url': config.source_url,
        'files_found': len(rows),
        'files_decoded': len(decoded),
        'decode_failures': len(decode_failures),
        'decode_failure_examples': decode_failures[:config.max_examples],
        'exact_duplicate_pairs': len(exact_pairs),
        'exact_duplicate_examples': exact_pairs[:config.max_examples],
        'perceptual_candidate_pairs': len(near_pairs),
        'perceptual_candidate_examples': near_pairs[:config.max_examples],
        'cross_source_exact_matches': len([m for m in cross_matches
                                           if m['kind'] != 'phash_candidate']),
        'cross_source_candidates': len([m for m in cross_matches
                                        if m['kind'] == 'phash_candidate']),
        'cross_source_examples': cross_matches[:config.max_examples],
        'images_duplicate_free': duplicate_free,
        'unique_class_labels': len({r['class_label'] for r in rows}),
        'unique_group_keys': len({r['group_key'] for r in rows}),
        'image_suffixes': sorted({Path(str(r['rel_path'])).suffix.lower()
                                  for r in rows}),
        'notes': ('Perceptual pairs are screening candidates, not proof of '
                  'duplication; hidden and stepped images differ by design.'),
    }
    result.rows = rows
    result.audit = audit
    return result


def write_manifest_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in MANIFEST_FIELDS})
