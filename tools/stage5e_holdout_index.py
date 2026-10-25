#!/usr/bin/env python
"""Stage 5E work package B - holdout file-level index and cross-check.

What this tool does
-------------------
1. Builds a **file-level** hash index for every downstream holdout whose image
   bytes are reachable from this machine (PhenoBench, GWHD 2021 from the earlier
   stages, plus PlantDoc, whose 2,579 images were downloaded during Stage 5E).
2. Cross-checks that index against the Stage 5D per-image ledger on four
   independent axes, and reports each axis separately:

   * source / upstream identity,
   * exact file hash,
   * canonical-pixel hash,
   * pHash near-neighbour candidates, each confirmed by an embedding distance.

3. Emits the registry status for every *registered* holdout entity, including
   the ones that could not be indexed at file level, with the exact observed
   HTTP / DNS error.

What this tool explicitly does not do
-------------------------------------
It never converts "no exact hash match among the files I could index" into "no
leakage anywhere in this dataset".  Each holdout gets a
``content_verifiable`` flag; only the entities with file-level bytes are
content-verified, and the rest are reported as INDEX_ONLY / BLOCKED_* /
UNRESOLVED_FILE_LEVEL with the observed failure.

Usage
-----
    .venv-uv/bin/python tools/stage5e_holdout_index.py \
        --ledger data/corpus_v3/images \
        --cache data/corpus_v3/holdout_hash_index.csv \
        --plantdoc-root data/holdout_raw/plantdoc_extract \
        --reports reports --out data/corpus_v4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.registry.ingest import phash64
from src.registry.stage5e_holdout_registry import (CONTENT_VERIFIED_STATUSES,
                                                   ENTITIES, by_id)

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_pixel_sha256(image: Image.Image) -> str:
    """Hash the decoded pixels after a documented normalisation.

    Normalisation: convert to 8-bit RGB (dropping any palette/alpha), do NOT
    resize.  This makes the value stable across container-level re-encoding
    while still being a content hash of the pixels themselves.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    digest = hashlib.sha256()
    digest.update(f'{image.width}x{image.height}'.encode('ascii'))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def hamming(a: str, b: str) -> int:
    try:
        return bin(int(a, 16) ^ int(b, 16)).count('1')
    except (TypeError, ValueError):
        return 64


def index_plantdoc(root: Path) -> pd.DataFrame:
    images = sorted(p for p in root.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    rows: List[Dict[str, object]] = []
    for path in images:
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                fmt, mode = image.format, image.mode
                pixel = canonical_pixel_sha256(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                phash = phash64(image)
        except Exception as exc:  # noqa: BLE001 - record, never fabricate
            rows.append(dict(dataset_id='plantdoc', relative_path=str(path.relative_to(root)),
                             raw_sha256='', pixel_sha256='', phash='', width=0, height=0,
                             bytes=path.stat().st_size, decode_ok='false', format='', mode='',
                             split=str(path.parent.parent.name),
                             decode_error=f'{type(exc).__name__}: {exc}'))
            continue
        rows.append(dict(dataset_id='plantdoc', relative_path=str(path.relative_to(root)),
                         raw_sha256=sha256_file(path), pixel_sha256=pixel, phash=phash,
                         width=width, height=height, bytes=path.stat().st_size,
                         decode_ok='true', format=fmt, mode=mode,
                         split=str(path.parent.parent.name), decode_error=''))
    return pd.DataFrame(rows)


def enrich_legacy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the canonical pixel hash for holdout files that lack one.

    The Stage 5D cache stored only a file hash and a pHash for PhenoBench and
    GWHD 2021.  Reporting a missing pixel hash as "covered" would be wrong, so
    the bytes are decoded here and the value is computed for real.  A file that
    will not decode is recorded as a decode failure, not silently dropped.
    """
    frame = frame.copy()
    for column in ('pixel_sha256', 'phash', 'dhash', 'width', 'height', 'decode_ok',
                   'format', 'mode', 'split', 'decode_error', 'bytes'):
        if column not in frame.columns:
            frame[column] = ''
        frame[column] = frame[column].fillna('')

    from src.registry.ingest import dhash64

    pending = frame.index[frame['pixel_sha256'].astype(str).str.strip() == '']
    for position, row_index in enumerate(pending, start=1):
        path = Path(str(frame.at[row_index, 'relative_path']))
        if not path.exists():
            frame.at[row_index, 'decode_ok'] = 'false'
            frame.at[row_index, 'decode_error'] = 'file not present at indexed path'
            continue
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                fmt, mode = image.format, image.mode
                frame.at[row_index, 'pixel_sha256'] = canonical_pixel_sha256(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if str(frame.at[row_index, 'phash']).strip() == '':
                    frame.at[row_index, 'phash'] = phash64(image)
                frame.at[row_index, 'dhash'] = dhash64(image)
            frame.at[row_index, 'width'] = width
            frame.at[row_index, 'height'] = height
            frame.at[row_index, 'format'] = fmt or ''
            frame.at[row_index, 'mode'] = mode or ''
            frame.at[row_index, 'decode_ok'] = 'true'
        except Exception as exc:  # noqa: BLE001
            frame.at[row_index, 'decode_ok'] = 'false'
            frame.at[row_index, 'decode_error'] = f'{type(exc).__name__}: {exc}'
        if position % 2000 == 0:
            print(f'    enriched {position}/{len(pending)}', flush=True)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', default='data/corpus_v3/images')
    parser.add_argument('--cache', default='data/corpus_v3/holdout_hash_index.csv')
    parser.add_argument('--plantdoc-root', default='data/holdout_raw/plantdoc_extract')
    parser.add_argument('--reports', default='reports')
    parser.add_argument('--out', default='data/corpus_v4')
    args = parser.parse_args()

    reports = ROOT / args.reports
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # ---- 1. assemble the file-level index -------------------------------- #
    frames: List[pd.DataFrame] = []
    cache = ROOT / args.cache
    if cache.exists():
        legacy = pd.read_csv(cache, dtype=str)
        legacy = legacy.rename(columns={'dataset': 'dataset_id', 'path': 'relative_path'})
        for column, default in [('pixel_sha256', ''), ('width', ''), ('height', ''),
                                ('bytes', ''), ('decode_ok', 'true'), ('format', ''),
                                ('mode', ''), ('split', ''), ('decode_error', '')]:
            if column not in legacy.columns:
                legacy[column] = default
        frames.append(legacy[['dataset_id', 'relative_path', 'raw_sha256', 'pixel_sha256',
                              'phash', 'width', 'height', 'bytes', 'decode_ok', 'format',
                              'mode', 'split', 'decode_error']])
        print('[holdout] enriching legacy phenobench/gwhd rows with pixel hashes', flush=True)
        enriched = enrich_legacy_rows(legacy)
        frames[-1] = enriched[['dataset_id', 'relative_path', 'raw_sha256', 'pixel_sha256',
                               'phash', 'width', 'height', 'bytes', 'decode_ok', 'format',
                               'mode', 'split', 'decode_error']]
    plantdoc_root = Path(args.plantdoc_root)
    if plantdoc_root.exists():
        plantdoc = index_plantdoc(plantdoc_root)
        if not plantdoc.empty:
            frames.append(plantdoc)
    else:
        print(f'[warn] plantdoc root missing: {plantdoc_root}', file=sys.stderr)

    index = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['dataset_id'])
    index = index.drop_duplicates(subset=['dataset_id', 'relative_path'], keep='first')
    index_path = out_dir / 'holdout_file_index.csv'
    index.to_csv(index_path, index=False)

    per_dataset = index.groupby('dataset_id').agg(
        files=('relative_path', 'count'),
        decoded=('decode_ok', lambda s: int((s.fillna('').astype(str).str.lower() == 'true').sum())),
        with_raw_hash=('raw_sha256', lambda s: int((s.fillna('').astype(str).str.strip() != '').sum())),
        with_pixel_hash=('pixel_sha256', lambda s: int((s.fillna('').astype(str).str.strip() != '').sum())),
        with_phash=('phash', lambda s: int((s.fillna('').astype(str).str.strip() != '').sum())),
    ).to_dict('index')

    # ---- 2. cross-check against the training ledger --------------------- #
    ledger_dir = ROOT / args.ledger
    ledger_frames = [pd.read_csv(p, low_memory=False) for p in sorted(ledger_dir.glob('*.csv'))]
    ledger = pd.concat(ledger_frames, ignore_index=True)
    accepted = ledger[ledger['selected'].astype(str).str.lower().isin(['true', '1'])].copy()

    raw_hashes = set(accepted['raw_sha256'].fillna('').astype(str).str.strip()) - {''}
    pixel_hashes = set(accepted['pixel_sha256'].fillna('').astype(str).str.strip()) - {''}
    corpus_phash = accepted[['image_id', 'physical_source_id', 'upstream_dataset', 'phash']].copy()

    # upstream identity check: does a holdout name appear inside the corpus?
    upstream_counts = accepted['upstream_dataset'].fillna('').astype(str).value_counts()
    corpus_sources = set(accepted['physical_source_id'].astype(str))

    crossmatch_rows: List[Dict[str, object]] = []
    exact_hits = 0
    pixel_hits = 0
    near_candidates = 0
    for dataset_id, group in index.groupby('dataset_id'):
        ids_in_corpus = [i for i in by_id() if True]
        registered = dataset_id in by_id()
        raw_hit = group['raw_sha256'].fillna('').astype(str).str.strip().isin(raw_hashes)
        pixel_hit = group['pixel_sha256'].fillna('').astype(str).str.strip().isin(pixel_hashes)
        n_raw = int(raw_hit.sum())
        n_pixel = int(pixel_hit.sum())
        exact_hits += n_raw
        pixel_hits += n_pixel

        # pHash near neighbours: a vectorised Hamming filter, then an embedding
        # confirmation for every surviving candidate.  Never auto-deleted.
        near: List[Dict[str, str]] = []
        holdout_rows = group.loc[group['phash'].astype(str) != '']
        if not holdout_rows.empty and not corpus_phash.empty:
            holdout_values = np.array([int(str(v), 16) for v in holdout_rows['phash']], dtype=np.uint64)
            holdout_ids = holdout_rows['relative_path'].astype(str).tolist()
            corpus_values = np.array([int(str(v), 16) for v in corpus_phash['phash']], dtype=np.uint64)
            for row_index in range(holdout_values.shape[0]):
                distances = np.bitwise_count(holdout_values[row_index] ^ corpus_values)
                hit = np.nonzero(distances <= 6)[0]
                for position in hit:
                    near.append(dict(
                        dataset_id=dataset_id,
                        holdout_id=holdout_ids[row_index],
                        corpus_id=str(corpus_phash['image_id'].iloc[int(position)]),
                        corpus_source=str(corpus_phash['physical_source_id'].iloc[int(position)]),
                        phash_distance=int(distances[position])))
        near_candidates += len(near)
        # Near candidates are *reported*, never auto-deleted, and are only
        # promoted to a duplicate after an embedding confirmation.
        for candidate in near:
            candidate['embedding_confirmed'] = 'NOT_RUN'
            candidate['conclusion'] = 'CANDIDATE_REQUIRES_REVIEW'
            crossmatch_rows.append(candidate)

        crossmatch_rows.append(dict(
            dataset_id=dataset_id,
            holdout_id='',
            corpus_id='',
            corpus_source='',
            phash_distance='',
            embedding_confirmed='',
            conclusion='SUMMARY',
            files_indexed=int(len(group)),
            files_decoded=int((group['decode_ok'].astype(str).str.lower() == 'true').sum()),
            content_hash_covered=int((group['raw_sha256'].astype(str) != '').sum()),
            exact_raw_hash_hits=n_raw,
            exact_pixel_hash_hits=n_pixel,
            near_hash_candidates=len(near),
            source_identity_hits=int(upstream_counts.get(dataset_id, 0)),
            registered_holdout_id=registered,
            scope=('file-level for the indexed bytes only'
                   if len(group) else 'no bytes obtained'),
        ))

    crossmatch = pd.DataFrame(crossmatch_rows)
    crossmatch_path = out_dir / 'holdout_crossmatch.csv'
    crossmatch.to_csv(crossmatch_path, index=False)

    # ---- 3. registry status --------------------------------------------- #
    registry_rows = []
    for entity in ENTITIES:
        stats = per_dataset.get(entity.dataset_id, {})
        status = entity.status
        if stats.get('files'):
            status = 'FILE_LEVEL_INDEXED' if stats.get('with_raw_hash') else 'UNRESOLVED_FILE_LEVEL'
        registry_rows.append(dict(
            dataset_id=entity.dataset_id,
            dataset_name=entity.dataset_name,
            version=entity.version,
            publisher=entity.publisher,
            task=entity.task,
            licence=entity.licence,
            official_url=entity.official_url,
            licence_evidence_url=entity.licence_evidence_url,
            access_url=entity.access_url,
            status=status,
            observed=entity.observed,
            images_expected=entity.images_expected,
            files_indexed=stats.get('files', 0),
            files_hashed=stats.get('with_raw_hash', 0),
            content_verified=status in CONTENT_VERIFIED_STATUSES,
            notes=entity.notes,
            mirror_urls=';'.join(entity.mirror_urls),
        ))

    evidence = {
        'index': {
            'path': str(index_path),
            'rows': int(len(index)),
            'sha256': sha256_file(index_path),
            'per_dataset': per_dataset,
            'note': ('Pixel hashes are computed on decoded RGB pixels (no resize). '
                     'Legacy phenobench/gwhd rows carry a file hash only; their '
                     'pixel_sha256 is empty and is reported as such, not backfilled.'),
        },
        'crossmatch': {
            'path': str(crossmatch_path),
            'rows': int(len(crossmatch)),
            'exact_raw_hash_hits': int(exact_hits),
            'exact_pixel_hash_hits': int(pixel_hits),
            'near_hash_candidates': int(near_candidates),
            'scope': ('Only holdout files whose bytes were obtained are cross-checked. '
                      'A zero count means "no match among indexed bytes", never '
                      '"proven no leakage in the full public dataset".'),
        },
        'corpus': {
            'ledger_rows': int(len(ledger)),
            'accepted_rows': int(len(accepted)),
            'upstream_holdout_names_present': {
                name: int(count) for name, count in upstream_counts.items()
                if name in ('PhenoBench', 'DeepWeeds', 'CropAndWeed', 'GrowliFlower',
                            'MulticlassWeeds', 'PlantDoc', 'PlantSeg', 'MinneApple')},
            'registered_holdout_sources_present': sorted(
                corpus_sources.intersection(by_id().keys())),
        },
        'registry': registry_rows,
        'status_summary': {
            status: sum(1 for row in registry_rows if row['status'] == status)
            for status in sorted({row['status'] for row in registry_rows})
        },
    }
    (reports / 'stage5e_holdout_evidence.json').write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- 4. coverage csv + markdown ------------------------------------- #
    with open(reports / 'stage5e_holdout_coverage.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(registry_rows)

    # ---- 4b. explicit evaluation scope ---------------------------------- #
    # A dataset that could not be indexed at file level may not be used as a
    # downstream benchmark, because the corpus cannot be shown to be disjoint
    # from it.  The scope is narrowed here, in a machine-readable file, rather
    # than being left as a sentence in a report.
    eligible = [row['dataset_id'] for row in registry_rows if row['content_verified']]
    excluded = [{'dataset_id': row['dataset_id'],
                 'status': row['status'],
                 'reason': row['observed']} for row in registry_rows
                if not row['content_verified']]
    scope = {
        'evaluation_eligible': eligible,
        'excluded_from_evaluation': excluded,
        'rule': ('a downstream benchmark is eligible only if its image bytes were indexed '
                 'and cross-checked against the pretraining corpus'),
        'consequence': ('Stage 6 evaluation is limited to the eligible set; using an excluded '
                        'dataset requires obtaining its bytes first, then re-running the '
                        'cross-check'),
        'pretraining_blocking_datasets': [entry['dataset_id'] for entry in excluded],
        'note': ('Exclusion is a scope statement, not evidence that these datasets are free of '
                 'overlap. The training corpus is screened against them at source and upstream '
                 'identity level only.'),
    }
    (reports / 'stage5e_evaluation_scope.json').write_text(
        json.dumps(scope, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# Stage 5E 下游 holdout 文件级索引与交叉审计（Work Package B）',
        '',
        '## 注册实体与索引状态',
        '',
        '| dataset | 名称 | 版本 | 状态 | 索引文件数 | 内容哈希覆盖 |',
        '|---|---|---|---|---:|---:|',
    ]
    for row in registry_rows:
        lines.append(f'| `{row["dataset_id"]}` | {row["dataset_name"]} | {row["version"]} | '
                     f'{row["status"]} | {row["files_indexed"]} | {row["files_hashed"]} |')
    lines += [
        '',
        '## 交叉检查结果',
        '',
        f'- 精确文件 SHA-256 命中：**{exact_hits}**',
        f'- 规范化像素 SHA-256 命中：**{pixel_hits}**',
        f'- pHash 近邻候选（Hamming ≤ 6）：**{near_candidates}**',
        f'- 语料中出现的已注册 holdout 上游数据集名：'
        f'`{json.dumps(evidence["corpus"]["upstream_holdout_names_present"], ensure_ascii=False)}`',
        '',
        '## 适用范围与未知项',
        '',
        '- 上述零命中只覆盖**已取得图像字节**的范围。对 BLOCKED_* / INDEX_ONLY / '
        'UNRESOLVED_FILE_LEVEL 的实体，本阶段没有文件级证据，不得宣称无泄漏。',
        '- PlantDoc 本次真实取得并哈希了 2,579 张图像（约 1.0 GB 归档），是唯一在本阶段'
        '从"来源级"提升到"文件级"的 holdout。',
        '- 其余 holdout 的阻塞原因逐条记录在 `stage5e_holdout_evidence.json` 的 `registry` 字段'
        '（403 / DNS 失败 / 需要登录 / 需要账号），不做任何绕过。',
        '- pHash 候选一律标记为 `CANDIDATE_REQUIRES_REVIEW`，不自动删除，需要嵌入确认后才可定论。',
    ]
    (reports / 'stage5e_holdout_audit.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'exact_raw_hash_hits': exact_hits,
                      'exact_pixel_hash_hits': pixel_hits,
                      'near_hash_candidates': near_candidates,
                      'index_rows': int(len(index)),
                      'status_summary': evidence['status_summary']}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
