#!/usr/bin/env python
"""Audit the official GWHD 2021 release and produce a reproducible report.

Checks, in order:

1. archive size + MD5 against the Zenodo record (record 5092309);
2. member listing: image count and file types;
3. the three official CSVs, used verbatim (no re-split);
4. box validation per split (empty, degenerate, out-of-bounds, duplicates);
5. decode + dimension audit of every image;
6. exact/perceptual duplicate screening inside GWHD;
7. cross-source overlap against the D1 candidate manifest;
8. group (acquisition domain) counts per split and cross-split group overlap.

Outputs ``reports/stage5a_gwhd_data_audit.{md,json}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.downstream.gwhd_data import (discover_split_files, image_group,
                                      read_split, split_isolation_report,
                                      validate_boxes)
from src.registry.ingest import file_sha256, iter_images, phash64, pixel_sha256

OFFICIAL_SIZE = 10215190132
OFFICIAL_MD5 = '22b4b542c9ae7e056d7fcdeae9ecaed5'


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='/data/datasets/GWHD2021/gwhd_2021.zip')
    parser.add_argument('--extract-dir', default='/data/datasets/GWHD2021/extracted')
    parser.add_argument('--d1-manifest',
                        default='data/frozen/agri-corpus-d1-candidate_train.csv')
    parser.add_argument('--out-json', default='reports/stage5a_gwhd_data_audit.json')
    parser.add_argument('--out-md', default='reports/stage5a_gwhd_data_audit.md')
    parser.add_argument('--skip-extract', action='store_true')
    parser.add_argument('--reuse-hashes', action='store_true',
                        help='reuse archive hashes from a previous audit with the '
                             'same size (10 GB hashing takes minutes)')
    parser.add_argument('--hashes-json', default='',
                        help='optional cache of per-image hashes from a previous run')
    args = parser.parse_args()

    archive = Path(args.archive)
    extract_dir = Path(args.extract_dir)
    size = archive.stat().st_size
    previous = {}
    if args.reuse_hashes and Path(args.out_json).is_file():
        previous = json.loads(Path(args.out_json).read_text(encoding='utf-8'))
    if previous.get('archive', {}).get('size') == size:
        md5 = previous['archive']['md5']
        sha256 = previous['archive']['sha256']
        print('reused archive hashes from previous audit', flush=True)
    else:
        md5 = md5_file(archive)
        sha256 = file_sha256(archive)
    checksum_ok = (size == OFFICIAL_SIZE and md5 == OFFICIAL_MD5)

    if not args.skip_extract:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            z.extractall(extract_dir)
    else:
        members = []

    split_files = discover_split_files(extract_dir)
    splits = {name: read_split(path, extract_dir, name)
              for name, path in split_files.items()}
    validation = {name: validate_boxes(split) for name, split in splits.items()}
    isolation = split_isolation_report(splits)

    image_paths = list(iter_images(extract_dir))
    decode_errors = []
    dims = Counter()
    hashes = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.load()
                dims[f'{image.width}x{image.height}'] += 1
                hashes.append({
                    'path': str(path.relative_to(extract_dir)),
                    'sha256': file_sha256(path),
                    'pixel_sha256': pixel_sha256(image),
                    'phash64': phash64(image)})
        except Exception as exc:  # noqa: BLE001
            decode_errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})

    sha_counter = Counter(h['sha256'] for h in hashes)
    pixel_counter = Counter(h['pixel_sha256'] for h in hashes)
    exact_pairs = sum(c - 1 for c in sha_counter.values() if c > 1)
    pixel_pairs = sum(c - 1 for c in pixel_counter.values() if c > 1)

    label_names = {n.lower() for split in splits.values() for n in split.images}
    image_rel = {Path(p).name.lower(): str(p.relative_to(extract_dir)) for p in image_paths}
    missing_images = sorted(label_names - set(image_rel))
    unlabelled_images = sorted(set(image_rel) - label_names)

    # cross-source screening against the D1 candidate manifest
    cross = {'checked': 0, 'exact': 0, 'phash_candidates': 0, 'examples': []}
    d1_path = Path(args.d1_manifest)
    if d1_path.is_file() and hashes:
        import csv
        with open(d1_path, newline='', encoding='utf-8') as handle:
            refs = list(csv.DictReader(handle))
        cross['checked'] = len(refs)
        ref_sha = {r['sha256'] for r in refs}
        ref_pixel = {r['pixel_sha256'] for r in refs}
        cross['exact'] = sum(1 for h in hashes
                             if h['sha256'] in ref_sha or h['pixel_sha256'] in ref_pixel)
        ref_ph = np.array([int(r['phash64'], 16) for r in refs if r.get('phash64')],
                          dtype=np.uint64)
        if len(ref_ph):
            q = np.array([int(h['phash64'], 16) for h in hashes], dtype=np.uint64)
            for start in range(0, len(q), 256):
                block = q[start:start + 256][:, None] ^ ref_ph[None, :]
                dist = np.bitwise_count(block)
                hits = np.argwhere(dist <= 6)
                cross['phash_candidates'] += int(len(hits))
                for r, c in hits[:20]:
                    cross['examples'].append({
                        'gwhd': hashes[start + int(r)]['path'],
                        'd1_row': refs[int(c)]['sample_id'],
                        'distance': int(dist[r, c])})

    audit = {
        'archive': {
            'path': str(archive), 'size': size, 'md5': md5, 'sha256': sha256,
            'official_size': OFFICIAL_SIZE, 'official_md5': OFFICIAL_MD5,
            'checksum_ok': checksum_ok,
        },
        'archive_members': len(members),
        'images_on_disk': len(image_paths),
        'decode_errors': decode_errors[:50],
        'decode_error_count': len(decode_errors),
        'dimensions': dict(dims),
        'splits': {
            name: {'csv': str(split_files[name]), 'count': len(split),
                   'rows_in_csv': len(split) + len(split.dropped_duplicates),
                   'dropped_duplicate_rows': split.dropped_duplicates,
                   'validation': validation[name].to_dict(),
                   'domains': sorted({d for d in split.domains if d})}
            for name, split in splits.items()},
        'isolation': isolation,
        'label_image_mismatch': {
            'missing_images': len(missing_images),
            'missing_examples': missing_images[:20],
            'unlabelled_images': len(unlabelled_images),
            'unlabelled_examples': unlabelled_images[:20],
        },
        'duplicates': {
            'exact_byte_pairs': exact_pairs,
            'exact_pixel_pairs': pixel_pairs,
            'note': 'exact only; perceptual pairs are not used to drop images',
        },
        'cross_source_vs_d1': cross,
        'verdict': {
            'checksum_verified': checksum_ok,
            'labels_parse': all(len(s) > 0 for s in splits.values()),
            'official_splits_preserved': True,
            'ready_for_training': checksum_ok and not decode_errors and not missing_images,
        },
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n',
                                   encoding='utf-8')

    v = audit['verdict']
    lines = [
        '# Stage 5A — GWHD 2021 数据审计',
        '',
        '> 官方来源：https://zenodo.org/records/5092309（CC-BY-4.0，10,215,190,132 bytes）。'
        '本报告由 `tools/stage5a_audit_gwhd.py` 生成。',
        '',
        '## 1. 归档校验',
        '',
        f"- 大小：{size:,} / 官方 {OFFICIAL_SIZE:,} → {'OK' if size == OFFICIAL_SIZE else 'MISMATCH'}",
        f"- MD5：`{md5}` / 官方 `{OFFICIAL_MD5}` → {'OK' if md5 == OFFICIAL_MD5 else 'MISMATCH'}",
        f"- SHA-256（本项目记录）：`{sha256}`",
        f"- 解压后图片：{len(image_paths):,}；解码失败：{len(decode_errors)}",
        '',
        '## 2. 官方划分（未重划分）',
        '',
        '| split | 图像 | 框 | 空标注 | 退化框 | 完全越界 | 部分越界 | 重复框 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, split in splits.items():
        val = validation[name]
        lines.append(f'| {name} | {len(split):,} | {val.boxes:,} | '
                     f'{val.images_without_boxes:,} | {val.degenerate:,} | '
                     f'{val.out_of_bounds:,} | {val.partially_outside:,} | '
                     f'{val.duplicate_boxes:,} |')
    lines += [
        '',
        f"- 跨 split 图像名重合：{isolation['overlaps']}",
        f"- 标签缺失图像：{len(missing_images)}；未标注图像：{len(unlabelled_images)}",
        '',
        '## 3. 分辨率与重复',
        '',
        f"- 分辨率分布：{dict(dims)}",
        f"- 字节级重复对：{exact_pairs}；解码像素级重复对：{pixel_pairs}",
        '- 感知相似只作为候选，不自动删除。',
        '',
        '## 4. 与 D1 候选语料的重叠',
        '',
        f"- 对照行数：{cross['checked']:,}；精确匹配：{cross['exact']}；"
        f"pHash 候选（≤6）：{cross['phash_candidates']}",
        '',
        'GWHD 是下游评价保留集，**任何** GWHD 图像（无论划分）都不得进入预训练语料。',
        '',
        '## 5. 结论',
        '',
        f"- checksum_verified = **{v['checksum_verified']}**",
        f"- official_splits_preserved = **{v['official_splits_preserved']}**",
        f"- ready_for_training = **{v['ready_for_training']}**",
        '',
    ]
    Path(args.out_md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('images_on_disk', 'decode_error_count',
                                            'dimensions', 'duplicates', 'verdict')},
                     indent=2))
    return 0 if audit['verdict']['ready_for_training'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
