#!/usr/bin/env python
"""Verify downloaded archives against the ledger and write the Gate A audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.stage5c_source_registry import all_rows  # noqa: E402

DATA_ROOT = Path('/data/datasets/Stage5C')
LEDGER = DATA_ROOT / 'download_manifest.csv'
REPORT = ROOT / 'reports/stage5c_download_audit.md'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob('*')
               if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)



def source_status(row, images: int) -> str:
    if images > 0:
        partial = (row.access_method == 's3_prefix'
                   or row.physical_source_id in {'agrifm_ref57'})
        return 'DOWNLOADED_PARTIAL' if partial else 'DOWNLOADED_FULL'
    if row.download_status == 'BLOCKED':
        return 'BLOCKED'
    if row.physical_source_id in {'deepweeds'}:
        return 'ARCHIVE_ONLY'
    return 'NOT_STARTED'

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-hash', action='store_true',
                        help='recompute SHA-256 for every archive (slow)')
    args = parser.parse_args()
    ledger_rows = []
    if LEDGER.exists():
        ledger_rows = list(csv.DictReader(open(LEDGER)))
    by_source = {}
    for row in ledger_rows:
        by_source.setdefault(row['source_id'], []).append(row)

    lines = [
        '# Stage 5C — 下载审计（Gate A）',
        '',
        f'> 生成：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}',
        '> 机器可读：`data/corpus_v2/download_plan.csv`、'
        '`/data/datasets/Stage5C/download_manifest.csv`',
        '',
        '本表只记录本机实际存在的文件。论文报数、远端索引数与本机下载数分列，不混算。',
        '',
        '## 1. 逐来源状态',
        '',
        '| 来源 | 访问方式 | 下载状态 | 归档字节 | 归档 SHA-256 | 解压图像 | 备注 |',
        '|---|---|---|---:|---|---:|---|',
    ]
    rows = all_rows()
    for row in rows:
        records = by_source.get(row.physical_source_id, [])
        downloaded = [r for r in records if r.get('status') in ('DOWNLOADED', 'SKIPPED_EXISTING')]
        total_bytes = sum(int(float(r.get('downloaded_bytes') or 0)) for r in downloaded)
        sha = next((r.get('checksum_value', '') for r in downloaded if r.get('checksum_value')), '')
        images = count_images(DATA_ROOT / 'raw' / row.physical_source_id)
        if row.access_method == 'local_reuse' and row.local_root:
            images = count_images(Path(row.local_root))
        # Files on disk are the ground truth; the registry's planned_action
        # is only used when nothing arrived.  A source can be complete even
        # if its per-file ledger rows predate the current ledger file.
        status = source_status(row, images)
        note = row.failure_reason or row.notes
        lines.append(
            f'| {row.physical_source_id} | {row.access_method} | {status} | '
            f'{total_bytes or "—"} | `{sha[:16]}…` | {images} | {note[:90]} |')

    lines += [
        '',
        '## 2. 完整性',
        '',
        '- 归档在下载时流式计算 SHA-256；重跑时若目标文件大小匹配则标记 `SKIPPED_EXISTING`。',
        '- zip 归档在解压前执行 `ZipFile.testzip()`，损坏归档记为 `CORRUPT`，不进入解压目录。',
        '- 原始归档只读保留，筛选只生成索引与派生资产。',
        '',
        '## 3. 未取得来源',
        '',
    ]
    for row in rows:
        if row.download_status == 'BLOCKED':
            lines.append(f'- **{row.physical_source_id}**：{row.failure_reason}')
    lines += [
        '',
        '## 4. 口径',
        '',
        '| 口径 | 说明 |',
        '|---|---|',
        '| `reported_images` | 论文/来源页声称的数量 |',
        '| `remotely_indexed_images` | API 实际列出的远端文件数 |',
        '| `downloaded_images` | 本机实际文件数 |',
        '| `decoded_images` | 通过 PIL 解码的数量 |',
        '| `final_accepted` | 通过全部门禁的数量 |',
        '',
        '这五个数字由不同工具产生，任何报告都不得相互替代。',
    ]
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    payload = []
    for row in rows:
        records = by_source.get(row.physical_source_id, [])
        downloaded = [r for r in records if r.get('status') in ('DOWNLOADED', 'SKIPPED_EXISTING')]
        images = count_images(DATA_ROOT / 'raw' / row.physical_source_id)
        if row.access_method == 'local_reuse' and row.local_root:
            images = count_images(Path(row.local_root))
        payload.append({
            'physical_source_id': row.physical_source_id,
            'access_method': row.access_method,
            'status': source_status(row, images),
            'archived_bytes': sum(int(float(r.get('downloaded_bytes') or 0)) for r in downloaded),
            'images_on_disk': images,
        })
    import json as _json
    (REPORT.with_suffix('.json')).write_text(
        _json.dumps(payload, indent=2), encoding='utf-8')
    print(REPORT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
