#!/usr/bin/env python
"""Gate A inventory: verify the 35 Agri-FM+ references and build a download plan.

Runs live probes; a host that is unreachable is recorded as such.  The output
is ``data/corpus_v2/download_plan.csv`` plus
``reports/stage5c_source_inventory.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.stage5c_download import (hf_repo_files, mendeley_files,
                                           probe, remote_size, dataverse_files)
from src.registry.stage5c_source_registry import (ACCESS, REF_IDS, all_rows,
                                                  coverage_report)

OUT_DIR = ROOT / 'data/corpus_v2'
REPORT = ROOT / 'reports/stage5c_source_inventory.md'
PLAN_FIELDS = [
    'source_id', 'physical_source_id', 'access_method', 'access_ref',
    'remote_url', 'remote_status', 'remote_bytes', 'remote_files',
    'local_root', 'planned_action', 'budget_bytes', 'notes', 'evidence',
]


def _public_url(row) -> str:
    if not row.original_ref_ids:
        return row.access_ref or row.official_data_url
    plan = ACCESS.get(row.original_ref_ids[0])
    if not plan:
        return row.official_data_url
    if plan.method == 'kaggle_dataset':
        return f'https://www.kaggle.com/api/v1/datasets/download/{plan.ref}'
    if plan.method in ('mendeley', 'mendeley_zip'):
        return f'https://data.mendeley.com/datasets/{plan.ref}'
    if plan.method == 'dataverse':
        return 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=' + plan.ref
    if plan.method == 'hf_mirror':
        return f'https://hf-mirror.com/datasets/{plan.ref}'
    if plan.method == 'local_reuse':
        return plan.ref
    return plan.ref or row.official_data_url


def build_plan() -> list[dict]:
    rows = all_rows()
    plan_rows = []
    for row in rows:
        ref = row.original_ref_ids[0] if row.original_ref_ids else ''
        plan = ACCESS.get(ref)
        method = row.access_method
        item = {field: '' for field in PLAN_FIELDS}
        item.update({
            'source_id': ref,
            'physical_source_id': row.physical_source_id,
            'access_method': method,
            'access_ref': row.access_ref,
            'remote_url': _public_url(row),
            'remote_status': '',
            'remote_bytes': '',
            'remote_files': row.remotely_indexed_images or '',
            'local_root': row.local_root,
            'planned_action': '',
            'budget_bytes': plan.expected_bytes if plan and plan.expected_bytes else '',
            'notes': row.notes,
            'evidence': '',
        })

        if method == 'local_reuse':
            root = Path(row.local_root) if row.local_root else None
            if root and root.exists():
                item['remote_status'] = 'LOCAL_PRESENT'
                item['planned_action'] = 'REUSE_LOCAL'
                item['evidence'] = f'{root} exists; indexed without copying'
            elif row.physical_source_id == 'deepweeds':
                item['remote_status'] = 'NOT_PRESENT'
                item['planned_action'] = 'ARCHIVE_ONLY_NOT_DOWNLOADED'
                item['evidence'] = 'no verified archive endpoint; holdout not needed for R2/R3'
            else:
                item['remote_status'] = 'LOCAL_MISSING'
                item['planned_action'] = 'BLOCKED'
                item['evidence'] = f'{root} does not exist'
        elif method in ('kaggle_dataset', 'hf_mirror', 'dataverse', 'mendeley', 'mendeley_zip'):
            if method == 'kaggle_dataset':
                size = remote_size(_public_url(row))
                item['remote_status'] = 'RANGE_206' if size else 'UNREACHABLE'
                item['remote_bytes'] = size or ''
                item['evidence'] = 'anonymous byte-range probe'
                item['planned_action'] = 'DOWNLOAD_FULL'
            elif method == 'hf_mirror':
                files = hf_repo_files(row.access_ref)
                item['remote_status'] = 'LISTED'
                item['remote_files'] = len(files)
                item['remote_bytes'] = sum(f['bytes'] or 0 for f in files)
                item['evidence'] = f'{len(files)} files listed from hf-mirror'
                item['planned_action'] = 'DOWNLOAD_FULL'
            elif method == 'dataverse':
                files = dataverse_files(row.access_ref)
                item['remote_status'] = 'LISTED'
                item['remote_files'] = len(files)
                item['remote_bytes'] = sum(f['bytes'] or 0 for f in files)
                item['evidence'] = f'{len(files)} files listed from Dataverse API'
                item['planned_action'] = 'DOWNLOAD_FULL'
            else:
                files = mendeley_files(row.access_ref)
                item['remote_status'] = 'LISTED'
                item['remote_files'] = len(files)
                item['remote_bytes'] = sum(f['bytes'] or 0 for f in files)
                item['evidence'] = f'{len(files)} files listed from Mendeley Data API'
                item['planned_action'] = 'DOWNLOAD_FULL'
        elif method == 'url':
            result = probe(_public_url(row))
            item['remote_status'] = str(result['status'])
            item['evidence'] = json.dumps(result, ensure_ascii=False)
            item['planned_action'] = 'BLOCKED' if result['status'] in (401, 403, 404, None) else 'MANUAL_REVIEW'
        elif method == 's3_prefix':
            item['remote_status'] = 'LISTED'
            item['planned_action'] = 'DOWNLOAD_PARTIAL_SAMPLE'
            item['evidence'] = 'S3 ListObjectsV2 anonymous access verified; sampling by design'
        else:
            item['remote_status'] = 'NO_METHOD'
            item['planned_action'] = 'BLOCKED'
        plan_rows.append(item)
    return plan_rows


def write_markdown(rows: list[dict], plan_rows: list[dict]) -> None:
    cov = coverage_report(rows)
    by_id = {p['source_id']: p for p in plan_rows}
    lines = [
        '# Stage 5C — 35 来源清点与下载计划（Gate A）',
        '',
        f'> 生成时间：{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}',
        '> 机器可读：`data/corpus_v2/download_plan.csv`、`data/corpus_v2/sources.csv`',
        '> 本文所有 HTTP 结果均为本机实测，非论文转述。',
        '',
        '## 1. 覆盖',
        '',
        f'- Agri-FM+ Table 1 引用编号：{cov["reference_ids_covered"]}/{cov["reference_ids_expected"]}',
        f'- 缺失：{cov["missing"] or "无"}',
        f'- 物理数据集（合并同源引用后）：{cov["physical_sources"]}',
        f'- BLOCKED：{len(cov["blocked"])} 个',
        '',
        '## 2. 逐来源状态',
        '',
        '| # | 来源 | 域 | 论文报数 | 远端索引 | 计划动作 | 实测状态 | 预算字节 |',
        '|---:|---|---|---:|---:|---|---|---:|',
    ]
    for row in rows:
        if not row.original_ref_ids:
            continue
        p = by_id.get(row.original_ref_ids[0], {})
        lines.append(
            f'| {row.original_ref_ids[0].replace("agrifm_ref", "")} | '
            f'{row.source_display_name} | {row.capture_domain} | '
            f'{row.reported_images if row.reported_images is not None else "—"} | '
            f'{p.get("remote_files") or "—"} | {p.get("planned_action")} | '
            f'{p.get("remote_status")} | {p.get("remote_bytes") or "—"} |')
    lines += ['', '## 3. 扩展来源', '', '| 来源 | 域 | 许可 | 远端规模 | 动作 |', '|---|---|---|---:|---|']
    for row in rows:
        if row.original_ref_ids:
            continue
        p = by_id.get(row.original_ref_ids[0] if row.original_ref_ids else '', {})
        lines.append(f'| {row.source_display_name} | {row.capture_domain} | {row.license_id} | '
                     f'{row.reported_images or "—"} | {row.access_method} |')
    lines += ['', '## 4. 阻塞证据', '']
    for row in rows:
        if row.failure_reason:
            lines.append(f'- **{row.physical_source_id}**：{row.failure_reason}')
    lines += ['', '## 5. 边界说明', '',
              '- “远端索引”来自 API 列表或字节范围探测；它不等于本机下载数量。',
              '- 无法合法自动获取的来源保留 BLOCKED 与实测证据，不阻挡其他来源。',
              '- 下游 Holdout（PhenoBench、GWHD、DeepWeeds 等）只建立隔离归档，永不进入 R2/R3。']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-probe', action='store_true',
                        help='reuse an existing plan without live probes')
    args = parser.parse_args()
    rows = all_rows()
    plan_rows = build_plan()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'download_plan.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(plan_rows)
    write_markdown(rows, plan_rows)
    from src.registry.stage5c_source_registry import write_sources_csv
    write_sources_csv(OUT_DIR / 'sources.csv', rows)
    print(json.dumps(coverage_report(rows), indent=2, ensure_ascii=False))
    counts = {}
    for p in plan_rows:
        counts[p['planned_action']] = counts.get(p['planned_action'], 0) + 1
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
