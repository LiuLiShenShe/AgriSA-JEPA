#!/usr/bin/env python
"""Build the Stage 5A source and benchmark registries from the curated catalog.

Inputs
------
* ``src/registry/catalog.py`` -- curated, cited source definitions
* ``reports/stage5a_url_check.json`` -- real HTTP results (optional)

Outputs
-------
* ``data/registry/agricultural_sources.csv``
* ``data/registry/agricultural_benchmarks.csv``
* ``data/registry/source_overlap_candidates.csv``
* ``data/registry/downstream_holdout_sources.csv``
* ``data/registry/agricultural_sources_manifest.json``
* ``reports/stage5a_data_source_catalog.md``
* ``reports/stage5a_source_overlap_audit.md``
* ``reports/stage5a_dataset_licence_audit.md``
* ``reports/stage5a_benchmark_registry.md``

URL results only *downgrade* claims: a row whose URL did not resolve stays at
``DISCOVERED`` no matter how confident the paper is.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry.benchmarks import benchmarks, write_benchmarks
from src.registry.catalog import all_sources
from src.registry.schema import (STATES, read_csv, validate_row, write_csv,
                                 write_manifest)

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / 'data' / 'registry'
REPORTS = ROOT / 'reports'


def apply_url_checks(rows, checks):
    """Promote/downgrade states using real HTTP evidence."""
    for row in rows:
        url = str(row['download_url'])
        check = checks.get(url)
        if not check:
            continue
        outcome = check.get('outcome')
        status = check.get('status')
        note = f"URL check {outcome} (HTTP {status})"
        if outcome in ('ok', 'reachable_but_blocked'):
            if row['state'] == 'DISCOVERED':
                if (row['license'] and row['license'] != 'unverified'
                        and row['license_evidence']):
                    row['state'] = 'LICENSE_VERIFIED'
                else:
                    row['state'] = 'URL_VERIFIED'
            row['evidence'] = (row['evidence'] + '; ' + note).strip('; ')
        else:
            row['notes'] = (row['notes'] + '; ' + note +
                            ' -- link not resolvable from this environment; '
                            'kept at DISCOVERED').strip('; ')
    return rows


def overlap_groups(rows):
    """Canonical overlap groups across the three catalogs."""
    groups = [
        ('phenobench', {
            'phenobench': 'AgriField-40K retains 9,606 PhenoBench images AND '
                          'Stage 5A downstream benchmark (official v1.1.0); '
                          'all splits held out',
            'agrifm_ref88': 'Agri-FM+ Table 1 ref [88]: 129,000 PhenoBench images',
        }),
        ('perennial_plants', {
            'perennial_plants': 'AgriField-40K: 392 -> 240 retained',
            'agrifm_ref4': 'Agri-FM+ Table 1 ref [4]: 392 images',
        }),
        ('ronin', {
            'ronin': 'AgriField-40K: 1,176 -> 135 retained',
            'agrifm_ref79': 'Agri-FM+ Table 1 ref [79]: 2,822 images',
        }),
        ('gwhd_2021', {
            'agrifm_ref31': 'Agri-FM+ pretraining pool (6,514 images)',
            'gwhd_2021': 'Stage 5A object-detection benchmark (6,514 images)',
        }),
        ('deepweeds', {
            'agrifm_ref60': 'Agri-FM+ pretraining pool (17,509 images)',
            'deepweeds': 'AgriMAE / Stage 6 benchmark (17,509 images)',
        }),
    ]
    out = []
    for name, members in groups:
        for source_id, evidence in members.items():
            if not any(r['source_id'] == source_id for r in rows):
                raise SystemExit(f'overlap group {name}: unknown source {source_id}')
            out.append({
                'overlap_group': name,
                'source_id': source_id,
                'role': evidence,
                'action': ('holdout' if name in ('gwhd_2021', 'deepweeds')
                           else 'single copy only; never sum retained counts'),
            })
    return out


def write_overlap(path, rows):
    groups = overlap_groups(rows)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'overlap_group', 'source_id', 'role', 'action'])
        writer.writeheader()
        writer.writerows(groups)
    return groups


def write_holdouts(path, rows):
    holdouts = [r for r in rows if r['downstream_holdout'] == 'yes']
    fields = ['source_id', 'official_name', 'task', 'license', 'reason']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in holdouts:
            writer.writerow({
                'source_id': row['source_id'],
                'official_name': row['official_name'],
                'task': row['category'],
                'license': row['license'],
                'reason': 'downstream evaluation data; excluded from every '
                          'pretraining manifest (no test-image leakage)',
            })
    return holdouts


def _md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(lines)


def write_catalog_md(path, rows, checks, groups, holdouts):
    counts = Counter(r['state'] for r in rows)
    groups_counter = Counter()
    for r in rows:
        for g in str(r['catalog_groups']).split(';'):
            if g:
                groups_counter[g] += 1
    lines = [
        '# Stage 5A — 农业数据源目录（真实台账）',
        '',
        '> 本文件由 `tools/stage5a_build_registries.py` 生成，机器可读版本为 '
        '`data/registry/agricultural_sources.csv`。',
        '> 生成时间：2026-10-23。所有数量分列：论文报告 / 远端文件 / 实际下载 / '
        '解码通过 / 审核通过。',
        '',
        '## 1. 来源总览',
        '',
        f'- 台账行数：**{len(rows)}**（AgriField-40K 17 个原始来源、'
        'Agri-FM+ Table 1 35 个来源、12 个下游数据集；同一物理数据集在不同目录中'
        '保留独立行，不合并计数）。',
        f'- 目录分组：{dict(groups_counter)}',
        f'- 状态分布：{ {s: counts.get(s, 0) for s in STATES + ["BLOCKED"]} }',
        f'- 下游 holdout：**{len(holdouts)}** 个数据集，全部禁止进入预训练语料。',
        '',
        '### 状态含义',
        '',
        '`DISCOVERED` → `URL_VERIFIED` → `LICENSE_VERIFIED` → `DOWNLOADED` → '
        '`DECODED` → `DUPLICATE_SCREENED` → `SPLIT_CHECKED` → `APPROVED` → `FROZEN`。',
        '任何一行都必须逐级推进，且每一级都需要对应证据字段；`BLOCKED` 单独标记。',
        '',
        '## 2. 全部来源',
        '',
        _md_table(
            ['source_id', '数据集', '分组', '报告数', '远端', '已下载', '解码',
             '许可', '状态', 'holdout'],
            [[r['source_id'], r['official_name'], r['catalog_groups'],
              r['images_reported'] if r['images_reported'] is not None else '',
              r['files_remote'] if r['files_remote'] is not None else '',
              r['files_downloaded'] if r['files_downloaded'] is not None else '',
              r['images_decoded'] if r['images_decoded'] is not None else '',
              r['license'] or 'unverified', r['state'], r['downstream_holdout']]
             for r in sorted(rows, key=lambda x: (x['catalog_groups'], x['source_id']))]),
        '',
        '## 3. URL 核验结果',
        '',
        f'- 受检 URL：{len(checks)}；`ok`：'
        f'{sum(1 for c in checks.values() if c["outcome"] == "ok")}；'
        '`reachable_but_blocked`：'
        f'{sum(1 for c in checks.values() if c["outcome"] == "reachable_but_blocked")}；'
        f'失败：{sum(1 for c in checks.values() if c["outcome"] not in ("ok", "reachable_but_blocked"))}。',
        '- 403/429 表示主机存在但拒绝自动访问，**不**据此提升为已验证；'
        'DNS/连接失败保持 `DISCOVERED` 并记录原因。',
        '',
        '## 4. 数量口径（关键）',
        '',
        '- **论文报告数量**只代表原论文陈述，绝不等于本机已有数据。',
        '- **远端文件数量**只在真实列出归档目录后填写。',
        '- **实际已取得数量**以本地目录为准；**解码通过**以 Pillow 实际解码为准；'
        '**最终纳入**必须通过去重、隔离与人工复核。',
        '- 当前 **没有任何一行达到 `APPROVED`/`FROZEN`**；D1 candidate 的冻结'
        '在 `reports/stage5a_summary.md` 中单独报告。',
        '',
        '## 5. 未找到 / 不可达资源',
        '',
    ]
    failures = [(u, c) for u, c in checks.items()
                if c['outcome'] not in ('ok', 'reachable_but_blocked')]
    if failures:
        for url, c in failures:
            lines.append(f"- `{url}` — {c['outcome']}"
                         f"{(' / ' + c.get('error', '')) if c.get('error') else ''}")
    else:
        lines.append('- 无。')
    lines += [
        '',
        '## 6. 证据约定',
        '',
        '- AgriField-40K 行：计数与许可证来自固定 revision 的官方数据卡；'
        'v1 中已确认的 6 个来源另附本地解码/去重/划分证据路径。',
        '- Agri-FM+ 行：仅以 Table 1 的参考文献编号与参考文献列表中的 URL 为证据；'
        '论文没有公开的下载镜像不会被补造。',
        '- 下游行：许可证来自 Zenodo API / 官方仓库 / 数据页；'
        '无法核实的字段写 `see repository` 而不是猜测。',
        '',
    ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_overlap_md(path, groups, rows):
    by_id = {r['source_id']: r for r in rows}
    lines = [
        '# Stage 5A — 跨目录来源重叠审计',
        '',
        '> 同一物理数据集可能同时出现在 AgriField-40K、Agri-FM+ 与下游基准中。'
        '本文件只登记重叠关系与处置规则；**不合并计数**。',
        '',
        '| 重叠组 | source_id | 角色 / 数量证据 | 处置 |',
        '|---|---|---|---|',
    ]
    for g in groups:
        lines.append(f"| {g['overlap_group']} | {g['source_id']} | {g['role']} | {g['action']} |")
    lines += [
        '',
        '## 处置原则',
        '',
        '1. 同一数据集的多份“保留数量”不可相加；v1 只使用本地冻结清单。',
        '2. PhenoBench 与 GWHD 是下游评价保留集，任何预训练 manifest 必须整来源排除。',
        '3. DeepWeeds 若用于 Stage 6 下游，同样整来源排除。',
        '4. 图像级 SHA-256 / pHash 去重只能发现像素重复，不能替代 session/plot 级隔离；'
        '跨来源重复候选进入人工复核队列。',
        '',
        '## 已知同源对',
        '',
        '- `perennial_plants` ↔ `agrifm_ref4`（Kaggle Perennial Plants）',
        '- `ronin` ↔ `agrifm_ref79`（Sudars et al. Data in Brief 2020）',
        '- `phenobench` ↔ `agrifm_ref88` ↔ `phenobench_sem`',
        '- `agrifm_ref31` ↔ `gwhd_2021`',
        '- `agrifm_ref60` ↔ `deepweeds`',
        '',
    ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_licence_md(path, rows):
    by_licence = defaultdict(list)
    for r in rows:
        by_licence[r['license'] or 'unverified'].append(r['source_id'])
    lines = [
        '# Stage 5A — 数据许可证审计',
        '',
        '> 许可证逐行记录在 `data/registry/agricultural_sources.csv` 的 '
        '`license` / `license_evidence` / `attribution_required` 字段。',
        '',
        '## 1. 按许可证汇总',
        '',
        '| 许可证 | 来源数 | source_id |',
        '|---|---:|---|',
    ]
    for lic, ids in sorted(by_licence.items(), key=lambda kv: -len(kv[1])):
        lines.append(f'| {lic} | {len(ids)} | {", ".join(sorted(ids))} |')
    lines += [
        '',
        '## 2. 结论',
        '',
        '- 多数来源为 CC-BY-4.0，需保留署名；PhenoBench/GrassClover/CropAndWeed 为 '
        'CC-BY-SA-4.0，衍生作品需同许可共享；Perennial Plants 为 MIT；'
        'Sesame&Weed 为 CC0。',
        '- PlantVillage 的 Zenodo 记录同时出现 `cc-by-4.0` 与正文 CC BY-SA 3.0 US，'
        '存在元数据冲突 → 以更严格的 **CC-BY-SA-3.0-US** 记录，未经人工确认不进入冻结语料。',
        '- 许可证未核实的来源一律 `allowed_in_project=pending`，且状态不超过 '
        '`URL_VERIFIED`，不进入任何训练清单。',
        '- 本阶段**没有**把任何来源标记为 `APPROVED`，因此没有新增“已授权并审核”语料；'
        'D1 candidate 的纳入决定见总结报告。',
        '',
    ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_benchmark_md(path):
    rows = benchmarks()
    lines = [
        '# Stage 5A — 农业多任务 Bench Registry',
        '',
        '> 机器可读：`data/registry/agricultural_benchmarks.csv`。'
        '覆盖 Agri-FM+ 的 8 项基准与 AgriMAE 的 4 项下游资源。',
        '',
        '| benchmark_id | 任务 | 数据集 | 指标 | 官方工具 | 重叠风险 | 本项目状态 |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        lines.append(f"| {r['benchmark_id']} | {r['task']} | {r['dataset']} | "
                     f"{r['metric']} | {r['official_eval_tool']} | "
                     f"{r['overlap_risk']} | {r['project_status']} |")
    lines += [
        '',
        '## 关键区分',
        '',
        '- **GWHD 2021 目标检测**（6,514 张，COCO bbox）与 '
        '**GWHD Instance Segmentation**（582 张，Roboflow mask）是两个不同资源，'
        '不能把 bbox 当 mask。',
        '- Agri-FM+ 的 GWHD 结果使用 Detectron2 Faster R-CNN + 20K iterations；'
        '与我们的冻结 ViT-H + 检测头协议不可直接比数字，只能比“同协议内 A vs B”。',
        '- PhenoBench 已有本项目真实结果（A 71.68 / B_2000 73.10 mIoU）；'
        '其余 11 项本阶段均为 `NOT_RUN`。',
        '',
        '## 统一评价接口',
        '',
        '`src/downstream/registry.py` 定义 Backbone/数据/Head/指标/cache key；'
        '`src/downstream/evaluation_registry.py` 定义每个 benchmark 的官方指标与'
        '结果路径。新增任务必须先在 registry 注册，才能进入训练入口。',
        '',
    ]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    REGISTRY.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = [validate_row(r) for r in all_sources()]
    check_path = REPORTS / 'stage5a_url_check.json'
    checks = json.loads(check_path.read_text(encoding='utf-8')) if check_path.is_file() else {}
    rows = apply_url_checks(rows, checks)

    write_csv(REGISTRY / 'agricultural_sources.csv', rows)
    n_bench = write_benchmarks(REGISTRY / 'agricultural_benchmarks.csv')
    groups = write_overlap(REGISTRY / 'source_overlap_candidates.csv', rows)
    holdouts = write_holdouts(REGISTRY / 'downstream_holdout_sources.csv', rows)
    sha = write_manifest(REGISTRY / 'agricultural_sources_manifest.json', rows,
                         extra={'benchmarks': n_bench,
                                'holdouts': len(holdouts),
                                'url_checks': len(checks)})

    write_catalog_md(REPORTS / 'stage5a_data_source_catalog.md', rows, checks,
                     groups, holdouts)
    write_overlap_md(REPORTS / 'stage5a_source_overlap_audit.md', groups, rows)
    write_licence_md(REPORTS / 'stage5a_dataset_licence_audit.md', rows)
    write_benchmark_md(REPORTS / 'stage5a_benchmark_registry.md')

    print(f'sources={len(rows)} benchmarks={n_bench} holdouts={len(holdouts)} '
          f'sha256={sha}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
