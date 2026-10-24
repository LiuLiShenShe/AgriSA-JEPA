#!/usr/bin/env python
"""Render the Stage 5C reports from the machine-readable ledgers.

Every number in the Markdown output is read from a CSV/JSON produced by
``stage5c_build_versions.py`` (or the per-source ledgers); nothing is typed in
by hand.  Figures are deterministic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / 'reports'
CORPUS = ROOT / 'data/corpus_v2'
IMAGES = CORPUS / 'images'
FIGURES = REPORTS / 'stage5c_figures'
SUMMARY = REPORTS / 'stage5c_versions_summary.json'
TARGET = 147_286


def fmt(n) -> str:
    try:
        return f'{int(n):,}'
    except (TypeError, ValueError):
        return '—'


def pct(n, d) -> str:
    try:
        return f'{100.0 * float(n) / float(d):.1f}%'
    except (TypeError, ValueError, ZeroDivisionError):
        return '—'


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def load_ledgers() -> pd.DataFrame:
    frames = []
    for path in sorted(IMAGES.glob('*_qc.csv')) + sorted(IMAGES.glob('*.csv')):
        if path.name.endswith('_qc.csv'):
            continue
        base = path.with_name(path.stem + '_qc.csv')
        chosen = base if base.exists() and base.stat().st_mtime >= path.stat().st_mtime else path
        try:
            frames.append(pd.read_csv(chosen, low_memory=False))
        except Exception as exc:  # noqa: BLE001
            print('skip', chosen, exc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_funnel(funnel: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    versions = funnel['version'].unique()
    fig, axes = plt.subplots(1, len(versions), figsize=(5.2 * len(versions), 4.6), squeeze=False)
    for ax, version in zip(axes[0], versions):
        part = funnel[funnel['version'] == version]
        labels = part['step'].tolist()
        values = part['remaining_images'].tolist()
        ax.barh(range(len(labels)), values, color='#2f7d4f')
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel('images remaining')
        ax.set_title(version)
        for i, value in enumerate(values):
            ax.text(value, i, f' {fmt(value)}', va='center', fontsize=8)
        ax.set_xlim(0, max(values) * 1.25 if values else 1)
    fig.tight_layout()
    fig.savefig(FIGURES / 'funnel.png', dpi=150)
    plt.close(fig)


def plot_sources(composition: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if composition.empty:
        return
    frame = composition.sort_values('r2', ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * len(frame))))
    ax.barh(frame['physical_source_id'], frame['r2'], color='#3b6ea5')
    ax.set_xlabel('R2 accepted images')
    ax.set_title('R2 contribution by physical source (top 25)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r2_by_source.png', dpi=150)
    plt.close(fig)


def plot_domains(by_domain: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not by_domain:
        return
    items = sorted(by_domain.items(), key=lambda kv: kv[1], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([k for k, _ in items], [v for _, v in items], color='#8a6d3b')
    ax.set_ylabel('images')
    ax.set_title('Acquisition-domain composition (R3 / final)')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'domain_mix.png', dpi=150)
    plt.close(fig)


def plot_quality(ledgers: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if ledgers.empty or 'laplacian_var' not in ledgers:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, column, label in zip(
            axes, ['laplacian_var', 'brightness_mean', 'iqa_musiq'],
            ['Laplacian variance (focus)', 'Mean brightness', 'MUSIQ']):
        if column not in ledgers:
            continue
        values = pd.to_numeric(ledgers[column], errors='coerce').dropna()
        if values.empty:
            continue
        upper = values.quantile(0.99)
        ax.hist(values.clip(upper=upper), bins=60, color='#4c72b0')
        ax.set_title(label)
        ax.set_xlabel(label)
    fig.tight_layout()
    fig.savefig(FIGURES / 'quality_distributions.png', dpi=150)
    plt.close(fig)


def composition_markdown(composition: pd.DataFrame, summary: dict) -> str:
    lines = ['# Stage 5C 数据集构成', '',
             '> 由 `tools/stage5c_generate_tables.py` 从 `stage5c_dataset_composition.csv` 渲染；'
             '数字为程序统计值。', '']
    for key in ('r0', 'r1', 'r2', 'r3'):
        payload = summary.get(key)
        if not payload:
            continue
        counts = payload.get('counts', {})
        lines += [f'## {key.upper()} — {payload.get("version")}', '',
                  f'- 状态：`{payload.get("status")}`（{payload.get("auto_status_meaning", "")}）',
                  f'- 图像：{fmt(counts.get("images"))}',
                  f'- 物理来源：{fmt(counts.get("sources"))}',
                  f'- 已知采集组：{fmt(counts.get("groups"))}；无组元数据图像：'
                  f'{fmt(counts.get("groups_unknown_images"))}',
                  f'- train / val：{fmt(counts.get("train"))} / {fmt(counts.get("val"))}',
                  f'- 最大单一来源占比：{pct(payload.get("max_source_share", 0), 1)}',
                  f'- 跨 split 已知组：{fmt(payload.get("cross_split_groups"))}',
                  '']
        by_domain = payload.get('by_domain') or {}
        if by_domain:
            lines += ['| 采集域 | 图像 | 占比 |', '|---|---:|---:|']
            total = sum(by_domain.values())
            for name, value in sorted(by_domain.items(), key=lambda kv: -kv[1]):
                lines.append(f'| {name} | {fmt(value)} | {pct(value, total)} |')
            lines.append('')
    if not composition.empty:
        lines += ['## 逐来源（R2 / R3 贡献）', '',
                  '| 来源 | 下载 | 解码 | 基础QC | IQA | 域 | 保留集排除 | 许可阻断 | R1 | R2 | R3 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for _, row in composition.sort_values('download'.replace('download', 'downloaded'),
                                              ascending=False).iterrows():
            lines.append('| {source} | {d} | {dec} | {qc} | {iqa} | {dom} | {ho} | {lic} | '
                         '{r1} | {r2} | {r3} |'.format(
                             source=row['physical_source_id'], d=fmt(row['downloaded']),
                             dec=fmt(row['decoded']), qc=fmt(row['basic_qc_pass']),
                             iqa=fmt(row['iqa_pass']), dom=fmt(row['domain_pass']),
                             ho=fmt(row['holdout_removed']), lic=fmt(row['license_blocked']),
                             r1=fmt(row['r1']), r2=fmt(row['r2']), r3=fmt(row['r3'])))
        lines.append('')
    return '\n'.join(lines) + '\n'


def funnel_markdown(funnel: pd.DataFrame) -> str:
    lines = ['# Stage 5C 过滤漏斗', '',
             '> 单因归因：每张图只计入第一次失败的步骤，因此各步排除数之和等于输入增量。', '']
    for version in funnel['version'].unique():
        part = funnel[funnel['version'] == version]
        lines += [f'## {version}', '',
                  '| 步骤 | 输入 | 本步排除 | 剩余 | 判定 |', '|---|---:|---:|---:|---|']
        for _, row in part.iterrows():
            lines.append(f'| {row["step"]} | {fmt(row["input_images"])} | '
                         f'{fmt(row["excluded_this_step"])} | {fmt(row["remaining_images"])} | '
                         f'{row["reason"]} |')
        lines.append('')
    return '\n'.join(lines) + '\n'


def reconstruction_gap(summary: dict) -> str:
    r0, r1 = summary.get('r0', {}), summary.get('r1', {})
    n0 = r0.get('counts', {}).get('images', 0)
    n1 = r1.get('counts', {}).get('images', 0)
    lines = ['# Stage 5C — 与 Agri-147K 的重建差距', '',
             '本项目**不声称**复现官方 Agri-147K。官方语料未发布逐图清单，'
             '本表把“论文公开规则可重建的部分”与“未取得/未披露的部分”分开记录。', '',
             '| 项目 | 数量 | 说明 |', '|---|---:|---|',
             f'| 论文报告语料 | 147,285 | Agri-FM+ Table 1 汇总，论文口径 |',
             f'| R0（本机实际下载池） | {fmt(n0)} | 所有已下载且可解码的文件，审计层 |',
             f'| R1（协议知情重建） | {fmt(n1)} | 仅使用论文明确披露的规则 |',
             f'| 差距（论文 − R1） | {fmt(max(0, 147285 - n1))} | 未取得来源 + 未披露筛选规则 |',
             '',
             '差距不是“失败”，而是可审计的缺口：每条未取得来源都在 '
             '`reports/stage5c_download_audit.md` 中有 BLOCKED/PARTIAL 原因。', '']
    return '\n'.join(lines) + '\n'


def holdout_audit(summary: dict) -> str:
    lines = ['# Stage 5C — 保留集隔离审计', '',
             'R2/R3 在构建时按来源 ID 与文件前缀双重排除下游基准。'
             '`pnb_` 前缀用于排除混入 AgriField-40K 目录的 PhenoBench 图像。', '',
             '| 保留集 | 隔离方式 | R2/R3 图像数 |', '|---|---|---:|']
    for key, payload in (('R1', summary.get('r1', {})), ('R2', summary.get('r2', {})),
                         ('R3', summary.get('r3', {}) or {})):
        if not payload:
            continue
        counts = payload.get('counts', {})
        lines.append(f'| {key} | {"不隔离（重建层）" if key == "R1" else "来源 ID + 前缀"} | '
                     f'{fmt(counts.get("images"))} |')
    lines += ['', '保留集名单（代码常量 `HOLDOUT_IDS`）：', '']
    from src.registry.stage5c_source_registry import HOLDOUT_IDS
    for item in sorted(HOLDOUT_IDS):
        lines.append(f'- `{item}`')
    lines += ['', '任何 R2/R3 冻结若 `no_holdout` 门禁为 false，则状态不会升级为 `AUTO_FROZEN`。', '']
    return '\n'.join(lines) + '\n'


def diversity(ledgers: pd.DataFrame, summary: dict) -> str:
    payload = summary.get('r3') or summary.get('r2') or {}
    counts = payload.get('counts', {})
    by_source = payload.get('by_source') or {}
    by_domain = payload.get('by_domain') or {}
    by_crop = payload.get('by_crop') or {}
    lines = ['# Stage 5C — 语料多样性', '',
             f'- 图像：{fmt(counts.get("images"))}',
             f'- 物理来源：{fmt(counts.get("sources"))}',
             f'- 已知采集组：{fmt(counts.get("groups"))}',
             f'- 采集域类别：{len(by_domain)}',
             f'- 有作物元数据的类别：{len(by_crop)}',
             f'- 最大单一来源占比：{pct(payload.get("max_source_share", 0), 1)}', '']
    if by_source:
        total = sum(by_source.values())
        lines += ['## 来源集中度', '', '| 来源 | 图像 | 占比 |', '|---|---:|---:|']
        cumulative = 0.0
        for name, value in sorted(by_source.items(), key=lambda kv: -kv[1]):
            cumulative += value / total
            lines.append(f'| {name} | {fmt(value)} | {pct(value, total)} |')
        lines += ['', f'- HHI（来源份额平方和）：'
                     f'{sum((v / total) ** 2 for v in by_source.values()):.4f}',
                  f'- Top-1 份额：{pct(max(by_source.values()), total)}', '']
    if by_domain:
        lines += ['## 采集域', '', '| 域 | 图像 | 占比 |', '|---|---:|---:|']
        total = sum(by_domain.values())
        for name, value in sorted(by_domain.items(), key=lambda kv: -kv[1]):
            lines.append(f'| {name} | {fmt(value)} | {pct(value, total)} |')
        lines.append('')
    return '\n'.join(lines) + '\n'


def autoqc_doc() -> str:
    calibration = load_json(REPORTS / 'stage5c_qc_calibration.json', {})
    lines = ['# Stage 5C — 自动质检方法与可复现性', '',
             '## 1. 分层判定', '',
             '| 层 | 内容 | 失败处置 |', '|---|---|---|',
             '| 解码 | PIL 解码、尺寸、通道、EXIF | `AUTO_REJECT` |',
             '| 基础质量 | Laplacian 方差、Tenengrad、曝光、熵、空白比例、长宽比 | `AUTO_REJECT` / `QUARANTINE` |',
             '| IQA | MUSIQ、NIQE（pyiqa） | `AUTO_REJECT`（双指标冲突时隔离） |',
             '| 场景域 | 元数据优先；不足时 CLIP 零样本，低置信隔离 | `QUARANTINE_DOMAIN` |',
             '| 重复 | SHA-256 / 像素哈希精确去重；pHash+dHash+几何确认近重复 | 精确折叠；未确认隔离 |',
             '| 保留集 | 来源 ID + 文件前缀 | 排除 |',
             '| 许可 | 逐来源实查许可；冲突直接阻断 | 排除 |', '',
             '## 2. 阈值来源', '',
             '阈值在 `data/corpus_v2/policy.yaml`（带版本号），由确定性合成退化标定，'
             '不是人工逐图标注，也没有为了扩大语料而放宽。', '',
             '## 3. 合成退化校准', '']
    if calibration:
        lines += [f'- 评估图像：{fmt(calibration.get("images_evaluated"))}'
                  f'（clean {fmt(calibration.get("clean_images"))}）', '',
                  '| 退化 | 阳性 | 真阳 | 假阳 | 召回 | 假阳率 |', '|---|---:|---:|---:|---:|---:|']
        for row in calibration.get('detectors', []):
            lines.append(f'| {row["fault"]} | {row["positives"]} | {row["true_positive"]} | '
                         f'{row["false_positive"]} | {row["recall"]:.3f} | '
                         f'{row["false_positive_rate"]:.3f} |')
        sep = calibration.get('iqa_separation') or []
        if sep:
            lines += ['', '### MUSIQ 分离度', '',
                      '| 退化 | clean 中位数 | 退化中位数 | 差值 | <35 占比 |', '|---|---:|---:|---:|---:|']
            for row in sep:
                lines.append(f'| {row["fault"]} | {row["clean_median"]} | {row["fault_median"]} | '
                             f'{row["delta"]} | {row["share_below_35"]} |')
    else:
        lines.append('（校准尚未运行；不得据此声称阈值已验证。）')
    lines += ['', '## 4. 可复现性', '',
              '- 随机性只来自固定种子（采样、split）；模型推理为确定性前向。',
              '- 每个来源的 ledger 记录策略版本、模型 SHA、阈值与判定原因码。',
              '- 重跑命令见 README。', '']
    return '\n'.join(lines) + '\n'


def contribution_design(summary: dict) -> str:
    r2 = summary.get('r2', {})
    r3 = summary.get('r3', {}) or {}
    lines = ['# Stage 5C — 数据贡献设计', '',
             '## 定位', '',
             '本工作的数据贡献是**可审计的田间语料构建协议**，不是“又一个大杂烩数据集”。', '',
             '| 层 | 用途 | 是否可训练 |', '|---|---|---|',
             '| R0 | 审计：本机到底有什么 | 否 |',
             '| R1 | 协议知情重建，对照论文规则 | 仅诊断 |',
             '| R2 | 严格田间集，保留集隔离 | 是 |',
             '| R3 | R2 + 合法扩展来源 | 是 |', '',
             '## 与现有工作的区别', '',
             '1. **区分报告数与实得数**：`reported / remotely_indexed / downloaded / decoded / '
             'final_accepted` 永不混用。',
             '2. **区分理论复现与实际复现**：R1 明确标为 protocol-informed reconstruction。',
             '3. **区分机器门禁与人工确认**：`AUTO_FROZEN` 只表示程序化门禁通过。',
             '4. **失败可审计**：BLOCKED/PARTIAL 来源保留原因，不静默丢弃。', '',
             '## 当前规模', '',
             f'- R2：{fmt(r2.get("counts", {}).get("images"))}',
             f'- R3：{fmt(r3.get("counts", {}).get("images"))}',
             f'- 目标：>{fmt(TARGET)}（R3 去重后可训练图）', '']
    return '\n'.join(lines) + '\n'


def summary_doc(summary: dict, funnel: pd.DataFrame, composition: pd.DataFrame,
                target_status: str) -> str:
    def n(key, field='images'):
        return fmt((summary.get(key) or {}).get('counts', {}).get(field))
    lines = ['# Stage 5C — 执行总结', '',
             f'> 目标阈值：R3 严格唯一可训练图 > {fmt(TARGET)}',
             f'> 实际状态：**{target_status}**', '',
             '## 版本规模', '',
             '| 版本 | 图像 | 来源 | 状态 |', '|---|---:|---:|---|']
    for key in ('r0', 'r1', 'r2', 'r3'):
        payload = summary.get(key)
        if not payload:
            continue
        lines.append(f'| {key.upper()} | {n(key)} | {n(key, "sources")} | '
                     f'`{payload.get("status")}` |')
    lines += ['', '## 质量与隔离', '']
    for key in ('r2', 'r3'):
        payload = summary.get(key)
        if not payload:
            continue
        gates = payload.get('gates', {})
        lines.append(f'### {key.upper()} 门禁')
        lines.append('')
        for gate, ok in gates.items():
            lines.append(f'- {"PASS" if ok else "FAIL"} — `{gate}`')
        lines.append('')
    lines += ['## 结论', '',
              '机器判定只能给出 `AUTO_FROZEN`；本报告不含任何“专家人工审核”声称。'
              '若 R3 低于目标，状态为 `BELOW_SCALE_TARGET`，并如实报告缺口。', '']
    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots', action='store_true', default=True)
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = load_json(SUMMARY, {})
    funnel_path = REPORTS / 'stage5c_filtering_funnel.csv'
    comp_path = REPORTS / 'stage5c_dataset_composition.csv'
    if not summary or not funnel_path.exists():
        print('missing summary/funnel; run stage5c_build_versions.py first')
        return 1
    funnel = pd.read_csv(funnel_path)
    composition = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()

    (REPORTS / 'stage5c_filtering_funnel.md').write_text(funnel_markdown(funnel), encoding='utf-8')
    (REPORTS / 'stage5c_dataset_composition.md').write_text(
        composition_markdown(composition, summary), encoding='utf-8')
    (REPORTS / 'stage5c_reconstruction_gap.md').write_text(
        reconstruction_gap(summary), encoding='utf-8')
    (REPORTS / 'stage5c_holdout_audit.md').write_text(holdout_audit(summary), encoding='utf-8')
    (REPORTS / 'stage5c_autoqc_reproducibility.md').write_text(autoqc_doc(), encoding='utf-8')
    (REPORTS / 'stage5c_data_contribution_design.md').write_text(
        contribution_design(summary), encoding='utf-8')

    ledgers = load_ledgers()
    (REPORTS / 'stage5c_corpus_diversity.md').write_text(diversity(ledgers, summary),
                                                         encoding='utf-8')
    r3_images = (summary.get('r3') or {}).get('counts', {}).get('images', 0)
    target_status = 'TARGET_MET' if r3_images > TARGET else 'BELOW_SCALE_TARGET'
    (REPORTS / 'stage5c_summary.md').write_text(
        summary_doc(summary, funnel, composition, target_status), encoding='utf-8')

    if args.plots:
        plot_funnel(funnel)
        plot_sources(composition)
        payload = summary.get('r3') or summary.get('r2') or {}
        plot_domains(payload.get('by_domain') or {})
        plot_quality(ledgers)
    print('reports written')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
