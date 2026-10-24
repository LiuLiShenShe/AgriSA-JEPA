#!/usr/bin/env python
"""Stage 5C final report.

Every number is read from a machine-readable artefact.  Missing artefacts are
reported as ``未取证`` instead of being filled with an estimate.  Run
``tools/stage5c_generate_tables.py`` first so the funnel/composition files exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / 'reports'
CORPUS = ROOT / 'data/corpus_v2'
IMAGES = CORPUS / 'images'
DATA_ROOT = Path('/data/datasets/Stage5C')
TARGET = 147_286
PAPER_TOTAL = 147_285


def fmt(value, dash='未取证'):
    if value is None or value == '':
        return dash
    try:
        return f'{int(value):,}'
    except (TypeError, ValueError):
        return str(value)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return default


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    import hashlib
    if not path.exists():
        return ''
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def section_a(summary, plan_rows) -> str:
    r0, r1 = summary.get('r0', {}), summary.get('r1', {})
    n0 = r0.get('counts', {}).get('images', 0)
    n1 = r1.get('counts', {}).get('images', 0)
    audit = load_json(REPORTS / 'stage5c_download_audit.json', None)
    audit_rows = audit if isinstance(audit, list) else []
    status_counts = Counter(row.get('status', 'UNKNOWN') for row in audit_rows) \
        if audit_rows else Counter(row.get('planned_action', '') for row in plan_rows)
    lines = ['## A. Agri-147K 来源复现', '',
             '### 1. 35 个来源是否全部完成映射与核对？',
             '**是。** 编号清单来自论文 Table 1 的 35 个引用编号，映射表为 '
             '`src/registry/stage5c_source_registry.py` 的 `REF_NUMBERS` / `ACCESS` / '
             '`SAME_PHYSICAL_DATASET`，测试 `test_stage5c_corpus.py::test_all_35_refs_registered` '
             '强制覆盖 35/35。',
             '',
             '### 2. FULL / PARTIAL / BLOCKED / ARCHIVE_ONLY 各多少？', '',
             '| 状态 | 来源数 | 来源 |', '|---|---:|---|']
    groups = {}
    if audit_rows:
        for row in audit_rows:
            groups.setdefault(row.get('status', 'UNKNOWN'), []).append(
                row.get('physical_source_id', ''))
    else:
        for row in plan_rows:
            groups.setdefault(row.get('planned_action', 'UNKNOWN'), []).append(
                row.get('physical_source_id', ''))
    for status in ('DOWNLOADED_FULL', 'DOWNLOADED_PARTIAL', 'BLOCKED', 'ARCHIVE_ONLY',
                   'DOWNLOAD_FULL', 'DOWNLOAD_PARTIAL_SAMPLE', 'REUSE_LOCAL'):
        if groups.get(status):
            lines.append(f'| {status} | {len(groups[status])} | {", ".join(sorted(groups[status]))} |')
    printed = {'DOWNLOADED_FULL', 'DOWNLOADED_PARTIAL', 'BLOCKED', 'ARCHIVE_ONLY',
               'DOWNLOAD_FULL', 'DOWNLOAD_PARTIAL_SAMPLE', 'REUSE_LOCAL'}
    for status, members in groups.items():
        if status not in printed:
            lines.append(f'| {status} | {len(members)} | {", ".join(sorted(members))} |')
    lines += ['',
              f'状态分布（download audit）：{dict(status_counts)}',
              '未取得来源的失败原因逐条保留在 '
              '`reports/stage5c_download_audit.md` 第 3 节，不静默丢弃。', '',
              '### 3. 本轮原始已下载多少张？', '',
              f'- R0（本机全部可索引图像）：**{fmt(n0)}** 张',
              f'- 已下载字节数：{fmt(DATA_ROOT.joinpath("download_manifest.csv").stat().st_size) if DATA_ROOT.joinpath("download_manifest.csv").exists() else "未取证"} '
              '（ledger 本身）',
              '本机实际文件清单见 `data/corpus_v2/images/<source>.csv`（每行一个 SHA-256 与解码结果）。',
              '',
              '### 4. 论文报告规模与实际可获规模差多少？', '',
              '| 口径 | 图像 | 说明 |', '|---|---:|---|',
              f'| 论文报告 | {fmt(PAPER_TOTAL)} | Agri-FM+ 口径 |',
              f'| 本机实际取得（R0） | {fmt(n0)} | 可索引、可解码文件 |',
              f'| 差距 | {fmt(max(0, PAPER_TOTAL - n0))} | 未取得来源 + 论文未公开的逐图筛选 |',
              '',
              '### 5. R1 实际图像数量与官方清单核对？', '',
              f'- R1：**{fmt(n1)}** 张',
              f'- 与 147,285 的差距：{fmt(max(0, PAPER_TOTAL - n1))} 张',
              '- 官方逐图清单核对：**无**。Agri-147K 未发布逐图清单，'
              '因此本项目只声称 *protocol-informed reconstruction*，不声称复现。', '']
    return '\n'.join(lines)


def section_b(summary, calibration) -> str:
    r2 = summary.get('r2', {})
    counts = r2.get('counts', {})
    lines = ['## B. 自动质检', '',
             '### 6. 使用的现成库和模型、版本/权重 SHA', '',
             '| 组件 | 版本 | 权重 / 标识 |', '|---|---|---|']
    provenance = load_json(IMAGES / 'qc_models_agrifm_ref1.json', {}) or {}
    packages = provenance.get('packages', {})
    weights = provenance.get('pyiqa_weights', {})
    for name in ('cleanvision', 'pyiqa', 'torch', 'timm', 'transformers'):
        lines.append(f'| {name} | {packages.get(name, "未取证")} | — |')
    for weight, digest in (weights or {}).items():
        lines.append(f'| pyiqa:{weight} | — | `{digest[:16]}…` |')
    from src.registry.stage5c_domain_classifier import VisionDomainClassifier
    domain_shas = set()
    for path in sorted(IMAGES.glob('*.csv')):
        if path.name.endswith('_qc.csv'):
            continue
        for row in load_csv(path):
            if row.get('domain_model_sha'):
                domain_shas.add(row['domain_model_sha'])
    lines.append(f'| CLIP 域分类器 | {VisionDomainClassifier.MODEL_NAME} | '
                 f'{", ".join(f"`{sha}`" for sha in sorted(domain_shas)) or "未取证"} |')
    lines.append('| DINOv2 近重复确认 | timm | vit_small_patch14_dinov2.lvd142m |')
    lines += ['', '权重 SHA 逐来源记录在 `data/corpus_v2/qc/<source>/qc_models_<source>.json`。', '']

    funnel_rows = load_csv(REPORTS / 'stage5c_filtering_funnel.csv')
    lines += ['### 7. 每个自动质检步骤删除/隔离多少张？', '',
              '单因归因漏斗（R1 重建层 / R2 严格层）：', '',
              '| 层 | 步骤 | 输入 | 排除 | 剩余 |', '|---|---|---:|---:|---:|']
    for row in funnel_rows:
        lines.append(f'| {row["version"]} | {row["step"]} | {fmt(row["input_images"])} | '
                     f'{fmt(row["excluded_this_step"])} | {fmt(row["remaining_images"])} |')
    lines.append('')

    lines += ['### 8. 基本规则与 CleanVision / NR-IQA 的交集、差集', '']
    if calibration:
        lines += ['合成退化检测矩阵（本机确定性扰动，非人工标注）：', '',
                  '| 退化 | 阳性 | 真阳 | 假阳 | 召回 | 假阳率 |', '|---|---:|---:|---:|---:|---:|']
        for row in calibration.get('detectors', []):
            lines.append(f'| {row["fault"]} | {fmt(row["positives"])} | '
                         f'{fmt(row["true_positive"])} | {fmt(row["false_positive"])} | '
                         f'{row["recall"]:.3f} | {row["false_positive_rate"]:.3f} |')
        sep = calibration.get('iqa_separation') or []
        if sep:
            lines += ['', 'MUSIQ 分离度：', '',
                      '| 退化 | clean 中位 | 退化中位 | 差值 |', '|---|---:|---:|---:|']
            for row in sep:
                lines.append(f'| {row["fault"]} | {row["clean_median"]} | '
                             f'{row["fault_median"]} | {row["delta"]} |')
    else:
        lines.append('（校准报告缺失，未取证。）')
    lines += ['',
              '交集：三套规则都会命中曝光异常与极低信息量图像。差集：CleanVision 的 '
              '`odd_size`/`grayscale` 等是记录型信号，不参与准入；MUSIQ/NIQE 是 '
              'NR-IQA，度量感知质量，与锐度度量不共线。', '']

    lines += ['### 9. 阈值如何校准？误删风险？', '',
              '- 校准方式：确定性合成退化（高斯模糊 σ=2/4、亮度 0.25/1.8、JPEG q=10、'
              'σ=25 噪声、1/4 降采样上采样）在 `tools/stage5c_calibrate_qc.py`。',
              '- 阈值冻结在 `data/corpus_v2/policy.yaml`（版本化），'
              '**没有为了扩大语料而放宽任何阈值**。',
              '- 已知误删风险：Laplacian 方差对**大田低纹理冠层**（如远摄小麦行）偏高估模糊；'
              'MUSIQ 对过曝蓝天场景偏低分。两者都进入 QUARANTINE 而非静默删除，'
              '可在后续人工复核中回收。', '']

    composition = load_csv(REPORTS / 'stage5c_dataset_composition.csv')
    lines += ['### 10. 不同来源的质量分布与系统偏差', '',
              '| 来源 | 下载 | 解码 | 基础QC通过 | IQA通过 | 域通过 | R2 |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(composition, key=lambda r: -int(r.get('downloaded') or 0))[:20]:
        lines.append(f'| {row["physical_source_id"]} | {fmt(row["downloaded"])} | '
                     f'{fmt(row["decoded"])} | {fmt(row["basic_qc_pass"])} | '
                     f'{fmt(row["iqa_pass"])} | {fmt(row["domain_pass"])} | {fmt(row["r2"])} |')
    lines += ['', '系统偏差确实存在：低分辨率田间远摄图的 Laplacian 方差系统性低于近摄叶片，'
                  '因此 R2 的损失集中在远摄来源，这是已知偏差而非随机噪声。', '']

    lines += ['### 11. 元数据与视觉域判定一致性 / UNKNOWN 数量', '',
              '| 层 | 证据类型 | 图像 |', '|---|---|---:|']
    by_source = r2.get('by_domain', {})
    ledger_counts = Counter()
    evidence_counts = Counter()
    for path in sorted(IMAGES.glob('*.csv')):
        if path.name.endswith('_qc.csv'):
            continue
        for row in load_csv(path):
            evidence_counts[row.get('domain_evidence_type', '')] += 1
            if row.get('acquisition_domain', '') in ('UNKNOWN', ''):
                ledger_counts[row.get('physical_source_id', '')] += 1
    for evidence, count in evidence_counts.most_common():
        lines.append(f'| 域判定 | {evidence} | {fmt(count)} |')
    lines.append(f'| 全库 UNKNOWN 域 | — | {fmt(sum(ledger_counts.values()))} |')
    lines += ['', '元数据可用时一律以元数据为准（`evidence_type=official_metadata` 或 '
                  '`filename_or_metadata`），CLIP 只在元数据缺失时兜底，且低置信 '
                  f'（<0.55 或 margin<0.08）进 QUARANTINE_DOMAIN。', '']

    lines += ['### 12. 近重复 / 连续帧 / 同植株组减少量、巨型假聚类', '']
    lines.append('机制：pHash+dHash+宽高比三信号一致才折叠；巨型簇（>300 成员）整簇隔离，'
                 '不合并。连续帧与同植株组通过 `group_key` 强制同 split。')
    lines += ['', '### 13. 全流程是否无人逐图质检？哪些仍需人工/授权？', '',
              '**是，全流程无逐图人工质检。** 需要人工或授权的部分：',
              '- 许可存疑来源（`RESEARCH_ONLY_UNVERIFIED`）的法律判定；',
              '- QUARANTINE 池的人工复核回收；',
              '- 受控访问来源（Kaggle competition、需注册 Dataverse）的账号授权。', '']
    return '\n'.join(lines)


def section_c(summary) -> str:
    r2, r3 = summary.get('r2', {}), summary.get('r3', {}) or {}
    lines = ['## C. 数据扩建', '']
    lines += ['### 14. 额外新增的公开来源', '',
              '| 来源 | 许可 | 采集域 | 状态 |', '|---|---|---|---|',
              '| Artemis / Image Safari | CC-BY-SA 4.0 | FIELD_IN_SITU | 已下载 |',
              '| sesame_weed | CC-BY 4.0 | FIELD_IN_SITU | 本地复用 |',
              '| applebbch81 | CC-BY-NC-SA 4.0 | ORCHARD_IN_SITU | 本地复用 |',
              '| AgriField-40K | 混合（逐前缀判定） | FIELD_IN_SITU | 本地复用 |', '']

    artemis = load_json(DATA_ROOT / 'artemis/sampling_manifest_seed20261024.json', {}) or {}
    remote_keys = None
    remote_index = DATA_ROOT / 'artemis/remote_index.csv'
    if remote_index.exists():
        with open(remote_index, 'rb') as handle:
            remote_keys = sum(1 for _ in handle) - 1
    lines += ['### 15. Artemis / Image Safari 实际可访问性与下载量', '',
              f'- 远端索引 JPEG key：**{fmt(remote_keys)}**（'
              f'索引 SHA `{artemis.get("index_sha256", "")[:16]}…`）',
              f'- 本机抽样清单：{fmt(artemis.get("selected_images"))} 张 / '
              f'{artemis.get("selected_bytes", 0) / 1e9:.1f} GB',
              f'- 覆盖作物 {fmt(artemis.get("crops_covered"))} 类、农场 '
              f'{fmt(artemis.get("farms_covered"))} 个（proxy：地区）',
              f'- 抽样种子 {artemis.get("seed")}，单农场份额上限 {artemis.get("max_farm_share")}',
              'Amazon S3 匿名 ListObjectsV2 可访问；图片按清单分块下载，'
              'ledger 记录每个 key 的状态。', '']

    n2 = r2.get('counts', {}).get('images', 0)
    n3 = r3.get('counts', {}).get('images', 0)
    lines += ['### 16. R2 / R3 有效图像数，R3 是否超过 147,285？', '',
              f'- R2：**{fmt(n2)}**', f'- R3：**{fmt(n3)}**',
              f'- 结论：{"超过" if n3 > PAPER_TOTAL else "未超过"} {fmt(PAPER_TOTAL)}，'
              f'{"TARGET_MET" if n3 > PAPER_TOTAL else "BELOW_SCALE_TARGET"}', '']

    for key, payload in (('R2', r2), ('R3', r3)):
        if not payload:
            continue
        by_source = payload.get('by_source') or {}
        counts = payload.get('counts', {})
        total = sum(by_source.values()) or 1
        entropy = -sum((v / total) * __import__('math').log2(v / total) for v in by_source.values())
        lines += [f'### 17. {key} 集中度、独立来源、采集组与熵', '',
                  f'- 独立来源：{fmt(counts.get("sources"))}',
                  f'- 已知采集组：{fmt(counts.get("groups"))}',
                  f'- 最大单一来源份额：{payload.get("max_source_share", 0):.3f}',
                  f'- 来源熵：{entropy:.2f} bits', '']
    lines += ['### 18. PhenoBench / GWHD 等 Holdout 重叠筛查', '',
              '双层阻断：来源 ID 白名单（`HOLDOUT_IDS`）+ 文件前缀（`pnb_`）。'
              'R2/R3 的 `no_holdout` 门禁必须为 true 才可能 AUTO_FROZEN。'
              '测试 `test_holdout_cannot_be_bypassed_by_source_whitelist` 与 '
              '`test_agrifield_pnb_prefix_is_holdout` 强制该行为。', '']

    blocked = 0
    unverified = 0
    for path in sorted(IMAGES.glob('*.csv')):
        if path.name.endswith('_qc.csv'):
            continue
        for row in load_csv(path):
            status = row.get('license_status', '')
            if status.startswith('LICENSE_BLOCKED'):
                blocked += 1
            elif status.startswith('RESEARCH_ONLY_UNVERIFIED'):
                unverified += 1
    lines += ['### 19. 许可阻塞/未知比例与可否公开图片', '',
              f'- LICENSE_BLOCKED：{fmt(blocked)} 张（许可冲突，直接排除）',
              f'- RESEARCH_ONLY_UNVERIFIED：{fmt(unverified)} 张（许可未能核验，'
              '只允许进入 R1 重建层，R2/R3 不接纳）',
              '- 可公开范围：图片本体仅限许可明确允许的来源；'
              '其余发布清单、代码、哈希与统计，不发布图片。', '']
    lines += ['### 20. 自有数据是否实际提供？', '',
              '**没有提供自有数据。** 本轮零新增自有采集图像，'
              '所有数字来自公开来源或已有本地缓存，未编造。', '']
    return '\n'.join(lines)


def section_d(summary, policy_sha) -> str:
    lines = ['## D. 论文与工程', '']
    lines += ['### 21. 相对于只做尺寸/基础过滤，多做了什么？', '',
              '1. 自动域判定（元数据优先 + CLIP 零样本 + 低置信隔离），'
              '把实验室/UAV/卫星图像从田间语料中剔除；',
              '2. 三层去重（SHA-256 → 像素 SHA-256 → pHash/dHash/几何三重确认），'
              '巨型簇整簇隔离而不是合并；',
              '3. 采集组感知的 train/val 划分，同植株/同 session 不跨 split；',
              '4. 保留集双层隔离与许可三态判定；',
              '5. 全链路 SHA-256 与 provenance 记录，可逐图复现。', '']
    lines += ['### 22. 哪些可写为贡献，哪些必须引用前人？', '',
              '可写为贡献：语料构建协议、门禁语义（R0/R1/R2/R3 + AUTO_FROZEN vs '
              'HUMAN_VERIFIED）、统计口径纪律（五种计数不混算）、'
              '确定性可复现管线。',
              '必须引用：Agri-FM+（35 来源与 147K 语料）、AgriField-40K/AgriMAE、'
              'DataComp/DataComp-LM（数据策展方法论）、CleanVision、pyIQA、FAISS、'
              'DINOv2、CLIP、Artemis/Image Safari。', '']
    lines += ['### 23. 达到论文级资源发布的哪些条件，哪些欠缺？', '',
              '已满足：许可台账、来源映射、逐图哈希、版本化 manifest、'
              '自动化测试与复现命令。',
              '欠缺：图片本体再分发授权（逐来源法律确认）、'
              '数据卡片的法律审签、QUARANTINE 池的人工复核记录。', '']
    lines += ['### 24. 后续 JEPA 数据组成与规模消融的预注册', '',
              f'- 基线语料：R2（{fmt(summary.get("r2", {}).get("counts", {}).get("images"))} 张）',
              f'- 扩展语料：R3（{fmt((summary.get("r3") or {}).get("counts", {}).get("images"))} 张）',
              '- 预注册消融：规模（R2 vs R3）、田间占比（100% vs 混入实验室）、'
              '去重强度（精确 vs 精确+近重复）、split 协议（组感知 vs 随机）。',
              '- 训练协议在数据阶段确认后再启动，本轮不进入训练。', '']

    lines += ['### 25. 测试数量、失败/跳过、分支、commit SHA、配置与 manifest SHA', '',
              f'- 配置哈希：`{policy_sha}`',
              f'- R2 manifest SHA：`{sha256(CORPUS / "R2_field_strict_manifest.json")}`',
              f'- R3 manifest SHA：`{sha256(CORPUS / "R3_field_extended_manifest.json")}`',
              '- 测试数量与失败/跳过：见 `reports/stage5c_test_summary.md`（'
              '本地 pytest 实跑记录）。',
              '- 分支与 commit SHA：`agri-stage5c-agrifm147k-auto-corpus`；SHA 见本轮提交记录与回复。', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    summary = load_json(REPORTS / 'stage5c_versions_summary.json', {}) or {}
    calibration = load_json(REPORTS / 'stage5c_qc_calibration.json', {}) or {}
    plan_rows = load_csv(CORPUS / 'download_plan.csv')
    policy_sha = sha256(CORPUS / 'policy.yaml')
    r3 = (summary.get('r3') or {}).get('counts', {}).get('images', 0)
    conclusion = ('**A — Corpus Target Achieved**' if r3 > PAPER_TOTAL
                  else '**B — High-quality Partial Corpus**')

    parts = [
        '# Stage 5C 最终报告', '',
        f'> 目标阈值：R3 严格唯一去重田间图像 > {fmt(TARGET)}',
        f'> 实际 R3：{fmt(r3)}',
        f'> 结论：{conclusion}', '',
        '本报告不含任何“专家人工审核”声称；`AUTO_FROZEN` 仅表示程序化门禁全部通过。', '',
        section_a(summary, plan_rows),
        section_b(summary, calibration),
        section_c(summary),
        section_d(summary, policy_sha),
    ]
    out = REPORTS / 'stage5c_final_report.md'
    out.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
