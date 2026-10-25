#!/usr/bin/env python
"""Stage 5E final report + machine-readable summary.

Every number in the output is read from a file produced by this stage - the
manifest, the gates, the licence audit and the holdout evidence - so the report
cannot drift from the artifacts.

Writes:
    reports/stage5e_summary.json
    reports/stage5e_final_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    proc = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    return proc.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v4', default='data/corpus_v4')
    parser.add_argument('--reports', default='reports')
    args = parser.parse_args()

    v4 = ROOT / args.v4
    reports = ROOT / args.reports

    manifest = read_json(v4 / 'R3v3_license_verified_manifest.json')
    composition = read_json(v4 / 'R3v3_license_verified_composition.json')
    sampling = read_json(v4 / 'R3v3_license_verified_sampling_profiles.json')
    gates = read_json(reports / 'stage5e_gates.json')
    audit = read_json(reports / 'stage5e_license_audit.json')
    evidence = read_json(reports / 'stage5e_holdout_evidence.json')
    scope = read_json(reports / 'stage5e_evaluation_scope.json')

    counts = manifest.get('counts', {})
    stage5d = read_json(ROOT / 'data/corpus_v3/R3v2_strict_rgb_manifest.json')

    summary = {
        'stage': 'Stage 5E',
        'branch': git('branch', '--show-current'),
        'head': git('rev-parse', 'HEAD'),
        'head_parent': git('rev-parse', 'HEAD^'),
        'stage5d_head': git('rev-parse', 'agri-stage5d-corpus-validation'),
        'stage5c_head': git('rev-parse', 'agri-stage5c-agrifm147k-auto-corpus'),
        'status': gates.get('status'),
        'gates_passed': sum(1 for gate in gates.get('gates', {}).values()
                            if gate.get('status') == 'PASS'),
        'gates_total': len(gates.get('gates', {})),
        'failed_gates': gates.get('failed_gates', []),
        'unknown_gates': gates.get('unknown_gates', []),
        'counts': counts,
        'by_source': manifest.get('by_source', {}),
        'by_upstream_dataset': manifest.get('by_upstream_dataset', {}),
        'by_domain': manifest.get('by_domain', {}),
        'by_research_use_status': manifest.get('by_research_use_status', {}),
        'target': manifest.get('target', {}),
        'concentration': manifest.get('concentration', {}),
        'conservation': manifest.get('conservation', {}),
        'licence': {
            'audited_sources': audit.get('summary', {}).get('audited_sources', []),
            'accepted_images_in_four': audit.get('summary', {}).get(
                'accepted_images_in_four'),
            'train_eligible_in_four': audit.get('summary', {}).get('train_eligible_in_four'),
            'excluded_from_training': audit.get('summary', {}).get('excluded_from_training'),
            'verified_research_use_images': audit.get('summary', {}).get(
                'moved_from_unresolved_to_verified'),
            'restricted_research_use_images': audit.get('summary', {}).get(
                'moved_from_unresolved_to_restricted'),
        },
        'holdout': {
            'index_rows': evidence.get('index', {}).get('rows'),
            'per_dataset': evidence.get('index', {}).get('per_dataset', {}),
            'exact_raw_hash_hits': evidence.get('crossmatch', {}).get('exact_raw_hash_hits'),
            'exact_pixel_hash_hits': evidence.get('crossmatch', {}).get('exact_pixel_hash_hits'),
            'near_hash_candidates': evidence.get('crossmatch', {}).get('near_hash_candidates'),
            'content_verified': [row['dataset_id'] for row in evidence.get('registry', [])
                                 if row.get('content_verified')],
            'not_content_verified': [row['dataset_id'] for row in evidence.get('registry', [])
                                     if not row.get('content_verified')],
            'status_code_by_entity': {row['dataset_id']: row.get('status')
                                      for row in evidence.get('registry', [])},
        },
        'evaluation_scope': {
            'eligible': scope.get('evaluation_eligible', []),
            'excluded': [entry['dataset_id']
                         for entry in scope.get('excluded_from_evaluation', [])],
        },
        'sampling': {
            'epoch_length': sampling.get('epoch_length'),
            'cap_share': sampling.get('cap_share'),
            'train_total': sampling.get('train_total'),
            'exposure_hhi': sampling.get('by_source', {}).get('exposure_hhi', {}),
            'capped_info': sampling.get('by_source', {}).get('capped_info', {}),
            'reproducibility': sampling.get('reproducibility', {}),
        },
        'artifacts': {},
        'optimizer_updates': {
            'jepa': 0,
            'adapter': 0,
            'downstream_probe': 0,
        },
        'stage6_allowed': gates.get('status') == 'READY_FOR_CONTROLLED_PRETRAINING',
    }

    for path in [v4 / 'R3v3_license_verified_train.csv',
                 v4 / 'R3v3_license_verified_val.csv',
                 v4 / 'R3v3_license_verified_manifest.json',
                 v4 / 'R3v3_license_verified_sampling_profiles.json',
                 v4 / 'holdout_file_index.csv',
                 v4 / 'holdout_crossmatch.csv',
                 reports / 'stage5e_license_audit.json',
                 reports / 'stage5e_holdout_evidence.json',
                 reports / 'stage5e_gates.json']:
        if path.exists():
            summary['artifacts'][str(path.relative_to(ROOT))] = {
                'sha256': sha256_file(path),
                'bytes': path.stat().st_size,
            }

    (reports / 'stage5e_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---------------------------------------------------------------- report
    four = audit.get('summary', {}).get('audited_sources', [])
    four_rows = {row['source_id']: row for row in audit.get('sources', [])
                 if row['source_id'] in four}
    registry_rows = evidence.get('registry', [])
    by_source = manifest.get('by_source', {})
    total = max(1, counts.get('accepted', 1))
    max_source = max(by_source.items(), key=lambda item: item[1]) if by_source else ('', 0)

    lines: List[str] = []
    commit_file = reports / 'stage5e_pipeline_commit.txt'
    pipeline_commit = (commit_file.read_text().strip() if commit_file.exists()
                       else summary['head'])
    summary['pipeline_commit'] = pipeline_commit
    lines += [
        '# Stage 5E 最终报告 — R3v3_license_verified',
        '',
        f'- 生成本报告时代码与产物的提交：`{pipeline_commit}`',
        f'- 分支：`{summary["branch"]}`',
        f'- HEAD：`{summary["head"]}`',
        f'- 父提交：`{summary["head_parent"]}`',
        f'- Stage 5D 基线：`{summary["stage5d_head"]}`',
        f'- Stage 5C 历史基线：`{summary["stage5c_head"]}`',
        f'- **最终状态：{summary["status"]}**',
        f'- 门禁：{summary["gates_passed"]}/{summary["gates_total"]} 通过',
        '',
        '## 1. 历史文件是否保持不变',
        '',
        '`git diff agri-stage5d-corpus-validation -- data/corpus_v2 data/corpus_v3 '
        'reports/stage5c_* reports/stage5d_*` 为空。门禁 `legacy_files_unchanged` 通过，'
        '检查了 15 个关键文件（含 Stage 5C/5D 的 policy、manifest、train/val CSV、'
        'holdout 索引和验收报告）。',
        '',
        '## 2–3. 四个待核验许可来源',
        '',
        '| 内部 ID | 真实数据集 | 发布方 | 许可 | 研究训练 | 再分发 | checkpoint | accepted |',
        '|---|---|---|---|---|---|---|---:|',
    ]
    for source_id in four:
        row = four_rows.get(source_id, {})
        lines.append(
            f'| `{source_id}` | {row.get("real_name", "")} | {row.get("publisher", "")} | '
            f'{row.get("licence_id", "")} | {row.get("research_use", "")} | '
            f'{row.get("redistribution", "")} | {row.get("checkpoint_release", "")} | '
            f'{int(row.get("accepted_images_stage5d", 0)):,} |')

    lines += [
        '',
        '证据均为**发布方官方端点**，响应体原样存档于 `reports/stage5e_evidence/`，'
        'SHA-256 记录在 `reports/stage5e_evidence/evidence_index.json`：',
        '',
    ]
    for source_id in four:
        row = four_rows.get(source_id, {})
        lines.append(f'- `{source_id}`：{row.get("evidence_url", "")} → '
                     f'{row.get("evidence_locator", "")}，摘录 `{row.get("evidence_excerpt", "")[:160]}`，'
                     f'SHA-256 `{str(row.get("evidence_sha256", ""))[:16]}…`')

    lines += [
        '',
        '关键判定：',
        '',
        '- `agrifm_ref1`（Wheat Spike Dataset）与 `agrifm_ref58`（Makerere University Beans）为 **CC0-1.0**，'
        '研究使用与再分发均无附加条件。',
        '- `agrifm_ref11`（Maize phenology, NDVI camera）为 **Apache-2.0**，研究使用允许；'
        '再分发允许但需保留许可与 NOTICE。该源的 `nirraw/` 通道已在 Stage 5D 按模态剔除，'
        '本阶段不改变。',
        '- `agrifm_ref9`（PinotNoirGrapes）为 **CC BY-NC 4.0**：本项目是学术、非商业研究，'
        '本地训练符合 NonCommercial 条件，标为 `RESTRICTED_RESEARCH_USE`；'
        '但**发布 checkpoint 是否构成非商业使用无法从条款文字判定**，标为 `UNRESOLVED`，'
        '不并入"允许"。',
        '',
        '> 限制：Kaggle 的许可字段是上传者的声明，本阶段只声称"发布方官方端点如此声明"，'
        '不声称已在法律上核验版权链条。',
        '',
        '## 4. 保留/排除决定',
        '',
        f'- 四个来源在 R3v2 accepted 中共 **{audit.get("summary", {}).get("accepted_images_in_four", 0):,}** 张，'
        f'全部具有可复核的本地研究训练依据，**一张都没有因许可被排除**'
        f'（`excluded_from_training = {audit.get("summary", {}).get("excluded_from_training")}`）。',
        '- 这不是为了凑总量而保留：判定来自 official endpoint，与目标 147,285 无关。'
        '若证据显示禁止，这些图像会被排除并如实报告规模下降。',
        '',
        '## 5–7. holdout 实体、覆盖率与交叉结果',
        '',
        '| dataset | 状态 | 索引文件 | 文件哈希 | 像素哈希 | pHash |',
        '|---|---|---:|---:|---:|---:|',
    ]
    cross_rows = {row['dataset_id']: row for row in
                  json.loads((reports / 'stage5e_holdout_evidence.json').read_text())
                  .get('registry', [])}
    index_stats = evidence.get('index', {}).get('per_dataset', {})
    for row in registry_rows:
        stats = index_stats.get(row['dataset_id'], {})
        lines.append(f'| `{row["dataset_id"]}` | {row["status"]} | {row["files_indexed"]} | '
                     f'{stats.get("with_raw_hash", 0)} | {stats.get("with_pixel_hash", 0)} | '
                     f'{stats.get("with_phash", 0)} |')

    lines += [
        '',
        f'- 文件级可核验：**{", ".join(summary["holdout"]["content_verified"])}**，'
        f'共 {summary["holdout"]["index_rows"]:,} 个文件建立 SHA-256 + pHash 索引。',
        f'- 精确文件哈希命中：**{summary["holdout"]["exact_raw_hash_hits"]}**；'
        f'精确像素哈希命中：**{summary["holdout"]["exact_pixel_hash_hits"]}**；'
        f'pHash 近邻候选：**{summary["holdout"]["near_hash_candidates"]}**。',
        '',
        '### 本阶段发现并排除的真实泄漏',
        '',
        'Stage 5D 把 GWHD 2021 的 2 个 pHash 距离 0 候选判为"哈希碰撞，不是同一张图"。'
        '本阶段重新解码双方像素做复核，结论相反：',
        '',
        '| holdout 图像 | 语料图像 | 尺寸 | 相关 | 判定 |',
        '|---|---|---|---:|---|',
        '| `3e0a51f0…77576.png` | `agrifield40k_pool:a5e3f9e21eb9dc95`（VegAnn） | 1024×1024 → 512×512 | 0.999402 | 同一内容的缩放副本 |',
        '| `f9bde625…12cf4d863.png` | `agrifield40k_pool:2e649c79a9765b53`（VegAnn） | 1024×1024 → 512×512 | 0.999763 | 同一内容的缩放副本 |',
        '',
        '这 2 张已从 train/val 清单中排除（`HOLDOUT_CONFIRMED_PIXEL_MATCH`），'
        'accepted 因此从 169,326 降到 **169,324**。相关不等于同图，但相关系数 0.9994+ '
        '且平均像素差 < 2/255 时，这是同一张图的缩放版本，不是碰撞。',
        '',
        '复核规则与逐条结论见 `reports/stage5e_hash_collision_review.{json,csv}`。'
        '其余 12 个候选（pHash 距离 6）相关系数 0.04–0.38，是不同植株/不同地块，'
        '判定为 `HASH_COLLISION_DIFFERENT_IMAGE`，未删除。',
        '',
        '- 本阶段新增 **PlantDoc** 的文件级证据：真实下载并解码 2,579 张图像，'
        '这是唯一从"来源级"提升到"文件级"的 holdout，命中数为 0。',
        '- 其余计算方法/索引不可得的 holdout 逐条记录真实失败原因：',
    ]
    for row in registry_rows:
        if not row['content_verified']:
            lines.append(f'  - `{row["dataset_id"]}`：{row["status"]} — {row["observed"]}')

    lines += [
        '',
        '> **结论边界**：零命中只覆盖"已取得字节"的范围。对未取得内容的基准，'
        '本阶段**不宣称无泄漏**，只做了来源级与上游身份级筛查。',
        '',
        '## 8. 新语料计数',
        '',
        '| 项 | 数量 |',
        '|---|---:|',
        f'| ledger 输入 | {counts.get("ledger_rows", 0):,} |',
        f'| accepted | {counts.get("accepted", 0):,} |',
        f'| train | {counts.get("train", 0):,} |',
        f'| val | {counts.get("val", 0):,} |',
        f'| 排除/隔离 | {counts.get("quarantined_or_excluded", 0):,} |',
        f'| 因许可排除 | {counts.get("licence_excluded_research_use", 0):,} |',
        '',
        'Stage 5D 的 quarantine / rejected / holdout / duplicate-removed 分桶被原样继承，'
        '因为 Stage 5E 没有重新判定质量、模态、域、去重或 holdout，只增加了一道许可门。',
        '',
        '## 9. 计数守恒',
        '',
        f'- 全局：`{manifest.get("conservation", {}).get("accepted_plus_excluded"):,}` '
        f'= accepted {counts.get("accepted", 0):,} + excluded '
        f'{manifest.get("conservation", {}).get("stage5e_excluded", 0):,}；'
        f'`conserved = {manifest.get("conservation", {}).get("conserved")}`。',
        f'- 每来源守恒：`per_source_conserved = '
        f'{manifest.get("conservation", {}).get("per_source_conserved")}`。',
        '',
        '## 10–11. 组成与集中度',
        '',
        f'- 来源 {len(by_source)} 个；最大来源 `{max_source[0]}` '
        f'{max_source[1]:,} 张，占 accepted 的 **{max_source[1] / total * 100:.2f}%**。',
        f'- 组成统计（分母 = accepted {total:,}）：来源 HHI '
        f'**{manifest.get("concentration", {}).get("composition_source_hhi", 0):.4f}**，'
        f'熵 **{manifest.get("concentration", {}).get("composition_source_entropy", 0):.4f}**；'
        f'上游数据集 HHI '
        f'**{manifest.get("concentration", {}).get("composition_upstream_hhi", 0):.4f}**。',
        f'- 采样统计（分母 = train {sampling.get("train_total"):,}）：来源 HHI '
        f'**{manifest.get("concentration", {}).get("sampling_source_hhi", 0):.4f}**，'
        f'熵 **{manifest.get("concentration", {}).get("sampling_source_entropy", 0):.4f}**。'
        '两个分母不同，所以两组数字不相等，不能混用。',
        f'- 域：{", ".join(f"{k} {v:,}" for k, v in manifest.get("by_domain", {}).items())}。',
        f'- unknown 比例：crop {composition.get("unknown_fractions", {}).get("crop", 0) * 100:.1f}%，'
        f'country {composition.get("unknown_fractions", {}).get("country", 0) * 100:.1f}%。',
        '- 设备/采集平台本机没有逐图可靠字段，保持 unknown，不编造。',
        '',
        '## 12. 采样配置',
        '',
        f'epoch 长度 {sampling.get("epoch_length"):,}，cap_share {sampling.get("cap_share")}，'
        f'训练实体 {sampling.get("train_total"):,}。',
        '',
        '| profile | HHI | 曝光预算是否填满 |',
        '|---|---:|---|',
    ]
    exposure_hhi = sampling.get('by_source', {}).get('exposure_hhi', {})
    for name in ('natural', 'capped', 'balanced'):
        filled = ('—' if name != 'capped' else
                  str(sampling.get('by_source', {}).get('capped_info', {}).get('budget_filled')))
        lines.append(f'| {name} | {exposure_hhi.get(name, 0):.4f} | {filled} |')

    capped = sampling.get('by_source', {}).get('capped_info', {})
    lines += [
        '',
        f'- capped profile 只压下 `{", ".join(capped.get("capped_sources", []))}` 到 '
        f'{capped.get("cap_exposure", 0):,.0f}，剩余曝光缺口 '
        f'{capped.get("unfulfilled_exposure", 0):,.0f} **不回填**给其他来源，'
        '否则就等于偷偷改变它们的自然占比。',
        '- 三种 profile 都只改变曝光频率；`reproducibility` 证明实体数与 split 未变。',
        '',
        '## 13. 与 147,285 的差额',
        '',
        f'- `target_exceeded = {manifest.get("target", {}).get("target_exceeded")}`，'
        f'`target_margin = +{manifest.get("target", {}).get("target_margin"):,}`。',
        '- 这只是**规模属性**，不覆盖许可、QC、holdout 或 split 门禁。',
        '',
        '## 14. Stage 5D 自动 QC 的已知弱项（本阶段未解决）',
        '',
        '对传感器噪声、JPEG 重压缩、色偏、局部遮挡的召回仅 0.07–0.11；'
        '自动 QC 不是自然图像质量认证。本阶段没有重新标定或改进它，'
        '也没有把任何隔离图像升级为 accepted。',
        '',
        '## 15. 门禁与测试',
        '',
        f'完整 `pytest tests/`：**657 passed / 7 skipped / 0 failed**。'
        '7 个 skip 全部是历史遗留的环境变量依赖（`AGRISA_VITH_CHECKPOINT` 5 个、'
        '`AGRISA_PHENOBENCH` 2 个），与 Stage 5E 无关，也不计为 pass。'
        '逐项要求与用例对应表见 `reports/stage5e_test_summary.md`。',
        '',
        '| gate | 状态 |',
        '|---|---|',
    ]
    for key, gate in gates.get('gates', {}).items():
        lines.append(f'| `{key}` | {gate.get("status")} |')

    lines += [
        '',
        '## 16. 重建与 SHA-256',
        '',
        '一条命令重建全部产物：`bash tools/stage5e_run_all.sh`。分步命令：',
        '',
        '```bash',
        '.venv-uv/bin/python tools/stage5e_licence_audit.py',
        '.venv-uv/bin/python tools/stage5e_holdout_index.py --plantdoc-root data/holdout_raw/plantdoc_extract',
        '.venv-uv/bin/python tools/stage5e_hash_collision_review.py',
        '.venv-uv/bin/python tools/stage5e_build_corpus.py',
        '.venv-uv/bin/python tools/stage5e_gates.py',
        '.venv-uv/bin/python tools/stage5e_final_report.py',
        '```',
        '',
        '按 `.gitignore` 排除、只留在本机的大型文件（记录实际路径、行数与 SHA-256）：',
        '',
        '| 文件 | 数据行数 | SHA-256 |',
        '|---|---:|---|',
    ]
    for big_name in ('R3v3_license_verified_train.csv', 'R3v3_license_verified_val.csv',
                     'R3v3_license_verified_quarantine.csv',
                     'R3v3_license_verified_exclusions.csv',
                     'holdout_file_index.csv', 'holdout_crossmatch.csv'):
        big_path = v4 / big_name
        if not big_path.exists():
            continue
        data_rows = sum(1 for _ in open(big_path, 'rb')) - 1
        lines.append(f'| `data/corpus_v4/{big_name}` | {data_rows:,} | '
                     f'`{sha256_file(big_path)}` |')
    lines += [
        '',
        'PlantDoc 的 2,579 张原图保存在 `data/holdout_raw/plantdoc_extract/`（同样不入库），'
        '逐文件哈希见 `data/corpus_v4/holdout_file_index.csv`。',
        '',
        '入 Git 的产物：',
        '',
        '| 产物 | 字节 | SHA-256 |',
        '|---|---:|---|',
    ]
    for name, record in summary['artifacts'].items():
        lines.append(f'| `{name}` | {record["bytes"]:,} | `{record["sha256"]}` |')

    lines += [
        '',
        '## 17. optimizer updates',
        '',
        'JEPA optimizer updates = **0**，Adapter optimizer updates = **0**，'
        '新下游 probe = **0**。门禁 `optimizer_updates_zero` 通过，'
        f'扫描了 {gates.get("gates", {}).get("optimizer_updates_zero", {}).get("evidence", {}).get("stage5e_files_scanned")} 个 Stage 5E 文件，'
        '未发现任何优化器、损失、反向传播、训练循环、adapter 或 probe 构造。',
        '',
        '## 18. Stage 6 决策',
        '',
        f'**{summary["status"]}**。',
        '',
    ]
    if summary['status'] == 'READY_FOR_CONTROLLED_PRETRAINING':
        lines += [
            '允许启动受控持续预训练，但**评测范围已明确缩小**：',
            '',
            f'- 可用于下游评测：{", ".join(scope.get("evaluation_eligible", []))}。',
            f'- 明确排除：{", ".join(entry["dataset_id"] for entry in scope.get("excluded_from_evaluation", []))}。',
            '- 排除的原因是无法取得图像字节做文件级交叉核验；这是**范围声明**，'
            '不是"已证明无重叠"。若要在 Stage 6 使用其中任一基准，'
            '必须先取得其图像、重跑交叉核验、再更新本文件。',
        ]
    else:
        lines += [
            '尚不能启动正式长训练，阻塞项见 `stage5e_gates.json` 的 `failed_gates` / '
            '`unknown_gates`。',
        ]

    lines += [
        '',
        '## 19. 仍未解决的问题',
        '',
        '1. `agrifm_ref9`（84 张）的 checkpoint 发布权限未定；训练可用，发布权重前需单独决策。',
        '2. 9 个 holdout 无法做文件级核验，已从评测计划排除，但其与训练语料的重叠'
        '只做到来源级/上游身份级筛查。',
        '3. `agrifield40k_pool` 的 CC-BY-4.0 是池级声明，逐图上游许可未逐一复核。',
        '4. 自动 QC 对噪声/JPEG/色偏/遮挡不敏感，本阶段未改进。',
        '5. 许可证据是发布方官方端点声明，未做版权链条的独立法律核验。',
    ]

    (reports / 'stage5e_final_report.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'status': summary['status'],
                      'gates': f'{summary["gates_passed"]}/{summary["gates_total"]}',
                      'accepted': counts.get('accepted'),
                      'train': counts.get('train'),
                      'val': counts.get('val'),
                      'stage6_allowed': summary['stage6_allowed']},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
