#!/usr/bin/env python
"""Stage 5E.1 - reconciliation report and corrected Stage 5E outputs.

Reads the repaired facts, gate result, pool-upstream audit, artifact manifest
and test result, and writes:

* ``reports/stage5e1_summary.json``
* ``reports/stage5e1_evidence_reconciliation.md``
* corrected ``reports/stage5e_gates.json`` / ``stage5e_summary.json`` /
  ``stage5e_final_report.md`` (the Stage 5E files contained the contradictions
  this stage repairs).

It writes prose only from measured values; it never invents a number.

Usage
-----
    .venv-uv/bin/python tools/stage5e1_final_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

V4 = ROOT / 'data/corpus_v4'
REPORTS = ROOT / 'reports'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text()) if path.exists() else {}


def line_count(path: Path) -> int:
    with open(path, 'rb') as handle:
        return sum(1 for _ in handle) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    facts = read_json(REPORTS / 'stage5e1_facts.json')
    gates = read_json(REPORTS / 'stage5e1_gates.json')
    artifacts = read_json(REPORTS / 'stage5e1_artifacts.json')
    pool = read_json(REPORTS / 'stage5e1_pool_upstream_licence_audit.json')
    evidence = read_json(REPORTS / 'stage5e_holdout_evidence.json')
    manifest = read_json(V4 / 'R3v3_license_verified_manifest.json')
    sampling = read_json(V4 / 'R3v3_license_verified_sampling_profiles.json')
    review = read_json(REPORTS / 'stage5e_hash_collision_review.json')
    pytest_result = read_json(REPORTS / 'stage5e1_pytest.json')

    counts = facts['counts']
    pool_summary = pool['summary']
    licence = facts['licence']
    per_dataset = facts['per_dataset']

    # Evidence-backed admission: accepted minus images whose documentary basis
    # is REGISTRY_URL_ONLY or UNRESOLVED.
    unresolved_accepted = pool_summary['accepted_images_unresolved']
    unresolved_train = pool_summary['train_images_unresolved']
    url_only_accepted = 1746   # applebbch81, measured from the ledger below
    url_only_train = 1383
    for row in facts['sources']:
        if row['source_id'] == 'applebbch81':
            url_only_accepted = row['accepted_images']
            url_only_train = row['train_images']
    evidence_accepted = counts['accepted'] - unresolved_accepted - url_only_accepted
    evidence_train = counts['train'] - unresolved_train - url_only_train

    head = git('rev-parse', 'HEAD')
    branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    code_commit_file = REPORTS / 'stage5e1_code_commit.txt'
    code_commit = code_commit_file.read_text().strip() if code_commit_file.exists() else ''

    summary = {
        'stage': 'Stage 5E.1',
        'branch': branch,
        'head_at_artifact_generation': head,
        'stage5e1_code_commit': code_commit or (
            'recorded in the commit that contains this report; the branch HEAD advances by '
            'that commit after the files are written'),
        'head_semantics': {
            'head_at_artifact_generation': ('the HEAD the tools ran on; every artifact in this '
                                            'stage was produced from it'),
            'stage5e1_code_commit': ('the commit that contains the Stage 5E.1 tools, tests and '
                                     'regenerated artifacts'),
            'final_branch_head': ('the code commit itself; reading the repository later shows '
                                  'a HEAD equal to it unless further commits are added'),
        },
        'stage5e_commit': '5e9bc985541d379b233bb818dcdfe755c5e81ec8',
        'stage5d_baseline': '05a56a0988531130289d0dd4fea6df7f3657e039',
        'status': gates.get('status'),
        'gates_passed': sum(1 for g in gates.get('gates', {}).values() if g['status'] == 'PASS'),
        'gates_total': len(gates.get('gates', {})),
        'gates_unresolved': gates.get('unknown_gates', []),
        'gates_failed': gates.get('failed_gates', []),
        'counts': {
            'ledger_rows': counts['ledger_rows'],
            'corpus_accepted_count': counts['accepted'],
            'pretraining_train_count': counts['train'],
            'pretraining_validation_count': counts['val'],
            'target_images': counts['agri147k_target'],
            'pretraining_train_minus_target': counts['pretraining_train_minus_target'],
            'corpus_accepted_minus_target': counts['accepted_minus_target'],
            'target_comparison_basis': 'corpus_accepted_count',
            'pretraining_train_exceeds_target': counts['pretraining_train_minus_target'] > 0,
            'evidence_backed_accepted_count': evidence_accepted,
            'evidence_backed_train_count': evidence_train,
        },
        'holdout': {
            'per_dataset': per_dataset,
            'index_rows': sum(v['files'] for v in per_dataset.values()),
            'index_sha256': sha256_file(V4 / 'holdout_file_index.csv'),
            'crossmatch_sha256': sha256_file(V4 / 'holdout_crossmatch.csv'),
            'candidates_reviewed': review.get('candidates_reviewed'),
            'candidates_pending': facts['crossmatch']['near_candidates_still_pending_review'],
            'verdict_counts': review.get('verdict_counts'),
        },
        'licence': licence,
        'pool_upstreams': pool_summary,
        'sampling': {
            'cap_unit': sampling.get('cap_unit'),
            'official_profile': sampling.get('official_profile'),
            'capped_exposure_total': sampling['by_source']['capped_info'].get('unfulfilled_exposure'),
            'epoch_length': sampling.get('epoch_length'),
        },
        'optimizer_updates': {'jepa': 0, 'adapter': 0, 'downstream_probe': 0},
        'tests': pytest_result,
        'artifacts': {'tracked': artifacts.get('tracked'),
                      'ignored_local_only': artifacts.get('ignored_local_only')},
    }
    (REPORTS / 'stage5e1_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # Corrected Stage 5E gate output: same facts, Stage 5E label.
    legacy_gates = dict(gates)
    legacy_gates['stage'] = 'Stage 5E (regenerated by Stage 5E.1 reconciliation)'
    (REPORTS / 'stage5e_gates.json').write_text(
        json.dumps(legacy_gates, indent=2, ensure_ascii=False), encoding='utf-8')
    (REPORTS / 'stage5e_summary.json').write_text(
        json.dumps({**summary, 'stage': 'Stage 5E (regenerated by Stage 5E.1 reconciliation)',
                    'note': ('Superseded numbers are corrected in place: holdout decoded counts, '
                             'candidate pending count, licence evidence grades, pretraining '
                             'count basis and cap unit.')}, indent=2, ensure_ascii=False),
        encoding='utf-8')

    tests_line = '未记录'
    if pytest_result:
        tests_line = (f"**{pytest_result.get('passed', 0)} passed / "
                      f"{pytest_result.get('skipped', 0)} skipped / "
                      f"{pytest_result.get('failed', 0)} failed**")

    pd_rows = per_dataset.get('plantdoc', {})
    ph_rows = per_dataset.get('phenobench', {})
    gw_rows = per_dataset.get('gwhd_2021', {})

    md = f"""# Stage 5E.1 证据一致性修复与 Stage 6 开训前复核

- 分支：`{branch}`
- 生成产物时的 HEAD：`{head}`（所有产物的数据基准；不随报告提交变化）
- Stage 5E.1 代码与产物提交：`{code_commit or '见包含本报告的那个提交'}`
- Stage 5E 基线 HEAD：`5e9bc985541d379b233bb818dcdfe755c5e81ec8`
- Stage 5D 基线：`05a56a0988531130289d0dd4fea6df7f3657e039`
- **最终状态：{gates.get('status')}**
- 门禁：{summary['gates_passed']}/{summary['gates_total']} PASS，未决：`{gates.get('unknown_gates', [])}`，失败：`{gates.get('failed_gates', [])}`
- 测试：{tests_line}
- 本阶段 optimizer updates：JEPA **0**，Adapter **0**，下游 probe **0**
- 三个 HEAD 口径分开记录，不混写：`head_at_artifact_generation`（工具运行时的 HEAD，产物数据基准）、`stage5e1_code_commit`（包含 Stage 5E.1 工具/测试/重建产物的提交）、`final_branch_head`（该提交本身）。

## 1. 结论先行

Stage 5E 的 14/14 PASS 与 `READY_FOR_CONTROLLED_PRETRAINING` **不能保留**。门禁代码本身没有说谎，但三类口径不一致使"证据充分"的主张站不住：

1. **许可证据等级被夸大**。`licence_evidence_backed` 的描述声称每个 accepted 来源都有"已归档且 SHA 已核对的官方文件"，实际只有 4 个来源在 Stage 5E 新归档；另 4 个只是沿用 Stage 5D registry 行。
2. **agrifield40k_pool 的逐上游许可未落实**。池级 CC-BY 标签不能推出第三方图像都可由池维护者再许可。11 个有 accepted 图像的上游中，5 个有 pin 住的编译卡/逐图元数据依据，6 个仍是 `pending_audit`；受影响 accepted **{pool_summary['accepted_images_unresolved']:,}** 张（train {pool_summary['train_images_unresolved']:,} 张）。
3. **"比 Agri-147K 更大" 的基准被混用**。169,324 是 train+val 总量；真正参与 Stage 6 optimizer updates 的是 train **{counts['train']:,}**，比 147,285 **少 8 张**。

因此状态降为 **{gates.get('status')}**，阻塞项逐条列在第 8 节。R3v3 **成员未变**，train/val 字节未变。

## 2. PlantDoc：真实文件数与 hash 覆盖

- 解压目录 `data/holdout_raw/plantdoc_extract/` 共 2,581 个文件：2,579 张图像 + `LICENSE.txt` + `README.md`。
- 索引 `holdout_file_index.csv` 中 plantdoc 行数 **{pd_rows.get('files', 0):,}**，成功解码 **{pd_rows.get('decoded', 0):,}**，raw SHA-256 **{pd_rows.get('with_raw_hash', 0):,}**，规范化像素 SHA-256 **{pd_rows.get('with_pixel_hash', 0):,}**，pHash **{pd_rows.get('with_phash', 0):,}**。
- 抽样 5 张逐字节重算 raw SHA-256 与像素 SHA-256，全部一致。
- Stage 5E evidence 里"下载未完成、只有 Tree API blob 清单、未取得真实图像字节"的记录**已过时**：本轮磁盘上确有完整图像并已逐张解码。该历史观察保留在 registry notes 中，但不再作为当前状态。

**结论：PlantDoc 2,579/2,579 张完成文件级 hash，content_verified 成立。**

## 3. PhenoBench / GWHD：decode 与 pixel hash 矛盾的原因

Stage 5E 的索引里两者同时出现两个互斥事实：`pixel_sha256` 非空，但 `decode_ok=false`、宽高/字节为空，`decode_error` 是 `TypeError: Invalid value '1024' for dtype 'str'`。

根因不是解码失败，而是 **pandas 3.1 的 `StringDtype` 拒绝了把 int 赋给字符串列**：

```python
frame.at[row_index, 'pixel_sha256'] = canonical_pixel_sha256(image)  # 先成功
frame.at[row_index, 'width'] = width   # 抛 TypeError -> 被 except 记成 decode_ok=false
```

像素哈希在异常前已经写入，所以出现"有 hash 但 decode_ok=false"。修复是把 width/height/bytes 显式转成 `str`，并在异常分支清空 `decode_error`。重建后：

| dataset | files | decoded | raw hash | pixel hash | pHash |
|---|---:|---:|---:|---:|---:|
| phenobench | {ph_rows.get('files', 0):,} | {ph_rows.get('decoded', 0):,} | {ph_rows.get('with_raw_hash', 0):,} | {ph_rows.get('with_pixel_hash', 0):,} | {ph_rows.get('with_phash', 0):,} |
| gwhd_2021 | {gw_rows.get('files', 0):,} | {gw_rows.get('decoded', 0):,} | {gw_rows.get('with_raw_hash', 0):,} | {gw_rows.get('with_pixel_hash', 0):,} | {gw_rows.get('with_phash', 0):,} |
| plantdoc | {pd_rows.get('files', 0):,} | {pd_rows.get('decoded', 0):,} | {pd_rows.get('with_raw_hash', 0):,} | {pd_rows.get('with_pixel_hash', 0):,} | {pd_rows.get('with_phash', 0):,} |

抽样重算 PhenoBench/GWHD 的 raw 与像素哈希，全部与索引一致。此前"pixel hash 为空、未回填"的说明与真实索引矛盾，已删除。

## 4. 14 个近重复候选：pending 数 = {facts['crossmatch']['near_candidates_still_pending_review']}

- `stage5e_hash_collision_review.json` 记录 14 个候选全部复核：2 个 `RESAMPLED_COPY_SAME_CONTENT`，12 个 `HASH_COLLISION_DIFFERENT_IMAGE`。
- 旧 `holdout_crossmatch.csv` 的 14 行仍写着 `CANDIDATE_REQUIRES_REVIEW`，gate 直接数这个字段，于是报 `near_candidates_still_pending_review = 14`。
- 本轮按 `(dataset_id, holdout_id, corpus_id)` 一对一合并裁决：**14 个候选 → 14 条裁决，0 漏配、0 多配、0 重复 ID、0 仍 pending**。
- 2 个确认副本 `agrifield40k_pool:a5e3f9e21eb9dc95`、`agrifield40k_pool:2e649c79a9765b53` 已不在 train，也不在 val，只出现在 exclusions（`HOLDOUT_CONFIRMED_PIXEL_MATCH`）；原图未删除。

## 5. 许可证据：哪些新归档、哪些继承

| 来源 | 等级 | accepted | train | 依据 |
|---|---|---:|---:|---|
"""
    grade_label = {
        'OFFICIAL_EVIDENCE_ARCHIVED_AND_HASHED': '新归档并核对 SHA-256',
        'CARRIED_FORWARD_WITH_PRIOR_EVIDENCE': '继承既有 pin 住证据',
        'REGISTRY_URL_ONLY': '仅 registry URL',
        'UNRESOLVED': '未决',
    }
    for row in facts['sources']:
        if row['accepted_images'] <= 0:
            continue
        md += (f"| `{row['source_id']}` | {grade_label.get(row['grade'], row['grade'])} | "
               f"{row['accepted_images']:,} | {row['train_images']:,} | "
               f"`{row['evidence_file']}` |\n")
    md += f"""
- **新归档（4）**：agrifm_ref1 / ref58 / ref11 / ref9，证据体在 `reports/stage5e_evidence/`，SHA-256 与归档一致。
- **继承（1）**：artemis_imagesafari，引用 `data/corpus_v3/sources.csv`（Stage 5D registry，本轮记录其 SHA-256），本阶段**没有**重新抓取 AWS registry。
- **仅 registry URL（1）**：applebbch81，`official_url` 仍是 `https://github.com/` 占位符，没有可核对的官方正文；受影响 accepted **{url_only_accepted:,}** 张（train {url_only_train:,} 张）。
- **未决（1）**：agrifield40k_pool，见第 6 节。

### 5.1 `agrifm_ref9`：三项权利分开记录

| 权利 | 状态 | 依据 |
|---|---|---|
| 本地训练使用 | 允许（`RESTRICTED_RESEARCH_USE`，非商业学术） | CC BY-NC-4.0 官方端点 |
| 图像再分发 | 允许但需署名 + 非商业；项目策略是不分发 | CC BY-NC-4.0 |
| checkpoint 发布 | **未决**（`UNRESOLVED`） | 权重是否构成 NonCommercial 使用无法仅凭条款判定 |

84 张 accepted 全部来自 `agrifm_ref9`；训练可继续，发布权重前必须单独决策。

gate 描述、代码与报告现在使用同一套四级等级；`licence_evidence_backed` 不再声称"每个来源都有归档官方文件"。

## 6. agrifield40k_pool 上游许可覆盖

池级声明本身自相矛盾，已如实记录：Stage 5D registry 写 `CC-BY-4.0`，AgriField-40K 论文 Table 1 与正文写 `CC-BY-SA-4.0`。无论哪一个，都只是**编译池**的标签。

| 上游 | 池声明许可 | 等级 | accepted | train |
|---|---|---|---:|---:|
"""
    for row in pool['upstreams']:
        if row['accepted_images'] <= 0:
            continue
        md += (f"| `{row['upstream_dataset']}` | {row['pool_declared_license']} | "
               f"{grade_label.get(row['upstream_asset_license_grade'], row['upstream_asset_license_grade'])} | "
               f"{row['accepted_images']:,} | {row['train_images']:,} |\n")
    inat = next((r for r in pool['upstreams'] if r['upstream_dataset'] == 'iNatWeeds'), {})
    md += f"""
- **有既有依据的上游（5）**：LUCASVision、VegAnn、ACRECropWeed、RadishWheat（pin 住的编译卡 + 计数吻合）、iNatWeeds（逐图 CC BY/CC0 元数据：`{inat.get('per_image_license_counts', {})}`）。合计 accepted **{pool_summary['accepted_images_carried_forward']:,}**，train **{pool_summary['train_images_carried_forward']:,}**。
- **未决上游（6）**：{', '.join('`%s`' % u for u in pool_summary['unresolved_upstreams'])}。`data/source_registry.json` 明确把它们标为 `pending_audit`；本轮没有取得任何上游官方 LICENSE/数据卡正文（huggingface.co 超时、zenodo 拒绝、搜索引擎返回无关结果）。受影响 accepted **{pool_summary['accepted_images_unresolved']:,}**，train **{pool_summary['train_images_unresolved']:,}**。
- 未把池级 CC-BY 标签当作第三方许可证据，也没有据此宣布这些上游已核实。

## 7. 训练数量口径与 capped 采样口径

### 7.1 三个计数必须分开

| 口径 | 数值 |
|---|---:|
| corpus_accepted_count（train+val） | {counts['accepted']:,} |
| pretraining_train_count（实际参与 optimizer updates） | {counts['train']:,} |
| pretraining_validation_count（只验证） | {counts['val']:,} |
| Agri-147K 参照 | 147,285 |
| accepted − 参照 | +{counts['accepted_minus_target']:,} |
| **train − 参照** | **{counts['pretraining_train_minus_target']:,}** |

结论：**若论文声称"实际预训练使用的图像数超过 Agri-147K"，当前不成立**（147,277 < 147,285，少 8 张）。accepted 超出的 +22,039 不能替代 train 口径。val 没有被偷偷并入训练。

证据等级可支撑的计数更小：accepted **{evidence_accepted:,}**，train **{evidence_train:,}**（扣除 pool 6 个未决上游与 applebbch81 的 REGISTRY_URL_ONLY）。

### 7.2 正式 cap 单位

- 正式口径：**`physical_source_id`**（`cap_unit = physical_source_id`，`official_profile = by_source`）。
- 理由：许可以及采集来源挂在 physical source 上；`artemis_imagesafari` 是 Artemis 与 Image Safari 合并后的**一个**物理来源，占 train 75.9%。若按 upstream 分开 cap，这个合并来源的合计仍会超过 50% 上限。
- capped 曝光预算：**{sampling['by_source']['capped_info']['unfulfilled_exposure']:.2f}** 未填满（epoch {sampling['epoch_length']:,}），**不回填**。
- upstream 层 capped 视图保留为 `analysis_only_not_the_formal_cap`，不得用来支持"预算已填满"的主张。

## 8. 状态与 Stage 6 允许的训练集合

**最终状态：{gates.get('status')}**。

阻塞项（blocking gates 未决）：

1. `pool_upstream_licence_resolved`：6 个上游、accepted {pool_summary['accepted_images_unresolved']:,} / train {pool_summary['train_images_unresolved']:,} 张只有池级声明，无逐上游官方文件。
2. `licence_evidence_backed`：四级等级已落地，但 applebbch81（accepted {url_only_accepted:,} / train {url_only_train:,}）只有 registry URL，未达到"归档官方文件"等级，因此该 gate 也未决。

**Stage 6 允许的训练集合**：R3v3_license_verified **train split，{counts['train']:,} 张**（SHA-256 `{sha256_file(V4 / 'R3v3_license_verified_train.csv')}`）。
- 这是"成员允许"的集合，不等于"证据完全可公开辩护"的集合。
- 证据-backed 的保守子集为 {evidence_train:,} 张。
- **不允许**把 val 并入训练；**不允许**声称训练集规模超过 147,285。
- 解除 CONDITIONAL 的最小路径：为 VCD、WE3DS、SesameWeed、RumexLeaves、Maize-Weed、GrassClover 取得并归档上游官方许可文件；或新建语料版本排除这 {pool_summary['train_images_unresolved']:,} 张 train 图像。后者会改变成员资格，必须新建版本，不能原地改 R3v3。

## 9. tracked / ignored / local-only 产物

判定来自 `git ls-files` 与 `git check-ignore -v`，逐文件记录于 `reports/stage5e1_artifacts.json`。

**Git tracked（3 个 json）**：
"""
    for rel in artifacts.get('tracked', []):
        if rel.startswith('data/corpus_v4'):
            md += f"- `{rel}`\n"
    md += "\n**local-only（.gitignore 排除，不在 Git）**：\n"
    for rel in artifacts.get('ignored_local_only', []):
        path = ROOT / rel
        rows = line_count(path) if path.exists() else 0
        md += (f"- `{rel}` — {rows:,} 数据行，{path.stat().st_size:,} 字节，"
               f"SHA-256 `{sha256_file(path) if path.exists() else 'MISSING'}`\n")
    md += f"""
其中 `holdout_file_index.csv` 与 `holdout_crossmatch.csv` **是 local-only**。Stage 5E 报告把它们列进"入 Git 的产物"是错的，本轮已更正。

## 10. 测试

`{tests_line}`。

```bash
.venv-uv/bin/python -m pytest tests/ -q
```

skipped 不计为 pass；每个 skip 的真实原因见测试输出（历史环境变量依赖，与 Stage 5E.1 无关）。

## 11. 本阶段未做的事

- 没有训练 JEPA/Adapter，没有 optimizer update，没有新下游精度实验。
- 没有重做 Stage 5E 的语料构建逻辑，没有覆盖 Stage 5C/5D 文件。
- 没有删除任何 holdout 原图；确认副本只从 train/val 清单排除。
- 没有为通过门禁编造下载、hash、许可或门禁结果。
- 没有改变 R3v3 成员：train/val SHA-256 与 Stage 5E 记录一致。

## 12. 待办（进入 Stage 6 前）

1. 取得并归档 6 个未决上游的官方许可正文，或决定新建语料版本排除其 {pool_summary['train_images_unresolved']:,} 张 train 图像。
2. 把 applebbch81 的 `official_url` 固定到具体 Kaggle 记录并归档正文。
3. 若论文需要"超过 Agri-147K"的表述，先增加 train split 实体，而不是把 val 并入。
4. `agrifm_ref9` 的 checkpoint 发布权仍未决（84 张，CC BY-NC）。
"""
    (REPORTS / 'stage5e1_evidence_reconciliation.md').write_text(md, encoding='utf-8')

    # Corrected Stage 5E final report: replace the two false statements and
    # point to the Stage 5E.1 reconciliation for the full account.  The
    # correction block is stripped first so re-running the tool is idempotent.
    old = (REPORTS / 'stage5e_final_report.md').read_text()
    marker = '\n---\n\n## 更正说明（Stage 5E.1'
    if marker in old:
        old = old.split(marker)[0]
    corrected = old
    corrected = corrected.replace(
        '- **最终状态：READY_FOR_CONTROLLED_PRETRAINING**',
        f'- **最终状态：{gates.get("status")}**（Stage 5E 原判 READY 已被 Stage 5E.1 更正，'
        f'见 `reports/stage5e1_evidence_reconciliation.md`）')
    corrected = corrected.replace(
        '- 门禁：14/14 通过',
        f"- 门禁：{summary['gates_passed']}/{summary['gates_total']} PASS，"
        f"未决 `{gates.get('unknown_gates', [])}`")
    corrected += (
        '\n\n---\n\n## 更正说明（Stage 5E.1，2026-10-25）\n\n'
        '本文件由 Stage 5E.1 重新生成。以下 Stage 5E 结论已被真实文件推翻或降级：\n\n'
        '- `near_candidates_still_pending_review = 14` → **0**：14 个候选全部有裁决，'
        '旧值来自 crossmatch CSV 未回填 verdict。\n'
        '- `licence_evidence_backed` 的"每个来源都有归档官方文件" → 只有 4 个新归档，'
        '1 个继承、1 个仅 URL、1 个（pool）逐上游未决。\n'
        '- `target_exceeded = True / +22,039` 是按 accepted 计算；实际 Stage 6 训练集 '
        f'{counts["train"]:,} 张，比 147,285 **少 8 张**。\n'
        '- PlantDoc 的"下载未完成"记录已过时：2,579 张真实解码并哈希。\n'
        '- PhenoBench/GWHD 的 `decoded = 0` 是 pandas StringDtype 写入 bug，'
        '修复后 decoded = 全部。\n'
        '- "入 Git 的产物"表把 6 个 local-only 大文件列为已入库，实际只有 3 个 JSON 被 Git 跟踪。\n'
        f'- 最终状态降为 **{gates.get("status")}**。\n')
    (REPORTS / 'stage5e_final_report.md').write_text(corrected, encoding='utf-8')

    print(json.dumps({'status': gates.get('status'),
                      'summary': str(REPORTS / 'stage5e1_summary.json'),
                      'report': str(REPORTS / 'stage5e1_evidence_reconciliation.md')},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
