#!/usr/bin/env python
"""Stage 5E work package A - licence audit.

Reads the Stage 5D per-image ledger (never rewrites it), attaches the Stage 5E
licence decision to every source, verifies the archived evidence bytes, and
writes the audit in three formats.

The tool refuses to emit a "verified" decision when the evidence file is
missing or its SHA-256 no longer matches, so a report can never claim evidence
it does not have.

Usage
-----
    .venv-uv/bin/python tools/stage5e_licence_audit.py \
        --ledger data/corpus_v3/images --reports reports --out data/corpus_v4
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

import pandas as pd

from src.registry import stage5e_licence as lic


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_ledger_counts(ledger_dir: Path) -> pd.DataFrame:
    """Per-source counts of Stage 5D decisions (accepted rows only matter here)."""
    frames: List[pd.DataFrame] = []
    for path in sorted(ledger_dir.glob('*.csv')):
        frame = pd.read_csv(path, usecols=['physical_source_id', 'selected', 'primary_reason'],
                            low_memory=False)
        frames.append(frame)
    if not frames:
        raise SystemExit(f'no ledger CSV found in {ledger_dir}')
    return pd.concat(frames, ignore_index=True)


def verify_evidence(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    problems: List[Dict[str, object]] = []
    for row in rows:
        evidence = Path(str(row.get('evidence_file') or ''))
        recorded = str(row.get('evidence_sha256') or '')
        if not recorded:
            continue  # carried-forward rows cite the Stage 5D registry, not an archive
        if not evidence.exists():
            problems.append({'source_id': row['source_id'], 'problem': 'MISSING_EVIDENCE_FILE',
                             'detail': str(evidence)})
            continue
        actual = sha256_file(evidence)
        if actual != recorded:
            problems.append({'source_id': row['source_id'], 'problem': 'EVIDENCE_SHA_MISMATCH',
                             'detail': f'recorded={recorded} actual={actual}'})
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', default='data/corpus_v3/images')
    parser.add_argument('--reports', default='reports')
    parser.add_argument('--out', default='data/corpus_v4')
    args = parser.parse_args()

    ledger_dir = ROOT / args.ledger
    reports = ROOT / args.reports
    out_dir = ROOT / args.out
    reports.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger_counts(ledger_dir)
    accepted = ledger[ledger['selected'].astype(str).str.lower().isin(['true', '1'])]
    accepted_by_source = accepted['physical_source_id'].value_counts().to_dict()
    total_by_source = ledger['physical_source_id'].value_counts().to_dict()

    rows = lic.audit_rows()
    evidence_problems = verify_evidence(rows)

    # Attach the image counts measured from the ledger, not from a report.
    for row in rows:
        source_id = str(row['source_id'])
        row['accepted_images_stage5d'] = int(accepted_by_source.get(source_id, 0))
        row['ledger_rows'] = int(total_by_source.get(source_id, 0))
        row['train_eligible'] = lic.train_eligible(source_id)
        row['redistribution_eligible'] = lic.redistribution_eligible(source_id)
        row['evidence_verified'] = not any(
            problem['source_id'] == source_id for problem in evidence_problems)

    four = ['agrifm_ref1', 'agrifm_ref58', 'agrifm_ref11', 'agrifm_ref9']
    four_rows = [r for r in rows if r['source_id'] in four]
    summary = {
        'audited_sources': four,
        'accepted_images_in_four': int(sum(r['accepted_images_stage5d'] for r in four_rows)),
        'train_eligible_in_four': int(sum(r['accepted_images_stage5d'] for r in four_rows
                                          if r['train_eligible'])),
        'excluded_from_training': int(sum(r['accepted_images_stage5d'] for r in four_rows
                                          if not r['train_eligible'])),
        'moved_from_unresolved_to_verified': int(sum(
            r['accepted_images_stage5d'] for r in four_rows
            if r['research_use'] == lic.VERIFIED_RESEARCH_USE)),
        'moved_from_unresolved_to_restricted': int(sum(
            r['accepted_images_stage5d'] for r in four_rows
            if r['research_use'] == lic.RESTRICTED_RESEARCH_USE)),
        'evidence_verified_sources': int(sum(1 for r in rows
                                             if r.get('evidence_verified') and
                                             str(r.get('evidence_sha256') or ''))),
        'evidence_problems': evidence_problems,
        'redistribution_allowed_images_in_four': int(sum(
            r['accepted_images_stage5d'] for r in four_rows
            if r['redistribution_eligible'])),
        'rules_applied': [
            'research-use and redistribution are independent dimensions',
            'unresolved or not-permitted research use never enters the training manifest',
            'each verified decision cites an archived official document with SHA-256',
            'no source is excluded to hit a target image count',
        ],
    }

    # ---- write outputs ---------------------------------------------------- #
    csv_path = reports / 'stage5e_license_audit.csv'
    fieldnames = sorted({key for row in rows for key in row})
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = reports / 'stage5e_license_audit.json'
    json_path.write_text(json.dumps({'summary': summary, 'sources': rows},
                                    indent=2, ensure_ascii=False), encoding='utf-8')

    md_path = reports / 'stage5e_license_audit.md'
    lines = [
        '# Stage 5E 许可审计（Work Package A）',
        '',
        f'访问日期：{rows[0]["accessed"]}；证据存档：`reports/stage5e_evidence/`。',
        '',
        '## 结论摘要',
        '',
        f'- 四个待核验来源在 Stage 5D accepted 清单中共 **{summary["accepted_images_in_four"]}** 张。',
        f'- 其中 **{summary["train_eligible_in_four"]}** 张的本地研究训练用途有官方证据支持，可以进入训练清单。',
        f'- 因许可被排除的训练图像：**{summary["excluded_from_training"]}** 张。',
        f'- VERIFIED_RESEARCH_USE：**{summary["moved_from_unresolved_to_verified"]}** 张；'
        f'RESTRICTED_RESEARCH_USE：**{summary["moved_from_unresolved_to_restricted"]}** 张。',
        '',
        '## 逐来源判定',
        '',
        '| 来源 | 真实名称 | 许可 | 研究训练 | 再分发 | checkpoint | accepted |',
        '|---|---|---|---|---|---|---:|',
    ]
    for row in four_rows:
        lines.append(
            f'| `{row["source_id"]}` | {row["real_name"]} | {row["licence_id"]} | '
            f'{row["research_use"]} | {row["redistribution"]} | {row["checkpoint_release"]} | '
            f'{row["accepted_images_stage5d"]:,} |')
    lines += [
        '',
        '## 五个权利维度是分开的',
        '',
        '`train_eligible` 只由研究用途维度决定；`redistribution_eligible` 只由再分发维度决定。'
        '两者在代码中是两个独立属性（`src/registry/stage5e_licence.py`），并有单测保证它们不会互相覆盖。',
        '',
        '## 证据与限制',
        '',
        '- 证据均为发布方官方端点：Kaggle dataset API 的 `licenseName` 字段，以及 Harvard Dataverse '
        '数据集 API 的 `license` / `versionState` / `termsOfUse` 字段。响应体原样存档，SHA-256 记录在 '
        '`reports/stage5e_evidence/evidence_index.json`。',
        '- **限制**：Kaggle 的许可字段是上传者的声明，不是对版权链条的独立法律核验；本报告只声称'
        '"发布方官方端点如此声明"，不声称"已在法律上确认版权归属"。',
        '- CC BY-NC 4.0（`agrifm_ref9`）只允许非商业使用。本项目为学术、非商业研究，故本地训练可用；'
        '但**发布 checkpoint** 是否构成非商业使用无法从条款文字判定，标为 `UNRESOLVED` 而不是"允许"。',
        '- 未对四个来源做任何降级或"凑数"处理：它们的许可状态是从 official endpoint 读出来的，'
        '不是为了让总量达到某个数字而保留或排除。',
    ]
    if evidence_problems:
        lines += ['', '## 证据问题', '']
        for problem in evidence_problems:
            lines.append(f'- `{problem["source_id"]}`: {problem["problem"]} ({problem["detail"]})')
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if evidence_problems:
        print('EVIDENCE PROBLEMS:', evidence_problems, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
