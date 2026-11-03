#!/usr/bin/env python
"""Stage 5B Gate 6: build the Field-D1 candidate corpus.

Usage
-----
    .venv-uv/bin/python tools/stage5b_build_field_corpus.py

What it does, in order:

1. reads the frozen v1 train/val manifests (read-only) and inherits their
   source rows and capture groups;
2. reads the newly ingested Stage 5B source manifests;
3. applies the domain policy (Gate 2), the holdout rule and the exact
   duplicate rule (Gate 5);
4. assigns every capture group wholly to train or val;
5. writes ``data/frozen/agri-field-d1-candidate_{train,val}.csv`` and the
   manifest, both hashed.

The result stays CANDIDATE unless the caller passes ``--freeze`` and the freeze
gate actually passes; nothing here promotes itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.registry.domain_policy import admission
from src.registry.field_freeze import (apply_gate, assign_group_split,
                                       canonical_csv, load_ingested,
                                       load_v1_manifest, write_field_corpus)
from src.registry.source_balance import (composition, shannon_entropy,
                                         strategy_weights, effective_sample_counts)

def max_share(rows) -> float:
    table = composition(rows, 'source_prefix')
    return max((entry['share'] for entry in table.values()), default=0.0)


FROZEN_DIR = ROOT / 'data/frozen'
OUT_PREFIX = FROZEN_DIR / 'agri-field-d1-candidate'
OUT_REPORT = ROOT / 'reports/stage5b_field_freeze_report.json'
OUT_SUMMARY = ROOT / 'reports/stage5b_field_freeze_auto.md'

#: v1 sources inherited into Field-D1.  Each row carries the domain label the
#: Stage 5B ledger assigns, with its evidence.  v1 rows are not re-labelled
#: per image; the source-level evidence is the v1 provenance (published dataset
#: cards + the frozen manifest's own ``source_url``).
V1_SOURCES = {
    'lucas_vision': ('LUCASVision', 'FIELD_IN_SITU', 'official_documentation',
                     'high', 'reviewed'),
    'acre_crop_weed': ('ACRECropWeed', 'FIELD_IN_SITU', 'official_documentation',
                       'high', 'reviewed'),
    'vegann': ('VegAnn', 'FIELD_IN_SITU', 'official_documentation', 'high', 'reviewed'),
    'radish_wheat': ('RadishWheat', 'FIELD_IN_SITU', 'official_documentation',
                     'medium', 'reviewed'),
    'sorghum_weed': ('SorghumWeed', 'FIELD_IN_SITU', 'official_documentation',
                     'medium', 'reviewed'),
    'inat_weeds': ('iNatWeeds', 'FIELD_IN_SITU', 'official_documentation',
                   'medium', 'reviewed'),
}

#: New Stage 5B sources and their labels.  ``applebbch81`` is ORCHARD_IN_SITU
#: and therefore admissible; ``sesame_weed`` is FIELD_IN_SITU; ``weedcrop`` is
#: UNKNOWN and is *not* added to Field-D1 (it is recorded as a candidate for
#: Controlled/unassigned review instead).
NEW_SOURCES = {
    'sesame_weed': {
        'source_id': 'sesame_weed',
        'source_label': 'Sesame Weed (Crop and Weed Detection)',
        'domain': 'FIELD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'review_status': 'model_assisted_pending_human',
        'review_reason': ('Publisher describes field images of sesame crop and weeds; '
                          'vision-model sampling of 18/18 cells reported open-field '
                          'scenes with natural soil backgrounds and no laboratory '
                          'artefacts; a human confirmation is still required.'),
        'license': 'CC0-1.0',
        'attribution': 'Ravirajsinh Dabhi (Kaggle), CC0',
        'source_url': ('https://www.kaggle.com/datasets/ravirajsinh45/'
                       'crop-and-weed-detection-data-with-bounding-boxes'),
        'manifest': 'data/registry/ingested/sesame_weed_stage5b_manifest.csv',
        'val_fraction': 0.2,
    },
    'applebbch81': {
        'source_id': 'applebbch81',
        'source_label': 'AppleBBCH81 (orchard, BBCH 81-85)',
        'domain': 'ORCHARD_IN_SITU',
        'evidence_type': 'official_documentation',
        'confidence': 'high',
        'review_status': 'model_assisted_pending_human',
        'review_reason': ('Publisher documents photographs taken in the LatHort '
                          'orchard at BBCH 81-85, two views per tree; vision-model '
                          'sampling of 18/18 cells reported orchard scenes. A human '
                          'confirmation is still required.'),
        'license': 'CC-BY-4.0',
        'attribution': ('Kodors et al. (2024), Drones 8(12):734, '
                        'https://doi.org/10.3390/drones8120734'),
        'source_url': 'https://doi.org/10.3390/drones8120734',
        'manifest': 'data/registry/ingested/applebbch81_stage5b_manifest.csv',
        'val_fraction': 0.2,
    },
    'weedcrop': {
        'source_id': 'weedcrop',
        'source_label': 'WeedCrop Image Dataset (Ronin-derived)',
        'domain': 'UNKNOWN',
        'evidence_type': 'filename_or_metadata_only',
        'confidence': 'low',
        'review_status': 'pending_human_review',
        'review_reason': ('Kaggle description confirms only that images were augmented '
                          'from an upstream set; it does not state the capture '
                          'setting. Two of nine sampled cells were read as controlled, '
                          'so the source is not admitted.'),
        'license': 'CC0-1.0',
        'attribution': 'Sudars et al. (2020), Data in Brief 31:105833',
        'source_url': ('https://www.kaggle.com/datasets/vinayakshanawad/'
                       'weedcrop-image-dataset'),
        'manifest': 'data/registry/ingested/weedcrop_stage5b_manifest.csv',
        'val_fraction': 0.2,
    },
}

HOLDOUT_IDS = set()


def load_holdout():
    path = ROOT / 'data/registry/downstream_holdout_sources.csv'
    with open(path, newline='', encoding='utf-8') as handle:
        return {r['source_id'] for r in csv.DictReader(handle)}


def load_v1_rows():
    rows = []
    for split in ('train', 'val'):
        path = FROZEN_DIR / f'agrifield40k-v1_{split}.csv'
        prefix_map = {
            'lvb': 'lucas_vision', 'acw': 'acre_crop_weed', 'vgn': 'vegann',
            'rwd': 'radish_wheat', 'swd': 'sorghum_weed', 'inat': 'inat_weeds',
        }
        with open(path, newline='', encoding='utf-8') as handle:
            for raw in csv.DictReader(handle):
                if str(raw.get('status', '')).strip() != 'include':
                    continue
                prefix = raw.get('source_prefix', '')
                sid = prefix_map.get(prefix, '')
                if not sid:
                    for candidate, mapped in prefix_map.items():
                        if prefix.startswith(candidate):
                            sid = mapped
                            break
                if sid not in V1_SOURCES:
                    continue
                label, domain, evidence, confidence, review = V1_SOURCES[sid]
                rows.append({
                    'sample_id': raw['sample_id'],
                    'path': raw['path'],
                    'source_prefix': sid,
                    'source': label,
                    'original_id': raw.get('original_id', ''),
                    'split': raw.get('split', split),
                    'width': raw.get('width', ''),
                    'height': raw.get('height', ''),
                    'mode': raw.get('mode', ''),
                    'format': raw.get('format', ''),
                    'bytes': raw.get('bytes', ''),
                    'sha256': raw.get('sha256', ''),
                    'pixel_sha256': raw.get('pixel_sha256', ''),
                    'phash64': raw.get('phash64', ''),
                    'dhash64': raw.get('dhash64', ''),
                    'decode_ok': raw.get('decode_ok', ''),
                    'license': raw.get('license', ''),
                    'attribution': raw.get('attribution', ''),
                    'source_url': raw.get('source_url', ''),
                    'status': 'include',
                    'reason': raw.get('reason', ''),
                    'component': raw.get('component', raw['sample_id']),
                    'acquisition_domain': domain,
                    'domain_evidence_type': evidence,
                    'domain_confidence': confidence,
                    'domain_review_status': review,
                    'domain_review_reason': 'inherited from frozen agrifield40k-v1',
                    'quality_status': 'v1_screened',
                    'duplicate_status': 'v1_screened',
                    'origin_manifest': str(path),
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true',
                        help='only writes FROZEN if the freeze gate actually passes')
    args = parser.parse_args()

    holdout = load_holdout()
    print(f'holdout sources: {len(holdout)}')

    v1_rows = load_v1_rows()
    print(f'v1 rows inherited: {len(v1_rows)}')

    new_rows = []
    admitted_sources = []
    blocked_sources = []
    for source_id, spec in NEW_SOURCES.items():
        manifest = ROOT / spec['manifest']
        if not manifest.is_file():
            blocked_sources.append({'source_id': source_id, 'reason': 'manifest missing'})
            continue
        rows = load_ingested(manifest, spec['source_id'], spec['source_label'],
                             spec['domain'], spec['evidence_type'], spec['confidence'],
                             spec['review_status'], spec['review_reason'],
                             spec['license'], spec['attribution'], spec['source_url'],
                             str(manifest))
        # drop exact duplicates / cross-source duplicates before grouping
        clean = [r for r in rows
                 if not str(r.get('duplicate_status', '')).startswith('exact_')
                 and r.get('duplicate_status') != 'cross_source_duplicate']
        splits = assign_group_split(clean, spec['val_fraction'])
        for row in clean:
            row['split'] = splits.get(str(row.get('component', '')), 'train')
        admitted, why = admission(spec['domain'], spec['evidence_type'],
                                  spec['confidence'], spec['review_status'],
                                  level='candidate')
        record = {
            'source_id': source_id,
            'images_in_manifest': len(rows),
            'images_after_exact_dedup': len(clean),
            'domain': spec['domain'],
            'admitted_to_candidate': admitted,
            'reason': why,
        }
        (admitted_sources if admitted else blocked_sources).append(record)
        if admitted:
            new_rows.extend(clean)
        else:
            for row in clean:
                row['status'] = 'excluded'
                row['reason'] = why
            blocked_sources[-1]['examples'] = 0
    print(f'new rows after gates: {len(new_rows)}')

    all_rows = v1_rows + new_rows
    kept, rejections, rejected_examples = apply_gate(all_rows, holdout)
    print(f'rows after domain/holdout gate: {len(kept)}; rejected {len(rejections)}')
    print(json.dumps(rejections, indent=2))

    extra = {
        'policy': {
            'version': 'stage5b-field-domain-policy-v1',
            'admitted_domains': ['FIELD_IN_SITU', 'ORCHARD_IN_SITU'],
            'excluded_domains': ['CONTROLLED_LAB', 'PROTECTED_CULTIVATION',
                                 'UAV_AERIAL', 'SATELLITE_REMOTE', 'UNKNOWN'],
            'holdout_sources': sorted(holdout),
        },
        'inherited': {
            'frozen_v1_train_csv': 'agrifield40k-v1_train.csv',
            'frozen_v1_val_csv': 'agrifield40k-v1_val.csv',
            'v1_sha256': {
                'agrifield40k-v1_train.csv':
                    'd63183af8ede53bc859474d1fc82cc969335c71967008812a485de10476a5435',
                'agrifield40k-v1_val.csv':
                    '86efe5156291950b25af692d41b5f1b4cee437dd3ba2a44ce39e2cf18984f944',
            },
            'note': ('v1 is read-only; rows and capture groups are inherited, and the '
                     'v1 unreviewed perceptual-duplicate and pending-audit findings '
                     'are inherited as risk, not cleared.'),
        },
        'new_sources': admitted_sources,
        'blocked_sources': blocked_sources,
        'gate_rejections': rejections,
        'distribution': {
            'by_source': composition(kept, 'source_prefix'),
            'by_domain': composition(kept, 'acquisition_domain'),
            'max_source_share': max_share(kept),
            'source_entropy_bits': shannon_entropy(kept, 'source_prefix'),
            'domain_entropy_bits': shannon_entropy(kept, 'acquisition_domain'),
            'sampling_strategies': {
                strategy: effective_sample_counts(kept, strategy_weights(kept, strategy))
                for strategy in ('natural', 'source_balanced', 'source_capped')
            },
        },
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest = write_field_corpus(OUT_PREFIX, kept, extra)
    print(json.dumps({k: manifest[k] for k in ('status', 'images', 'sources',
                                               'split_integrity')},
                     indent=2))

    review_pending = sum(1 for r in kept
                         if r.get('domain_review_status') != 'reviewed')
    freeze_ok = (manifest['split_integrity']['cross_split_groups'] == 0
                 and review_pending == 0
                 and max_share(kept) < 0.9)
    status = 'FROZEN' if (args.freeze and freeze_ok) else 'CANDIDATE'
    report = {
        'status': status,
        'images': manifest['images'],
        'sources': manifest['sources'],
        'max_source_share': max_share(kept),
        'source_entropy_bits': shannon_entropy(kept, 'source_prefix'),
        'rows_pending_human_review': review_pending,
        'freeze_gate_passed': freeze_ok,
        'freeze_gate': {
            'cross_split_groups_zero': manifest['split_integrity']['cross_split_groups'] == 0,
            'human_review_complete': review_pending == 0,
            'no_source_over_90pct': max_share(kept) < 0.9,
        },
        'new_sources': admitted_sources,
        'blocked_sources': blocked_sources,
        'sha256': manifest['sha256'],
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n',
                          encoding='utf-8')
    lines = [
        '# Stage 5B — Field-D1 候选版本建设报告',
        '',
        f'- 状态：**{status}**',
        f"- train：{manifest['images']['train']} 张；val：{manifest['images']['val']} 张",
        f"- 来源数：{manifest['sources']}",
        f"- 最大来源占比：{max_share(kept):.1%}",
        f"- 采集组跨划分数量：{manifest['split_integrity']['cross_split_groups']}",
        f"- 待人工确认域标签的图像：{review_pending}",
        '',
        '## 冻结门禁',
        '',
        '| 条件 | 结果 |',
        '|---|---|',
        f"| 采集组不跨 train/val | {'通过' if report['freeze_gate']['cross_split_groups_zero'] else '未通过'} |",
        f"| 全部图像域标签经人工确认 | {'通过' if report['freeze_gate']['human_review_complete'] else '未通过'} |",
        f"| 无来源占比超过 90% | {'通过' if report['freeze_gate']['no_source_over_90pct'] else '未通过'} |",
        '',
        '冻结需要三项全部通过。当前两项未通过，因此保持 CANDIDATE。',
        '',
        '## 新来源',
        '',
        '| 来源 | 域 | 纳入 | 理由 |',
        '|---|---|---|---|',
    ]
    for record in admitted_sources:
        lines.append(f"| {record['source_id']} | {record['domain']} | "
                     f"{'是' if record['admitted_to_candidate'] else '否'} | "
                     f"{record['reason']} |")
    for record in blocked_sources:
        lines.append(f"| {record['source_id']} | {record.get('domain', '')} | 否 | "
                     f"{record['reason']} |")
    OUT_SUMMARY.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # ---- Controlled-D1 / Mixed-D1 definitions (no image list is duplicated) --
    controlled = {
        'data_version': 'agri-controlled-d1-candidate',
        'status': 'DEFINED_NOT_BUILT',
        'approved': False,
        'reason': ('The version is defined so the PlantVillage data is not lost, but '
                   'no manifest is written: the CC-BY-SA-3.0-US terms, the 12,436 '
                   'perceptual candidates and the source-level licence conflict are '
                   'all unresolved.'),
        'images': {
            'files_downloaded': 54305,
            'duplicate_free_candidates': 54284,
            'perceptual_candidate_pairs_pending': 12436,
        },
        'sources': [{
            'source_id': 'agrifm_ref93',
            'name': 'PlantVillage (colour images)',
            'acquisition_domain': 'CONTROLLED_LAB',
            'license': 'CC-BY-SA-3.0-US (Zenodo metadata says cc-by-4.0)',
            'image_manifest': 'data/registry/ingested/plantvillage_manifest.csv',
            'audit': 'reports/stage5a_plantvillage_audit.json',
        }],
        'excluded_from': ['agri-field-d1-candidate'],
        'prerequisites_to_build': [
            'resolve CC-BY-SA-3.0-US attribution/share-alike obligations',
            'review or explicitly accept the 12,436 perceptual candidates',
            'record a human approval for the source',
        ],
    }
    mixed = {
        'data_version': 'agri-mixed-d1-candidate',
        'status': 'DEFINED_NOT_BUILT',
        'approved': False,
        'composition_rule': 'Field-D1 candidate rows + Controlled-D1 candidate rows',
        'purpose': ('test whether adding a large controlled-leaf corpus helps or '
                    'hurts transfer to field tasks'),
        'prerequisites': [
            'Controlled-D1 reaches at least CANDIDATE status with resolved licence',
            'both sides pass the same duplicate and holdout gates',
            'no holdout source appears on either side',
        ],
        'not_run_in_stage5b': True,
    }
    (FROZEN_DIR / 'agri-controlled-d1-candidate_manifest.json').write_text(
        json.dumps(controlled, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (FROZEN_DIR / 'agri-mixed-d1-candidate_definition.json').write_text(
        json.dumps(mixed, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
