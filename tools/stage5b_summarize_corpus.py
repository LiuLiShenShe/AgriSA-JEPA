#!/usr/bin/env python
"""Stage 5B: corpus composition, sampling strategies and DataLoader smoke test.

    .venv-uv/bin/python tools/stage5b_summarize_corpus.py --smoke-test

Writes ``reports/stage5b_source_distribution.md`` and
``reports/stage5b_source_distribution.json``, then (with ``--smoke-test``)
loads a handful of samples through the real ``AgriFieldDataset``/``DataLoader``
used by I-JEPA training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.registry.source_balance import (composition, shannon_entropy,
                                         strategy_weights, effective_sample_counts,
                                         STRATEGIES)


def load_rows(path: Path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def max_source_share_readable(rows) -> float:
    table = composition(rows, 'source_prefix')
    return max((e['share'] for e in table.values()), default=0.0)


def describe_manifest(name: str, rows):
    return {
        'images': len(rows),
        'sources': len({r['source_prefix'] for r in rows}),
        'max_source_share': max_source_share_readable(rows),
        'source_entropy_bits': shannon_entropy(rows, 'source_prefix'),
        'domain_entropy_bits': shannon_entropy(rows, 'acquisition_domain'),
        'groups': len({r.get('component', r['sample_id']) for r in rows}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--smoke-samples', type=int, default=8)
    args = parser.parse_args()

    train_path = ROOT / 'data/frozen/agri-field-d1-candidate_train.csv'
    val_path = ROOT / 'data/frozen/agri-field-d1-candidate_val.csv'
    v1_train = ROOT / 'data/frozen/agrifield40k-v1_train.csv'
    v1_val = ROOT / 'data/frozen/agrifield40k-v1_val.csv'

    train_rows = load_rows(train_path)
    val_rows = load_rows(val_path)
    v1_rows = load_rows(v1_train) + load_rows(v1_val)
    # v1 uses per-batch filename prefixes (lvb11, lvb32, ...) for LUCASVision.
    # Canonicalise them so the entropy comparison is at the source level, not the
    # filename-batch level.
    prefix_map = {
        'lvb': 'lucas_vision', 'acw': 'acre_crop_weed', 'vgn': 'vegann',
        'rwd': 'radish_wheat', 'swd': 'sorghum_weed', 'inat': 'inat_weeds',
    }
    for row in v1_rows:
        prefix = row.get('source_prefix', '')
        canonical = prefix_map.get(prefix, '')
        if not canonical:
            for candidate, mapped in prefix_map.items():
                if prefix.startswith(candidate):
                    canonical = mapped
                    break
        row['source_prefix'] = canonical or prefix
    all_rows = train_rows + val_rows

    v1_by_source = Counter(r.get('source', '') for r in v1_rows if r.get('status') == 'include')
    d1_by_source = Counter(r.get('source_prefix', '') for r in train_rows)

    strategies = {}
    for strategy in STRATEGIES:
        weights = strategy_weights(all_rows, strategy)
        strategies[strategy] = effective_sample_counts(all_rows, weights)

    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'field_d1_train': describe_manifest('train', train_rows),
        'field_d1_val': describe_manifest('val', val_rows),
        'v1_for_reference': describe_manifest('v1', v1_rows),
        'v1_max_source_share': (max(v1_by_source.values()) / sum(v1_by_source.values())
                                if v1_by_source else 0.0),
        'composition': {
            'train_by_source': composition(train_rows, 'source_prefix'),
            'train_by_domain': composition(train_rows, 'acquisition_domain'),
            'val_by_source': composition(val_rows, 'source_prefix'),
            'val_by_domain': composition(val_rows, 'acquisition_domain'),
        },
        'sampling_strategies': strategies,
        'notes': [
            'The reference target of keeping any single source under 50% of sampled '
            'mass is a distribution hypothesis, not a data-deletion rule.',
            'source_balanced repeats small sources, which can show the model the same '
            'image many times; source_capped avoids that by capping the large source '
            'and proportionally up-weighting the rest.',
        ],
    }

    if args.smoke_test:
        from src.datasets.agri_dataset import AgriFieldDataset
        report = {'status': 'ok', 'errors': []}
        try:
            # the production transform the I-JEPA loop applies to pretrain data
            from torchvision import transforms as tv
            transform = tv.Compose([
                tv.Resize((224, 224)),
                tv.ToTensor(),
            ])
            dataset = AgriFieldDataset(
                manifest_path=str(train_path), split='train',
                transform=transform, max_samples=args.smoke_samples,
                root_path='/data/datasets/AgriField-40K')
            report['dataset_length'] = len(dataset)
            image, sample_id = dataset[0]
            report['first_sample'] = {
                'type': 'tuple',
                'sample_id': str(sample_id),
                'image_shape': list(image.shape),
                'image_dtype': str(image.dtype),
            }
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            report['batch'] = {
                'images': list(batch[0].shape),
                'ids': [str(i) for i in batch[1]],
            }
            def _resolve(rel: str) -> Path:
                path = Path(rel)
                if path.is_absolute():
                    return path
                return Path('/data/datasets/AgriField-40K') / path
            report['paths_exist'] = all(
                _resolve(r['path']).is_file()
                for r in load_rows(train_path)[:args.smoke_samples])
            report['paths_checked'] = [
                str(_resolve(r['path'])) for r in
                load_rows(train_path)[:args.smoke_samples]]
        except Exception as exc:  # noqa: BLE001 - report the failure, do not hide it
            report['status'] = 'failed'
            report['errors'].append(f'{type(exc).__name__}: {exc}')
        payload['loader_smoke_test'] = report

    (ROOT / 'reports/stage5b_source_distribution.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def table(counter: Counter) -> str:
        total = sum(counter.values()) or 1
        return '\n'.join(f'| {k} | {v} | {v / total:.1%} |'
                         for k, v in counter.most_common())

    md = [
        '# Stage 5B — Field-D1 来源分布与采样策略',
        '',
        f"- train：{payload['field_d1_train']['images']}；"
        f"val：{payload['field_d1_val']['images']}",
        f"- 来源数：{payload['field_d1_train']['sources']}",
        f"- 最大来源占比：{payload['field_d1_train']['max_source_share']:.1%}"
        f"（v1 参考值 {payload['v1_max_source_share']:.1%}）",
        f"- 来源熵：{payload['field_d1_train']['source_entropy_bits']} bit"
        f"（v1 参考值 {payload['v1_for_reference']['source_entropy_bits']} bit）",
        '',
        '## train 来源分布',
        '',
        '| 来源 | 图像 | 占比 |',
        '|---|---:|---:|',
        table(d1_by_source),
        '',
        '## 采样策略（每来源采样后占比）',
        '',
    ]
    for strategy, entries in strategies.items():
        md.append(f'### {strategy}')
        md.append('')
        md.append('| 来源 | 语料图像 | 权重 | 采样后占比 |')
        md.append('|---|---:|---:|---:|')
        for source, entry in sorted(entries.items(),
                                    key=lambda kv: -kv[1]['share_after_sampling']):
            md.append(f"| {source} | {entry['images']} | {entry['weight']} | "
                      f"{entry['share_after_sampling']:.1%} |")
        md.append('')
    (ROOT / 'reports/stage5b_source_distribution.md').write_text(
        '\n'.join(md) + '\n', encoding='utf-8')

    print(json.dumps({k: payload[k] for k in (
        'field_d1_train', 'field_d1_val', 'v1_max_source_share',
        'loader_smoke_test') if k in payload}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
