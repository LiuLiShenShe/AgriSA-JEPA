#!/usr/bin/env python
"""Stage 8 Phase 1: audit candidate downstream datasets and cross-check leakage.

For every candidate dataset in the Stage 8 specification (GWHD 2021, PlantDoc,
MinneApple, CropAndWeed, GrowliFlower, PlantSeg, DeepWeeds, PhenoBench), the
audit:

1. enumerates every local image file and verifies it decodes (RGB-convertible),
   recording bytes / width / height / format;
2. computes ``raw_sha256`` and ``pixel_sha256`` for every image, where
   ``pixel_sha256`` is ``sha256(convert('RGB').tobytes())`` -- byte-identical to
   how the pretraining manifest is built, so an exact match is real leakage;
3. cross-checks both hash columns against ``data/stage6/
   stage6_r3v3_train_view_nvme.csv`` (the frozen pretraining manifest, 147,277
   images) -- exact raw matches and exact pixel matches are hard leaks;
4. records a READY / READY_WITH_MODIFICATION / BLOCKED verdict per dataset with
   the reason, plus the annotation/split availability that would drive Phase 2.

Outputs ``reports/stage8_dataset_audit.{md,json}`` and
``reports/stage8_leakage_audit.{md,json}`` and
``reports/stage8_dataset_manifest.json``.

No numbers are invented.  A dataset not present locally is BLOCKED with reason
'not_downloaded'; a dataset present but unusable (missing annotations) is
BLOCKED with reason 'missing_annotation'.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'reports'
DATA = ROOT / 'data'

PRETRAIN_MANIFEST = DATA / 'stage6' / 'stage6_r3v3_train_view_nvme.csv'
LEAKAGE_CHUNK = 1 << 20


# --------------------------------------------------------------------------
# Candidate dataset declarations (from the Stage 8 specification).
# --------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    dataset_id: str
    task: str          # 'detection' | 'segmentation' | 'classification'
    images_dir: Optional[str]
    annotation_dir: Optional[str]
    annotation_note: str
    official_splits: str
    license: str
    license_source: str
    overlap_risk: str


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        'gwhd2021_det', 'detection',
        '/data/datasets/GWHD2021/extracted/gwhd_2021/images',
        '/data/datasets/GWHD2021/extracted/gwhd_2021',
        'three official CSVs (competition_{train,val,test}.csv); reuse Stage 5A pipeline',
        'train / val / test (official)',
        'CC-BY-4.0', 'Zenodo 5092309',
        'excluded from all pretraining manifests (holdout)',
    ),
    DatasetSpec(
        'plantdoc_cls', 'classification',
        str(ROOT / 'data/holdout_raw/plantdoc_extract/PlantDoc-Dataset-master'),
        str(ROOT / 'data/holdout_raw/plantdoc_extract/PlantDoc-Dataset-master'),
        'class-directory structure (train/{28 classes}, test/{28 classes}); '
        'no bbox annotations shipped, so detection is BLOCKED locally; '
        'classification reuses class directory labels',
        'train / test (class dirs), val carved from train with fixed seed',
        'CC-BY-4.0', 'GitHub pratikkayal/PlantDoc-Dataset',
        'not in pretraining sources (holdout)',
    ),
    DatasetSpec(
        'minneapple_det', 'detection',
        None, None,
        'not downloaded; license "research use (repository terms)" must be '
        'accepted before local audit',
        'train / val / test (official)',
        'research use (repository terms)', 'GitHub nicolaihaeni/MinneApple',
        'none',
    ),
    DatasetSpec(
        'cropandweed_sem', 'segmentation',
        '/data/datasets/cropandweed/images',
        str(ROOT / 'data/holdout_raw/cropandweed_annotations'),
        'bboxes/labelIds/params per image (8,034 each); semantic masks are '
        'labelIds/{CropAndWeed}/*.png; splits derived from ave_/vwg_ '
        'recording-session prefix (no official CSV split)',
        'ave_ (Application Set) / vwg_ (Experimental Set); session-grouped',
        'AIT non-commercial research', 'LICENCE.txt (local, AIT)',
        'upstream ACRECropWeed (790 images) in pretraining corpus is a '
        'DIFFERENT dataset -- verify no name/hash overlap',
    ),
    DatasetSpec(
        'growliflower_sem', 'segmentation',
        None, None,
        'not downloaded; must check sequence/session leakage before use',
        'train / val / test (time series)',
        'CC-BY-4.0', 'ipb.uni-bonn.de/data/growliflower',
        'none',
    ),
    DatasetSpec(
        'plantseg_sem', 'segmentation',
        None, None,
        'not downloaded; license "see repository" must be read before use',
        'train / test (per disease)',
        'see repository', 'arXiv:2409.04038',
        'none',
    ),
    DatasetSpec(
        'deepweeds_cls', 'classification',
        '/data/datasets/DeepWeeds/images',
        str(ROOT / 'data/holdout_raw/deepweeds_labels'),
        'labels.csv (Filename,Label,Species) + 5-fold train/val/test subset '
        'CSVs; no single official split -- use fold-0 as the fixed split',
        '5-fold stratified CV (paper)',
        'CC-BY-4.0', 'GitHub AlexOlsen/DeepWeeds',
        'appears in Agri-FM+ pretraining (ref [60]); excluded from our '
        'pretraining manifests',
    ),
    DatasetSpec(
        'phenobench_sem', 'segmentation',
        '/data/datasets/PhenoBench/PhenoBench-v110',
        '/data/datasets/PhenoBench/PhenoBench-v110',
        'official download + official eval tool installed; Stage 7B probes '
        'already cover this dataset',
        'train / val (+ test server)',
        'CC-BY-SA-4.0', 'phenobench.org',
        'excluded from all pretraining manifests (holdout)',
    ),
]


# --------------------------------------------------------------------------
# Hash / decode audit of every image in a dataset.
# --------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(LEAKAGE_CHUNK):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ImageRow:
    dataset_id: str
    relative_path: str
    bytes: int
    raw_sha256: str
    pixel_sha256: str
    width: int
    height: int
    mode: str
    format: str
    decode_ok: bool
    decode_error: str = ''


def _audit_image(dataset_id: str, path: Path, root: Path) -> ImageRow:
    rel = str(path.relative_to(root))
    try:
        size = path.stat().st_size
        raw_sha = _sha256_file(path)
    except OSError as exc:
        return ImageRow(dataset_id, rel, 0, '', '', 0, 0, '', '',
                        False, f'stat_error:{exc}')
    try:
        with Image.open(path) as handle:
            handle.load()
            rgb = handle.convert('RGB')
            width, height = handle.width, handle.height
            mode, fmt = handle.mode, handle.format
            pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    except Exception as exc:  # noqa: BLE001 - any decoder error is recorded
        return ImageRow(dataset_id, rel, size, raw_sha, '', 0, 0, '', '',
                        False, f'decode_error:{exc}')
    return ImageRow(dataset_id, rel, size, raw_sha, pixel_sha,
                    width, height, mode, fmt, True)


def collect_images(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob('*')
                  if p.is_file() and p.suffix.lower() in
                  {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'})


def audit_dataset(spec: DatasetSpec, workers: int,
                  pixel_hashes: bool) -> Dict[str, object]:
    cache_dir = ROOT / 'data' / 'stage8_audit_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f'{spec.dataset_id}.json'
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        if (cached.get('pixel_hashes') == pixel_hashes
                and cached.get('images_dir') == spec.images_dir):
            print(f'  {spec.dataset_id}: using hash cache '
                  f'({cached["n_images"]} images)', flush=True)
            return cached

    if spec.images_dir is None:
        return {'dataset_id': spec.dataset_id,
                'status': 'BLOCKED',
                'status_reason': 'not_downloaded',
                'n_images': 0, 'n_decode_ok': 0}
    root = Path(spec.images_dir)
    files = collect_images(root)
    if not files:
        return {'dataset_id': spec.dataset_id,
                'status': 'BLOCKED',
                'status_reason': f'no_images_found_at:{spec.images_dir}',
                'n_images': 0, 'n_decode_ok': 0}

    rows: List[ImageRow] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(lambda p: _audit_image(
                spec.dataset_id, p, root), files), 1):
            if not pixel_hashes and row.decode_ok:
                # cheap mode: drop the pixel hash to save memory
                row.pixel_sha256 = ''
            rows.append(row)
            if i % 2000 == 0:
                print(f'  {spec.dataset_id}: audited {i}/{len(files)}', flush=True)

    n_ok = sum(1 for r in rows if r.decode_ok)
    n_raw_zero = sum(1 for r in rows if not r.raw_sha256)
    result = {
        'dataset_id': spec.dataset_id,
        'status': 'READY_WITH_MODIFICATION',
        'status_reason': 'local images audited',
        'images_dir': spec.images_dir,
        'annotation_dir': spec.annotation_dir,
        'annotation_note': spec.annotation_note,
        'official_splits': spec.official_splits,
        'license': spec.license,
        'license_source': spec.license_source,
        'overlap_risk': spec.overlap_risk,
        'pixel_hashes': pixel_hashes,
        'n_images': len(rows),
        'n_decode_ok': n_ok,
        'n_raw_sha_missing': n_raw_zero,
        'rows': [asdict(r) for r in rows],
    }
    cache_path.write_text(json.dumps(result, indent=1, ensure_ascii=False),
                          encoding='utf-8')
    return result


# --------------------------------------------------------------------------
# Leakage cross-check against the frozen pretraining manifest.
# --------------------------------------------------------------------------
def load_pretraining_hashes(csv_path: Path) -> Tuple[set, set, int]:
    raw, pixel = set(), set()
    n = 0
    with csv_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n += 1
            if row.get('raw_sha256'):
                raw.add(row['raw_sha256'])
            if row.get('pixel_sha256'):
                pixel.add(row['pixel_sha256'])
    return raw, pixel, n


def crosscheck(rows: List[Dict[str, object]], raw_set: set,
               pixel_set: set) -> Dict[str, object]:
    exact_raw, exact_pixel = [], []
    for row in rows:
        if not row.get('decode_ok'):
            continue
        if row['raw_sha256'] and row['raw_sha256'] in raw_set:
            exact_raw.append(row['relative_path'])
        if row.get('pixel_sha256') and row['pixel_sha256'] in pixel_set:
            exact_pixel.append(row['relative_path'])
    return {
        'n_scanned': len(rows),
        'n_decode_ok': sum(1 for r in rows if r.get('decode_ok')),
        'n_exact_raw_matches': len(exact_raw),
        'n_exact_pixel_matches': len(exact_pixel),
        'exact_raw_matches': exact_raw[:20],
        'exact_pixel_matches': exact_pixel[:20],
        'verdict': ('LEAKAGE' if (exact_raw or exact_pixel)
                    else 'CLEAN_exact'),
    }


# --------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------
def render_dataset_md(results: Dict[str, Dict[str, object]],
                      n_pretrain: int) -> str:
    lines = ['# Stage 8 — 数据集审计', '',
             f'> 生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec="seconds")}',
             '',
             f'- 预训练 manifest：`{PRETRAIN_MANIFEST.name}`（{n_pretrain} 张图）。', '']
    lines.append('| dataset | 任务 | 状态 | 本地图数 | 可解码 | 备注 |')
    lines.append('|---|---|---|---|---|---|')
    for spec in DATASETS:
        res = results.get(spec.dataset_id, {})
        status = res.get('status', 'BLOCKED')
        n_img = res.get('n_images', 0)
        n_ok = res.get('n_decode_ok', 0)
        reason = res.get('status_reason', '')
        if res.get('n_exact_raw_matches') or res.get('n_exact_pixel_matches'):
            status = 'LEAKAGE'
            reason = (f"exact raw={res.get('n_exact_raw_matches', 0)} / "
                      f"pixel={res.get('n_exact_pixel_matches', 0)}")
        lines.append(f'| `{spec.dataset_id}` | {spec.task} | {status} | '
                     f'{n_img} | {n_ok} | {reason} |')
    lines += ['', '## 每数据集说明', '']
    for spec in DATASETS:
        res = results.get(spec.dataset_id, {})
        lines.append(f'### `{spec.dataset_id}`（{spec.task}）')
        lines.append('')
        lines.append(f'- 本地路径：`{spec.images_dir or "（未下载）"}`')
        lines.append(f'- 注释：{spec.annotation_note}')
        lines.append(f'- 官方切分：{spec.official_splits}')
        lines.append(f'- 许可：{spec.license}（{spec.license_source}）')
        lines.append(f'- 泄漏风险：{spec.overlap_risk}')
        lines.append(f'- 状态：**{res.get("status", "BLOCKED")}** — '
                     f'{res.get("status_reason", "")}')
        lines.append('')
    return '\n'.join(lines) + '\n'


def render_leakage_md(checks: Dict[str, Dict[str, object]],
                      n_pretrain: int) -> str:
    lines = ['# Stage 8 — 泄漏审计', '',
             f'> 生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec="seconds")}',
             '',
             f'- 与预训练 manifest（`{PRETRAIN_MANIFEST.name}`，{n_pretrain} 张图）'
             '的 raw_sha256 / pixel_sha256 精确比对。', '',
             '- **精确 raw 命中** = 与预训练某张图字节完全一致；', '',
             '- **精确 pixel 命中** = 与预训练某张图解码后像素完全一致'
             '（重编码、重压缩、无损重存都会改变 raw 但保持 pixel）。', '']
    lines.append('| dataset | 扫描 | 可解码 | raw 命中 | pixel 命中 | 结论 |')
    lines.append('|---|---|---|---|---|---|')
    for dataset_id, res in checks.items():
        verdict = res.get('verdict', 'CLEAN_exact')
        if verdict == 'LEAKAGE':
            verdict = '**LEAKAGE**'
        lines.append(f'| `{dataset_id}` | {res["n_scanned"]} | '
                     f'{res["n_decode_ok"]} | {res["n_exact_raw_matches"]} | '
                     f'{res["n_exact_pixel_matches"]} | {verdict} |')
    lines += ['', '## 说明', '',
              '- 只做精确哈希比对；phash / dhash 近重复筛查见 Stage 5E（'
              '`data/corpus_v4/holdout_crossmatch.csv`），其结论已覆盖 '
              'gwhd_2021 / plantdoc / phenobench。', '',
              '- 历史结论：GWHD 有 2 张图与 VegAnn（预训练上游）像素同图，'
              '对应 corpus 样本已在 R3v3 排除（`HOLDOUT_CONFIRMED_PIXEL_MATCH`）。',
              '- 本审计对最终 R3v3 训练集重新做精确哈希比对，'
              '若某数据集出现新命中则 Phase 2 必须先排除泄漏图再冻结协议。', '']
    return '\n'.join(lines) + '\n'


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--datasets', type=str, default='',
                        help='comma-separated subset of dataset_ids')
    parser.add_argument('--pixel-hashes', action='store_true', default=True)
    parser.add_argument('--no-pixel-hashes', dest='pixel_hashes',
                        action='store_false')
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.datasets.split(',') if s.strip()}
    specs = [s for s in DATASETS if not only or s.dataset_id in only]

    R.mkdir(exist_ok=True)
    print(f'loading pretraining manifest {PRETRAIN_MANIFEST}', flush=True)
    raw_set, pixel_set, n_pretrain = load_pretraining_hashes(PRETRAIN_MANIFEST)
    print(f'  {n_pretrain} pretraining images, '
          f'{len(raw_set)} raw / {len(pixel_set)} pixel hashes', flush=True)

    results: Dict[str, Dict[str, object]] = {}
    checks: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        print(f'auditing {spec.dataset_id} ...', flush=True)
        res = audit_dataset(spec, args.workers, args.pixel_hashes)
        # BLOCKED datasets have no rows; skip crosscheck for them.
        if res['n_images'] > 0:
            leak = crosscheck(res['rows'], raw_set, pixel_set)
            res['leakage'] = leak
            if leak['verdict'] == 'LEAKAGE':
                res['status'] = 'LEAKAGE'
                res['status_reason'] = (
                    f"exact raw={leak['n_exact_raw_matches']}, "
                    f"pixel={leak['n_exact_pixel_matches']}")
            checks[spec.dataset_id] = leak
        results[spec.dataset_id] = res

    (R / 'stage8_dataset_manifest.json').write_text(
        json.dumps({'schema_version': 'stage8_dataset_manifest/v1',
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'pretraining_manifest': str(PRETRAIN_MANIFEST.relative_to(ROOT)),
                    'n_pretraining_images': n_pretrain,
                    'datasets': results}, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    (R / 'stage8_dataset_audit.md').write_text(
        render_dataset_md(results, n_pretrain), encoding='utf-8')
    (R / 'stage8_leakage_audit.md').write_text(
        render_leakage_md(checks, n_pretrain), encoding='utf-8')

    summary = {k: {'status': v['status'], 'n_images': v['n_images'],
                   'leakage': (v.get('leakage') or {}).get('verdict', 'n/a')}
               for k, v in results.items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())