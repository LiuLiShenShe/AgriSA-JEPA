"""Batch quality evaluation: CleanVision + pyIQA.

Run per source over the raw files that already decoded.  CleanVision produces
its own flags; pyIQA (MUSIQ / NIQE) produces reference-free scores.  The
module writes a compact CSV per source so the main ``images.parquet`` ledger
never has to touch pixels twice.

Model provenance is recorded (package versions + checkpoint SHA-256) so the
numbers stay traceable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# huggingface.co is unreachable from this machine; the mirror works.
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# One QC shard owns one CPU core.  Left at the default, torch spawns a
# 64-thread intra-op pool per shard; a dozen shards then run ~1000 spinning
# threads and JPEG decode slows by an order of magnitude.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: NR-IQA models resize internally to <=384 px; decoding beyond 512 px only
#: costs time.  Override with --max-side when a different metric needs it.
MAX_SIDE = 512


def package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in ('cleanvision', 'pyiqa', 'torch', 'timm', 'transformers'):
        try:
            module = __import__(name)
            out[name] = getattr(module, '__version__', 'unknown')
        except Exception:  # noqa: BLE001
            out[name] = 'not installed'
    return out


def checkpoint_sha256(path: Path) -> str:
    if not path.exists():
        return ''
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def pyiqa_weights() -> Dict[str, str]:
    cache = Path.home() / '.cache/torch/hub/pyiqa'
    return {p.name: checkpoint_sha256(p) for p in sorted(cache.glob('*')) if p.is_file()}


def cleanvision_report(image_paths: Sequence[str], work_dir: Path,
                       sample_per_source: int) -> Dict[str, object]:
    """Run CleanVision on a deterministic, evenly spaced sample.

    CleanVision is a *diagnostic* signal here (its flags are recorded but do
    not decide acceptance); the per-image gates are the basic metrics and
    pyIQA.  Even spacing avoids the alphabetical bias of ``paths[:N]``.
    """
    from cleanvision import Imagelab
    paths = list(image_paths)
    if sample_per_source and len(paths) > sample_per_source:
        stride = len(paths) / sample_per_source
        paths = [paths[min(len(paths) - 1, int(i * stride))]
                 for i in range(sample_per_source)]
    imagelab = Imagelab(filepaths=list(paths))
    imagelab.find_issues()
    summary_path = work_dir / 'cleanvision_issues.csv'
    imagelab.issues.to_csv(summary_path)
    stats = imagelab.issue_summary
    return {'report': str(summary_path), 'summary': stats.to_dict(),
            'sampled': len(paths), 'available': len(image_paths)}


def _load_picture(path: str, max_side: int):
    """Decode one image at a reduced DCT scale.

    ``draft`` (libjpeg scaled decode) cuts a 4000x3000 JPEG to 1/8 scale before
    the IDCT, which is several times cheaper than a full decode.  MUSIQ/NIQE
    resize to <=384 px internally, so the score is unchanged.
    """
    from PIL import Image
    try:
        with Image.open(path) as image:
            try:
                image.draft('RGB', (max_side, max_side))
            except Exception:  # noqa: BLE001
                pass
            picture = image.convert('RGB')
        if max(picture.size) > max_side:
            picture.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return picture
    except Exception:  # noqa: BLE001
        return None


def score_pyiqa(image_paths: Sequence[str], metrics: Sequence[str], device: str,
                batch_size: int = 64,
                max_side: int = MAX_SIDE) -> Dict[str, Dict[str, float]]:
    import torch
    import pyiqa
    from concurrent.futures import ThreadPoolExecutor
    from torchvision import transforms
    results: Dict[str, Dict[str, float]] = {str(p): {} for p in image_paths}
    to_tensor = transforms.ToTensor()
    # Decoding is the wall-clock cost (HDD random reads + JPEG), so decode a
    # batch of 32 with 8 threads while the GPU scores the previous batch.  PIL
    # releases the GIL in libjpeg, so threads — not processes — are enough.
    with ThreadPoolExecutor(max_workers=16) as decode_pool:
        for name in metrics:
            metric = pyiqa.create_metric(name, device=device)
            for start in range(0, len(image_paths), batch_size):
                batch = image_paths[start:start + batch_size]
                tensors = list(decode_pool.map(
                    lambda path: _load_picture(path, max_side), batch))
                valid = [i for i, t in enumerate(tensors) if t is not None]
                if not valid:
                    continue
                # Score in batches of 8.  Per-image calls put a full round-trip
                # on a GPU shared with other jobs; 8 keeps latency low while
                # staying small enough that a bad image can be isolated.
                scores: List[float] = []
                for chunk_start in range(0, len(valid), 8):
                    chunk = valid[chunk_start:chunk_start + 8]
                    try:
                        stack = torch.stack([to_tensor(tensors[i]) for i in chunk]).to(device)
                        with torch.inference_mode():
                            values = metric(stack)
                        values = values.reshape(-1).tolist()
                        if len(values) != len(chunk):
                            raise RuntimeError(f'unexpected score shape {len(values)}')
                        scores.extend(float(v) for v in values)
                    except Exception as exc:  # noqa: BLE001
                        print(f'    iqa batch fallback ({len(chunk)} images): {exc}', flush=True)
                        for i in chunk:
                            try:
                                with torch.inference_mode():
                                    scores.append(float(metric(
                                        to_tensor(tensors[i]).unsqueeze(0).to(device))))
                            except Exception as inner:  # noqa: BLE001
                                print(f'    iqa error {batch[i]}: {inner}', flush=True)
                                scores.append(float('nan'))
                for position, index in enumerate(valid):
                    if position < len(scores):
                        try:
                            results[str(batch[index])][name] = float(scores[position])
                        except Exception:  # noqa: BLE001
                            pass
                done = min(start + batch_size, len(image_paths))
                if done % 320 == 0 or done == len(image_paths):
                    print(f'  {name} {done}/{len(image_paths)}', flush=True)
                del tensors
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
    return results


def merge_cleanvision(rows: List[Dict[str, object]],
                      issue_path: Path) -> List[Dict[str, object]]:
    import pandas as pd
    if not issue_path.exists():
        for row in rows:
            row['cleanvision_flags'] = ''
        return rows
    frame = pd.read_csv(issue_path)
    column = next((c for c in ('image_path', 'index', 'Unnamed: 0') if c in frame.columns), None)
    if column is None:
        for row in rows:
            row['cleanvision_flags'] = ''
        return rows
    flags: Dict[str, Dict[str, object]] = {}
    for _, record in frame.iterrows():
        key = str(record.get(column))
        flags[key] = {k: v for k, v in record.items()
                      if k not in ('image_path', 'index', 'Unnamed: 0')
                      and str(v) == 'True' and k.startswith('is_')}
    for row in rows:
        key = str(row.get('raw_path') or row.get('path'))
        found = flags.get(key) or flags.get(Path(key).name) or {}
        row['cleanvision_flags'] = json.dumps(found, sort_keys=True)
    return rows


def main() -> int:
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--paths-csv', required=True,
                        help='CSV with a path column for decoded images of one source')
    parser.add_argument('--out', required=True)
    parser.add_argument('--sample', type=int, default=400)
    parser.add_argument('--skip-cleanvision', action='store_true')
    parser.add_argument('--skip-iqa', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--iqa-metrics', nargs='*', default=['musiq', 'niqe'])
    parser.add_argument('--max-side', type=int, default=None)
    args = parser.parse_args()

    all_rows = [dict(r) for r in csv.DictReader(open(args.paths_csv))]
    path_key = 'raw_path' if all_rows and 'raw_path' in all_rows[0] else 'path'
    rows = [r for r in all_rows if str(r.get('decode_ok')).lower() in ('1', 'true', 'true')]
    work_dir = Path(args.out).parent / 'qc' / args.source
    work_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_cleanvision and args.sample > 0:
        report = cleanvision_report([r[path_key] for r in rows], work_dir, args.sample)
        rows = merge_cleanvision(rows, work_dir / 'cleanvision_issues.csv')
        print('cleanvision', report['summary'])
    else:
        for row in rows:
            row['cleanvision_flags'] = ''

    if not args.skip_iqa:
        scores = score_pyiqa([r[path_key] for r in rows], args.iqa_metrics,
                             args.device, max_side=args.max_side or MAX_SIDE)
        for row in rows:
            found = scores.get(row[path_key], {})
            for metric, value in found.items():
                row[f'iqa_{metric}'] = value
    for row in rows:
        row.setdefault('cleanvision_flags', '')
        row.setdefault('iqa_musiq', '')
        row.setdefault('iqa_niqe', '')

    scored = {str(r.get(path_key)): r for r in rows}
    for row in all_rows:
        found = scored.get(str(row.get(path_key)))
        if found is None:
            row.setdefault('cleanvision_flags', '')
            row.setdefault('iqa_musiq', '')
            row.setdefault('iqa_niqe', '')
        else:
            for key in ('cleanvision_flags', 'iqa_musiq', 'iqa_niqe'):
                row[key] = found.get(key, '')
        row.setdefault('cleanvision_flags', '')
        row.setdefault('iqa_musiq', '')
        row.setdefault('iqa_niqe', '')
    with open(args.out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    provenance = {
        'source_id': args.source,
        'packages': package_versions(),
        'pyiqa_weights': pyiqa_weights(),
        'images_scored': len(rows),
        'images_total': len(all_rows),
        'cleanvision_sampled': min(args.sample, len(rows)),
    }
    (work_dir / f'qc_models_{args.source}.json').write_text(
        json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
