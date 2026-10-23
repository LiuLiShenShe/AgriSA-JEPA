#!/usr/bin/env python
"""Ingest one new source: decode, hash, dedupe-screen against the existing corpus.

Usage
-----
    .venv-uv/bin/python tools/stage5a_ingest_source.py \
        --source-id plantvillage --source-name "PlantVillage (colour)" \
        --root /data/datasets/PlantVillage/extracted \
        --manifest data/registry/ingested/plantvillage_manifest.csv \
        --audit reports/stage5a_plantvillage_audit.json \
        --license "CC-BY-SA-3.0-US" --source-url https://doi.org/10.5281/zenodo.1204914 \
        --archive /data/datasets/PlantVillage/plantvillage.7z
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.registry.ingest import IngestConfig, ingest, write_manifest_csv

ROOT = Path(__file__).resolve().parent.parent


def load_reference(path: Path) -> list:
    if not path.is_file():
        return []
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-id', required=True)
    parser.add_argument('--source-name', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--license', default='')
    parser.add_argument('--attribution', default='')
    parser.add_argument('--source-url', default='')
    parser.add_argument('--archive', default='')
    parser.add_argument('--reference-manifest', default='data/all_samples.csv')
    parser.add_argument('--phash-threshold', type=int, default=6)
    parser.add_argument('--dhash-threshold', type=int, default=6)
    args = parser.parse_args()

    archive_sha = ''
    if args.archive:
        from src.registry.ingest import file_sha256
        archive_sha = file_sha256(Path(args.archive))

    reference = load_reference(Path(args.reference_manifest))
    config = IngestConfig(
        source_id=args.source_id,
        source_name=args.source_name,
        root=Path(args.root),
        out_csv=Path(args.manifest),
        license=args.license,
        attribution=args.attribution,
        source_url=args.source_url,
        source_archive_sha256=archive_sha,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    result = ingest(config, reference_rows=reference)
    write_manifest_csv(Path(args.manifest), result.rows)

    audit = dict(result.audit)
    audit['archive_sha256'] = archive_sha
    audit['manifest'] = str(args.manifest)
    audit['reference_rows'] = len(reference)
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n',
                                encoding='utf-8')
    print(json.dumps({k: audit[k] for k in (
        'source_id', 'files_found', 'files_decoded', 'decode_failures',
        'exact_duplicate_pairs', 'perceptual_candidate_pairs',
        'cross_source_exact_matches', 'cross_source_candidates',
        'images_duplicate_free')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
