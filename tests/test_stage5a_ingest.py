"""Stage 5A tests: ingestion hashes, duplicate screening, D1 freeze gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.registry.freeze import (CorpusRow, filter_candidate, load_ingested,
                                 write_corpus)
from src.registry.ingest import (IngestConfig, dhash64, file_sha256, ingest,
                                 phash64, pixel_sha256, write_manifest_csv)


def _save(path: Path, array: np.ndarray, mode: str = 'RGB'):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def test_hashes_are_deterministic(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / 'a.png'
    _save(path, rng.integers(0, 255, (32, 32, 3), dtype='uint8'))
    with Image.open(path) as image:
        image.load()
        p1, d1, x1 = phash64(image), dhash64(image), pixel_sha256(image)
        p2, d2, x2 = phash64(image), dhash64(image), pixel_sha256(image)
    assert (p1, d1, x1) == (p2, d2, x2)
    assert len(p1) == 16 and len(d1) == 16 and len(x1) == 64


def test_identical_image_has_identical_pixel_hash(tmp_path):
    rng = np.random.default_rng(1)
    array = rng.integers(0, 255, (32, 32, 3), dtype='uint8')
    _save(tmp_path / 'a.png', array)
    _save(tmp_path / 'b.png', array)
    with Image.open(tmp_path / 'a.png') as a, Image.open(tmp_path / 'b.png') as b:
        assert pixel_sha256(a) == pixel_sha256(b)
        assert phash64(a) == phash64(b)


def test_ingest_detects_exact_duplicates(tmp_path):
    rng = np.random.default_rng(2)
    root = tmp_path / 'imgs'
    array = rng.integers(0, 255, (32, 32, 3), dtype='uint8')
    _save(root / 'classA' / 'one.png', array)
    _save(root / 'classA' / 'two.png', array)
    _save(root / 'classB' / 'three.png',
          rng.integers(0, 255, (32, 32, 3), dtype='uint8'))
    config = IngestConfig(source_id='s', source_name='S', root=root,
                          out_csv=tmp_path / 'm.csv')
    result = ingest(config)
    assert result.audit['files_found'] == 3
    assert result.audit['files_decoded'] == 3
    assert result.audit['exact_duplicate_pairs'] == 2  # sha256 + pixel_sha256
    assert result.audit['images_duplicate_free'] == 2


def test_ingest_records_decode_failures(tmp_path):
    root = tmp_path / 'imgs'
    root.mkdir()
    (root / 'broken.jpg').write_bytes(b'not an image')
    config = IngestConfig(source_id='s', source_name='S', root=root,
                          out_csv=tmp_path / 'm.csv')
    result = ingest(config)
    assert result.audit['files_found'] == 1
    assert result.audit['files_decoded'] == 0
    assert result.audit['decode_failures'] == 1
    assert result.rows[0]['decode_ok'] is False


def test_ingest_cross_source_match(tmp_path):
    rng = np.random.default_rng(3)
    root = tmp_path / 'imgs'
    array = rng.integers(0, 255, (32, 32, 3), dtype='uint8')
    _save(root / 'x.png', array)
    sha = file_sha256(root / 'x.png')
    reference = [{'sample_id': 'existing:1', 'sha256': sha,
                  'pixel_sha256': 'z', 'source': 'old'}]
    config = IngestConfig(source_id='s', source_name='S', root=root,
                          out_csv=tmp_path / 'm.csv')
    result = ingest(config, reference_rows=reference)
    assert result.audit['cross_source_exact_matches'] == 1
    assert result.rows[0]['duplicate_status'] == 'cross_source_duplicate'


def test_manifest_csv_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    root = tmp_path / 'imgs'
    _save(root / 'a.png', rng.integers(0, 255, (16, 16, 3), dtype='uint8'))
    config = IngestConfig(source_id='s', source_name='S', root=root,
                          out_csv=tmp_path / 'm.csv')
    result = ingest(config)
    write_manifest_csv(tmp_path / 'm.csv', result.rows)
    with open(tmp_path / 'm.csv', newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]['source_id'] == 's'
    assert rows[0]['sha256']
    assert rows[0]['decode_ok'] == 'True'


# --------------------------------------------------------------------------- #
# D1 freeze gates
# --------------------------------------------------------------------------- #

def _corpus_row(**kw):
    base = dict(sample_id='s1', source_id='ACRECropWeed', source='ACRECropWeed',
                path='/x.jpg', sha256='a' * 64, pixel_sha256='b' * 64,
                width=512, height=512, group_key='g', license='CC-BY-4.0',
                attribution='x', split_role='pretrain_train',
                review_status='ok', duplicate_status='v1_screened',
                origin_manifest='m.csv')
    base.update(kw)
    return CorpusRow(**base)


def test_filter_excludes_holdout_sources():
    rows = [_corpus_row(sample_id='x', source_id='phenobench',
                        sha256='c' * 64, pixel_sha256='d' * 64)]
    assert filter_candidate(rows) == []


def test_filter_excludes_exact_duplicates():
    rows = [_corpus_row(), _corpus_row(sample_id='s2', sha256='a' * 64,
                                       pixel_sha256='b' * 64)]
    kept = filter_candidate(rows)
    assert len(kept) == 1


def test_filter_excludes_cross_source_duplicates():
    rows = [_corpus_row(duplicate_status='cross_source_duplicate')]
    assert filter_candidate(rows) == []


def test_filter_allows_pending_perceptual_rows():
    # Screening candidates are output for review, not silently dropped.
    rows = [_corpus_row(review_status='pending_perceptual_review')]
    assert len(filter_candidate(rows)) == 1


def test_write_corpus_marks_candidate_and_hashes(tmp_path):
    rows = [_corpus_row()]
    digest = write_corpus(tmp_path / 'd1.csv', rows)
    manifest = json.loads((tmp_path / 'd1.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'CANDIDATE'
    assert manifest['approved'] is False
    assert manifest['sha256_canonical_csv'] == digest
    assert manifest['images'] == 1


def test_write_corpus_is_deterministic(tmp_path):
    rows = [_corpus_row()]
    d1 = write_corpus(tmp_path / 'a.csv', rows)
    d2 = write_corpus(tmp_path / 'b.csv', rows)
    assert d1 == d2


def test_load_ingested_produces_corpus_rows(tmp_path):
    path = tmp_path / 'm.csv'
    path.write_text(
        'sample_id,source_id,source_name,rel_path,path,class_label,group_key,'
        'width,height,mode,format,bytes,sha256,pixel_sha256,phash64,dhash64,'
        'decode_ok,decode_error,duplicate_status,duplicate_of,license,'
        'attribution,source_url,source_archive_sha256\n'
        's1,pv,PV,r/a.jpg,/x/a.jpg,cls,g,32,32,RGB,JPEG,10,'
        + 'a' * 64 + ',' + 'b' * 64 + ',0,0,True,,unique,,CC-BY-4.0,attr,url,\n',
        encoding='utf-8')
    rows = load_ingested(path, 'pv')
    assert len(rows) == 1
    assert rows[0].source_id == 'pv'
    assert rows[0].review_status == 'pending_perceptual_review'
