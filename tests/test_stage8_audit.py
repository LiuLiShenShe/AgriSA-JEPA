"""Stage 8 tests: audit cross-check logic (no GPUs, no large downloads).

``crosscheck`` must classify a dataset as leaking when either its raw or its
pixel SHA-256 appears in the pretraining sets, and clean otherwise.  BLOCKED
datasets (no local images) must short-circuit audit_dataset without touching
the image directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.stage8_audit_datasets import (DatasetSpec, audit_dataset,
                                         collect_images, crosscheck)

ROOT = Path(__file__).resolve().parent.parent


def test_crosscheck_clean():
    rows = [
        {'relative_path': 'a.jpg', 'raw_sha256': 'a1', 'pixel_sha256': 'p1',
         'decode_ok': True},
        {'relative_path': 'b.jpg', 'raw_sha256': 'a2', 'pixel_sha256': 'p2',
         'decode_ok': True},
    ]
    res = crosscheck(rows, {'zz'}, {'zz'})
    assert res['verdict'] == 'CLEAN_exact'
    assert res['n_exact_raw_matches'] == 0
    assert res['n_exact_pixel_matches'] == 0


def test_crosscheck_raw_leak():
    rows = [
        {'relative_path': 'a.jpg', 'raw_sha256': 'leakraw',
         'pixel_sha256': 'p1', 'decode_ok': True},
    ]
    res = crosscheck(rows, {'leakraw'}, {'nope'})
    assert res['verdict'] == 'LEAKAGE'
    assert res['n_exact_raw_matches'] == 1
    assert res['exact_raw_matches'] == ['a.jpg']


def test_crosscheck_pixel_leak():
    rows = [
        {'relative_path': 'a.jpg', 'raw_sha256': 'r1',
         'pixel_sha256': 'leakpix', 'decode_ok': True},
    ]
    res = crosscheck(rows, {'nope'}, {'leakpix'})
    assert res['verdict'] == 'LEAKAGE'
    assert res['n_exact_pixel_matches'] == 1


def test_crosscheck_undecoded_skipped():
    rows = [
        {'relative_path': 'broken.jpg', 'raw_sha256': '', 'pixel_sha256': '',
         'decode_ok': False},
    ]
    res = crosscheck(rows, {'x'}, {'x'})
    assert res['n_decode_ok'] == 0
    assert res['verdict'] == 'CLEAN_exact'


def test_audit_blocked_untouched(tmp_path: Path):
    spec = DatasetSpec('x', 'detection', None, None, 'n/a', 'n/a',
                       'CC-BY-4.0', 'src', 'none')
    res = audit_dataset(spec, workers=2, pixel_hashes=True)
    assert res['status'] == 'BLOCKED'
    assert res['status_reason'] == 'not_downloaded'
    assert res['n_images'] == 0


def test_collect_images_filters_by_extension(tmp_path: Path):
    (tmp_path / 'a.jpg').write_bytes(b'x')
    (tmp_path / 'b.txt').write_bytes(b'y')
    (tmp_path / 'c.PNG').write_bytes(b'z')
    got = collect_images(tmp_path)
    assert [p.suffix.lower() for p in got] == ['.jpg', '.png']


def test_audit_dataset_creates_cache(tmp_path: Path):
    from PIL import Image
    img = Image.new('RGB', (4, 4), (10, 200, 30))
    img.save(tmp_path / 'im.jpg')
    spec = DatasetSpec('cachetest', 'classification', str(tmp_path), None,
                       'n/a', 'n/a', 'CC-BY-4.0', 'src', 'none')
    res = audit_dataset(spec, workers=1, pixel_hashes=True)
    assert res['n_images'] == 1
    assert res['n_decode_ok'] == 1
    assert res['rows'][0]['width'] == 4
    assert res['rows'][0]['raw_sha256']
    assert res['rows'][0]['pixel_sha256']
    cache = ROOT / 'data' / 'stage8_audit_cache' / 'cachetest.json'
    assert cache.is_file()
    cache.unlink()  # do not leak the fixture into the repository cache


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))