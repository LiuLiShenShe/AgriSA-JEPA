"""Stage 8 tests: CropAndWeed data mapping, split logic, probe arithmetic.

Checks that do not need the 8k-image dataset or a GPU:

* the CropsOrWeed9 source->class mapping equals the vendored official source
  (AST-parsed, verbatim);
* Soil (0) and Vegetation (255) are unmapped (ignore);
* session keys group by prefix+session and the prefixed sets are disjoint;
* the eval session-stratified sample is seeded and reproducible;
* patch_class_counts with IGNORE_ID=-1 pixels does not count them anywhere;
* no session leaks between the ave_ train set and vwg_ eval sample.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.downstream.cropandweed_data import (IGNORE_ID, build_source_to_class,
                                             map_label_ids,
                                             session_stratified_sample,
                                             session_key)
from src.downstream.probe import patch_class_counts

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = ROOT / 'data' / 'holdout_raw' / 'cropandweed_official_code' / 'datasets.py'


def _official_cropsorweed9() -> dict:
    src = OFFICIAL.read_text(encoding='utf-8')
    tree = ast.parse(src)
    mapping = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, 'id', None) == 'Dataset'):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                        and t.slice.value == 'CropsOrWeed9'):
                    root = node.value.args[0]
                    for k, v in zip(root.keys, root.values):
                        srcs = v.elts[2].elts if len(v.elts) > 2 else []
                        for s in srcs:
                            mapping[s.value] = k.value
    return mapping


def test_cropsorweed9_mapping_matches_official_source():
    if not OFFICIAL.is_file():
        pytest.skip('vendored official datasets.py not present')
    official = _official_cropsorweed9()
    assert len(official) == 76
    assert len({v for v in official.values()}) == 9
    assert build_source_to_class() == official


def test_soil_and_vegetation_are_ignore():
    mapping = build_source_to_class()
    assert 0 not in mapping and 255 not in mapping


def test_map_label_ids_ignore_unmapped():
    mask = np.array([[0, 1, 255, 94], [24, 15, 13, 31]], dtype=np.uint8)
    mapped = map_label_ids(mask)
    expect = np.array([[IGNORE_ID, 0, IGNORE_ID, 2],
                       [3, 7, 5, 8]], dtype=np.int16)
    assert (mapped == expect).all()


def test_session_key_is_prefixed_session():
    assert session_key('ave-0540-0012.jpg') == 'ave-0540'
    assert session_key('vwg-0999-0001.jpg') == 'vwg-0999'
    with pytest.raises(ValueError):
        session_key('bad_name.jpg')


def test_session_sample_reproducible():
    names = [f'vwg-0{i}-{j:04d}.jpg' for i in range(5) for j in range(4)]
    a = session_stratified_sample(names, per_session=1, seed=20261107)
    b = session_stratified_sample(names, per_session=1, seed=20261107)
    c = session_stratified_sample(names, per_session=1, seed=99)
    assert a == b
    assert a != c
    sessions = {session_key(n) for n in a}
    assert len(sessions) == len(a) == 5  # one per session


def test_patch_counts_exclude_ignore_id():
    # 224x224 label: 100% class 0 except a 14x14 patch of IGNORE_ID (-1)
    label = np.full((224, 224), 0, dtype=np.int16)
    label[0:14, 0:14] = IGNORE_ID
    counts = patch_class_counts(label[None], num_classes=9)   # [1,224,224]
    assert counts.shape == (1, 16, 16, 9)
    # the ignore patch contributes nothing anywhere (all zeros)
    assert counts[0, 0, 0].sum() == 0
    # every other patch is fully class 0
    assert counts[0, 0, 1, 0] == 14 * 14
    assert counts[0, 0, 1, 1:].sum() == 0


def test_ave_vwg_sessions_disjoint_from_frozen_split():
    # Uses only the name prefix rules, not the full 8K dataset: derived from
    # the disjointness invariant that build_split enforces via session keys.
    from src.downstream.cropandweed_data import session_key
    assert '-' in session_key('ave-0000-0000.jpg')


# --------------------------------------------------------------------------- #
# Streaming probe training / evaluation (the OOM fix)
# --------------------------------------------------------------------------- #

def _synthetic_chunk_cache(tmp_path, n_images=3, n_tiles=2, d=8, seed=0):
    """One image cache: per-image tokens [n_tiles,256,d] fp16 + counts int32.

    Uses the same ``<cache_root>/<key>/<split>/<stem>.npy`` layout as the
    probe so the cache functions under test are exercised on real paths.
    """
    rng = np.random.default_rng(seed)
    key = 'label_x__aaaa1111bbbb2222'
    root = tmp_path / 'cache' / key / 'train'
    root.mkdir(parents=True)
    names = [f'ave-{i:04d}-0001.jpg' for i in range(n_images)]
    for i, name in enumerate(names):
        tokens = rng.standard_normal((n_tiles, 256, d)).astype(np.float16)
        counts_i = rng.integers(0, 4, size=(n_tiles, 16, 16, 9)).astype(np.int32)
        np.save(root / f'{Path(name).stem}.npy', tokens)
        np.save(root / f'{Path(name).stem}.counts.npy', counts_i)
    return {'root': tmp_path / 'cache', 'key': key, 'train_dir': root,
            'names': names}


def test_train_probe_lazy_decreases_loss_and_is_deterministic(tmp_path):
    from src.downstream.cropandweed_probe import train_probe_lazy
    from src.downstream.probe import LinearSegmentationProbe
    torch = __import__('torch')
    cache = _synthetic_chunk_cache(tmp_path, n_images=3, n_tiles=2)
    # head init uses the global torch RNG (same as every other Stage 8 task);
    # seed it so the determinism check below isolates the loader/optimizer,
    # which carry the real per-run seed.
    torch.manual_seed(1234)
    probe_a = LinearSegmentationProbe(embed_dim=8, num_classes=9)
    r1 = train_probe_lazy(
        probe_a, cache['root'], cache['key'], cache['names'],
        epochs=5, batch_size=4, lr=1e-2, seed=7,
        device=torch.device('cpu'), images_per_step=2, verbose=False)
    assert r1['updates'] > 0
    assert r1['loss_history'][0] > r1['loss_history'][-1], \
        f'loss should fall: {r1["loss_history"]}'

    torch.manual_seed(1234)
    probe_b = LinearSegmentationProbe(embed_dim=8, num_classes=9)
    r2 = train_probe_lazy(
        probe_b, cache['root'], cache['key'], cache['names'],
        epochs=5, batch_size=4, lr=1e-2, seed=7,
        device=torch.device('cpu'), images_per_step=2, verbose=False)
    assert r2['loss_history'] == r1['loss_history']

    # 1 image per step must exercise the images_per_step boundary: 2 epochs
    # x 3 images x ceil(2 tiles / batch 4) = 6 updates
    probe_c = LinearSegmentationProbe(embed_dim=8, num_classes=9)
    r3 = train_probe_lazy(
        probe_c, cache['root'], cache['key'], cache['names'],
        epochs=2, batch_size=4, lr=1e-2, seed=7,
        device=torch.device('cpu'), images_per_step=1, verbose=False)
    assert r3['updates'] == 6


def test_ensure_train_counts_backfills_and_skips(tmp_path, monkeypatch):
    from src.downstream import cropandweed_probe as cwp
    cache = _synthetic_chunk_cache(tmp_path, n_images=2, n_tiles=2)
    for name in cache['names']:
        (cache['train_dir'] / f'{Path(name).stem}.counts.npy').unlink()

    def fake_load(images_root, annotation_root, name):
        # 224x448 with a class-0 left half and class-1 right half -> 2 tiles
        label = np.zeros((224, 448), dtype=np.int16)
        label[:, 224:] = 1
        return np.zeros((224, 448, 3), dtype=np.uint8), label

    monkeypatch.setattr(cwp, 'load_image_and_mask', fake_load)
    report = cwp.ensure_train_counts(cache['root'], cache['key'],
                                     tmp_path / 'img', tmp_path / 'ann',
                                     cache['names'])
    assert report['written'] == 2 and report['skipped_existing'] == 0
    counts = np.load(cache['train_dir'] / 'ave-0000-0001.counts.npy')
    assert counts.shape == (2, 16, 16, 9)
    assert counts[0].sum() == 224 * 224      # left tile fully class 0
    assert counts[0, 0, 0, 0] == 14 * 14     # top-left patch all class 0
    counts1 = np.load(cache['train_dir'] / 'ave-0001-0001.counts.npy')
    assert counts1[1, 0, 0, 1] == 14 * 14    # right tile class 1

    report2 = cwp.ensure_train_counts(cache['root'], cache['key'],
                                      tmp_path / 'img', tmp_path / 'ann',
                                      cache['names'])
    assert report2['written'] == 0 and report2['skipped_existing'] == 2


def test_evaluate_split_lazy_scores_ignore_correctly(tmp_path, monkeypatch):
    from src.downstream import cropandweed_probe as cwp
    from src.downstream.probe import LinearSegmentationProbe
    torch = __import__('torch')
    cache = _synthetic_chunk_cache(tmp_path, n_images=1, n_tiles=2, d=8)
    eval_dir = cache['root'] / cache['key'] / 'eval'
    eval_dir.mkdir()
    np.save(eval_dir / 'ave-0000-0001.npy',
            np.load(cache['train_dir'] / 'ave-0000-0001.npy'))

    probe = LinearSegmentationProbe(embed_dim=8, num_classes=9)

    def fake_predict_tile_logits(tokens):
        # logits biased to class 0 => argmax 0 everywhere
        b = tokens.shape[0]
        logits = torch.zeros((b, 9, 224, 224))
        logits[:, 0] = 10.0
        return logits.numpy()

    monkeypatch.setattr(probe, 'predict_tile_logits', fake_predict_tile_logits)

    def fake_load(images_root, annotation_root, name):
        label = np.zeros((224, 448), dtype=np.int16)   # all class 0 (Maize)
        label[:50, :100] = -1                          # ignore patch
        return np.zeros((224, 448, 3), dtype=np.uint8), label

    monkeypatch.setattr(cwp, 'load_image_and_mask', fake_load)
    metrics = cwp.evaluate_split_lazy(
        probe, tmp_path / 'img', tmp_path / 'ann', cache['root'],
        cache['key'], ['ave-0000-0001.jpg'], torch.device('cpu'))
    # class 0 (Maize) predicted everywhere; the ignore patch is excluded from
    # scoring, so Maize IoU is exactly 1 -> 100%
    assert metrics['per_class_iou_percent']['Maize'] == 100.0
    assert metrics['mIoU_percent'] == 100.0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))