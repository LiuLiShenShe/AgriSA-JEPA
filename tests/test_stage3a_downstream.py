"""Stage 3A downstream tests: PhenoBench data, tiling, probe, official eval format.

The PhenoBench fixtures here are tiny synthetic images written in the official
directory layout (``<split>/images`` + ``<split>/semantics``).  They exercise the
loader, label handling, tiling/stitching, probe forward/backward and the
prediction-PNG format without downloading the 7.6 GB dataset.  Anything that
needs the real dataset is skipped unless ``AGRISA_PHENOBENCH`` points at it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.downstream.features import (
    EMBED_DIM,
    TOKEN_GRID,
    TOKENS_PER_TILE,
    arm_config,
    build_encoder,
    extract_tile_features,
    load_backbone_state,
    normalize_tiles,
)
from src.downstream.phenobench_data import (
    CLASS_NAMES,
    NUM_CLASSES,
    PhenoBenchSemantic,
    dataset_summary,
    list_split_files,
    load_image_and_label,
    map_partial_to_parent,
)
from src.downstream.probe import (
    IoUAccumulator,
    LinearSegmentationProbe,
    count_parameters,
    patch_ce_loss,
    patch_class_counts,
    train_probe,
)
from src.downstream.tiling import (
    TILE_SIZE,
    covered_mask,
    covered_region,
    crop_fraction,
    n_tiles_per_axis,
    stitch,
    stitch_logits,
    tile_coords,
    tile_image,
    tile_label,
    tile_offsets,
)

PHENOBENCH_ENV = 'AGRISA_PHENOBENCH'
VITH_ENV = 'AGRISA_VITH_CHECKPOINT'
ARMB_CHECKPOINT = Path('runs/stage3_armB_pilot_500/jepa-latest.pth.tar')


# --------------------------------------------------------------------------- #
# Fixtures: a tiny synthetic PhenoBench in the official layout
# --------------------------------------------------------------------------- #

def _write_phenobench(root: Path, sizes=(1024,), splits=('train', 'val'),
                      n_per_split=2) -> Path:
    from PIL import Image
    rng = np.random.default_rng(0)
    for split in splits:
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'semantics').mkdir(parents=True, exist_ok=True)
        for size in sizes:
            for i in range(n_per_split):
                name = f'{size}_{split}_{i:03d}.png'
                image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
                Image.fromarray(image).save(root / split / 'images' / name)
                label = np.zeros((size, size), dtype=np.uint8)
                label[:, : size // 2] = 1        # crop
                label[:, size // 2:] = 2         # weed
                label[: size // 8] = 3           # partial crop
                Image.fromarray(label, mode='L').save(root / split / 'semantics' / name)
    return root


@pytest.fixture(scope='module')
def synthetic_root(tmp_path_factory):
    return _write_phenobench(tmp_path_factory.mktemp('phenobench'))


# --------------------------------------------------------------------------- #
# 1. PhenoBench loading
# --------------------------------------------------------------------------- #

class TestPhenoBenchData:

    def test_list_and_load(self, synthetic_root):
        names = list_split_files(synthetic_root, 'val')
        assert names == sorted(names)
        assert len(names) == 2
        image, label = load_image_and_label(synthetic_root, 'val', names[0])
        assert image.shape == (1024, 1024, 3) and image.dtype == np.uint8
        assert label.shape == (1024, 1024)
        assert set(np.unique(label)) <= {0, 1, 2, 3}

    def test_dataset_index(self, synthetic_root):
        ds = PhenoBenchSemantic(synthetic_root, 'train')
        assert len(ds) == 2
        sample = ds[0]
        assert set(sample) == {'name', 'image', 'label'}

    def test_missing_split_rejected(self, synthetic_root):
        with pytest.raises(ValueError, match='train.*val|val.*train'):
            PhenoBenchSemantic(synthetic_root, 'test')

    def test_summary_counts(self, synthetic_root):
        summary = dataset_summary(synthetic_root)
        assert summary['splits']['train']['images'] == 2
        assert summary['splits']['train']['semantics'] == 2


class TestSemanticLabelIds:

    def test_partial_maps_to_parent(self):
        label = np.array([[0, 1, 2, 3, 4]], dtype=np.int16)
        mapped = map_partial_to_parent(label)
        assert mapped.tolist() == [[0, 1, 2, 1, 2]]

    def test_mapping_does_not_alias_soil(self):
        label = np.array([[3, 4]], dtype=np.int16)
        assert (map_partial_to_parent(label) == 0).sum() == 0

    def test_num_classes_matches_official(self):
        assert NUM_CLASSES == 3
        assert CLASS_NAMES == ('soil', 'crop', 'weed')

    def test_mapping_matches_official_devkit(self):
        # cross-check against the installed official implementation
        convert = pytest.importorskip('phenobench.evaluation.auxiliary.convert')
        raw = torch.arange(0, 5, dtype=torch.int32).view(1, 1, 5)
        expected = convert.convert_partial_semantics(raw).numpy().ravel()
        actual = map_partial_to_parent(np.arange(0, 5, dtype=np.int16))
        assert actual.tolist() == expected.tolist()


# --------------------------------------------------------------------------- #
# 2. Tiling / padding / stitching
# --------------------------------------------------------------------------- #

class TestTiling:

    def test_1024_yields_5x5_full_coverage_tiles(self):
        tiles = tile_coords(1024, 1024)
        assert len(tiles) == 25
        assert tile_offsets(1024) == [0, 200, 400, 600, 800]
        assert tiles[0] == tiles[0].__class__(0, 0, 224, 224)
        assert tiles[-1].bottom == 1024 and tiles[-1].right == 1024

    def test_tiles_never_go_out_of_bounds(self):
        for t in tile_coords(1024, 1024):
            assert 0 <= t.top and t.bottom <= 1024
            assert 0 <= t.left and t.right <= 1024
            assert t.height == t.width == TILE_SIZE

    def test_every_pixel_is_covered(self):
        assert covered_region(1024, 1024) == (1024, 1024)
        mask = covered_mask(1024, 1024)
        assert mask.all(), 'no pixel may be left unpredicted'

    def test_offsets_are_uniform_and_gapless(self):
        for size in (1024, 1000, 768, 224, 500):
            offsets = tile_offsets(size)
            assert offsets[0] == 0 and offsets[-1] == size - TILE_SIZE
            for a, b in zip(offsets, offsets[1:]):
                assert b > a
                assert b <= a + TILE_SIZE, 'a gap would leave pixels uncovered'

    def test_no_padding_and_no_crop(self):
        # the grid must not extend past the image (no padding) and must reach it
        for size in (1024, 1000):
            offsets = tile_offsets(size)
            assert offsets[-1] + TILE_SIZE == size

    def test_pixels_below_tile_are_rejected(self):
        with pytest.raises(ValueError, match='smaller than one'):
            tile_coords(100, 1024)

    def test_overlap_fraction_is_small(self):
        assert 0.0 < crop_fraction(1024, 1024) < 0.25

    def test_tile_counts_per_axis(self):
        assert n_tiles_per_axis(1024) == 5
        assert n_tiles_per_axis(1000) == 5
        assert n_tiles_per_axis(448) == 2

    def test_tile_image_and_label_shapes(self):
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        label = np.random.randint(0, 3, (1024, 1024)).astype(np.int16)
        tiles = tile_coords(1024, 1024)
        assert tile_image(image, tiles).shape == (25, 224, 224, 3)
        assert tile_label(label, tiles).shape == (25, 224, 224)


class TestStitching:

    def test_hard_round_trip_identity(self):
        # each tile repeats the full label, so every placement is exact
        label = np.random.randint(0, 3, (1024, 1024)).astype(np.int16)
        tiles = tile_coords(1024, 1024)
        pieces = tile_label(label, tiles)
        stitched = stitch(pieces, 1024, 1024, tiles)
        assert np.array_equal(stitched, label)

    def test_logit_stitch_reproduces_label(self):
        tiles = tile_coords(1024, 1024)
        label = np.random.randint(0, 3, (1024, 1024))
        onehot = np.zeros((len(tiles), 3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
        for i, t in enumerate(tiles):
            piece = label[t.top:t.bottom, t.left:t.right]
            for c in range(3):
                onehot[i, c] = (piece == c) * 10.0
        stitched = stitch_logits(onehot, 1024, 1024, tiles)
        assert np.array_equal(stitched, label.astype(np.int16))

    def test_logit_stitch_averages_overlaps(self):
        tiles = tile_coords(1024, 1024)
        # tile 0 votes class 0, every other tile votes class 2
        logits = np.zeros((len(tiles), 3, TILE_SIZE, TILE_SIZE), dtype=np.float32)
        logits[:, 2] = 1.0
        logits[0, 0] = 5.0
        out = stitch_logits(logits, 1024, 1024, tiles)
        # overlap region covered by both tile 0 and a neighbour is averaged
        assert out[0, 0] in (0, 2)
        assert out[500, 500] == 2

    def test_stitch_rejects_wrong_tile_shape(self):
        tiles = tile_coords(1024, 1024)
        bad = np.zeros((len(tiles), 100, 100), dtype=np.int16)
        with pytest.raises(ValueError, match='expected'):
            stitch(bad, 1024, 1024, tiles)

    def test_stitch_predictions_match_official_filenames(self, synthetic_root):
        ds = PhenoBenchSemantic(synthetic_root, 'val')
        tiles = tile_coords(1024, 1024)
        preds = {name: stitch(np.zeros((len(tiles), 224, 224), dtype=np.int16),
                              1024, 1024, tiles)
                 for name in ds.filenames}
        assert set(preds) == set(ds.filenames)


# --------------------------------------------------------------------------- #
# 3. ViT-H token geometry and feature extraction
# --------------------------------------------------------------------------- #

class TestViTHTokens:

    def test_token_shape_constants(self):
        assert TOKEN_GRID == 16
        assert TOKENS_PER_TILE == 256
        assert EMBED_DIM == 1280

    def test_encoder_output_shape_cpu(self):
        from src.models.vision_transformer import vit_huge
        encoder = vit_huge(img_size=[TILE_SIZE], patch_size=14).eval()
        with torch.no_grad():
            out = encoder(torch.randn(2, 3, TILE_SIZE, TILE_SIZE))
        assert out.shape == (2, TOKENS_PER_TILE, EMBED_DIM)

    def test_normalization_matches_imagenet_constants(self):
        tiles = np.full((2, TILE_SIZE, TILE_SIZE, 3), 128, dtype=np.uint8)
        x = normalize_tiles(tiles)
        assert x.shape == (2, 3, TILE_SIZE, TILE_SIZE)
        expected = [(128 / 255 - m) / s for m, s in
                    zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        for c, value in enumerate(expected):
            assert torch.allclose(x[:, c], torch.full_like(x[:, c], value), atol=1e-5)
        # and it is not a no-op rescale
        assert not torch.allclose(x[:, 0], x[:, 1])

    def test_extract_features_shape_and_freeness(self):
        from src.models.vision_transformer import vit_huge
        encoder = vit_huge(img_size=[TILE_SIZE], patch_size=14).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        tiles = np.random.randint(0, 255, (3, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        feats = extract_tile_features(encoder, tiles, torch.device('cpu'),
                                      batch_size=2, use_bfloat16=False)
        assert feats.shape == (3, TOKENS_PER_TILE, EMBED_DIM)
        grid = extract_tile_features(encoder, tiles, torch.device('cpu'),
                                     batch_size=2, use_bfloat16=False, return_grid=True)
        assert grid.shape == (3, TOKEN_GRID, TOKEN_GRID, EMBED_DIM)

    def test_extract_rejects_bad_tile_shape(self):
        from src.models.vision_transformer import vit_huge
        encoder = vit_huge(img_size=[TILE_SIZE], patch_size=14).eval()
        with pytest.raises(ValueError, match='Expected'):
            extract_tile_features(encoder, np.zeros((2, 100, 100, 3), dtype=np.uint8),
                                  torch.device('cpu'), use_bfloat16=False)


class TestArmLoading:

    def test_arm_configs_distinct_checkpoints(self):
        a, b = arm_config('A'), arm_config('B')
        assert a['checkpoint'] != b['checkpoint']
        assert a['continued_training'] == 'none'
        assert '500' in b['continued_training']

    def test_unknown_arm_rejected(self):
        with pytest.raises(ValueError, match="Unknown arm"):
            arm_config('C')

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_backbone_state(tmp_path / 'nope.pth.tar')

    @pytest.mark.skipif(not ARMB_CHECKPOINT.is_file(),
                        reason='Arm B pilot checkpoint not present')
    def test_armB_weights_load_exactly(self):
        state = load_backbone_state(ARMB_CHECKPOINT)
        from src.pretrained import compare_state_dicts
        from src.models.vision_transformer import vit_huge
        model = vit_huge(img_size=[TILE_SIZE], patch_size=14)
        diff = compare_state_dicts(model.state_dict(), state)
        assert diff['is_exact'], diff

    @pytest.mark.skipif(not os.environ.get(VITH_ENV),
                        reason=f'{VITH_ENV} not set')
    def test_armA_weights_load_exactly(self):
        state = load_backbone_state(os.environ[VITH_ENV])
        from src.pretrained import compare_state_dicts
        from src.models.vision_transformer import vit_huge
        model = vit_huge(img_size=[TILE_SIZE], patch_size=14)
        diff = compare_state_dicts(model.state_dict(), state)
        assert diff['is_exact'], diff


# --------------------------------------------------------------------------- #
# 4. Linear probe forward / backward
# --------------------------------------------------------------------------- #

class TestLinearProbe:

    def test_forward_shapes(self):
        probe = LinearSegmentationProbe()
        logits = probe(torch.randn(4, TOKENS_PER_TILE, EMBED_DIM))
        assert logits.shape == (4, 3, TOKEN_GRID, TOKEN_GRID)

    def test_parameter_count(self):
        probe = LinearSegmentationProbe()
        # 1280*3 weights + 3 biases
        assert count_parameters(probe) == 1280 * 3 + 3

    def test_predict_tiles_full_resolution(self):
        probe = LinearSegmentationProbe().eval()
        preds = probe.predict_tiles(torch.randn(2, TOKENS_PER_TILE, EMBED_DIM))
        assert preds.shape == (2, TILE_SIZE, TILE_SIZE)

    def test_forward_rejects_bad_input(self):
        probe = LinearSegmentationProbe()
        with pytest.raises(ValueError, match='tokens'):
            probe(torch.randn(4, EMBED_DIM))

    def test_patch_class_counts_sum_to_pixels(self):
        label = np.random.default_rng(0).integers(0, 3, (2, TILE_SIZE, TILE_SIZE))
        counts = patch_class_counts(label)
        assert counts.shape == (2, TOKEN_GRID, TOKEN_GRID, 3)
        assert counts.sum(axis=(1, 2, 3)).tolist() == [
            TILE_SIZE * TILE_SIZE] * 2
        for c in range(3):
            assert counts[..., c].sum() == int((label == c).sum())

    def test_patch_class_counts_rejects_indivisible_tile(self):
        with pytest.raises(ValueError, match='divisible'):
            patch_class_counts(np.zeros((1, 100, 100), dtype=np.int64))

    def test_patch_loss_equals_upsampled_pixel_loss(self):
        # the exactness property the whole probe depends on
        torch.manual_seed(0)
        label = torch.randint(0, 3, (2, TILE_SIZE, TILE_SIZE))
        logits = torch.randn(2, 3, TOKEN_GRID, TOKEN_GRID, requires_grad=True)
        counts = torch.from_numpy(patch_class_counts(label.numpy()))

        fast = patch_ce_loss(logits, counts)
        up = torch.nn.functional.interpolate(
            logits, size=(TILE_SIZE, TILE_SIZE), mode='nearest')
        slow = torch.nn.functional.cross_entropy(up, label)
        assert torch.allclose(fast, slow, atol=1e-5), (float(fast), float(slow))

        fast.backward()
        grad_fast = logits.grad.clone()
        logits.grad = None
        slow.backward()
        assert torch.allclose(grad_fast, logits.grad, atol=1e-5)

    def test_coverage_masking_changes_loss(self):
        logits = torch.randn(1, 3, TOKEN_GRID, TOKEN_GRID)
        label = np.zeros((1, TILE_SIZE, TILE_SIZE), dtype=np.int64)
        label[:, :, :TILE_SIZE // 2] = 1
        counts = torch.from_numpy(patch_class_counts(label))
        full = patch_ce_loss(logits, counts)
        coverage = torch.ones(1, TOKEN_GRID, TOKEN_GRID)
        coverage[:, :, TOKEN_GRID // 2:] = 0        # hide the crop half
        masked = patch_ce_loss(logits, counts, coverage)
        assert float(full) != float(masked)

    def test_training_reduces_loss(self):
        # a tiny, linearly separable problem: the probe must fit it
        torch.manual_seed(0)
        n = 24
        tokens = torch.randn(n, TOKENS_PER_TILE, EMBED_DIM)
        labels = torch.zeros(n, TILE_SIZE, TILE_SIZE, dtype=torch.int64)
        # class is a linear function of token 0's first 3 dims -> learnable
        weights = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        scores = tokens[:, 0, :3] @ weights.T
        cls = scores.argmax(dim=1)
        labels[:] = cls.view(-1, 1, 1)
        counts = patch_class_counts(labels.numpy())
        probe = LinearSegmentationProbe()
        out = train_probe(probe, tokens, torch.from_numpy(counts), epochs=3,
                          batch_size=8, lr=0.05, seed=0)
        assert out['loss_history'][-1] < out['loss_history'][0]
        assert out['updates'] == 3 * (n // 8)

    def test_fully_masked_loss_is_finite(self):
        logits = torch.randn(1, 3, TOKEN_GRID, TOKEN_GRID)
        label = np.zeros((1, TILE_SIZE, TILE_SIZE), dtype=np.int64)
        counts = torch.from_numpy(patch_class_counts(label))
        loss = patch_ce_loss(logits, counts, coverage=torch.zeros(1, TOKEN_GRID, TOKEN_GRID))
        assert torch.isfinite(loss)


class TestIoUAccumulator:

    def test_perfect_prediction(self):
        acc = IoUAccumulator(3)
        label = np.array([[0, 1, 2]])
        acc.update(label.copy(), label)
        assert acc.compute() == {0: 1.0, 1: 1.0, 2: 1.0}

    def test_absent_class_counts_one(self):
        acc = IoUAccumulator(3)
        label = np.array([[0, 0]])
        acc.update(label.copy(), label)
        # crop/weed never appear in either -> official convention gives 1.0
        assert acc.compute()[1] == 1.0

    def test_valid_mask_excludes_border(self):
        acc_full = IoUAccumulator(3)
        acc_masked = IoUAccumulator(3)
        label = np.array([[0, 1, 2, 2]])
        pred = np.array([[0, 1, 2, 0]])
        acc_full.update(pred, label)
        # exclude the last pixel, where the prediction is wrong
        valid = np.array([[True, True, True, False]])
        acc_masked.update(pred, label, valid)
        assert acc_full.union[2] == 2      # two weed true-or-pred pixels
        assert acc_masked.union[2] == 1    # the wrong one was excluded
        assert acc_masked.intersection[2] == 1

    def test_matches_torchmetrics(self):
        tm = pytest.importorskip('torchmetrics')
        from torchmetrics.classification import MulticlassJaccardIndex
        rng = np.random.default_rng(0)
        pred = rng.integers(0, 3, (64, 64))
        true = rng.integers(0, 3, (64, 64))
        acc = IoUAccumulator(3)
        acc.update(pred, true)
        ours = acc.compute()
        metric = MulticlassJaccardIndex(num_classes=3, average=None)
        metric.update(torch.from_numpy(pred), torch.from_numpy(true))
        official = metric.compute().numpy()
        for c in range(3):
            assert abs(ours[c] - float(official[c])) < 1e-6


# --------------------------------------------------------------------------- #
# 5. Real dataset integration (skipped unless AGRISA_PHENOBENCH is set)
# --------------------------------------------------------------------------- #

REAL_ROOT = os.environ.get(PHENOBENCH_ENV)
needs_real_data = pytest.mark.skipif(
    not REAL_ROOT or not Path(REAL_ROOT).is_dir(),
    reason=f'{PHENOBENCH_ENV} is not set to a PhenoBench root')


@needs_real_data
class TestRealPhenoBench:

    def test_splits_load(self):
        root = Path(REAL_ROOT)
        for split in ('train', 'val'):
            names = list_split_files(root, split)
            assert names, split
            _, label = load_image_and_label(root, split, names[0])
            assert set(np.unique(label)) <= {0, 1, 2, 3, 4}

    def test_label_ids_are_official(self):
        root = Path(REAL_ROOT)
        hist = PhenoBenchSemantic(root, 'val').label_histogram(limit=5)
        assert set(hist) <= {0, 1, 2, 3, 4}