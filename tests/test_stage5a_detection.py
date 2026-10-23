"""Stage 5A tests: GWHD labels/splits, ViT detection backbone, coordinate mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.downstream.gwhd_data import (GWHDDataset, GWHDSplit, clip_boxes,
                                      image_group, keep_valid_boxes, parse_boxes,
                                      read_split, split_isolation_report,
                                      validate_boxes)
from src.downstream.vit_detection import (SquareResizeTransform, ViTFPNBackbone,
                                          resize_pos_embed, trainable_parameter_count)
from src.models.vision_transformer import vit_tiny


# --------------------------------------------------------------------------- #
# box parsing / validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('value,expected', [
    ('[[1, 2, 3, 4]]', [(1, 2, 3, 4)]),
    ('[[1,2,3,4],[5,6,7,8]]', [(1, 2, 3, 4), (5, 6, 7, 8)]),
    ('1 2 3 4; 5 6 7 8', [(1, 2, 3, 4), (5, 6, 7, 8)]),
    ('1,2,3,4', [(1, 2, 3, 4)]),
    ('', []),
    ('[]', []),
    (None, []),
])
def test_parse_boxes_formats(value, expected):
    assert parse_boxes(value) == expected


def _split(images, boxes, widths=None, heights=None):
    return GWHDSplit('train', Path('x.csv'), images, boxes,
                     widths or [1024] * len(images), heights or [1024] * len(images))


def test_validate_boxes_counts_defects():
    split = _split(
        ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'],
        [[(10, 10, 100, 100)],
         [(500, 500, 500, 600)],                 # degenerate
         [(900, 900, 1200, 1200)],               # partially outside
         [(-50, -50, -10, -10)],                 # fully outside
         ])
    result = validate_boxes(split)
    assert result.images == 4
    assert result.boxes == 4
    assert result.degenerate == 1
    assert result.partially_outside == 1
    assert result.out_of_bounds == 1
    assert result.images_without_boxes == 0


def test_validate_boxes_counts_empty_and_duplicates():
    split = _split(['a.jpg', 'b.jpg'], [[], [(1, 1, 10, 10), (1, 1, 10, 10)]])
    result = validate_boxes(split)
    assert result.images_without_boxes == 1
    assert result.duplicate_boxes == 1


def test_clip_boxes_clamps_to_image():
    boxes = torch.tensor([[-5.0, -5.0, 20.0, 20.0]])
    assert clip_boxes(boxes, 10, 15).tolist() == [[0.0, 0.0, 10.0, 15.0]]


def test_keep_valid_boxes_drops_degenerate_after_clip():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 50.0], [5.0, 5.0, 20.0, 20.0]])
    kept = keep_valid_boxes(boxes, 10, 10, min_size=2.0)
    assert kept.tolist() == [[5.0, 5.0, 10.0, 10.0]]


def test_image_group_uses_leading_token():
    assert image_group('ethz_1_0001.jpg') == 'ethz'
    assert image_group('UQ_123.png') == 'uq'


# --------------------------------------------------------------------------- #
# official split handling
# --------------------------------------------------------------------------- #

def _write_split_csv(path: Path, rows):
    path.write_text('image_name,width,height,bbox\n' + '\n'.join(rows) + '\n',
                    encoding='utf-8')


def test_read_split_and_official_isolation(tmp_path):
    for split in ('train', 'val', 'test'):
        _write_split_csv(tmp_path / f'competition_{split}.csv',
                         [f'{split}_1.jpg,1024,1024,"[[10,10,50,50]]"',
                          f'{split}_2.jpg,1024,1024,"[]"'])
    from src.downstream.gwhd_data import discover_split_files
    files = discover_split_files(tmp_path)
    splits = {name: read_split(path, tmp_path, name)
              for name, path in files.items()}
    assert {k: len(v) for k, v in splits.items()} == {'train': 2, 'val': 2, 'test': 2}
    report = split_isolation_report(splits)
    assert report['overlaps']['train_val']['shared_image_names'] == 0
    assert not report['overlaps']['train_test']['shared_groups']


def test_read_split_rejects_duplicate_names(tmp_path):
    path = tmp_path / 'competition_train.csv'
    _write_split_csv(path, ['a.jpg,1024,1024,[]', 'a.jpg,1024,1024,[]'])
    with pytest.raises(ValueError):
        read_split(path, tmp_path, 'train')


def test_read_split_rejects_missing_box_column(tmp_path):
    path = tmp_path / 'competition_train.csv'
    path.write_text('image_name,width,height\nx.jpg,1024,1024\n', encoding='utf-8')
    with pytest.raises(ValueError):
        read_split(path, tmp_path, 'train')


def test_dataset_returns_boxes_and_flip_is_consistent(tmp_path):
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    array = (np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    Image.fromarray(array).save(image_dir / 'train_1.jpg')
    split = GWHDSplit('train', tmp_path / 'c.csv', ['train_1.jpg'],
                      [[(4, 4, 16, 16)]], [32], [32])
    dataset = GWHDDataset(tmp_path, split, train=False)
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    assert target['boxes'].tolist() == [[4.0, 4.0, 16.0, 16.0]]
    assert target['labels'].tolist() == [1]


def test_dataset_never_invents_boxes(tmp_path):
    image_dir = tmp_path / 'images'
    image_dir.mkdir()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_dir / 'x.jpg')
    split = GWHDSplit('train', tmp_path / 'c.csv', ['x.jpg'], [[]], [32], [32])
    _, target = GWHDDataset(tmp_path, split)[0]
    assert target['boxes'].shape == (0, 4)
    assert target['labels'].shape == (0,)


# --------------------------------------------------------------------------- #
# position embedding and backbone
# --------------------------------------------------------------------------- #

def test_resize_pos_embed_shape_and_identity():
    torch.manual_seed(0)
    original = torch.randn(1, 16 * 16, 1280)
    same = resize_pos_embed(original, 16)
    assert same.shape == original.shape
    assert torch.allclose(same, original, atol=1e-5)
    larger = resize_pos_embed(original, 37)
    assert larger.shape == (1, 37 * 37, 1280)
    assert torch.isfinite(larger).all()


def test_resize_pos_embed_rejects_non_square():
    with pytest.raises(ValueError):
        resize_pos_embed(torch.randn(1, 300, 1280), 16)


def test_backbone_output_shapes_and_frozen_encoder():
    encoder = vit_tiny(img_size=[224], patch_size=14)
    backbone = ViTFPNBackbone(encoder, input_size=224, out_channels=64,
                              embed_dim=192, frozen=True)
    x = torch.randn(1, 3, 224, 224)
    features = backbone(x)
    grid = 16
    assert list(features.keys()) == ['0', '1', '2', '3']
    assert features['1'].shape == (1, 64, grid, grid)
    assert features['0'].shape == (1, 64, grid * 2, grid * 2)
    assert features['2'].shape == (1, 64, grid // 2, grid // 2)
    for param in encoder.parameters():
        assert not param.requires_grad
    assert trainable_parameter_count(backbone) > 0


def test_backbone_rejects_wrong_input_size():
    encoder = vit_tiny(img_size=[224], patch_size=14)
    backbone = ViTFPNBackbone(encoder, input_size=224, out_channels=64,
                              embed_dim=192, frozen=True)
    with pytest.raises(ValueError):
        backbone(torch.randn(1, 3, 210, 210))


def test_backbone_freezes_weights_through_a_backward():
    encoder = vit_tiny(img_size=[224], patch_size=14)
    backbone = ViTFPNBackbone(encoder, input_size=224, out_channels=64,
                              embed_dim=192, frozen=True)
    before = [p.detach().clone() for p in encoder.parameters()]
    out = backbone(torch.randn(1, 3, 224, 224))['1']
    out.sum().backward()
    after = list(encoder.parameters())
    for old, new in zip(before, after):
        assert torch.equal(old, new.detach())
        assert new.grad is None
    assert backbone.lateral.weight.grad is not None


# --------------------------------------------------------------------------- #
# coordinate transform
# --------------------------------------------------------------------------- #

def test_square_resize_transform_maps_boxes():
    transform = SquareResizeTransform(518, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    image = torch.zeros(3, 1024, 1024)
    target = {'boxes': torch.tensor([[0.0, 0.0, 1024.0, 1024.0]]),
              'labels': torch.tensor([1]), 'area': torch.tensor([1024.0 ** 2])}
    images, targets = transform([image], [target])
    assert images.tensors.shape[-2:] == (518, 518)
    expected = 518.0
    assert torch.allclose(targets[0]['boxes'],
                          torch.tensor([[0.0, 0.0, expected, expected]]), atol=1e-4)


def test_square_resize_transform_keeps_already_square_images():
    transform = SquareResizeTransform(518, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    image = torch.zeros(3, 518, 518)
    images, _ = transform([image])
    assert images.tensors.shape[-2:] == (518, 518)


def test_positive_transformed_boxes_have_positive_area():
    transform = SquareResizeTransform(518, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    target = {'boxes': torch.tensor([[100.0, 200.0, 140.0, 260.0]]),
              'labels': torch.tensor([1]), 'area': torch.tensor([2400.0])}
    _, targets = transform([torch.zeros(3, 1024, 1024)], [target])
    box = targets[0]['boxes'][0]
    assert box[2] > box[0] and box[3] > box[1]
