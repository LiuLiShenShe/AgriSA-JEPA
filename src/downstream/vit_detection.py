"""Frozen ViT-H/14 detection backbone with an interpolated position encoding.

Why this file exists
--------------------
The I-JEPA ViT-H/14 was pretrained at 224x224 (16x16 tokens).  GWHD 2021 images
are 1024x1024 and wheat heads are small, so 224 is not a defensible input size.
Two engineering options were compared before choosing:

1. **Raise the input resolution** and interpolate the 2D sine-cosine position
   embedding (this file).  One encoder pass per image.
2. **Tile at 224** and run the encoder on every tile.  At stride 168 a 1024
   image needs 36 tiles (6x6): 36x the encoder passes and, if cached as fp16
   patch features, ~154 GB for the train split alone.

Measured on one RTX A6000 with BF16 + SDPA (no grad):

=====  ======  ==========
input  tokens  images/s
=====  ======  ==========
224    256     204
518    1369    37
700    2500    18
1008   5184    7.4
=====  ======  ==========

The formal Stage 5A protocol therefore uses **518x518** (37x37 tokens): 5.5x
more linear detail than 224 at 37 images/s, while 1008 is 5x slower and tiling
is 36x slower.  This is recorded in ``reports/stage5a_gwhd_protocol.md``; both
arms use it and it is not tuned per arm.

The position-embedding interpolation is the *only* modification to the encoder.
It is applied to the frozen copy used for detection; the pretraining code path
is untouched.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as TF

from src.downstream.features import EMBED_DIM, PATCH_SIZE, build_encoder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DETECTION_INPUT_SIZE = 518          # 37 * 14
DETECTION_GRID = DETECTION_INPUT_SIZE // PATCH_SIZE


class SquareResizeTransform(GeneralizedRCNNTransform):
    """Resize every image to exactly ``(size, size)`` without the 32-pixel rounding.

    ``GeneralizedRCNNTransform`` rounds the requested short side up to a
    multiple of 32 (518 -> 544), which breaks the ViT patch grid.  This
    transform scales directly to the patch-compatible square and rescales the
    target boxes by the same factors.  Normalisation and post-processing are
    inherited, so predicted boxes are mapped back to original image pixels.
    """

    def __init__(self, size: int, image_mean, image_std):
        super().__init__(min_size=size, max_size=size,
                         image_mean=image_mean, image_std=image_std,
                         size_divisible=1)
        self.size = size

    def resize(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]]):
        height, width = image.shape[-2:]
        if (height, width) != (self.size, self.size):
            image = TF.resize(image, [self.size, self.size],
                              interpolation=TF.InterpolationMode.BILINEAR,
                              antialias=True)
        if target is not None and 'boxes' in target and target['boxes'].numel():
            scale = torch.tensor([self.size / width, self.size / height,
                                  self.size / width, self.size / height],
                                 dtype=target['boxes'].dtype)
            target['boxes'] = target['boxes'] * scale.to(target['boxes'].device)
            if 'area' in target:
                target['area'] = target['area'] * (scale[0] * scale[1]).to(
                    target['area'].device)
        return image, target


def resize_pos_embed(pos_embed: torch.Tensor, grid_to: int) -> torch.Tensor:
    """Bicubically resize a cls-free 2D sine-cosine position embedding.

    ``pos_embed`` is ``[1, g*g, D]``; the returned tensor is ``[1, t*t, D]``.
    The I-JEPA encoder has no class token, and its in-model
    ``interpolate_pos_encoding`` assumes one, so the detection path resizes the
    parameter once at construction and then calls the encoder at its native
    (already resized) length.
    """
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
        raise ValueError(f'expected [1, N, D] pos_embed, got {tuple(pos_embed.shape)}')
    n = pos_embed.shape[1]
    grid_from = int(round(math.sqrt(n)))
    if grid_from * grid_from != n:
        raise ValueError(f'pos_embed length {n} is not a square grid')
    dim = pos_embed.shape[2]
    x = pos_embed.reshape(1, grid_from, grid_from, dim).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(grid_to, grid_to), mode='bicubic',
                      align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(1, grid_to * grid_to, dim)


class ViTFPNBackbone(nn.Module):
    """Frozen ViT-H patch tokens -> four feature maps for a Faster R-CNN neck.

    The "simple feature pyramid" mirrors the ViTDet recipe: one lateral 1x1
    convolution on the token grid (stride 14), one upsampled level (stride 7)
    and two pooled levels (strides 28, 56).  No pretrained neck weights exist
    for I-JEPA, so the neck is always trained from scratch -- identically for
    every arm.
    """

    def __init__(self, encoder: nn.Module, input_size: int = DETECTION_INPUT_SIZE,
                 out_channels: int = 256, embed_dim: int = EMBED_DIM,
                 frozen: bool = True, use_bfloat16: bool = True) -> None:
        super().__init__()
        if input_size % PATCH_SIZE:
            raise ValueError(f'input size must be a multiple of {PATCH_SIZE}')
        self.encoder = encoder
        self.input_size = input_size
        self.grid = input_size // PATCH_SIZE
        self.out_channels = out_channels
        self.frozen = frozen
        self.use_bfloat16 = use_bfloat16
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder.eval()

        self.lateral = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(4)
        ])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> 'OrderedDict[str, torch.Tensor]':
        if x.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                f'backbone expects {self.input_size}x{self.input_size}, '
                f'got {tuple(x.shape[-2:])}')
        if self.frozen:
            with torch.no_grad(), torch.amp.autocast(
                    'cuda', dtype=torch.bfloat16,
                    enabled=self.use_bfloat16 and x.is_cuda):
                tokens = self.encoder(x)
        else:
            tokens = self.encoder(x)
        tokens = tokens.float()          # bf16 encoder output -> fp32 neck
        b, n, d = tokens.shape
        if n != self.grid * self.grid:
            raise RuntimeError(
                f'encoder produced {n} tokens, expected {self.grid ** 2}')
        feat = tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid)
        p3 = self.lateral(feat)
        p2 = F.interpolate(p3, scale_factor=2.0, mode='nearest')
        p4 = F.max_pool2d(p3, kernel_size=2, stride=2)
        p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
        levels = [p2, p3, p4, p5]
        out: 'OrderedDict[str, torch.Tensor]' = OrderedDict()
        for idx, (level, conv) in enumerate(zip(levels, self.smooth)):
            out[str(idx)] = conv(level)
        return out


def build_frozen_vit_backbone(checkpoint_path: str, device: torch.device,
                              input_size: int = DETECTION_INPUT_SIZE,
                              frozen: bool = True) -> ViTFPNBackbone:
    """Load the checkpoint *strictly*, then resize its position embedding.

    Strict loading happens first so a mismatched checkpoint can never be
    silently accepted; the resize only changes the (fixed) positional tensor
    for the higher-resolution grid.
    """
    encoder = build_encoder(checkpoint_path, device=torch.device('cpu'))
    grid_to = input_size // PATCH_SIZE
    encoder.pos_embed = nn.Parameter(
        resize_pos_embed(encoder.pos_embed.detach(), grid_to), requires_grad=False)
    encoder.to(device)
    backbone = ViTFPNBackbone(encoder, input_size=input_size, frozen=frozen)
    return backbone.to(device)


def build_detector(checkpoint_path: str, device: torch.device,
                   input_size: int = DETECTION_INPUT_SIZE,
                   num_classes: int = 2,
                   frozen: bool = True,
                   min_size: Optional[int] = None,
                   max_size: Optional[int] = None) -> FasterRCNN:
    """Faster R-CNN with the identical architecture for every arm."""
    backbone = build_frozen_vit_backbone(checkpoint_path, device,
                                         input_size=input_size, frozen=frozen)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4)
    roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                  output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=min_size or input_size,
        max_size=max_size or input_size,
        image_mean=list(IMAGENET_MEAN),
        image_std=list(IMAGENET_STD),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pool,
        box_score_thresh=0.01,
        box_nms_thresh=0.5,
        box_detections_per_img=300,
    )
    model.transform = SquareResizeTransform(input_size, list(IMAGENET_MEAN),
                                            list(IMAGENET_STD))
    return model.to(device)


def trainable_parameter_names(model: nn.Module) -> List[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def frozen_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def assert_only_head_trainable(model: nn.Module) -> None:
    """Raise if any *encoder* parameter is trainable (the neck may train)."""
    bad = [name for name, param in model.named_parameters()
           if param.requires_grad and name.startswith('backbone.encoder.')]
    if bad:
        raise AssertionError(f'frozen backbone has trainable parameters: {bad[:5]}')
