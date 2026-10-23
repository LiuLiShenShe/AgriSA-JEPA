"""GWHD 2021 detection: frozen ViT-H backbone + trained Faster R-CNN head.

Identical pipeline for every arm
--------------------------------
The backbone is frozen, the neck/head is trained from scratch, the data split is
the official one, the input is 518x518, the optimizer/schedule/budget/seed and
the evaluation are the same.  The only permitted difference is the checkpoint
passed with ``--arm``/``--checkpoint``.

Usage
-----
    CUDA_VISIBLE_DEVICES=1 .venv-uv/bin/python -m src.downstream.detection_train \
        --arm A --data-root /data/datasets/GWHD2021/extracted \
        --out-dir runs/stage5a_gwhd/A --iterations 20000 --batch-size 2 --seed 0

Outputs (per run directory)
---------------------------
``protocol.json``      resolved config + backbone/data SHA-256 + run key
``train_log.csv``      iteration, losses, LR, images/s
``eval.jsonl``         one line per evaluation point (val)
``results.json``       final val + test metrics, cost, parameter counts
``head.pth``           trainable-head state dict only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.downstream.backbone_factory import resolve
from src.downstream.evaluation_registry import get as get_metric
from src.downstream.gwhd_data import (GWHDDataset, collate, discover_split_files,
                                      read_split, split_isolation_report,
                                      validate_boxes)
from src.downstream.registry import make_spec
from src.downstream.vit_detection import (DETECTION_INPUT_SIZE, assert_only_head_trainable,
                                          build_detector, trainable_parameter_count)
from src.registry.ingest import file_sha256

PROTOCOL_FIELDS = ('input_size', 'iterations', 'batch_size', 'lr', 'momentum',
                   'weight_decay', 'warmup_iterations', 'eval_every',
                   'min_box_size', 'hflip_prob', 'score_thresh', 'seed',
                   'architecture', 'anchors', 'optimizer', 'scheduler')


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def precision_recall(preds: List[Dict[str, torch.Tensor]],
                     targets: List[Dict[str, torch.Tensor]],
                     iou_threshold: float = 0.5,
                     score_threshold: float = 0.5) -> Tuple[int, int, int]:
    """Greedy one-to-one matching P/R at a fixed score cutoff."""
    tp = fp = fn = 0
    for pred, target in zip(preds, targets):
        scores = pred['scores']
        keep = scores >= score_threshold
        boxes = pred['boxes'][keep]
        order = torch.argsort(-scores[keep])
        boxes = boxes[order]
        gt = target['boxes']
        used = torch.zeros(len(gt), dtype=torch.bool)
        for box in boxes:
            if len(gt) == 0:
                fp += 1
                continue
            x1 = torch.maximum(box[0], gt[:, 0]); y1 = torch.maximum(box[1], gt[:, 1])
            x2 = torch.minimum(box[2], gt[:, 2]); y2 = torch.minimum(box[3], gt[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_p = (box[2] - box[0]) * (box[3] - box[1])
            area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
            iou = inter / (area_p + area_g - inter).clamp(min=1e-6)
            iou = torch.where(used, torch.zeros_like(iou), iou)
            best = int(torch.argmax(iou)) if len(iou) else -1
            if best >= 0 and float(iou[best]) >= iou_threshold:
                tp += 1
                used[best] = True
            else:
                fp += 1
        fn += int((~used).sum())
    return tp, fp, fn


def evaluate(model, loader, device, max_images: Optional[int] = None
             ) -> Dict[str, object]:
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
                                  backend='pycocotools')
    model.eval()
    preds_all: List[Dict[str, torch.Tensor]] = []
    targets_all: List[Dict[str, torch.Tensor]] = []
    seen = 0
    start = time.time()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            cpu_outputs = [{k: v.detach().cpu() for k, v in out.items()}
                           for out in outputs]
            cpu_targets = [{k: v.detach().cpu() for k, v in t.items()}
                           for t in targets]
            preds_all.extend(cpu_outputs)
            targets_all.extend(cpu_targets)
            seen += len(images)
            if max_images is not None and seen >= max_images:
                break
    elapsed = time.time() - start
    metric.update(preds_all, targets_all)
    raw = metric.compute()
    metrics = {k: (float(v) if v.numel() == 1 else v.tolist())
               for k, v in raw.items()}
    tp, fp, fn = precision_recall(preds_all, targets_all)
    metrics['precision@0.5'] = tp / (tp + fp) if tp + fp else 0.0
    metrics['recall@0.5'] = tp / (tp + fn) if tp + fn else 0.0
    metrics['images'] = len(preds_all)
    metrics['seconds'] = round(elapsed, 2)
    metrics['ms_per_image'] = round(1000 * elapsed / max(len(preds_all), 1), 2)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['A', 'B'], default=None)
    parser.add_argument('--checkpoint', default=None,
                        help='explicit backbone checkpoint (overrides --arm)')
    parser.add_argument('--label', default=None)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-iterations', type=int, default=500)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=DETECTION_INPUT_SIZE)
    parser.add_argument('--score-thresh', type=float, default=0.01)
    parser.add_argument('--min-box-size', type=float, default=2.0)
    parser.add_argument('--hflip-prob', type=float, default=0.5)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-eval', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--final-test', action='store_true', default=True)
    args = parser.parse_args()

    if args.checkpoint:
        from src.downstream.backbone_factory import BackboneSpec
        spec = BackboneSpec(name=args.label or 'custom',
                            checkpoint=args.checkpoint,
                            label=args.label or Path(args.checkpoint).stem,
                            continued_training='custom',
                            data_version=None, input_size=args.input_size)
        backbone_key = args.label or 'custom'
    else:
        if not args.arm:
            raise SystemExit('either --arm or --checkpoint is required')
        spec = resolve(args.arm)
        backbone_key = args.arm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    data_root = Path(args.data_root)
    split_files = discover_split_files(data_root)
    splits = {name: read_split(path, data_root, name)
              for name, path in split_files.items()}
    isolation = split_isolation_report(splits)
    if any(v['shared_image_names'] for v in isolation['overlaps'].values()):
        raise SystemExit(f'official splits share image names: {isolation}')

    train_ds = GWHDDataset(data_root, splits['train'], train=True,
                           hflip_prob=args.hflip_prob,
                           min_box_size=args.min_box_size)
    val_ds = GWHDDataset(data_root, splits['val'], train=False,
                         min_box_size=args.min_box_size)
    test_ds = GWHDDataset(data_root, splits['test'], train=False,
                          min_box_size=args.min_box_size)
    if args.limit_train:
        train_ds.split.images = train_ds.split.images[:args.limit_train]
        train_ds.split.boxes = train_ds.split.boxes[:args.limit_train]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, collate_fn=collate,
                             pin_memory=True)

    model = build_detector(spec.checkpoint, device=device,
                           input_size=args.input_size, num_classes=2,
                           frozen=True)
    assert_only_head_trainable(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup_iterations:
            return args.lr * (step + 1) / max(args.warmup_iterations, 1)
        progress = (step - args.warmup_iterations) / max(
            args.iterations - args.warmup_iterations, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    protocol = {
        'backbone': backbone_key,
        'backbone_checkpoint': spec.checkpoint,
        'backbone_sha256': file_sha256(spec.checkpoint),
        'input_size': args.input_size,
        'iterations': args.iterations,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_iterations': args.warmup_iterations,
        'eval_every': args.eval_every,
        'min_box_size': args.min_box_size,
        'hflip_prob': args.hflip_prob,
        'score_thresh': args.score_thresh,
        'seed': args.seed,
        'architecture': 'FasterRCNN + SimpleFeaturePyramid(ViT-H/14 patch tokens)',
        'anchors': '4 levels (16,32,64,128) x (0.5,1.0,2.0)',
        'optimizer': 'SGD(momentum, nesterov=False) on head/neck only',
        'scheduler': 'linear warmup + cosine to 0',
        'data_root': str(data_root),
        'split_files': {k: str(v) for k, v in split_files.items()},
        'split_counts': isolation['counts'],
        'split_isolation': isolation,
        'trainable_parameters': trainable_parameter_count(model),
        'frozen_parameters': sum(p.numel() for p in model.parameters()
                                 if not p.requires_grad),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
    }
    (out_dir / 'protocol.json').write_text(
        json.dumps(protocol, indent=2) + '\n', encoding='utf-8')

    log_path = out_dir / 'train_log.csv'
    log_handle = open(log_path, 'w', newline='', encoding='utf-8')
    logger = csv.writer(log_handle)
    logger.writerow(['iteration', 'loss', 'loss_ce', 'loss_box', 'loss_rpn_box',
                     'loss_objectness', 'lr', 'images_per_s', 'peak_gb'])
    eval_path = out_dir / 'eval.jsonl'
    eval_handle = open(eval_path, 'w', encoding='utf-8')

    print(f'[{backbone_key}] train={len(train_ds)} val={len(val_ds)} '
          f'test={len(test_ds)} head_params={trainable_parameter_count(model)/1e6:.1f}M',
          flush=True)
    iterator = iter(train_loader)
    t0 = time.time()
    running = {}
    for step in range(args.iterations):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, targets = next(iterator)
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        lr = lr_at(step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        for key, value in loss_dict.items():
            running[key] = running.get(key, 0.0) + float(value)
        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - t0
            ips = (step + 1) * args.batch_size / max(elapsed, 1e-6)
            peak = (torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == 'cuda' else 0.0)
            logger.writerow([step + 1, float(loss), running.get('loss_classifier', 0) / 50,
                             running.get('loss_box_reg', 0) / 50,
                             running.get('loss_rpn_box_reg', 0) / 50,
                             running.get('loss_objectness', 0) / 50, lr,
                             round(ips, 2), round(peak, 3)])
            log_handle.flush()
            print(f'  step {step+1}/{args.iterations} loss={float(loss):.4f} '
                  f'lr={lr:.5f} {ips:.1f} img/s peak={peak:.1f}GB', flush=True)
            running = {}
        if (step + 1) % args.eval_every == 0 or step + 1 == args.iterations:
            metrics = evaluate(model, val_loader, device,
                               max_images=args.limit_eval or None)
            record = {'iteration': step + 1, 'split': 'val', **metrics}
            eval_handle.write(json.dumps(record) + '\n')
            eval_handle.flush()
            print(f'  [val @{step+1}] AP={metrics.get("map", 0):.4f} '
                  f'AP50={metrics.get("map_50", 0):.4f} '
                  f'AP75={metrics.get("map_75", 0):.4f}', flush=True)

    log_handle.close()
    eval_handle.close()
    train_seconds = time.time() - t0
    final_val = evaluate(model, val_loader, device,
                         max_images=args.limit_eval or None)
    final_test = (evaluate(model, test_loader, device,
                           max_images=args.limit_eval or None)
                  if args.final_test else None)

    results = {
        'protocol': protocol,
        'final_val': final_val,
        'final_test': final_test,
        'train_seconds': round(train_seconds, 1),
        'peak_gpu_gb': (torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == 'cuda' else 0.0),
        'metric_definition': get_metric('gwhd_coco_ap').__dict__,
        'metric_note': ('AP50:95 follows COCO IoU 0.50:0.95. Absolute mAP is not '
                        'comparable to Agri-FM+ Detectron2 numbers; A vs B is.'),
    }
    (out_dir / 'results.json').write_text(
        json.dumps(results, indent=2) + '\n', encoding='utf-8')
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith('backbone.')}, out_dir / 'head.pth')
    print(f'[{backbone_key}] val AP={final_val.get("map", 0):.4f} '
          f'AP50={final_val.get("map_50", 0):.4f}; '
          f'test AP={final_test.get("map", 0) if final_test else float("nan"):.4f} '
          f'-> {out_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
