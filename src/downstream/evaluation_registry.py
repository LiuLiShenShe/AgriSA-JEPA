"""Official metric definitions per benchmark.

The point is to make "mAP" unambiguous: GWHD uses COCO-style AP over IoU
0.50:0.95, PhenoBench uses the official ``phenobench-eval --task semantics``
mIoU, and so on.  A result file must name one of these entries; free-form
metrics are not accepted.

``metric_definition`` is deliberately descriptive: the same number computed
under a different definition is not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    dataset: str
    task: str
    primary: str
    secondary: List[str]
    tool: str
    notes: str = ''


METRICS: Dict[str, MetricDefinition] = {
    'gwhd_coco_ap': MetricDefinition(
        name='gwhd_coco_ap', dataset='gwhd2021', task='object_detection',
        primary='AP50:95',
        secondary=['AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                   'AR100', 'precision@0.5', 'recall@0.5'],
        tool='pycocotools COCOeval via torchmetrics MeanAveragePrecision',
        notes='Single class (wheat head). 0.50:0.95 IoU, 101-point interpolation. '
              'Absolute numbers are not comparable to Agri-FM+ Detectron2 runs; '
              'only A vs B under this identical protocol is.'),
    'phenobench_semantics_miou': MetricDefinition(
        name='phenobench_semantics_miou', dataset='phenobench',
        task='semantic_segmentation',
        primary='mIoU', secondary=['Crop IoU', 'Weed IoU'],
        tool='phenobench-eval --task semantics (official)',
        notes='Unchanged from Stage 3A/4B; never recompute with a different label map.'),
    'classification_accuracy': MetricDefinition(
        name='classification_accuracy', dataset='deepweeds',
        task='classification', primary='macro-F1', secondary=['accuracy'],
        tool='official split + sklearn-style confusion matrix'),
    'segmentation_miou': MetricDefinition(
        name='segmentation_miou', dataset='generic', task='semantic_segmentation',
        primary='mIoU', secondary=['per-class IoU'],
        tool='official tool of the dataset'),
}


def get(name: str) -> MetricDefinition:
    if name not in METRICS:
        raise KeyError(f'unknown metric {name!r}; registered: {sorted(METRICS)}')
    return METRICS[name]