"""Benchmark registry: the twelve downstream tasks Stage 5A/5B must cover.

Eight come from Agri-FM+ (five detection, two semantic segmentation, one
instance segmentation) and four from AgriMAE (DeepWeeds, PhenoBench,
GrowliFlower, CropAndWeed).  GWHD detection and GWHD instance segmentation are
kept as *separate* rows: bounding boxes are not instance masks.

``project_status`` records only what has actually run:

* ``RUN``             -- real numbers exist in ``reports/``
* ``RUN_PARTIAL``     -- a pilot ran but the protocol is not final
* ``DATA_AUDITED``    -- data obtained and validated, no model yet
* ``NOT_RUN``         -- nothing executed
* ``BLOCKED``         -- data or licence obstacle
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping

BENCHMARK_FIELDS: List[str] = [
    'benchmark_id', 'task', 'dataset', 'paper_version', 'paper_url',
    'download_url', 'license', 'images_reported', 'images_available',
    'official_splits', 'label_classes', 'metric', 'official_eval_tool',
    'overlap_risk', 'project_status', 'comparable_to_agrifm', 'notes',
]


def _b(**kw) -> Dict[str, object]:
    row = {f: '' for f in BENCHMARK_FIELDS}
    row.update(kw)
    return row


def benchmarks() -> List[Dict[str, object]]:
    return [
        # ---- object detection (Agri-FM+ x4 + crop detection) ----
        _b(benchmark_id='gwhd2021_det', task='object_detection',
           dataset='Global Wheat Head Dataset 2021',
           paper_version='GWHD 2021 (Zenodo 5092309)',
           paper_url='https://doi.org/10.34133/2021/9846152',
           download_url='https://zenodo.org/records/5092309',
           license='CC-BY-4.0', images_reported='6514',
           images_available='official full release (10.2 GB)',
           official_splits='train / val / test',
           label_classes='1 class (wheat head); boxes, domain labels',
           metric='mAP (COCO definition) / AP50 / AP50:95',
           official_eval_tool='pycocotools COCOeval (official split files)',
           overlap_risk='HIGH in Agri-FM+ (ref [31] in Agri-147K); excluded from '
                        'all of our pretraining manifests',
           project_status='RUN',
           comparable_to_agrifm='yes, if COCO mAP and official split are used',
           notes='Stage 5A frozen ViT-H + Faster R-CNN, 518 input, 20K iters, '
                 '2 head seeds. test AP50: A 0.0582±0.0018, B 0.0786±0.0001 '
                 '(delta +0.0204, 2/2 seeds); test AP50:95: A 0.0146, B 0.0189. '
                 'A single pretraining seed and a single detection dataset: '
                 'exploratory, not a paper-level claim. '
                 'See reports/stage5a_gwhd_results.md.'),
        _b(benchmark_id='plantdoc_det', task='object_detection',
           dataset='PlantDoc', paper_version='Singh et al. 2020',
           paper_url='https://doi.org/10.1145/3371158.3371196',
           download_url='https://github.com/pratikkayal/PlantDoc-Dataset',
           license='CC-BY-4.0', images_reported='2569',
           images_available='2,328 train / 239 test (8,851 instances)',
           official_splits='train / test', label_classes='30 classes, 13 species',
           metric='mAP', official_eval_tool='COCOeval',
           overlap_risk='not in our pretraining sources',
           project_status='NOT_RUN', comparable_to_agrifm='yes'),
        _b(benchmark_id='minneapple_det', task='object_detection',
           dataset='MinneApple', paper_version='Haeni et al. 2020',
           paper_url='https://doi.org/10.1109/LRA.2020.2965078',
           download_url='https://github.com/nicolaihaeni/MinneApple',
           license='research use (repository terms)', images_reported='1001',
           images_available='701 train / 200 val / 100 test',
           official_splits='train / val / test',
           label_classes='apple instance boxes', metric='mAP',
           official_eval_tool='COCOeval', overlap_risk='none',
           project_status='NOT_RUN', comparable_to_agrifm='yes'),
        _b(benchmark_id='disease_det', task='object_detection',
           dataset='Roboflow Detecting-Diseases',
           paper_version='Roboflow Universe v6 (2022)',
           paper_url='https://universe.roboflow.com/artificial-intelligence-82oex/detecting-diseases/dataset/6',
           download_url='https://universe.roboflow.com/artificial-intelligence-82oex/detecting-diseases/dataset/6',
           license='CC-BY-4.0 (dataset page)', images_reported='5493',
           images_available='2,904 train / 1,416 val / 1,163 test',
           official_splits='train / val / test', label_classes='13 disease categories',
           metric='mAP', official_eval_tool='COCOeval', overlap_risk='none',
           project_status='NOT_RUN', comparable_to_agrifm='yes'),
        _b(benchmark_id='crop_det', task='object_detection',
           dataset='Roboflow Crops (maize/cassava/sugarcane vs grass)',
           paper_version='Roboflow Universe (2023)',
           paper_url='https://universe.roboflow.com/school-el5qm/crops-maize-cassava-sugarcane-vs-grass',
           download_url='https://universe.roboflow.com/school-el5qm/crops-maize-cassava-sugarcane-vs-grass',
           license='CC-BY-4.0 (dataset page)', images_reported='570',
           images_available='392 train / 107 val / 71 test',
           official_splits='train / val / test', label_classes='crop vs grass',
           metric='mAP', official_eval_tool='COCOeval', overlap_risk='none',
           project_status='NOT_RUN', comparable_to_agrifm='yes'),
        # ---- semantic segmentation (Agri-FM+ x2) ----
        _b(benchmark_id='plantseg_sem', task='semantic_segmentation',
           dataset='PlantSeg',
           paper_version='Wei et al. 2024 (arXiv:2409.04038)',
           paper_url='https://arxiv.org/abs/2409.04038',
           download_url='https://arxiv.org/abs/2409.04038',
           license='see repository', images_reported='11458',
           images_available='official 20% test per disease category',
           official_splits='train / test (per disease)', label_classes='115 diseases',
           metric='mIoU', official_eval_tool='MMSegmentation mIoU',
           overlap_risk='none', project_status='NOT_RUN',
           comparable_to_agrifm='yes'),
        _b(benchmark_id='multiclass_weeds_sem', task='semantic_segmentation',
           dataset='Multiclass Weeds Dataset',
           paper_version='Yadav et al. 2023 (figshare)',
           paper_url='https://doi.org/10.6084/m9.figshare.22643434.v1',
           download_url='https://doi.org/10.6084/m9.figshare.22643434.v1',
           license='CC-BY-4.0', images_reported='7872',
           images_available='7,872 augmented images with pixel masks',
           official_splits='not stated', label_classes='2 weed species',
           metric='mIoU', official_eval_tool='MMSegmentation mIoU',
           overlap_risk='none', project_status='NOT_RUN',
           comparable_to_agrifm='yes'),
        # ---- instance segmentation (distinct from GWHD detection) ----
        _b(benchmark_id='gwhd_instance_seg', task='instance_segmentation',
           dataset='GWHD Instance Segmentation (Roboflow wheat-seg)',
           paper_version='Roboflow Universe (2024)',
           paper_url='https://universe.roboflow.com/wheat-seg-xcxow/wheat-segmentation-sr7mm',
           download_url='https://universe.roboflow.com/wheat-seg-xcxow/wheat-segmentation-sr7mm',
           license='CC-BY-4.0 (dataset page)', images_reported='582',
           images_available='439 train / 102 val / 41 test',
           official_splits='train / val / test', label_classes='wheat head instances',
           metric='mask AP', official_eval_tool='COCOeval (segm)',
           overlap_risk='none', project_status='NOT_RUN',
           comparable_to_agrifm='yes',
           notes='NOT the same resource as GWHD 2021 boxes.'),
        # ---- AgriMAE x4 ----
        _b(benchmark_id='deepweeds_cls', task='classification',
           dataset='DeepWeeds', paper_version='Olsen et al. 2019',
           paper_url='https://doi.org/10.1038/s41598-018-38343-3',
           download_url='https://github.com/AlexOlsen/DeepWeeds',
           license='CC-BY-4.0', images_reported='17509',
           images_available='official train/val/test', official_splits='train/val/test',
           label_classes='9 classes (8 weeds + negative)',
           metric='accuracy / macro-F1', official_eval_tool='official split accuracy',
           overlap_risk='appears in Agri-FM+ pretraining (ref [60]) and in AgriMAE; '
                        'hold out if used',
           project_status='NOT_RUN', comparable_to_agrifm='yes (classification not '
           'reported by Agri-FM+)'),
        _b(benchmark_id='phenobench_sem', task='semantic_segmentation',
           dataset='PhenoBench v1.1.0', paper_version='v1.1.0',
           paper_url='https://doi.org/10.1109/TPAMI.2024.3393571',
           download_url='https://www.phenobench.org/',
           license='CC-BY-SA-4.0', images_reported='1407 train / 772 val',
           images_available='official download + official eval tool installed',
           official_splits='train / val (+ test server)',
           label_classes='crop / weed / soil (+ 6 weed species semantics)',
           metric='mIoU (official semantics task)', official_eval_tool='phenobench-eval --task semantics',
           overlap_risk='excluded from all pretraining manifests',
           project_status='RUN',
           comparable_to_agrifm='partially: AgriMAE protocol differs (crop/weed binary or 3-class)',
           notes='A=71.68, B_2000=73.10 mIoU (Stage 4B, 3 probe seeds).'),
        _b(benchmark_id='growliflower_sem', task='semantic_segmentation',
           dataset='GrowliFlower', paper_version='Kierdorf et al. 2023',
           paper_url='https://doi.org/10.1002/rob.22126',
           download_url='https://www.ipb.uni-bonn.de/data/growliflower/',
           license='CC-BY-4.0', images_reported='1400',
           images_available='official train/val/test time series',
           official_splits='train / val / test', label_classes='cauliflower / weed / soil',
           metric='mIoU', official_eval_tool='official segmentation metric',
           overlap_risk='none', project_status='NOT_RUN',
           comparable_to_agrifm='not evaluated by Agri-FM+'),
        _b(benchmark_id='cropandweed_sem', task='semantic_segmentation',
           dataset='CropAndWeed', paper_version='Steininger et al. 2023',
           paper_url='https://doi.org/10.1109/WACV56688.2023.00372',
           download_url='https://github.com/cropandweed/cropandweed-dataset',
           license='CC-BY-SA-4.0', images_reported='5616',
           images_available='official train/val/test', official_splits='train/val/test',
           label_classes='crop / weed species', metric='mIoU',
           official_eval_tool='official segmentation metric',
           overlap_risk='none', project_status='NOT_RUN',
           comparable_to_agrifm='not evaluated by Agri-FM+'),
    ]


def write_benchmarks(path: Path) -> int:
    rows = benchmarks()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)
