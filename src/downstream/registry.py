"""Unified experiment registry.

Every downstream run must be described by an :class:`ExperimentSpec` *before*
it starts.  The registry:

* binds a result to the exact backbone SHA-256, data manifest, head config,
  budget, seed and metric;
* derives a ``run_key`` so two different configurations can never overwrite
  the same result file;
* refuses to register a run whose backbone is missing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.downstream.backbone_factory import BackboneSpec, resolve


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset: str
    task: str                      # object_detection / semantic_segmentation ...
    backbone: str                  # 'A' / 'B' / explicit label
    backbone_checkpoint: str
    backbone_sha256: str
    frozen_backbone: bool
    head: str
    data_manifest: str
    data_manifest_sha256: str
    seed: int
    budget: str
    metric: str
    result_path: str
    input_size: int
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def run_key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data['run_key'] = self.run_key
        return data


def make_spec(experiment_id: str, dataset: str, task: str,
              backbone: str, head: str, data_manifest: str,
              seed: int, budget: str, metric: str, result_path: str,
              input_size: int, frozen_backbone: bool = True,
              extra: Optional[Dict[str, str]] = None) -> ExperimentSpec:
    spec: BackboneSpec = resolve(backbone)
    manifest = Path(data_manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f'data manifest not found: {manifest}')
    from src.downstream.features import sha256_file
    return ExperimentSpec(
        experiment_id=experiment_id,
        dataset=dataset,
        task=task,
        backbone=backbone,
        backbone_checkpoint=spec.checkpoint,
        backbone_sha256=spec.checkpoint_sha256,
        frozen_backbone=frozen_backbone,
        head=head,
        data_manifest=str(manifest),
        data_manifest_sha256=sha256_file(manifest),
        seed=seed,
        budget=budget,
        metric=metric,
        result_path=result_path,
        input_size=input_size,
        extra=extra or {},
    )


def assert_comparable(spec_a: ExperimentSpec, spec_b: ExperimentSpec) -> None:
    """A/B may differ only in the backbone checkpoint and its hash."""
    fields = ('dataset', 'task', 'frozen_backbone', 'head', 'data_manifest',
              'data_manifest_sha256', 'seed', 'budget', 'metric', 'input_size')
    mismatches = [f for f in fields if getattr(spec_a, f) != getattr(spec_b, f)]
    if mismatches:
        raise AssertionError(f'A/B protocols differ in {mismatches}')
    if spec_a.backbone_sha256 == spec_b.backbone_sha256:
        raise AssertionError('A and B point at identical weights')