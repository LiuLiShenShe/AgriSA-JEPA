"""Stage 5D quality policy: one source of truth, explicit metric semantics.

The Stage 5C code carried two incompatible threshold sets: ``policy.yaml``
declared ``accept_min: 0.35 / reject_max: 0.22`` while ``iqa_accept()`` used
hard-coded ``45.0 / 25.0``.  pyIQA's MUSIQ emits a 0-100 score, so the YAML
values could never have been applied to it.  This module removes the
possibility of that class of bug:

* every metric declares its own ``direction`` and ``score_range``;
* thresholds are read from the policy file, never from a function default;
* a metric that has no measured value is reported as ``MISSING`` and can never
  become a silent PASS.

``MetricSpec.metric_name`` alone cannot decide whether a score is good, so the
direction is mandatory and validated against the observed range.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

STATUS_PASS = 'PASS'
STATUS_QUARANTINE = 'QUARANTINE'
STATUS_REJECT = 'REJECT'
STATUS_MISSING = 'MISSING'

ROLE_GATE = 'GATE'
ROLE_DIAGNOSTIC = 'DIAGNOSTIC_ONLY'

HIGHER_IS_BETTER = 'higher_is_better'
LOWER_IS_BETTER = 'lower_is_better'


class PolicyError(ValueError):
    """Raised when a policy is ambiguous or self-contradictory."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    score_range: Tuple[float, float]
    accept_threshold: float
    reject_threshold: float
    role: str = ROLE_GATE
    model: str = ''
    model_sha256: str = ''
    scale_evidence: str = ''

    def validate(self) -> 'MetricSpec':
        if self.direction not in (HIGHER_IS_BETTER, LOWER_IS_BETTER):
            raise PolicyError(f'{self.name}: invalid direction {self.direction!r}')
        low, high = self.score_range
        if not low < high:
            raise PolicyError(f'{self.name}: empty score range {self.score_range}')
        if self.role not in (ROLE_GATE, ROLE_DIAGNOSTIC):
            raise PolicyError(f'{self.name}: invalid role {self.role!r}')
        if self.direction == HIGHER_IS_BETTER and not (
                self.reject_threshold <= self.accept_threshold):
            raise PolicyError(
                f'{self.name}: higher_is_better requires reject <= accept, got '
                f'{self.reject_threshold} > {self.accept_threshold}')
        if self.direction == LOWER_IS_BETTER and not (
                self.accept_threshold <= self.reject_threshold):
            raise PolicyError(
                f'{self.name}: lower_is_better requires accept <= reject, got '
                f'{self.accept_threshold} > {self.reject_threshold}')
        return self

    def evaluate(self, value: object) -> Tuple[str, str]:
        """Return ``(status, reason)``.  ``MISSING`` never becomes ``PASS``."""
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return STATUS_MISSING, f'{self.name.upper()}_MISSING'
        if score is None or not math.isfinite(score):
            return STATUS_MISSING, f'{self.name.upper()}_MISSING'
        low, high = self.score_range
        if score < low or score > high:
            return STATUS_MISSING, f'{self.name.upper()}_OUT_OF_RANGE'
        if self.direction == HIGHER_IS_BETTER:
            if score < self.reject_threshold:
                return STATUS_REJECT, f'{self.name.upper()}_REJECT'
            if score < self.accept_threshold:
                return STATUS_QUARANTINE, f'{self.name.upper()}_QUARANTINE'
            return STATUS_PASS, f'{self.name.upper()}_PASS'
        if score > self.reject_threshold:
            return STATUS_REJECT, f'{self.name.upper()}_REJECT'
        if score > self.accept_threshold:
            return STATUS_QUARANTINE, f'{self.name.upper()}_QUARANTINE'
        return STATUS_PASS, f'{self.name.upper()}_PASS'


@dataclass(frozen=True)
class BasicThresholds:
    min_short_side: int
    laplacian_var_min: float
    tenengrad_min: float
    dark_mean_max: float
    bright_mean_min: float
    entropy_min: float
    extreme_aspect_ratio_max: float
    blank_pixel_fraction_max: float
    near_threshold_band: float
    quarantine_on_conflict: bool


@dataclass(frozen=True)
class QualityPolicy:
    version: str
    path: Path
    sha256: str
    basic: BasicThresholds
    metrics: Dict[str, MetricSpec]
    missing_score_action: str

    def gate_metrics(self) -> List[MetricSpec]:
        return [m for m in self.metrics.values() if m.role == ROLE_GATE]

    def diagnostic_metrics(self) -> List[MetricSpec]:
        return [m for m in self.metrics.values() if m.role == ROLE_DIAGNOSTIC]

    def as_dict(self) -> Dict[str, object]:
        return {
            'version': self.version,
            'policy_sha256': self.sha256,
            'basic': self.basic.__dict__,
            'metrics': {name: spec.__dict__ | {'score_range': list(spec.score_range)}
                        for name, spec in self.metrics.items()},
            'missing_score_action': self.missing_score_action,
        }


def load_policy(path: str | Path) -> QualityPolicy:
    policy_path = Path(path)
    raw = yaml.safe_load(policy_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise PolicyError(f'{policy_path}: policy must be a mapping')
    quality = raw.get('quality') or {}
    basic_raw = quality.get('basic') or {}
    required_basic = set(BasicThresholds.__dataclass_fields__)
    missing_basic = required_basic - set(basic_raw)
    if missing_basic:
        raise PolicyError(f'policy missing quality.basic keys: {sorted(missing_basic)}')
    basic = BasicThresholds(**{key: basic_raw[key] for key in required_basic})

    iqa = quality.get('iqa') or {}
    metric_raw = iqa.get('metrics') or {}
    if not metric_raw:
        raise PolicyError('policy must declare quality.iqa.metrics')
    metrics: Dict[str, MetricSpec] = {}
    for name, spec in metric_raw.items():
        for key in ('direction', 'score_range', 'accept_threshold',
                    'reject_threshold'):
            if key not in spec:
                raise PolicyError(f'metric {name}: missing {key}')
        metrics[name] = MetricSpec(
            name=name,
            direction=spec['direction'],
            score_range=tuple(float(x) for x in spec['score_range']),  # type: ignore[arg-type]
            accept_threshold=float(spec['accept_threshold']),
            reject_threshold=float(spec['reject_threshold']),
            role=spec.get('role', ROLE_GATE),
            model=spec.get('model', name),
            model_sha256=spec.get('model_sha256', ''),
            scale_evidence=spec.get('scale_evidence', ''),
        ).validate()

    missing_action = iqa.get('missing_score_action', STATUS_QUARANTINE)
    if missing_action not in (STATUS_QUARANTINE, STATUS_REJECT):
        raise PolicyError(
            'quality.iqa.missing_score_action must QUARANTINE or REJECT; '
            'a missing measurement can never PASS')
    return QualityPolicy(
        version=raw.get('version', 'unknown'),
        path=policy_path,
        sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        basic=basic,
        metrics=metrics,
        missing_score_action=missing_action,
    )


def combine_metric_statuses(statuses: List[str],
                            missing_action: str = STATUS_QUARANTINE
                            ) -> Tuple[str, List[str]]:
    """Combine several metric outcomes into one gate status.

    ``REJECT`` dominates ``QUARANTINE`` dominates ``MISSING`` dominates
    ``PASS``.  A missing score is never treated as a pass.
    """
    if not statuses:
        return STATUS_PASS, []
    if STATUS_REJECT in statuses:
        return STATUS_REJECT, [s for s in statuses if s == STATUS_REJECT]
    if STATUS_QUARANTINE in statuses:
        return STATUS_QUARANTINE, [s for s in statuses if s == STATUS_QUARANTINE]
    if STATUS_MISSING in statuses:
        return missing_action, [STATUS_MISSING]
    return STATUS_PASS, []
