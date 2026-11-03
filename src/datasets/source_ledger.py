"""Record which ``physical_source_id`` every emitted training sample belongs to.

Stage 6 must report *realised* source exposure, not the configured ratios.
The training loop therefore feeds the ``sample_id`` of every micro-batch it
actually consumes into this ledger.  The ledger is written as a per-sample CSV
(so counts can be recomputed independently of the code) and as a compact
summary JSON.

Rows from a resumed process are appended with a ``part`` label recording the
optimizer-step position at which that process started, so an interrupted run
and its continuation are still separable.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from src.datasets.stage6_sampling import concentration

COLUMNS = [
    'part', 'epoch', 'itr', 'micro_step', 'optimizer_step',
    'sample_id', 'physical_source_id',
]


class SourceExposureLedger:
    def __init__(
        self,
        path: str | Path,
        sample_to_source: Mapping[str, str],
        part: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        self.summary_path = self.path.with_suffix('.summary.json')
        self.sample_to_source = sample_to_source
        self.part = str(part) if part is not None else '0'
        self.rows: List[Dict[str, object]] = []
        self.counts: Counter = Counter()
        self.missing: List[str] = []
        # -- rows already written to disk. ``flush`` runs at every epoch end, so
        #    only the newly accumulated rows may be appended; re-writing the
        #    whole buffer would duplicate every earlier epoch.
        self._flushed = 0

    def record(self, epoch: int, itr: int, micro_step: int,
               optimizer_step: int, sample_ids: Iterable[object]) -> None:
        for sample_id in sample_ids:
            key = str(sample_id)
            source = self.sample_to_source.get(key)
            if source is None:
                self.missing.append(key)
                continue
            self.counts[source] += 1
            self.rows.append({
                'part': self.part,
                'epoch': int(epoch),
                'itr': int(itr),
                'micro_step': int(micro_step),
                'optimizer_step': int(optimizer_step),
                'sample_id': key,
                'physical_source_id': source,
            })

    def summary(self) -> Dict[str, object]:
        if self.missing:
            raise RuntimeError(
                f'{len(self.missing)} emitted sample ids are absent from the '
                f'manifest, e.g. {self.missing[:5]}')
        payload = concentration(dict(self.counts))
        payload['samples'] = len(self.rows)
        payload['part'] = self.part
        return payload

    def flush(self) -> None:
        summary = self.summary()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.is_file() and self.path.stat().st_size > 0
        pending = self.rows[self._flushed:]
        with self.path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(pending)
        self._flushed = len(self.rows)
        existing: Dict[str, object] = {}
        if self.summary_path.is_file():
            existing = json.loads(self.summary_path.read_text(encoding='utf-8'))
        parts = dict(existing.get('parts', {}))
        parts[self.part] = summary
        # -- merged over every part written so far
        merged: Counter = Counter()
        for part_summary in parts.values():
            merged.update(part_summary.get('counts', {}))
        combined = concentration(dict(merged))
        combined['parts'] = parts
        combined['samples'] = sum(part['samples'] for part in parts.values())
        self.summary_path.write_text(
            json.dumps(combined, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


__all__ = ['SourceExposureLedger', 'COLUMNS']
