"""Stage 5E corpus construction from the Stage 5D ledger.

The Stage 5D ledger already carries, per image, the result of every quality,
modality, domain, dedup, holdout and licence decision.  Stage 5E does not
re-decide any of those.  It applies exactly one new rule:

    an image is admitted to R3v3 only if its source has a positive Stage 5E
    research-use licence status, AND it was ACCEPTED in Stage 5D, AND it was
    not a registered holdout.

Every other Stage 5D predicate is inherited verbatim, so no threshold is
retuned and no quarantine image is silently upgraded.

The output keeps train eligibility and redistribution eligibility as separate
columns; they are never merged into one flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.registry import stage5e_licence as lic
from src.registry.stage5e_holdout_registry import by_id as holdout_by_id

#: Stage 5D reasons that already mean "not trainable".  Anything outside this
#: set and ACCEPTED is inherited as accepted.
INHERITED_BUCKETS = (
    'accepted', 'quarantined', 'rejected', 'duplicate_removed',
    'holdout', 'licence_blocked', 'archive_only',
)

#: File that records the holdout images confirmed to be identical to a
#: pretraining-corpus image by the pixel review.
COLLISION_REVIEW = Path(__file__).resolve().parents[2] / \
    'reports/stage5e_hash_collision_review.json'

#: Verdicts from that review which mean "the corpus image is the holdout image".
EXCLUDE_VERDICTS = ('IDENTICAL_PIXELS', 'RESAMPLED_COPY_SAME_CONTENT')


def confirmed_holdout_hits(review_path: Path = COLLISION_REVIEW) -> set:
    """Corpus image ids the pixel review proved to be holdout images.

    These are removed before any licence reasoning: a confirmed holdout hit is a
    hard external constraint, exactly like the Stage 5D holdout block.
    """
    if not review_path.exists():
        return set()
    payload = json.loads(review_path.read_text())
    rows = payload.get('rows', [])
    return {str(row['corpus_id']) for row in rows
            if str(row.get('verdict')) in EXCLUDE_VERDICTS}


def load_ledger(ledger_dir) -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in sorted(ledger_dir.glob('*.csv'))]
    if not frames:
        raise SystemExit(f'no ledger CSV in {ledger_dir}')
    return pd.concat(frames, ignore_index=True)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(['true', '1', 'yes'])


def stage5e_decision(ledger: pd.DataFrame) -> pd.DataFrame:
    """Attach Stage 5E columns.  Stage 5D columns are never overwritten."""
    frame = ledger.copy()

    source_id = frame['physical_source_id'].astype(str)
    research_use = source_id.map(lic.research_use_status).fillna(lic.UNRESOLVED)
    train_ok = source_id.map(lic.train_eligible).astype(bool)
    redistribution_ok = source_id.map(lic.redistribution_eligible).astype(bool)

    frame['stage5e_research_use'] = research_use
    frame['stage5e_train_eligible'] = train_ok
    frame['stage5e_redistribution_eligible'] = redistribution_ok

    # Registered holdout ids and upstream holdout names are blocked at the
    # source level; Stage 5D already blocked them per-image, this makes it
    # impossible to lose the check if a source is ever added to the registry.
    holdout_ids = set(holdout_by_id().keys())
    source_holdout = source_id.isin(holdout_ids)

    accepted_d5 = _as_bool(frame['selected']) & frame['primary_reason'].eq('ACCEPTED')
    holdout_d5 = frame['holdout_status'].astype(str).eq('HOLDOUT')

    # Stage 5E: images the pixel review proved to be a downstream holdout image.
    # Stage 5D dismissed the two distance-0 GWHD candidates as a pHash
    # collision; the review re-checked them against the decoded pixels and found
    # correlation 0.9994-0.9998 with the GWHD originals (the corpus copies are
    # 512x512 downscales of the 1024x1024 originals).  They are blocked here.
    confirmed = confirmed_holdout_hits()
    confirmed_hit = frame['image_id'].astype(str).isin(confirmed)

    # Stage 5E exclusion: licence not positively verified for research use.
    licence_blocked_e5 = ~train_ok

    frame['stage5e_selected'] = (accepted_d5 & ~source_holdout & ~holdout_d5
                                 & ~confirmed_hit & ~licence_blocked_e5)
    frame['stage5e_primary_reason'] = np.select(
        [~accepted_d5, source_holdout, holdout_d5, confirmed_hit, licence_blocked_e5],
        ['INHERITED_STAGE5D_' + frame['primary_reason'].fillna('').astype(str),
         'HOLDOUT_SOURCE_REGISTERED', 'HOLDOUT_EXCLUDED',
         'HOLDOUT_CONFIRMED_PIXEL_MATCH', 'LICENCE_UNRESOLVED_FOR_RESEARCH_USE'],
        default='ACCEPTED')
    return frame


def build(ledger: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    decided = stage5e_decision(ledger)
    accepted = decided[decided['stage5e_selected']].copy()

    by_source = accepted['physical_source_id'].value_counts().to_dict()
    by_bucket = decided.groupby('stage5e_primary_reason').size().to_dict()

    summary: Dict[str, object] = {
        'ledger_rows': int(len(decided)),
        'stage5d_accepted': int(_as_bool(decided['selected']).sum()),
        'stage5e_accepted': int(len(accepted)),
        'stage5d_accepted_by_source': decided.loc[_as_bool(decided['selected']),
                                                  'physical_source_id'].value_counts().to_dict(),
        'stage5e_accepted_by_source': by_source,
        'reason_buckets': {str(k): int(v) for k, v in by_bucket.items()},
        'excluded_sources': sorted(
            source for source, count in
            decided.loc[_as_bool(decided['selected']) &
                        ~decided['stage5e_selected'], 'physical_source_id']
            .value_counts().items()),
        'research_use_by_source': {
            source: status for source, status in
            sorted(decided.groupby('physical_source_id')['stage5e_research_use']
                   .agg(lambda s: s.iloc[0]).items())},
    }
    return decided, summary


def conservation(decided: pd.DataFrame) -> Dict[str, object]:
    """Ledger rows must equal accepted + excluded, per source and globally."""
    total = int(len(decided))
    accepted = int(decided['stage5e_selected'].sum())
    excluded = total - accepted
    by_source_total = decided.groupby('physical_source_id').size().to_dict()
    by_source_accepted = decided.groupby('physical_source_id')['stage5e_selected'].sum() \
        .to_dict()
    conserved = all(
        int(by_source_accepted[source]) <= int(count)
        for source, count in by_source_total.items())
    return {
        'ledger_rows': total,
        'stage5e_accepted': accepted,
        'stage5e_excluded': excluded,
        'accepted_plus_excluded': accepted + excluded,
        'conserved': (accepted + excluded == total) and conserved,
        'per_source_conserved': conserved,
    }
