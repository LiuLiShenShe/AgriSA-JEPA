"""Stage 5E publication gates.

Unlike Stage 5D, where a gate failure only affected a *candidate* corpus, the
Stage 5E gates are the last check before anyone is allowed to start a
controlled pretraining run.  The status vocabulary is:

``READY_FOR_CONTROLLED_PRETRAINING``
    Every gate below is green, the licence audit is evidence-backed, and the
    holdout entities that cannot be indexed have been explicitly excluded from
    the evaluation plan.
``CONDITIONAL``
    The corpus builds and the counts conserve, but at least one gate is
    unresolved (typically a holdout that could not be indexed at file level).
``BLOCKED``
    A gate that makes the counts or the licence claim meaningless failed.

A gate is never "green because nothing contradicted it"; each gate returns a
dict with a boolean and the evidence it read.
"""

from __future__ import annotations

from typing import Dict, List

STATUS_READY = 'READY_FOR_CONTROLLED_PRETRAINING'
STATUS_CONDITIONAL = 'CONDITIONAL'
STATUS_BLOCKED = 'BLOCKED'

GATES: List[tuple] = [
    ('legacy_files_unchanged',
     'Stage 5C/5D manifests, policies, ledgers and reports are byte-identical', True),
    ('licence_evidence_backed',
     'every accepted source has a research-use decision with an archived official '
     'document whose SHA-256 still matches', True),
    ('licence_unresolved_not_trained',
     'no source with UNRESOLVED or NOT_PERMITTED research use appears in the '
     'training manifest', True),
    ('train_and_redistribution_independent',
     'train eligibility and redistribution eligibility are separate fields and '
     'neither derives from the other', True),
    ('holdout_file_level_reported',
     'every registered holdout reports a real index status; no unreachable '
     'dataset is marked as passing', True),
    ('holdout_confirmed_hits_blocked',
     'confirmed exact-duplicate holdout hits are absent from the training '
     'manifest, and near-duplicate candidates are flagged for review', True),
    ('evaluation_scope_declared',
     'the evaluation plan names exactly the holdouts with file-level evidence and '
     'explicitly excludes the rest; no unindexed benchmark is treated as usable', True),
    ('quality_rules_not_relaxed',
     'R3v3 accepted is a subset of R3v2 accepted; no quarantine/IQA-missing/'
     'domain-unknown image was upgraded', True),
    ('group_split_inherited',
     'known-group train/val disjointness and the UNKNOWN_GROUP train-only rule '
     'are inherited from Stage 5D without modification', True),
    ('counts_conserved',
     'ledger rows equal accepted plus excluded, globally and per source', True),
    ('rebuild_deterministic',
     'the same ledger and policy rebuild the same manifest with the same hashes', True),
    ('sampling_exposure_only',
     'every sampling profile preserves the entity count and the split', True),
    ('composition_from_manifest',
     'source share, HHI and entropy are computed from the new manifest, not '
     'copied from a previous report', True),
    ('optimizer_updates_zero',
     'this stage ran no JEPA/Adapter forward, backward or optimizer step', True),
]


def evaluate(checks: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """``checks`` maps gate key -> {'passed': bool, 'evidence': {...}}."""
    results: Dict[str, Dict[str, object]] = {}
    failed: List[str] = []
    unresolved: List[str] = []
    for key, description, blocking in GATES:
        check = checks.get(key)
        if check is None:
            results[key] = {'description': description, 'blocking': blocking,
                            'status': 'UNRESOLVED', 'evidence': None}
            if blocking:
                unresolved.append(key)
            continue
        passed = bool(check.get('passed'))
        status = 'PASS' if passed else 'FAIL'
        results[key] = {'description': description, 'blocking': blocking,
                        'status': status, 'evidence': check.get('evidence')}
        if not passed and blocking:
            failed.append(key)
    if failed:
        status = STATUS_BLOCKED
    elif unresolved:
        status = STATUS_CONDITIONAL
    else:
        status = STATUS_READY
    return {'status': status, 'failed_gates': failed, 'unknown_gates': unresolved,
            'gates': results,
            'status_meaning': {
                STATUS_READY: 'all blocking gates passed with evidence',
                STATUS_CONDITIONAL: ('builds and conserves, but a blocking gate is '
                                     'unresolved; the unresolved gate must be listed in '
                                     'the report'),
                STATUS_BLOCKED: 'a blocking gate failed; the corpus must not be used',
            }}
