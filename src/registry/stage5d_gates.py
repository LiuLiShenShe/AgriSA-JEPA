"""The twelve Stage 5D publication gates.

The status vocabulary is deliberately narrow:

``READY_AUTO_VALIDATED``
    All twelve programmatic gates passed.  This is an *automatic* validation;
    it never means human-verified and never means PROVEN_NO_LEAKAGE.

``AUTO_CANDIDATE``
    The corpus builds and runs, but at least one gate is unresolved.

``AUDIT_BLOCKED``
    A gate that makes the counts meaningless failed (unreadable files, policy
    disagreement, non-conserving funnel).

``TARGET_EXCEEDED`` is a *scale* property and is reported separately.  It can
never promote a corpus to ``READY_AUTO_VALIDATED``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

STATUS_READY = 'READY_AUTO_VALIDATED'
STATUS_CANDIDATE = 'AUTO_CANDIDATE'
STATUS_BLOCKED = 'AUDIT_BLOCKED'
STATUS_INCOMPLETE = 'INCOMPLETE'

#: The twelve publication gates: (key, description, blocking).
GATES: List[tuple] = [
    ('files_readable',
     'every accepted image exists on disk and a deterministic sample decodes', True),
    ('policy_and_iqa_scale_verified',
     'manifest policy hash equals the executed file; every IQA metric declares '
     'direction, range, thresholds and model SHA', True),
    ('visible_rgb_evidenced',
     'every accepted image has documented visible-RGB evidence', True),
    ('field_domain_evidenced',
     'every accepted image has a field/orchard domain with recorded evidence', True),
    ('licence_clear',
     'every accepted image has a positively accepted licence', True),
    ('dedup_resolved',
     'exact duplicates resolved deterministically; near-duplicate decisions '
     'recorded and unresolved candidates quarantined', False),
    ('holdout_screened',
     'source, prefix and file-hash holdout screening completed and disclosed', True),
    ('groups_disjoint_and_unknown_flagged',
     'strong groups do not cross splits; unknown-group images are label-separated', False),
    ('counts_conserved',
     'funnel, manifest and per-source counts are the same decision table', True),
    ('rebuild_byte_identical',
     'two independent builds produced identical train/val SHA-256', False),
    ('composition_from_manifest',
     'composition tables were generated from the accepted manifest', False),
    ('legacy_unchanged',
     'Stage 5C policy, manifests and reports are byte-identical', True),
]


def status(gates: Dict[str, Optional[bool]],
           target_exceeded: bool = False) -> Dict[str, object]:
    missing = [key for key, _, _ in GATES if key not in gates]
    failed = [key for key, _, _ in GATES if gates.get(key) is False]
    unknown = [key for key, _, _ in GATES if gates.get(key) is None]
    blocking_fail = [key for key, _, blocking in GATES
                     if blocking and gates.get(key) is False]

    if blocking_fail:
        value = STATUS_BLOCKED
    elif not failed and not unknown and not missing:
        value = STATUS_READY
    elif missing:
        value = STATUS_INCOMPLETE
    else:
        value = STATUS_CANDIDATE
    return {
        'status': value,
        'target_exceeded': bool(target_exceeded),
        'target_exceeded_is_scale_property_only': True,
        'failed_gates': failed,
        'unknown_gates': unknown,
        'missing_gates': missing,
        'blocking_failures': blocking_fail,
        'all_gates_passed': value == STATUS_READY,
    }
