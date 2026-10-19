"""Stage 2C duplicate-candidate review workflow (human-review tooling).

This tool is a companion to the 784 candidate edges in
``data/duplicate_review.csv``. It helps a HUMAN reviewer inspect, record and
consistency-check verdicts. It never invents a verdict and never pre-fills a
reviewer: an unattributed verdict is not a verdict.

Hard rules
----------
- The tool NEVER writes a verdict on its own. ``record`` only persists rows the
  caller explicitly supplied, with a non-empty ``reviewer`` that is not a
  reserved AI/automation sentinel.
- It never mutates ``data/frozen/`` or any row ``status``. ``impact`` and
  ``refreeze-plan`` only *describe* what a re-freeze would require; they do not
  execute it.

Subcommands
-----------
- ``status``   -- counts by review_status, by section, pending total.
- ``record``   -- persist fully-attributed verdicts from a JSON map (atomic).
- ``check``    -- consistency / conflict checks over the CSV.
- ``impact``   -- given CURRENT verdicts, classify which frozen rows would be
                  returned vs removed, and emit the re-freeze command.
- ``refreeze-plan`` -- print the explicit step list for a post-adjudication
                  re-freeze, without executing it.

Repo style: argparse CLI, reuses ``fmt_int``/``write_text``/``sha256_file`` from
the existing Stage 2 audit module so the two tools stay in sync.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Reuse the helpers already defined for the Stage 2 audit so this tool cannot
# drift from the generator that produced the CSV it reads.
try:
    from tools.stage2_data_audit import fmt_int, write_text, sha256_file  # type: ignore
except Exception:  # pragma: no cover - path fallback for direct execution
    try:
        from stage2_data_audit import fmt_int, write_text, sha256_file  # type: ignore
    except Exception:
        def fmt_int(value) -> str:
            return f'{int(value):,}'

        def write_text(path: Path, text: str) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')

        def sha256_file(path: Path) -> str:
            import hashlib
            with path.open('rb') as f:
                return hashlib.file_digest(f, 'sha256').hexdigest()


REVIEW_CSV = Path('data/duplicate_review.csv')
FROZEN_DIR = Path('data/frozen')
FROZEN_TRAIN = FROZEN_DIR / 'agrifield40k-v0_train.csv'
FROZEN_VAL = FROZEN_DIR / 'agrifield40k-v0_val.csv'

PENDING = 'pending_manual_review'
VALID_VERDICTS = {
    'exact_duplicate',
    'confirmed_duplicate',
    'near_duplicate',
    'not_duplicate',
}
# Reserved identifiers that must never be accepted as a human reviewer. An AI
# or automation tag is not a human attribution.
AI_REVIEWER_SENTINELS = {
    'ai', 'a.i.', 'automation', 'auto', 'bot', 'script', 'system', 'claude',
    'gpt', 'llm', 'agent', 'duplicate_review_workflow', 'tool', 'stage2c',
    'assistant', 'machine', 'model', 'openai', 'chatgpt', 'codex', 'copilot',
}
_ISO_RE = re.compile(r'^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?'
                     r'(Z|[+-]\d{2}:?\d{2})?)?$')
_REQUIRED_FIELDS = ('review_status', 'reviewer', 'review_notes', 'reviewed_at')


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _is_true(value) -> bool:
    return str(value).strip().lower() in ('true', '1', 'yes')


def _section(row: dict) -> str:
    """Cross-split / manifest-affecting / same-split section of one edge."""
    if _is_true(row.get('cross_split')):
        return 'cross_split'
    if _is_true(row.get('in_train_manifest')) or _is_true(row.get('in_val_manifest')):
        return 'manifest_affecting'
    return 'same_split'


def _section_label(sec: str) -> str:
    return {
        'cross_split': 'Cross-split (train<->val leakage risk)',
        'manifest_affecting': 'Manifest-affecting (one endpoint in a manifest)',
        'same_split': 'Same-split screening (no manifest impact)',
    }[sec]


def _is_valid_iso(value: str) -> bool:
    value = (value or '').strip()
    if not _ISO_RE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


def _emit(args, text: str) -> None:
    """Write markdown to args.out, or to stdout when out is '-'."""
    out = args.out
    if out is None or str(out) == '-':
        sys.stdout.write(text)
    else:
        write_text(Path(out), text)
        print(f'wrote {out}')


def _load_frozen_sets():
    """Return (train_ids, val_ids) from the frozen manifests, if present."""
    train, val = set(), set()
    if FROZEN_TRAIN.exists():
        train = {r['sample_id'] for r in _load_rows(FROZEN_TRAIN)}
    if FROZEN_VAL.exists():
        val = {r['sample_id'] for r in _load_rows(FROZEN_VAL)}
    return train, val


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_status(args):
    rows = _load_rows(args.review_csv)
    by_status = Counter(r['review_status'] for r in rows)
    by_section = Counter(_section(r) for r in rows)
    pending = sum(1 for r in rows if r['review_status'] == PENDING)
    out = []
    out.append('# Duplicate-review status')
    out.append('')
    out.append(f'Total candidate edges: {fmt_int(len(rows))}')
    out.append('')
    out.append('## By review_status')
    out.append('')
    out.append('| review_status | edges |')
    out.append('| --- | ---: |')
    for status, n in sorted(by_status.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f'| {status} | {fmt_int(n)} |')
    out.append('')
    out.append('## By section')
    out.append('')
    out.append('| section | edges |')
    out.append('| --- | ---: |')
    total_sec = 0
    for sec in ('cross_split', 'manifest_affecting', 'same_split'):
        n = by_section.get(sec, 0)
        total_sec += n
        out.append(f'| {_section_label(sec)} | {fmt_int(n)} |')
    out.append(f'| **total** | **{fmt_int(total_sec)}** |')
    out.append('')
    out.append('## Still pending manual review')
    out.append('')
    out.append(f'{fmt_int(pending)} of {fmt_int(len(rows))} edges are `{PENDING}`. '
               'No verdict was auto-filled; a human must review each edge.')
    out.append('')
    out.append('The 258 cross-split / manifest-affecting edges are the '
               'highest-priority group (direct pretrain->val leakage risk).')
    out.append('')
    _emit(args, '\n'.join(out) + '\n')
    print(f'edges={len(rows)} pending={pending} sections={dict(by_section)}')
    return 0


def validate_verdict(edge_id, verdict):
    """Return an error string, or None if the verdict record is acceptable.

    A verdict must carry all four fields, a recognised status, a real (non-
    sentinel) reviewer, notes, and an ISO-8601 reviewed_at.
    """
    missing = [f for f in _REQUIRED_FIELDS if f not in verdict]
    if missing:
        return f'edge {edge_id}: missing required field(s): {", ".join(missing)}'
    status = verdict['review_status'].strip()
    if status not in VALID_VERDICTS:
        return (f'edge {edge_id}: invalid review_status {status!r} '
                f'(allowed: {sorted(VALID_VERDICTS)})')
    reviewer = verdict['reviewer'].strip()
    if not reviewer:
        return f'edge {edge_id}: reviewer is empty (an unattributed verdict is not a verdict)'
    if reviewer.lower() in AI_REVIEWER_SENTINELS:
        return (f'edge {edge_id}: reviewer {reviewer!r} is a reserved '
                'AI/automation sentinel and cannot author a human verdict')
    if not verdict['review_notes'].strip():
        return f'edge {edge_id}: review_notes is empty'
    if not _is_valid_iso(verdict['reviewed_at']):
        return (f'edge {edge_id}: reviewed_at {verdict["reviewed_at"]!r} is not '
                'ISO-8601')
    return None


def apply_verdicts(rows, verdicts):
    """Validate then apply a {edge_id: {field: value}} map. Returns error list.

    Pure: mutates ``rows`` only when the whole batch validates.
    """
    by_id = {r['edge_id']: r for r in rows}
    errors = []
    for edge_id, verdict in verdicts.items():
        if edge_id not in by_id:
            errors.append(f'unknown edge_id {edge_id!r}')
            continue
        err = validate_verdict(edge_id, verdict)
        if err:
            errors.append(err)
    if errors:
        return errors
    for edge_id, verdict in verdicts.items():
        row = by_id[edge_id]
        for f in _REQUIRED_FIELDS:
            row[f] = str(verdict[f]).strip()
    return []


def cmd_record(args):
    """Record fully-attributed verdicts from a JSON file (atomic write-back)."""
    review_csv = Path(args.review_csv)
    verdicts = json.loads(Path(args.verdicts).read_text(encoding='utf-8'))
    if not isinstance(verdicts, dict) or not verdicts:
        print('ERROR: --verdicts must be a non-empty JSON object keyed by '
              'edge_id', file=sys.stderr)
        return 2
    rows = _load_rows(review_csv)
    errors = apply_verdicts(rows, verdicts)
    if errors:
        for e in errors:
            print(f'ERROR: {e}', file=sys.stderr)
        return 1

    fields = list(rows[0].keys())
    tmp = review_csv.with_name(review_csv.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\r\n')
        w.writeheader()
        w.writerows(rows)
    shutil.move(str(tmp), str(review_csv))
    print(f'Recorded {len(verdicts)} verdict(s): {", ".join(sorted(verdicts))}')
    return 0


def check_rows(rows):
    """Return a list of human-readable consistency issues."""
    issues = []
    for eid, n in Counter(r['edge_id'] for r in rows).items():
        if n > 1:
            issues.append(f'duplicate edge_id {eid} appears {n} times')
    for r in rows:
        st = (r.get('review_status') or '').strip()
        eid = r.get('edge_id', '?')
        reviewer = (r.get('reviewer') or '').strip()
        if st not in VALID_VERDICTS and st != PENDING:
            issues.append(f'{eid}: unknown verdict {st!r}')
        # A verdict with no reviewer is unattributed: not a verdict.
        if st in VALID_VERDICTS and not reviewer:
            issues.append(f'{eid}: verdict {st} has blank reviewer '
                          '(unattributed verdict)')
        if st in VALID_VERDICTS and reviewer.lower() in AI_REVIEWER_SENTINELS:
            issues.append(f'{eid}: verdict attributed to reserved sentinel '
                          f'{reviewer!r}')
        # reviewer present but no verdict -> inconsistent half-state.
        if st == PENDING and reviewer:
            issues.append(f'{eid}: reviewer set but status still {PENDING!r}')
        if st in VALID_VERDICTS:
            if not (r.get('reviewed_at') or '').strip():
                issues.append(f'{eid}: verdict {st} has blank reviewed_at')
            elif not _is_valid_iso(r['reviewed_at']):
                issues.append(f'{eid}: reviewed_at {r["reviewed_at"]!r} is not '
                              'ISO-8601')
            if not (r.get('review_notes') or '').strip():
                issues.append(f'{eid}: verdict {st} has blank review_notes')
    return issues


def cmd_check(args):
    rows = _load_rows(args.review_csv)
    issues = check_rows(rows)
    out = ['# Duplicate-review consistency check', '']
    out.append(f'Edges checked: {fmt_int(len(rows))}')
    out.append('')
    if issues:
        out.append(f'## {fmt_int(len(issues))} issue(s) found')
        out.append('')
        out.append('| # | issue |')
        out.append('| --- | --- |')
        for i, iss in enumerate(issues, 1):
            out.append(f'| {i} | {iss} |')
        out.append('')
        out.append('Every flagged row carries a verdict the tool cannot '
                   'attribute to a named human reviewer, or an internally '
                   'inconsistent status/reviewer/reviewed_at set. Fix these '
                   'before any re-freeze.')
    else:
        out.append('No consistency issues found. All verdicts are fully '
                   'attributed and all statuses are valid.')
    out.append('')
    _emit(args, '\n'.join(out) + '\n')
    print(f'issues={len(issues)}')
    return 1 if issues else 0


def classify_impact(rows, frozen):
    """Pure classifier over (review rows, frozen sample_id set).

    Returns a dict:
      return_candidates  -- endpoints whose edge is ``not_duplicate``
      remove_required    -- endpoints, both of a confirmed/near edge, in frozen
      remove_from_frozen -- the subset of remove_required currently frozen
    """
    return_candidates = set()
    remove_required = set()
    for r in rows:
        st = (r.get('review_status') or '').strip()
        a, b = r.get('sample_a'), r.get('sample_b')
        if st == 'not_duplicate':
            return_candidates.update({a, b})
        elif st in ('confirmed_duplicate', 'near_duplicate'):
            if a in frozen and b in frozen:
                remove_required.update({a, b})
    return {
        'return_candidates': return_candidates,
        'remove_required': remove_required,
        'remove_from_frozen': remove_required & set(frozen),
        'return_in_frozen': return_candidates & set(frozen),
        'return_outside_frozen': return_candidates - set(frozen),
    }


def cmd_impact(args):
    """Given CURRENT verdicts, classify frozen-row impact and emit re-freeze cmd."""
    rows = _load_rows(args.review_csv)
    train_ids, val_ids = _load_frozen_sets()
    frozen = train_ids | val_ids
    cls = classify_impact(rows, frozen)

    out = ['# Duplicate-review impact (given CURRENT verdicts)', '']
    out.append('This is a DRY-RUN. It does not modify `data/frozen/`.')
    out.append('')
    by_status = Counter(r['review_status'] for r in rows)
    recorded = sum(n for s, n in by_status.items() if s != PENDING)
    out.append('## Verdicts currently recorded')
    out.append('')
    out.append(f'Recorded (non-pending) verdicts: {fmt_int(recorded)} of '
               f'{fmt_int(len(rows))}.')
    out.append('')
    out.append('## Isolated endpoints that COULD be returned')
    out.append('')
    out.append('- endpoints on a `not_duplicate` edge: '
               f'{fmt_int(len(cls["return_candidates"]))}')
    out.append(f'  - already in the frozen manifest (no-op return): '
               f'{fmt_int(len(cls["return_in_frozen"]))}')
    out.append(f'  - still outside the manifest (pending_audit / exclude): '
               f'{fmt_int(len(cls["return_outside_frozen"]))}')
    out.append('')
    out.append('## Frozen rows that would have to be REMOVED')
    out.append('')
    out.append(f'- endpoints on a `confirmed_duplicate`/`near_duplicate` edge '
               f'where BOTH endpoints are in the frozen manifest: '
               f'{fmt_int(len(cls["remove_from_frozen"]))}')
    out.append('')
    if cls['remove_from_frozen']:
        out.append('**The current freeze would be INVALIDATED.** A re-freeze '
                   'under a new `data_version` is required.')
    else:
        out.append('No frozen row is currently invalidated by a recorded '
                   'duplicate verdict. Note: `near_duplicate` is recorded but '
                   'not treated as leakage by the conservative protocol; a '
                   'curator may still choose to remove such rows.')
    out.append('')
    bump = args.bump or 'agrifield40k-v1'
    out.append('## Re-freeze command (do NOT run unless a human approves)')
    out.append('')
    out.append('```bash')
    out.append(f'# next data version: bump agrifield40k-v0 -> {bump}')
    out.append('./.venv/bin/python -u -m tools.stage2_data_audit '
               f'--generated-at {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
    out.append('```')
    out.append('')
    out.append('After regenerating, record fresh SHA-256 digests in '
               '`data/frozen/freeze_manifest.json` and `reports/data_freeze.md`. '
               'Bumping `data_version` also requires editing the `DATA_VERSION` '
               'constant in `tools/stage2_data_audit.py` first.')
    out.append('')
    _emit(args, '\n'.join(out) + '\n')
    print(f'return_candidates={len(cls["return_candidates"])} '
          f'remove_from_frozen={len(cls["remove_from_frozen"])}')
    return 0


def cmd_refreeze_plan(args):
    bump = args.bump or 'agrifield40k-v1'
    out = ['# Post-adjudication re-freeze plan', '']
    out.append('This plan is informational. It is NOT executed by this tool.')
    out.append('')
    out.append(f'Target data version: `{bump}` (bump from `agrifield40k-v0`).')
    out.append('')
    out.append('## Steps')
    out.append('')
    steps = [
        'Finish human adjudication of all 784 edges in '
        '`data/duplicate_review.csv` (minimum bar: the 258 cross-split / '
        'manifest-affecting edges) via `duplicate_review_workflow.py record`.',
        'Run `duplicate_review_workflow.py check` and confirm zero issues (no '
        'unattributed verdicts, no unknown statuses).',
        'Run `duplicate_review_workflow.py impact` to confirm which isolated '
        'endpoints can return and whether any frozen rows must be removed.',
        'If any frozen row is invalidated, decide removal vs. replacement with '
        'the curator; never edit a frozen manifest in place.',
        'Promote the adopted prefixes (per `stage2_source_index_evidence.md`) '
        'so newly included rows are also duplicate-screened; this is a separate '
        'data-version decision owned by the curator.',
        f'Set `DATA_VERSION = "{bump}"` in `tools/stage2_data_audit.py`, then '
        'regenerate: `./.venv/bin/python -u -m tools.stage2_data_audit '
        '--generated-at <ISO-8601>`.',
        'Recompute and record SHA-256 of every frozen file in '
        '`data/frozen/freeze_manifest.json` and `reports/data_freeze.md`.',
        'Update `reports/data_freeze.md` gate status and the "unfinished items" '
        'list; if the source/license gate is now met, set `gate_met: true` and '
        'flip the freeze off PROVISIONAL.',
        'Record the change and its evidence in the Stage 3 baseline protocol '
        'before any training run consumes the new split.',
    ]
    for i, s in enumerate(steps, 1):
        out.append(f'{i}. {s}')
    out.append('')
    _emit(args, '\n'.join(out) + '\n')
    print(f'plan written for {bump}')
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--review-csv', default=str(REVIEW_CSV),
                    help='path to duplicate_review.csv (default data/duplicate_review.csv)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('status', help='print review counts')
    p.add_argument('--out', default='-', help='markdown output path (default stdout)')
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('record', help='record one or more fully-attributed verdicts')
    p.add_argument('--verdicts', required=True,
                   help='JSON file: {"edge-0001": {"review_status": ..., '
                        '"reviewer": ..., "review_notes": ..., "reviewed_at": ...}}')
    p.set_defaults(func=cmd_record)

    p = sub.add_parser('check', help='consistency/conflict check')
    p.add_argument('--out', default='-', help='markdown output path (default stdout)')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('impact', help='classify frozen-row impact (dry-run)')
    p.add_argument('--out', default='-', help='markdown output path (default stdout)')
    p.add_argument('--bump', default=None, help='next data version (default agrifield40k-v1)')
    p.set_defaults(func=cmd_impact)

    p = sub.add_parser('refreeze-plan', help='print re-freeze step list')
    p.add_argument('--out', default='-', help='markdown output path (default stdout)')
    p.add_argument('--bump', default=None, help='next data version (default agrifield40k-v1)')
    p.set_defaults(func=cmd_refreeze_plan)
    return ap


def main(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
