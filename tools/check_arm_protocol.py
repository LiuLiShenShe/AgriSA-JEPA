#!/usr/bin/env python
"""Assert the Arm A / Arm B downstream configs differ ONLY in the backbone.

A downstream comparison is only interpretable if every non-backbone knob is
identical.  This tool diffs the two YAML files field by field and fails on any
difference outside the documented arm-identity keys.

Usage:
    ./.venv/bin/python -m tools.check_arm_protocol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

#: The only keys allowed to differ between the two arm configs.
ALLOWED_DIFFERENT = {
    ('meta', 'arm'),
    ('meta', 'arm_label'),
    ('meta', 'backbone_checkpoint'),
    ('meta', 'backbone_checkpoint_sha256'),
    ('meta', 'continued_training'),
    ('meta', 'pretraining_data_version'),
    ('outputs', 'out_dir'),
    ('outputs', 'report'),
}


def flatten(node, prefix=()):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, prefix + (str(key),)))
    else:
        out[prefix] = node
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', default='configs/phenobench_armA_linear.yaml')
    parser.add_argument('--b', default='configs/phenobench_armB_linear.yaml')
    args = parser.parse_args()

    flat_a = flatten(yaml.safe_load(Path(args.a).read_text()))
    flat_b = flatten(yaml.safe_load(Path(args.b).read_text()))

    problems = []
    for key in sorted(set(flat_a) | set(flat_b)):
        if key not in flat_a:
            problems.append(f'{key}: missing from A')
            continue
        if key not in flat_b:
            problems.append(f'{key}: missing from B')
            continue
        if flat_a[key] != flat_b[key] and key not in ALLOWED_DIFFERENT:
            problems.append(
                f'{key}: A={flat_a[key]!r} B={flat_b[key]!r}')

    # the two arms must actually differ in the backbone, or the comparison is void
    if flat_a[('meta', 'backbone_checkpoint')] == flat_b[('meta', 'backbone_checkpoint')]:
        problems.append('both arms point at the same backbone checkpoint')

    if problems:
        print('ARM PROTOCOL MISMATCH')
        for line in problems:
            print('  ' + line)
        return 1
    print(f'ARM PROTOCOL OK: {len(flat_a)} fields, only arm identity differs')
    print(f"  A backbone: {flat_a[('meta', 'backbone_checkpoint')]}")
    print(f"  B backbone: {flat_b[('meta', 'backbone_checkpoint')]}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())