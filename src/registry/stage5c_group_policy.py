"""Capture-group derivation and leakage-safe split assignment.

A group is only claimed when the metadata supports it.  ``GROUP_UNKNOWN`` is a
first-class value: it is reported, and those images are split individually,
which is the conservative choice for isolation (an unknown group cannot be
*proven* independent, so it never contributes a second "scene").

Whole groups move to one split.  Holdout sources are never split into R2/R3 at
all: they are blocked at the source level before this module is consulted.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

GROUP_UNKNOWN = 'GROUP_UNKNOWN'

#: Filename patterns that encode a plant / plot / session identity.
PATTERNS = [
    # AppleBBCH81: DSC_<tree>_<session>_<view>.jpg
    ('applebbch81', re.compile(r'(DSC[_-]?(?P<tree>\d+)[_-](?P<session>\d+))')),
    # PlantVillage: <species>___<plant id>.<ext>
    ('plantvillage', re.compile(r'___(?P<plant>[^.]+)\.')),
    # AgriField LUCAS: lvb<nn>_LUCAS<year>_<point>_Cover
    ('lucas', re.compile(r'(?P<survey>lvb\d+)_LUCAS\d+_(?P<point>\d+)_')),
    # MuST-C: mc_plot<id>_day<yymmdd>_pos<nn>
    ('mustc', re.compile(r'(?P<plot>plot\d+)_(?P<day>day\d+)_')),
    # PhenoBench: pnb_<date>_<split>_<type>_<mission>_<x>_<y>
    ('phenobench', re.compile(r'pnb_(?P<date>\d{8})_\w+_(?P<type>\w+)_(?P<mission>P\d+)')),
    # VCD: vcd_<date>_<site>_<crop>_<plot>_<idx>
    ('vcd', re.compile(r'vcd_(?P<date>[\d-]+)_(?P<site>[^_]+)_(?P<crop>[^_]+)_(?P<plot>\d+)')),
]

#: Artemis paths encode farm/session/plot/date directly.
ARTEMIS_GROUP = re.compile(r'/(?P<crop>[^/]+)/images/(?P<farm>[^/]+)/.*?/(?P<date>\d{4}-\d{2}-\d{2})/plot-(?P<plot>\d+)/')


def derive_group(source_id: str, rel_path: str,
                 row: Optional[Dict[str, object]] = None) -> Tuple[str, bool, str]:
    row = row or {}
    for field in ('session_id', 'plant_id', 'plot_id', 'farm_id'):
        value = row.get(field)
        if value not in (None, '', 'nan'):
            return f'{field}:{value}', True, field
    name = Path(rel_path).name
    if source_id == 'artemis_imagesafari':
        match = ARTEMIS_GROUP.search(rel_path.replace('\\', '/'))
        if match:
            key = 'artemis|{crop}|{farm}|{date}|plot-{plot}'.format(**match.groupdict())
            return key, True, 'path_metadata'
    for tag, pattern in PATTERNS:
        match = pattern.search(name)
        if match:
            group = '|'.join(f'{k}={v}' for k, v in sorted(match.groupdict().items()) if v)
            return f'{tag}|{group}', True, 'filename_metadata'
    if source_id == 'agrifield40k_pool':
        # Prefix + numeric original id is only a coarse identity, so keep it
        # explicit rather than pretending it is a session.
        prefix = name.split('_', 1)[0]
        stem = Path(name).stem
        return f'agrifield|{prefix}|{stem}', True, 'filename_identity'
    return GROUP_UNKNOWN, False, 'none'


def _bucket(text: str, salt: str) -> int:
    digest = hashlib.sha256((salt + '|' + text).encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(group_key: str, group_known: bool, image_id: str,
                 salt: str = 'agrisa-stage5c-groups-v1',
                 val_fraction: float = 0.15) -> str:
    """Deterministic split; whole groups stay together."""
    basis = group_key if group_known else f'image:{image_id}'
    threshold = int(round(val_fraction * 100))
    return 'val' if _bucket(basis, salt) < threshold else 'train'
