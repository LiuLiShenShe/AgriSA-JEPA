"""Capture-group derivation and split assignment for Stage 5D.

The Stage 5C module assigned ``GROUP_UNKNOWN`` images to train/val per image and
then reported ``groups_disjoint: true``.  That claim is false: an image-level
assignment proves nothing about scene independence.  This module separates the
two questions:

1. ``group_key`` with an explicit trust tier.  ``OFFICIAL_GROUP`` comes from a
   published index (Artemis S3 keys); ``DOCUMENTED_PATH_GROUP`` comes from a
   published filename convention (AgriField ``component``/observation id);
   ``WEAK_PROXY`` covers identity that is real but coarse (a photo id);
   ``UNKNOWN_GROUP`` means nothing is known.
2. ``split``.  Only strong groups may produce ``GROUP_DISJOINT_VAL``.  Unknown
   groups become ``TRAIN_ONLY_GROUP_UNKNOWN`` unless the caller explicitly asks
   for a diagnostic random val, which is a *different* object and is named
   ``DIAGNOSTIC_RANDOM_VAL``.

The Stage 2D repairs are preserved rather than re-derived:

* **ACRECropWeed** stays a single capture campaign and is never split (the
  robot run continued across the old split boundary).
* **iNatWeeds** is split by observation id parsed from ``observation_url``;
  images without an observation id are forced to train.

The split is a pure function of the sorted group ids and a salt, so two builds
on the same rows always agree.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .stage5d_metadata_schema import (GROUP_DOCUMENTED_PATH, GROUP_OFFICIAL,
                                      GROUP_UNKNOWN, GROUP_WEAK_PROXY)

SPLIT_TRAIN = 'train'
SPLIT_VAL = 'val'
SPLIT_TRAIN_ONLY_UNKNOWN = 'TRAIN_ONLY_GROUP_UNKNOWN'
SPLIT_DIAGNOSTIC_RANDOM_VAL = 'DIAGNOSTIC_RANDOM_VAL'

STRONG_GROUPS = {GROUP_OFFICIAL, GROUP_DOCUMENTED_PATH}

OBSERVATION_RE = re.compile(r'/observations?/(\d+)')


@dataclass(frozen=True)
class GroupAssignment:
    group_key: str
    tier: str
    evidence: str
    confidence: float
    source_dataset: str = ''


def _bucket(text: str, salt: str) -> int:
    digest = hashlib.sha256((salt + '|' + text).encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % 100


def assign_splits(groups: Sequence[GroupAssignment], val_fraction: float,
                  salt: str) -> Dict[str, str]:
    """Assign a split per group id.

    Only ``OFFICIAL_GROUP`` / ``DOCUMENTED_PATH_GROUP`` receive ``val``.
    ``WEAK_PROXY`` and ``UNKNOWN_GROUP`` go to train, or to
    ``TRAIN_ONLY_GROUP_UNKNOWN`` when nothing at all is known.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError('val_fraction must be in [0, 1)')
    ids = sorted({g.group_key for g in groups})
    ranked = sorted(ids, key=lambda g: hashlib.sha256(
        f'{salt}|group|{g}'.encode('utf-8')).hexdigest())
    n_val = max(1, int(round(len(ranked) * val_fraction))) if val_fraction else 0
    val_groups: Set[str] = set(ranked[:n_val])

    by_id: Dict[str, GroupAssignment] = {g.group_key: g for g in groups}
    out: Dict[str, str] = {}
    for group_id in ids:
        tier = by_id[group_id].tier
        if tier in STRONG_GROUPS:
            out[group_id] = SPLIT_VAL if group_id in val_groups else SPLIT_TRAIN
        elif tier == GROUP_WEAK_PROXY:
            out[group_id] = SPLIT_TRAIN
        else:
            out[group_id] = SPLIT_TRAIN_ONLY_UNKNOWN
    return out


def assign_diagnostic_random(assignments: Sequence[GroupAssignment],
                             val_fraction: float, salt: str,
                             strong_only: bool = True) -> Dict[str, str]:
    """Image-level random val used ONLY for the diagnostic split.

    ``strong_only=True`` keeps unknown-group images out of even the diagnostic
    val unless the caller overrides it, in which case the caller must label the
    result ``DIAGNOSTIC_RANDOM_VAL`` and never ``GROUP_DISJOINT_VAL``.
    """
    eligible = [g for g in assignments
                if (g.tier in STRONG_GROUPS or not strong_only)]
    ids = sorted({g.group_key for g in eligible})
    ranked = sorted(ids, key=lambda g: hashlib.sha256(
        f'{salt}|diag|{g}'.encode('utf-8')).hexdigest())
    n_val = max(1, int(round(len(ranked) * val_fraction))) if val_fraction else 0
    val_groups = set(ranked[:n_val])
    return {group_id: (SPLIT_DIAGNOSTIC_RANDOM_VAL if group_id in val_groups
                       else SPLIT_TRAIN)
            for group_id in ids}


# --------------------------------------------------------------------------- #
# Group derivation per physical source.
# --------------------------------------------------------------------------- #
def agrifield_group(row: Dict[str, object]) -> GroupAssignment:
    """Group an AgriField-40K image using its published component metadata.

    ``component`` is the curated group column written by the AgriField audit;
    it encodes the capture unit (iNaturalist observation, robot run, LUCAS
    point, MuST-C plot/day/position, PhenoBench mission).  The prefix decides
    which upstream dataset the image came from, so two paper mirrors of the
    same scene are not merged by a shared string alone.
    """
    component = str(row.get('component') or '').strip()
    prefix = str(row.get('source_prefix') or '').strip()
    dataset = str(row.get('upstream_dataset') or '').strip()
    if not component:
        return GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, dataset)

    if prefix == 'inat':
        url = str(row.get('observation_url') or '')
        match = OBSERVATION_RE.search(url)
        if match:
            return GroupAssignment(
                f'inat|observation|{match.group(1)}', GROUP_DOCUMENTED_PATH,
                'observation_url_parsed', 0.95, 'iNatWeeds')
        # Stage 2D: no observation id -> forced to train, never val.
        return GroupAssignment(f'inat|photo|{component}', GROUP_WEAK_PROXY,
                               'component_without_observation_url', 0.35,
                               'iNatWeeds')

    if prefix == 'acw':
        # Stage 2D: one continuous robot campaign; the source is never split.
        return GroupAssignment('acw|single_campaign', GROUP_DOCUMENTED_PATH,
                               'stage2d_single_campaign', 0.9, 'ACRECropWeed')

    if prefix == 'pnb':
        return GroupAssignment(f'phenobench|mission|{component}',
                               GROUP_DOCUMENTED_PATH, 'component_mission_id',
                               0.9, 'PhenoBench')
    return GroupAssignment(f'{prefix}|{component}', GROUP_DOCUMENTED_PATH,
                           'agrifield_component_column', 0.85, dataset)


def agri_jepa_group(row: Dict[str, object]) -> GroupAssignment:
    """Group an AppleBBCH81 image by its published tree/session filename."""
    import re as _re
    name = str(row.get('filename') or '')
    match = _re.search(r'DSC[_-]?(?P<tree>\d+)[_-](?P<session>\d+)', name)
    if match:
        return GroupAssignment(
            f'applebbch81|tree{match.group("tree")}|session{match.group("session")}',
            GROUP_DOCUMENTED_PATH, 'filename_tree_session', 0.9, 'AppleBBCH81')
    return GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, 'AppleBBCH81')


def simple_group(row: Dict[str, object], tier: str, evidence: str,
                 template: str) -> GroupAssignment:
    """Group by a documented field such as an official dataset id."""
    key = template.format(**{k: row.get(k, '') for k in
                             _template_fields(template)})
    if not key or '{' in key:
        return GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, '')
    return GroupAssignment(key, tier, evidence, 0.9, '')


def _template_fields(template: str) -> Set[str]:
    import string
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def derive_groups(frame, group_rules: Dict[str, object]) -> List[GroupAssignment]:
    """Apply the per-source group rule declared in the policy."""
    out: List[GroupAssignment] = []
    for row in frame.to_dict('records'):
        source = str(row.get('physical_source_id') or '')
        rule = group_rules.get(source)
        if rule is None:
            out.append(GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, ''))
            continue
        kind = str(rule.get('kind'))
        if kind == 'artemis_key':
            key = str(row.get('group_key') or '')
            if key:
                out.append(GroupAssignment(key, GROUP_OFFICIAL,
                                           'published_s3_key', 1.0,
                                           str(row.get('original_dataset_id') or '')))
            else:
                out.append(GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, ''))
        elif kind == 'agrifield_component':
            out.append(agrifield_group(row))
        elif kind == 'filename':
            out.append(agri_jepa_group(row))
        elif kind == 'constant':
            out.append(GroupAssignment(str(rule['group_key']), GROUP_DOCUMENTED_PATH,
                                       'documented_constant', 0.9,
                                       str(rule.get('dataset', ''))))
        else:
            out.append(GroupAssignment('', GROUP_UNKNOWN, 'MISSING', 0.0, ''))
    return out


def cross_split_group_count(frame, group_col: str = 'group_key',
                            tier_col: str = 'group_tier',
                            split_col: str = 'split') -> int:
    strong = frame[frame[tier_col].isin(STRONG_GROUPS)]
    if strong.empty:
        return 0
    return int(strong.groupby(group_col)[split_col].nunique().gt(1).sum())
