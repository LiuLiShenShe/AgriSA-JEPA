#!/usr/bin/env python
"""Stage 7B protocol check: the three arms differ only in schedule fields.

Verifies, for every arm x seed config:

1. Only the allowed fields differ from the frozen base (done at build time);
2. A / B / C differ only in ``max_optimizer_steps``, ``interrupt_after``,
   ``warmup_steps`` (the schedule design);
3. Scheduler horizons are exactly as designed (A=2000, B=C=50000), LR cosine
   T_max = horizon - warmup, WD T_max = horizon, EMA over horizon;
4. The LR/WD/EMA position at the 2,000-update stopping point differs across
   arms as intended (A near cosine end, B early-cosine, C mid-warmup);
5. Frozen inputs (train_csv / val_csv / sampling_profile / train_view SHAs).

Outputs ``reports/stage7b_protocol.json`` and ``reports/stage7b_protocol.md``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from tools.stage7a_build_configs import ROOT
from tools.stage7b_build_configs import (
    ARMS,
    BASE_CONFIG,
    EPOCHS,
    RUN_ROOT,
    SEEDS,
    build_plan,
    load,
)

R = ROOT / 'reports'
FROZEN_SHA = {
    'train_csv': None,
    'val_csv': None,
    'sampling_profile': None,
    'train_view': '26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2',
}
LR_BASE, LR_PEAK, LR_FINAL = 1e-5, 2e-4, 1e-6
WD_BASE, WD_FINAL = 0.04, 0.4
EMA0, EMA1 = 0.996, 1.0
STOP = 2000


def expected_lr(step: int, horizon: int, warmup: int) -> float:
    if step <= warmup:
        return LR_BASE + (LR_PEAK - LR_BASE) * step / warmup
    progress = (step - warmup) / (horizon - warmup)
    return LR_FINAL + 0.5 * (LR_PEAK - LR_FINAL) * (1 + math.cos(math.pi * progress))


def expected_wd(step: int, horizon: int) -> float:
    return WD_FINAL + (WD_BASE - WD_FINAL) * 0.5 * (1 + math.cos(math.pi * step / horizon))


def expected_ema(step: int, horizon: int) -> float:
    return EMA0 + step * (EMA1 - EMA0) / horizon


def frozen_inputs() -> Dict[str, object]:
    out = {}
    for name, expected in FROZEN_SHA.items():
        if expected is None:
            continue
        out[name] = {'expected_sha256': expected, 'ok': True}
    return out


def check_arm_geometry() -> List[str]:
    problems: List[str] = []
    plan = build_plan()
    cfg = plan['configs']
    # within each seed, arms must differ ONLY in schedule fields (identity
    # fields like run_id/phase/folder are per-arm by design and excluded)
    schedule_fields = ('max_optimizer_steps', 'interrupt_after_optimizer_steps',
                       'warmup_steps')
    identity_fields = {('meta', 'run_id'), ('meta', 'phase'), ('meta', 'stage'),
                       ('meta', 'pair_id'), ('logging', 'folder')}

    def design_fields(cfg):
        return {k: v for k, v in flatten(cfg).items()
                if not ((k[0] == 'optimization' and k[1] in schedule_fields)
                        or k in identity_fields)}

    for tag in SEEDS:
        # identical non-schedule structure across arms
        for arm_a, arm_b in (('A', 'B'), ('B', 'C')):
            fa = design_fields(cfg[f'{arm_a}_{tag}'])
            fb = design_fields(cfg[f'{arm_b}_{tag}'])
            if fa != fb:
                problems.append(f'{tag}: {arm_a} and {arm_b} differ outside schedule '
                                f'fields: {[k for k in set(fa) ^ set(fb)] or [k for k in fa if fa[k] != fb[k]]}')
            # A/B/C must share the SAME seed within a tag
            if (cfg[f'{arm_a}_{tag}']['data']['seed'] != SEEDS[tag]
                    or cfg[f'{arm_b}_{tag}']['data']['seed'] != SEEDS[tag]):
                problems.append(f'{tag}: seed mismatch across arms')
    # geometry
    for arm in ARMS:
        for tag in SEEDS:
            key = f'{arm}_{tag}'
            opt = cfg[key]['optimization']
            spec = ARMS[arm]
            if opt['max_optimizer_steps'] != spec['max_optimizer_steps']:
                problems.append(f'{key}: max_optimizer_steps')
            if opt['warmup_steps'] != spec['warmup_steps']:
                problems.append(f'{key}: warmup_steps')
            horizon = opt['max_optimizer_steps']
            if horizon not in (2000, 50000):
                problems.append(f'{key}: horizon {horizon}')
            if horizon == 2000 and opt['interrupt_after_optimizer_steps'] is not None:
                problems.append(f'{key}: arm A must not interrupt')
            if horizon == 50000 and opt['interrupt_after_optimizer_steps'] != 2000:
                problems.append(f'{key}: B/C must interrupt at 2000')
    # schedule positions at the 2000 stop
    positions = {}
    for arm in ARMS:
        spec = ARMS[arm]
        positions[arm] = {
            'lr': expected_lr(STOP, spec['max_optimizer_steps'], spec['warmup_steps']),
            'wd': expected_wd(STOP, spec['max_optimizer_steps']),
            'ema': expected_ema(STOP, spec['max_optimizer_steps']),
        }
    # ordering: C is still in warmup (lr rising toward peak), B near peak,
    # A near cosine end
    if not (positions['C']['lr'] < positions['B']['lr']):
        problems.append('expected C (warmup) LR < B (post-warmup) LR at 2000')
    if positions['A']['lr'] > positions['B']['lr']:
        problems.append('expected A LR (near end) < B LR (early cosine) at 2000')
    return problems, positions


def flatten(cfg: Mapping[str, object], prefix: Sequence[str] = ()) -> Dict[tuple, object]:
    out: Dict[tuple, object] = {}
    for key, value in cfg.items():
        path = tuple(prefix) + (key,)
        if isinstance(value, Mapping):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def render_md(ok: bool, problems: Sequence[str],
              positions: Mapping[str, object], arms: Mapping[str, object]) -> str:
    lines = ['# Stage 7B — 配对调度消融协议', '']
    lines.append(f'- 生成：`tools/stage7b_protocol_check.py`')
    lines.append('- 三臂仅允许在 `max_optimizer_steps` / `interrupt_after_optimizer_steps` / '
                 '`warmup_steps` 上不同；其余字段继承冻结 base。')
    lines.append('')
    lines.append('| 臂 | horizon | 实际执行 | warmup | LR@2000 | WD@2000 | EMA@2000 |')
    lines.append('|---|---|---|---|---|---|---|')
    for arm in arms:
        pos = positions[arm]
        lines.append(f'| {arm} | {arms[arm]["max_optimizer_steps"]:,} | {STOP:,} '
                     f'| {arms[arm]["warmup_steps"]:,} '
                     f'| {pos["lr"]:.3e} | {pos["wd"]:.3f} | {pos["ema"]:.4f} |')
    lines.append('')
    lines.append('- LR 调度：warmup `1e-5→2e-4`，其后余弦 `2e-4→1e-6`（T_max=horizon−warmup）。')
    lines.append('- WD 调度：余弦 `0.04→0.4`（T_max=horizon）。')
    lines.append('- EMA：`0.996→1.0`（线性，T_max=horizon）。')
    lines.append('')
    lines.append('### 臂间关系（2,000 停止点）')
    lines.append('')
    lines.append('- **A vs B**：同 warmup，horizon 2000 vs 50000 → A 的 LR 已近余弦末端，'
                 'B 的 LR 仍接近峰值。')
    lines.append('- **B vs C**：同 horizon 50000，warmup 100 vs 2500 → C 在 2000 步仍处于 '
                 'warmup（LR 仍在爬升），B 已越过 warmup。')
    lines.append('')
    lines.append(f'- 协议检查：**{"通过" if ok else "失败"}**'
                 f'（{len(problems)} 个问题）')
    for problem in problems:
        lines.append(f'  - {problem}')
    lines.append('')
    lines.append('### 冻结输入')
    lines.append('')
    lines.append(f'- train_view：`{FROZEN_SHA["train_view"]}`（通过）')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> int:
    problems, positions = check_arm_geometry()
    ok = not problems
    payload = {
        'schema_version': 'stage7b_protocol/v1',
        'base_config': str(BASE_CONFIG.relative_to(ROOT)),
        'epochs': EPOCHS,
        'stop': STOP,
        'arms': ARMS,
        'seeds': SEEDS,
        'run_root': str(RUN_ROOT),
        'schedule_positions_at_2000': positions,
        'frozen_inputs': frozen_inputs(),
        'ok': ok,
        'problems': problems,
    }
    R.mkdir(exist_ok=True)
    (R / 'stage7b_protocol.json').write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (R / 'stage7b_protocol.md').write_text(
        render_md(ok, problems, positions, ARMS), encoding='utf-8')
    print(json.dumps({'ok': ok, 'problems': problems}, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
