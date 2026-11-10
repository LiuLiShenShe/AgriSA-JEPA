# Stage 7B — 配对调度消融协议

- 生成：`tools/stage7b_protocol_check.py`
- 三臂仅允许在 `max_optimizer_steps` / `interrupt_after_optimizer_steps` / `warmup_steps` 上不同；其余字段继承冻结 base。

| 臂 | horizon | 实际执行 | warmup | LR@2000 | WD@2000 | EMA@2000 |
|---|---|---|---|---|---|---|
| A | 2,000 | 2,000 | 100 | 1.000e-06 | 0.400 | 1.0000 |
| B | 50,000 | 2,000 | 100 | 1.993e-04 | 0.041 | 0.9962 |
| C | 50,000 | 2,000 | 2,500 | 1.620e-04 | 0.041 | 0.9962 |

- LR 调度：warmup `1e-5→2e-4`，其后余弦 `2e-4→1e-6`（T_max=horizon−warmup）。
- WD 调度：余弦 `0.04→0.4`（T_max=horizon）。
- EMA：`0.996→1.0`（线性，T_max=horizon）。

### 臂间关系（2,000 停止点）

- **A vs B**：同 warmup，horizon 2000 vs 50000 → A 的 LR 已近余弦末端，B 的 LR 仍接近峰值。
- **B vs C**：同 horizon 50000，warmup 100 vs 2500 → C 在 2000 步仍处于 warmup（LR 仍在爬升），B 已越过 warmup。

- 协议检查：**通过**（0 个问题）

### 冻结输入

- train_view：`26c55431c63791d4f7d427d432840c21aa47a717a5daadbb5990a68a546f71f2`（通过）

