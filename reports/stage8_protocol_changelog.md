# Stage 8 — 协议变更日志（Phase 2 冻结之后）

冻结协议规定：任何偏差必须以新行追加并记录原因，不得原地改写。本文件
记录 Phase 2 冻结（83 行注册表）之后的全部实现层偏差及其理由。

## 2026-11-07 — cropandweed_sem OOM 修复（probe 实现层）

- **偏差**：`cropandweed_probe` 的训练数据加载方式改变。
  - 原（冻结时实现）：一次性载入全部训练 token 张量（136,980 × 256 × 1280
    fp16 ≈ 89.8 GB）→ 2026-11-07 OFFICIAL_seed0 实际运行 OOM（exit 137，
    宿主 125 GB RAM），任务不可执行。
  - 新：分块流式加载（每块 8 张图 ≈ 0.25 GB，从 NVMe 特征缓存读）。
  - 影响域：mini-batch 顺序由"全局 tile 乱序"改为"图像级乱序+块内乱序"。
  - 不变性：损失函数（patch CE over 非忽略像素）、batch size（32）、
    epochs（10）、优化器（AdamW, lr 1e-3, wd 0）、特征缓存绑定
    （cache key = label + checkpoint SHA-256）全部不变。
  - 结果 JSON 的 protocol 字段记录 `training_loader`，可追溯。
- **修复**：同日发现的 phase4_all.sh 启动 bug —— `--roles exploratory`
  与状态 `FROZEN_EXPLORATORY` 精确匹配失败，C 臂 6+6+6 行静默跳过
  （deepweeds/cropandweed/gwhd 的 C_p1/C_p2 从未运行）。已改为
  `--roles exploratory,frozen_exploratory`。注册表本身无改动。
- **附带**：新增 `tools/stage8_prune_feature_cache.py`（NVMe 空间管理，
  每 backbone ~102 GB 缓存 × 7 > 217 GB 可用；仅当该 backbone 的全部
  注册行都有 result.json 时才删除其缓存）。

以上均为可执行性 / 基础设施修复，不改变任何实验的科学内容或比较方向。
