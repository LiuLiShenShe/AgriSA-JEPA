# Stage 5A — Adapter 可行性与实验设计

> 状态：**设计完成 + 非侵入式烟雾测试已执行**；正式 Adapter 对照实验 **NOT RUN**（安排在 Stage 6）。
> 代码：`src/downstream/adapters.py`；测试：`tests/test_stage5a_adapters.py`（6 passed）。
> 本文件不改变当前训练主线、不替换任何 checkpoint、不修改 `src/models/vision_transformer.py`。

## 0. 结论先行

1. **Downstream Adapter + Head 可行**：零初始化残差 Adapter 在真实 I-JEPA ViT-H/14 上
   初始输出与原始 Encoder 一致（实测 `allclose` 通过），只训练 Adapter + Head。
2. 参数量极小：32 个 Block 全插 Adapter 时，bottleneck 8/16/32 分别约
   **0.70M / 1.35M / 2.66M** 可训练参数，占 632M 主干的 0.11%–0.42%。
3. 224×224、fp32 实测前向开销约 **2.7%**（48.5 → 47.2 img/s，bs=4）。
4. **Adapter 一旦参与梯度，Frozen-Encoder 特征缓存立即失效**；必须把 Adapter 配置
   写进 cache key，否则会静默复用错误特征。
5. 建议 Stage 6 先做「Adapter + Head」小规模对照；「Adapter-JEPA 预训练」作为后续，
   不作为 Stage 5 主线前置条件。

## 1. 已执行的烟雾测试（真实 ViT-H/14）

环境：RTX A6000（仅 GPU1），fp32，bs=4，224×224。

| bottleneck | 插入 Block 数 | 可训练参数 | 初始恒等 | 前向吞吐 |
|---:|---:|---:|---|---:|
| 8  | 32 | 0.70 M | ✅ | 47.4 img/s |
| 16 | 32 | 1.35 M | ✅ | 47.2 img/s |
| 32 | 32 | 2.66 M | ✅ | 47.2 img/s |
| 无 Adapter | — | 0 | — | 48.5 img/s |

恒等性判定：Adapter 关闭与开启（零初始化）两次前向 `torch.allclose(atol=1e-5)`。

单元测试另外验证了：

- 关闭 Adapter 后输出与**未插入 Adapter 的原始 Encoder** 逐元素相等；
- 反向传播后主干权重逐元素不变，且主干 `grad is None`；
- 仅 Adapter 参数 `requires_grad=True`；
- 一次优化器 step 后 Adapter 参数确实变化；
- 非法层索引被拒绝。

## 2. 方案 A — Downstream Adapter + Head

### 结构

在选定的 Transformer Block **输出残差**上插入 bottleneck Adapter：

```
x = Block(x)
x = x + W_up · GELU(W_down · x)      # W_up 零初始化
```

- 默认插入全部 32 个 Block；也可只插后 12 层（成本更低）。
- 主干（Encoder）完全冻结；Neck/Detection Head 或 Linear Probe 照常训练。
- Adapter 与 Head 使用同一优化器或两个参数组（建议 Adapter lr 更小）。

### 必须验证的项目

| 检查 | 状态 | 证据 |
|---|---|---|
| 原始主干权重不变 | ✅ | `test_backbone_unchanged_after_backward` |
| Adapter 关闭 ≡ 原始输出 | ✅ | `test_adapter_disabled_restores_original_forward` |
| 新旧 checkpoint 独立加载 | ✅（设计） | Adapter 单独保存 `adapters.pth` + config；主干 checkpoint 不变 |
| 可训练参数量 | ✅ | 0.70–2.66 M（32 blocks） |
| 训练显存 | 预估 | 与冻结主干 + Head 基本相同；Adapter 梯度可忽略（<0.1 GB） |
| 推理成本 | ✅ | +2.7%（fp32, 224） |
| 旧 PhenoBench 评价不被破坏 | 设计约束 | 不启用 Adapter 时走原路径；新增回归测试只验证“关闭即原样” |

### 显存与缓存

- **不能复用 Frozen Encoder 的预计算特征**：Adapter 在主干内部改变每层激活，
  缓存 key 必须包含 `adapter_layers / bottleneck / enabled`。
- 若仍要缓存，只能缓存 **Adapter 输出**，其大小与 patch tokens 相同
  （PhenoBench：1,407+772 图 × 25 tiles × 256 × 1280 × 2 B ≈ 35 GB）；
  建议先不做缓存，直接在线前向。

### Checkpoint 协议

```
checkpoints/
  encoder.pth            # 原始 I-JEPA / 农业 JEPA，不变
  adapters.pth           # {config, state_dict}，严格加载
  head.pth               # 下游 Head
```

- 加载顺序：主干严格加载 → Adapter 严格加载 → Head；
- 缺失 Adapter 时拒绝启动 Adapter 实验，而不是随机初始化后继续；
- `adapter.enabled=False` 时评估必须与主干原模型一致。

## 3. 方案 B — 自监督 Adapter-JEPA（Stage 6+）

### 结构

- Context Encoder：冻结主干 + 可训练 Adapter；
- Target Encoder：冻结主干 + **EMA Adapter**（从 Context Adapter 初始化，按 I-JEPA
  的 EMA 规则更新，不接收梯度）；
- Predictor：保持原 I-JEPA Predictor（可训练），或加 Adapter；
- 目标函数、Mask（官方 Random Multi-Block）、块尺寸、EMA 动量：**全部不变**。

### 需要处理的问题

1. **EMA Adapter**：Target 侧 Adapter 参数必须与 Context 侧同构；EMA 只更新 Adapter，
   主干保持官方权重。若 Target 不更新 Adapter，目标会与 Context 表示漂移。
2. **Optimizer 参数组**：Adapter（Context）与 Predictor 分开设 lr / weight decay；
   Target 不进入 optimizer。
3. **Checkpoint / Resume**：必须同时保存 Context Adapter、EMA Target Adapter、
   Predictor、optimizer、scaler、RNG、dataloader generator、mask step；
   任何一项缺失都拒绝 resume。
4. **容量与预算**：Adapter 参数量小，2000-step 预算下可能欠拟合；Stage 6 应先做
   500-step 烟雾测试，再决定是否扩到 2000。
5. **评估**：Adapter-JEPA 的 Encoder 不能再走“Frozen Encoder 特征缓存”路径；
   PhenoBench/GWHD 都必须在线前向，cache key 需包含 Adapter 状态。

### Stage 6 建议的最小对照（4 臂）

| Arm | 预训练 | 下游 |
|---|---|---|
| A0 | 官方 I-JEPA（冻结） | Head |
| B0 | 农业 JEPA 2k（冻结） | Head |
| A1 | 官方 I-JEPA + Adapter（冻结主干） | Adapter + Head |
| B1 | 农业 JEPA 2k + Adapter（冻结主干） | Adapter + Head |

先做 A0/B0 vs A1/B1（不重训预训练），确认 Adapter 是否值得；再决定是否做
Adapter-JEPA 预训练。

## 4. 风险与边界

- Adapter 不是“免费提升”：它增加可训练容量，可能在小数据上过拟合；
  必须与同参数量级的 Head-only / 全 Head 微调对照。
- 不得把 Adapter 带来的提升解释为“自监督算法更优”。
- 不得在 Adapter 实验中使用与主线不同的数据划分或测试集调参。
- 本阶段没有运行任何 Adapter 下游训练；上表数字只有吞吐与参数量，没有精度。

## 5. 结论

Adapter 值得进入 Stage 6 的正式小规模对照，但**不应阻碍** Stage 5 的数据建设与
GWHD/PhenoBench 下游评价。当前主线的 Frozen Encoder + Head 仍是唯一已产生真实
A/B 结果的协议。
