# Stage 6C — PhenoBench A / B1 / B2 冻结线性探针评测（验收报告）

**分支**：`agri-stage6c-phenobench-eval`
**基线 HEAD**：`f057721530d687fc37cba864106b39f484298ca5`（远端 HEAD 相同，已核实）
**代码 commit（本阶段最终）**：见第 12 节
**设备**：物理 GPU1（RTX A6000，`CUDA_VISIBLE_DEVICES=1`，进程内 `cuda:0`）
**预训练语料状态**：R3v3_license_verified train，147,277 张，审计状态 **CONDITIONAL**（未更改）

---

## 1. A / B1 / B2 checkpoint 路径与 SHA-256

| arm | 路径 | SHA-256（期望 = 记录 = 磁盘） |
|---|---|---|
| A | `/data/checkpoints/ijepa/IN1K-vit.h.14-300e.pth.tar` | `0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22` |
| B1 | `runs/stage6_b1_natural_2000/jepa-encoder-step2000.pth` | `19796cebc65e29a07eafd7222f5a642d4ef9f28c33f2e4f4f2b65e34a2bfde6b` |
| B2 | `runs/stage6_b2_capped_no_backfill_2000/jepa-encoder-step2000.pth` | `6e7a83c9156e556757016145fb804e2b67e7b94f5e8b45bda4ffd24cb5975a62` |

- A 的 SHA 在开始前即与预期一致；B1/B2 磁盘 SHA 与 `reports/stage6_run_manifest_b1.json` /
  `stage6_run_manifest_b2.json` 记录的 encoder SHA 逐字节相同。三者均经
  `build_encoder(..., strict=True)` 严格加载成功（630,762,240 参数，256 tokens/tile）。
- 未误用 full training checkpoint 或旧 Stage 3B/4B/5A checkpoint。

## 2. A baseline：复用还是重跑？可比性证据

**复用 Stage 3A Arm A 的同协议历史结果**，未重跑。

可比性证据（全部机器可读、当前在盘可验）：
- `runs/stage3a_phenobench/armA/result.json` 存在且可解析；
- A checkpoint SHA 与预期一致，当前在盘复算一致；
- 协议字段（tiling/head/optimizer/LR/WD/epochs/batch/seeds/train-val 数）与 Stage 6C
  固定协议逐字段相同（由 `tools/compare_stage6c_phenobench.py` 重新断言）；
- 3 个 probe seeds 齐全；每 seed 772 张 val 预测，文件名与官方 val 列表一致；
- 官方评价器原始输出齐全；自实现/官方指标最大偏差 < 0.01 pp；
- 数据版本与清单 SHA 与本轮相同。

=> 这是**已核实的同协议 reference**。详见 `reports/stage6c_phenobench_A_official.md`。

## 3. 数据版本、文件清单 SHA、train/val 数量

- 版本：**PhenoBench v1.1.0**，路径 `/data/datasets/PhenoBench/PhenoBench-v110`。
- 文件清单：`reports/phenobench_file_manifest.sha256.txt`
  （SHA-256 `13fa2810d9fe87c2d65b2dc5b7c7e8ac55e31812aee44af21aea98da8e98bd5d`），
  13,074 项，逐文件尺寸校验 **0 不符**。本地重算清单 SHA 与审计记录一致。
- 数量：**train 1,407 / val 772 / test 693（无公开标注，未使用）**。
- 只用 train 训练 probe，只用 val 评价；PhenoBench 不进入预训练语料。

## 4. 三臂是否使用完全相同的评价协议？

**是。** 三臂除 backbone checkpoint 与 arm/pretraining metadata 外，协议逐字段一致，
并由比较脚本在聚合前断言拒绝任何偏差。固定项：ViT-H/14 冻结、224×224 tiles
（stride 200，5×5，无 padding，全覆盖，重叠 logits 平均）、Linear(1280→3)、
类别映射 3→1/4→2、AdamW LR 1e-3 WD 0、batch 32、10 epochs、seeds 0/1/2、
特征 BF16、head FP32、官方评价器 + 自实现指标交叉核验。

## 5. 每 arm、每 probe seed 的 soil/crop/weed IoU 与 mIoU（%）

| arm | seed | Soil | Crop | Weed | mIoU |
|---|---:|---:|---:|---:|---:|
| A | 0 | 97.6186 | 81.66 | 38.72 | 72.67 |
| A | 1 | 97.6104 | 81.46 | 38.74 | 72.60 |
| A | 2 | 97.5964 | 81.25 | 30.48 | 69.78 |
| B1 | 0 | 97.6899 | 82.35 | 40.35 | 73.46 |
| B1 | 1 | 97.6937 | 82.29 | 40.53 | 73.50 |
| B1 | 2 | 97.6757 | 82.01 | 35.12 | 71.60 |
| B2 | 0 | 97.6951 | 82.37 | 40.21 | 73.43 |
| B2 | 1 | 97.6913 | 82.15 | 39.95 | 73.27 |
| B2 | 2 | 97.6763 | 82.16 | 33.87 | 71.23 |

（soil 来自自实现 `IoUAccumulator`；crop/weed/mIoU 来自官方 `phenobench-eval`。）

## 6. mean±std、B1−A、B2−A、B2−B1（百分点）

| 项 | Soil | Crop | Weed | mIoU |
|---|---:|---:|---:|---:|
| A mean±std | 97.6085±0.0092 | 81.4567±0.1674 | 35.98±3.8891 | 71.6833±1.3462 |
| B1 mean±std | 97.6864±0.0077 | 82.2167±0.1482 | 38.6667±2.5089 | 72.8533±0.8864 |
| B2 mean±std | 97.6876±0.0081 | 82.2267±0.1014 | 38.01±2.9293 | 72.6433±1.0015 |
| B1−A mean | +0.08 | +0.76 | +2.69 | **+1.17** |
| B2−A mean | +0.08 | +0.77 | +2.03 | **+0.96** |
| B2−B1 mean | +0.00 | +0.01 | −0.66 | **−0.21** |

逐 seed 配对差值见 `reports/stage6c_phenobench_comparison.csv` 与 `.md`。

## 7. 官方评价器与 self metric 最大偏差

**最大偏差 0.0050 pp**（< 0.01 pp 容差），逐 seed、逐 arm 均通过交叉核验
（crop/weed/mIoU 行取官方，soil 行由 mIoU 恒等式间接核验）。

## 8. 是否完整评价 772 张 val 图像？

**是。** 三臂共 3×3=9 组结果，每组均生成 772 张预测 PNG，文件名与官方 val 列表
逐张比对一致；比较脚本在缺图时**拒绝汇总**。官方评价器遍历全部 772 张 GT 计算指标。

## 9. 缓存是否 SHA 绑定、删除了哪个精确目录？

- 缓存键由 `arm-label + checkpoint SHA[:16]` 绑定（`src/downstream/features.cache_key`），
  绝不复用旧 Stage 3B/4B/5A 特征。
- B1 完成并核验后**精确删除**：`runs/stage6c_phenobench/cache/stage6c_b1_natural__19796cebc65e29a0/`
  （34 GB）。
- B2 完成并核验后**精确删除**：`runs/stage6c_phenobench/cache/stage6c_b2_capped_no_backfill__6e7a83c9156e5567/`
  （34 GB）。
- 未通配删除 cache root；未删除用户数据或历史产物。删除后 `runs/stage6c_phenobench/cache/`
  为空目录。

## 10. 哪些结论得到支持，哪些仍不能支持？

**得到支持（直接实测 + 同协议复用）：**
- B1、B2 相对 A 在 3/3 probe seeds 上 mIoU 为正（+1.17 / +0.96 pp），且由
  crop（+0.76 / +0.77）与 weed（+2.69 / +2.03）驱动，soil 几乎不变——收益**不是**
  由占比最大的 soil 类驱动。
- 三臂优化路径可比（loss / probe 耗时 / 显存见比较报告）；未修改协议。

**仍不能支持（受每个 arm 单预训练 checkpoint 限制）：**
- 不能宣称自然采样或 capped no-backfill 已被证明“优越”：每种采样只有一个
  预训练 checkpoint；3 个 probe seeds 只衡量分割头/训练随机性，不估计预训练方差。
- B2−B1 为**轻微负向且类间方向不一致**（mIoU −0.21 pp；weed 3/3 为负，soil/crop
  几近持平）；|差值| 小于两 arm 的 probe-seed std，不能判定采样策略优劣。
- 无法基于本轮断定 GWHD 检测或更长训练一定会带来收益；需要独立预训练 seeds 先复现 B>A。

## 11. 新增测试、全套件测试、inherited failure

- 新增 `tests/test_stage6c_phenobench.py`：**11 passed**（详见
  `reports/stage6c_phenobench_test_summary.md`）。
- 全套件：**772 passed / 7 skipped / 1 failed**。
- 唯一 failed 为既有 `tests/test_stage5e1_evidence.py::test_artifact_manifest_agrees_with_git`
  （Stage 5E.1 历史产物不一致），按约束不修、不删、不放宽断言——记为 **inherited
  baseline failure**。本阶段**未引入新失败**。skipped 7 项逐条列出，未计入 passed。

## 12. 分支、commit SHA、远端 HEAD、工作树状态

- 分支：`agri-stage6c-phenobench-eval`（从 `agri-stage6-controlled-pretraining`
  HEAD `f0577215…` 切出）。
- 本阶段代码/测试/报告 commit：`42c1a2558eca166dad2fd42b56a9e3a3d314f7ce`。
- 元数据 commit 与远端 HEAD：见 `reports/stage6c_commit.txt` 的实时记录
  （后续更新只改本文件/比较产物，不改代码）。
- 远端 HEAD（执行前核实）：`f057721530d687fc37cba864106b39f484298ca5`（与本地一致）。
- 工作树状态：保留未跟踪的 `CLAUDE.md` 及其他用户未提交文件；新增代码/测试/报告
  已提交并 push；`runs/` 下原始图像、预测 PNG、特征缓存、checkpoint 按 `.gitignore`
  不提交。

## 13. 下一步应优先 GWHD 检测还是独立预训练 seed？

**优先独立预训练 seeds（≥2 个额外 seed/arm），再决定是否进入 GWHD 检测或更长训练。**
理由：
- 当前每个 arm 只有一个预训练 checkpoint，B1/B2 相对 A 的 +1.17/+0.96 pp mIoU
  虽在 3/3 probe seeds 同向，但仍可能是单一预训练随机性的偶然结果；先用独立
  seeds 复现 B>A 才能确认迁移信号可信。
- 在确认迁移信号稳定前，进入 GWHD 检测（目标域迁移）或更长训练（2,000 → 更多
  updates）会放大尚未被证明的方差风险。
- 一旦独立 seeds 显示 B>A 稳定，再按预算顺序推进：① 更长训练（验证 2k 是否过少）；
  ② GWHD 检测（验证域外农业迁移）。B2−B1 的采样方向差异因类间不一致，不足以单独
  决定优先采用哪种采样画像。

---

## 附：研究问题直接回答

1. **B1/B2 是否优于官方 I-JEPA 冻结特征？** 在本设置下呈初步正迁移（B1 +1.17、
   B2 +0.96 pp mIoU，3/3 seeds 同向），但受单 checkpoint 限制，称为初步迁移信号，
   非已证明的“优越”。
2. **B2 相对 B1 方向如何？** mIoU −0.21 pp（轻微负）、weed 3/3 为负、soil/crop
   持平——探索性负向且不一致；不能判定优劣。
3. **是否值得进入 GWHD 检测 / 独立预训练 seed / 更长训练？** 先独立预训练 seeds
   复现 B>A；之后再评更长训练与 GWHD 检测。
