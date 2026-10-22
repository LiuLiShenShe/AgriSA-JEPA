# Stage 4B 总结报告

> 生成时间：2026-10-22 · 分支 `agri-stage4b` · 数据 v1（PROVISIONAL）
> 全量几何审计：12567 张 × 9 个 arm；下游：PhenoBench v1.1.0，3 个 Probe seed。

## 0. 最终结论

**D — Negative**

* G0 − B（几何控制本身）mIoU 0.377 pp
* G1 − G0（植被结构额外收益）mIoU -0.607 pp，0/3 seeds
* G2 − G0（边界结构额外收益）mIoU -0.363 pp，1/3 seeds
* G2 − G3（正确空间对齐）mIoU -0.523 pp
* G1 − C1_500 -0.383 pp；G2 − C2_500 0.207 pp

判定规则在实验前写入 `tools/make_stage4b_reports.py`：只有结构臂相对 G0
达到 +0.5 pp 且至少 2/3 seeds 同向，才记为结构信号；只有几何控制本身
达到 +0.5 pp 才记为几何收益。3 个 Probe seed 的最小双侧符号检验 p=0.25，
因此这些标签是探索性筛查，不是显著性结论。

## A. Stage 4A 失效原因

1. **C1/C2 的 Target Union 是否系统性低于 B？** 是。全量审计：B 114.92，C1 96.30（Δ -18.62），C2 96.91（Δ -18.02），C3 100.51（Δ -14.41）。配对胜/负：C1 2855/9574，C2 2940/9465。
2. **Context 可见数量差异多大？** 在真实 batch 裁剪后，B 80.59，C1 95.17（+14.57），C2 94.58（+13.99）。Stage 4A 报告的 109/132/123/120 是**截断前**数字；见 `stage4b_geometry_diagnosis.md` 第 4 节。
3. **扩大样本后是否仍存在？** 是。12,567 张全量、配对、来源分层后，方向与 60 张时一致，符号检验 p < 1e-8。
4. **重复预测有多严重？** 重复目标面积 B 62.12，C1 80.74，C2 80.13。
5. **结构覆盖提升是否集中在重复区域？** 是。C1 的植被重复质量比 0.975，B 为 0.582；C2 的边界重复质量比 0.927，B 为 0.578。
6. **支持哪些机制假设？** 支持“结构引导使目标块集中、Target Union 下降、Context 上升”，这是可复现的几何混杂；不支持“结构信息本身无用”的更强结论。下游因果仍不能仅凭几何相关性断言。

## B. 新采样器

7. **几何控制方式**：候选块权重 `p_i ∝ exp(λ ŝ_i − β o_i)`，`o_i` 为候选块与该图已选 Target 并集的重叠比例；β=3.5 由离线几何扫描选定。
8. **G0/G1/G2/G3 几何是否可比？** G1/G2：Union 114.90/115.80 vs B 114.92，Context 78.84/78.95 vs B 80.59，均值与分布均通过门禁。G3：Union 119.46、Context 73.84，均值接近但分布 KS 仍失配（control note）。G0：Union 130.62、Context 68.72，过分散，作为机制对照而非几何匹配臂。
9. **几何匹配让结构引导强度减少了多少？** 与 Stage 4A 的 C1/C2 相比，G1 的植被 Union 均值 0.5514（C1 0.5668，B 0.4841）；G2 的边界 Union 均值 0.2970（C2 0.3006，B 0.2807）。结构偏好保留，重复质量比明显下降。
10. **原始 JEPA 目标函数**：保留。未改 Loss、Encoder/Predictor、EMA、块尺寸/数量、Context 排斥、`min_keep`、官方初始化。
11. **G3 是否比原 C3 更适合检验空间对齐？** 是。G3 保留直方图 100%、lag-1 自相关保留 0.981，与原图平均相关 -0.090；C3 的置换会同时破坏局部连续性。残余相关 >0.5 的图像比例 0.0060。
12. **诊断分母问题**：已修复。`mean_candidate_prior_score` 现在除以有效候选数，不是 `candidates_generated`；有手工构造分数的单元测试，且证明 Mask/RNG/梯度不变。

## C. 训练

13. **哪些 Arm 完成 500-step 试点？** G0/G1/G2/G3（见 `stage4b_training_pilot.md`）。
14. **初始化与总调度是否一致？** 是。四臂均从官方 checkpoint 独立初始化，`max_optimizer_steps=2000`、`interrupt_after_optimizer_steps=500`，`save_at_optimizer_steps=[500]`；`tools/check_stage4b_protocol.py` 逐字段验证。
15. **是否可恢复？** checkpoint 保存了 encoder/predictor/target/optimizer/scaler/RNG/loader generator/mask_collator_step；resume 路径与 Stage 2C 相同，并额外记录了 Mask 计数器。
16. **成本**：单卡 A6000；每臂 500 updates，~27–30 min 训练段，峰值 ~22.2 GB；Mask 额外开销见 `stage4b_mask_benchmark.json`。
17. **哪些实验未执行？** U 未单独训练（离线分布等价，降为诊断）；未跑 2,000-step 四臂全长；未用下游指标调 β。

## D. 下游

B_500: mIoU 72.64，Crop 82.40，Weed 37.79。
G0_500: mIoU 73.02，Crop 82.57，Weed 38.76。
G1_500: mIoU 72.41，Crop 82.25，Weed 37.29。
G2_500: mIoU 72.66，Crop 82.39，Weed 37.86。
G3_500: mIoU 73.18，Crop 82.39，Weed 39.42。

18–19. **mIoU / Crop / Weed** 见上表。Soil 在官方评价器的 semantics 任务中不单独输出；自实现指标中的 Soil 约 97.7，各臂差异 <0.02 pp。

20. **G0 − B 是否有收益？** mIoU 0.377 pp（3/3 seeds：[0.05, 0.47, 0.61]），Weed 0.970 pp，Crop 0.163 pp。方向一致但幅度低于预先设定的 +0.5 pp 探索性参考线，只能算弱提示。
21. **G1 − G0 是否有收益？** 否，mIoU -0.607 pp，0/3 seeds；Weed -1.463 pp。控制几何后，植被结构没有带来额外收益。
22. **G2 − G0 是否有收益？** 否，mIoU -0.363 pp，1/3 seeds；Weed -0.893 pp。边界结构同样没有额外收益。
23. **G2 − G3 是否支持空间对齐的价值？** 不支持。mIoU -0.523 pp，1/3 seeds；G3（空间错位先验）反而略高。结合 C2≈C3 与 G2≈G3，没有证据表明“正确的植物边界位置”本身带来了可测收益。
23b. **控制臂 G3 意外最高（mIoU 73.18）怎么解释？** G3 − B = 0.537 pp（3/3 seeds），是新增臂中最大的单点差异；但 G3 是**空间错位**的先验，且 G2 − G3 = -0.523 pp。因此不能把它解释为结构或对齐收益。它最多提示：G3 这一臂的几何（Union +4.5、Context −6.8）值得用纯几何对照单独复核，而不是回到结构先验。
24. **是否以牺牲 Crop 换 Weed？** 没有出现这种交换。G1 crop -0.320 / weed -1.463，二者同降；G2 crop -0.173 / weed -0.893，同样同降。

25. **三个 Probe seeds 方向是否一致？** G1−G0、G2−G0、G2−G3 的 mIoU 方向分别为 0/3、1/3、1/3 一致；G0−B 为 3/3 一致。逐 seed 表见 `stage4b_phenobench_comparison.md`。

## E. 下一步

停止当前 Mask 路线，把预算转向农业持续预训练本身与跨数据集迁移；除非出现新的、与几何无关的结构先验假设，否则不再增加 Mask 消融。

> 任何 Positive/Mixed 结果都必须用独立预训练 seed 与第二个数据集复核；不能把 3 个 Probe seed 当作 3 次独立预训练。
