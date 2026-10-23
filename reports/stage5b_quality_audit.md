# Stage 5B — 质量与采集域图像审计

> 产物：`data/registry/field_image_review.csv`（10 个本地来源 × 40 张分层抽样）、
> `reports/stage5b_domain_audit.json`
> 生成工具：`tools/stage5b_audit_domains.py`
> 视觉抽样：Agnes 3.0 Flash（图像由该 API 识别，脚本在 Git 之外执行）

## 1. 客观质量指标

### 1.1 新来源

| source_id | 图像 | 短边最小值 | 短边中位数 | <224 px | 灰度 | 解码失败 |
|---|---:|---:|---:|---:|---:|---:|
| `sesame_weed` | 1,300 | 512 | 512 | 0 | 0 | 0 |
| `applebbch81` | 1,838 | 640 | 640 | 0 | 0 | 0 |
| `weedcrop` | 2,822 | 360 | 720 | 0 | 0 | 0 |

所有新图像都在缩放前满足 224 px 下限，**没有通过放大低分辨率图像来过门**。

### 1.2 模糊分布（Laplacian 方差，抽样 40 张/来源）

| source_id | p10 | p50 | p90 |
|---|---:|---:|---:|
| `sesame_weed` | 292.9 | 1,001.5 | 3,975.1 |
| `applebbch81` | 323.2 | 682.7 | 1,114.6 |
| `weedcrop` | 56.1 | 104.5 | 1,246.4 |
| `plantvillage`（对照） | 190.9 | 3,477.8 | 8,476.3 |

`weedcrop` 的整体清晰度明显低于其他来源（p50 仅 104.5），部分原因是增强中的旋转/剪切
与重采样；这也支持了它不进入 Field-D1 的决定。田间来源中，苹果果园的 p10 为 323，
说明浅景深果实特写在数值上会偏低，但这不等于不可用——全局阈值会误删这类合法样本，
因此本轮只报告分布，不设全局模糊门禁。

### 1.3 曝光与颜色模式

抽样 400 张全部通过 `mean_brightness` 与模式检查；无灰度、无全黑/全白帧。

## 2. 图像级域分诊结果

`field_image_review.csv` 中的标签来自启发式统计（边框一致性、ExG 植被比例、路径关键词），
**只用于排序复核队列**，一律标记为 `model_assisted_inspection` + `unknown` 置信度。

| source_id | 分诊 FIELD | 分诊 UNKNOWN | 分诊 CONTROLLED | 分诊 PROTECTED |
|---|---:|---:|---:|---:|
| `lucas_vision` | 39 | 1 | 0 | 0 |
| `acre_crop_weed` | 40 | 0 | 0 | 0 |
| `radish_wheat` | 37 | 3 | 0 | 0 |
| `sorghum_weed` | 18 | 22 | 0 | 0 |
| `vegann` | 32 | 8 | 0 | 0 |
| `inat_weeds` | 38 | 2 | 0 | 0 |
| `sesame_weed` | 36 | 4 | 0 | 0 |
| `applebbch81` | 36 | 4 | 0 | 0 |
| `weedcrop` | 19 | 21 | 0 | 0 |
| `plantvillage` | 34 | 1 | 1 | 4 |

**这张表本身证明了启发式不可作为最终标签**：PlantVillage 是公认的受控拍摄来源，
但基于边框颜色/亮度的规则把其中 34/40 判成了 FIELD——因为它的背景多为中灰，
既不是纯白也不是纯黑。因此本项目采用"来源级文档 + 抽样验证"作为准入依据，
启发式只负责排队。第 4 节的视觉抽样是另一条独立证据。

## 3. 来源级判定

63 行全部有采集域标签。判定规则与证据列在
`data/registry/field_source_review.csv` 与 `reports/stage5b_source_verification.md`。

## 4. 视觉模型抽样（模型辅助证据）

对每个新来源抽取 18 张（按文件大小分层，拼成 2 张 3×3 联系表），由 Agnes 3.0 Flash
逐格描述采集场景。原始回答保存在 Git 之外（`/tmp/vision_*.json`），结论摘要如下：

| source_id | 判为真实田间/果园 | 判为受控 | 无法判断 | 结论 |
|---|---:|---:|---:|---|
| `sesame_weed` | 18/18 | 0 | 0 | 与官方描述一致：干燥土壤、自然光照、无纯色背景 |
| `applebbch81` | 18/18 | 0 | 0 | 果园树行、天空、支柱与滴灌设施清晰可见 |
| `weedcrop` | 14/18 | 4/18 | 0 | 多数为田间，少数近景被读作受控；不足以支撑统一标签 |
| `plantvillage`（方法对照） | 0/9 | 8/9 | 1/9 | 方法能识别实验室痕迹，说明上面三行不是"什么都判成田间" |

视觉模型的结论被记录为 `model_assisted_inspection`，复核状态统一为
`model_assisted_pending_human`，**没有**写成"专家人工确认"。它足以支持 CANDIDATE
版本，但不足以支持 FROZEN 版本。

## 5. 待人工处理的队列

1. `sesame_weed`、`applebbch81` 各 1,300 / 1,838 张的域标签需要一名人工复核者签字，
   才能把 `model_assisted_pending_human` 升级为 `reviewed`。
2. `weedcrop` 2,822 张需要判定：可作为 Controlled/混合研究候选，或整源弃用。
3. PlantVillage 的 12,436 条感知相似候选与 v1 的 784 条候选仍待复核，但它们都不在
   Field-D1 中，不阻塞本次候选版本。
4. 采样中 `sorghum_weed` 有 22/40 被判为 UNKNOWN，原因是俯拍近景土壤占比高；
   它是 v1 继承来源，本轮不重新标注，只记录该信号供后续复核。
