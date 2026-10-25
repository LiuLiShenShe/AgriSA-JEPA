# Stage 5E 许可审计（Work Package A）

访问日期：2026-10-25；证据存档：`reports/stage5e_evidence/`。

## 结论摘要

- 四个待核验来源在 Stage 5D accepted 清单中共 **15892** 张。
- 其中 **15892** 张的本地研究训练用途有官方证据支持，可以进入训练清单。
- 因许可被排除的训练图像：**0** 张。
- VERIFIED_RESEARCH_USE：**15808** 张；RESTRICTED_RESEARCH_USE：**84** 张。

## 逐来源判定

| 来源 | 真实名称 | 许可 | 研究训练 | 再分发 | checkpoint | accepted |
|---|---|---|---|---|---|---:|
| `agrifm_ref1` | Wheat Spike Dataset (voglinio) | CC0-1.0 | VERIFIED_RESEARCH_USE | ALLOWED | ALLOWED | 40 |
| `agrifm_ref11` | Maize phenology images (NDVI camera) | Apache-2.0 | VERIFIED_RESEARCH_USE | ALLOWED_WITH_CONDITIONS | ALLOWED_WITH_CONDITIONS | 382 |
| `agrifm_ref58` | Makerere University Beans Image Dataset | CC0-1.0 | VERIFIED_RESEARCH_USE | ALLOWED | ALLOWED | 15,386 |
| `agrifm_ref9` | PinotNoirGrapes (nicolaasregnier) | CC-BY-NC-4.0 | RESTRICTED_RESEARCH_USE | ALLOWED_WITH_CONDITIONS | UNRESOLVED | 84 |

## 五个权利维度是分开的

`train_eligible` 只由研究用途维度决定；`redistribution_eligible` 只由再分发维度决定。两者在代码中是两个独立属性（`src/registry/stage5e_licence.py`），并有单测保证它们不会互相覆盖。

## 证据与限制

- 证据均为发布方官方端点：Kaggle dataset API 的 `licenseName` 字段，以及 Harvard Dataverse 数据集 API 的 `license` / `versionState` / `termsOfUse` 字段。响应体原样存档，SHA-256 记录在 `reports/stage5e_evidence/evidence_index.json`。
- **限制**：Kaggle 的许可字段是上传者的声明，不是对版权链条的独立法律核验；本报告只声称"发布方官方端点如此声明"，不声称"已在法律上确认版权归属"。
- CC BY-NC 4.0（`agrifm_ref9`）只允许非商业使用。本项目为学术、非商业研究，故本地训练可用；但**发布 checkpoint** 是否构成非商业使用无法从条款文字判定，标为 `UNRESOLVED` 而不是"允许"。
- 未对四个来源做任何降级或"凑数"处理：它们的许可状态是从 official endpoint 读出来的，不是为了让总量达到某个数字而保留或排除。
