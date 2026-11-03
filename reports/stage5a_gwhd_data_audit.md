# Stage 5A — GWHD 2021 数据审计

> 官方来源：https://zenodo.org/records/5092309（CC-BY-4.0，10,215,190,132 bytes）。本报告由 `tools/stage5a_audit_gwhd.py` 生成。

## 1. 归档校验

- 大小：10,215,190,132 / 官方 10,215,190,132 → OK
- MD5：`22b4b542c9ae7e056d7fcdeae9ecaed5` / 官方 `22b4b542c9ae7e056d7fcdeae9ecaed5` → OK
- SHA-256（本项目记录）：`87649215c9dd413af26d295cbf54f6f245da8d49747a2f13058133ff64b01f68`
- 解压后图片：6,515；解码失败：0

## 2. 官方划分（未重划分）

| split | 图像 | 框 | 空标注 | 退化框 | 完全越界 | 部分越界 | 重复框 |
|---|---:|---:|---:|---:|---:|---:|---:|
| test | 1,381 | 67,407 | 47 | 0 | 7 | 12 | 1 |
| train | 3,655 | 163,603 | 50 | 0 | 0 | 0 | 5 |
| val | 1,476 | 44,347 | 28 | 0 | 0 | 0 | 0 |

- 跨 split 图像名重合：{'train_val': {'shared_image_names': 0, 'shared_groups': [], 'shared_group_count': 0}, 'train_test': {'shared_image_names': 0, 'shared_groups': [], 'shared_group_count': 0}, 'val_test': {'shared_image_names': 0, 'shared_groups': [], 'shared_group_count': 0}}
- 标签缺失图像：0；未标注图像：3

## 3. 分辨率与重复

- 分辨率分布：{'1024x1024': 6515}
- 字节级重复对：1；解码像素级重复对：1
- 感知相似只作为候选，不自动删除。

## 4. 与 D1 候选语料的重叠

- 对照行数：66,851；精确匹配：0；pHash 候选（≤6）：0

GWHD 是下游评价保留集，**任何** GWHD 图像（无论划分）都不得进入预训练语料。

## 5. 结论

- checksum_verified = **True**
- official_splits_preserved = **True**
- ready_for_training = **True**

