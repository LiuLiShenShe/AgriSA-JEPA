# Stage 5B — 下载审计

> 产物：`reports/stage5b_source_verification.json`、`reports/stage5b_ingest_*.json`
> 原始文件位置：`/data/datasets/Stage5B/`（Git 之外）
> 执行环境日期：2026-10-23

## 1. 网络可达性（实测）

| 主机 | 结果 | 影响 |
|---|---|---|
| `huggingface.co` | 不可达（IPv6 解析，连接失败） | 无法用官方 HF 入口；AgriField-40K 已在本机，无需下载 |
| `zenodo.org` | DNS 返回 `0.0.0.0`，不可达 | VCD、WE3DS 无法下载 |
| `www.sciencedirect.com` | HTTP 403 | VCD 论文页不可用 |
| `data.mendeley.com` | 页面 HTTP 200，但文件桶 `prod-dcd-datasets-cache-zipfiles.s3...` 返回 **403 AccessDenied** | Crop Pest、Winter Wheat Leaf、Tomato Plant Factory 无法下载 |
| `www.ipb.uni-bonn.de` | 超时 | MuST-C、GrowliFlower 页面不可用 |
| `www.kaggle.com` | HTTP 200；公开数据集下载接口返回 302 + 签名 URL，**无需账号**即可下载 | 本轮三个新来源全部由 Kaggle 取得 |
| `weed-ai.sydney.edu.au` | 页面可达但 Cloudflare 挑战 | PalmerAmaranth、RumexLeaves 无法自动下载 |
| `vision.eng.au.dk` | 页面 HTTP 200，但 `?download=` 处理器返回 HTML 页面而不是文件 | GrassClover 无法下载 |

不下结论说"这些数据集不存在"，只记录：**本机网络无法取得其文件**。

## 2. 实际下载的三个来源

| source_id | 入口 | 归档字节 | 归档 SHA-256 | 解压图像 | 解码成功 |
|---|---|---:|---|---:|---:|
| `sesame_weed` | `https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes` | 83,008,512 | `01de68257cddbd22d8579c6b7f912dc2bd1a592fec676e14c070c140895741c4` | 1,300 | 1,300 |
| `applebbch81` | `https://www.kaggle.com/datasets/projectlzp201910094/applebbch81` | 168,020,771 | `d5f47554e00e19c7e73c766ee8d986d8cd2be5036c7a39a97b1981059ace519a` | 1,838 | 1,838 |
| `weedcrop` | `https://www.kaggle.com/datasets/vinayakshanawad/weedcrop-image-dataset` | 263,674,982 | `db7a8ccb22a9385ef1603e1e624b08c512efc4eb2bb14d34c038966f46bbdfbb` | 2,822 | 2,822 |

三个归档均通过 `unzip -tq` 完整性检查。原始归档与解压目录均保留：
`/data/datasets/Stage5B/<source>/`。图像不进入 Git。

### 2.1 Sesame-Weed

- 解压后为 1,300 个 `.jpeg` + 1,300 个 YOLO `.txt` + `classes.txt`，与数据集页声称的
  1,300 张一致。
- 图像尺寸全部 512×512，RGB，无灰度、无解码失败。
- **与 AgriField-40K 的 `saw_*` 行逐字节相同**：1,300/1,300 命中 SHA-256 或解码像素哈希。
  这是一条正面证据（确认 `saw` 前缀映射），不是重复计数，因为 v1 未采纳这些行。
  详见 `reports/stage5b_source_verification.md` 第 7 节。

### 2.2 AppleBBCH81

- 1,838 张 640×640 RGB 裁剪图，来自 3,008×2,000 原图的 30% 重叠滑窗裁剪；
  文件名形如 `DSC_<树号>_<会话>_<视图>.jpg`，共 **140 棵树的 150 个采集会话**。
- 与 AgriField-40K、v1、PhenoBench、GWHD 无任何哈希或感知命中。
- 该来源在 Stage 5A 台账中不存在，`agricultural_sources_v2.csv` 新增了一行。

### 2.3 WeedCrop（未纳入 Field-D1）

- 2,822 张图像 = **1,544 张原始图的 2,822 个增强版本**（639 张有 3 个版本、905 张单版本）。
  增强为 90° 旋转、±15° 剪切、±25% 亮度。
- 官方说明它派生自 Sudars et al. 2020（即已登记的 `ronin`），但**没有说明原始采集环境**。
- 判定为 `UNKNOWN`，因此不进入 Field-D1；数据保留。

## 3. 尝试但受阻的来源

| 来源 | 入口 | 失败证据 | 状态 |
|---|---|---|---|
| `agrifm_ref57` Crop Pest and Disease Detection | Mendeley `10.17632/bwh3zbpkpv.1` | 元数据 API 返回 S3 地址，实际 GET 返回 `403 AccessDenied`（XML Error） | BLOCKED（网络/权限） |
| `vcd` | DOI `10.1016/j.dib.2022.108035` → Zenodo/ScienceDirect | ScienceDirect 403；Zenodo DNS 不可达 | BLOCKED（网络） |
| `we3ds` | Zenodo `10.5281/zenodo.7457982` | Zenodo 不可达 | BLOCKED（网络） |
| `grass_clover` | `vision.eng.au.dk` | `?download=` 返回 163 KB HTML 页面，不是 zip | BLOCKED（下载处理器失效） |
| `palmer_amaranth`、`rumex_leaves` | weed-ai | Cloudflare 挑战页 | BLOCKED（反爬） |
| `agrifm_ref20` Winter Wheat Leaf | Mendeley `10.17632/gj72pjfvb2.1` | S3 `403 AccessDenied` | BLOCKED（网络/权限） |
| `agrifm_ref90` Tomato Plant Factory | Mendeley `10.17632/8h3s6jkyff.1` | S3 `403 AccessDenied` | BLOCKED（网络/权限） |
| `perennial_plants` | Kaggle | 可下载，但 Kaggle 许可字段为 "Other"，与台账记录的 MIT 不一致；且官方描述未说明采集环境 | 未下载（许可与域均未确认） |

## 4. 数量口径

| 口径 | 数值 |
|---|---:|
| 本轮实际下载的图像文件 | 5,960 |
| 解码成功 | 5,960 |
| 解码失败 | 0 |
| 通过采集域政策进入 Field-D1 | 3,138（1,300 + 1,838） |
| 因 `UNKNOWN` 被排除 | 2,822（weedcrop） |
| 通过与下游保留集的精确哈希筛查 | 3,138 / 3,138 |

"论文报告数量"与"本机实际数量"继续分列在不同字段，未合并。
