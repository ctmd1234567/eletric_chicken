# 第三方模型与素材说明

本仓库随 Git 分发两个第三方训练权重；其余候选权重（D-Fire YOLOv8n、Safety Detection、LwST-FSD）仅保存在本地，需自行下载。所有保留文件均**未修改任何权重字节**。图片、视频与数据集不随仓库分发。

2026-10-30 已移除 `pyronear_sensitive`、`firesmockdetect_y11s`、`fasdd_yolo26n` 的登记和本地权重。下文保留被移除模型的来源与许可记录，供历史报告追溯，不表示当前仍分发或可通过下载脚本选择。

## 模型来源

| 权重 | 上游仓库 | revision | 许可 |
|---|---|---|---|
| `models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt` | [HF: TianchenGuan/FireSmockDetect-models](https://huggingface.co/TianchenGuan/FireSmockDetect-models) | `a914a2a036686cf09bac8efc738971fdc8f31a29` | AGPL-3.0 |
| `models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt` | [HF: mfranzon/fire-smoke-yolov8](https://huggingface.co/mfranzon/fire-smoke-yolov8) | `f1c6426b069c1849cbf13b1ef5d2a260289286db` | AGPL-3.0 |

2026-10-29 核对上表权重的模型卡，均标注 `agpl-3.0`。本地文件的 SHA-256 与 Hugging Face 返回的 LFS oid 逐字节一致，见 `sentinel/models/catalog.py` 的 `MODELS`。已移除的 `expB_y11s_deployed.pt` 来自相同 FireSmockDetect 仓库及 revision，许可同为 AGPL-3.0；历史 SHA-256 为 `cc56224c84092e1e374cee1b3ce317f17f7f80e7945bc6c9a3a24a083321ff0e`。

FireSmockDetect 的代码与实验报告在 [GitHub](https://github.com/TianchenGuan/FireSmockDetect)，其中代码为 AGPL-3.0、报告与文档为 CC BY 4.0。Early Fire and Smoke 只发布了权重，模型卡描述的连续帧状态机没有公开实现。

## 新增候选与 D-Fire 地址

2026-10-29 新增候选来源（历史登记；Pyronear 与 FASDD 已于 2026-10-30 移除）：

- `models/fire_smoke/pyronear_sensitive/best.pt`：现为 Pyronear 官方 [yolov11s / Tenacious Tiger v8.3.0](https://huggingface.co/pyronear/yolov11s)，固定 revision `de398b5192b2c1d5f8963aef56bf8df1f13c159e`，19,225,626 字节，SHA-256 `c4b5b846b957cef2465c4f02f657442ec05151b1c3f95cb2e5702c2aa25bc848`。固定版本模型卡及 LICENSE 标注 AGPL-3.0。仍为 YOLO11s、1024 输入、早期野火烟雾检测；原始类别 `item` 仅在内存中显示为 `smoke`。保留旧目录名/选项名用于兼容，旧 Sensitive Detector v1.1.0 已被替换。
- `models/fire_smoke/dfire_yolov8n/best.pt`：[rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n)，固定 revision `13017fe8af477c25f5298d168e2dfede4b000753`，6,229,802 字节，SHA-256 `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f`，模型卡标注 AGPL-3.0。由第三方在 D-Fire 上训练，类别为 `0=smoke, 1=fire`；不是 D-Fire 数据集作者的 YOLOv5 权重。

- `models/fire_smoke/fasdd_yolo26n/best.pt`：[seawsurf/fire_yolo](https://huggingface.co/seawsurf/fire_yolo)（旧地址 `fire_smoke_detection_box` 已重定向），固定 revision `9403f231bab33e0ea4048424ff64023936957ba5`，上游文件 `weights/best.pt`，5,377,541 字节，SHA-256 `d2b0be214f45fa01d2044b960c3cb4da4bf65c06a89cf0a1967b0d40e8db4b98`。FASDD_CV 训练的 YOLO26n，640 输入，权重类别为 `0=fire, 1=smoke`。上游模型卡标注 CC BY 4.0；该声明与 Ultralytics 基础模型许可的关系尚未核清，仅本地保存。
- `models/fire_smoke/safety_yolo26/yolo_smoke_fire.pt`：[e1250/safety_detection](https://huggingface.co/e1250/safety_detection)，固定 revision `f2049aa749dcc844e406d2507ffa3b683ce83aac`，15,735,298 字节，SHA-256 `cd2c8c1108b29589d77d4a21cc6e5f111c7586b225e2d8668fda50d791a2cbea`。模型卡称使用 Home Fire Dataset；checkpoint 与 args.yaml 均为 YOLO26n、640 输入。**上游未声明权重许可**，仅作为本地展示/待核实候选，不作为选型决策依据或随库分发。checkpoint 的类别为 `0=smoke, 1=fire`，与模型卡相反；代码保留 checkpoint 映射，实际标签语义仍需核实。

四者均从固定 revision 的 Hugging Face 镜像下载，大小及 SHA-256 与仓库元数据一致。车棚薄烟召回率和误报率均待测。旧 Pyronear v1.1.0 的固定来源为 `pyronear/yolo11s_sensitive-detector@3df695158beed0a34e2f5a48245773fd9855ddd6`，旧哈希 `a9bfa11c559e4b221c43bb24a0ec0857bc75ec653283ba6146aef44c088da51a`；历史报告中的 16/76 属于旧权重，不能记在新版名下。

D-Fire 的准确入口：

- [原始数据集仓库](https://github.com/gaia-solutions-on-demand/DFireDataset)（本次不下载数据集）。
- 数据集作者关联的 [Fire-Detection 模型仓库](https://github.com/pedbrgs/Fire-Detection)，README 提供 [YOLOv5s 权重](https://1drv.ms/u/c/c0bd25b6b048b01d/EeZYmpKPBppNr3lo8oaOqecB9GDj1dDvbogCJyegO0PY1Q?e=u76Iei) 和 [YOLOv5l 权重](https://1drv.ms/u/c/c0bd25b6b048b01d/ERy9-UpeDeRHkEb_eqPeC7EBdblsWLwujJ1BlssUWfz_Lg?e=e71j1u)。本机访问作者网盘连接失败，旧 Google Drive 脚本地址返回 404；不能将本次下载的第三方 YOLOv8n 冒充作者原始权重。

## 为什么本仓库也是 AGPL-3.0

本仓库的脚本、文档及随库分发的两个权重采用 AGPL-3.0（全文见 `LICENSE`）。本地 D-Fire 候选同样由上游标注 AGPL-3.0。Safety YOLO26 和 LwST-FSD 的许可问题单独列于本文，不应把仓库许可证视为替上游授权。

如果你把本仓库的推理能力以网络服务形式提供给他人，AGPL-3.0 第 13 条要求你向使用者提供完整源码。

## 数据集与素材

- LwST-FSD：作者 [WCUSTC/LwST-FSD](https://github.com/WCUSTC/LwST-FSD)，固定源码 revision `af3ed8ac9686a607ec2330275fbb95cf8d1f910d`。作者公开 [FIgLib 五帧模型分享](https://pan.ustc.edu.cn/share/index/5b015f2c73e24fd48b82) 的 `epoch_10.pth` 为 141,991,741 字节；本次从作者地址下载的 SHA-256 为 `6ac5ca325d59f1bb48a8f63750ba1efd49b53cc5b34659395e8edf14ad8b3add`。**上游未提供独立摘要/签名，不能称为官方 LFS 哈希验证**；本项目固定该摘要以检测后续漂移。仓库未声明明确项目许可，权重许可未核清，仅限本地演示/复现，不随 Git 分发，不作为车棚选型结论依据。权重位于 `models/fire_smoke/lwst_fsd/`，作者源码按固定版本和 SHA-256 下载至被忽略的 `samples/lwst_fsd/upstream/`，不复制到本仓库分发源码中。推理适配器保留原始 TSM/PTA/CSP/YOLOX 前向层，替代训练框架中的标准 Conv/BN/SiLU 包装及训练日志反序列化，不训练、不重新导出或改写权重。FIgLib 数据集未下载。

- 训练数据（pyro-sdis、D-Fire、Domestic Fire and Smoke）**不随本仓库分发**，需从原始出处获取，许可归各自作者。
- FIgLib / HPWREN 为 CC BY-NC-ND，上游只用于回放评测，从未参与训练，本仓库也不分发。
- 本地 `samples/rf100_smoke/` 为 [RF100 smoke](https://universe.roboflow.com/roboflow-100/smoke-uvylj/dataset/2) 测试子集（CC BY 4.0），仅用于本机运行检查，已被 `.gitignore` 排除。
- 本地 `video/` 为第三方新闻视频，版权不在本项目，已被 `.gitignore` 排除，不得公开分发。

## 免责

这些模型是研究工具，不是经过认证的火灾报警装置——上游作者也明确这样声明。本仓库未在电动车车棚场景做过任何性能验证，不要把其中的模型直接当作生命安全保障设备部署。
