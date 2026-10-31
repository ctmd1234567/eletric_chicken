# 第三方模型与素材说明

本轮本地候选 `dfire_ludan_focused`：从原始 `dfire_ludan` 继续训练，30 轮中按参考验证集选择第 22 轮。权重 `runs/dfire_ebike_focused/experiment_002/train/weights/best.pt`，6,259,818 字节，SHA-256 `fb335751a28ebcc7f3d35651d4eb8ef4a9805b2e62db32b14b690a4010f7f919`。继承上游 AGPL-3.0 要求；训练素材按各来源许可及用户已确认授权使用，不随 Git 分发。合成素材的 CC BY-NC 4.0 等约束仍适用，不据此推定商业使用许可。训练配方、用户确认和实际退化见 `docs/ebike_focused_local.md`。

本仓库随 Git 分发两个第三方训练权重，以及三份本地微调最佳权重（`dfire_small_smoke`、`dfire_ludan`、`dfire_ludan_focused`）；其余候选权重（D-Fire YOLOv8n、Safety Detection、LwST-FSD）需自行下载。所有保留文件均**未修改任何权重字节**。图片、视频与数据集不随仓库分发。

2026-10-30 已移除 `pyronear_sensitive`、`firesmockdetect_y11s`、`fasdd_yolo26n` 的登记和本地权重。下文保留被移除模型的来源与许可记录，供历史报告追溯，不表示当前仍分发或可通过下载脚本选择。

## 模型来源

2026-10-31 后续状态：用户明确确认新收集新闻素材“已有训练使用授权”，本项目据此记录本地/私有训练使用条件，不推定公开再分发许可。用户提供 `ebike_smoke_100_annotated.zip`（SHA-256 `f9b3d4d0fc631e21a10e84f633fa1af2ac3c8dd933acfea8b9fe626977daca28`），86 张有框，另 14 张由用户明确判定为无肉眼可辨认烟雾目标，按其要求补空标签；不把空标签解释为真实无火灾风险。用户另明确要求将 9 个越界框裁到图像边界、保留图内部分。源 ZIP 保留不改，衍生标签及裁剪记录在本地 `annotations/import_004/`。下文“授权待核实、未纳入训练”的描述是本次确认前状态。当前定向微调清单为 `datasets/ebike_focused_20261031/`，从 `dfire_ludan` 初始化；原图、标注和数据包不随 Git 分发；后续按用户明确要求分发上述三份最佳权重，其他实验产物不分发。

本轮额外整理的 `datasets/ebike_early_smoke_review_20261031/` 为第三方新闻/消防事故素材的本地待标注副本。原始页面包括《20起电动自行车爆炸起火瞬间 一篇看完！》（澎湃政务 `28427063`）及天津南开公交站附近电动车起烟报道（澎湃政务 `27945147`）。100 帧来自 13 段素材，多数选择早期局部烟雾，17 帧为已形成局部烟团的补充；不等于 100 起事故，也不能证明全部事件独立。未获得明确训练/再分发许可，全部仅供本地审阅与手工标注，不纳入当前训练、验证、测试清单或云端包，不随 Git 分发；新闻水印和来源记录保留，用户标注不能代替原素材许可。

2026-10-31 按用户要求移除上一轮退化的 `dfire_expanded` 权重和模型登记，保留小型来源及实验记录供追溯；下文该权重的接入/候选描述为历史状态。新训练默认起点为 `dfire_ludan`，新模型尚未训练。

2026-10-31 本轮场景补充训练从现有 `dfire_ludan` 初始化，保留用户手工标注的 40 张训练图片并提高采样频次；用户确认标签正确，标签文件不由模型重新生成。用户视频及衍生图片仅用于本地/私有云端训练，不公开分发。

本轮新来源：**Indoor Fire Smoke**，Arya Krisna Putra，Zenodo record `15826133`（DOI `10.5281/zenodo.15826133`），发布页和包内 Roboflow 元数据均声明 CC BY 4.0。原包 `Indoor Fire Smoke.zip` 为 200,547,120 字节，发布者 MD5 `086fbc3b874139276097f4057bc45a3c`。本次只通过 HTTP Range 提取原训练划分的候选图片和标签，逐条验证 ZIP CRC32 并保存本地 SHA-256，**不宣称已验证整个 ZIP 的 MD5**。包内类别名仅为 `0/1`，以带框抽查确认原 `0=fire,1=smoke` 后，衍生标签转换为本项目 `0=smoke,1=fire`；原文件保留。仅纳入视觉核对通过的图片，包含建筑/仓库火烟及停放摩托车附近的受控燃烧烟雾，不宣称后者为电池热失控实拍。镜头/事件身份不完整，保守地全部只补入训练集，不作为独立测试。

本轮新来源：**Simuletic CCTV Smoke & Fire Emergency Detection Dataset** 免费样本，作者 Hugging Face 仓库 `Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset` 固定 revision `b9df600c82aa5033a31d0f8448ba4863698b978c`。该 revision 的 README 和结构化许可统一为 **CC BY-NC 4.0**，仅用于本轮非商业研究；不采用旧缓存页面里的 CC BY 4.0 声明。数据为合成的城市停车区、自行车架、建筑外部及垃圾桶烟火，共 240 张原图及 YOLO 标签；图片按官方 LFS SHA-256 校验，普通 Git 标签按 blob ID 校验。源 `data.yaml` 的 `nc: 1` 与两类 names 不一致，衍生配置明确使用两类并转换 `0=fire,1=smoke` 为本项目顺序，源文件不改。合成图片仅作低占比训练补充，不代表真实电瓶车事故或预警能力。图片、标签、来源快照和私有上传包不随 Git 分发。

2026-10-31 后续回放检查后，两个编辑器入口已恢复默认 `dfire_ludan`；`dfire_expanded` 仅作本地对比候选，以下接入记录的默认选择已被本条取代。权重、来源及许可记录不变。

2026-10-31 接入云端微调产物 `dfire_expanded`：由原始 AGPL-3.0 D-Fire YOLOv8n 初始化，使用本项目扩充数据的 balanced 清单在 RTX 3080 Ti 上完成 30 轮训练。权重原样保存在 `outputs/autodl_results/runs/dfire_expanded/experiment_001/train/weights/best.pt`，6,234,922 字节，SHA-256 `d1d7219bcafb8cad426a172350eaa6ff81381c6efa448ab821ea96c9ac1ef87a`；不随 Git 分发，不提供网络下载替代。数据来源与许可沿用下文“扩充火烟训练数据”记录，各来源条款分别保留。内部测试可能存在上游训练数据重叠及事件身份未知，不能证明真实车棚薄烟预警能力。旧模型及其权重保留，两个编辑器入口现默认选择此云端版本。

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

- `dfire_ludan` 当前指向负样本增量训练的 `experiment_002/train/weights/best.pt`（6,257,322 字节，SHA-256 `22fd0c9f6d2498e3ea3d5c3f456f1f13393d66e068a4d1d6c698b82777f2f810`）。在原 experiment_001 基础上混合 400 张 FASDD 含烟图、50 张 FASDD 背景图、40 张原 CVAT 标注图、99 张网络背景图与 32 张 jjs 背景帧训练；另用 45 张网络背景图加入验证集。保留模型名，没有新增模型选项；experiment_001 权重未覆盖，以下为其历史来源记录。

- 本地困难负样本（2026-10-30）：从 Wikimedia Commons 筛选 144 张无可见烟火的光头、墙壁及地板反光图片，逐图来源、作者、许可（CC BY / CC BY-SA / CC0 / Public domain）与摘要保存在被忽略的 `datasets/web_interference_negatives/sources.csv`；源图片与衍生 JPEG 不随 Git 分发，许可按每张文件页面分别保留。先按文件摘要和 dHash 去重，再按作者分组留出 99 张训练、45 张验证背景图。用户新增 `video/jjs.mp4`（第三方短视频，仅限本地）声明全片无烟火，30 FPS、565 帧，按 stride=15 提取 38 张；视觉核对后排除末尾 6 张结束页，32 张实际场景仅用于训练，不作为独立验证视频。所有新负样本标签为空，不能把它们再标为 smoke/fire；既有 16 张模糊有烟帧仍不参与训练。

- `dfire_ludan`：在本地 `dfire_small_smoke` 基础上，使用用户提供的 `ludan.zip` 中 40 张有标签的原视频帧（41 个烟雾框、8 个火焰框）和 200 张原 FASDD 训练样本继续训练；16 张不清晰且无标签的帧全部排除，未视为负样本。原视频为已登记的第三方本地素材，本次用户标注不改变原视频版权，图片、标注与衍生权重均不随 Git 分发。最佳权重位于 `runs/dfire_ludan/experiment_001/train/weights/best.pt`，6,261,802 字节，SHA-256 `d8942c16f18cb24b1fa4069245451426639cf5f6541386a417427ec09d40849e`；上轮权重未覆盖。此模型仅用于本地试验，不作为真实车棚选型依据。

- 本地微调产物 `dfire_small_smoke` 登记于 `sentinel/models/catalog.py`，权重为 `runs/dfire_fasdd_small_smoke/experiment_001/train/weights/best.pt`，6,262,442 字节，SHA-256 `4c76642b14ef471f966220115391c3226b248e974bacf154f49a2da8a30d66ad`。这是本项目从 AGPL-3.0 的 D-Fire YOLOv8n 初始化训练得到的衍生产物，不是上游下载文件；仅保存在本地、不随 Git 分发，数据许可与来源差异见下文。下载脚本只校验该产物，不从网络获取或覆盖；原始 `dfire_yolov8n` 仍可独立选择。

- FASDD_CV 小烟雾微调实验（2026-10-30，用户明确要求）：原始来源为 Wang 等的 [FASDD / ScienceDB](https://doi.org/10.57760/sciencedb.j00104.00103)，ScienceDB 页面结构化元数据标注 **CC BY-SA 4.0**。实际读取第三方镜像 [seawsurf/fire_smoke_dataset_fasdd_cv](https://huggingface.co/datasets/seawsurf/fire_smoke_dataset_fasdd_cv)，固定 revision `896dec15c93396ed4990ad502a187be9271e38e7`。镜像及其 Roboflow 元数据标注 CC BY 4.0，与原始来源不同，不能视为取消原始许可条件。镜像为经过 Roboflow 整理的版本，不宣称与原始 ScienceDB 压缩包逐字节一致。本地通过 HTTP Range 提取 ZIP 条目，校验各文件 CRC32 和尺寸；未完整下载压缩包，**未验证完整压缩包 SHA-256**。数据、标注、抽样清单和训练产物只保存到已忽略目录，不随 Git 分发。标签从镜像 `0=fire,1=smoke` 转为 D-Fire 的 `0=smoke,1=fire`，保留入选图片全部原有框。微调使用原始 D-Fire 权重的独立副本，原始权重不覆盖；实验不自动替换默认推理模型。原始事件、机位身份不可核实，且 FASDD 与原模型 D-Fire 训练数据可能重叠，因此内部测试不作为独立车棚泛化能力证据。

- LwST-FSD：作者 [WCUSTC/LwST-FSD](https://github.com/WCUSTC/LwST-FSD)，固定源码 revision `af3ed8ac9686a607ec2330275fbb95cf8d1f910d`。作者公开 [FIgLib 五帧模型分享](https://pan.ustc.edu.cn/share/index/5b015f2c73e24fd48b82) 的 `epoch_10.pth` 为 141,991,741 字节；本次从作者地址下载的 SHA-256 为 `6ac5ca325d59f1bb48a8f63750ba1efd49b53cc5b34659395e8edf14ad8b3add`。**上游未提供独立摘要/签名，不能称为官方 LFS 哈希验证**；本项目固定该摘要以检测后续漂移。仓库未声明明确项目许可，权重许可未核清，仅限本地演示/复现，不随 Git 分发，不作为车棚选型结论依据。权重位于 `models/fire_smoke/lwst_fsd/`，作者源码按固定版本和 SHA-256 下载至被忽略的 `samples/lwst_fsd/upstream/`，不复制到本仓库分发源码中。推理适配器保留原始 TSM/PTA/CSP/YOLOX 前向层，替代训练框架中的标准 Conv/BN/SiLU 包装及训练日志反序列化，不训练、不重新导出或改写权重。FIgLib 数据集未下载。

- 训练数据（pyro-sdis、D-Fire、Domestic Fire and Smoke）**不随本仓库分发**，需从原始出处获取，许可归各自作者。
- FIgLib / HPWREN 为 CC BY-NC-ND，上游只用于回放评测，从未参与训练，本仓库也不分发。
- 本地 `samples/rf100_smoke/` 为 [RF100 smoke](https://universe.roboflow.com/roboflow-100/smoke-uvylj/dataset/2) 测试子集（CC BY 4.0），仅用于本机运行检查，已被 `.gitignore` 排除。
- 本地 `video/` 为第三方新闻视频，版权不在本项目，已被 `.gitignore` 排除，不得公开分发。

## 2026-10-30 扩充火烟训练数据

本轮原始文件、元数据快照、摘要、准备脚本及衍生 YOLO 数据均仅存于被忽略的 `datasets/acquisition_20261030/`。实际下载和验证状态以该目录的 `download_results.json`、各来源 `source_manifest.jsonl` 及 `merged/counts.json` 为准；以下登记来源不等于宣称所有文件已下载或用于训练。

- **FASDD_CV 完整镜像包**：沿用前述 `seawsurf/fire_smoke_dataset_fasdd_cv@896dec15c93396ed4990ad502a187be9271e38e7`，`FASDD_CV.zip` 为 3,410,626,268 字节，上游镜像 LFS SHA-256 为 `c7d34d3511fd3a4bd16fdf5412e974d3e31b8f1b400bc566d1f115a807f36152`。原始 [ScienceDB / DOI](https://doi.org/10.57760/sciencedb.j00104.00103) 页面本轮再次取得，其 JSON-LD 明示 CC BY-SA 4.0；页面和结构化元数据本地留档。镜像为经过 Roboflow 处理的版本，其 CC BY 4.0 声明不能替代原始条件。保留署名、来源及相同方式共享义务，不公开重分发原图。
- **D-Fire**：[作者原仓库](https://github.com/gaia-solutions-on-demand/DFireDataset) 的集合声明为 CC0 1.0，但 README 明确指出 `PublicDataset` 前缀图片仍受 HPWREN / FESB / FireNet 等原提供者条款约束。本轮实际下载 [hanvithSai/gavin-dfire-raw](https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw) 固定 revision `fee27ef6934d5bb569ae6271c123673f10a877c8` 的 `train.tar`、`test.tar`，以其 LFS 摘要校验；镜像卡片的 CC BY 4.0 不应理解为作者授权变化。原包完整保留，准备阶段将 `PublicDataset` 前缀图片隔离，不进入合并训练/验证/测试清单。未证明镜像与作者网盘包逐字节一致。
- **Pyro-SDIS v2**：[Pyronear 官方数据](https://huggingface.co/datasets/pyronear/pyro-sdis)，固定 revision `ebfc02d0cba7f7dfe0623d13bcf0f0aa8a38e3ee`，Apache-2.0。官方记录 12,378 张图像，其中 1,634 张为系统误报的无烟图；逐个 Parquet 文件与官方 LFS SHA-256 对比。保留 partner / camera / date、原始划分与署名；若相同摄像头日期组跨训练与留出集，剔除该组训练帧。v2 的烟雾类别为 `0`，不能套用 v1 的类别 `1`。
- **Annotated Fire–Smoke Image Dataset**：[Shouthiri Partheepan / Figshare](https://doi.org/10.25946/28747046.v1)，文章 ID `28747046`、文件 ID `53486522`，`MyData_Fire.zip`，544,994,383 字节。作者 API 元数据明示 CC BY 4.0，并提供 MD5 `fd9ed42cd67fd5f7c016ca127985e245`。但本轮读取包内 `data.yaml` 发现其 Roboflow 项目 `fire-smoke-obstacle-dataset` version 10 的 `license: Private`，与发布页不一致；尚未核实这仅为发布前项目状态还是来源权利冲突，因此完整原包保留、转换后全量隔离，不进入默认训练/验证/测试。使用提供者 MD5 验证原包，另记录本地 SHA-256；不将本地 SHA-256 冒充官方发布的 SHA-256。
- **RISE 困难负样本候选**：[CMU CREATE Lab / deep-smoke-machine](https://github.com/CMU-CREATE-Lab/deep-smoke-machine) 作者明确声明数据为 CC0（代码为 BSD-3-Clause）。本轮仅从作者发布的 2020-02-24 元数据及 camera split 0 选择最终 `label=0` 且研究者状态为 strong/gold negative 的片段，不把工业有烟片段转成无烟图。固定源码 revision、视频摘要、片段 ID 和帧号保存在该来源本地清单；是否进入双类训练集须以实际视觉核对和合并清单为准，片段无工业烟标注本身不提供火焰框。
- **新增 Wikimedia Commons 干扰候选**：仅选择逐图标明 CC BY / CC BY-SA / CC0 / Public domain 的来源，保留原页面、作者、许可链接和图像摘要。搜索结果不等于负样本真值；蒸汽火车、可能有烟火的摊位、绘画、图文扫描件等不进入训练集。只有经视觉核对批准的图片才产生空标签；原图和候选图片均不随 Git 分发。

不同来源许可分别保留，不把混合数据统一重新许可。所有训练框统一为 `0=smoke, 1=fire`；空标签只表示该来源明确标注或本轮核对后无可见烟火的图片。去重和保留原始划分不能证明所有来源事件独立，也不能消除基础权重预训练数据重叠。

训练数据预检查时，对入选数据的 35 张本地 JPEG 衍生副本仅补上结束标记 EOI，逐张确认解码像素不变，保留原始来源摘要并记录新的衍生文件摘要，见本地 `merged/jpeg_repairs.json`。8 个下载原包保持不变；这不修改模型权重或替代上游许可。

## 免责

这些模型是研究工具，不是经过认证的火灾报警装置——上游作者也明确这样声明。本仓库未在电动车车棚场景做过任何性能验证，不要把其中的模型直接当作生命安全保障设备部署。
