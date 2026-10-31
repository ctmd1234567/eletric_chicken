# 电鸡哨兵 · eletric_chicken

当前快照随 Git 保存三份本地微调最佳权重：`dfire_small_smoke`、`dfire_ludan`、`dfire_ludan_focused`，均保持 `catalog.py` 登记的原路径与 SHA-256。克隆后可直接加载；完整训练目录、训练数据和第三方视频不分发。视频入口当前选用 `dfire_ludan_focused`、960 输入、置信度 0.15，素材路径由使用者自行设置。用户在新视频上的观察为烟雾识别效果好、仍有较多误报；尚无这些新视频的逐帧真值评测，不把该观察表述为已测精确率。

本次本地定向微调已完成，候选登记名 `dfire_ludan_focused`，保留第 22 轮最佳和 960 输入；旧参考集存在退化，原始 `dfire_ludan` 默认保留。调用、相同阈值回放与限制见 [本轮记录](docs/ebike_focused_local.md)。

2026-10-31 最新训练路线：用户要求本地从 `dfire_ludan` 继续微调，使用 762 张不同训练图，重点采样 140 张人工审核图及 32 张 jjs 干扰帧，不直接使用三万多张混合集。入口、采样占比、14 张人工确认背景和 9 个越界框裁剪见 [本地定向微调说明](docs/ebike_focused_local.md)。下文 AutoDL 打包、授权待核实及尚未开训等描述属于此前阶段，实际运行状态见本轮输出，不表示已完成训练。

2026-10-31 本轮准备从 `dfire_ludan` 继续训练的场景补充数据及私有 AutoDL 包：用户手工标注 40 张重点采样，另加入 52 张核对过的实际场景图片和 239 张合成监控图片。训练清单、叠框预览、来源限制及云端命令见 [本轮训练说明](docs/urban_smoke_autodl.md)。推理入口继续使用 `dfire_ludan`；按用户要求移除上一轮退化的 `dfire_expanded` 权重和登记，下文相关接入内容仅为历史记录。新一轮尚未启动正式训练。

2026-10-31 回放检查后，两个编辑器入口默认恢复为 `dfire_ludan`，云端 `dfire_expanded` 保留为注释候选。同为 960 / conf=0.15 / IoU=0.7，jjs 的 565 帧里旧模型 6 帧、新模型 273 帧输出烟雾框；该视频参与过旧模型训练，此结果仅说明这段视频的行为差异，不能作为独立泛化指标。云端模型从原始 D-Fire 初始化，未继承 ludan/jjs 专项微调，云端报告只与原始 D-Fire 比较。下文接入时默认启用云端模型的描述已被本条取代，调用名和保留路径仍有效。

**用 YOLO 识别烟火，让强化学习决定云台看哪里，面向电动自行车停放充电场所研究早期火情预警。**

车棚里的车辆会互相遮挡，远处的少量白烟在固定监控中也可能只有很少的像素。本项目希望通过云台转动、变焦与可疑点复核，提高车位覆盖率，并缩短发现可疑烟雾的时间。

**当前进度：保留 5 个上游候选权重及多个本地微调版本，支持图片对比、流式视频基线和连续帧告警对照。强化学习、仿真环境、摄像机闭环和管理平台仍待开发。**

2026-10-31 已取回 AutoDL RTX 3080 Ti 的 balanced / 960 / batch=8 / 30 轮训练结果，登记为 `dfire_expanded`；两个编辑器入口现默认启用此模型，其他版本保留。权重位于 `outputs/autodl_results/runs/dfire_expanded/experiment_001/train/weights/best.pt`，请保留该目录。云端训练和后续评测正常完成，内部结果见 [模型对比记录](docs/model_comparison.md)，不代表车棚预警能力。下文 2026-10-30 的准备和默认模型描述为当时记录，当前选择以入口配置为准。

2026-10-30 按用户要求完成首次 [D-Fire 小烟雾微调实验](docs/dfire_small_smoke_finetune.md)：使用 FASDD_CV 筛选的 1,000 张含小烟雾图片和 200 张背景图，独立保存微调权重与对照结果。原始模型保留，编辑器入口已选择 `dfire_small_smoke`；内部数据集上的提升不代表真实车棚预警能力。

同日已完成 [监控火烟训练数据扩充](docs/training_data_expansion.md)：下载并校验 8.5 GB 原包，合并去重及隔离后保留 73,855 张，包含小烟雾候选及反光、水汽等背景；提供全量与背景约占三分之一的训练清单，并通过小批量 CUDA 训练入口检查。该轮仅准备数据，未进行全量微调或替换现有模型。

用户自行开训使用 `scripts/train_expanded_smoke.py`（默认 35,122 张、960、batch=8、最多 30 轮）；本机短时测速外推约 9–14 小时，具体规模、耗时限制及检验命令见 [训练报告与手动启动说明](docs/expanded_training_runbook.md)。

[AutoDL 3080 Ti 部署说明](docs/autodl_3080ti.md) 对应的私有上传包已生成在 `outputs/cloud/`（约 3.2 GB，含校验文件）。Linux 路径重定位已用 WSL 小样本检查，云端安装与 3080 Ti 实测仍需在租到的实例上执行；本轮没有租赁、上传或启动云端训练。

**改代码先看 [代码修改入口](docs/code_map.md)**：日常参数在 `scripts/run_models.py` / `scripts/run_video_baseline.py`；模型配置集中在 `sentinel/models/catalog.py`；推理流程、视频读写、告警评估已分目录。

[详细准备情况与核查记录](read.md) · [模型对比记录](docs/model_comparison.md) · [协作规范 AGENTS.md](AGENTS.md) · [项目方案 PDF](电鸡哨兵-基于强化学习与yolo算法.pdf) · [第三方模型说明](NOTICE.md)

## 核心思路

1. **检测：看到了什么。** 用 YOLO 检测烟雾、明火，后续针对车棚早期白烟采集和标注数据。
2. **决策：接下来看哪里。** 结合充电风险、检测可疑度、区域未查看时间和云台姿态，选择视角、倍率、复核及告警动作。
3. **训练：在仿真中学习巡视。** 计划通过全景裁剪模拟云台转动和变焦，将烟雾素材合成到场景中，以 Gymnasium 环境训练带动作掩码的 PPO，并用 DQN 作对照。
4. **落地：连接真实设备。** 计划接入 ONVIF 云台摄像机与网页平台，展示监控画面、风险分布及告警记录。

研究将比较固定摄像头、固定巡航、规则复核、风险贪心和强化学习策略，评价发现时间、30/60 秒内召回率、误报次数和遮挡车位覆盖率。方案中的改善幅度属于研究目标，目前没有完成对应实验。

## 已有内容

- **模型下载与校验：** 固定上游版本、大小和 SHA-256，避免下载损坏或版本漂移。
- **模型验证：** 加载权重、读取类别与参数量、执行单张图片推理。
- **多模型比较：** 对图片目录或视频抽帧运行多个模型，输出检测明细、耗时和可选带框图片。
- **CUDA 检查：** 打印实际解释器、PyTorch/CUDA 版本及显卡，并执行真实 GPU 矩阵乘法。
- **环境：** Windows、Python 3.13.14、PyTorch `2.8.0+cu129`、torchvision `0.23.0+cu129`，本机为 RTX 5070 Laptop GPU。

本项目环境已通过 CUDA 矩阵运算、依赖检查及多个候选模型在 GPU 上的推理验证，实际结果设备为 `cuda:0`，见 [GPU 验证记录](reports/gpu_check_2026-10-29.json)。

当前保留以下五个候选；2026-10-30 已移除 Pyronear、FireSmockDetect YOLO11s 和 FASDD YOLO26n，历史实验记录见 [模型记录](docs/model_comparison.md)：

| 模型 | 大小 | 类别 | 输入 | 是否随仓库分发 |
|---|---|---|---|---|
| YOLO11m / FireSmockDetect | 40.58 MB | `smoke` | 1280 | 是 |
| YOLOv8n / Early Fire and Smoke | 6.26 MB | `fire`, `smoke` | 640 | 是 |
| YOLOv8n / D-Fire 微调 | 6.23 MB | `smoke`, `fire` | 640 | 否，本地候选 |
| YOLO26n / Safety Detection | 15.74 MB | `smoke`, `fire` | 640 | 否，本地候选 |
| LwST-FSD | 141.99 MB | `smoke` | 1024，五帧 | 否，本地候选 |

后三个用 `download_models.py` 自行下载。这些是待评估的候选模型。野外烟雾与车棚白烟存在场景差异；当前也没有飞线充电、消防通道堵塞检测能力。本地的 RF100 图片只用于运行检查，"有检测框的比例"不等于准确率或召回率。

## 快速开始

以下命令在 Windows PowerShell、仓库根目录执行，面向与 CUDA 12.9 构建兼容的 NVIDIA 环境。其他平台需按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/)选择对应构建。

```powershell
git clone https://github.com/ctmd1234567/eletric_chicken.git
cd eletric_chicken
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_cuda.py
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/download_models.py --check
```

`requirements.txt` 固定 GPU 版 PyTorch 并包含官方 CUDA 下载源。下载模型默认走 `hf-mirror.com`；需要官方端点时：

```powershell
.\.venv\Scripts\python.exe scripts/download_models.py --endpoint https://huggingface.co
```

自行准备一张图片 `samples/test.jpg` 或图片目录 `samples/images/` 后，运行：

也可以在 PyCharm 中选择项目 `.venv\Scripts\python.exe` 解释器，打开 `scripts/run_models.py`，在文件顶部设置素材路径、模型和设备，然后直接点击运行，无需填写命令行参数。当前默认 `MODELS = ["dfire_expanded"]`；取消其他模型行的注释可同时比较。`SOURCE` 仍为原来的视频路径，可改成图片、图片目录或其他视频。

云端模型的调用与旧模型相同：`run_models.py` 使用列表中的 `"dfire_expanded"`，视频专用入口 `run_video_baseline.py` 使用字典项 `"dfire_expanded": 960`。后者额外输出连续帧告警对照。参数仍在文件顶部调整，`CONFIDENCE=None` 时新模型使用库默认 0.25；视频入口沿用手动值 0.15，尚未针对车棚校准。`IOU=None` 时读取 checkpoint 的 0.7。两入口保存和播放结果的方式不变。

```powershell
# 直接运行编辑器入口，使用文件顶部的 SOURCE、MODELS 和其他设置
.\.venv\Scripts\python.exe scripts/run_models.py
.\.venv\Scripts\python.exe scripts/run_video_baseline.py

# 命令行选择云端模型；不自动打开播放器
.\.venv\Scripts\python.exe scripts/compare_models.py --source "video/Video Project 8.mp4" --models dfire_expanded --device 0 --stride 1 --save-images
.\.venv\Scripts\python.exe scripts/video_baseline.py --source "video/Video Project 8.mp4" --models dfire_expanded --device 0 --conf 0.15 --save-images
```

比较 D-Fire 与 YOLO11m 时，可设置 `MODELS = ["firesmockdetect_y11m", "dfire_yolov8n"]`。车棚薄烟效果仍待测，不能宣称哪个模型更好。D-Fire 的数据集、作者原始模型及已下载第三方模型的区别见 [NOTICE.md](NOTICE.md)。

`safety_yolo26`（Home Fire Dataset）原生输入为 640。Safety Detection 未声明权重许可，且模型卡与 checkpoint 类别顺序相反；代码保留 checkpoint 映射，详情见 [NOTICE.md](NOTICE.md) 和 [模型对比记录](docs/model_comparison.md)。

两个编辑器入口独立配置：`run_models.py` 支持图片和视频，快速输出检测明细、汇总及带框画面，耗时包含首次运行开销；`run_video_baseline.py` 仅处理视频，支持 LwST 五帧模型、单独预热、逐帧时间戳和连续 N 帧烟雾告警对照。正式记录视频时序实验使用后者，详见 [视频基线](docs/video_baseline.md)。

两个入口的手动参数统一位于文件顶部：`CONFIDENCE` 调阈值，`IOU` 调重叠框抑制；数字直接用于所选模型，只有 `None` 才采用模型默认值。启用哪些模型以当前 `MODELS` 为准，视频入口字典中的数字就是输入尺寸。参数及依据见 [上游推理配置](docs/upstream_inference.md)；手动只跑某个模型时注释其他行即可。

```powershell
# 在首张 GPU 上验证候选模型
.\.venv\Scripts\python.exe scripts/verify_models.py --image samples/test.jpg --device 0 --json reports/verify/infer_check.json

# 比较候选模型，输出 CSV
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/images --device 0

# 保存带框图片
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/images --device 0 --save-images
```

`--device 0` 指定首张 GPU；`--device cpu` 使用 CPU。结果默认保存在 `samples/images_results/` 中，每次运行新建 `run_001/`、`run_002/` 等编号子目录。可用 `--out` 指定其他结果父目录。`run_models.py` 中的 `SAVE_IMAGES` 控制结果保存：图片保存有检测框的 JPG，视频按模型分别保存 H.264 / yuv420p 编码的 MP4，无框的视频帧也保留。保存视频需要 PATH 中可用的 FFmpeg（含 libx264）；本机已验证。MP4 不含音轨，按原帧率除以 `VIDEO_STRIDE` 写入，时长与原视频近似一致；默认 `1` 逐帧检测并保留原帧率。命令行 `--save-images` 或 `--save-results` 均可开启保存。`SHOW_IMAGES=True` 时，图片在图片窗口中按键查看下一张；视频全部处理并完成编码后交给系统播放器打开，可以连续播放、暂停和拖动进度。关闭保存但开启显示时，会生成临时 MP4 用于播放；CSV 仍会输出。图片目录也可以放常见视频。图片和视频均逐帧处理，检测 CSV 随推理写入，不再将整段视频载入内存。

## 视频基线与连续帧告警对照

流式命令行入口 `scripts/video_baseline.py` 默认逐帧运行 YOLO11m、Early Fire and Smoke 和 D-Fire，各用对应上游示例阈值；编辑器入口 `scripts/run_video_baseline.py` 当前启用本地微调版 `dfire_small_smoke`（960 输入），原模型仍可单独选择。链路记录时间戳、完整检测框和预热/稳定态耗时，并自动比较连续 1、3、5 帧烟雾命中的规则。结果及人工标注模板保存在 `samples/video_results/run_00N/`；没有人工标签时不报告召回率、误报率或预警延迟。运行方法和指标定义见 [视频评测说明](docs/video_baseline.md)。

也可在 PyCharm 中打开 `scripts/run_video_baseline.py`，修改文件顶部的视频、模型等设置后直接运行。`SAVE_IMAGES` 和 `SHOW_IMAGES` 可控制保存和播放带框视频；视频编码需要 FFmpeg。

新入口的 `MODELS` 使用“模型名 → 输入尺寸”字典，启用一行即选择对应模型；已支持 `"lwst_fsd": 1024` 的原作者五帧模型。它需等待中心帧之后两帧，结果中单独记录上下文可用时间。权重、预处理与适用范围见 [视频评测说明](docs/video_baseline.md)。

```powershell
.\.venv\Scripts\python.exe scripts/video_baseline.py --source video --device 0
```

恶劣天气评测借鉴 WACV 2024 的研究，优先保留原始视频基线并按真实天气分层；方案与尚未实施的对照见 [恶劣天气评测设计](docs/weather_robustness.md)。

## 目录说明

```text
scripts/                 运行入口、日常参数；保留原命令
sentinel/
  models/catalog.py      唯一模型清单：权重来源、哈希、原生配置
  models/runtime.py      模型加载、预测参数与 IoU 来源
  models/lwst/           LwST 网络装配、五帧窗口和推理适配
  pipelines/             图片比较、视频基线流程
  media/                 读帧、编码、预览和素材下载
  evaluation/            连续帧告警与人工标注评估
  reports.py             CSV、实验目录和耗时统计
tests/                   回归测试
docs/code_map.md         按修改目的查找代码
models/fire_smoke/       权重：三个随库分发，其余仅本地
samples/、video/         本地素材和运行结果，不入库
```

其中三个权重已随仓库提交（合计约 66 MB，单文件均小于 100 MB，无需 Git LFS），克隆后即可直接推理；也可以运行 `download_models.py` 重新获取并校验。虚拟环境、样例图片、测试视频及运行生成的图片不上传 GitHub。

## 后续开发

1. 补齐有来源许可、标签及固定测试划分的车棚数据，加入水汽、反光等负样本。
2. 采集全景和烟雾素材，搭建仿真环境，完成巡航、规则、贪心基线。
3. 训练 PPO/DQN，开展风险先验、变焦、复核动作消融与误报约束评测。
4. 接入真实云台、告警服务和网页平台，开展实地验证。

## 使用边界

本项目为研究原型，尚未验证真实场景下的早期告警性能。第三方权重以 AGPL-3.0 原样分发（未修改字节），本仓库因此整体采用 AGPL-3.0；数据集与测试素材的许可独立，不随仓库分发。具体来源与使用说明见 [NOTICE.md](NOTICE.md)。
