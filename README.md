# 电鸡哨兵 · eletric_chicken

**用 YOLO 识别烟火，让强化学习决定云台看哪里，面向电动自行车停放充电场所研究早期火情预警。**

车棚里的车辆会互相遮挡，远处的少量白烟在固定监控中也可能只有很少的像素。本项目希望通过云台转动、变焦与可疑点复核，提高车位覆盖率，并缩短发现可疑烟雾的时间。

**当前进度：保留 5 个候选权重，均已完成本机校验和推理链路检查，支持图片对比、流式视频基线和连续帧告警对照。强化学习、仿真环境、摄像机闭环和管理平台仍待开发。**

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

也可以在 PyCharm 中选择项目 `.venv\Scripts\python.exe` 解释器，打开 `scripts/run_models.py`，在文件顶部设置素材路径、模型和设备，然后直接点击运行，无需填写命令行参数。当前启用四个 YOLO，设置以文件顶部为准。修改 `MODELS` 列表可选择模型，例如 `MODELS = ["early_fire_smoke"]` 只测试 YOLOv8n；列表中填写多个名称可同时比较。

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

流式命令行入口 `scripts/video_baseline.py` 默认逐帧运行 YOLO11m、Early Fire and Smoke 和 D-Fire，各用对应上游示例阈值；编辑器入口 `scripts/run_video_baseline.py` 已启用全部五个模型。链路记录时间戳、完整检测框和预热/稳定态耗时，并自动比较连续 1、3、5 帧烟雾命中的规则。结果及人工标注模板保存在 `samples/video_results/run_00N/`；没有人工标签时不报告召回率、误报率或预警延迟。运行方法和指标定义见 [视频评测说明](docs/video_baseline.md)。

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
