# 电鸡哨兵 · eletric_chicken

面向电动自行车停放、充电场所的烟雾与火情检测研究项目。

项目以 YOLO 检测烟雾和明火，探索在车辆遮挡、远端小目标和复杂背景下识别可疑烟雾。后续计划结合云台转动、变焦与强化学习，实现主动巡视和可疑区域复核。

**目前已实现模型微调、图片与视频检测、多模型对比，以及连续帧告警规则评估。强化学习、仿真环境、真实云台闭环和管理平台尚待开发。**

[快速开始](#快速开始) · [主模型](#主模型) · [视频评测](docs/video_baseline.md) · [代码导览](docs/code_map.md) · [模型与素材来源](NOTICE.md)

## 主模型

**`dfire_ludan_focused`：面向电动车烟雾场景定向微调的 YOLOv8n 模型。**

- 检测类别：`smoke`、`fire`。
- 输入尺寸：960。
- 权重选择：30 轮训练中，按参考验证集 mAP50–95 保留第 22 轮。
- 文件大小：约 6.26 MB，已随仓库提供。
- [下载最佳权重](runs/dfire_ebike_focused/experiment_002/train/weights/best.pt) · [训练与评测记录](docs/ebike_focused_local.md) · [评测摘要](reports/training/ebike_focused_20261031.json)

模型从 `dfire_ludan` 继续微调，训练包含 762 张不同图片，对 140 张人工审核图片和 32 张干扰帧提高采样比例。相同阈值下，目标训练素材上的烟雾框匹配和干扰视频回放表现有所改善，但部分背景的误报增加，旧参考测试集指标也有下降。训练图上的改善不能作为未见场景的泛化结论；真实车棚的薄烟召回、误报率和首次报警时间仍需独立测试。

输入尺寸 960、1280、1536 的对比中，960 在现有专项回放上表现更好。置信度可从 0.15 或 0.25 开始比较，实际部署阈值需要结合场景标注校准。

其他基线与微调版本保留用于对照，模型标识、权重哈希及输入配置统一维护在 [`sentinel/models/catalog.py`](sentinel/models/catalog.py)。上游模型来源、许可和历史结果见 [NOTICE.md](NOTICE.md) 与 [模型对比记录](docs/model_comparison.md)。

## 已实现的功能

- **图片与视频检测：** 支持单张图片、图片目录和视频，输出检测明细与带框画面。
- **流式视频评测：** 逐帧处理并记录时间戳、检测框、预热与稳定态耗时。
- **连续帧告警对照：** 比较连续 1、3、5 帧命中规则，结合人工标注评估告警行为。
- **多模型比较：** 在相同素材上比较不同模型与阈值，保存 CSV 结果。
- **训练与标注处理：** 导入 YOLO 标注、裁剪越界框、检查图像与标签配对，并配置重点样本采样。
- **权重校验：** 按固定来源版本、文件大小和 SHA-256 验证模型文件。

## 快速开始

已验证环境为 Windows、Python 3.13、PyTorch 2.8.0 + CUDA 12.9 和 NVIDIA GPU。依赖文件固定了对应版本；其他平台需选择兼容的 PyTorch 构建。

在 Windows PowerShell 中执行：

```powershell
git clone https://github.com/ctmd1234567/eletric_chicken.git
cd eletric_chicken
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_cuda.py

# 校验仓库附带的主模型
.\.venv\Scripts\python.exe scripts/download_models.py --only dfire_ludan_focused --check
```

保存带框视频需要安装 FFmpeg，并将其加入 PATH。图片、测试视频和数据集不随仓库提供，需要自行准备有权使用的素材。

### 图片检测

将图片放在 `samples/test.jpg`，执行：

```powershell
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/test.jpg --models dfire_ludan_focused --device 0 --conf 0.25 --save-images
```

将 `--source` 换成图片目录即可批量处理。结果默认保存到 `samples/images_results/run_编号/`，包含 CSV 和带框图片。

### 视频检测

将视频放在 `video/test.mp4`，执行：

```powershell
.\.venv\Scripts\python.exe scripts/video_baseline.py --source video/test.mp4 --models dfire_ludan_focused --device 0 --conf 0.15 --stride 1 --save-images
```

结果默认保存到 `samples/video_results/run_编号/`，包括带框视频、逐帧检测、耗时统计和连续帧告警对照。每次运行创建独立目录，保留之前的结果；保存的视频不含音轨。

`--stride 1` 表示逐帧检测，`--device 0` 表示第一张 GPU。使用 CPU 时改为 `--device cpu`。详细参数及人工标注方法见 [视频评测说明](docs/video_baseline.md)。

### 在编辑器中运行

也可以在 PyCharm 等编辑器中选择项目的 `.venv` 解释器，修改入口顶部参数后直接运行：

- [`scripts/run_models.py`](scripts/run_models.py)：图片、图片目录或视频的快速检测与比较；主模型设为 `MODELS = ["dfire_ludan_focused"]`。
- [`scripts/run_video_baseline.py`](scripts/run_video_baseline.py)：视频检测与连续帧告警对照；主模型设为 `MODELS = {"dfire_ludan_focused": 960}`。

`SOURCE` 指定素材路径，`CONFIDENCE` 设置置信度，`IOU` 设置重叠框抑制阈值，`SAVE_IMAGES` 和 `SHOW_IMAGES` 控制保存与预览。输入路径需指向实际存在的本地素材。

## 评测与适用范围

本项目关注薄烟召回、无烟场景误报、推理延迟、首次有效报警时间与告警稳定性。没有人工真值的素材只记录检测输出，不计算召回率或精确率；有框帧占比也不等于识别准确率。

现有结果主要用于模型比较和训练回放，尚未建立独立、按事故划分的真实车棚测试集。水汽、反光、尾气等干扰仍可能触发误报。连续帧规则已支持离线对照，其对实际预警的收益仍需验证。

- [定向微调与固定阈值对比](docs/ebike_focused_local.md)
- [模型对比与实验记录](docs/model_comparison.md)
- [视频评测与告警指标](docs/video_baseline.md)
- [推理配置与参数来源](docs/upstream_inference.md)
- [恶劣天气评测设计](docs/weather_robustness.md)

## 研究方向

1. 建立包含早期薄烟、遮挡车位和常见干扰源的独立测试集。
2. 比较固定摄像头、固定巡航、规则复核与主动巡视策略。
3. 搭建模拟云台转动和变焦的环境，研究 PPO / DQN 巡视策略。
4. 接入 ONVIF 云台摄像机、告警服务与可视化平台，开展实地验证。

研究方案见 [项目方案 PDF](电鸡哨兵-基于强化学习与yolo算法.pdf)。

## 目录结构

```text
scripts/                 检测、比较、训练及数据处理入口
sentinel/
  models/                模型清单、加载与推理适配
  pipelines/             图片与视频处理流程
  media/                 素材读取、视频编码与下载
  evaluation/            连续帧告警与标注评估
  training/              标注导入、数据准备、训练与专项评测
  reports.py             CSV、实验目录与耗时统计
docs/                    使用说明、设计与实验记录
reports/training/        公开训练评测摘要
runs/                    已分发的指定最佳权重；其余实验产物忽略
tests/                   回归测试
samples/、video/         本地素材与运行结果，不随仓库提供
```

代码职责和修改入口见 [代码导览](docs/code_map.md)。训练数据准备与历史实验说明见 [训练数据扩充](docs/training_data_expansion.md)、[本地微调](docs/ebike_focused_local.md) 和 [AutoDL 部署](docs/autodl_3080ti.md)。

## 许可

项目采用 [AGPL-3.0](LICENSE)。第三方模型、训练数据和素材具有各自的许可条件，详见 [NOTICE.md](NOTICE.md)；模型权重的分发不表示训练素材可以再分发或用于商业用途。数据集、第三方视频和完整训练输出不随仓库提供。
