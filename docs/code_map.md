# 代码在哪里改

日常运行只改 `scripts/run_models.py` 或 `scripts/run_video_baseline.py` 顶部的设置，然后直接运行该文件。当前选中的模型、素材和自定义尺寸以这两个文件为准；整理期间的本地配置修改也已保留。命令行入口仍支持原来的 `python scripts/xxx.py ...`。

## 按修改目的找文件

- **图片/视频快速比较的素材、模型、设备、阈值和保存开关**：`scripts/run_models.py`。
- **视频基线的素材、模型尺寸、抽帧、预热、连续帧数**：`scripts/run_video_baseline.py`。`MODELS` 中启用一行即选择该模型，数字就是本次输入尺寸。
- **新增模型、修改权重版本、哈希或原生推理设置**：`sentinel/models/catalog.py` 的 `MODELS`，同时更新 `NOTICE.md` 和 `docs/model_comparison.md`。
- **加载模型、IoU 优先级、类别显示和调用预测**：`sentinel/models/runtime.py`。
- **各模型上游推理配置与参数出处**：`docs/upstream_inference.md`；模型卡显式阈值登记在 `catalog.py`，无额外阈值的作者示例按库 predict 默认值解析。入口 `CONFIDENCE=None` 自动按模型选择，填写数字则显式覆盖。
- **下载与权重校验**：`sentinel/models/weights.py`；加载/单图验证：`sentinel/models/verify.py`。
- **快速对比流程**：`sentinel/pipelines/compare.py`。`compare_model` 处理一个模型，`run` 负责组织模型和报告。
- **视频逐帧推理与计时**：`sentinel/pipelines/video.py` 的 `process_video`；`run` 组织模型、素材、实验清单和离线评估。
- **连续 N 帧告警规则**：`sentinel/evaluation/temporal.py` 的 `alarm_states`。标注读取和指标计算也在此文件，不需要重新推理就能重算。
- **读图片/视频、抽帧**：`sentinel/media/inputs.py`。
- **画框、保存与预览**：`sentinel/media/output.py`；FFmpeg 编码和系统播放器：`sentinel/media/video.py`。
- **B 站/腾讯视频下载**：`sentinel/media/fetch_bilibili.py`、`fetch_qqvideo.py`，命令行仍从 `scripts/` 运行。
- **LwST 五帧模型**：`sentinel/models/lwst/` 下 `detector.py` 是预处理、推理和解码，`architecture.py` 是固定网络装配，`video.py` 是五帧窗口，`sources.py` 固定作者源码版本与摘要。
- **CSV、实验编号目录、分位数**：`sentinel/reports.py`。项目根目录只在 `sentinel/paths.py` 定义。
- **测试**：`tests/`。

业务代码只依赖 `sentinel` 内部模块，不再从 `scripts` 导入实现。`scripts/_bootstrap.py` 只为直接运行脚本添加项目导入路径，没有推理逻辑。

## IoU 与输入尺寸

IoU 优先级为：本次显式覆盖 → 已登记的上游推理推荐 → checkpoint 的 `train_args.iou` → Ultralytics 默认值。报告同时记录最终值和 `iou_source`，训练参数不冒充上游推理推荐。

- LwST 保留作者推理值 `0.65`。
- 当前四个 YOLO 权重的 checkpoint 都记录 `0.7`，代码读取各自记录，没有给全部模型强制赋同一个值。
- 2026-10-30 已移除 Pyronear、FireSmockDetect YOLO11s、FASDD YOLO26n 的登记、权重和 Pyronear 专用参数；历史报告保留。
- 原生尺寸同样只维护在 `catalog.py`。快速对比入口的 Early Fire and Smoke / D-Fire 默认由旧的 1280 恢复为 640；运行入口中显式指定的尺寸优先。

## 本次整理的行为变化

删除重复的模型/尺寸/IoU 表、`video_detectors.py` 转接模块、入口的“设置转命令行再解析”及元数据读取中的吞错回退。视频不再整段载入内存；检测明细边推理边写入。重复的编码、预览、CSV 和实验目录代码各保留一份。

权重校验、视频解码完整性、标注有效性和资源释放仍保留；LwST 的训练日志兼容方法是 checkpoint 反序列化实际需要的符号，不是可删的空壳。

缺失或哈希不符的模型会明确失败，避免“选了多个模型却悄悄只跑部分”。快速对比 P95 改用与视频基线相同的 nearest-rank 算法。保存文件名包含素材完整名称，例如 `dfire_yolov8n_clip.avi.mp4`，同名不同扩展名不会互相覆盖。预览在每个素材完成编码后打开；关闭 `SHOW_IMAGES` 可批量运行。仅预览时的临时 MP4 留在系统临时目录，供外部播放器读取。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/download_models.py --check
.\.venv\Scripts\python.exe scripts/verify_models.py --image samples/test.jpg --device 0
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/images --device 0
.\.venv\Scripts\python.exe scripts/video_baseline.py --source video/example.mp4 --device 0
```

2026-10-30 本机重构验证：17 项测试通过；8 个权重通过 SHA-256、加载及 GPU 单图推理；7 个 YOLO 模型完成图片比较。由本地 RF100 图片生成的 7 帧短视频以 stride=2 对比重构前提交 `9237600`：8 模型共 32 条检测记录除计时字段外一致，连续帧评估 CSV 一致；8 个保存视频均为 H.264/yuv420p、4 帧。上述只验证软件链路，不评价车棚检测能力。播放器人工交互和第三方下载网络未在本次重构中复测。

同日移除三个模型后的复核：17 项测试通过；剩余 5 个模型通过权重校验、加载及 GPU 单图推理（`reports/verify/after_remove_models_2026-10-30.json`）；4 个 YOLO 完成单图快速比较（`reports/verify/remove_models_images/run_001/`）；5 个模型各完成 7 帧短视频和连续帧评估（`reports/verify/remove_models_video/run_001/`）。本次未开启保存画面或播放器，结果仅用于检查删除后的软件链路。
