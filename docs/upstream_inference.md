# 按上游示例运行模型

2026-10-30 核对当前五份固定 revision 的模型卡、本机 Ultralytics 8.4.192 的 `Model.predict` 实现，以及 LwST 固定权重内的 `meta.cfg`。这是一组用于人工回放的上游示例配置，尚未经过车棚场景的阈值校准。

## 两个编辑器入口

视频入口的 `MODELS` 控制启用模型，每个模型依次处理同一素材；只想跑一个时，注释其他行即可。尺寸恢复为各自原生值，`CONFIDENCE=None` 自动使用下面对应阈值。`scripts/run_models.py` 同样使用自动阈值；LwST 仍使用视频基线入口。

手动调参统一放在运行入口顶部：`CONFIDENCE` 设置置信度，`IOU` 设置 NMS 阈值，视频入口 `MODELS` 中的数字设置模型尺寸。填写数字就直接用于所选模型，只有 `None` 才回退到下面的默认配置。已删除之前新增的 LwST 专用变量及 `model_conf` / `model_iou` 单模型覆盖接口，不再存在第二层阈值覆盖。模型目录仍登记上游原值，不会根据视频结果自动调参。

`CONFIDENCE` / `IOU` 填数字或命令行提供 `--conf` / `--iou` 时，直接用于全部所选模型。控制台打印最终配置及来源；视频基线 `manifest.json` 的每个模型记录 `conf` / `conf_source`，`summary.csv` 记录同样信息。顶层 `conf=null` 表示使用模型默认阈值。带框视频底部显示本次尺寸、阈值、IoU 和权重哈希。

保存与预览保持开启，逐帧运行 `VIDEO_STRIDE=1`。预热三次和连续 1/3/5 帧对照属于本项目的运行设置，作者没有为这五个模型共同指定这组告警规则。

## 参数与依据

- **FireSmockDetect YOLO11m**：`imgsz=1280, conf=0.10, iou=0.7, rect=True`。尺寸和阈值来自固定版本[模型卡 Usage](https://huggingface.co/TianchenGuan/FireSmockDetect-models/blob/a914a2a036686cf09bac8efc738971fdc8f31a29/README.md)；作者没有在该示例另设 NMS，当前 checkpoint 记录 `iou=0.7`，与库默认值一致。`rect=True` 是库的 predict 默认。作者明确要求按部署场景另行校准置信度，0.10 是示例值。
- **Early Fire and Smoke**：`imgsz=640, conf=0.25, iou=0.7, rect=True`。固定版本[模型卡 Usage](https://huggingface.co/mfranzon/fire-smoke-yolov8/blob/f1c6426b069c1849cbf13b1ef5d2a260289286db/README.md) 直接调用 `model(video, stream=True)`，没有声明专用阈值。640 是已校验 checkpoint 的模型覆盖尺寸；0.25 是本机库的 predict 默认，0.7 与 checkpoint 及库默认一致，rect 使用 predict 默认。
- **D-Fire YOLOv8n**：`imgsz=640, conf=0.25, iou=0.7, rect=True`。固定版本[模型卡 Usage](https://huggingface.co/rabahdev/fire-smoke-yolov8n/blob/13017fe8af477c25f5298d168e2dfede4b000753/README.md) 直接调用 `model(image)`；640 与作者训练说明及 checkpoint 一致，阈值与补边跟随库的 predict 默认，IoU 与 checkpoint / 库默认一致。作者没有给出额外推荐阈值。
- **Safety YOLO26**：`imgsz=640, conf=0.25, iou=0.7, rect=True`。固定版本[模型卡 Usage](https://huggingface.co/e1250/safety_detection/blob/f2049aa749dcc844e406d2507ffa3b683ce83aac/README.md) 同样只调用 `model(image)`；640 来自 checkpoint 的模型覆盖尺寸，阈值与补边跟随库 predict 默认，IoU 与 checkpoint / 库默认一致。实际检测头 `end2end=False`，使用传统 NMS。类别保持 checkpoint 的 `0=smoke, 1=fire`；模型卡中的相反顺序不用于改写权重标签，语义差异仍待核实。
- **LwST-FSD FIgLib 五帧**：`imgsz=1024, conf=0.0001, iou=0.65, rect=False`。依据当前固定摘要 `epoch_10.pth` 的 `meta.cfg` 中 `model.test_cfg` 及测试 resize/pad 配置；采用五帧输入、BGR、0..255 浮点、等比例缩放和右下 114 补边。对应[作者固定源码版本](https://github.com/WCUSTC/LwST-FSD/tree/af3ed8ac9686a607ec2330275fbb95cf8d1f910d) 的配置文件另写有 `score_thr=0.01`，与所发布 checkpoint 保存的 `0.0001` 有差异；本次按当前权重内的测试配置运行，并保留差异说明。0.0001 是复现配置，出现更多低分框不代表检测正确；尚未完成与原 MMDetection 框架的数值一致性验证。

未另设的 YOLO 参数保持 Ultralytics 默认，包括 `max_det=300`、`agnostic_nms=False`、`augment=False`。本机 `Model.predict` 显式设置 `conf=0.25, rect=True`；不能用训练参数或全局配置中的 `conf=None / rect=False` 代替实际预测默认。官方说明见 [Ultralytics predict 参数](https://docs.ultralytics.com/modes/predict/)。

固定版本的上游原文副本保存在本地忽略目录 `reports/verify/upstream_profiles_2026-10-30/`，Hugging Face 文件通过镜像读取同一 revision；权重版本与字节保持不变。

## 本次验证

20 项单元测试通过，覆盖模型阈值选择、手动覆盖与零阈值。五个权重通过 SHA-256、加载和 GPU 单图推理；四个 YOLO 图片比较完成。五模型各完成同一段 7 帧短视频推理、连续帧评估和 H.264 保存，实际尺寸、conf 与 IoU 已核对；本地记录在 `reports/verify/after_upstream_profiles.json`、`reports/verify/upstream_profiles_images/run_001/`、`reports/verify/upstream_profiles_video/run_001/`。本次没有运行完整素材回放或启动系统播放器，完整视频效果留待人工检查。

后续 LwST 本地调参验证：20 项测试通过；保持 `vvvv.mp4`、1024 输入及逐帧运行，重新推理 `conf=0.01 / IoU=0.30` 后，532 帧总框数为 177、152 帧有框、23 帧多于一个框、单帧最多 3 框。原上游配置记录 `samples/video_results/run_037/` 为 3506 个框、单帧最多 33 框。调后 CSV、完整 MP4 和抽查画面位于本地 `reports/verify/lwst_tuned_vvvv/run_001/`；仍有错误候选，不能将框数减少解释为已验证的误报率改善。另用 YOLO11m 与 LwST 同跑 7 帧，确认单模型覆盖只修改 LwST：YOLO11m 保持 `conf=0.10 / IoU=0.7`，记录在 `reports/verify/lwst_tuning_isolation/run_001/`。
