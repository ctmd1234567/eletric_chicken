# 视频基线与连续帧告警离线对照

本链路用于模型准备阶段，不训练模型。现有 `compare_models.py` 和编辑器入口保持原有行为；新增入口逐帧读取视频，不把整段解码帧放进内存。所有原始视频、标签和输出留在本地。

## 运行

在 PyCharm 中选择项目 `.venv` 解释器，打开 `scripts/run_video_baseline.py`，在文件顶部设置 `SOURCE`、`MODELS`、`CONFIDENCE`、`VIDEO_STRIDE` 和 `CONSECUTIVE`，然后直接运行。`MODELS` 现在是唯一的“模型名 → 输入尺寸”字典，启用哪一行就运行哪个模型；取消注释多行会分别运行多个模型，右侧数值决定各自的输入尺寸，不再另设 `MODEL_IMGSZ`。例如：

启用模型以入口当前 `MODELS` 为准；手动参数统一在入口顶部的 `CONFIDENCE`、`IOU` 及 `MODELS` 尺寸处修改，数字直接生效，只有 `None` 才使用各模型默认值。已删除 LwST 专用阈值和单模型覆盖接口。精确默认参数和出处见 [上游推理配置](upstream_inference.md)；下文自定义尺寸示例不代表当前默认配置。

```python
MODELS = {
    "dfire_yolov8n": 1920,
    "lwst_fsd": 1024,
    # "safety_yolo26": 640,
}
```

上例会运行 D-Fire 和 LwST-FSD，不运行 Safety Detection。控制台打印入口路径、所选模型、权重路径、SHA-256、类别、参数量和实际配置；带框视频底部也标注模型、尺寸和哈希前缀，便于核对切换。原来的 `run_models.py` 仍是另一入口，设置互不联动。

默认对照连续 1、3、5 帧；`SAVE_IMAGES=True` 将带框视频保存在结果目录，`SHOW_IMAGES=True` 在完成后交给系统播放器打开。编码使用原有 FFmpeg H.264 保存器，保留无框帧，视频无音轨；抽帧时输出帧率为原帧率除以 `VIDEO_STRIDE`。仅开启显示时会生成临时视频用于播放。视频显示检测框，连续帧告警对照在 CSV 中。

也可在仓库根目录使用命令行：

```powershell
# 命令行默认 YOLO11m、Early Fire and Smoke、D-Fire，各用上游阈值；逐帧、预热三次
.\.venv\Scripts\python.exe scripts/video_baseline.py --source video --device 0

# 指定视频和模型；同一次运行中的模型使用同一素材与抽帧间隔
.\.venv\Scripts\python.exe scripts/video_baseline.py --source "video/Video Project 8.mp4" --models firesmockdetect_y11m early_fire_smoke --device 0
```

输出到 `samples/video_results/run_00N/`，每次新建目录：

- `manifest.json`：素材 SHA-256、FPS、尺寸、声明/实读帧数、权重 SHA-256 与 revision、实际输入尺寸/IoU/类别、依赖版本、完成状态。只有 `status=complete` 是完整实验。
- `detections.csv`：每模型/视频/采样帧一行，含原视频帧号（从 0 开始）、时间、烟/火框数、烟雾最高置信度、完整框坐标/类别/置信度、推理与读取耗时。
- `summary.csv`：每模型/视频的检出帧占比、预热逐次耗时、预热后平均与 p95 推理耗时、平均读取耗时。
- `annotations_template.csv`：整段默认 `unknown`，用于人工拆分和填写区间。
- `temporal/frames.csv`：N=1/3/5 每个采样帧的连续命中数、告警状态、新触发状态。
- `temporal/events.csv`：每次新告警触发的帧号和视频时间；即使无触发也保留表头。
- `temporal/summary.csv`：各规则触发次数、告警帧数、首次触发时间；无标签时评价指标留空。
- `模型名_视频名.mp4`：启用保存时的带框视频；此前未开启保存的运行不会自动补出视频。

PyCharm 入口在 `MODELS` 中同时选择模型及尺寸，尺寸须为至少 32 的 32 倍数。命令行的 `--models` 用于选模型，`--model-imgsz dfire_yolov8n=1280 lwst_fsd=1024` 用于覆盖尺寸；未指定尺寸时使用 `sentinel/models/catalog.py` 的原生尺寸。实际参数记录在 `manifest.json` 和 `summary.csv`。命令行默认运行 YOLO11m、Early Fire and Smoke、D-Fire；LwST-FSD 和 Safety YOLO26 需显式选择。Pyronear、FireSmockDetect YOLO11s、FASDD YOLO26n 及其专用参数已移除。

## LwST-FSD 接入

`lwst_fsd` 使用作者公开的 FIgLib 五帧权重，默认 1024 输入、IoU=0.65。权重和来源许可说明见 `NOTICE.md`；缺少权重时运行 `scripts/download_models.py --only lwst_fsd`，首次加载会按固定 revision 和 SHA-256 下载四个作者源码文件至本地忽略目录。作者仓库只包含关键源码，本项目在现有 PyTorch 环境下执行其中的 TSM/CSP/PTA/YOLOX 前向层，提供标准 Conv/BN/SiLU 包装、预处理和 NMS，未安装或替换现有环境的 MMDetection/MMCV；也不加载无关的 DCN/XFeat 实验依赖。所有 567 个参数/缓冲区必须严格匹配，不能用其他 YOLO 权重替代。

网络输入顺序为 `[t-2, t-1, t, t+1, t+2]`，只有一个中心帧结果，越界支持帧按作者数据集流程替换为中心帧。原生预处理保持 BGR、0..255 浮点像素、等比例缩放和右下方 114 方形补边，不使用 YOLO 的 RGB/除以255/居中补边；用户可自行调整输入尺寸，但时序模型显存和耗时会随之增加。当前支持帧间隔为一个原视频帧，`VIDEO_STRIDE` 只控制评测目标帧间隔；这不等同于上游 FIgLib 图像序列的拍摄时间间隔。单图验证会重复五帧，只验证加载和前向。

该五帧配置需要读取未来两帧，不能当作无等待的实时模型。CSV 中 `timestamp_s` 是目标帧时间，`available_frame_index`、`decision_timestamp_s` 是凑齐上下文后的可用帧号/时间，`support_frame_indices` 记录五帧的原始帧号。时序规则的首次触发和标注后的报警延迟使用 `decision_timestamp_s`，包含这部分等待；画框仍对应中心帧。视频尾部以重复帧补齐，不额外虚构未来时间。分类帧率仍按目标帧标签计算。耗时包含五帧读取/预处理/网络/后处理的调用开销，预热后窗口缓存会减少读取开销，不是摄像机至报警服务的完整延迟。

这是原作者结构和权重的推理适配，尚未复现其完整 MMDetection 基准或论文指标，也没有车棚召回、误报和预警延迟的有效数据。测试会确认同一中心帧下改变支持帧会改变网络输出，避免把五帧模型误接成五次独立单帧推理。

2026-10-30 本地实跑：`samples/video_results/run_011/` 对 `Video Project 8.mp4` 逐帧运行 D-Fire（1920）和 LwST-FSD（1024），各 836 帧，两份权重哈希、预测内容不同，MP4 均可解码。统一 `conf=0.10` 时 LwST-FSD 在该片段没有框，不能把接入成功写成检出有效。作者 checkpoint 的测试配置为 `score_thr=0.0001`；另在 `video/vvvv.mp4` 用这一阈值运行的本地演示位于 `samples/video_results/lwst_native_threshold/run_001/`（532 帧）。阈值极低时出现框也不能说明正确；该阈值不应视为车棚推荐值。不同素材、阈值的结果不得混成同条件选型比较。

## 2026-10-30 LwST 无框排查

本次权重校验、加载及 GPU 前向通过，完整推理和 H.264 保存也通过，未复现启动失败。保持输入 1280、IoU=0.65，在 `vvvv.mp4` 上以 `conf=0.15` 运行 532 帧，39 帧输出框；结果在本地 `reports/verify/lwst_reproduce/run_001/`，`manifest.json` 为 `complete`。直接从保存视频提取第 432 帧确认框已画出，但最高分 0.633 的框位于画面顶部台标附近。此前 `samples/video_results/run_020/` 在 `Video Project 8.mp4` 上的 836 帧仅有第 677 帧输出框；复查保存画面，该框落在前景路面，未覆盖明显烟雾。

对 `s3248s55a3f.mp4` 保持输入 1280，以权重元数据 `meta.cfg` 中的 `score_thr=0.0001` 作诊断，1208 帧完整运行，891 帧有候选框；按输出分数过滤到 0.15 后仅剩 2 帧，过滤到 0.01 为 107 帧。记录在本地 `reports/verify/lwst_score_check/run_001/`。检查最高分所在的第 686 帧，多个候选框落在字幕附近；不能将降低阈值后出现更多框称作检测修复。这些是输出计数和少量画面核对，不是召回率或误报率，也未完成与原 MMDetection 框架的数值一致性验证。此次保留用户正在调整的运行入口配置，没有修改权重、推理代码或公共阈值。

## 规则定义

仅 `smoke` 类命中参与烟雾告警，单独出现 `fire` 不计为烟雾。规则是场景级的“连续 N 个采样帧存在至少一个烟雾框”；不同位置的框也可能构成连续命中，尚未做目标跟踪或烟羽关联。

命中数达到 N 时触发，后续持续命中保持告警；一次漏检立即解除并清零。缺帧、视频边界均清零，无保持时间和迟滞。N=1 是单帧基线，N=3/5 是对照参数，均不是已验证的部署配置。`stride>1` 时连续指采样帧，不能称作连续原始帧；从首次命中起，理论最少额外等待为 `(N-1)*stride/fps`。

`first_trigger_s` 是从视频开头计时的触发位置。时间采用 `frame_index / nominal_fps`，假设恒定帧率；对变帧率、拼接、变速或缺少起烟过程的视频，不能据此推断真实起烟时间。当前读取耗时独立记录；`infer_ms` 覆盖预处理/预测/后处理并同步 GPU，不含磁盘读取、CSV、可视化、告警服务及摄像机传输。预热另记，不计入稳定态均值；`warmup=0` 时首次启动开销仍在推理统计中。

## 人工标注后再评价

复制生成的模板为另一个本地 CSV，按不重叠的区间拆行：

```csv
video_id,start_frame,end_frame,state,interference,smoke_onset_frame
example.mp4,0,99,clear,reflection,
example.mp4,100,119,unknown,,
example.mp4,120,299,smoke,,120
```

上述仅为格式示例，不是任何本地视频的真实标签。`video_id` 必须与 detections 一致；起止帧均包含。`state` 为 `smoke`、`clear` 或 `unknown`；未覆盖部分也视为 unknown。干扰类型按实际情况填写，例如 steam/reflection/exhaust/person。只有核实了真实起烟帧才填 `smoke_onset_frame`；已有烟雾的截取片段不要把首帧当起烟帧。含剪辑的视频应按独立片段处理，避免规则跨剪辑延续。

```powershell
# 不必重新推理；out 必须是尚不存在的目录
.\.venv\Scripts\python.exe scripts/evaluate_temporal.py --detections samples/video_results/run_001/detections.csv --annotations samples/video_results/run_001/annotations_reviewed.csv --consecutive 1 3 5 --out samples/video_results/run_001/reviewed_temporal
```

标签用于评价，不改变告警状态。`smoke_frame_alarm_rate` 是已标注烟雾采样帧中的告警帧占比，`clear_frame_alarm_rate` 是已标注无烟采样帧中的告警帧占比；unknown 不计入分母。`frames.csv` 保留干扰类型供分类核对。`smoke_segments.csv` 按人工划分的烟雾片段记录是否出现告警与首次有效触发延迟；片段划分会影响片段级统计，需固定标注规范。延迟仅在已填真实起烟帧、且片段内有完全由起烟后 N 个采样帧构成的新告警时给出；起烟前已经持续的告警和跨起烟帧累计的命中不冒充该事件的成功检测，无有效新触发时留空并标记 `no_new_trigger`，不能用 0 代替漏检。

本地现有视频在 NOTICE 中登记为第三方新闻素材，缺少可用于选型的已核实标签和场景许可，只用于本地展示与链路检查。检出占比、触发次数减少和首次触发位置均不等于召回率、误报改善或提前预警。车棚场景能力仍待有许可的真实车棚视频及人工标注验证。

## 规则验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_evaluate_temporal.py -v
```

测试覆盖无烟/有烟、仅火框、连续命中、断帧/跨视频复位、抽帧等待、标注重叠、重复帧、unknown 排除、误报比例、起烟后延迟以及起烟前持续误报。

代码修改位置见 [代码地图](code_map.md)。IoU 由统一模型配置解析，保留各模型差异并记录 `iou_source`；推理实现位于 `sentinel/pipelines/video.py`，连续帧规则位于 `sentinel/evaluation/temporal.py`。
