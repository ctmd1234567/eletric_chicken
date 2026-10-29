# 电鸡哨兵：项目说明与准备情况

核查日期：2026-10-29（Asia/Shanghai）。本页依据项目 PDF、当前脚本和本次实际运行结果编写。

## 1. 项目做什么

[方案 PDF](电鸡哨兵-基于强化学习与yolo算法.pdf)提出：在电动自行车停放充电场所，用一台云台摄像机主动巡视，争取在明火出现前识别可疑白烟并告警。

YOLO 负责判断画面中的烟火；强化学习根据充电风险、可疑检测历史、区域未被查看的时长及云台姿态，决定看哪里、放大多少、是否复核或告警。方案计划用全景裁剪和真实烟雾素材合成半真实环境，以带动作掩码的 PPO 为主、DQN 为对照，再接入 ONVIF 云台与网页管理平台。

计划比较固定摄像头、固定巡航、规则复核、风险贪心和强化学习策略。指标包括发现时间、30/60 秒内召回率、每天误报次数、遮挡车位覆盖率。PDF 中“比固定巡航缩短发现时间 30%、比贪心缩短 10%”是待验证目标，尚无本项目实验支持。

## 2. 当前已完成与缺口

已具备：

- 三个候选权重，以及固定上游 revision、大小、SHA-256 的下载清单。
- 下载/离线校验脚本、真实加载与单图推理脚本、多模型图片/视频比较脚本。
- 本机 Python 虚拟环境；40 张本地烟雾 JPG 图片。
- 本次三个模型的哈希、加载、单图推理和 40 图比较均完成；`pip check` 通过。
- 项目 `.venv` 已改为 PyTorch `2.8.0+cu129` 与 torchvision `0.23.0+cu129`，GPU 验证记录见下文。

尚未具备：

- 可训练的完整数据集：本地样例没有标签、训练/验证/测试划分或来源许可说明，不能当正式评测集。
- 车棚全景、可合成的烟雾素材、带起烟时间的视频和水汽/反光等负样本。
- Gymnasium 仿真环境、PPO/DQN 及已训练策略权重。
- 飞线充电、消防通道堵塞检测器。现有模型只能识别烟或火，不能覆盖 PDF 中的全部目标。
- ONVIF 控制、真实摄像机联调、告警服务、网页管理平台。
- 尚未开展实地预警验证；本地 GPU 推理检查不等于满足项目告警性能目标。

## 3. 模型下载检查

以下文件本地均存在，大小和完整 SHA-256 与 `scripts/download_models.py` 内固定清单一致，加载和推理均成功。共 66,082,217 字节（约 66.08 MB）。本次没有重新下载已通过校验的文件。

### FireSmockDetect / YOLO11m

- 文件：`models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt`
- 大小：40,578,796 字节；类别：`smoke`；参数：20,053,779；输入：1280。
- SHA-256：`1b6cb82295767ae9d0d5c4fb94dffcaf17bbbafc0e3af001d3fe517015902298`
- 上游：[TianchenGuan/FireSmockDetect-models](https://huggingface.co/TianchenGuan/FireSmockDetect-models/tree/a914a2a036686cf09bac8efc738971fdc8f31a29)。

### FireSmockDetect / YOLO11s

- 文件：`models/fire_smoke/firesmockdetect/expB_y11s_deployed.pt`
- 大小：19,241,370 字节；类别：`smoke`；参数：9,428,179；输入：1280。
- SHA-256：`cc56224c84092e1e374cee1b3ce317f17f7f80e7945bc6c9a3a24a083321ff0e`
- 与 YOLO11m 使用同一上游 revision：`a914a2a036686cf09bac8efc738971fdc8f31a29`。

这两个模型的上游任务是固定瞭望摄像头/无人机的野外烟雾检测；迁移到室内外车棚的早期白烟仍需场景验证，尚不能选定为最终检测器。

### Early Fire and Smoke / YOLOv8n

- 文件：`models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt`
- 大小：6,262,051 字节；类别：`fire`、`smoke`；参数：3,011,238；输入：640。
- SHA-256：`ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16`
- 上游：[mfranzon/fire-smoke-yolov8](https://huggingface.co/mfranzon/fire-smoke-yolov8/tree/f1c6426b069c1849cbf13b1ef5d2a260289286db)。
- 上游定位于家庭早期烟火场景；目前仅作为轻量候选。

上述上游用途及许可证已于核查日通过固定 revision 的原始模型卡核对。SHA 校验确认本地文件与清单相符，不证明模型准确率或适用性。

## 4. 本次运行结果

此前模型比较使用项目 `.venv` 的旧版本 PyTorch `2.15.0+cpu`。旧环境返回 `torch.cuda.is_available()=False` 是因为安装了 CPU 构建；现已按用户要求替换成 GPU 版，当前依赖见 [requirements.txt](requirements.txt)。本地 40 张 JPG 共 7,475,773 字节；未发现 `.part` 下载残留。没有发现本项目已下载的训练集、视频、全景或强化学习权重。

历史 CPU 记录：在同一批 40 张图片上，`conf=0.10`，各模型按上面分辨率运行：

- YOLO11m：37/40 帧出现检测框，平均 324.8 ms/帧。
- YOLO11s：35/40 帧出现检测框，平均 104.5 ms/帧。
- YOLOv8n：36/40 帧出现检测框，平均 16.2 ms/帧。

这些数字只验证可运行性。样例无标签，也没有独立的负样本集；“有框比例”不等于召回率或准确率。分辨率不同，计时包含首帧开销、未专门预热，不能当严格性能基准或最终选型结论。

历史下载与 CPU 核查记录见 [reports/download_check_2026-10-29.json](reports/download_check_2026-10-29.json)，其中 CPU 环境字段记录当时情况。原始本地运行输出位于 `reports/verify/`，被 Git 忽略。

### 当前 GPU 验证

项目 `.venv` 切换为 PyTorch `2.8.0+cu129` 后，CUDA 矩阵乘法通过，`pip check` 通过。三个模型均重新完成单图验证与 40 张图片推理，检测结果张量设备均为 `cuda:0`：

- YOLO11m：37/40 帧出现检测框，平均 54.4 ms/帧。
- YOLO11s：35/40 帧出现检测框，平均 14.2 ms/帧。
- YOLOv8n：36/40 帧出现检测框，平均 8.0 ms/帧。

原始结果记录见 [reports/gpu_check_2026-10-29.json](reports/gpu_check_2026-10-29.json)。上述计时包含首帧开销、未专门预热，输入分辨率也不同，仅用于确认当前环境可运行，不能视为严格速度基准；无标签样例不能证明检测准确性。

## 5. 安装与运行

### PyCharm 与 CUDA

2026-10-29 补充检查结果：

- 显卡：NVIDIA GeForce RTX 5070 Laptop GPU；驱动：576.80；`nvidia-smi` 显示驱动支持 CUDA 12.9。
- 先前已在 PyCharm 登记的 JupyterProject 环境中验证 CUDA 运算。
- 本次将本项目 `.venv` 也切换到相同 GPU 构建：PyTorch `2.8.0+cu129`、torchvision `0.23.0+cu129`、CUDA runtime `12.9`。保留本项目 Ultralytics 和其他依赖。

在 PyCharm 的项目 Python Interpreter 中选择本项目 `D:\eletric_chicken\.venv\Scripts\python.exe`，再运行 [scripts/verify_cuda.py](scripts/verify_cuda.py) 复核。也可在 PowerShell 直接运行：

```powershell
.\.venv\Scripts\python.exe scripts/verify_cuda.py
```

本次使用项目 Python 可执行文件验证，没有通过 PyCharm 界面点击运行，也没有修改 IDE 全局配置。PyCharm 中应选择上述项目解释器。

`requirements.txt` 已固定 `+cu129` GPU 构建并包含 PyTorch 官方下载源，不会再次按旧清单安装 CPU 版本。版本组合与安装源参见 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。这份清单面向支持该 CUDA 构建的平台；没有 NVIDIA GPU 或使用 macOS 时需另选适合平台的 PyTorch。

### 创建项目环境与下载模型

以下命令从仓库根目录执行。仓库不附带 `.venv`、权重和样例，克隆后需要创建环境、下载权重并自行准备图片。

Windows PowerShell（GPU 环境）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/verify_cuda.py
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/download_models.py --check
.\.venv\Scripts\python.exe scripts/verify_models.py --json reports/verify/load_check.json
```

Linux / WSL 使用独立的 Linux 虚拟环境，不能复用 Windows 的 `.venv`。例如将环境放在用户目录：

```bash
python3 -m venv ~/.venvs/eletric_chicken
source ~/.venvs/eletric_chicken/bin/activate
python -m pip install -r requirements.txt
python scripts/download_models.py
python scripts/download_models.py --check
python scripts/verify_models.py --json reports/verify/load_check.json
```

本次只验证了现有 Windows 环境，未重新创建干净环境，也未验证 Linux 安装。下载和哈希校验仅使用 Python 标准库，可以先于依赖安装单独执行。使用 GPU 时应先根据操作系统、驱动和 PyTorch 官方安装说明选择 torch/torchvision，不能仅凭安装成功判断 CUDA 可用。

下载默认使用 `https://hf-mirror.com`，可通过参数或 `HF_ENDPOINT` 修改；文件地址锁定 revision，完成后校验 SHA-256。官方端点命令如下：

```bash
python scripts/download_models.py --endpoint https://huggingface.co
python scripts/download_models.py --only firesmockdetect
python scripts/download_models.py --only early_fire_smoke
```

网络失败时检查端点和代理；脚本以 `.part` 写入临时文件，通过校验才替换目标文件。当前脚本不支持断点续传，重试会重新下载对应文件。本次未测试重新下载链路。

准备 `samples/test.jpg` 或将图片放入 `samples/images/` 后，使用所在环境的 Python 执行：

```bash
python scripts/verify_models.py --image samples/test.jpg --device 0 --json reports/verify/infer_check.json
python scripts/compare_models.py --source samples/images --conf 0.10 --device 0
python scripts/compare_models.py --source samples/carport.mp4 --stride 5
python scripts/compare_models.py --source samples/images --models early_fire_smoke --save-images
```

比较脚本默认在 `samples/images_results/` 中每次新建 `run_001/`、`run_002/` 等编号子目录，包含 `detections.csv`、`summary.csv` 和可选检测结果（`--save-images` 或 `--save-results`）。图片保存 JPG；视频通过 FFmpeg 的 libx264 编码为 H.264 / yuv420p MP4，无音轨，无框的视频帧也保留。文件完成编码后才以最终名称出现，避免打开未完成的 MP4。输出帧率为原帧率除以 `--stride`，保持近似原时长；`--stride 1` 逐帧检测并保留原帧率。可用 `--out` 指定其他结果父目录。`run_models.py` 的 `SAVE_IMAGES` 控制结果保存；`SHOW_IMAGES=True` 时图片逐张显示，视频处理完成后直接用系统播放器打开带框 MP4。关闭保存但开启视频显示时，使用临时 MP4 播放。目录可以包含图片和常见视频；`--stride` 使用正整数。视频抽帧会先载入内存，长视频建议先裁短。

`--device 0` 强制使用首张 GPU，`--device cpu` 使用 CPU；省略时由 Ultralytics 自动选择。比较汇总及单图推理记录包含实际设备字段，可核对是否为 `cuda:0`。

## 6. 后续工作顺序

1. 补齐有来源和许可的车棚数据、负样本及标签，固定独立测试划分，评价烟雾漏检和误报。
2. 采集全景与烟雾素材，构建裁剪/变焦/合成环境，建立固定巡航、规则复核、风险贪心基线。
3. 在同一环境中训练 PPO/DQN，以发现时间与误报告警约束评价，开展风险先验、变焦、复核消融。
4. 完成 ONVIF 云台闭环、告警与管理界面，再开展实地验证。

当前不需要为了检查下载而安装全部后续框架；尚未引入 Gymnasium、强化学习库或摄像机 SDK。

## 7. 仓库内容与许可

- `scripts/`：下载、验证、比较脚本。
- `requirements.txt`：本次环境的直接依赖版本。
- `reports/download_check_2026-10-29.json`：可提交的核查摘要。
- `models/`、`samples/`、`.venv/`、`.tmp_hf/`：仅本地使用，不上传 GitHub。
- [NOTICE.md](NOTICE.md)：第三方权重来源与许可证说明。

本项目当前为研究原型，尚未验证早期告警性能，不能作为已验收的消防报警系统。第三方模型卡标注 AGPL-3.0，数据集许可仍归各自权利人；仓库不重新分发权重或样例，也未替上游授予额外许可。
