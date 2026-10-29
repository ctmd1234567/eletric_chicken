# AGENTS.md — 电鸡哨兵项目规范

面向在本仓库里工作的编码代理与协作者。内容只描述**已经核实的事实**；未验证的内容统一标注为"待验证"。

## 1. 项目目标

研究电动自行车停放充电场所的**早期薄烟预警**：锂电池热失控通常先冒白烟、后起明火，中间可能只有几十秒。固定摄像头下车棚遮挡严重、远端烟雾像素极少，本项目让一台云台摄像机在强化学习控制下主动巡视，YOLO 负责"看到了什么"，策略负责"看哪里、怎么看"。

**当前阶段只做模型侧准备**：下载、校验、加载、在同一批素材上跑通对比链路。训练、仿真环境、强化学习策略、云台闭环、管理平台均**尚未开始**。

明确不做的事：

- 不把野火/航拍场景的论文指标当作电动车车棚的检测能力。
- 不虚构召回率、误报率、预警时间等未实测的数字。
- 不在没有许可的情况下分发数据集或第三方视频。

## 2. 目录结构与职责

2026-10-30 重构后，`scripts/` 只保留运行入口；实现位于 `sentinel/models/`、`sentinel/pipelines/`、`sentinel/media/`、`sentinel/evaluation/`，测试位于 `tests/`。按修改目的查阅 [docs/code_map.md](docs/code_map.md)。同日已移除 `pyronear_sensitive`、`firesmockdetect_y11s`、`fasdd_yolo26n` 的登记、本地权重和专用参数。当前保留 YOLO11m、Early Fire and Smoke、D-Fire、Safety YOLO26、LwST-FSD 五个模型，以 `sentinel/models/catalog.py` 和 `docs/model_comparison.md` 为准。下方模型清单、分发数量和五模型实测数据是早期历史记录，涉及已移除模型的描述不代表当前状态。

```text
README.md                  项目概览、快速开始
read.md                    完整方案说明与历史核查记录
NOTICE.md                  第三方权重来源与许可声明
AGENTS.md                  本文件：协作规范
LICENSE                    AGPL-3.0 全文（随权重许可要求）
requirements.txt           固定版本的依赖清单
docs/
  model_comparison.md      模型对比表（未测项写"待测"）
models/fire_smoke/
  firesmockdetect/         FireSmockDetect 的 YOLO11m / YOLO11s 权重
  early_fire_smoke/        Early Fire and Smoke 的 YOLOv8n 权重
scripts/
  download_models.py       按固定 revision + SHA-256 下载权重
  verify_models.py         哈希校验 + 加载 + 单图推理
  compare_models.py        多模型批量对比，输出 CSV
  run_models.py            可在编辑器里直接运行的入口（无命令行参数）
  verify_cuda.py           CUDA 真实矩阵运算自检
  fetch_bilibili.py        下载 B 站测试视频（仅本地，不入库）
reports/
  verify/                  加载/推理验证的 JSON 记录（不入库）
  comparison/              对比 CSV 与带框截图（不入库）
samples/                   本地测试素材（不入库）
video/                     本地测试视频（不入库）
datasets/                  数据集（不入库）
```

## 3. 模型清单

| 权重 | 路径 | 类别 | 输入 | 参数量 | 许可 |
|---|---|---|---|---|---|
| `v5bA_y11m_best.pt` | `models/fire_smoke/firesmockdetect/` | `smoke`（单类） | 1280 | 20,053,779 | AGPL-3.0 |
| `expB_y11s_deployed.pt` | `models/fire_smoke/firesmockdetect/` | `smoke`（单类） | 1280 | 9,428,179 | AGPL-3.0 |
| `fire_smoke_yolov8.pt` | `models/fire_smoke/early_fire_smoke/` | `fire`, `smoke` | 640 | 3,011,238 | AGPL-3.0 |
| `best.pt` | `models/fire_smoke/pyronear_sensitive/` | `item`（单类，类名无语义） | 1024 | 9,428,179 | AGPL-3.0 |
| `best.pt` | `models/fire_smoke/dfire_yolov8n/` | `smoke`, `fire` | 640 | 3,011,238 | AGPL-3.0 |

上游来源：

- FireSmockDetect：<https://huggingface.co/TianchenGuan/FireSmockDetect-models>，revision `a914a2a036686cf09bac8efc738971fdc8f31a29`；代码与报告 <https://github.com/TianchenGuan/FireSmockDetect>。
- Early Fire and Smoke：<https://huggingface.co/mfranzon/fire-smoke-yolov8>，revision `f1c6426b069c1849cbf13b1ef5d2a260289286db`。
- Pyronear Sensitive Detector：<https://huggingface.co/pyronear/yolo11s_sensitive-detector>，revision `3df695158beed0a34e2f5a48245773fd9855ddd6`（v1.1.0）；训练数据为 pyronear/pyro-dataset。
- D-Fire YOLOv8n：<https://huggingface.co/rabahdev/fire-smoke-yolov8n>，revision `13017fe8af477c25f5298d168e2dfede4b000753`；D-Fire 微调，作者报告 test split mAP50 0.754。

前三个权重随仓库分发；`pyronear_sensitive` 与 `dfire_yolov8n` 是本地候选，**未**纳入版本管理（`.gitignore` 只放行前两个目录），需要时用 `download_models.py` 自行下载。

五者的 SHA-256 与官方 LFS oid 一致（2026-10-29 核对），完整值见 `sentinel/models/catalog.py` 的 `MODELS`。

**两个 FireSmockDetect 权重不是两种不同算法**：都是 YOLO11 + 同一套训练配方（pyro-sdis + D-Fire），差别只在骨干规模（m / s）与训练轮次。`expB_y11s_deployed.pt` 是上游服务的生产默认，`v5bA_y11m_best.pt` 是冻结测试集上固定摄像头表现更好的重训版本。

## 4. 技术栈与环境

- Windows + Python 3.13（项目内 `.venv`）
- PyTorch `2.8.0+cu129`、torchvision `0.23.0+cu129`，本机 RTX 5070 Laptop，`cuda:0` 可用
- ultralytics `8.4.192`、opencv-python `5.0.0.93`、numpy `2.5.4`

**已知环境坑（务必注意）：**

1. `dill` 必须固定在 `0.3.8`。ultralytics 加载老 checkpoint 时会自动装 `dill` 最新版（0.4.1），而 torch 2.8 的 `torch.utils._import_utils` 调用了已被移除的 `dill.extend`，装完 0.4.1 会导致**整个 torch 无法导入**。修复：
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --ignore-installed --no-deps "dill==0.3.8"
   ```
2. 国内直连 `huggingface.co` 不通，`hf-mirror.com` 可用。`download_models.py` 默认走镜像，下载后仍以官方 SHA-256 校验，可保证与官方逐字节一致。
3. 本机 `requests` 依赖链曾因中断的 pip 安装损坏（`certifi`/`idna` 缺 RECORD）。爬虫脚本已改为纯标准库 `urllib`，不依赖 `requests`。

安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --ignore-installed --no-deps "dill==0.3.8"
.\.venv\Scripts\python.exe scripts/verify_cuda.py
```

## 5. 常用命令

```powershell
# 下载权重（默认镜像；--check 只校验不下载）
.\.venv\Scripts\python.exe scripts/download_models.py
.\.venv\Scripts\python.exe scripts/download_models.py --check

# 加载验证 + 单图推理
.\.venv\Scripts\python.exe scripts/verify_models.py --image samples/rf100_smoke/images/xxx.jpg --device 0 --json reports/verify/infer_check.json

# 批量对比（图片目录或视频，视频按 stride 抽帧）
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/rf100_smoke/images --device 0
.\.venv\Scripts\python.exe scripts/compare_models.py --source video/xxx.mp4 --stride 3 --device 0 --save-images
```

结果默认落在 `samples/images_results/run_00N/`，含 `detections.csv`（逐帧明细）与 `summary.csv`（每模型汇总）。

## 6. 怎么在同一批素材上比较

1. 素材必须**完全一致**：同一目录、同一视频、同一抽帧间隔。
2. 五个模型的输入分辨率不同（1280 / 1280 / 640 / 1024 / 640），NMS 的 `iou` 也不统一（`pyronear_sensitive` 上游推荐 `iou=0.01`，其余为默认 0.7）。这些都是模型固有属性，**不要**为了"公平"强行统一——按各自推荐配置跑，并把这个差异写进结论。
3. 置信度阈值先统一用 `0.10`（`--conf`），再针对每个模型单独做阈值扫描；上游明确说明固定摄像头与航拍场景的阈值**不可互换**。
4. 视频场景务必按帧记录，因为要算的是"首次有效报警时间"，不是单帧准确率。

## 7. 必须记录的指标

| 指标 | 定义 | 采集方式 |
|---|---|---|
| 薄烟召回率 | 人工标注的薄烟片段中被检出的比例 | 逐帧人工核对 `detections.csv` |
| 误报率 | 无烟帧被判为有烟的比例，按干扰源分类（水汽、反光、尾气、行人） | 同上 |
| 推理延迟 | 单帧端到端耗时，区分预热与稳定态 | `summary.csv` 的 `mean_ms` / `p95_ms` |
| 首次有效报警时间 | 从真实起烟帧到首次满足告警条件的帧，单位秒 | 视频逐帧核对 |
| 告警稳定性 | 连续 N 帧命中才算报警，N 需在文档中写明 | 人工按帧序列核对 |

**禁止**把 `detect_rate`（有框帧占比）当作召回率。没有标签的素材只能说明"模型有没有输出"，不能说明"对不对"。

## 8. 数据集与权重管理

- 数据集一律不入库。`datasets/`、`samples/`、`video/` 已在 `.gitignore` 中排除。
- 权重按固定 revision 下载并校验 SHA-256，不手工改名、不重新导出、不修改权重字节。
- 新增权重必须同时更新：`sentinel/models/catalog.py` 的 `MODELS`、`NOTICE.md`、`docs/model_comparison.md`。
- 数据集使用前先确认许可，并在 `NOTICE.md` 里登记；许可不明的数据集只能做展示，不能作为决策依据。
- 第三方视频（如 B 站素材）仅限本地分析，**不得**提交到公开仓库。

## 9. 修改代码前 / 提交 / 发布

修改前：

1. `git status` 确认工作区状态，不要覆盖他人未提交的改动。
2. 涉及推理链路时先跑 `scripts/verify_models.py`，确认基线可用。
3. 改 `.gitignore` 时要确认没有把权重或素材误放进来。

提交前：

1. 跑一次 `verify_models.py` 与 `compare_models.py`，确认脚本仍可执行。
2. 检查 `git diff` 中是否混入绝对路径、Token、Cookie、个人目录。
3. 提交信息说明"改了什么 + 为什么"。

发布时：

1. 权重按 AGPL-3.0 分发，`LICENSE` 必须随仓库存在；修改推理代码后同样受 AGPL-3.0 约束。
2. 不使用 `git push --force`，不重写历史。
3. 单文件超过 100 MB 时改用 Git LFS 或 GitHub Releases，不要硬塞进普通提交。

## 10. 已知限制与待验证假设

**已核实：**

- 五个权重可加载、可推理，SHA-256 与官方一致。
- 在 RF100 `smoke` 测试子集（CC BY 4.0，76 张）上，五个模型分别有 73 / 71 / 71 / 16 / 75 张输出至少一个框，平均延迟 45.6 / 13.0 / 6.5 / 9.8 / 5.7 ms（`cuda:0`，含首次预热）。
- `pyronear_sensitive` 的低检出率（16/76）来自上游推荐的 `iou=0.01`——NMS 近乎最大抑制，为"单场景一个烟羽"设计，不是模型失效。换默认 `iou=0.7` 会得到完全不同的数字，比较时必须写明用的哪个值。
- FireSmockDetect 上游**没有**做过早期预警的时间指标验证——模型卡明确写"no early-warning claim is made"，因为缺少合格的起烟瞬间素材。
- Early Fire and Smoke 的模型卡描述了"连续帧状态机"（clear → smoke → early warning → fire），但**没有公开实现代码**，需要自己实现。

**未验证 / 待验证：**

- 两个模型在**电动车车棚**场景的薄烟召回率：**完全没有数据**。训练数据是野火（pyro-sdis、D-Fire）和家用火灾（Domestic Fire and Smoke），与车棚存在域差异。
- 车棚常见干扰源（水汽、地面反光、尾气、吸烟）下的误报率：**未测**。
- 从冒烟到明火的时间窗口内，模型能否稳定给出预警：**未测**。
- 上游报告的 R@10%FP（YOLO11m 0.388 / YOLO11s 0.354）只在 pyro-sdis 冻结测试集、IoU≥0.4、匹配负样本误报率的条件下成立，**不能**外推到本项目场景。
- 单类 `smoke` 的 FireSmockDetect 无法区分白烟与其他白色干扰物；双类模型虽有 `fire`，但本项目尚未验证其对锂电池明火的响应。
