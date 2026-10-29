# 电鸡哨兵 · eletric_chicken

**用 YOLO 识别烟火，让强化学习决定云台看哪里，面向电动自行车停放充电场所研究早期火情预警。**

车棚里的车辆会互相遮挡，远处的少量白烟在固定监控中也可能只有很少的像素。本项目希望通过云台转动、变焦与可疑点复核，提高车位覆盖率，并缩短发现可疑烟雾的时间。

**当前进度：已完成三个候选模型的下载、完整性校验和推理验证，项目环境已切换为 NVIDIA GPU 版 PyTorch。强化学习、仿真环境、摄像机闭环和管理平台仍待开发。**

[详细准备情况与核查记录](read.md) · [项目方案 PDF](电鸡哨兵-基于强化学习与yolo算法.pdf) · [第三方模型说明](NOTICE.md)

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

本项目环境已通过 CUDA 矩阵运算、依赖检查及三个模型各 40 张图片的 GPU 推理验证，实际结果设备为 `cuda:0`，见 [GPU 验证记录](reports/gpu_check_2026-10-29.json)。

三个候选权重本地均已下载并通过 SHA-256 校验，总计约 66.08 MB：

- **YOLO11m / FireSmockDetect：** 约 40.58 MB，单类 `smoke`，1280 输入。
- **YOLO11s / FireSmockDetect：** 约 19.24 MB，单类 `smoke`，1280 输入。
- **YOLOv8n / Early Fire and Smoke：** 约 6.26 MB，`fire`、`smoke` 两类，640 输入。

这些是待评估的候选模型。野外烟雾与车棚白烟存在场景差异；当前也没有飞线充电、消防通道堵塞检测能力。40 张本地图片没有标签，只用于运行检查，“有检测框的比例”不等于准确率或召回率。

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

也可以在 PyCharm 中选择项目 `.venv\Scripts\python.exe` 解释器，打开 `scripts/run_models.py`，在文件顶部设置素材路径、模型和设备，然后直接点击运行，无需填写命令行参数。默认运行三个模型并保存有检测框的图片。修改 `MODELS` 列表可选择模型，例如 `MODELS = ["early_fire_smoke"]` 只测试 YOLOv8n；列表中填写多个名称可同时比较。

```powershell
# 在首张 GPU 上验证三个模型
.\.venv\Scripts\python.exe scripts/verify_models.py --image samples/test.jpg --device 0 --json reports/verify/infer_check.json

# 比较三个模型，输出 CSV
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/images --device 0

# 保存带框图片
.\.venv\Scripts\python.exe scripts/compare_models.py --source samples/images --device 0 --save-images
```

`--device 0` 指定首张 GPU；`--device cpu` 使用 CPU。结果默认保存在 `samples/images_results/` 中，每次运行新建 `run_001/`、`run_002/` 等编号子目录。可用 `--out` 指定其他结果父目录。在 `run_models.py` 中可用 `SAVE_IMAGES` 控制是否保存带框图片，用 `SHOW_IMAGES` 控制是否逐张显示，显示时按任意键继续；CSV 仍会保存。图片目录也可以放常见视频，视频按 `VIDEO_STRIDE` 抽帧处理。长视频会占用较多内存。

## 目录说明

```text
README.md             项目概览与快速开始
read.md               完整方案说明、下载检查与运行记录
NOTICE.md             第三方模型来源及许可说明
requirements.txt      当前 GPU 环境的直接依赖
scripts/
  download_models.py  下载权重并校验大小和 SHA-256
  verify_models.py    权重加载与单图推理验证
  compare_models.py   多模型图片/视频比较
  run_models.py       可在编辑器中直接运行的模型入口
  verify_cuda.py      CUDA 真实运算检查
reports/              可提交的环境与验证摘要
```

模型权重、虚拟环境、临时下载文件、样例图片及运行生成的图片不上传 GitHub。克隆后需要自行安装依赖、下载权重并准备素材。

## 后续开发

1. 补齐有来源许可、标签及固定测试划分的车棚数据，加入水汽、反光等负样本。
2. 采集全景和烟雾素材，搭建仿真环境，完成巡航、规则、贪心基线。
3. 训练 PPO/DQN，开展风险先验、变焦、复核动作消融与误报约束评测。
4. 接入真实云台、告警服务和网页平台，开展实地验证。

## 使用边界

本项目为研究原型，尚未验证真实场景下的早期告警性能。第三方权重的模型卡标注 AGPL-3.0，数据集许可独立；本仓库不重新分发权重或素材。具体来源与使用说明见 [NOTICE.md](NOTICE.md)。
