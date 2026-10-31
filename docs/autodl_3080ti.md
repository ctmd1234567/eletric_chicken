# AutoDL RTX 3080 Ti 部署与训练

2026-10-31 新一轮场景补充包、重点采样和从 `dfire_ludan` 初始化的命令见 [本轮训练说明](urban_smoke_autodl.md)。训练入口现默认 `--dataset urban --base-model dfire_ludan`；本文下方 balanced / 原始 D-Fire / experiment_001 的配置属于首轮历史记录。旧上传包已按用户要求清理，不要再使用。

适用日期：2026-10-30。此文和上传包用于用户自行租用的私有实例；本轮没有创建付费实例、上传数据或启动正式训练。

## 时间和机器选择

3080 Ti 参考规格为 12 GB 显存，见 [NVIDIA 官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/)。本项目已在 5070 Laptop 8 GB 上测过 YOLOv8n、960、batch=8 的前向/反向，实测约 39.3 张/秒；**尚未在 3080 Ti 上实测**，不能按显卡名称承诺固定加速倍数。

先按以下时间预留，均含逐轮验证及初末对照，未包含上传、安装环境时间：

- 初训 balanced：35,122 张训练图、960、batch=8、最多 30 轮，先按 **8–14 小时**预算。
- 640 输入的初训：先按 **4–7 小时**预算。
- 全量 57,118 张训练图的 960 方案：先按 **13–21 小时**预算；本次上传包仅含 balanced，不能在该包上直接选择 full。

这些范围是参考本机实测留出的规划余量，不是 3080 Ti 的基准测试。云端先观察一轮时间，再用“已用时间 + 剩余轮数 × 每轮时间 + 最终测试余量”重算。保持 batch=8、AMP=False 与本地方案一致，先不要为了追求速度同时改变 batch、AMP 和其他参数；后续优化需单独验证。

AutoDL 选择一张 3080 Ti，优先 Python **3.12 或 3.13**、PyTorch 2.8.0、CUDA 12.6/12.8 的 Linux 镜像（以当时可选镜像为准）。NumPy 2.5.4 要求 Python>=3.12，不能直接使用 Python 3.10/3.11 镜像安装本项目固定依赖。数据及安装缓存放数据盘，建议预留至少 **30 GB 空闲空间**。

## 上传包里有什么

本地文件：`outputs/cloud/autodl_balanced_20261030.tar`，附 `.tar.sha256` 校验文件。解压后目录叫 `eletric_chicken/`。

- 训练图 35,122、验证图 7,781、测试图 8,956，共 **51,859 张图片及配对标签**，图片/标签净大小约 2.94 GB；最终 TAR 实际大小以旁边 JSON 记录为准。
- 训练所需源码、原始 D-Fire 初始化权重、许可和来源记录。
- Linux 安装脚本 `install_cloud.sh`，以及路径重定位/文件校验入口 `scripts/prepare_autodl.py`。
- 不含 Windows `.venv`、原始数据压缩包、第三方本地视频、Figshare 隔离数据、其他候选权重或之前实验产物。

来源条件在 `NOTICE.md` 和每图 provenance 中保留，上传包只用于你的私有训练实例，不发布到公开下载链接。

## 1. 上传到 AutoDL

可以先用无卡模式上传和解压，正常开机恢复 GPU 后再执行安装/检查脚本。无卡模式不保证稍后仍有原 GPU 空闲，详见 [官方省钱说明](https://www.autodl.com/docs/save_money/)。

在实例页面打开 JupyterLab，进入 `/root/autodl-tmp/`，上传 TAR 和 SHA256 两个文件。也可在**本地 Windows PowerShell**用 SCP，端口、地址必须换成实例页面提供的真实值；不需要把密码写进命令或发到聊天里。[上传文档](https://www.autodl.com/docs/scp/)

```powershell
# 将下方 PORT 和 HOST 换为实例 SSH 登录指令里的端口和主机名
scp -P PORT .\outputs\cloud\autodl_balanced_20261030.tar .\outputs\cloud\autodl_balanced_20261030.tar.sha256 root@HOST:/root/autodl-tmp/
```

## 2. 解压并装环境

以下命令在 **AutoDL JupyterLab 终端**执行：

```bash
cd /root/autodl-tmp
sha256sum -c autodl_balanced_20261030.tar.sha256
tar -xf autodl_balanced_20261030.tar
cd /root/autodl-tmp/eletric_chicken
python --version
```

确认是 Python 3.12/3.13；如是其他版本，换合适镜像或先用实例自带 Conda 创建相应版本环境。切换到**正常 GPU 开机模式**后执行：

```bash
nvidia-smi
bash install_cloud.sh
```

脚本保留已经可用的 PyTorch 2.8.0 / torchvision 0.23.0；不符合时按 [PyTorch 官方 2.8 安装命令](https://pytorch.org/get-started/previous-versions/)安装 cu126 构建。随后安装固定依赖、固定 dill=0.3.8、校验包内文件摘要，把路径生成到当前解压目录，执行真实 CUDA 矩阵检查，并只显示训练配置；**不会自动开训**。若换目录或实例迁移，重新运行 `python scripts/prepare_autodl.py prepare --verify` 即可重建路径。

若导入 OpenCV 提示缺少 `libGL.so.1` 或 `libglib-2.0.so.0`，在该实例中补装对应系统库后重试：

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
```

环境或 CUDA 检查失败时先处理报错，不直接启动长时间任务。本地没有真实 3080 Ti 云实例，安装/GPU 检查需要在你租到的机器上完成。

## 3. 开始训练

3080 Ti 的 12 GB 是 GPU 显存，本次 YOLOv8n、960、batch=8 可以使用；服务器 CPU 内存建议至少 16 GB，训练保持 cache=False，不需要把所有图片载入内存。云端实际显存占用仍以运行结果为准。

需要跑完自动关机时，将新增 `train_and_shutdown.sh` 上传到解压后的项目根目录，然后用它代替下面的普通启动命令：

```bash
cd /root/autodl-tmp/eletric_chicken
bash train_and_shutdown.sh
```

它运行默认 balanced / 960 / batch=8 / 30 轮，等待训练和后续评测全部完成后调用 AutoDL 官方 `/usr/bin/shutdown`；训练报错退出也会保存日志并关机，避免无人值守空耗。结果和日志留在实例数据盘，不删除实例；日志在 `runs/cloud_logs/`。按 Ctrl+C 暂停或收到 SIGTERM 不会触发自动关机。`--check` 仅显示行为、`--dry-run` 只检查训练配置，两者均不会训练或关机。额外参数可直接追加，例如 `bash train_and_shutdown.sh --epochs 10`。

此前已生成的 TAR 不包含该新增脚本，已上传旧包的用户只需补传这一个小文件；后续重新打包会包含它。关机方式依据 [AutoDL 官方文档](https://www.autodl.com/docs/save_money/)；只用于该实例的单个训练任务。本轮仅做脚本检查，没有实际执行关机。

确认 dry-run 显示 train=35122 / val=7781 / test=8956 后，在 JupyterLab 终端执行：

```bash
python scripts/train_expanded_smoke.py --dataset balanced --epochs 30 --imgsz 960 --batch 8
```

脚本先做原模型基线测试，再开始训练，最后对新模型做同尺寸测试；开始时看到验证进度是正常行为。输出在 `runs/dfire_expanded/experiment_001/`，若目录已存在，使用新 `--out` 名称，避免覆盖。

JupyterLab 终端可在本地浏览器断开后继续运行；如果通过普通 SSH 运行，使用 screen/tmux，见 [AutoDL 守护进程说明](https://api.autodl.com/docs/daemon/)。示例：

```bash
screen -S smoke_train
python scripts/train_expanded_smoke.py --dataset balanced --epochs 30 --imgsz 960 --batch 8
```

离开 screen 会话用 `Ctrl+A` 然后 `D`；稍后执行 `screen -r smoke_train` 回看。离开会话会继续计费训练；只有中断程序/关闭实例才会停下。

## 4. 暂停、继续与取回

每一轮训练及验证结束保存 `last.pt`。要暂停，等待本轮保存完成再按 `Ctrl+C`，随后按需关闭实例。中途停止最多重跑尚未保存的这一轮；如果首轮尚未产生 checkpoint，则没有可恢复断点。恢复命令：

```bash
cd /root/autodl-tmp/eletric_chicken
yolo detect train model=runs/dfire_expanded/experiment_001/train/weights/last.pt resume=True
```

正常完成训练的 checkpoint 与中断断点不同；不要先把目标设成 15 轮并正常结束，再当作原 30 轮任务直接恢复。独立 CLI 恢复只续训练，无法自动补完原包装器的对照报告，后续手动测试见 [手动训练说明](expanded_training_runbook.md)。

正常跑完查看 `comparison.json` 的 status=complete，并下载整个 `runs/dfire_expanded/experiment_001/`，至少保留 `best.pt`、`last.pt`、`results.csv`、`args.yaml` 和 `comparison.json`；权重位于其 `train/weights/` 下，CSV/YAML 位于 `train/`。下载示例同样在本地 Windows 执行：

```powershell
scp -r -P PORT root@HOST:/root/autodl-tmp/eletric_chicken/runs/dfire_expanded/experiment_001 .\runs\autodl_3080ti_001
```

关闭实例与释放/删除实例不同。确认结果已下载、需要的训练数据和环境已保留后，再决定是否释放实例；不要删除尚未取回结果的实例。本文没有代你充值、租赁或上传。
