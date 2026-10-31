# 扩充数据训练报告与手动启动说明

日期：2026-10-30。数据准备、文件审核、小批量训练入口检查和短时测速已经完成；**正式全量训练尚未启动，由用户手动开始并检验**。来源、许可、排除原因见 [数据扩充记录](training_data_expansion.md)。

使用 AutoDL 3080 Ti 的上传包、安装、路径迁移和启动步骤见 [云端部署说明](autodl_3080ti.md)。3080 Ti 的时间只是规划估算，尚无该卡实测。

## 训练多少

整理后的可用清单共 73,855 张。训练、验证、测试互不挪用；35,122 和 57,118 都是**训练侧图片数量**，不是包含验证/测试的总量。

- **初训方案 `balanced`**：35,122 张训练图，含 23,415 张正样本（烟或火）、11,707 张背景。保留全部训练正样本，其中有烟图 18,777 张、小烟框候选 3,917 张；专项背景来源保留 1,811 张，其余背景固定抽样补足。跑满 30 轮对应 1,053,660 次训练图片呈现，不是新增一百万张独立图片。
- **全量方案 `full`**：57,118 张训练图，含相同的 23,415 张正样本、33,703 张背景。跑满 30 轮对应 1,713,540 次训练图片呈现。
- 两方案共用 **7,781 张验证图**选择最佳权重，以及 **8,956 张测试图**做前后对照；训练过程中不使用测试集更新参数。
- 背景包含实际系统误报、反光地面/玻璃、眩光、湿地面、雾、水汽及工业无烟片段。多数普通背景没有“反光/水汽”等逐图细分类，不能把所有 42,862 张背景都叫做已核验的反光图。

先使用 `balanced`：既保留全部烟/火正样本和专项干扰，又避免把所有普通背景一次混入。`full` 清单完整保留，需要时可以用相同流程另开实验。图像近似去重不保证事件完全独立；“小烟框”也不是“薄烟/早期/遮挡”真值标签。

## 大概多久

在本机 RTX 5070 Laptop 8 GB、现有 PyTorch/CUDA 环境实测，batch=8、AMP=False、workers=0；从初训清单固定抽取 512 张，每个尺寸先预热 4 个批次，再测 32 个批次（256 张）。包含真实读取、增强、GPU 前向及反向，**没有 optimizer.step()，没有保存新权重**。另用 64 张测读取/前处理/前向/NMS 速度，不计算精度指标。

- **640 输入**：训练约 83.8 张/秒；预测约 137.5 张/秒。
- **960 输入**：训练约 39.3 张/秒；预测约 83.6 张/秒。

按每轮训练加同一验证集估算，并对持续满载、散热、I/O 和系统占用留出约 50% 的上浮空间：

- `balanced` + **960**：每轮约 **17–26 分钟**；跑满 30 轮，含初末对照及启动余量，约 **9–14 小时**。
- `full` + **960**：每轮约 **27–40 分钟**；30 轮整体约 **13.5–20.5 小时**。
- `balanced` + **640**：每轮约 **8–13 分钟**；30 轮整体约 **4.5–6.5 小时**。
- `full` + **640**：每轮约 **13–19 分钟**；30 轮整体约 **6.5–10 小时**。

这些是短时测量外推，**没有实际跑完一个大数据集 epoch**，范围不是保证。验证的额外开销按预测耗时的 1.5 倍估计，未实测整套 AP 计算。当前训练流程还会额外执行原模型/新模型各 640、目标尺寸测试，四次合计 35,824 张测试图片推理，通常再预留约 10–20 分钟。笔记本长期降频、其他程序占用和首轮扫描可能让实际时间更长；连续 8 轮无改善会早停，实际也可能更短。实际设置可在第一轮结束后用日志的 epoch 时间重算。原始测量与计算见本地 `datasets/acquisition_20261030/merged/training_throughput.json`。

## 这仍然算微调吗

**算。** 这次从已有 `models/fire_smoke/dfire_yolov8n/best.pt` 初始化，在扩充数据上更新主干和检测头，可称为“全参数微调”或“基于预训练权重的再训练”。微调不是只训练检测头，也不按新旧图片数量划分。Ultralytics 的 [微调说明](https://docs.ultralytics.com/guides/finetuning-guide/) 和 [Python 训练用法](https://docs.ultralytics.com/usage/python/) 同样以是否沿用预训练权重区分。

35,122 张训练图的数量超过 [D-Fire 公开集合](https://github.com/gaia-solutions-on-demand/DFireDataset) 的 21,527 张，但其中有 11,707 张背景，而且不同集合可能互相收录；这不等于获得了更多独立火灾事件，也不能据此证明覆盖了作者及基础模型的全部训练经历。下载原始包里的约 14 万张也不是最终训练量。

从模型结构 YAML 建立随机权重、再训练所有层才是从头训练。它需要单独的初始化、学习率及轮数实验，不能只把现有脚本名字从“微调”改成“训练”。本轮先用预训练权重，保留当前可复核起点；若后续想比较从头训练，另建同划分实验。

## 怎么开始

新入口是 `scripts/train_expanded_smoke.py`，可以直接在 PyCharm 打开运行；默认已经选中 `balanced`、960、batch=8、最多 30 轮。它只负责显式选择这次新数据，再调用已有训练实现，不会误用之前 1,200 张的小子集。**本轮仅执行了 `--dry-run`，没有替你启动正式训练。**

PowerShell 从项目目录启动：

```powershell
# 先进入你克隆的项目目录

# 可先查看配置；这行不会训练
.\.venv\Scripts\python.exe scripts/train_expanded_smoke.py --dry-run

# 正式开始：初训数据、960 输入、最多 30 轮
.\.venv\Scripts\python.exe scripts/train_expanded_smoke.py --dataset balanced --epochs 30 --imgsz 960 --batch 8
```

入口会先做两次原模型测试，再进入训练，开始时看到验证/测试进度是正常行为。训练使用 AdamW、lr0=0.0003、warmup=2、patience=8、seed=20261030、AMP=False、轻度 Mosaic；按验证集选取 `best.pt`，完成后自动做同尺寸的训练前后对照。

首次实验目录为 `runs/dfire_expanded/experiment_001/`。若目录已存在，脚本会拒绝覆盖；新实验显式指定 `--out runs/dfire_expanded/experiment_002`。要跑全量背景，使用：

```powershell
.\.venv\Scripts\python.exe scripts/train_expanded_smoke.py --dataset full --epochs 30 --imgsz 960 --batch 8 --out runs/dfire_expanded/full_001
```

想先缩短一轮实验，可以用 `--epochs 10`；960 下初训约 3–5 小时，全量约 4.5–7 小时，仍需加初末对照时间，且较少轮数不保证足够收敛。若显存不足，把 `--batch 8` 改为 `--batch 4`，相应重新观察耗时。运行期间保持插电，避免系统睡眠中断。

## 训练后怎么检验

正常跑完先检查 `runs/dfire_expanded/experiment_001/comparison.json` 的 `status` 是否为 `complete`。核心产物：

- `train/weights/best.pt`：按验证集挑选的最佳权重；`train/weights/last.pt`：最后一轮权重。
- `train/results.csv`：逐轮训练损失与验证指标，可检查是否继续改善或明显过拟合。
- `comparison.json`：原模型与新模型在同一测试集、同一输入尺寸下的指标。先看 `classes.smoke` 的 AP50、AP50–95 和召回变化，同时检查火焰类别，不能只看总体均值。

这里的 P/R 使用库选定的 F1 工作点，不是固定部署阈值的误报率；数据可能与预训练来源重叠，因此这些是内部对照结果。正式验收还要固定置信度（例如项目已有的 0.10 工作点）检查无烟图误报，特别关注单独留出的反光、水汽和未参与训练的新监控视频；无标签视频的“有框帧占比”不能当召回率。

正常流程已经自动评测，不用重复跑同一组。如果以后使用其他训练入口或断点恢复，可以手动补测：

```powershell
.\.venv\Scripts\yolo.exe detect val model=runs/dfire_expanded/experiment_001/train/weights/best.pt data=datasets/acquisition_20261030/merged/data_balanced.yaml split=test imgsz=960 batch=8 device=0 conf=0.001 iou=0.7 project=runs/dfire_expanded/experiment_001 name=manual_test_960

# 对本地 video 目录做定性回放，先保持与既有实验相同的 conf=0.10
.\.venv\Scripts\yolo.exe detect predict model=runs/dfire_expanded/experiment_001/train/weights/best.pt source=video imgsz=960 device=0 conf=0.10 iou=0.7 save=True project=runs/dfire_expanded/experiment_001 name=local_video_check
```

若中断后需要续训，使用 [官方 checkpoint 恢复方式](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings)：

```powershell
.\.venv\Scripts\yolo.exe detect train model=runs/dfire_expanded/experiment_001/train/weights/last.pt resume=True
```

这会恢复 checkpoint 的训练配置，不会重新执行本项目包装器的四组前后对照；因此恢复完成后用上面的手动测试补齐，不把旧 `comparison.json` 未完成状态当成完整评估。

## 数据交付检查

73,855 张的配对、坐标、类别和文件摘要审核已通过。短时速度测量发现框架会自动重编码末尾没有标准 EOI 标记的 JPEG，本轮将所选数据中的 35 张仅在衍生副本补上 EOI，并逐张验证解码像素完全一致，记录原始和新摘要；8 个原包全部保持不变。此后独立重新审核全部 73,855 张通过，见 `merged/jpeg_repairs.json`、`merged/audit.json`。正式入口的 balanced/full 两种 `--dry-run` 均已检查，未改动现有模型目录或默认推理选择。
