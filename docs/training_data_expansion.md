# 监控烟雾训练数据扩充（2026-10-30）

本轮目标是获取可训练的大批量烟雾框标注及无烟干扰背景，场景不限于电动自行车。数据和标注仅保存在本地，不进入 Git。原始数据、复现脚本、许可快照与校验结果的根目录为 `datasets/acquisition_20261030/`；具体许可见 [NOTICE](../NOTICE.md)。本记录区分已找到的来源、实际下载文件与进入训练清单的样本，不使用论文指标推断本项目效果。

## 优先获取的来源

1. [FASDD / ScienceDB](https://doi.org/10.57760/sciencedb.j00104.00103)：多场景火烟及混淆背景。实际获取 [FASDD_CV 的 YOLO 镜像](https://huggingface.co/datasets/seawsurf/fire_smoke_dataset_fasdd_cv)，固定 revision `896dec15c93396ed4990ad502a187be9271e38e7`，ZIP 索引包含 95,126 张图像，原划分为 train 76,101 / valid 9,513 / test 9,512。镜像经过 Roboflow 处理，不能宣称是原始完整 FASDD（包括 UAV/RS）的 122,634 张，也不能将镜像标签当作逐张复核的真值。
2. [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset)：作者描述的 21,527 张图像来自监控、自采、网络和既有数据，包含 9,838 张无火无烟图。获取固定 revision 的第三方原始格式镜像，保留 train/test 归属。作者特别提示的 `PublicDataset` 前缀图像因来源条款未逐一核清，保留原包但隔离出合并清单。
3. [Pyro-SDIS v2](https://huggingface.co/datasets/pyronear/pyro-sdis)：来自实际固定摄像头，12,378 张，其中 10,744 张含烟、1,634 张是系统误报背景。保留 partner / camera / date 及原划分；官方是 Apache-2.0，类别 `0=smoke`。这比将清晰网络火灾图全部混在一起更贴近远端监控需求，但仍主要是野外火情。
4. [Annotated Fire–Smoke Image Dataset / Figshare](https://doi.org/10.25946/28747046.v1)：作者记录 11,027 张、YOLO 框标注、CC BY 4.0。读取包内 `data.yaml` 后发现 Roboflow `fire-smoke-obstacle-dataset` version 10 标为 `license: Private`，与发布页不一致。暂不能判断这是私有项目导出后公开授权还是其他来源条件冲突，因此原包下载保留，转换后的全部样本暂隔离，不进入默认训练清单。原文件 `MyData_Fire.zip` 的提供者 MD5 和本地 SHA-256 均记录在下载结果中。
5. [RISE / CMU CREATE Lab](https://github.com/CMU-CREATE-Lab/deep-smoke-machine)：原任务是工业烟排放视频分类，不能把其正样本直接当成烟雾检测框。另取作者 camera split 0 中研究者 strong/gold negative、最终 `label=0` 的 180 个视频，提取 720 张 320×320 背景帧，保留官方视频划分。对全部帧进行拼图视觉检查，排除 1 个可见橙色炉口/火花不明确的片段（4 帧），716 帧获准进入去重阶段。保留工业“无烟”标签语义；蒸汽/燃烧组成没有逐帧独立测量。
6. [Wikimedia Commons](https://commons.wikimedia.org/)：本轮检索得到 83 张候选，经视觉及来源描述检查批准 59 张，包含反射地面、玻璃窗、湿路面、雾、逆光眩光、电水壶水汽及地热蒸汽。排除 24 张可能含燃烧烟的蒸汽火车、摊位、图文扫描和无关内容。与既有 144 张逐图许可、已复核背景一起纳入统一去重；不会覆盖旧数据或把旧数量冒充本轮新增。

前四项提供者的标称数量相加为 140,058，但这是去重前的来源规模，不是独立事件数量。下载原始文件的计划体积为 8,512,512,379 字节，另有 RISE 视频、元数据和 Commons 图片；本地保留原包及衍生文件，磁盘占用会更大。全部摘要和数据审核通过后，已仅清理本轮数字续传块及重复拼接临时文件，回收 11,928,643,671 字节；原包、来源记录及图片/标签均保留，见 `cache_cleanup.json`。

FASDD 镜像的完整标签块已按 ZIP 条目尺寸和 CRC 解包清点：43,473 张含烟，39,119 张为空背景，2,513 张满足所有烟雾框面积不超过画面 2% 的候选尺度条件。这里还未扣除异常框、图像解码问题及重复项，不能把它们直接当作最终训练数量；完整分划统计在本地 `fasdd_label_inventory.json`。

## 进一步查到、未直接混入的来源

- [MIVIA LFDN](https://mivia.unisa.it/large-fire-dataset-with-negative-samples-lfdn/)：36,554 张，作者列明有反光、眩光、车灯、雾、灰尘等干扰，4,154 张为负样本。下载跳转到作者申请入口，本机访问入口超时；未下载，不宣称已获训练授权。
- [NEMO](https://github.com/SayBender/Nemo)：监控及云台视角、低/中/高密度烟，约 2,934 张的整理版本；官方目录提供 COCO 标注及 Nevada Box 图片分享，README 首页仍有占位下载文字。候选用途贴近需求，本轮未取得完整图片，不把代码仓库 Apache-2.0 自动等同于所有原始视频授权。
- [MIVIA Smoke Detection Dataset](https://mivia.unisa.it/datasets/video-analysis-datasets/smoke-detection-dataset/)：低分辨率固定机位视频及掩膜，包含无烟干扰，下载需作者网站流程；本轮未取得。
- [SmokeBench](https://arxiv.org/abs/2509.12701)：真实监控中有烟/清晰成对图像，原任务是去烟，不是现成 YOLO 检测框；暂不直接混入框监督。
- [MultiNatSmoke](https://github.com/henryzhao0615/MultiNatSmoke)、[SmokeMM](https://huggingface.co/datasets/ksdbc/SmokeMM)：烟雾分割汇编，来自多个既有来源，需要逐来源许可检查和掩膜转换。不能因为总量更大就视为新增独立框数据。
- FIgLib / HPWREN、AI for Mankind：有大量远端监控序列，但不同版本有 NC/ND 等限制，并且可能已被其他汇编收录；本轮不另行混入训练。
- [FireSpot-CNX](https://github.com/Biometrix-4/FireSpot-CNX)：真实泰国固定监控、早期烟雾及无烟背景，约 4,000 张，图片经 Google Drive 分享。官方 README 的文字/表格数量不完全一致，且本轮未在原发布页确认明确数据许可；不根据其他汇编转述的 CC BY 4.0 擅自入训。
- [S-Firedata / 国家地球系统科学数据中心](https://www.geodata.cn/data/datadetails.html?dataguid=221643895015416&docId=1271)：检索摘要提及复杂光照、遮挡及自采视频帧，完整数据需订单审核，平台另有限科研、复制修改等条款；本轮仅记录来源，没有申请、联系作者或下载使用。

## 准备和验证规则

- 原包只读保留，Hugging Face 的每个原文件与固定 revision 的 LFS SHA-256 对比，Figshare 文件与提供者 MD5 对比，另外计算本地 SHA-256。ZIP 读取同时检验条目 CRC；镜像摘要通过只证明与该镜像文件一致。
- 图片实际解码、标签配对、类别、有限数值、归一化坐标及边界均检查。缺标签不自动等同于负样本；仅显式空标签或审核批准的背景产生空标签。不执行下载文件内的脚本。
- 统一 `0=smoke, 1=fire`，保留同图全部合法框；Pyro-SDIS 的单类烟雾仍为 0。原始图片、框及来源信息在来源清单中保留，异常单独记录。
- 合并按原 test、val、train 的优先顺序处理，以 SHA-256、原图 ID、全局 dHash 距离不超过 4 保守排除重复/近似候选。同事件不同视角仍可能漏过；dHash 相似也可能排除本来不同的画面，排除记录可追溯。
- 不把原 test/val 调到训练。Pyro-SDIS 同摄像头同日期若跨训练/留出集，去除训练侧；Commons 以作者分组沿用既有划分规则；RISE 沿用视频/机位划分，不逐帧随机分组。
- `small_smoke` 仅表示所有烟框面积不超过图像 2%，不是“薄烟”或“早期起烟”标签。低分辨率边缘方差只作为待人工检查的画面清晰度线索。遮挡、首帧起烟时间没有统一真值，不能虚构这些标注。

## 本地复现入口

使用现有 Windows 虚拟环境进行网络下载及图片处理。WSL 可读同一工作区，但当前无 Pillow / PyArrow，且启动提示 Windows localhost 代理无法直接映射；本轮不重装项目环境。读取 Parquet 用的 PyArrow 单独放在被忽略的 `python_packages/`，没有修改项目依赖或全局配置。

```powershell
# 原包分块续传和摘要验证
.\.venv\Scripts\python.exe datasets/acquisition_20261030/download.py

# 原包验证通过后，解码、转换标签并生成全量合并清单
.\.venv\Scripts\python.exe datasets/acquisition_20261030/prepare.py
```

完整训练配置为 `datasets/acquisition_20261030/merged/data.yaml`，原始来源清单在 `prepared/<source>/source_manifest.jsonl`，统一清单为 `merged/manifest.jsonl`，排除原因在 `merged/excluded.jsonl`，数量在 `merged/counts.json`。这些文件本轮均已实际生成并通过核对。另有 `data_balanced.yaml`：训练侧保留全部合法正样本及专项困难背景，再固定抽样普通背景，验证/测试集不变。

本轮获取数据不会自动替换当前推理权重。后续训练可以使用现有入口指定此配置及新的实验输出目录；这是长时间训练命令，不代表本轮已经训练：

```powershell
.\.venv\Scripts\python.exe scripts/finetune_dfire_small_smoke.py --data datasets/acquisition_20261030/merged/data_balanced.yaml --out runs/dfire_expanded/experiment_001 --epochs 30 --imgsz 960 --batch 8
```

该入口会先后做原模型和微调模型测试，数据量变大后耗时明显增加。原 D-Fire 权重预训练过部分来源，合并数据也可能互相收录；所有测试只用于内部比较，不是独立事件泛化或真实火灾预警验收。

## 原包与来源检查实测

8 个原文件、共 8,512,512,379 字节已全部下载并通过提供者摘要校验（7 个 Hugging Face SHA-256、1 个 Figshare MD5），每个文件另记录本地 SHA-256，见 `download_results.json`。RISE 的 180 个视频下载和抽帧均成功，另保留各视频本地 SHA-256；该摘要没有作者独立签名或摘要佐证。

D-Fire 共 21,527 张原图，21,278 张通过图片/标签检查，249 张因坐标非法或框越界被隔离，具体原因在该来源 `rejected.json`；其中 9,838 张显式空标签背景全部通过，1,333 张合法样本另有 `PublicDataset` 来源条款待核。Pyro-SDIS 的 12,378 张全部通过检查，1,634 张为空背景；原训练集中 1,026 张与原验证集共享摄像头/日期组，合并时进一步排除训练侧。

Figshare 原包实际只有 11,021 张图片（train 8,826 / val 1,088 / test 1,107），均可解码且标签可解析，与作者描述的 11,027 张差 6 张；全量按包内 `Private` 与发布页 CC BY 4.0 的差异隔离。图片可解码不意味着许可问题或逐框真值已经解决。

## 合并完成结果与训练入口验证

FASDD_CV 的 95,126 张图像全部可解码，所有标签可解析。合并准备阶段共有 140,722 张合法来源图及背景候选，最终清单保留 **73,855 张、76,615 个框**。全量清单包含 24,907 张含烟图、42,862 张空背景，以及不含烟的火焰正样本；4,703 张属于上述小烟框尺度条件。

- 训练：57,118 张，其中含烟 18,777 张、小烟框候选 3,917 张、背景 33,703 张。
- 验证：7,781 张，其中含烟 3,018 张、小烟框候选 504 张、背景 3,974 张。
- 测试：8,956 张，其中含烟 3,112 张、小烟框候选 282 张、背景 5,185 张。
- 来源保留量：FASDD_CV 56,814、D-Fire 10,977、Pyro-SDIS 5,292、RISE 575、既有 Commons 143、本轮新增 Commons 54。最后两项是去重后的数量；不能把既有 143 张算作本轮新下载。

排除 66,867 张：52,836 张为 dHash 距离不超过 4 的**近似候选**、76 张为文件摘要相同；1,333 张为 D-Fire `PublicDataset`，另有 1,088 张与这批图像内容相似而一并隔离；Figshare 11,021 张按许可差异隔离；去重后剩余 Pyro 训练侧另排除 513 张跨验证摄像头/日期组的图像。近似候选并不等于全部经过人工确认的重复图片，部分可能是不同帧、不同烟量或相似背景；原包与来源清单完整保留，便于后续复核这些排除。

`data_balanced.yaml` 的训练清单为 **35,122 张**：23,415 张正样本（含烟或火焰）、11,707 张背景，其中 1,811 张来自 Pyro-SDIS 系统误报、RISE 及 Commons 的专项背景来源；全量清单不删图。分划的 `*_small_smoke.txt`、`*_negative.txt` 和 `difficulty_review.csv` 提供进一步筛选入口。

独立审核已重新核对全部 73,855 张的 SHA-256、标签与清单一致性和路径配对；训练/验证/测试之间 SHA-256 及原图 ID 交集均为 0，Pyro 训练与留出集的摄像头/日期组交集为 0，结果见 `merged/audit.json`。这不是对全部框的人工复标，也不证明所有来源事件完全独立。抽查拼图为 `merged/preview_small_smoke.jpg`、`merged/preview_interference.jpg`。

原始 D-Fire 权重通过 `verify_models.py --only dfire_yolov8n` 的哈希、加载及 CUDA 单图推理。另在新的 `runs/dataset_readiness/check_002/` 使用 72 张训练图、21 张验证图、640 输入执行 1 轮输入/梯度检查，CUDA 训练与验证完成、损失有限、原始权重哈希不变，见 `merged/training_readiness.json`。第一次检查的训练本身也已完成，但结果记录代码把当前库返回的损失字典当成张量，记录步骤失败；修正兼容后重新完整执行通过。两个小批量检查均不代表全量微调，不用于报告精度提升，未替换默认推理模型。

按用户后续要求，另完成不更新/保存权重的短时速度测量，训练规模、耗时外推与手动启动命令见 [训练报告](expanded_training_runbook.md)。框架预扫描发现 JPEG 末尾标记问题后，已对入选数据的 35 张本地衍生副本补上 EOI，验证解码像素逐字节相同，保留 `original_source_sha256`、更新衍生副本摘要，原包不变；随后重新独立核对全部 73,855 张通过，详见 `merged/jpeg_repairs.json` 与 `audit_final.log`。正式训练由用户启动。
