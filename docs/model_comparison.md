# 模型对比记录

本文件是后续实验的**填写模板 + 已知事实**。规则很简单：**没测过的字段写"待测"，不写估计值。**

核对日期：2026-10-29

## 1. 对比表

| 字段 | YOLO11m / FireSmockDetect | YOLO11s / FireSmockDetect | YOLOv8n / Early Fire and Smoke |
|---|---|---|---|
| **模型名称** | FireSmockDetect `v5bA_y11m_best`（v5b arm A，seed 3） | FireSmockDetect `expB_y11s_deployed`（上游生产默认） | Early Fire and Smoke（YOLOv8n 微调） |
| **权重文件** | `models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt` | `models/fire_smoke/firesmockdetect/expB_y11s_deployed.pt` | `models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt` |
| **文件大小** | 40,578,796 B | 19,241,370 B | 6,262,051 B |
| **SHA-256** | `1b6cb82295767ae9d0d5c4fb94dffcaf17bbbafc0e3af001d3fe517015902298` | `cc56224c84092e1e374cee1b3ce317f17f7f80e7945bc6c9a3a24a083321ff0e` | `ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16` |
| **官方来源** | [HF: TianchenGuan/FireSmockDetect-models](https://huggingface.co/TianchenGuan/FireSmockDetect-models) @ `a914a2a` | 同上 | [HF: mfranzon/fire-smoke-yolov8](https://huggingface.co/mfranzon/fire-smoke-yolov8) @ `f1c6426` |
| **上游代码** | [GitHub: TianchenGuan/FireSmockDetect](https://github.com/TianchenGuan/FireSmockDetect) | 同上 | 无（模型卡只描述设计，未公开实现） |
| **输入尺寸** | 1280（训练记录与模型卡一致） | 1280（训练记录与模型卡一致） | 640（训练记录与模型卡一致） |
| **检测类别** | `smoke`（单类） | `smoke`（单类） | `fire`, `smoke`（两类） |
| **参数量** | 20,053,779 | 9,428,179 | 3,011,238 |
| **许可证** | AGPL-3.0 | AGPL-3.0 | AGPL-3.0 |
| **加载结果** | 成功（0.095 s） | 成功（0.047 s） | 成功（0.085 s，需 `dill==0.3.8`） |
| **推理耗时** | 44.6 ms/帧（含预热，见注 1） | 14.6 ms/帧 | 7.3 ms/帧 |
| **薄烟召回率** | 待测 | 待测 | 待测 |
| **误报率** | 待测 | 待测 | 待测 |
| **首次有效报警时间** | 待测 | 待测 | 待测 |
| **已知问题** | 单类，无法区分白烟与水汽/反光；显存与延迟最高 | 与 11m 同配方，仅骨干更小 | 640 输入对远处小目标天然不利；无公开的连续帧实现 |

注 1：耗时为 `cuda:0` 上 76 张图逐帧平均，**包含首次推理的预热开销**，因此 11m 的均值被显著抬高（其 p95 为 29.5 ms）。稳定态延迟应看 p95，或先跑若干张热身帧再计时。完整数据见 `reports/comparison/rf100_smoke_2026-10-29_summary.csv`。

## 2. 训练数据与评估条件（来自上游，非本项目实测）

### FireSmockDetect

- **训练数据**：pyro-sdis + D-Fire 的训练划分。FIgLib / HPWREN 只用于回放评测，**从未**参与训练（其许可为 CC BY-NC-ND）。
- **评估方法**：冻结的 pyro-sdis TEST，4,124 张图 / 917 张负样本；匹配负样本误报率下的召回，IoU ≥ 0.4。
- **作者报告**：

  | 模型 | R@5%FP | R@10%FP | R@20%FP |
  |---|---|---|---|
  | v5bA_y11m_best | 0.287 | **0.388** | 0.553 |
  | expB_y11s_deployed | 0.258 | 0.354 | 0.501 |

- **告警阈值**：上游的告警策略为 0.05 / 0.30 / 0.15（三级），并注明在逐机位校准前保持不变。sdis 验证集上 5/10/20% 误报对应的置信度约为 0.729 / 0.673 / 0.534（11m）。
- **场景**：固定瞭望摄像头与无人机航拍，任务是**野火早期烟羽**。作者明确说明阈值**不可**在固定机位与航拍之间迁移。
- **作者自己声明的局限**：没有合格的起烟瞬间素材，因此**不提出任何早期预警时间指标**；薄烟低误报下的召回是瓶颈；定位是辅助工具，不是认证报警系统。

### Early Fire and Smoke

- **训练数据**：D-Fire + Domestic Fire and Smoke。
- **作者报告**：Apple Silicon MPS 上约 70 FPS。**没有**报告召回率、误报率或任何定量检测指标。
- **场景**：家庭早期火灾与烟雾（烹饪、电气、香烟）。模型卡声称针对"电气"场景训练过，但**没有任何证据**表明它验证过电动车电池热失控。
- **连续帧机制**：模型卡描述了一套状态机（clear → smoke present → early warning（烟雾增长）→ fire confirmed），设计目的是对持续增长的烟雾告警而不是单帧噪声。但作者 GitHub 账号与其所属组织下的公开仓库中**均未找到该实现**，需要自行实现。
- **作者自己声明的局限**：视觉手段永远不是生命安全系统，这是研究工具，不能替代认证的烟感/火灾探测器。

## 3. 本地候选模型（不随仓库分发）

以下两个权重已下载并校验，但**没有**纳入版本管理，需要用 `download_models.py` 自行获取。

| 字段 | Pyronear Sensitive Detector | D-Fire YOLOv8n 微调 |
|---|---|---|
| **模型名称** | yolo11s_sensitive-detector v1.1.0 | fire-smoke-yolov8n |
| **权重文件** | `models/fire_smoke/pyronear_sensitive/best.pt` | `models/fire_smoke/dfire_yolov8n/best.pt` |
| **文件大小** | 19,225,626 B | 6,229,802 B |
| **SHA-256** | `a9bfa11c559e4b221c43bb24a0ec0857bc75ec653283ba6146aef44c088da51a` | `b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f` |
| **官方来源** | [HF: pyronear/yolo11s_sensitive-detector](https://huggingface.co/pyronear/yolo11s_sensitive-detector) @ `3df6951` | [HF: rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) @ `13017fe` |
| **输入尺寸** | 1024（训练记录与模型卡一致） | 640（训练记录与模型卡一致） |
| **检测类别** | `item`（单类，类名无语义，按上游任务语义即烟雾） | `smoke`, `fire` |
| **参数量** | 9,428,179 | 3,011,238 |
| **许可证** | AGPL-3.0 | AGPL-3.0 |
| **加载结果** | 成功 | 成功 |
| **推理耗时** | 9.8 ms/帧（`iou=0.01`，见注 2） | 5.7 ms/帧 |
| **薄烟召回率** | 待测 | 待测 |
| **误报率** | 待测 | 待测 |
| **首次有效报警时间** | 待测 | 待测 |
| **训练数据** | `pyronear/pyro-dataset` v4.0.0-corrected（野火，排除背景图） | D-Fire（train 14,122 / val 3,099 / test 4,306） |
| **作者报告指标** | 模型卡未给出定量检测指标 | test split mAP50 0.754、mAP50-95 0.430、P 0.766、R 0.688 |
| **已知问题** | 上游推荐 `conf=0.2, iou=0.01`，NMS 近乎最大抑制，一个场景只留一个烟羽；换阈值结果差异极大 | 第三方在 D-Fire 上自训，**不是** D-Fire 数据集作者的原始权重；只验过 D-Fire 划分 |

注 2：`iou=0.01` 是上游模型卡明确给出的推荐值。这不是笔误——野火瞭望场景下一张图通常只有一个烟羽，用极低 IoU 抑制重叠框。但在车棚场景（一辆车一个烟源、可能同时多处冒烟）这个设置会严重漏检，实测前必须重新标定。

## 4. 本项目的实测记录

### 2026-10-29 · RF100 `smoke` 测试子集（冒烟测试，非正式评测）

- **素材**：`samples/rf100_smoke/images`，76 张，单类 `smoke` 标注 74 份。来源 [RF100 smoke](https://universe.roboflow.com/roboflow-100/smoke-uvylj/dataset/2)，许可 CC BY 4.0。
- **参数**：`conf=0.10`，各自原生 imgsz 与上游推荐 `iou`，`cuda:0`。
- **结果**：

  | 模型 | 有检出帧 | 占比 | 平均耗时 | p95 | iou |
  |---|---|---|---|---|---|
  | firesmockdetect_y11m | 73 / 76 | 96.1% | 45.6 ms | 32.5 ms | 0.7 |
  | firesmockdetect_y11s | 71 / 76 | 93.4% | 13.0 ms | 13.5 ms | 0.7 |
  | early_fire_smoke | 71 / 76 | 93.4% | 6.5 ms | 7.1 ms | 0.7 |
  | pyronear_sensitive | 16 / 76 | 21.1% | 9.8 ms | 10.9 ms | 0.01 |
  | dfire_yolov8n | 75 / 76 | 98.7% | 5.7 ms | 6.8 ms | 0.7 |

- **这个结果能说明什么**：五个权重都是真实可用的，推理链路通畅，延迟量级符合预期。
- **这个结果不能说明什么**：**不是**召回率。RF100 的烟雾多为明显烟柱，不是车棚里的早期薄烟；图片没有逐帧人工复核。`pyronear_sensitive` 的 21.1% 是 `iou=0.01` 造成的，不代表它更差；`fire_smoke_yolov8` 在首张图上以 0.10 阈值输出 0 个框，说明其置信度分布明显更保守。任何"哪个模型更好"的结论都必须等车棚实测。

## 5. 证据缺口（必须补的）

1. **没有电动车电池早期冒烟的真实视频**。当前只有一段 18 秒的第三方新闻素材（本地 `video/`，版权不在本项目，不入库），且是已经明显冒烟、4 秒后爆燃的后期阶段，**不能**用来衡量"早期"预警。
2. **没有车棚场景的负样本**。水汽、地面反光、车灯、尾气、吸烟者这些干扰源一张都没有。
3. **没有固定测试集划分**，因此任何跨模型的召回率比较目前都不成立。
4. **没有时间维度标注**（起烟帧、首次可见帧、明火帧），因此"首次有效报警时间"无法计算。

补数据的顺序建议：先采一批车棚全景与干扰源负样本 → 再拍可控烟雾（烟雾机 + alpha 合成）→ 最后才是真实起烟片段。

## 6. 公开数据集结果不能直接代表车棚效果

上游的 0.388 / 0.354 是在**野火**数据、**固定瞭望机位**、**匹配误报率**、**IoU≥0.4** 条件下测得的。电动车车棚在以下方面完全不同：目标尺度更小（远处车位）、遮挡更严重、干扰源更杂（水汽/反光）、背景更结构化（成排车辆）。把这两个数字写进申报书当作车棚场景的性能，属于错误引用。
