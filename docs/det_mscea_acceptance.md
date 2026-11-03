# Det-MSCEA 阶段验收记录

用户日期：2026-11-03。初始审计提交 `79bab7c`。本地实施在独立 worktree 的 `feature/det-mscea-full` 上，原 `main` 的未提交修改未动。PR #2 保持 Draft，不合并。

## 当前结论

实现已有增量修复；合成优化、CPU Mock Detector 数据生成和训练入口门槛通过测试；真实 YOLO/CUDA **适配器**通过加载和前向检查。真实 Mask 增强、人工样本验收和 B0–B8 正式实验尚未进行。MSCEA 全文仍缺失，因此算法原文一致性验收没有完成。

## 第一阶段：证据审计

见 [算法审计](det_mscea_audit.md)，包含与原文一致、检测适配、实现偏差、明确错误、证据不足五类。DAS 与 FROFI 分别核对作者稿，未用一篇代替另一篇。MSCEA 全文没有取到；旧文档的 Canny / 分类目标等说法已降为待核实，未据此宣称复现。

## 第二阶段：问题与修复结果

1. 烟雾参考库：所有行在模型加载前核对目标为 smoke=0，拒绝混入火焰；检测模型也要求 class 0 为 smoke。合成测试覆盖非目标 fire 行污染。
2. 尺度分桶：按 letterbox 缩放后的实例尺度选层，同分辨率比例变化不改变层选择；源、候选沿用同一层与同库。
3. Hook：原提交在本机真实前向失败，错误为 `cannot pickle '_thread.lock' object`，原因是当前 Ultralytics 的预测器初始化会深拷贝包含绑定 Hook 的模型。现在先初始化预测器，再挂实际推理模型的 Detect pre-hook 和输入形状 Hook。
4. 坐标：使用实际输入 H/W、整数 resize 和 padding；覆盖非整 stride 请求与奇数 padding。真实 650 请求实际进入 672 网格。
5. 实例匹配：最高 IoU 优先，置信度仅打破同 IoU 平局，避免用邻团高分替代目标。匹配阈值仍为 0.10，没有为测试放宽。
6. 背景误报：ROI 外烟雾框与 ROI 外 baseline 做一对一 IoU>=0.50 匹配，旧检测不算新增，一个旧框不能豁免多个新框。C2 仍是中心在 ROI 外且新烟雾分数不超过 0.10。
7. Mask/标签：仍使用同一逆映射、最近邻 mask、tight bbox 和 YOLO 转换；增加 mask/原标签 IoU>=0.50 校验，ROI 碰到其他标注实例时跳过，避免移动其他对象像素却保留旧框。
8. ROI：外部像素逐字节保持；移动目标点限在 ROI 内，重合固定控制点由固定锚点接管，目标控制点塌缩则判无效。
9. 可行档案：最终图像只进入满足全部约束（违反度严格为零）的 top-k；搜索用 FROFI archive 允许不可行项，两者分离。移除种群 payload 的完整图像、mask 和特征引用，只由 top-k 保留图像。
10. 扩张：保持邻域盒子之并，不能把包围盒空隙当合法空间；增加解码前的绝对 ROI 位移界限，按预算配置盒子容量，避免默认 256 上限截断实例搜索。此工程边界未宣称来自 MSCEA 原式。
11. 预算：不足一对完整相关代的配置明确拒绝；每次正常或无效适应度评价都计数；末代截断，FROFI 单坐标变异也计数。归档替换不占评价次数，预算恰好用尽也执行。
12. 资源：CPU 测试断言种群不保留完整 warp；真实重复前向见下。尚缺真实多候选 MLS 优化资源观测，不能声称已排除全部泄漏。

检测适配输出增加逐次评估轨迹、阶段状态、生成耗时和文件哈希。输出一律 `human_review=pending`；训练构建器要求人工批准、审核人和检查说明，并核对源/生成文件、参考库哈希与训练划分。参考库中的任何验证/测试图也不能进入正式训练。相同内容但不同路径的复制图片仍需数据集去重/分组划分验收，路径门槛不能代替它。

## 第三阶段：自动化验证

- 原始两模块：12 项测试通过。
- 更新后三个相关模块：32 项测试（MLS、优化器、CPU 适配器/生成集成）。
- 全仓库：61 项测试，60 项成功、1 项跳过。跳过的是既有 `test_lwst_backend`，需要未提供的上游 LwST 权重和源码资源。
- CPU 集成生成的是明确标识的合成测试数据与 Mock 权重占位符，不是真实增强成果；临时测试目录不进入数据集。
- GitHub Actions 只安装 NumPy、OpenCV headless、PyYAML，执行这 32 项测试；不依赖 GPU、Torch 或真实模型。CI 结果以 PR 当前提交的 Actions 记录为准，不代替算法与效果验收。

测试命令：

```powershell
<project-python> -m unittest tests.test_mls_augmentation tests.test_mscea_optimizer tests.test_mscea_adapter -v
<project-python> -m unittest discover -s tests -v
```

## 第四阶段：真实环境和适配器

公开可审查原始数值日志位于：

- `reports/det_mscea/original_hook.json`：原实现真实失败证据。
- `reports/det_mscea/real_hook_960.json`：30 次 GPU 前向、特征统计、内存与输入清单检查。
- `reports/det_mscea/real_hook_650.json`：非整 stride 请求（实际 672）检查。

本机环境：Windows、Python 3.13.14、Torch 2.8.0+cu129、Ultralytics 8.4.192、NumPy 2.5.4、OpenCV 5.0.0、CUDA 12.9、RTX 5070 Laptop GPU。此次核对以实际运行输出优先于旧环境笔记。

真实权重为 `dfire_ludan`，SHA-256 `22fd0c9f6d2498e3ea3d5c3f456f1f13393d66e068a4d1d6c698b82777f2f810`。`verify_models.py` 完成该权重哈希、加载和真实图片推理；`compare_models.py` 在同图成功生成 CSV。该图在默认 conf=0.25 没有框，不能解释成召回率；适配器检查用 conf=0.001，主要验证特征捕获。没有宣称这张图是经过人工核查的本项目实例 Mask 样本。

960 输入三层形状为 `[1,64,120,120]`、`[1,128,60,60]`、`[1,256,30,30]`，均有限。30 次前向后 CUDA allocated 每次为 12,267,008 bytes；RSS 从 1,461,657,600 到 1,462,980,608 bytes，有约 1.26 MiB 增长。该短测试支持“未观察到 GPU 分配持续增长”，不支持“无内存泄漏”。

### 阻塞材料

扫描原项目 `datasets/`、`samples/`、`outputs/`、`data/` 的 100 份 manifest 文件，未找到具备 `image/mask/label/target_label_index` 的实例 Mask 清单。现有人工审核图片/YOLO 检测框及 CVAT 检测框导出不等于经过人工验收的烟雾分割掩膜。

同时查阅可读 Codex 记忆、相关历史会话、`.workbuddy-ai` 项目笔记、当前 B3 文档与研究 PDF；没有恢复出 Mask Manifest 和 B0–B8 完整定义。记忆仅作定位线索，旧笔记不当作当前路径存在的证据。本仓库当前仅明确定义 B3 随机局部 MLS。

因此未进行随机 3 实例、population=10/max_evals=30/top_k=1 的真实增强试运行；没有伪造 Mask、没有把 bbox 当 Mask。

## 第五阶段：人工样本检查

状态：未开始；真实增强图片 0、人工通过样本 0。后续逐样本必须检查原图/原 Mask/原框与增强图/Mask/框，同时记录 mask 与 bbox 面积比、底边位移、源与候选分数、ROI 外新增误报分数、特征多样性，并排除结构不自然、断裂、边界伪影、标签漂移和新误报。

审核结论写入 manifest 的 `human_review`；只有真实审核人可填 approved，不把代理视觉观察或约束通过自动当作人工批准。真实候选不存在时没有可填的审核记录。

## 第六阶段：正式对照

状态：未开始，训练运行 0。`reports/det_mscea/experiment_status.json` 保留 B0–B8 全部 ID，缺失定义不猜测；所有指标为 null。计划配对种子 20261103/20261104/20261105 仅为后续建议，未执行。

待小规模生成与人工验收通过后，恢复/确定各组定义，冻结同一基础 checkpoint、训练/验证/独立测试划分、输入尺寸、轮数和超参。训练前登记原始清单及哈希、事故/视频分组与复制图片去重、tiny/low-confidence 子集定义和阈值，测试集不能参与参考库、参数选择或结果挑选。

正式结果须同时报告 mAP50、mAP50–95、smoke Recall、tiny/低置信度子集 Recall、误报数及每图误报、PR 曲线、增强耗时与实际生成数量，并做同种子配对比较。没有改善或 Recall/误报存在代价时照实报告。
