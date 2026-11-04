# Det-MSCEA / B3：局部 MLS 随机增强

本目录实现的是 **B3 对照组**，不是完整 MSCEA。它先回答一个基础问题：在不使用差分进化的情况下，局部 MLS 形变本身能否为 YOLO 烟雾检测带来收益。

## 当前实现范围

- 从单个烟雾实例的二值 PNG 掩膜提取轮廓控制点。
- 在局部 ROI 中使用 affine MLS；ROI 边界布置固定锚点。
- 在原控制点的局部 alpha 邻域内随机采样目标位置，默认 \`alpha=0.07\`。
- 对烟雾实例底部的控制点降低位移幅度。
- 对图像和掩膜使用同一逆向 MLS 映射；掩膜使用最近邻插值。
- 只在原/新烟雾掩膜附近做羽化合成，ROI 之外及远离烟雾的背景保持不变。
- 由变形后的掩膜重新计算 YOLO 框，原图其他检测框保持不变。
- 对面积变化和底边位移做硬约束；不满足条件的样本重新采样。

## 重要前置条件：必须有实例掩膜

当前公开数据主要是 YOLO 矩形框标签，没有烟雾多边形/分割掩膜。**不能把矩形框直接当烟雾掩膜**，否则 MLS 会把大量背景当成烟雾一起变形，实验结论不可信。

每个增强实例需要一张与原图同尺寸的单通道 PNG：
- 黑色（0）：背景
- 白色（任意非零值）：当前这一团烟雾
- 一张 mask 只对应一个 smoke 实例；不要把同图多团烟雾合成一张 mask
- 掩膜边缘允许粗略，但应覆盖可见烟雾主体；不确定的区域不要凭空扩张

可用 CVAT 多边形标注或 SAM 辅助生成，再人工检查。第一轮不必给全部图片做掩膜，建议先选 20–50 个有代表性的烟雾实例做算子可视化验证。当前脚本要求 PNG 掩膜已存在，不会把 bbox 自动伪造成 mask。

## 从 CVAT YOLO Segmentation 导出生成掩膜清单

如果 CVAT 导出的是 Ultralytics YOLO Segmentation 多边形格式，可先解压导出包，并分别准备：
- 图片目录，例如 \`obj_train_data/\`
- 分割多边形标签目录，例如 \`labels/\`
- 与图片对应的 YOLO 检测框标签目录（现有数据集标签）

运行转换器：

\`\`\`powershell
.\\.venv\\Scripts\\python.exe scripts/import_yolo_segmentation_masks.py --images datasets/cvat_seg/images --seg-labels datasets/cvat_seg/labels --det-labels datasets/ebike_focused_20261031_v2/labels/train --out datasets/mscea_b3/mask_import --smoke-class-id 0 --fire-class-id 1 --min-iou 0.5
\`\`\`

转换器会用多边形外接框与现有检测框做 IoU 匹配，生成每个烟雾实例独立的 PNG 掩膜和 \`manifest.jsonl\`。如果 CVAT 导出类别顺序不是 \`0=smoke,1=fire\`，必须显式调整类别 ID；不要直接套用默认值。若检测框标签与分割标签不是同一批图像/同一标注版本，匹配会失败或错配，应先统一数据。

## 生成 B3 样本

\`\`\`powershell
.\\.venv\\Scripts\\python.exe scripts/generate_mls_dataset.py --manifest datasets/mscea_b3/mask_import/manifest.jsonl --out outputs/mscea_b3/generated --count 2 --seed 20261103 --attempts 20 --roi-scale 2.0 --alpha 0.07
\`\`\`

输出目录独立保存：
- \`images/train/\`：增强图片
- \`labels/train/\`：更新后的 YOLO 检测标签
- \`masks/train/\`：增强后的实例掩膜
- \`manifest.jsonl\`：源文件、SHA-256、参数和几何审计信息
- \`summary.json\`：生成数量及失败原因

脚本不会修改原始数据、原始标签或模型权重。

## 构建 B3 训练配置

\`\`\`powershell
.\\.venv\\Scripts\\python.exe scripts/prepare_mls_b3.py --base-data datasets/ebike_focused_20261031_v2/merged/data_balanced.yaml --generated-root outputs/mscea_b3/generated --out outputs/mscea_b3/prepared --repeat 1
\`\`\`

这会创建单独的 \`data_b3.yaml\` 和 \`train_b3.txt\`：在原训练列表后追加增强图片，验证集和测试集沿用基线配置，不把增强图混入留出集。

## 训练 B3

先人工抽查 \`outputs/mscea_b3/generated/\` 中的图片、掩膜和框。确认形变自然、标签正确后再训练：

\`\`\`powershell
.\\.venv\\Scripts\\python.exe scripts/finetune_dfire_small_smoke.py --data outputs/mscea_b3/prepared/data_b3.yaml --out runs/dfire_mscea_b3/experiment_001 --base-model dfire_ludan --epochs 30 --imgsz 960 --batch 8 --lr0 0.0001 --freeze 10
\`\`\`

这会从 \`dfire_ludan\` 开始，输出到新的实验目录，不覆盖原权重。为公平比较，B2 和 B3 应使用相同训练/验证/测试划分、初始化权重、训练轮数策略和随机种子；后续再做至少 3 个种子的重复实验。

## 单元测试

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m unittest tests.test_mls_augmentation -v
\`\`\`

## 目前不能声称的内容

- 还没有实现 MSCEA 的 DE、DASA-DE 约束处理、可行解邻域扩张和多样性目标。
- B3 不是 MSCEA 的完整复现，只是用于分离“局部 MLS 算子收益”和“进化搜索收益”的必要消融。
- 当前还没有在真实烟雾掩膜上运行，也没有训练结果；单元测试只证明几何变换、标签更新和数据配置链路可运行。
- B3 有收益后，再实现完整 Det-MSCEA；B3 无收益时，应先检查掩膜质量、参数和场景覆盖，不能直接认定论文方法无效。
