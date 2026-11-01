# 本地电动车起烟定向微调

当前使用规范配对目录 `datasets/ebike_focused_20261031_v2/`。首轮 experiment_001 因新图标签路径不符合 YOLO 自动查找规则，已标记无效，不作为候选。修复后已用实际 YOLODataset 验证全部 1,374 个采样条目的标签数组，包括新图的 404 个烟雾框、8 个火焰框和 56 个用户确认背景条目；experiment_002 重新从原始 dfire_ludan 开始。

2026-10-31 用户明确要求在本机从 `dfire_ludan` 继续训练。本轮不使用 AutoDL，不直接加载三万多张混合训练清单，不覆盖现用权重。训练输出独立保存在 `runs/dfire_ebike_focused/experiment_002/`，日志为 `outputs/local_training/focused_experiment_002.log`；完成状态以实际 `comparison.json` 为准。

## 数据与用户确认

- 762 张不同训练图片；重复采样后每轮 1,374 个样本。
- 人工审核组共 140 张：昨日 40 张，加本次 100 张。本次导出包含 86 张框标注，用户确认另 14 张烟雾肉眼不可辨认，明确要求作为背景一起训练，已生成空标签。空标签不代表没有真实火灾风险。
- 9 个越界框按用户要求裁到图片边界，仅保留图内部分；原始导出 ZIP 不变，裁剪前后坐标见 `datasets/ebike_early_smoke_review_20261031/annotations/import_004/clipping.json`。
- 人工审核组每图采样 4 次，共 560 次，约占 40.76%；jjs 的 32 张干扰帧每图采样 7 次，共 224 次，约占 16.30%。其他 590 张图片各采样一次，约占 42.94%。
- 其余训练图包括旧专项微调的 200 张回放、99 张已审阅网络背景、52 张筛选过的实际场景、239 张合成监控场景。各来源许可分别保留。
- 参考验证 209 张、参考测试 150 张，均沿用先前留出清单；图像 SHA-256 不跨集合重复。

用户明确确认新新闻素材已有训练使用授权。本项目记录该用户确认，不推定公开再分发许可。人工标注导出包 SHA-256 为 `f9b3d4d0fc631e21a10e84f633fa1af2ac3c8dd933acfea8b9fe626977daca28`，类别 `0=smoke,1=fire`，新图共 101 个烟雾框、2 个火焰框。图片与数据集不随 Git 分发。

所有人工审核图和 jjs 抽帧都参与训练，不能将其上的改善说成未见场景泛化能力；参考留出集也已用于旧模型选择。新场景尚无独立留出事故，不能用这些数字宣称车棚召回率或预警能力。

## 入口与参数

基础权重为 `runs/dfire_ludan/experiment_002/train/weights/best.pt`，SHA-256 `22fd0c9f6d2498e3ea3d5c3f456f1f13393d66e068a4d1d6c698b82777f2f810`。输入 960、batch=8、最多 30 轮、AdamW、初始学习率 0.0001、冻结前 10 层、patience=8；其余实际参数保存在本次 `train/args.yaml`。执行前先做原模型参考测试，训练后测试新 best 权重。

```powershell
# 只检查配置和基础权重，不开训
.\.venv\Scripts\python.exe scripts/train_expanded_smoke.py --dataset focused --base-model dfire_ludan --dry-run

# 正式训练；experiment_002 已存在时不要重复执行，以免与当前任务冲突
.\.venv\Scripts\python.exe scripts/train_expanded_smoke.py --dataset focused --base-model dfire_ludan --epochs 30 --imgsz 960 --batch 8 --lr0 0.0001 --freeze 10
```

新实验应通过 `--out runs/dfire_ebike_focused/experiment_003` 等独立名称创建目录。脚本拒绝覆盖已有实验。新权重不自动覆盖旧基础权重或推理入口；应先查看参考测试和相同阈值的干扰回放，再决定是否切换。

## 训练后的固定阈值检查

检查入口为 `scripts/evaluate_focused_smoke.py`。默认比较原始 `dfire_ludan` 和本轮验证集最佳 `best.pt`，保持置信度 0.15 / 0.25、NMS IoU=0.7、框匹配 IoU≥0.5；分别记录昨日人工、本次人工、jjs 抽帧和参考网络背景，另外逐帧检查整个 jjs 视频。

```powershell
# 分别运行，报告保存到本轮实验目录。
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_focused_smoke.py --imgsz 960
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_focused_smoke.py --imgsz 1280
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_focused_smoke.py --imgsz 1536
```

人工标注结果采用烟雾框正确匹配数与多余检测框数，不用“有框图片占比”代替召回率。jjs 全视频仅报告输出帧数；视频没有逐帧事故真值，不能直接称为实测误报率。输入尺寸的选择以同一阈值下的实际结果为准。

## 实际结果与保留权重

已完成 30 轮；按参考验证集 mAP50–95 选择第 **22** 轮（轮次从 1 开始），验证值 0.25130。最佳权重为 `runs/dfire_ebike_focused/experiment_002/train/weights/best.pt`，大小 6,259,818 字节，SHA-256 `fb335751a28ebcc7f3d35651d4eb8ef4a9805b2e62db32b14b690a4010f7f919`，登记名 `dfire_ludan_focused`。原始 `dfire_ludan` 哈希不变；`last.pt` 和周期快照暂留供核对。第 30 轮虽然对训练图拟合稍多，但参考验证较差，0.25 阈值下训练图多余烟雾框也更多，因此保留第 22 轮作为本轮最佳。

在相同输入 960、置信度 0.15、框匹配 IoU≥0.5 下：本次人工组 101 个烟雾框，正确匹配从原模型 2 个提高到 93 个，昨日组 41 个从 13 个提高到 28 个；本次组多余烟雾框从 45 个减到 34 个。14 张人工确认背景的出框图片从 8 张减到 1 张。jjs 全视频 565 帧任意类别出框从 10 帧减到 0 帧。上述人工与 jjs 素材均参与训练，只能说明拟合与回放变化。

同时存在明确退化：45 张参考网络背景的出框图片从 8 张增加到 13 张；昨日组的多余烟雾框从 10 个增加到 37 个。150 张旧参考测试集上，整体 mAP50–95 从 0.45023 降至 0.38201，烟雾 AP50 从 0.71564 降至 0.58243。因此本轮登记为定向候选，编辑器默认仍保留原始 `dfire_ludan`，不宣称全面提高准确率或降低所有误报。

同为 0.15 阈值，本轮最佳权重在 960 / 1280 / 1536 下，新人工组烟雾正确匹配为 93 / 81 / 58（共 101 个），jjs 视频出框为 0 / 17 / 72 帧，参考网络背景出框图片为 13 / 20 / 21（共 45 张）。因此本轮建议输入 **960**；并非分辨率越高越好。0.25 阈值下 960 输入的新人工组匹配 92 个、多余烟雾框 13 个，jjs 视频 0 帧出框；两档阈值的完整分组数据在 `target_regression_*.json`。

调用方式：在 `scripts/run_models.py` 启用 `"dfire_ludan_focused"`，或在 `scripts/run_video_baseline.py` 启用 `"dfire_ludan_focused": 960`。这两处均按原有风格添加了中文注释；同时保留 `dfire_ludan` 可直接对比，同一阈值比较。命令行也可执行：

```powershell
.\.venv\Scripts\python.exe scripts/compare_models.py --source video/jjs.mp4 --models dfire_ludan dfire_ludan_focused --device 0 --conf 0.15 --stride 1
```

完整选择记录见本次实验 `selection.json`；训练和参考测试见 `comparison.json`；960、1280、1536 的固定阈值回放分别见 `target_regression_960.json`、`target_regression_1280.json`、`target_regression_1536.json`；末轮训练图核对见 `target_regression_last_960.json`。无新场景独立标注测试，不作车棚泛化能力结论。

最终验证：9 项标注/采样针对性测试通过；新候选哈希、GPU 加载及单图推理通过；原比较入口双模型调用通过；三档尺寸全量回放完成。

## GitHub 快照与用户新视频反馈

用户明确要求将当前代码和微调模型上传 GitHub。随快照保存三份已登记的最佳权重，原路径、文件名和字节保持不变；只跟踪指定 `best.pt`，不提交其余训练产物或数据。视频入口当前由用户选中 `dfire_ludan_focused`，960 输入、阈值 0.15；图片快速入口仍选原始 `dfire_ludan`，保留各入口当前选择。

用户在新视频上的观察为“误报率有点高但识别精准度非常好”。这是用户实际观看反馈，相关视频没有逐帧真值，尚未量化其精确率和误报率。此前参考测试的退化与训练图拟合改善仍保留，不能用该反馈替代定量结果。公开选择记录另存 `reports/training/ebike_focused_20261031.json`，不含图片或视频。
