# 本轮场景补充与 AutoDL 训练

后续状态：用户要求优先补充明确的电动车早期起烟图片后，另整理了 `datasets/ebike_early_smoke_review_20261031/` 的 100 张待标注帧。它们尚未标注，素材训练许可也未核清，未进入下述包。对旧集的 48 张抽查已见明显场景混杂；因此下述混合包虽已生成并通过全包摘要检查，**当前建议保留备份，先完成目标场景标注和来源核对，再生成最终训练包**，不因包可运行就立即开训。本机未启动正式训练。

本轮从现有 `dfire_ludan`（experiment_002）继续训练，不使用已退化的 `dfire_expanded`。本机只做数据准备、检查和打包，不启动正式训练。仅供非商业研究：新增 Simuletic 样本为 CC BY-NC 4.0，其他来源保留各自条件，详见 NOTICE.md。

## 数据和重点样本

- `datasets/urban_smoke_20261031/merged/train_balanced.txt`：本轮实际训练清单，35,823 张不同图片；提高采样频次后每轮 42,020 个样本。
- 用户手工标注 40 张：原标签不改，每张在训练清单出现 105 次，约占每轮 10%。另外 16 张缺标签帧仍排除。同一事件没有独立本地验证集，不能用训练集拟合代表泛化。
- 新增 52 张实际场景图片：Zenodo Indoor Fire Smoke 原训练集的核对样本，含建筑、仓库、室内火烟和停放摩托车附近的受控燃烧烟雾。不是已确认的电池热失控实拍。
- 新增 239 张合成监控图片：Simuletic 城市停车区、自行车架、建筑外围等；原 240 张中缺标签的 1 张排除。按 48 个场景代表图抽查，全部图片/标签做结构和完整性检查，不能宣称所有标注完美。
- 新增 291 张各采样 8 次，总计约占每轮 5.54%；原训练数据和旧专项训练的有效回放样本保留。重复采样不复制文件，也不增加不同图片数量。
- 验证 7,781 张、测试 8,956 张，沿用原留出清单。新场景全部只加入训练，避免同一采样视频/合成场景被拆到不同集合。合并后图像 SHA-256 不跨集合重复；尚不能证明全部事件独立，也不能消除基础模型先前看过部分留出素材的风险。

原始下载和许可证快照在 `datasets/urban_smoke_20261031/raw/`、`research/`；衍生标签统一 `0=smoke,1=fire`。`previews/user_manual_*.jpg` 为用户标注叠框图，`previews/real_reviewed_*.jpg` 为新增实际场景叠框图，`previews/new_scenes_*.jpg` 为全部新场景叠框图；绿色烟雾框，红色火焰框。来源、原始成员名、摘要和采样计数见 `new_manifest.jsonl`、`review.json`、`merged/audit.json`。

## 上传和启动

上传 `outputs/cloud/autodl_urban_ludan_20261031.tar` 及同名 `.sha256` 到实例 `/root/autodl-tmp/`。解压到独立目录，避免与上一轮项目混用。在 AutoDL 终端执行：

```bash
cd /root/autodl-tmp
sha256sum -c autodl_urban_ludan_20261031.tar.sha256
mkdir -p urban_ludan_20261031
tar -xf autodl_urban_ludan_20261031.tar -C urban_ludan_20261031
cd urban_ludan_20261031/eletric_chicken
bash install_cloud.sh
```

安装脚本校验文件摘要、重建 Linux 数据路径、检查 CUDA，然后只显示训练配置。确认 `base_model=dfire_ludan`、训练清单 42,020 条、`imgsz=960` 后，由用户启动：

```bash
bash train_and_shutdown.sh
```

默认 30 轮、batch=8、学习率 0.0001、冻结前 10 层；这是为了保留已有特征的保守起点，不保证效果提高。训练入口会在训练前评估所选 `dfire_ludan`，训练后评估新权重；不会用原始 D-Fire 冒充本轮基线。结果独立保存到 `runs/dfire_urban/experiment_001/`，不会覆盖旧权重。训练及评测完成或报错退出后保留日志并关机；Ctrl+C / SIGTERM 跳过自动关机。

如需调整参数，命令追加 `--epochs 20 --batch 8 --lr0 0.0001 --freeze 10`。基础模型可用 `--base-model` 显式选择，默认固定为 `dfire_ludan`。不要仅因内部指标提高就替换推理默认；还需比较真实新视频的烟雾检出、干扰和漏检。

本地检查命令（不训练）：

```powershell
.\.venv\Scripts\python.exe scripts/train_expanded_smoke.py --dry-run
.\.venv\Scripts\python.exe scripts/prepare_autodl.py pack
```

上传包只包含需要的训练/验证/测试图片、标签、所选基础权重、训练代码、摘要及来源记录，不包含其他候选模型、第三方测试视频或完整下载缓存。文件和数据集均不进入 Git。
