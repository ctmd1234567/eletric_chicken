# 电动车人工审核训练图与旧参考测试集（交接整理包）

2026-11-01 整理；目录名 real_ebike 按交接约定保留，不表示包内全部图片均为电动车场景。

- images/train_manual 与 labels/train_manual：140 张人工审核训练图及一一配对的 YOLO 标签，昨日 40 张 + 本次 100 张。含 14 张用户确认无肉眼可辨认烟雾的背景帧，空标签是有意保留的。这 140 张已参加定向训练，每张原图仅打包一次；不恢复训练时的四倍重复采样。
- images/test_reference 与 labels/test_reference：150 张 FASDD 旧参考测试图及标签，不是 150 张独立真实电动车测试图。该参考集已有历史评估和模型选择使用，不能作为新场景独立测试。
- 类别固定为 0=smoke、1=fire。原图和现行标签按字节复制，无重新编码或本次改框；此前按用户要求裁剪的标签仍保留现状。
- manifest.jsonl 保留仓库相对来源、原始 SHA-256、标签 SHA-256、来源组、历史划分及采样记录。audit.json 为本次实际校验结果。精确图像去重不证明事件独立，也不能排除基础模型预训练重叠。

## 接入说明

reference_test.yaml 仅声明 test，未编造 val。使用时将 path 设为解压后 real_ebike 的绝对路径，并在现有评估流程中明确选择 test。140 张训练图只用于训练回放/拟合检查，不应通过随机拆帧包装成独立验证集。本包不覆盖已有训练配置，不启动训练。

## 论文与答辩表述

可写：本地整理了 140 张人工审核电动车相关帧用于定向训练，其中包括 14 张肉眼不可辨认目标的背景帧；另使用 150 张 FASDD 历史参考测试图检查性能变化。训练图上的改善属于拟合表现，不等于未见事故泛化能力；暂无新场景独立标注测试，不据此声称车棚薄烟召回率或预警时间。

不能将本包描述为 290 张真实电动车烟雾样本，不能把 140 张帧写成 140 起事故，不能把有框帧占比写成召回率。“M4 唯一样本”是外部交接描述，本仓库未核实 M4 范围。

## 来源和使用边界

SOURCE_NOTICE.md 保留现有来源与许可说明，permission_record.json 保留此前用户确认的训练授权。该授权不推定公开再分发权；FASDD 使用条件见来源声明。后续用户于 2026-11-01 明确确认已取得这 140 张图片公开再分发许可，并要求上传 GitHub，记录见 public_permission_20261101.json；本次据该确认发布本交接包，不扩展至原始视频或其他数据。

EXPERIMENT_NOTES.md 与 model_selection_record.json 为已有实验记录的原样副本，包含历史状态；当前实现和现行入口应以源仓库为准。论文引用需结合实验条件及限制，不可直接把内部观察写成通用能力结论。

## 本次发布与原始记录的关系

SOURCE_NOTICE.md 和 permission_record.json 是此前记录的原样快照，其中“仅本地”“不公开分发”描述当时的范围；本次仅针对本交接包的发布范围由 public_permission_20261101.json 补充，不改变其他文件或数据集的使用条件。

150 张参考图来自 FASDD_CV，经现有流程转换为 0=smoke、1=fire；保留该衍生版本的 CC BY-SA 4.0 条件。原始数据集：An open flame and smoke detection dataset for deep learning in remote sensing based fire detection，Science Data Bank，https://doi.org/10.57760/sciencedb.j00104.00103 。许可：https://creativecommons.org/licenses/by-sa/4.0/ 。取得的镜像：seawsurf/fire_smoke_dataset_fasdd_cv，revision 896dec15c93396ed4990ad502a187be9271e38e7。来源成员及 SHA-256 见 manifest.jsonl。人工审核组保留原素材权利，不因与 FASDD 同包而统一重新许可。

## 762 张训练图中其余 622 张

- 200 张旧回放：160 张有烟雾框、40 张背景；来源为 FASDD 等旧参考素材，未确认是电动车事故图。
- 131 张干扰背景：jjs 32 张、网络背景 99 张，均为空标签。
- 52 张 Zenodo 真实相近场景：46 张有烟雾框，包含建筑、仓库、室内火烟及停放摩托车附近受控燃烧；不等于已确认的电池热失控实拍。
- 239 张 Simuletic 合成监控图：141 张有烟雾框，其余为火焰标注图；为合成场景，不能算真实电动车事故。

上述组别按现行训练 manifest 的实际标签统计；烟雾和火焰可在同图共存。除 140 张人工审核组外，尚未确认有其他“真实电动车事故烟雾”的明确标注组。这不等于声称其余每一张图片都经过本次视觉排查。其余 622 张未收入本交接目录。

本次按交接约定直接提交到 GitHub 仓库的 data/yolo/real_ebike/，不使用 Release。目录内 .gitattributes 禁止换行自动转换，保证克隆后的图片和标签 SHA-256 与本地一致。
