# SAM 2.1 small 本地掩膜预览

后续状态（2026-11-16）：用户要求移除明显错误项，已从 `run_001`、`run_002` 的候选、掩膜、标签副本和预览中删除 `manual_box_02`、`old_case_008_0018`、`old_case_009_0018`，每轮保留 9 项；原训练来源未删除。下文 12 张运行数字是当时完整试跑的历史记录。之后新获取无新增框的原媒体抽帧候选 315 张，并按用户要求独立执行 SAM 自动分割，不经 YOLO 筛选，见 [原始素材与自动分割记录](mscea_raw_review.md)。

2026-11-16：按用户要求部署在当前 D 盘项目内，只预览 12 张，不训练、不批量扩充负样本。首批为新获取的北京充电车棚监控图 1 张、之前网上电动车/电池场景 8 张、既有人工检测框的待分割图片 3 张。同一事件的不同帧不是独立事件。

## 安装与资源

- 环境：`outputs/sam21_local/env/`；源码：`outputs/sam21_local/source/`。使用 Windows Python 3.13，独立环境通过 `project_runtime.pth` 读取当前项目 `.venv` 的 PyTorch 2.8.0+cu129 等依赖；没有升级或改写 `.venv`。移动项目后需重新生成这个路径文件。
- 本机 WSL 有 GPU，但 Python 3.14 环境没有 pip，因此本次采用已有 CUDA 可用的 Windows Python；没有为此重新安装一套 PyTorch。
- 单独安装 hydra-core 1.3.2、iopath 0.1.10、tqdm 4.67.1 及其依赖。上游 SAM 源码 revision `2b90b9f5ceec907a1c18123530e92e794ad901a4`，安装时 `SAM2_BUILD_CUDA=0`，未编译可选 CUDA 后处理；不是禁用 GPU 推理。
- 原始权重：`models/segmentation/sam2.1_hiera_small.pt`，184,416,285 字节，官方 Hugging Face revision `ee5bba1d82bb8749febdf90f45e84b687142ba03`；SHA-256 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` 与官方 LFS 一致。没有改名重导出或修改权重字节。
- 分割辅助模型登记在 `sentinel/models/catalog.py` 的 `SEGMENTATION_MODELS`，与检测器的 `MODELS` 分开，避免被 YOLO 加载器误调用。权重、源码副本、独立环境和素材均被现有忽略规则覆盖，不入 Git。

## 显存保护与实际验证

只做单图、batch=1、推理模式，不与 YOLO 同时加载，不编译模型，不缓存整段视频。启动前检查可用显存；仅当预算至少 2 GiB 时使用 GPU，预算不超过 3 GiB，并预留当前可用显存中的至少 2 GiB。每张图释放图像 embedding 和缓存；可用显存不足或触及 CUDA 分配上限时切 CPU。

同一批 12 张做初始提示和提示修正各一次，均在 CUDA 上成功完成，没有 OOM 或 CPU 回退。单轮 PyTorch 峰值 allocated 为 627,620,864 字节（598.5 MiB），reserved 为 694,157,312 字节（662 MiB）；这些是本进程 PyTorch 统计，不是整机总显存，也不能保证其他程序抢占显存时永远不失败。原图最大 1892×1062，SAM 编码器使用其原生 1024 输入。

另对同批中的一张做真实 CPU 单图验证成功；自动 OOM 回退分支没有通过刻意制造 OOM 来实测。4 项提示输入校验测试通过，覆盖超额批次、重名/路径穿越、越界/非有限点框和正负提示。原有 `verify_models.py`、`compare_models.py` 在同张图、`dfire_ludan`、CPU 下均执行成功，仅验证入口兼容，不用于判断模型效果。

## 查看与重跑

首轮 `outputs/sam21_ebike_pilot_20261116/run_001/`；同图提示修正版 `run_002/`。每轮有：

- `contact_01.jpg` 至 `contact_03.jpg`：提示图、绿色分割叠图、黑白 PNG 掩膜对照。
- `images/`、`masks/`、`labels/`：原图副本、原尺寸二值实例掩膜、检测框标签副本或候选框。
- `candidates.jsonl`、`summary.json`：输入提示、来源哈希、候选状态和显存记录。

重跑示例，在项目根目录执行，输出目录必须全新或为空：

```powershell
.\outputs\sam21_local\env\Scripts\python.exe scripts/preview_sam_masks.py --prompts datasets/sam21_ebike_pilot_20261116/prompts_refined.jsonl --checkpoint models/segmentation/sam2.1_hiera_small.pt --out outputs/sam21_ebike_pilot_20261116/run_003
```

可以增加 `--device cpu` 强制 CPU。单次入口最多接受 12 个实例；提示文件为逐行 JSON，`image` 和可选 `label` 相对提示文件目录解析，`box_xyxy` 和可选 `points` 使用原图像素，`point_labels` 中 1 为前景、0 为背景。提示来自既有人工检测框或代理目视框点，不能描述为用户逐张确认的分割真值。

## 效果限制与 Det-MSCEA 下一步

目视检查发现：较明显的车棚/街边白烟可形成可读掩膜；低对比度透明烟、遮挡和电池附近小烟仍有漏分或误选车身/背景。`manual_box_02`、`old_case_008_0018`、`old_case_009_0018` 尤其需要重新提示或手工多边形修正。第二轮只是修正同一批提示，没有新增 12 张，也没有人工验收通过记录。

`sam_predicted_iou` 是模型自身估计，不是与人工真值比较的实测准确率。全部输出标记 `training_eligible=false`、`annotation_status=sam_candidate_needs_human_review`；刻意输出 `candidates.jsonl` 而不生成可直接用于增强的 `manifest.jsonl`。人工通过后，还要检查每实例掩膜质量、检测标签目标索引、原有同图其他类别框、素材许可和训练/留出事件划分，再为 Det-MSCEA 导出已审核清单。不能将框当作掩膜，不能将未标注的 fire 或 smoke 当背景，也不能仅因用户觉得视觉效果好就推断外部素材许可已解决。

新抓取仅保留北京日报充电车棚冒烟监控图；另一个新闻页面取得的灭火喷水、残骸、装饰动画经目视排除并删除，来源排除记录保留。没有扩充新的反面素材。新闻素材仅限本地效果审阅，未进入训练/验证/测试或云端包。

## 旧素材清理

按本轮授权，对 acquisition_20261030 中 7 个已经下载验证且有解压数据的混合来源原包重新核对 SHA-256 后删除，合计 7,967,517,996 字节（约 7.97 GB / 7.42 GiB）。本次删的是冗余原包，不等于逐图清理了全部无关场景。已有训练清单仍引用解压图片，因此保留它们及人工标签、权重、来源/许可/下载记录。清理逐文件记录及重新下载地址在本地 `outputs/sam21_local/cleanup_20261116.json`。早期文档所述“8 个原包保留”是历史状态。
