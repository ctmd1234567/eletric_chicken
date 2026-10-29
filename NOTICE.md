# 第三方模型与素材说明

本仓库随 Git 分发三个第三方训练权重，原样转发，**未修改任何权重字节**。图片、视频与数据集不随仓库分发。

## 模型来源

| 权重 | 上游仓库 | revision | 许可 |
|---|---|---|---|
| `models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt` | [HF: TianchenGuan/FireSmockDetect-models](https://huggingface.co/TianchenGuan/FireSmockDetect-models) | `a914a2a036686cf09bac8efc738971fdc8f31a29` | AGPL-3.0 |
| `models/fire_smoke/firesmockdetect/expB_y11s_deployed.pt` | 同上 | `a914a2a036686cf09bac8efc738971fdc8f31a29` | AGPL-3.0 |
| `models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt` | [HF: mfranzon/fire-smoke-yolov8](https://huggingface.co/mfranzon/fire-smoke-yolov8) | `f1c6426b069c1849cbf13b1ef5d2a260289286db` | AGPL-3.0 |

2026-10-29 核对上述固定版本的模型卡，三者均标注 `agpl-3.0`。本地文件的 SHA-256 与 Hugging Face 返回的 LFS oid 逐字节一致，见 `scripts/download_models.py` 的 `MANIFEST`。

FireSmockDetect 的代码与实验报告在 [GitHub](https://github.com/TianchenGuan/FireSmockDetect)，其中代码为 AGPL-3.0、报告与文档为 CC BY 4.0。Early Fire and Smoke 只发布了权重，模型卡描述的连续帧状态机没有公开实现。

## 为什么本仓库也是 AGPL-3.0

三个权重都是 Ultralytics YOLO11 / YOLOv8 的微调产物，上游以 AGPL-3.0 发布。AGPL-3.0 允许原样再分发，但要求衍生作品整体同样以 AGPL-3.0 授权并提供对应源码，因此本仓库的脚本与文档统一采用 AGPL-3.0（全文见 `LICENSE`）。

如果你把本仓库的推理能力以网络服务形式提供给他人，AGPL-3.0 第 13 条要求你向使用者提供完整源码。

## 数据集与素材

- 训练数据（pyro-sdis、D-Fire、Domestic Fire and Smoke）**不随本仓库分发**，需从原始出处获取，许可归各自作者。
- FIgLib / HPWREN 为 CC BY-NC-ND，上游只用于回放评测，从未参与训练，本仓库也不分发。
- 本地 `samples/rf100_smoke/` 为 [RF100 smoke](https://universe.roboflow.com/roboflow-100/smoke-uvylj/dataset/2) 测试子集（CC BY 4.0），仅用于本机运行检查，已被 `.gitignore` 排除。
- 本地 `video/` 为第三方新闻视频，版权不在本项目，已被 `.gitignore` 排除，不得公开分发。

## 免责

这两个模型是研究工具，不是经过认证的火灾报警装置——上游作者也明确这样声明。本仓库未在电动车车棚场景做过任何性能验证，不要把其中的模型直接当作生命安全保障设备部署。
