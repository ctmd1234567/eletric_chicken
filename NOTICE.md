# 第三方模型与素材说明

本仓库提供固定版本的模型下载与验证脚本，不随 Git 分发第三方权重、图片或视频。

## 模型来源

- FireSmockDetect 的 `v5bA_y11m_best.pt` 和 `expB_y11s_deployed.pt`：
  [模型仓库及模型卡](https://huggingface.co/TianchenGuan/FireSmockDetect-models/tree/a914a2a036686cf09bac8efc738971fdc8f31a29)，revision `a914a2a036686cf09bac8efc738971fdc8f31a29`；[上游代码](https://github.com/TianchenGuan/FireSmockDetect)。
- Early Fire and Smoke 的 `fire_smoke_yolov8.pt`：
  [模型仓库及模型卡](https://huggingface.co/mfranzon/fire-smoke-yolov8/tree/f1c6426b069c1849cbf13b1ef5d2a260289286db)，revision `f1c6426b069c1849cbf13b1ef5d2a260289286db`。

2026-10-29 核对上述固定版本模型卡，两者均标注 `agpl-3.0`。使用、修改和分发时应查阅对应上游许可及条款；本说明不是替代许可证，也不额外授权。

## 软件和数据

Ultralytics、PyTorch、OpenCV 等依赖遵循各自许可证。训练数据和测试素材的许可独立于模型许可。

本地 `samples/rf100_smoke/images/` 有 40 张 JPG，但没有随附完整来源及许可记录，也没有标签，因此仅用于本次本机运行检查，不随仓库上传，不作为正式评测集。本仓库未将第三方模型的研究指标作为本项目成绩。
