"""修改下方设置后，在 PyCharm 中直接运行视频基线和连续帧对照。"""

from pathlib import Path

from video_baseline import main

ROOT = Path(__file__).resolve().parent.parent

# 视频文件或视频目录；相对路径以项目根目录为准。
SOURCE = "video/Video Project 8.mp4"
# 模型名 → 输入尺寸：取消注释即选中该模型，右侧数字由你决定。
# 同时启用多行会分别运行多个模型，所有模型处理相同视频和相同帧。
# firesmockdetect_y11m、firesmockdetect_y11s：FireSmockDetect 烟雾模型。
# early_fire_smoke：YOLOv8n 火和烟雾；dfire_yolov8n：D-Fire 火和烟雾。
# pyronear_sensitive：当前本地权重为 Tenacious Tiger v8.3.0。
# lwst_fsd：作者 FIgLib 五帧时序模型；前后各两帧，内置 IoU=0.65。
# fasdd_yolo26n、safety_yolo26：本地候选，许可和类别限制见 NOTICE.md。
MODELS = {
    # "firesmockdetect_y11m": 1920,
    # "firesmockdetect_y11s": 1280,
    # "early_fire_smoke": 1280,
    # "pyronear_sensitive": 1280,
    "dfire_yolov8n": 1920,
    "lwst_fsd": 1024,
    # "fasdd_yolo26n": 640,
    # "safety_yolo26": 640,
}
DEVICE = "0"  # 首张 NVIDIA GPU；使用 CPU 时改为 "cpu"。
CONFIDENCE = 0.10  # 检测置信度阈值；不是已验证的告警阈值。
# Pyronear 的 NMS IoU 内置为 0.01；这里仅调整补边方式。
PYRONEAR_RECT = False  # False 为方形补边，True 为矩形补边。
VIDEO_STRIDE = 1  # 每几帧检测一次；1 为逐帧检测。
WARMUP = 3  # 每个模型和视频的预热次数；不计入稳定态耗时。
CONSECUTIVE = [1, 3, 5]  # 分别比较连续多少个采样帧检测到烟雾。
SAVE_IMAGES = True  # 保存带检测框的视频，需 PATH 中有 FFmpeg。
SHOW_IMAGES = True  # 完成后用系统播放器打开带框视频。
OUTPUT = "samples/video_results"  # 结果父目录，每次运行创建编号子目录。


def run() -> int:
    if not MODELS:
        raise ValueError("请在 MODELS 中至少启用一个模型，并填写输入尺寸。")
    print(f"运行入口: {Path(__file__).resolve()}", flush=True)
    print("选中模型: " + ", ".join(f"{name} (imgsz={size})" for name, size in MODELS.items()), flush=True)
    source = Path(SOURCE)
    if not source.is_absolute():
        source = ROOT / source
    output = Path(OUTPUT)
    if not output.is_absolute():
        output = ROOT / output
    args = [
        "--source", str(source), "--models", *MODELS,
        "--device", DEVICE, "--conf", str(CONFIDENCE),
        "--stride", str(VIDEO_STRIDE), "--warmup", str(WARMUP),
        "--consecutive", *(str(n) for n in CONSECUTIVE),
        "--out", str(output),
    ]
    args.extend(["--model-imgsz", *(f"{name}={size}" for name, size in MODELS.items())])
    if "pyronear_sensitive" in MODELS:
        args.append("--pyronear-rect" if PYRONEAR_RECT else "--no-pyronear-rect")
    if SAVE_IMAGES:
        args.append("--save-images")
    if SHOW_IMAGES:
        args.append("--show-images")
    return main(args)


if __name__ == "__main__":
    raise SystemExit(run())
