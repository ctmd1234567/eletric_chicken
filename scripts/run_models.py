"""修改下方设置后，在 PyCharm 中直接运行此文件。"""

from pathlib import Path

from compare_models import main

ROOT = Path(__file__).resolve().parent.parent

# 图片、图片目录或视频；相对路径以项目根目录为准。
SOURCE = "samples/images"
# 在列表中保留要测试的模型，也可同时填写多个进行对比：
# firesmockdetect_y11m：YOLO11m，烟雾；firesmockdetect_y11s：YOLO11s，烟雾。
# early_fire_smoke：YOLOv8n，火和烟雾。
# 例如只测试 YOLOv8n：MODELS = ["early_fire_smoke"]
MODELS = ["firesmockdetect_y11m", "firesmockdetect_y11s", "early_fire_smoke"]
DEVICE = "0"  # 首张 NVIDIA GPU；使用 CPU 时改为 "cpu"。
CONFIDENCE = 0.15
VIDEO_STRIDE = 5
SAVE_IMAGES = True  # 保存有检测框的图片。
SHOW_IMAGES = False  # True：显示每张检测结果，按任意键查看下一张。
OUTPUT = "samples/images_results"  # 结果父目录，每次运行创建编号子目录。


def run() -> int:
    source = Path(SOURCE)
    if not source.is_absolute():
        source = ROOT / source
    args = [
        "--source", str(source), "--models", *MODELS,
        "--device", DEVICE, "--conf", str(CONFIDENCE),
        "--stride", str(VIDEO_STRIDE),
    ]
    if SAVE_IMAGES:
        args.append("--save-images")
    if SHOW_IMAGES:
        args.append("--show-images")
    if OUTPUT is not None:
        output = Path(OUTPUT)
        if not output.is_absolute():
            output = ROOT / output
        args.extend(["--out", str(output)])
    return main(args)


if __name__ == "__main__":
    raise SystemExit(run())
