"""修改下方设置后，在 PyCharm 中直接运行此文件。"""

from pathlib import Path

from compare_models import main

ROOT = Path(__file__).resolve().parent.parent

# 图片、图片目录或视频；相对路径以项目根目录为准。
SOURCE = "video/vvvv.mp4"
# 在列表中保留要测试的模型，也可同时填写多个进行对比：
# firesmockdetect_y11m：YOLO11m，烟雾；firesmockdetect_y11s：YOLO11s，烟雾。
# early_fire_smoke：YOLOv8n，火和烟雾。
# pyronear_sensitive：保留旧选项名，权重已换为 Pyronear Tenacious Tiger v8.3.0。
#   仍是 YOLO11s 野火烟雾模型；官方输入 1024，下面 PYRONEAR_IMGSZ 可覆盖。
# dfire_yolov8n：rabahdev 基于 D-Fire 训练的 YOLOv8n，烟雾和火。
# fasdd_yolo26n：seawsurf/fire_yolo，FASDD_CV 训练的 YOLO26n，640 输入。
# safety_yolo26：e1250/safety_detection，Home Fire Dataset 训练的 YOLO26n，640 输入。
#   权重实际类别为 0=smoke、1=fire，与模型卡相反；保留权重原始映射，不擅自交换。
#   未声明权重许可，仅作为本地待核实候选；详见 NOTICE.md。
# 两个 YOLO26 权重实测 end2end=False，当前均走传统 NMS（默认 iou=0.7）。
# 例如只测试 YOLOv8n：MODELS = ["early_fire_smoke"]
# 对比本次三个更新候选：
# MODELS = ["pyronear_sensitive", "fasdd_yolo26n", "safety_yolo26"]
MODELS = [
    "safety_yolo26",
]
DEVICE = "0"  # 首张 NVIDIA GPU；使用 CPU 时改为 "cpu"。
CONFIDENCE = 0.05  # 所选模型共用的诊断阈值，非已验证告警阈值；Pyronear 官方示例为 0.20。
# 仅作用于 pyronear_sensitive；其他模型仍使用各自的默认输入和 NMS。
PYRONEAR_IMGSZ = 1280  # 保留当前自定义尺寸；复现 v8.3.0 官方示例时改为 1024。
PYRONEAR_IOU = 0.01  # 官方推荐值；无框问题不能仅归因于 NMS。

PYRONEAR_RECT = False  # 完整方形补边；不拉伸原图，不保证优于默认矩形补边。
VIDEO_STRIDE = 1  # 每几帧检测一次；1 为逐帧检测，保留原视频帧率。
SAVE_IMAGES = True  # 保存结果：图片→JPG，视频→H.264 MP4（无音轨，需要 FFmpeg）。
SHOW_IMAGES = True  # 图片逐张显示；视频处理完成后，用系统播放器播放带框 MP4。
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
    if "pyronear_sensitive" in MODELS:
        args.extend([
            "--pyronear-imgsz", str(PYRONEAR_IMGSZ),
            "--pyronear-iou", str(PYRONEAR_IOU),
            "--pyronear-rect" if PYRONEAR_RECT else "--no-pyronear-rect",
        ])
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
