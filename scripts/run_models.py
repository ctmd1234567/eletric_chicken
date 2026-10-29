"""修改下方设置后，在 PyCharm 中直接运行此文件。"""

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from sentinel.paths import ROOT
from sentinel.pipelines.compare import run as run_comparison


# 图片、图片目录或视频；相对路径以项目根目录为准。
SOURCE = "video/Video Project 8.mp4"
# 在列表中保留要测试的模型，也可同时填写多个进行对比：
# firesmockdetect_y11m：YOLO11m，烟雾。
# early_fire_smoke：YOLOv8n，火和烟雾。
# dfire_yolov8n：rabahdev 基于 D-Fire 训练的 YOLOv8n，烟雾和火。
# safety_yolo26：e1250/safety_detection，Home Fire Dataset 训练的 YOLO26n，640 输入。
#   权重实际类别为 0=smoke、1=fire，与模型卡相反；保留权重原始映射，不擅自交换。
#   未声明权重许可，仅作为本地待核实候选；详见 NOTICE.md。
# Safety YOLO26 权重实测 end2end=False，当前走传统 NMS（默认 iou=0.7）。
# 例如只测试 YOLOv8n：MODELS = ["early_fire_smoke"]
MODELS = [
    "firesmockdetect_y11m",
    "early_fire_smoke",
    "dfire_yolov8n",
    "safety_yolo26",
]
DEVICE = "0"  # 首张 NVIDIA GPU；使用 CPU 时改为 "cpu"。
CONFIDENCE = None  # 各用上游示例：YOLO11m=0.10，其余三个=库默认0.25；填写数字会统一覆盖。
IOU = None  # 填写数字直接设置所选模型的 NMS 阈值；None 使用各自默认。
# 原生输入：YOLO11m=1280，其余三个=640；具体来源见 docs/upstream_inference.md。
VIDEO_STRIDE = 1  # 每几帧检测一次；1 为逐帧检测，保留原视频帧率。
SAVE_IMAGES = True  # 保存结果：图片→JPG，视频→H.264 MP4（无音轨，需要 FFmpeg）。
SHOW_IMAGES = True  # 图片逐张显示；视频处理完成后，用系统播放器播放带框 MP4。
OUTPUT = "samples/images_results"  # 结果父目录，每次运行创建编号子目录。


def run() -> int:
    return run_comparison(
        ROOT / SOURCE, MODELS, device=DEVICE, conf=CONFIDENCE, iou=IOU, stride=VIDEO_STRIDE,
        out=ROOT / OUTPUT, save_images=SAVE_IMAGES, show_images=SHOW_IMAGES,
    )


if __name__ == "__main__":
    raise SystemExit(run())
