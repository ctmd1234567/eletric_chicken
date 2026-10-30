"""修改下方设置后，在 PyCharm 中直接运行视频基线和连续帧对照。"""

if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.paths import ROOT
from sentinel.pipelines.video import run as run_baseline

# 视频文件或视频目录；相对路径以项目根目录为准。
SOURCE = "video/vvvv.mp4"
# 模型名 → 输入尺寸：已按各自上游示例/权重原生尺寸设置，注释一行即可排除。
# 同时启用多行会分别运行多个模型，所有模型处理相同视频和相同帧。
# firesmockdetect_y11m：FireSmockDetect 烟雾模型。
# early_fire_smoke：YOLOv8n 火和烟雾；
# dfire_yolov8n：D-Fire 火和烟雾。
# lwst_fsd：作者 FIgLib 五帧时序模型；前后各两帧。
# safety_yolo26：本地候选，许可和类别限制见 NOTICE.md。
MODELS = {
    # "firesmockdetect_y11m": 1280,
    # "early_fire_smoke": 640,
    "dfire_yolov8n": 640,
    # "safety_yolo26": 640,
    # "lwst_fsd": 1024,
}
DEVICE = "0"  # 首张 NVIDIA GPU；使用 CPU 时改为 "cpu"。
CONFIDENCE = 0.10  # 手动阈值，直接用于所选模型；None 才使用各自上游默认。
IOU = None # 手动 NMS 阈值，越低重叠框抑制越强；None 使用各自默认（LwST=0.65）。
# 只在这里调参。填写数字后直接生效，没有单模型参数再次覆盖。
VIDEO_STRIDE = 1  # 每几帧检测一次；1 为逐帧检测。
WARMUP = 0  # 每个模型和视频的预热次数；不计入稳定态耗时。
CONSECUTIVE = [1, 3, 5]  # 分别比较连续多少个采样帧检测到烟雾。
SAVE_IMAGES = True  # 保存带检测框的视频，需 PATH 中有 FFmpeg。
SHOW_IMAGES = True  # 完成后用系统播放器打开带框视频。
OUTPUT = "samples/video_results"  # 结果父目录，每次运行创建编号子目录。


def run() -> int:
    return run_baseline(
        ROOT / SOURCE, list(MODELS), model_imgsz=MODELS, device=DEVICE, conf=CONFIDENCE, iou=IOU,
        stride=VIDEO_STRIDE, warmup=WARMUP, consecutive=CONSECUTIVE,
        out=ROOT / OUTPUT, save_images=SAVE_IMAGES, show_images=SHOW_IMAGES,
    )


if __name__ == "__main__":
    raise SystemExit(run())
