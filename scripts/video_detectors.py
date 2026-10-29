"""视频入口的检测器注册与加载；旧图片/视频对比入口保持原有模型集合。"""
from compare_models import MODELS as YOLO_MODELS, NMS_IOU as YOLO_IOU

MODELS = {**YOLO_MODELS, 'lwst_fsd': ('models/fire_smoke/lwst_fsd/epoch_10.pth', 1024)}
NMS_IOU = {**YOLO_IOU, 'lwst_fsd': .65}


def load_detector(name, path):
    if name == 'lwst_fsd':
        from lwst_backend import LwSTDetector
        return LwSTDetector(path)
    from ultralytics import YOLO
    return YOLO(str(path))
