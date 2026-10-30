"""所有推理入口共用的加载与参数解析；不依赖运行脚本。"""
from sentinel.paths import ROOT
from .catalog import MODELS
from .weights import sha256_of


def load_detector(name):
    entry = MODELS[name]
    path = ROOT / entry['dest']
    digest = sha256_of(path)
    if digest != entry['sha256'] or path.stat().st_size != entry['size']:
        raise ValueError(f'权重校验失败: {name}')
    if name == 'lwst_fsd':
        from .lwst.detector import LwSTDetector
        model = LwSTDetector(path)
    else:
        from ultralytics import YOLO
        model = YOLO(str(path))
    if 'display_names' in entry and model.names == {0: 'item'}:
        model.model.names = entry['display_names']
    return model


def prediction_settings(name, model, *, imgsz=None, iou=None, rect=None, conf=None):
    """显式覆盖 > 上游推荐 > checkpoint 原值 > 推理库默认，并记录来源。"""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    entry = MODELS[name]
    checkpoint_args = model.ckpt.get('train_args') or {}
    if imgsz is not None and (imgsz < 32 or imgsz % 32):
        raise ValueError('输入尺寸必须是至少 32 的 32 倍数')
    if iou is not None and not 0 <= iou <= 1:
        raise ValueError('IoU 必须在 [0,1] 内')
    if conf is not None and not 0 <= conf <= 1:
        raise ValueError('置信度必须在 [0,1] 内')
    if conf is not None:
        conf_source = 'explicit override'
    elif 'conf' in entry:
        conf, conf_source = entry['conf'], entry['conf_source']
    else:
        # 作者示例 model(...) 未设阈值；8.4.192 的 Model.predict 使用 0.25。
        # 不能取 train_args.conf 或 DEFAULT_CFG_DICT.conf（后者为 None）。
        conf, conf_source = .25, 'ultralytics predict default (author example has no override)'
    if iou is not None:
        source = 'explicit override'
    elif entry.get('iou') is not None:
        iou, source = entry['iou'], 'upstream recommendation'
    elif checkpoint_args.get('iou') is not None:
        iou, source = checkpoint_args['iou'], 'checkpoint train_args'
    else:
        iou, source = DEFAULT_CFG_DICT['iou'], 'ultralytics default'
    return dict(imgsz=imgsz if imgsz is not None else entry['imgsz'], conf=conf, conf_source=conf_source, iou=iou,
                iou_source=source, rect=entry.get('rect', True) if rect is None else rect)


def predict(model, frame, settings, conf, device, frame_index=None):
    options = dict(imgsz=settings['imgsz'], iou=settings['iou'], rect=settings['rect'],
                   conf=conf, device=device, verbose=False)
    if frame_index is not None:
        options['frame_index'] = frame_index
    return model.predict(frame, **options)[0]


def detections(result):
    """一次转到 CPU，供 CSV 和评估使用。"""
    return [dict(cls=result.names[int(cls)], conf=conf, xyxy=box)
            for cls, conf, box in zip(result.boxes.cls.cpu().tolist(),
                                      result.boxes.conf.cpu().tolist(),
                                      result.boxes.xyxy.cpu().tolist())]
