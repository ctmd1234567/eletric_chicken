"""模型的唯一登记处：来源、哈希、原生尺寸和明确的上游推理设置。

iou 缺省表示读取 checkpoint，不能在各入口补一个统一常数。
"""

MODELS = {
    'lwst_fsd': {
        'group': 'lwst_fsd',
        'repo': 'WCUSTC/LwST-FSD',
        'revision': 'af3ed8ac9686a607ec2330275fbb95cf8d1f910d',
        'file': 'epoch_10.pth',
        'provider': 'ustc_share',
        'share_id': '5b015f2c73e24fd48b82',
        'size': 141991741,
        'sha256': '6ac5ca325d59f1bb48a8f63750ba1efd49b53cc5b34659395e8edf14ad8b3add',
        'dest': 'models/fire_smoke/lwst_fsd/epoch_10.pth',
        'imgsz': 1024,
        'conf': 0.0001,
        'conf_source': 'checkpoint meta.cfg model.test_cfg.score_thr',
        'iou': 0.65,
        'rect': False,
        'temporal_frames': 5,
    },
    'firesmockdetect_y11m': {
        'group': 'firesmockdetect',
        'repo': 'TianchenGuan/FireSmockDetect-models',
        'revision': 'a914a2a036686cf09bac8efc738971fdc8f31a29',
        'file': 'v5bA_y11m_best.pt',
        'size': 40578796,
        'sha256': '1b6cb82295767ae9d0d5c4fb94dffcaf17bbbafc0e3af001d3fe517015902298',
        'dest': 'models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt',
        'imgsz': 1280,
        'conf': 0.10,
        'conf_source': 'upstream model card usage example',
    },
    'early_fire_smoke': {
        'group': 'early_fire_smoke',
        'repo': 'mfranzon/fire-smoke-yolov8',
        'revision': 'f1c6426b069c1849cbf13b1ef5d2a260289286db',
        'file': 'fire_smoke_yolov8.pt',
        'size': 6262051,
        'sha256': 'ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16',
        'dest': 'models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt',
        'imgsz': 640,
    },
    'dfire_yolov8n': {
        'group': 'dfire_yolov8n',
        'repo': 'rabahdev/fire-smoke-yolov8n',
        'revision': '13017fe8af477c25f5298d168e2dfede4b000753',
        'file': 'best.pt',
        'size': 6229802,
        'sha256': 'b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f',
        'dest': 'models/fire_smoke/dfire_yolov8n/best.pt',
        'imgsz': 640,
    },
    'safety_yolo26': {
        'group': 'safety_yolo26',
        'repo': 'e1250/safety_detection',
        'revision': 'f2049aa749dcc844e406d2507ffa3b683ce83aac',
        'file': 'yolo_smoke_fire.pt',
        'size': 15735298,
        'sha256': 'cd2c8c1108b29589d77d4a21cc6e5f111c7586b225e2d8668fda50d791a2cbea',
        'dest': 'models/fire_smoke/safety_yolo26/yolo_smoke_fire.pt',
        'imgsz': 640,
    },
}

YOLO_MODELS = [name for name in MODELS if name != "lwst_fsd"]
# 保留原默认集合中仍在用的模型；候选 Safety / LwST 需显式选择。
DEFAULT_VIDEO_MODELS = ['firesmockdetect_y11m', 'early_fire_smoke', 'dfire_yolov8n']
