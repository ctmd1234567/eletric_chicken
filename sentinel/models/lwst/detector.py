"""LwST-FSD FIgLib 五帧推理适配器；保持作者的 BGR/0..255/右下补边流程。"""
import pickle
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics.engine.results import Results

from .architecture import build_network
from .video import FiveFrameReader


class TrainingHistory:
    """仅容纳 checkpoint 中不参与推理的 MMEngine 训练日志状态。"""
    # checkpoint pickle 引用了这四个方法；即使不计算日志，加载时也必须存在。
    @staticmethod
    def _unused(*args, **kwargs):
        raise RuntimeError('推理适配器不执行训练日志统计')

    min = max = mean = current = _unused


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('mmengine.logging.history_buffer', 'HistoryBuffer'):
            return TrainingHistory
        return super().find_class(module, name)


def load_checkpoint(path):
    # 仅在调用方核对固定 SHA-256 后使用；训练日志不需要安装训练框架。
    module = SimpleNamespace(Unpickler=CheckpointUnpickler, __name__='pickle')
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=module)


def preprocess_clip(frames, size, device):
    height, width = frames[2].shape[:2]
    ratio = size / max(height, width)
    new_width, new_height = int(width * ratio + .5), int(height * ratio + .5)
    images = []
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError('五帧视频尺寸必须一致')
        image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[:new_height, :new_width] = image
        images.append(padded.transpose(2, 0, 1))
    inputs = torch.from_numpy(np.stack(images)).to(device=device, dtype=torch.float32)
    return inputs, (new_width / width, new_height / height)


def decode_outputs(outputs, threshold, iou, scale):
    boxes_all, scores_all = [], []
    classes, regressions, objectnesses = outputs
    for stride, cls, reg, obj in zip((8, 16, 32), classes, regressions, objectnesses):
        height, width = cls.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(height, device=reg.device),
                                torch.arange(width, device=reg.device), indexing='ij')
        priors = torch.stack((xx, yy), -1).reshape(-1, 2) * stride
        reg = reg.permute(0, 2, 3, 1).reshape(-1, 4)
        center = reg[:, :2] * stride + priors
        half = reg[:, 2:].exp() * stride / 2
        boxes = torch.cat((center - half, center + half), 1)
        scores = cls.sigmoid().reshape(-1) * obj.sigmoid().reshape(-1)
        mask = scores >= threshold
        boxes_all.append(boxes[mask])
        scores_all.append(scores[mask])
    boxes, scores = torch.cat(boxes_all), torch.cat(scores_all)
    # 作者先还原原图尺度再做 NMS，保留同样顺序。
    boxes = boxes / boxes.new_tensor([scale[0], scale[1], scale[0], scale[1]])
    keep = nms(boxes, scores, iou)
    return torch.cat((boxes[keep], scores[keep, None], torch.zeros_like(scores[keep, None])), 1)


class LwSTDetector:
    """仅负责模型推理；视频窗口、网络结构和输出后处理在独立模块。"""
    def __init__(self, path):
        checkpoint = load_checkpoint(path)
        self.model = build_network()
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        self.names = {0: 'smoke'}
        self.task = 'detect'
        self.ckpt = {'meta': checkpoint.get('meta', {})}
        self.reader = None
        self.support_indices = []
        self.available_frame_index = 0

    def start_video(self, path):
        self.close_video()
        self.reader = FiveFrameReader(path)

    def close_video(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    @torch.inference_mode()
    def predict(self, frame, imgsz=1024, conf=.10, iou=.65, rect=False,
                device='0', verbose=False, frame_index=None):
        if rect:
            raise ValueError('LwST-FSD 使用作者的方形右下补边，rect 必须为 False')
        if self.reader is None:
            frames = [frame] * 5
            self.support_indices = [0] * 5
            self.available_frame_index = 0
        else:
            frames, self.support_indices, self.available_frame_index = self.reader.read(frame_index)
        target_device = torch.device('cuda:0' if device is None else f'cuda:{device}' if str(device).isdigit() else device)
        self.model.to(target_device)
        inputs, scale = preprocess_clip(frames, imgsz, target_device)
        outputs = self.model.bbox_head(self.model.neck(self.model.backbone(inputs)))
        boxes = decode_outputs(outputs, conf, iou, scale)
        return [Results(orig_img=frame, path='', names=self.names, boxes=boxes)]
