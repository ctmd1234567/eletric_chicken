"""验证五帧边界、BGR/像素尺度、YOLOX 解码和真实时序特征路径。"""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sentinel.models.lwst.detector import LwSTDetector, preprocess_clip, decode_outputs
from sentinel.models.lwst.sources import SOURCE_ROOT, SOURCE_HASHES
from sentinel.models.lwst.video import FiveFrameReader


class WindowTests(unittest.TestCase):
    def test_edges_order_repeat_and_stream_reset(self):
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(6)]
        class Capture:
            def __init__(self):
                self.index = 0
            def read(self):
                if self.index == len(frames):
                    return False, None
                frame = frames[self.index]
                self.index += 1
                return True, frame
            def release(self):
                pass
        with patch('sentinel.models.lwst.video.cv2.VideoCapture', side_effect=lambda _: Capture()):
            reader = FiveFrameReader('a')
            _, indices, available = reader.read(0)
            self.assertEqual(indices, [0, 0, 0, 1, 2])
            self.assertEqual(available, 2)
            self.assertEqual(reader.read(0)[1], indices)
            self.assertEqual(reader.read(2)[1], [0, 1, 2, 3, 4])
            self.assertEqual(reader.read(5)[1:], ([3, 4, 5, 5, 5], 5))
            reader.close()
            reader = FiveFrameReader('b')
            self.assertEqual(reader.read(0)[1], [0, 0, 0, 1, 2])
            reader.close()

    def test_preprocess_preserves_bgr_and_raw_pixel_scale(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[:] = [10, 20, 30]
        inputs, scale = preprocess_clip([image] * 5, 32, 'cpu')
        self.assertEqual(tuple(inputs.shape), (5, 3, 32, 32))
        self.assertEqual(inputs[2, :, 0, 0].tolist(), [10, 20, 30])
        self.assertEqual(inputs[2, :, -1, -1].tolist(), [114, 114, 114])
        self.assertEqual(scale, (8, 8))

    def test_decoder_uses_objectness_and_zero_offset_grid(self):
        classes = [torch.full((1, 1, 1, 1), value) for value in (10., -100., -100.)]
        objects = [x.clone() for x in classes]
        boxes = [torch.zeros((1, 4, 1, 1)) for _ in range(3)]
        result = decode_outputs((classes, boxes, objects), .1, .65, (2, 2))
        self.assertEqual(tuple(result.shape), (1, 6))
        torch.testing.assert_close(result[0, :4], torch.tensor([-2., -2., 2., 2.]))
        self.assertGreater(float(result[0, 4]), .99)
        empty = decode_outputs((classes, boxes, objects), 1., .65, (2, 2))
        self.assertEqual(tuple(empty.shape), (0, 6))


WEIGHTS = Path(__file__).resolve().parent.parent / 'models/fire_smoke/lwst_fsd/epoch_10.pth'


@unittest.skipUnless(WEIGHTS.exists() and all((SOURCE_ROOT / name).exists() for name in SOURCE_HASHES),
                     '需要本地作者权重和源码；测试不负责下载')
class RealNetworkTests(unittest.TestCase):
    def test_support_frames_change_predictions_with_same_center_frame(self):
        torch.set_num_threads(2)
        model = LwSTDetector(WEIGHTS).model
        self.assertEqual(len(model.state_dict()), 567)
        generator = torch.Generator().manual_seed(7)
        inputs = torch.rand((5, 3, 64, 64), generator=generator) * 255
        changed = inputs.clone()
        changed[[0, 1, 3, 4]] = 114
        with torch.inference_mode():
            feats = model.neck(model.backbone(inputs))
            first = model.bbox_head(feats)
            second = model.bbox_head(model.neck(model.backbone(changed)))
        self.assertEqual([tuple(x.shape) for x in feats], [(1, 128, 8, 8), (1, 128, 4, 4), (1, 128, 2, 2)])
        difference = max(float((a - b).abs().max()) for group_a, group_b in zip(first, second)
                         for a, b in zip(group_a, group_b))
        self.assertGreater(difference, 1e-5)
        self.assertTrue(all(torch.isfinite(x).all() for group in first for x in group))


if __name__ == '__main__':
    unittest.main()
