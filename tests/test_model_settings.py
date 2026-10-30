"""防止再次将不同模型的 IoU 强行统一。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sentinel.models.runtime import prediction_settings, load_detector


class ModelSettingsTests(unittest.TestCase):
    def test_upstream_confidence_examples_are_model_specific(self):
        model = SimpleNamespace(ckpt={'train_args': {'conf': .9}})
        self.assertEqual(prediction_settings('firesmockdetect_y11m', model)['conf'], .10)
        self.assertEqual(prediction_settings('lwst_fsd', model)['conf'], .0001)
        for name in ('early_fire_smoke', 'dfire_yolov8n', 'safety_yolo26'):
            with self.subTest(name=name):
                settings = prediction_settings(name, model)
                self.assertEqual(settings['conf'], .25)
                self.assertEqual(settings['imgsz'], 640)
                self.assertIn('predict default', settings['conf_source'])

    def test_explicit_confidence_including_zero_overrides_upstream(self):
        for value in (0., .15):
            settings = prediction_settings('lwst_fsd', SimpleNamespace(ckpt={}), conf=value)
            self.assertEqual(settings['conf'], value)
            self.assertEqual(settings['conf_source'], 'explicit override')

    def test_invalid_confidence_is_rejected(self):
        for value in (-.1, 1.1):
            with self.assertRaisesRegex(ValueError, '置信度'):
                prediction_settings('lwst_fsd', SimpleNamespace(ckpt={}), conf=value)

    def test_checkpoint_iou_survives_for_an_ordinary_yolo_model(self):
        model = SimpleNamespace(ckpt={'train_args': {'iou': .37}})
        settings = prediction_settings('dfire_yolov8n', model)
        self.assertEqual(settings['iou'], .37)
        self.assertEqual(settings['iou_source'], 'checkpoint train_args')
        self.assertEqual(settings['imgsz'], 640)

    def test_recommendations_override_training_settings(self):
        model = SimpleNamespace(ckpt={'train_args': {'iou': .2}})
        lwst = prediction_settings('lwst_fsd', model)
        self.assertEqual(lwst['iou'], .65)
        self.assertFalse(lwst['rect'])
        explicit = prediction_settings('lwst_fsd', model, iou=.4)
        self.assertEqual(explicit['iou'], .4)
        self.assertEqual(explicit['iou_source'], 'explicit override')

    def test_absent_checkpoint_iou_uses_library_default(self):
        with patch.dict('ultralytics.cfg.DEFAULT_CFG_DICT', {'iou': .43}):
            settings = prediction_settings('dfire_yolov8n', SimpleNamespace(ckpt={}))
        self.assertEqual(settings['iou'], .43)
        self.assertEqual(settings['iou_source'], 'ultralytics default')

    def test_bad_digest_stops_before_checkpoint_loading(self):
        with patch('sentinel.models.runtime.sha256_of', return_value='bad'), \
             patch('ultralytics.YOLO') as yolo:
            with self.assertRaisesRegex(ValueError, '权重校验失败'):
                load_detector('dfire_yolov8n')
        yolo.assert_not_called()
