"""验证单处配置同时控制实际模型选择和输入尺寸。"""
import unittest
from unittest.mock import patch

import run_video_baseline as runner


class EntrySettingsTests(unittest.TestCase):
    def test_selected_models_and_sizes_are_forwarded_together(self):
        with patch.object(runner, 'MODELS', {'dfire_yolov8n': 1920}), \
             patch.object(runner, 'main', return_value=0) as main:
            self.assertEqual(runner.run(), 0)
        args = main.call_args.args[0]
        self.assertIn('--model-imgsz', args)
        self.assertIn('dfire_yolov8n=1920', args)
        self.assertIn('dfire_yolov8n', args)
        self.assertNotIn('safety_yolo26', args)

    def test_partial_size_overrides_and_internal_iou(self):
        with patch.object(runner, 'MODELS', {'pyronear_sensitive': 1024, 'safety_yolo26': 640}), \
             patch.object(runner, 'PYRONEAR_RECT', False), \
             patch.object(runner, 'main', return_value=0) as main:
            self.assertEqual(runner.run(), 0)
        args = main.call_args.args[0]
        self.assertIn('pyronear_sensitive=1024', args)
        self.assertNotIn('--pyronear-iou', args)
        self.assertIn('--no-pyronear-rect', args)

    def test_no_model_selected_fails_before_backend(self):
        with patch.object(runner, 'MODELS', {}), patch.object(runner, 'main') as main:
            with self.assertRaisesRegex(ValueError, 'MODELS'):
                runner.run()
        main.assert_not_called()


if __name__ == '__main__':
    unittest.main()
