"""编辑器配置直接传给视频流程，尺寸不会与模型选择脱节。"""
import unittest
from unittest.mock import patch

from scripts import run_video_baseline as runner


class EntrySettingsTests(unittest.TestCase):
    def test_selected_models_and_sizes_are_forwarded_together(self):
        with patch.object(runner, 'MODELS', {'dfire_yolov8n': 1920}), patch.object(runner, 'run_baseline', return_value=0) as run:
            self.assertEqual(runner.run(), 0)
        self.assertEqual(run.call_args.args[1], ['dfire_yolov8n'])
        self.assertEqual(run.call_args.kwargs['model_imgsz'], {'dfire_yolov8n': 1920})
        self.assertEqual(run.call_args.kwargs['conf'], runner.CONFIDENCE)
        self.assertEqual(run.call_args.kwargs['iou'], runner.IOU)
        self.assertNotIn('model_conf', run.call_args.kwargs)
        self.assertNotIn('model_iou', run.call_args.kwargs)

    def test_no_model_selected_fails_before_backend(self):
        with patch.object(runner, 'MODELS', {}), patch('sentinel.pipelines.video.load_detector') as load:
            with self.assertRaises(ValueError):
                runner.run()
        load.assert_not_called()
