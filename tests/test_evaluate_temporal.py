"""覆盖跨视频/缺帧复位、类别语义、标注边界及误报与时间计算。"""
import csv
import tempfile
import unittest
from pathlib import Path

from sentinel.evaluation.temporal import evaluate
from sentinel.reports import read_csv


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def detections(self, videos):
        path = self.root / 'detections.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', 'video_id', 'frame_index', 'timestamp_s',
                                                        'fps', 'stride', 'smoke_boxes', 'fire_boxes'])
            writer.writeheader()
            for video, samples in videos.items():
                for index, smoke in samples:
                    writer.writerow(dict(model='test', video_id=video, frame_index=index,
                                         timestamp_s=index / 10, fps=10, stride=1,
                                         smoke_boxes=smoke, fire_boxes=1))
        return path

    def test_resets_and_fire_does_not_count_as_smoke(self):
        path = self.detections({'a': [(0, 1), (1, 1), (2, 1), (3, 0), (4, 1), (6, 1), (7, 1), (8, 1)],
                                'b': [(0, 1), (1, 1)]})
        summary = evaluate(path, self.root / 'out', [1, 3, 5])
        three = next(r for r in summary if r['video_id'] == 'a' and r['consecutive'] == 3)
        self.assertEqual(three['alarm_events'], 2)
        self.assertEqual(three['first_trigger_s'], .2)
        self.assertEqual(three['alarm_active_frames'], 2)
        self.assertEqual(three['clear_frame_alarm_rate'], '')
        self.assertEqual(next(r for r in summary if r['video_id'] == 'b' and r['consecutive'] == 3)['alarm_events'], 0)

    def test_labels_delay_false_alarm_and_unknown_exclusion(self):
        path = self.detections({'a': [(i, int(i != 3)) for i in range(10)]})
        labels = self.root / 'labels.csv'
        labels.write_text('video_id,start_frame,end_frame,state,interference,smoke_onset_frame\n'
                          'a,0,2,clear,reflection,\n'
                          'a,4,8,smoke,,4\n', encoding='utf-8')
        summary = evaluate(path, self.root / 'out', [1, 3], labels)
        three = summary[1]
        self.assertEqual(three['clear_frame_alarm_rate'], 1 / 3)
        self.assertEqual(three['smoke_frame_alarm_rate'], 3 / 5)
        self.assertEqual(three['labeled_clear_frames'], 3)
        segment = read_csv(self.root / 'out/smoke_segments.csv')[1]
        self.assertAlmostEqual(float(segment['delay_s']), .2)
        frame = read_csv(self.root / 'out/frames.csv')[3]
        self.assertEqual(frame['annotation_state'], 'unknown')

    def test_overlapping_labels_and_duplicates_rejected(self):
        path = self.detections({'a': [(0, 1), (1, 1)]})
        labels = self.root / 'labels.csv'
        labels.write_text('video_id,start_frame,end_frame,state\na,0,1,clear\na,1,2,smoke\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluate(path, self.root / 'overlap', [1], labels)
        path = self.detections({'a': [(0, 1), (0, 1)]})
        with self.assertRaises(ValueError):
            evaluate(path, self.root / 'duplicate', [1])

    def test_preexisting_alarm_is_not_new_onset_detection(self):
        path = self.detections({'a': [(i, 1) for i in range(8)]})
        labels = self.root / 'labels.csv'
        labels.write_text('video_id,start_frame,end_frame,state,smoke_onset_frame\na,4,7,smoke,4\n', encoding='utf-8')
        evaluate(path, self.root / 'out', [3], labels)
        segment = read_csv(self.root / 'out/smoke_segments.csv')[0]
        self.assertEqual(segment['delay_s'], '')
        self.assertEqual(segment['timing_status'], 'no_new_trigger')

    def test_stride_changes_wait_and_gaps_reset(self):
        path = self.detections({'a': [(0, 1), (2, 1), (4, 1)]})
        content = path.read_text(encoding='utf-8').replace(',10,1,', ',10,2,')
        path.write_text(content, encoding='utf-8')
        summary = evaluate(path, self.root / 'out', [3])
        self.assertEqual(summary[0]['first_trigger_s'], .4)
        self.assertEqual(summary[0]['theoretical_min_extra_wait_s'], .4)

    def test_run_started_before_onset_is_not_effective_detection(self):
        path = self.detections({'a': [(i, 1) for i in range(5)]})
        labels = self.root / 'labels.csv'
        labels.write_text('video_id,start_frame,end_frame,state,smoke_onset_frame\na,1,4,smoke,1\n', encoding='utf-8')
        evaluate(path, self.root / 'out', [3], labels)
        segment = read_csv(self.root / 'out/smoke_segments.csv')[0]
        self.assertEqual(segment['delay_s'], '')
        self.assertEqual(segment['timing_status'], 'no_new_trigger')


    def test_lookahead_is_included_in_first_alarm_and_onset_delay(self):
        path = self.detections({'a': [(0, 1), (1, 1), (2, 1)]})
        rows = read_csv(path)
        for row in rows:
            row['decision_timestamp_s'] = float(row['timestamp_s']) + .2
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        labels = self.root / 'labels.csv'
        labels.write_text('video_id,start_frame,end_frame,state,smoke_onset_frame\na,0,2,smoke,0\n', encoding='utf-8')
        summary = evaluate(path, self.root / 'out', [3], labels)
        self.assertAlmostEqual(summary[0]['first_trigger_s'], .4)
        self.assertAlmostEqual(float(read_csv(self.root / 'out/smoke_segments.csv')[0]['delay_s']), .4)


if __name__ == '__main__':
    unittest.main()
