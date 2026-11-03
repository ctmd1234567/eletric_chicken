"""CPU-only adapter and dataset integration checks (no torch or weights)."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.generate_det_mscea import (
    FrozenYOLOFeatureEvaluator, _bbox_from_yolo, _matched_confidence,
    _new_background_confidence, _validate_source, feature_crop_bounds, generate,
)
from scripts.prepare_mls_b3 import build
from sentinel.augmentation.mls import mask_to_bbox


def det(box, confidence=.4, cls=0):
    return {"bbox": box, "confidence": confidence, "class_id": cls}


class MockDetector:
    strides = [8, 16, 32]
    closed = False

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, image):
        return [det((20, 12, 30, 28))], [np.array([.1, .2])]

    def level_for_bbox(self, bbox, image_shape):
        return 0

    def feature_for_bbox(self, maps, bbox, shape, level):
        return maps[level]

    def close(self):
        MockDetector.closed = True


class AdapterTests(unittest.TestCase):
    def test_match_uses_instance_iou_before_confidence(self):
        box = (20, 20, 40, 40)
        self.assertEqual(_matched_confidence([
            det((15, 15, 50, 50), .99), det(box, .3), det(box, .9, 1)], box), .3)
        self.assertIsNone(_matched_confidence([det(box, .9, 1)], box))

    def test_background_preserves_old_detections_but_not_duplicates(self):
        box = (80, 80, 90, 90)
        baseline = [det(box)]
        roi = (0, 0, 40, 40)
        self.assertEqual(_new_background_confidence([det(box)], baseline, (0, 0, 10, 10), roi), 0)
        self.assertEqual(_new_background_confidence([det(box), det(box, .8)], baseline,
                                                    (0, 0, 10, 10), roi), .4)
        self.assertEqual(_new_background_confidence([det(box, .8, 1)], [], box, roi), 0)
        # Target overlap alone must not suppress a new outside-ROI detection.
        self.assertEqual(_new_background_confidence([det(box, .8)], [], box, roi), .8)

    def test_letterbox_integer_padding_and_stride_aligned_shape(self):
        # 333x1000 -> resized 224x672, top=224 with actual 672 input,
        # even when the requested imgsz was 650.
        self.assertEqual(feature_crop_bounds((0, 0, 1000, 333), (333, 1000),
                                             (672, 672), (84, 84)), (0, 28, 84, 56))
        # 101x200 -> 323x640, 317 padding => top 158, bottom 159.
        self.assertEqual(feature_crop_bounds((0, 0, 200, 101), (101, 200),
                                             (640, 640), (640, 640)), (0, 158, 640, 482))
        self.assertEqual(feature_crop_bounds((-5, -5, 105, 105), (100, 100),
                                             (640, 640), (80, 80)), (0, 0, 80, 80))

    def test_scale_bucket_is_invariant_to_source_resolution(self):
        evaluator = FrozenYOLOFeatureEvaluator.__new__(FrozenYOLOFeatureEvaluator)
        evaluator.imgsz = 960
        evaluator.strides = [8, 16, 32]
        self.assertEqual(evaluator.level_for_bbox((0, 0, 160, 160), (480, 640)),
                         evaluator.level_for_bbox((0, 0, 320, 320), (960, 1280)))

    def make_source(self, root):
        image = np.full((40, 60, 3), 80, dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[12:28, 20:30] = 255
        image[mask > 0] = 200
        cv2.imwrite(str(root / "source.png"), image)
        cv2.imwrite(str(root / "mask.png"), mask)
        (root / "source.txt").write_text("0 .416666667 .5 .166666667 .4\n", encoding="utf-8")
        row = dict(image="source.png", mask="mask.png", label="source.txt", target_label_index=0)
        (root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return row

    def test_fire_reference_and_misaligned_mask_fail_before_detector_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.make_source(root)
            (root / "source.txt").write_text("1 .416666667 .5 .166666667 .4\n")
            with self.assertRaisesRegex(ValueError, "smoke class"):
                _validate_source(row, root)
            (root / "source.txt").write_text("0 .1 .1 .1 .1\n")
            with self.assertRaisesRegex(ValueError, "IoU"):
                _validate_source(row, root)
            row["target_label_index"] = .5
            with self.assertRaisesRegex(ValueError, "target_label_index"):
                _validate_source(row, root)

    def test_non_target_fire_row_cannot_enter_reference_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.make_source(root)
            (root / "fire.txt").write_text("1 .416666667 .5 .166666667 .4\n")
            fire = {**row, "label": "fire.txt"}
            (root / "manifest.jsonl").write_text(json.dumps(row)+"\n"+json.dumps(fire)+"\n")
            (root / "weights").write_bytes(b"mock")
            with patch("scripts.generate_det_mscea.FrozenYOLOFeatureEvaluator") as detector:
                with self.assertRaisesRegex(ValueError, "smoke class"):
                    generate(root / "manifest.jsonl", root / "weights", root / "out", max_instances=1)
                detector.assert_not_called()

    def test_overlapping_other_label_is_skipped_without_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_source(root)
            with (root / "source.txt").open("a") as f:
                f.write("1 .416666667 .5 .1 .2\n")
            (root / "weights").write_bytes(b"mock")
            with patch("scripts.generate_det_mscea.FrozenYOLOFeatureEvaluator", MockDetector):
                with patch("scripts.generate_det_mscea.optimize_mls_instance") as search:
                    result = generate(root / "manifest.jsonl", root / "weights", root / "out")
                    search.assert_not_called()
            self.assertEqual(result["generated"], 0)
            self.assertIn("another labeled", result["skipped"][0]["reason"])

    def test_mock_detector_generation_labels_logs_and_manual_training_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_source(root)
            weights = root / "mock_weights"
            weights.write_bytes(b"mock, never a real checkpoint")
            out = root / "generated"
            with patch("scripts.generate_det_mscea.FrozenYOLOFeatureEvaluator", MockDetector):
                summary = generate(root / "manifest.jsonl", weights, out,
                                   population_size=10, max_evals=30, top_k=1,
                                   alpha=.001, contour_points=8, max_instances=1)
            self.assertTrue(MockDetector.closed)
            self.assertEqual(summary["generated"], 1)
            record = json.loads((out / "manifest.jsonl").read_text())
            mask = cv2.imread(str(out / record["mask"]), 0)
            box = list(map(float, (out / record["label"]).read_text().split()))
            np.testing.assert_allclose(_bbox_from_yolo(box, 60, 40), mask_to_bbox(mask), atol=1e-4)
            trace = json.loads((out / "diagnostics/row_00000.json").read_text())
            self.assertEqual(len(trace["evaluation_history"]), 30)
            self.assertEqual(trace["evaluations"], 30)
            (root / "train.txt").write_text(str(root / "source.png") + "\n")
            (root / "val.txt").write_text(str(root / "val.png") + "\n")
            (root / "data.yaml").write_text("train: train.txt\nval: val.txt\nnames: [smoke, fire]\n")
            with self.assertRaisesRegex(ValueError, "human_review"):
                build(root / "data.yaml", out, root / "blocked")
            record["human_review"] = dict(status="approved", reviewer="synthetic-test", notes="mock only")
            (out / "manifest.jsonl").write_text(json.dumps(record) + "\n")
            summary = build(root / "data.yaml", out, root / "prepared")
            self.assertEqual(summary["generated_images"], 1)
            (root / "val.txt").write_text(str(root / "source.png") + "\n")
            with self.assertRaisesRegex(ValueError, "exclusively"):
                build(root / "data.yaml", out, root / "heldout_blocked")
            (root / "val.txt").write_text(str(root / "val.png") + "\n")
            bank = out / record["reference_bank"]
            original_bank = bank.read_bytes()
            bank.write_text(json.dumps([dict(image=str(root / "val.png"), image_sha256="unused")]))
            record["reference_bank_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest()
            (out / "manifest.jsonl").write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "reference bank must use training"):
                build(root / "data.yaml", out, root / "heldout_reference_blocked")
            bank.write_bytes(original_bank)
            record["reference_bank_sha256"] = hashlib.sha256(original_bank).hexdigest()
            (out / "manifest.jsonl").write_text(json.dumps(record) + "\n")
            (out / record["label"]).write_text("0 .5 .5 .1 .1\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build(root / "data.yaml", out, root / "changed_after_review")


if __name__ == "__main__":
    unittest.main()
