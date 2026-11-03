"""Tests for local MLS augmentation and the isolated B3 dataset workflow."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from sentinel.augmentation.mls import _mls_affine_map, mask_to_bbox, warp_local_instance
from scripts.generate_mls_dataset import generate
from scripts.import_yolo_segmentation_masks import convert as convert_segmentation
from scripts.prepare_mls_b3 import build


class LocalMLSTests(unittest.TestCase):
    def _sample(self):
        height, width = 180, 240
        image = np.full((height, width, 3), (45, 65, 85), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (118, 82), (18, 34), -12, 0, 360, 1, -1)
        cv2.circle(mask, (128, 55), 9, 1, -1)
        image[mask > 0] = (215, 220, 225)
        return image, mask

    def test_affine_mls_reproduces_affine_transform(self):
        source = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 3]], dtype=np.float32)
        matrix = np.array([[1.1, 0.2], [-0.1, 0.9]], dtype=np.float32)
        offset = np.array([3, -2], dtype=np.float32)
        target = source @ matrix.T + offset
        query = np.array([[2, 4], [6, 6], [8, 1]], dtype=np.float32)
        actual = _mls_affine_map(query, source, target)
        expected = query @ matrix.T + offset
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_mask_bbox_uses_exclusive_max_coordinates(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:9, 7:13] = 1
        self.assertEqual(mask_to_bbox(mask), (7, 4, 13, 9))

    def test_local_warp_changes_instance_and_preserves_outside_roi(self):
        image, mask = self._sample()
        result = warp_local_instance(image, mask, np.random.default_rng(7))
        self.assertNotEqual(result.bbox_xyxy, mask_to_bbox(mask))
        self.assertTrue(0.7 <= result.area_ratio <= 1.4)
        self.assertLessEqual(result.bottom_shift_ratio, 0.05)
        x1, y1, x2, y2 = result.roi_xyxy
        outside = np.ones(mask.shape, dtype=bool)
        outside[y1:y2, x1:x2] = False
        np.testing.assert_array_equal(result.image[outside], image[outside])
        np.testing.assert_array_equal(result.mask[outside], mask[outside])
        self.assertGreater(result.control_points, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        local_support = cv2.dilate(mask, kernel) > 0
        self.assertTrue(np.array_equal(result.image[~local_support], image[~local_support]))

    def test_empty_or_mismatched_mask_is_rejected(self):
        image, mask = self._sample()
        with self.assertRaisesRegex(ValueError, "empty"):
            warp_local_instance(image, np.zeros_like(mask), np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "matching image size"):
            warp_local_instance(image, np.zeros((10, 10), dtype=np.uint8))

    def test_yolo_segmentation_export_becomes_per_instance_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            seg_labels = root / "seg_labels"
            det_labels = root / "det_labels"
            for folder in (images, seg_labels, det_labels):
                folder.mkdir()
            image, mask = self._sample()
            image_path = images / "frame.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            x1, y1, x2, y2 = mask_to_bbox(mask)
            h, w = mask.shape
            points = [
                (x1 / w, y1 / h), (x2 / w, y1 / h),
                (x2 / w, y2 / h), (x1 / w, y2 / h),
            ]
            polygon = "0 " + " ".join(f"{x} {y}" for x, y in points)
            (seg_labels / "frame.txt").write_text(polygon + "\n", encoding="utf-8")
            cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            (det_labels / "frame.txt").write_text(
                f"0 {cx} {cy} {bw} {bh}\n1 .2 .2 .1 .1\n", encoding="utf-8"
            )
            output = root / "mask_import"
            summary = convert_segmentation(images, seg_labels, det_labels, output)
            self.assertEqual(summary["smoke_instances"], 1)
            record = json.loads((output / "manifest.jsonl").read_text().strip())
            self.assertEqual(record["target_label_index"], 0)
            imported_mask = cv2.imread(str(output / record["mask"]), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(imported_mask.shape, mask.shape)
            self.assertGreater(int((imported_mask > 0).sum()), 0)

    def test_generator_and_b3_config_are_isolated_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            image, mask = self._sample()
            image_path = source / "smoke.png"
            mask_path = source / "smoke_mask.png"
            label_path = source / "smoke.txt"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(mask_path), mask * 255))
            x1, y1, x2, y2 = mask_to_bbox(mask)
            h, w = mask.shape
            cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            label_path.write_text(f"0 {cx} {cy} {bw} {bh}\n1 .2 .2 .1 .1\n", encoding="utf-8")
            manifest = source / "manifest.jsonl"
            manifest.write_text(json.dumps(dict(
                image=image_path.name, mask=mask_path.name, label=label_path.name,
                target_label_index=0,
            )) + "\n", encoding="utf-8")

            generated = root / "generated"
            summary = generate(manifest, generated, count=2, seed=123)
            self.assertEqual(summary["generated"], 2)
            records = [json.loads(line) for line in (generated / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertTrue((generated / record["image"]).is_file())
                self.assertTrue((generated / record["label"]).is_file())
                lines = (generated / record["label"]).read_text().splitlines()
                self.assertEqual(len(lines), 2)
                self.assertTrue(lines[1].startswith("1 "))

            base_root = root / "base"
            (base_root / "images" / "train").mkdir(parents=True)
            (base_root / "images" / "val").mkdir(parents=True)
            (base_root / "images" / "test").mkdir(parents=True)
            base_train_image = base_root / "images" / "train" / "base.png"
            base_val_image = base_root / "images" / "val" / "val.png"
            base_test_image = base_root / "images" / "test" / "test.png"
            for path in (base_train_image, base_val_image, base_test_image):
                self.assertTrue(cv2.imwrite(str(path), image))
            for split, path in (("train", base_train_image), ("val", base_val_image), ("test", base_test_image)):
                (base_root / f"{split}.txt").write_text(path.as_posix() + "\n", encoding="utf-8")
            base_data = base_root / "data.yaml"
            base_data.write_text(yaml.safe_dump(dict(
                path=base_root.as_posix(), train="train.txt", val="val.txt", test="test.txt",
                names={0: "smoke", 1: "fire"},
            )), encoding="utf-8")

            prepared = root / "prepared_b3"
            result = build(base_data, generated, prepared)
            self.assertEqual(result["base_train_images"], 1)
            self.assertEqual(result["generated_images"], 2)
            self.assertEqual(result["final_train_entries"], 3)
            config = yaml.safe_load((prepared / "data_b3.yaml").read_text())
            self.assertEqual(len(Path(config["train"]).read_text().splitlines()), 3)
            self.assertEqual(config["val"], "val.txt")
            self.assertEqual(config["test"], "test.txt")


if __name__ == "__main__":
    unittest.main()
