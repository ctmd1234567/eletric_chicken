"""Generate Det-MSCEA samples from a YOLO-segmentation mask manifest.

This is an expensive offline search: each candidate invokes the frozen detector.
Use --max-instances for a pilot before running a full manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from scripts.generate_mls_dataset import _read_yolo_label, _updated_yolo_line
from sentinel.augmentation.mls import mask_to_bbox
from sentinel.augmentation.mscea import optimize_mls_instance


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(
        0.0, (bx2 - bx1) * (by2 - by1)
    ) - inter
    return inter / union if union > 0 else 0.0


def _bbox_from_yolo(box, width, height):
    _, cx, cy, bw, bh = box
    return (
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    )


def _matched_confidence(detections, bbox, smoke_class_id=0, match_iou=0.10):
    scores = [
        det["confidence"] for det in detections
        if det["class_id"] == smoke_class_id and _box_iou(det["bbox"], bbox) >= match_iou
    ]
    return max(scores) if scores else None


def _new_background_confidence(
    candidate_detections,
    baseline_detections,
    target_bbox,
    roi_xyxy,
    *,
    smoke_class_id=0,
    baseline_iou=0.50,
):
    rx1, ry1, rx2, ry2 = roi_xyxy
    baseline_boxes = [
        det["bbox"] for det in baseline_detections
        if det["class_id"] == smoke_class_id
    ]
    new_scores = []
    for det in candidate_detections:
        if det["class_id"] != smoke_class_id:
            continue
        box = det["bbox"]
        if _box_iou(box, target_bbox) >= 0.10:
            continue
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            continue
        if any(_box_iou(box, old) >= baseline_iou for old in baseline_boxes):
            continue
        new_scores.append(det["confidence"])
    return max(new_scores, default=0.0)


class FrozenYOLOFeatureEvaluator:
    """YOLO predictor plus Detect-head input feature capture."""

    def __init__(self, weights: Path, *, imgsz=960, conf=0.001, iou=0.7, device="0"):
        self.model = YOLO(str(weights))
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.feature_maps = None

        detector = self.model.model
        layers = getattr(detector, "model", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError("Unsupported Ultralytics model: Detect head not found")
        self.detect_head = layers[-1]
        self.hook = self.detect_head.register_forward_pre_hook(self._capture_features)
        stride = getattr(self.detect_head, "stride", None)
        if stride is None:
            self.strides = [8, 16, 32]
        elif torch.is_tensor(stride):
            self.strides = [int(x) for x in stride.detach().cpu().tolist()]
        else:
            self.strides = [int(x) for x in stride]

    def _capture_features(self, module, inputs):
        if not inputs:
            self.feature_maps = None
            return
        values = inputs[0]
        if isinstance(values, (list, tuple)):
            self.feature_maps = [value.detach() for value in values]
        elif torch.is_tensor(values):
            self.feature_maps = [values.detach()]
        else:
            self.feature_maps = None

    def predict(self, image):
        self.feature_maps = None
        with torch.inference_mode():
            results = self.model.predict(
                source=image,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                rect=False,
                half=False,
                verbose=False,
                save=False,
                augment=False,
            )
        if not results:
            raise RuntimeError("YOLO returned no result for a single image")
        if self.feature_maps is None:
            raise RuntimeError("Detect-head feature hook did not capture feature maps")
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            detections = []
        else:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            detections = [
                {"bbox": tuple(map(float, box)), "confidence": float(score), "class_id": int(class_id)}
                for box, score, class_id in zip(xyxy, conf, cls)
            ]
        maps = [feature.detach() for feature in self.feature_maps]
        return detections, maps

    def level_for_bbox(self, bbox):
        width = max(1.0, float(bbox[2] - bbox[0]))
        height = max(1.0, float(bbox[3] - bbox[1]))
        target_stride = max(width, height) / 8.0
        return min(
            range(len(self.strides)),
            key=lambda i: abs(np.log(max(1e-6, self.strides[i] / target_stride))),
        )

    def feature_for_bbox(self, feature_maps, bbox, image_shape, level):
        if not 0 <= level < len(feature_maps):
            raise ValueError("requested feature level is unavailable")
        image_height, image_width = image_shape[:2]
        ratio = min(self.imgsz / image_width, self.imgsz / image_height)
        resized_width = image_width * ratio
        resized_height = image_height * ratio
        pad_x = (self.imgsz - resized_width) / 2.0
        pad_y = (self.imgsz - resized_height) / 2.0

        feature = feature_maps[level]
        if feature.ndim != 4 or feature.shape[0] != 1:
            raise ValueError("expected a single-image BCHW feature map")
        _, _, feature_height, feature_width = feature.shape
        sx, sy = feature_width / self.imgsz, feature_height / self.imgsz
        x1 = int(np.floor((bbox[0] * ratio + pad_x) * sx))
        y1 = int(np.floor((bbox[1] * ratio + pad_y) * sy))
        x2 = int(np.ceil((bbox[2] * ratio + pad_x) * sx))
        y2 = int(np.ceil((bbox[3] * ratio + pad_y) * sy))
        x1, x2 = max(0, x1), min(feature_width, max(x1 + 1, x2))
        y1, y2 = max(0, y1), min(feature_height, max(y1 + 1, y2))
        crop = feature[:, :, y1:y2, x1:x2]
        if crop.numel() == 0:
            raise ValueError("instance ROI maps to an empty feature crop")
        pooled = F.adaptive_avg_pool2d(crop.float(), output_size=(1, 1))
        vector = pooled.flatten().detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("non-finite detector feature vector")
        return vector

    def close(self):
        self.hook.remove()


def _load_manifest(path: Path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("manifest is empty")
    required = {"image", "mask", "label", "target_label_index"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"manifest row {index + 1} missing fields: {sorted(missing)}")
    return rows


def generate(
    manifest: Path,
    weights: Path,
    output: Path,
    *,
    imgsz=960,
    conf=0.001,
    iou=0.7,
    device="0",
    max_instances=10,
    max_reference_instances=256,
    population_size=30,
    max_evals=300,
    correlation_evals=None,
    alpha=0.07,
    roi_scale=2.0,
    contour_points=32,
    seed=20261103,
    top_k=3,
):
    manifest = manifest.resolve()
    weights = weights.resolve()
    output = output.resolve()
    if not manifest.is_file() or not weights.is_file():
        raise FileNotFoundError("manifest or model weights do not exist")
    if output.exists():
        raise ValueError(f"output already exists; choose a new directory: {output}")
    if max_instances < 1 or max_reference_instances < 1:
        raise ValueError("max_instances and max_reference_instances must be positive")

    rows = _load_manifest(manifest)
    target_indices = list(range(min(max_instances, len(rows))))
    level_by_row = {}
    bbox_by_row = {}
    mask_by_row = {}
    label_by_row = {}
    image_by_row = {}
    for index, row in enumerate(rows):
        image_path = _resolve(manifest.parent, row["image"])
        mask_path = _resolve(manifest.parent, row["mask"])
        label_path = _resolve(manifest.parent, row["label"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"cannot read instance mask: {mask_path}")
        bbox = mask_to_bbox((mask > 0).astype(np.uint8))
        image_by_row[index] = image_path
        mask_by_row[index] = mask_path
        label_by_row[index] = label_path
        bbox_by_row[index] = bbox

    evaluator = FrozenYOLOFeatureEvaluator(
        weights, imgsz=imgsz, conf=conf, iou=iou, device=device
    )
    rng = np.random.default_rng(seed)
    try:
        # Select a deterministic, size-matched reference bank for each feature level.
        buckets = defaultdict(list)
        for index in range(len(rows)):
            buckets[evaluator.level_for_bbox(bbox_by_row[index])].append(index)
        reference_indices = set(target_indices)
        for level, indices in buckets.items():
            if len(indices) > max_reference_instances:
                selected = rng.choice(indices, size=max_reference_instances, replace=False)
                reference_indices.update(int(x) for x in selected)
            else:
                reference_indices.update(indices)

        needed_by_image = defaultdict(list)
        for index in reference_indices:
            needed_by_image[image_by_row[index]].append(index)

        feature_by_row = {}
        confidence_by_row = {}
        detections_by_row = {}
        for image_path, indices in needed_by_image.items():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot read source image: {image_path}")
            detections, feature_maps = evaluator.predict(image)
            for index in indices:
                bbox = bbox_by_row[index]
                level = evaluator.level_for_bbox(bbox)
                feature_by_row[index] = evaluator.feature_for_bbox(
                    feature_maps, bbox, image.shape, level
                )
                confidence_by_row[index] = _matched_confidence(detections, bbox)
                if index in target_indices:
                    detections_by_row[index] = detections
            del feature_maps

        feature_banks = {}
        for level in range(len(evaluator.strides)):
            vectors = [
                feature_by_row[index]
                for index in sorted(reference_indices)
                if evaluator.level_for_bbox(bbox_by_row[index]) == level
                and index in feature_by_row
            ]
            if vectors:
                feature_banks[level] = np.vstack(vectors)

        if not feature_banks:
            raise RuntimeError("no reference feature vectors were extracted")

        images_dir = output / "images" / "train"
        labels_dir = output / "labels" / "train"
        masks_dir = output / "masks" / "train"
        for folder in (images_dir, labels_dir, masks_dir):
            folder.mkdir(parents=True, exist_ok=True)

        records, skipped = [], []
        for order, index in enumerate(target_indices, start=1):
            original_conf = confidence_by_row.get(index)
            level = evaluator.level_for_bbox(bbox_by_row[index])
            if original_conf is None:
                skipped.append({
                    "manifest_row": index,
                    "reason": "no baseline smoke detection matched the source mask bbox",
                })
                continue
            if level not in feature_banks:
                skipped.append({"manifest_row": index, "reason": "empty feature-size bucket"})
                continue

            image_path = image_by_row[index]
            mask_path = mask_by_row[index]
            label_path = label_by_row[index]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                skipped.append({"manifest_row": index, "reason": "source image/mask unreadable"})
                continue
            label_lines, parsed, nonempty_indices = _read_yolo_label(label_path)
            target_index = int(rows[index]["target_label_index"])
            if not 0 <= target_index < len(parsed):
                raise ValueError(f"target_label_index out of range in row {index + 1}")
            target_box = parsed[target_index]
            if int(target_box[0]) != 0:
                skipped.append({"manifest_row": index, "reason": "target is not smoke class 0"})
                continue
            baseline_detections = detections_by_row[index]

            def evaluate_candidate(warped_image, warped_mask, bbox, roi):
                candidate_detections, candidate_maps = evaluator.predict(warped_image)
                feature = evaluator.feature_for_bbox(
                    candidate_maps, bbox, warped_image.shape, level
                )
                candidate_conf = _matched_confidence(candidate_detections, bbox)
                if candidate_conf is None:
                    candidate_conf = 0.0
                background_conf = _new_background_confidence(
                    candidate_detections, baseline_detections, bbox, roi
                )
                return {
                    "feature": feature,
                    "confidence": candidate_conf,
                    "background_confidence": background_conf,
                }

            result = optimize_mls_instance(
                image,
                mask,
                feature_banks[level],
                evaluate_candidate,
                original_confidence=float(original_conf),
                population_size=population_size,
                max_evals=max_evals,
                correlation_evals=correlation_evals,
                alpha=alpha,
                roi_scale=roi_scale,
                contour_points=contour_points,
                seed=seed + index,
                top_k=top_k,
            )
            if not result.candidates:
                skipped.append({
                    "manifest_row": index,
                    "reason": "no feasible candidate found",
                    "evaluations": result.optimizer.evaluations,
                    "allocation": result.optimizer.allocation,
                    "comparison": result.optimizer.comparison,
                })
                continue

            target_line_index = nonempty_indices[target_index]
            for candidate_index, candidate in enumerate(result.candidates):
                stem = f"{image_path.stem}__row{index:05d}__det_mscea_{candidate_index:02d}"
                image_out = images_dir / f"{stem}.png"
                mask_out = masks_dir / f"{stem}.png"
                label_out = labels_dir / f"{stem}.txt"
                if not cv2.imwrite(str(image_out), candidate.warp.image):
                    raise OSError(f"failed to write {image_out}")
                if not cv2.imwrite(str(mask_out), candidate.warp.mask * 255):
                    raise OSError(f"failed to write {mask_out}")
                updated_lines = list(label_lines)
                updated_lines[target_line_index] = _updated_yolo_line(
                    target_box[0], candidate.warp.bbox_xyxy, image.shape[1], image.shape[0]
                )
                label_out.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
                records.append({
                    "image": image_out.relative_to(output).as_posix(),
                    "label": label_out.relative_to(output).as_posix(),
                    "mask": mask_out.relative_to(output).as_posix(),
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "source_label": str(label_path),
                    "manifest_row": index,
                    "target_label_index": target_index,
                    "feature_level": level,
                    "original_confidence": float(original_conf),
                    "candidate_confidence": candidate.confidence,
                    "background_confidence": candidate.background_confidence,
                    "feature_diversity": candidate.diversity,
                    "constraint_components": candidate.constraint_components,
                    "mask_area_ratio": candidate.warp.area_ratio,
                    "bbox_area_ratio": candidate.warp.bbox_area_ratio,
                    "bottom_shift_ratio": candidate.warp.bottom_shift_ratio,
                    "roi_xyxy": list(candidate.warp.roi_xyxy),
                    "bbox_xyxy": list(candidate.warp.bbox_xyxy),
                    "optimizer": {
                        "evaluations": result.optimizer.evaluations,
                        "kappa": result.optimizer.kappa,
                        "allocation": result.optimizer.allocation,
                        "comparison": result.optimizer.comparison,
                        "correlation_objective": result.optimizer.correlation_objective,
                        "correlation_constraint": result.optimizer.correlation_constraint,
                        "expanded_region_boxes": result.expanded_region_boxes,
                    },
                    "seed": seed + index,
                    "alpha": alpha,
                    "roi_scale": roi_scale,
                })
            print(
                f"[{order}/{len(target_indices)}] {image_path.name}: "
                f"{len(result.candidates)} feasible variants",
                flush=True,
            )

        (output / "manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        summary = {
            "status": "complete" if records else "no_feasible_samples",
            "source_manifest": str(manifest),
            "weights": str(weights),
            "weights_sha256": _sha256(weights),
            "source_rows": len(rows),
            "target_instances": len(target_indices),
            "reference_instances": len(reference_indices),
            "generated": len(records),
            "skipped": skipped,
            "settings": {
                "imgsz": imgsz, "conf": conf, "iou": iou, "device": device,
                "population_size": population_size, "max_evals": max_evals,
                "correlation_evals": correlation_evals, "alpha": alpha,
                "roi_scale": roi_scale, "contour_points": contour_points,
                "seed": seed, "top_k": top_k,
            },
            "limitations": [
                "This is a detection-task adaptation, not a reproduction of the classification experiments.",
                "The objective uses pooled YOLO neck features grouped by instance scale.",
                "The C1 confidence constraint is relative to the frozen detector's source-instance confidence.",
                "C2 rejects newly introduced smoke detections outside the local ROI.",
                "The current implementation processes one candidate per detector forward pass.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        evaluator.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-instances", type=int, default=10,
                        help="pilot limit; increase only after reviewing the first outputs")
    parser.add_argument("--max-reference-instances", type=int, default=256)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--correlation-evals", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--roi-scale", type=float, default=2.0)
    parser.add_argument("--contour-points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    generate(
        args.manifest, args.weights, args.out,
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device,
        max_instances=args.max_instances,
        max_reference_instances=args.max_reference_instances,
        population_size=args.population_size, max_evals=args.max_evals,
        correlation_evals=args.correlation_evals, alpha=args.alpha,
        roi_scale=args.roi_scale, contour_points=args.contour_points,
        seed=args.seed, top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
