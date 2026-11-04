"""Convert YOLO polygon-segmentation labels into per-instance PNG masks and MLS manifest."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_detection_labels(path: Path):
    rows = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f"Invalid detection label: {path}:{line_index + 1}")
        cls, cx, cy, bw, bh = values
        if cls not in (0, 1) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid detection box: {path}:{line_index + 1}")
        rows.append(dict(line_index=line_index, cls=int(cls), box=(cx, cy, bw, bh)))
    return rows


def _bbox_from_polygon(points):
    xs, ys = points[:, 0], points[:, 1]
    x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def _box_iou(a, b):
    def corners(box):
        cx, cy, w, h = box
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    ax1, ay1, ax2, ay2 = corners(a)
    bx1, by1, bx2, by2 = corners(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def convert(images_dir: Path, segmentation_labels_dir: Path, detection_labels_dir: Path,
            output: Path, *, smoke_class_id=0, fire_class_id=1, min_iou=0.5):
    images_dir = images_dir.resolve()
    segmentation_labels_dir = segmentation_labels_dir.resolve()
    detection_labels_dir = detection_labels_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be in [0, 1]")
    if smoke_class_id == fire_class_id:
        raise ValueError("Smoke and fire class IDs must differ")

    image_by_stem = {}
    for image in images_dir.rglob("*"):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            if image.stem in image_by_stem:
                raise ValueError(f"Duplicate image stem: {image.stem}")
            image_by_stem[image.stem] = image
    if not image_by_stem:
        raise ValueError(f"No images found under {images_dir}")

    masks_dir = output / "masks"
    masks_dir.mkdir(parents=True)
    manifest = []
    skipped_fire_only = []
    for seg_path in sorted(segmentation_labels_dir.rglob("*.txt")):
        image_path = image_by_stem.get(seg_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"No image found for segmentation label: {seg_path}")
        detection_path = detection_labels_dir / f"{seg_path.stem}.txt"
        if not detection_path.is_file():
            raise FileNotFoundError(f"No detection label found: {detection_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        height, width = image.shape[:2]
        detection_rows = _read_detection_labels(detection_path)
        available = {
            index for index, row in enumerate(detection_rows)
            if row["cls"] == smoke_class_id
        }
        smoke_count = 0
        for polygon_index, line in enumerate(seg_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) < 7 or (len(values) - 1) % 2 != 0 or not np.isfinite(values).all():
                raise ValueError(f"Invalid YOLO segmentation polygon: {seg_path}:{polygon_index + 1}")
            cls = int(values[0])
            if values[0] != cls or cls not in (smoke_class_id, fire_class_id):
                raise ValueError(f"Unexpected segmentation class: {seg_path}:{polygon_index + 1}")
            if cls == fire_class_id:
                continue
            coords = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2)
            if len(coords) < 3 or np.any(coords < 0) or np.any(coords > 1):
                raise ValueError(f"Polygon coordinates must be normalized to [0,1]: {seg_path}:{polygon_index + 1}")
            polygon_box = _bbox_from_polygon(coords)
            candidates = [
                (_box_iou(polygon_box, detection_rows[index]["box"]), index)
                for index in available
            ]
            if not candidates:
                raise ValueError(f"No unmatched smoke detection box for polygon in {seg_path}")
            best_iou, target_index = max(candidates)
            if best_iou < min_iou:
                raise ValueError(
                    f"Polygon/box match IoU {best_iou:.3f} below {min_iou:.3f} "
                    f"for {seg_path}:{polygon_index + 1}"
                )
            available.remove(target_index)
            polygon_pixels = np.rint(coords * np.array([width - 1, height - 1])).astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_pixels], 255)
            if not np.any(mask):
                raise ValueError(f"Polygon produced an empty mask: {seg_path}:{polygon_index + 1}")
            mask_name = f"{image_path.stem}__smoke_{smoke_count:02d}.png"
            mask_path = masks_dir / mask_name
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Failed to write mask: {mask_path}")
            manifest.append(dict(
                image=image_path.as_posix(),
                mask=mask_path.relative_to(output).as_posix(),
                label=detection_path.as_posix(),
                target_label_index=target_index,
                segmentation_label=seg_path.as_posix(),
                segmentation_polygon_index=polygon_index,
                polygon_box_iou=float(best_iou),
            ))
            smoke_count += 1
        if smoke_count == 0:
            skipped_fire_only.append(seg_path.as_posix())

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    summary = dict(
        status="ready" if manifest else "no_smoke_masks",
        source_images=len(image_by_stem),
        smoke_instances=len(manifest),
        fire_only_or_empty_label_files=len(skipped_fire_only),
        smoke_class_id=smoke_class_id,
        fire_class_id=fire_class_id,
        minimum_polygon_box_iou=min_iou,
        limitations=[
            "The input segmentation export must use the same class IDs as the detection labels.",
            "Each polygon is treated as one smoke instance; verify merged/fragmented smoke annotations manually.",
            "Polygon masks are approximate human annotations, not physical smoke boundaries.",
        ],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--seg-labels", type=Path, required=True,
                        help="YOLO segmentation polygon label directory")
    parser.add_argument("--det-labels", type=Path, required=True,
                        help="Matching YOLO detection bbox label directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-class-id", type=int, default=0)
    parser.add_argument("--fire-class-id", type=int, default=1)
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args(argv)
    convert(args.images, args.seg_labels, args.det_labels, args.out,
            smoke_class_id=args.smoke_class_id, fire_class_id=args.fire_class_id,
            min_iou=args.min_iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
