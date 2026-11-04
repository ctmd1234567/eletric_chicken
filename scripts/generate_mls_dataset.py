"""Generate local-MLS B3 variants from image/mask/YOLO-label triples.

Manifest JSONL fields:
  image: source image path
  mask: binary PNG mask for exactly one smoke instance
  label: YOLO detection label file for the source image
  target_label_index: zero-based index among non-empty label lines (class 0=smoke)
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from sentinel.augmentation.mls import mask_to_bbox, warp_local_instance


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yolo_label(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = []
    nonempty_indices = []
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError(f"Malformed YOLO label line: {path}:{line_index + 1}")
        cls, cx, cy, bw, bh = values
        if cls not in (0, 1) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid YOLO box: {path}:{line_index + 1}")
        parsed.append(values)
        nonempty_indices.append(line_index)
    return lines, parsed, nonempty_indices


def _mask_coverage(mask_bbox, box, image_width, image_height):
    mx1, my1, mx2, my2 = mask_bbox
    cls, cx, cy, bw, bh = box
    bx1, by1 = (cx - bw / 2) * image_width, (cy - bh / 2) * image_height
    bx2, by2 = (cx + bw / 2) * image_width, (cy + bh / 2) * image_height
    ix1, iy1 = max(mx1, bx1), max(my1, by1)
    ix2, iy2 = min(mx2, bx2), min(my2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    mask_area = max(1.0, (mx2 - mx1) * (my2 - my1))
    return intersection / mask_area


def _updated_yolo_line(cls, bbox, width, height):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    values = (cx, cy, bw, bh)
    if not all(np.isfinite(values)) or min(bw, bh) <= 0:
        raise ValueError("Warp produced an invalid detection box")
    return f"{int(cls)} " + " ".join(f"{v:.8f}" for v in values)


def generate(manifest: Path, output: Path, *, count=2, seed=20261103,
             attempts=20, roi_scale=2.0, alpha=0.07, min_mask_coverage=0.95):
    manifest = manifest.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if count < 1 or attempts < 1:
        raise ValueError("count and attempts must be positive")
    if not 0 < min_mask_coverage <= 1:
        raise ValueError("min_mask_coverage must be in (0, 1]")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Manifest is empty")

    images_dir = output / "images" / "train"
    labels_dir = output / "labels" / "train"
    masks_dir = output / "masks" / "train"
    for folder in (images_dir, labels_dir, masks_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    records = []
    failures = []
    for row_index, row in enumerate(rows):
        for field in ("image", "mask", "label", "target_label_index"):
            if field not in row:
                raise ValueError(f"Manifest row {row_index + 1} is missing {field}")
        image_path = _resolve(manifest.parent, row["image"])
        mask_path = _resolve(manifest.parent, row["mask"])
        label_path = _resolve(manifest.parent, row["label"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read source image: {image_path}")
        if mask is None:
            raise ValueError(f"Cannot read binary instance mask: {mask_path}")
        height, width = image.shape[:2]
        if mask.shape != (height, width):
            raise ValueError(f"Mask/image size mismatch: {image_path}")
        mask = (mask > 0).astype(np.uint8)
        original_bbox = mask_to_bbox(mask)
        label_lines, parsed, line_indices = _read_yolo_label(label_path)
        target_index = int(row["target_label_index"])
        if not 0 <= target_index < len(parsed):
            raise ValueError(f"target_label_index out of range in manifest row {row_index + 1}")
        target_line_index = line_indices[target_index]
        target_box = parsed[target_index]
        if int(target_box[0]) != 0:
            raise ValueError("MLS B3 currently augments smoke instances only (class 0)")
        coverage = _mask_coverage(original_bbox, target_box, width, height)
        if coverage < min_mask_coverage:
            raise ValueError(
                f"Mask bbox is not contained by target YOLO box (coverage={coverage:.3f}); "
                "check the mask/target_label_index pair"
            )

        made = 0
        for sample_index in range(count):
            last_error = None
            result = None
            for _ in range(attempts):
                try:
                    result = warp_local_instance(image, mask, rng, roi_scale=roi_scale, alpha=alpha)
                    break
                except ValueError as exc:
                    last_error = str(exc)
            if result is None:
                failures.append(dict(row=row_index, sample=sample_index, reason=last_error))
                continue

            stem = (
                f"{image_path.stem}__row{row_index:04d}"
                f"__box{target_index:02d}__mls{sample_index:02d}"
            )
            image_out = images_dir / f"{stem}.png"
            label_out = labels_dir / f"{stem}.txt"
            mask_out = masks_dir / f"{stem}.png"
            if not cv2.imwrite(str(image_out), result.image):
                raise OSError(f"Failed to write image: {image_out}")
            if not cv2.imwrite(str(mask_out), result.mask * 255):
                raise OSError(f"Failed to write mask: {mask_out}")
            updated = list(label_lines)
            updated[target_line_index] = _updated_yolo_line(
                target_box[0], result.bbox_xyxy, width, height
            )
            label_out.write_text("\n".join(updated) + "\n", encoding="utf-8")
            records.append(dict(
                image=image_out.relative_to(output).as_posix(),
                label=label_out.relative_to(output).as_posix(),
                mask=mask_out.relative_to(output).as_posix(),
                source_image=str(image_path),
                source_mask=str(mask_path),
                source_label=str(label_path),
                target_label_index=target_index,
                source_image_sha256=_sha256(image_path),
                image_sha256=_sha256(image_out),
                label_sha256=_sha256(label_out),
                mask_sha256=_sha256(mask_out),
                augmentation="local_mls_random_b3",
                seed=seed,
                roi_xyxy=list(result.roi_xyxy),
                bbox_xyxy=list(result.bbox_xyxy),
                area_ratio=result.area_ratio,
                bottom_shift_ratio=result.bottom_shift_ratio,
                control_points=result.control_points,
                alpha=alpha,
                roi_scale=roi_scale,
            ))
            made += 1
        print(f"[{row_index + 1}/{len(rows)}] {image_path.name}: generated {made}/{count}", flush=True)

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = dict(
        status="generated" if records else "no_valid_samples",
        source_manifest=str(manifest),
        source_rows=len(rows),
        requested_per_row=count,
        generated=len(records),
        failed=len(failures),
        seed=seed,
        alpha=alpha,
        roi_scale=roi_scale,
        failures=failures,
        limitations=[
            "B3 uses random sampling within the local alpha neighborhood; it does not implement DE or DASA-DE.",
            "Each mask must be a manually reviewed binary mask for exactly one smoke instance.",
            "The mask is not inferred from a detection bounding box.",
            "Generated variants are training-only and must not enter validation or test sets.",
        ],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--roi-scale", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--min-mask-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)
    generate(args.manifest, args.out, count=args.count, seed=args.seed,
             attempts=args.attempts, roi_scale=args.roi_scale, alpha=args.alpha,
             min_mask_coverage=args.min_mask_coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
