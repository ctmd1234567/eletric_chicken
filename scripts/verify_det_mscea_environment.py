"""Inspect pilot prerequisites and real YOLO features without inventing masks.

This verifies the adapter only. Missing reviewed masks block augmentation,
sample review, and training even when a real detector forward succeeds.
"""
import argparse
import gc
import json
import platform
from pathlib import Path
import time

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

import cv2
import numpy as np

from scripts.generate_det_mscea import FrozenYOLOFeatureEvaluator, _sha256


def verify(data_root, weights, image_path, output, *, repeats=30, imgsz=960, device="0"):
    import torch
    import ultralytics

    if repeats < 2:
        raise ValueError("repeats must be >=2")
    if output.exists():
        raise ValueError("choose a new output file")
    if not weights.is_file() or not image_path.is_file():
        raise FileNotFoundError("weights or real source image missing")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; real GPU verification stopped")
    inventory = []
    mask_manifests = []
    for folder in ("datasets", "samples", "outputs", "data"):
        base = data_root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*manifest*.json*")):
            if not path.is_file():
                continue
            keys = []
            mask_rows = 0
            try:
                if path.suffix == ".jsonl":
                    for line in path.read_text(encoding="utf-8-sig").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if isinstance(row, dict):
                            keys = sorted(row)
                            mask_rows += int({"image", "mask", "label", "target_label_index"} <= row.keys())
                else:
                    obj = json.loads(path.read_text(encoding="utf-8-sig"))
                    keys = sorted(obj) if isinstance(obj, dict) else ["JSON array"]
            except (ValueError, OSError):
                keys = ["unreadable manifest"]
            name = path.relative_to(data_root).as_posix()
            inventory.append(dict(path=name, keys=keys, instance_mask_rows=mask_rows))
            if mask_rows:
                mask_manifests.append(name)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("real image cannot be decoded")
    evaluator = FrozenYOLOFeatureEvaluator(weights, imgsz=imgsz, device=device)
    forwards = []
    try:
        import psutil
        process = psutil.Process()
        rss = lambda: process.memory_info().rss
    except ImportError:
        rss = lambda: None
    try:
        for index in range(repeats):
            started = time.perf_counter()
            detections, maps = evaluator.predict(image)
            if device != "cpu":
                torch.cuda.synchronize()
            boxes = [d["bbox"] for d in detections if d["class_id"] == 0]
            box = boxes[0] if boxes else (0, 0, image.shape[1], image.shape[0])
            level = evaluator.level_for_bbox(box, image.shape)
            vector = evaluator.feature_for_bbox(maps, box, image.shape, level)
            stats = [dict(shape=list(m.shape), finite=bool(torch.isfinite(m).all().item()),
                          minimum=float(m.min().item()), maximum=float(m.max().item())) for m in maps]
            if not all(s["finite"] for s in stats):
                raise RuntimeError("real feature maps contain non-finite values")
            del maps
            gc.collect()
            forwards.append(dict(index=index+1, seconds=time.perf_counter()-started,
                                 input_shape=list(evaluator.input_shape), feature_maps=stats,
                                 selected_level=level, feature_dimension=len(vector),
                                 feature_finite=bool(np.isfinite(vector).all()),
                                 detection_count=len(detections), rss_bytes=rss(),
                                 cuda_allocated_bytes=(torch.cuda.memory_allocated() if device != "cpu" else None),
                                 cuda_reserved_bytes=(torch.cuda.memory_reserved() if device != "cpu" else None),
                                 hook_released=evaluator.feature_maps is None))
        names = evaluator.model.names
    finally:
        evaluator.close()
    result = dict(
        adapter_status="real_forward_verified",
        augmentation_pilot_status="blocked_missing_reviewed_instance_masks" if not mask_manifests else "requires_manifest_review",
        generated_samples=0, reviewed_samples=0, formal_training_runs=0,
        instance_mask_manifests=mask_manifests, manifest_inventory=inventory,
        environment=dict(python=platform.python_version(), platform=platform.system(),
                         torch=torch.__version__, ultralytics=ultralytics.__version__,
                         numpy=np.__version__, opencv=cv2.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if device != "cpu" else None),
        weights_sha256=_sha256(weights), source_sha256=_sha256(image_path),
        source_image_shape=list(image.shape), class_names=names, imgsz=imgsz, device=device,
        repeated_forwards=forwards,
        limitations=["Adapter forward verification is not a real-mask optimization pilot.",
                     "No generated sample or training result is produced by this command.",
                     "Short repeated-forward memory observations do not prove absence of all leaks."])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("adapter_status", "augmentation_pilot_status", "environment")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    verify(args.data_root, args.weights, args.image, args.out,
           repeats=args.repeats, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
