#!/usr/bin/env python
"""在同一批图片/视频上跑多个模型，逐帧记录检测结果与耗时，便于人工比对。

用法:
    # 一批图片
    python scripts/compare_models.py --source samples/images --conf 0.10
    # 一段视频（每 5 帧取 1 帧）
    python scripts/compare_models.py --source samples/carport.mp4 --stride 5
    # 只跑某一个模型
    python scripts/compare_models.py --source samples/images --models early_fire_smoke

产出:
    默认在 samples/images_results/ 下创建 run_001、run_002 等子目录。
    detections.csv   逐帧明细
    summary.csv      每个模型的汇总（平均延迟、有检出帧占比）
    *.jpg            可选，带框的截图（--save-images）

注意:
    本脚本只负责"跑得动、跑多快、报了什么"，不做准确率判定。
    薄烟召回率、误报率需要人工逐帧核对，评测限制见 read.md。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "firesmockdetect_y11m": ("models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt", 1280),
    "firesmockdetect_y11s": ("models/fire_smoke/firesmockdetect/expB_y11s_deployed.pt", 1280),
    "early_fire_smoke": ("models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt", 640),
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def collect_frames(source: Path, stride: int):
    """返回 [(标签, 帧号或文件名, BGR 图或 None, 路径)]；视频按 stride 抽帧。"""
    import cv2

    items = []
    if source.is_dir():
        for p in sorted(source.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT | VIDEO_EXT:
                items.extend(collect_frames(p, stride))
    elif source.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(source))
        if img is not None:
            items.append((source.name, 0, img, str(source)))
        else:
            print(f"无法读取图片: {source}")
    else:
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                items.append((f"{source.name}#{idx}", idx, frame, f"{source}@{idx / fps:.2f}s"))
            idx += 1
        cap.release()
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--stride", type=int, default=5, help="视频抽帧间隔")
    ap.add_argument("--out", type=Path, help="结果父目录；默认 samples/images_results")
    ap.add_argument("--save-images", action="store_true", help="保存带检测框的帧")
    ap.add_argument("--show-images", action="store_true", help="显示每帧检测结果，按任意键继续")
    ap.add_argument("--device", help="推理设备，如 0（首张 GPU）或 cpu；省略时自动选择")
    args = ap.parse_args(argv)
    if args.stride < 1:
        ap.error("--stride 必须是正整数")

    from ultralytics import YOLO

    if not args.source.exists():
        print(f"素材不存在: {args.source}")
        return 2

    if args.out is None:
        args.out = ROOT / "samples" / "images_results"

    frames = collect_frames(args.source, args.stride)
    if not frames:
        print("没有可处理的帧")
        return 2
    print(f"素材: {args.source}  帧数: {len(frames)}")

    args.out.mkdir(parents=True, exist_ok=True)
    parent = args.out
    numbers = [int(p.name[4:]) for p in parent.iterdir()
               if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit()]
    number = max(numbers, default=0) + 1
    while True:
        args.out = parent / f"run_{number:03d}"
        try:
            args.out.mkdir()
            break
        except FileExistsError:
            number += 1
    print(f"结果目录: {args.out.resolve()}")
    rows, summary = [], []

    for name in args.models:
        rel, imgsz = MODELS[name]
        path = ROOT / rel
        if not path.exists():
            print(f"[跳过] {name}: 权重不存在 {rel}")
            continue
        model = YOLO(str(path))
        times, hit = [], 0
        for label, idx, frame, ref in frames:
            t0 = time.perf_counter()
            res = model.predict(frame, imgsz=imgsz, conf=args.conf, device=args.device, verbose=False)
            ms = (time.perf_counter() - t0) * 1000
            times.append(ms)
            r = res[0]
            confs = [float(c) for c in r.boxes.conf]
            clss = [r.names[int(c)] for c in r.boxes.cls]
            hit += 1 if confs else 0
            rows.append(
                {
                    "model": name,
                    "frame": label,
                    "frame_index": idx,
                    "imgsz": imgsz,
                    "conf_threshold": args.conf,
                    "n_boxes": len(confs),
                    "max_conf": round(max(confs), 4) if confs else "",
                    "classes": "|".join(sorted(set(clss))),
                    "infer_ms": round(ms, 1),
                }
            )
            if args.save_images and confs or args.show_images:
                import cv2

                plotted = r.plot()
                if args.save_images and confs:
                    filename = f"{name}_{Path(label.split('#')[0]).stem}"
                    if "#" in label:
                        filename += f"_frame_{idx:06d}"
                    cv2.imwrite(str(args.out / f"{filename}.jpg"), plotted)
                if args.show_images:
                    cv2.imshow(f"{name} - {label}", plotted)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        summary.append(
            {
                "model": name,
                "weights": rel,
                "device": str(res[0].boxes.data.device),
                "imgsz": imgsz,
                "frames": len(times),
                "frames_with_detection": hit,
                "detect_rate": round(hit / len(times), 3),
                "mean_ms": round(sum(times) / len(times), 1),
                "p95_ms": round(sorted(times)[int(len(times) * 0.95) - 1], 1) if len(times) > 1 else round(times[0], 1),
                "fps_equiv": round(1000 / (sum(times) / len(times)), 1),
            }
        )
        print(f"[完成] {name}: {len(times)} 帧, 有检出 {hit} 帧, 平均 {summary[-1]['mean_ms']} ms/帧")

    if not rows:
        print("没有可运行的模型，请检查模型权重是否存在。")
        return 2

    with (args.out / "detections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n明细: {args.out / 'detections.csv'}")
    print(f"汇总: {args.out / 'summary.csv'}")
    print("\n提醒：detect_rate 只反映'有没有框'，不等于召回率；薄烟召回与误报需人工核对。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
