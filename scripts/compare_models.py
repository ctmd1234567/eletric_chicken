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
    *.jpg / *.mp4    可选，图片保存截图，视频保存带框 MP4（--save-images）

注意:
    本脚本只负责"跑得动、跑多快、报了什么"，不做准确率判定。
    薄烟召回率、误报率需要人工逐帧核对，评测限制见 read.md。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from download_models import sha256_of

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "firesmockdetect_y11m": ("models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt", 1280),
    "firesmockdetect_y11s": ("models/fire_smoke/firesmockdetect/expB_y11s_deployed.pt", 1280),
    "early_fire_smoke": ("models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt", 1280),
    # 兼容旧选项名：此路径现为 Tenacious Tiger v8.3.0；1280 是当前本地设置，
    # 官方训练/示例为 1024，实际值可由 --pyronear-imgsz 单独覆盖。
    "pyronear_sensitive": ("models/fire_smoke/pyronear_sensitive/best.pt", 1280),
    "dfire_yolov8n": ("models/fire_smoke/dfire_yolov8n/best.pt", 1280),
    "fasdd_yolo26n": ("models/fire_smoke/fasdd_yolo26n/best.pt", 640),
    "safety_yolo26": ("models/fire_smoke/safety_yolo26/yolo_smoke_fire.pt", 640),
}

# Pyronear 模型卡的推荐 NMS IoU；其他模型保留 Ultralytics 默认值。
NMS_IOU = {"pyronear_sensitive": 0.01}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


class MP4Writer:
    """通过 FFmpeg 写入 H.264 MP4；编码完成后才生成最终文件。"""

    def __init__(self, output: Path, fps: float, size: tuple[int, int]):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("保存兼容 MP4 需要 FFmpeg，请安装并将 ffmpeg 加入 PATH。")
        self.output = output
        self.temporary = output.with_suffix(".mp4.part")
        self.closed = False
        self.errors = tempfile.TemporaryFile()
        width, height = size
        try:
            self.process = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                 "-r", str(fps), "-i", "pipe:0", "-an",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4",
                 str(self.temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.errors.close()
            raise

    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.release()
            raise RuntimeError(f"FFmpeg 在写入视频时退出: {self.output}") from None

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            self.errors.seek(0)
            error = self.errors.read().decode("utf-8", errors="replace")
            if code != 0:
                raise RuntimeError(f"MP4 编码失败（退出码 {code}）: {error}")
            if not self.temporary.exists() or self.temporary.stat().st_size == 0:
                raise RuntimeError(f"MP4 编码未生成有效文件: {self.output}")
            self.temporary.replace(self.output)
        finally:
            self.errors.close()


def open_video(path: Path):
    """交给系统播放器播放完整视频，与图片预览分开。"""
    if sys.platform == "win32":
        os.startfile(str(path.resolve()))
    else:
        command = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([command, str(path.resolve())], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)


def collect_frames(source: Path, stride: int, video_fps: dict[str, float] | None = None):
    """返回 [(标签, 帧号或文件名, BGR 图或 None, 路径)]；视频按 stride 抽帧。"""
    import cv2

    items = []
    if source.is_dir():
        for p in sorted(source.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT | VIDEO_EXT:
                items.extend(collect_frames(p, stride, video_fps))
    elif source.suffix.lower() in IMG_EXT:
        img = cv2.imread(str(source))
        if img is not None:
            items.append((source.name, 0, img, str(source)))
        else:
            print(f"无法读取图片: {source}")
    else:
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        if video_fps is not None:
            video_fps[str(source)] = fps / stride
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
    ap.add_argument("--pyronear-imgsz", type=int, default=1024,
                    help="仅覆盖 Pyronear 输入尺寸，默认 1024")
    ap.add_argument("--pyronear-iou", type=float, default=0.01,
                    help="仅覆盖 Pyronear NMS IoU，官方示例为 0.01")
    ap.add_argument("--pyronear-rect", action=argparse.BooleanOptionalAction, default=True,
                    help="Pyronear 矩形补边；--no-pyronear-rect 使用完整方形补边")
    ap.add_argument("--stride", type=int, default=5, help="视频抽帧间隔")
    ap.add_argument("--out", type=Path, help="结果父目录；默认 samples/images_results")
    ap.add_argument("--save-images", "--save-results", dest="save_images", action="store_true",
                    help="保存结果：图片为 JPG，视频为无音轨的带框 MP4")
    ap.add_argument("--show-images", action="store_true",
                    help="图片按键继续；视频检测完成后用系统播放器打开带框 MP4")
    ap.add_argument("--device", help="推理设备，如 0（首张 GPU）或 cpu；省略时自动选择")
    args = ap.parse_args(argv)
    if args.stride < 1:
        ap.error("--stride 必须是正整数")
    if args.pyronear_imgsz < 32 or args.pyronear_imgsz % 32:
        ap.error("--pyronear-imgsz 必须是至少 32 的 32 倍数")
    if not 0 <= args.pyronear_iou <= 1:
        ap.error("--pyronear-iou 必须在 0 到 1 之间")
    if not 0 <= args.conf <= 1:
        ap.error("--conf 必须在 0 到 1 之间")

    from ultralytics import YOLO

    if not args.source.exists():
        print(f"素材不存在: {args.source}")
        return 2

    if args.out is None:
        args.out = ROOT / "samples" / "images_results"

    video_fps: dict[str, float] = {}
    frames = collect_frames(args.source, args.stride, video_fps)
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
        # 路径/选项名可跨版本复用，CSV 保存实际权重哈希以区分新旧结果。
        weights_sha256 = sha256_of(path)
        end2end = bool(getattr(model.model, "end2end", False))
        if name == "pyronear_sensitive" and model.names == {0: "item"}:
            # 上游单类烟雾 checkpoint 的名称是 item；只修正显示，不修改权重。
            model.model.names = {0: "smoke"}
        iou = NMS_IOU.get(name, 0.7)
        rect = True
        if name == "pyronear_sensitive":
            imgsz = args.pyronear_imgsz
            iou = args.pyronear_iou
            rect = args.pyronear_rect
        print(f"推理设置: {name}, imgsz={imgsz}, conf={args.conf}, iou={iou}, rect={rect}")
        if end2end:
            print("  此权重使用端到端检测头；默认预测不执行传统 NMS，iou 不参与删框。")
        times, hit = [], 0
        writers = {}
        video_paths: dict[str, Path] = {}
        preview_directory = None
        with ExitStack() as video_outputs:
            if args.show_images:
                import cv2

                video_outputs.callback(cv2.destroyAllWindows)
            for label, idx, frame, ref in frames:
                t0 = time.perf_counter()
                res = model.predict(frame, imgsz=imgsz, conf=args.conf, iou=iou,
                                    rect=rect, device=args.device, verbose=False)
                ms = (time.perf_counter() - t0) * 1000
                times.append(ms)
                r = res[0]
                confs = [float(c) for c in r.boxes.conf]
                clss = [r.names[int(c)] for c in r.boxes.cls]
                hit += 1 if confs else 0
                rows.append(
                    {
                        "model": name,
                        "weights_sha256": weights_sha256,
                        "end2end": end2end,
                        "frame": label,
                        "frame_index": idx,
                        "imgsz": imgsz,
                        "conf_threshold": args.conf,
                        "iou_threshold": iou,
                        "rect": rect,
                        "n_boxes": len(confs),
                        "max_conf": round(max(confs), 4) if confs else "",
                        "classes": "|".join(sorted(set(clss))),
                        "infer_ms": round(ms, 1),
                    }
                )
                video_source = ref.rsplit("@", 1)[0]
                is_video = video_source in video_fps
                if (args.save_images and (confs or is_video)) or args.show_images:
                    import cv2

                    plotted = r.plot()
                    if is_video:
                        if args.save_images or args.show_images:
                            if video_source not in writers:
                                if args.save_images:
                                    video_directory = args.out
                                else:
                                    # 不保存结果时，播放器仍需读取一个临时带框视频。
                                    if preview_directory is None:
                                        preview_directory = Path(tempfile.mkdtemp(prefix="chicken_preview_"))
                                    video_directory = preview_directory
                                output = video_directory / f"{name}_{Path(video_source).stem}.mp4"
                                # 同目录内允许 clip.mp4 和 clip.avi 共存，避免覆盖。
                                if output.exists() or output in video_paths.values():
                                    video_path = Path(video_source)
                                    output = video_directory / f"{name}_{video_path.stem}_{video_path.suffix[1:]}.mp4"
                                height, width = plotted.shape[:2]
                                writer = MP4Writer(output, video_fps[video_source], (width, height))
                                video_outputs.callback(writer.release)
                                writers[video_source] = writer
                                video_paths[video_source] = output
                                print(f"正在编码 H.264 视频（完成后可打开）: {output}")
                            # 无检测框的帧也保留，避免只剩有烟片段。
                            writers[video_source].write(plotted)
                    else:
                        if args.save_images and confs:
                            cv2.imwrite(str(args.out / f"{name}_{Path(label).stem}.jpg"), plotted)
                        if args.show_images:
                            title = f"{name} - {label}"
                            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                            height, width = plotted.shape[:2]
                            scale = min(1.0, 1280 / width, 800 / height)
                            cv2.resizeWindow(title, int(width * scale), int(height * scale))
                            cv2.imshow(title, plotted)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
        if args.show_images:
            for output in video_paths.values():
                try:
                    open_video(output)
                except OSError as exc:
                    print(f"无法打开系统播放器，请手动打开 {output}: {exc}")
        summary.append(
            {
                "model": name,
                "weights": rel,
                "weights_sha256": weights_sha256,
                "end2end": end2end,
                "device": str(res[0].boxes.data.device),
                "imgsz": imgsz,
                "iou_threshold": iou,
                "conf_threshold": args.conf,
                "rect": rect,
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
