#!/usr/bin/env python
"""流式视频基线；逐帧保存烟/火框，再离线比较连续帧规则。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import tempfile
import time
from pathlib import Path

from compare_models import VIDEO_EXT, MP4Writer, open_video
from video_detectors import MODELS, NMS_IOU, load_detector
from download_models import MANIFEST, sha256_of
from verify_models import NATIVE_IMGSZ

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = list(MODELS)[:5]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='视频或包含视频的目录（不递归）')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=DEFAULT_MODELS)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.10)
    parser.add_argument('--model-imgsz', nargs='+', default=[], metavar='模型=尺寸',
                        help='每个所选模型的输入尺寸，例如 dfire_yolov8n=1280')
    parser.add_argument('--pyronear-rect', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--consecutive', nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--out', type=Path, default=ROOT / 'samples/video_results')
    parser.add_argument('--save-images', action='store_true', help='保存带检测框的视频')
    parser.add_argument('--show-images', action='store_true', help='完成后用系统播放器打开视频')
    args = parser.parse_args(argv)
    if args.stride < 1 or args.warmup < 0 or not 0 <= args.conf <= 1:
        parser.error('stride 必须 >=1、warmup >=0、conf 在 [0,1] 内')
    if any(n < 1 for n in args.consecutive):
        parser.error('连续帧数必须 >=1')
    sizes = {}
    for item in args.model_imgsz:
        name, separator, size = item.partition('=')
        if not separator or name not in args.models or name in sizes or not size.isdigit():
            parser.error(f'输入尺寸格式无效、模型未选择或重复: {item}')
        value = int(size)
        if value < 32 or value % 32:
            parser.error(f'输入尺寸必须是至少 32 的 32 倍数: {item}')
        sizes[name] = value
    source = args.source.resolve()
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_EXT) if source.is_dir() else [source]
    if not paths or any(not p.is_file() or p.suffix.lower() not in VIDEO_EXT for p in paths):
        parser.error('没有可读取的视频')
    if len(set(args.models)) != len(args.models):
        parser.error('模型不能重复')

    import cv2
    import torch
    import ultralytics

    videos = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            ok, _ = cap.read()
            if not ok or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f'无法解码视频或 FPS 无效: {path.name}')
            videos.append(dict(video_id=path.name, sha256=sha256_of(path), fps=fps,
                               declared_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                               width=int(cap.get(3)), height=int(cap.get(4))))
        finally:
            cap.release()

    model_records = []
    for name in args.models:
        rel, _ = MODELS[name]
        entry = next(e for e in MANIFEST if e['dest'] == rel)
        path = ROOT / rel
        digest = sha256_of(path)
        if digest != entry['sha256'] or path.stat().st_size != entry['size']:
            raise ValueError(f'权重校验失败: {name}')
        model_records.append(dict(model=name, weights=rel, weights_sha256=digest,
                                  revision=entry['revision'],
                                  imgsz=sizes.get(name, NATIVE_IMGSZ.get(rel, NATIVE_IMGSZ.get(path.name))),
                                  iou=NMS_IOU.get(name, 0.7),
                                  rect=False if name == 'lwst_fsd' else args.pyronear_rect if name == 'pyronear_sensitive' else True,
                                  temporal_frames=5 if name == 'lwst_fsd' else 1))
        settings = model_records[-1]
        if name == 'lwst_fsd':
            from lwst_sources import REVISION, SOURCE_HASHES
            settings['source_revision'] = REVISION
            settings['source_sha256'] = SOURCE_HASHES
            settings['context_offsets'] = [-2, -1, 0, 1, 2]
        print(f"推理设置: {name}, imgsz={settings['imgsz']}, conf={args.conf}, "
              f"iou={settings['iou']}, rect={settings['rect']}", flush=True)
        print(f"权重: {path.resolve()}\nSHA-256: {digest}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        out = args.out / f'run_{number:03d}'
        try:
            out.mkdir()
            break
        except FileExistsError:
            number += 1
    manifest = dict(status='running', conf=args.conf, stride=args.stride, warmup=args.warmup,
                    time_basis='frame_index / nominal_fps; CFR assumed; not wall-clock alarm latency',
                    evaluation_scope='local third-party video demonstration; no verified carport ground truth',
                    versions=dict(python=platform.python_version(), torch=torch.__version__,
                                  ultralytics=ultralytics.__version__, opencv=cv2.__version__),
                    device_requested=args.device, models=model_records, videos=videos,
                    save_video=args.save_images, show_video=args.show_images)
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'结果: {out}', flush=True)
    summaries = []
    counts = {}
    video_outputs = []
    preview_dir = None
    fields = ['model', 'video_id', 'frame_index', 'timestamp_s', 'fps', 'stride',
              'decision_timestamp_s', 'available_frame_index', 'support_frame_indices',
              'smoke_boxes', 'fire_boxes', 'max_smoke_conf', 'detections_json', 'infer_ms', 'read_ms']
    try:
        with (out / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for settings in model_records:
                model = load_detector(settings['model'], ROOT / settings['weights'])
                if settings['model'] == 'pyronear_sensitive' and model.names == {0: 'item'}:
                    model.model.names = {0: 'smoke'}
                settings['classes'] = dict(model.names)
                settings['end2end'] = bool(getattr(model.model, 'end2end', False))
                print(f"已加载: {settings['model']}, 类别={model.names}, "
                      f"参数量={sum(p.numel() for p in model.model.parameters()):,}", flush=True)
                for path, video in zip(paths, videos):
                    if settings['temporal_frames'] > 1:
                        model.start_video(path)
                    cap = cv2.VideoCapture(str(path))
                    times, reads, cold_times = [], [], []
                    index, hits = 0, 0
                    video_writer = None
                    try:
                        while True:
                            start = time.perf_counter()
                            ok, frame = cap.read()
                            read_ms = (time.perf_counter() - start) * 1000
                            if not ok:
                                break
                            current = index
                            index += 1
                            if current % args.stride:
                                continue

                            def predict():
                                options = {'frame_index': current} if settings['temporal_frames'] > 1 else {}
                                return model.predict(frame, imgsz=settings['imgsz'], conf=args.conf,
                                                     iou=settings['iou'], rect=settings['rect'],
                                                     device=args.device, verbose=False, **options)[0]

                            def sync():
                                device = next(model.model.parameters()).device
                                if device.type == 'cuda':
                                    torch.cuda.synchronize(device)

                            if not times:
                                for _ in range(args.warmup):
                                    sync()
                                    start = time.perf_counter()
                                    predict()
                                    sync()
                                    cold_times.append((time.perf_counter() - start) * 1000)
                            sync()
                            start = time.perf_counter()
                            result = predict()
                            sync()
                            ms = (time.perf_counter() - start) * 1000
                            times.append(ms)
                            reads.append(read_ms)
                            boxes = [dict(cls=result.names[int(cls)], conf=float(conf), xyxy=xyxy)
                                     for cls, conf, xyxy in zip(result.boxes.cls.cpu().tolist(),
                                                               result.boxes.conf.cpu().tolist(),
                                                               result.boxes.xyxy.cpu().tolist())]
                            smoke = [b for b in boxes if b['cls'] == 'smoke']
                            hits += bool(smoke)
                            writer.writerow(dict(model=settings['model'], video_id=video['video_id'],
                                                 frame_index=current, timestamp_s=current / video['fps'],
                                                 decision_timestamp_s=getattr(model, 'available_frame_index', current) / video['fps'],
                                                 available_frame_index=getattr(model, 'available_frame_index', current),
                                                 support_frame_indices=json.dumps(getattr(model, 'support_indices', [current])),
                                                 fps=video['fps'], stride=args.stride, smoke_boxes=len(smoke),
                                                 fire_boxes=sum(b['cls'] == 'fire' for b in boxes),
                                                 max_smoke_conf=max((b['conf'] for b in smoke), default=''),
                                                 detections_json=json.dumps(boxes), infer_ms=ms, read_ms=read_ms))
                            if args.save_images or args.show_images:
                                plotted = result.plot()
                                cv2.putText(plotted, f"{settings['model']} | imgsz={settings['imgsz']} | "
                                            f"sha={settings['weights_sha256'][:12]}",
                                            (12, max(24, plotted.shape[0] - 18)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                                if video_writer is None:
                                    if args.save_images:
                                        video_dir = out
                                    else:
                                        if preview_dir is None:
                                            preview_dir = Path(tempfile.mkdtemp(prefix='chicken_baseline_preview_'))
                                        video_dir = preview_dir
                                    same_stem = sum(p.stem == path.stem for p in paths) > 1
                                    suffix = f'_{path.suffix[1:]}' if same_stem else ''
                                    video_path = video_dir / f"{settings['model']}_{path.stem}{suffix}.mp4"
                                    height, width = plotted.shape[:2]
                                    video_writer = MP4Writer(video_path, video['fps'] / args.stride, (width, height))
                                    video_outputs.append(video_path)
                                    print(f'正在编码带框视频: {video_path}', flush=True)
                                video_writer.write(plotted)
                            if len(times) % 300 == 0:
                                print(f"{settings['model']} / {path.name}: {len(times)} 帧", flush=True)
                    finally:
                        cap.release()
                        if settings['temporal_frames'] > 1:
                            model.close_video()
                        if video_writer is not None:
                            video_writer.release()
                    if not times:
                        raise ValueError(f'视频没有推理帧: {path.name}')
                    if path.name in counts and counts[path.name] != index:
                        raise ValueError(f'不同模型读取的帧数不一致: {path.name}')
                    counts[path.name] = index
                    # 对已声明长度的视频，提早解码结束视为失败，避免把截断结果当完整实验。
                    if video['declared_frames'] > 0 and index != video['declared_frames']:
                        raise ValueError(f'视频声明 {video["declared_frames"]} 帧，实际解码 {index} 帧: {path.name}')
                    summaries.append(dict(model=settings['model'], video_id=path.name,
                                          weights_sha256=settings['weights_sha256'], imgsz=settings['imgsz'],
                                          iou=settings['iou'], conf=args.conf, rect=settings['rect'],
                                          device=str(result.boxes.data.device), frames=len(times),
                                          smoke_hit_frames=hits, smoke_hit_rate=hits / len(times),
                                          warmup_ms_json=json.dumps(cold_times),
                                          mean_ms=sum(times) / len(times), p95_ms=percentile(times, .95),
                                          mean_read_ms=sum(reads) / len(reads)))
                    stream.flush()
                    print(f"完成 {settings['model']} / {path.name}: {len(times)} 帧", flush=True)
                del model
        write_csv(out / 'summary.csv', summaries)
        write_csv(out / 'annotations_template.csv', [dict(video_id=v['video_id'], start_frame=0,
                  end_frame=counts[v['video_id']] - 1, state='unknown', interference='', smoke_onset_frame='')
                  for v in videos])
        from evaluate_temporal import evaluate
        evaluate(out / 'detections.csv', out / 'temporal', args.consecutive)
        manifest['status'] = 'complete'
        manifest['decoded_frames'] = counts
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.show_images:
        for video_path in video_outputs:
            try:
                open_video(video_path)
            except OSError as exc:
                print(f'无法打开播放器，请手动打开 {video_path}: {exc}', flush=True)
    print('完成；未标注视频的触发时间不代表预警延迟，检出占比不代表召回率。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
