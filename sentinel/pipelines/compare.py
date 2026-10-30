"""图片/视频快速对比：逐帧推理、CSV、可选标框输出。"""
import argparse
import csv
import time
from contextlib import closing, nullcontext
from pathlib import Path

from sentinel.paths import ROOT
from sentinel.models.catalog import MODELS, YOLO_MODELS
from sentinel.models.runtime import load_detector, prediction_settings, predict, detections
from sentinel.media.inputs import source_paths, inspect_video, video_frames, image_frame, VIDEO_EXTENSIONS
from sentinel.media.output import ResultOutput
from sentinel.reports import new_run, write_csv, percentile


def compare_model(name, model, settings, paths, videos, writer, out, *, conf, device, stride, save_images, show_images):
    entry = MODELS[name]
    metadata = dict(model=name, weights_sha256=entry['sha256'],
                    end2end=bool(getattr(model.model, 'end2end', False)),
                    imgsz=settings['imgsz'], conf_threshold=conf, conf_source=settings['conf_source'],
                    iou_threshold=settings['iou'], iou_source=settings['iou_source'], rect=settings['rect'])
    times, hits = [], 0
    for path in paths:
        video = videos.get(path)
        frames = video_frames(path, stride, video) if video else iter([image_frame(path)])
        # generator.close 保证推理异常时立即关闭视频文件。
        with closing(frames) if video else nullcontext(frames):
            with ResultOutput(out, name, path, save=save_images, show=show_images,
                              fps=video['fps'] / stride if video else None) as output:
                for frame in frames:
                    start = time.perf_counter()
                    result = predict(model, frame.image, settings, conf, device)
                    ms = (time.perf_counter() - start) * 1000
                    boxes = detections(result)
                    times.append(ms)
                    hits += bool(boxes)
                    writer.writerow(dict(metadata, frame=f'{path.name}#{frame.index}' if video else path.name,
                                         frame_index=frame.index, n_boxes=len(boxes),
                                         max_conf=round(max(b['conf'] for b in boxes), 4) if boxes else '',
                                         classes='|'.join(sorted({b['cls'] for b in boxes})), infer_ms=round(ms, 1)))
                    output.write(result)
    mean = sum(times) / len(times)
    return dict(metadata, weights=entry['dest'], device=str(result.boxes.data.device),
                frames=len(times), frames_with_detection=hits, detect_rate=round(hits / len(times), 3),
                mean_ms=round(mean, 1), p95_ms=round(percentile(times), 1), fps_equiv=round(1000 / mean, 1))


def run(source, models=YOLO_MODELS, *, conf=None, iou=None, device=None, stride=5,
        out=ROOT / 'samples/images_results', save_images=False, show_images=False,
        model_imgsz=None):
    if stride < 1 or (conf is not None and not 0 <= conf <= 1) or not models:
        raise ValueError('stride 必须 >=1，conf 在 [0,1] 内，并选择至少一个模型')
    paths = source_paths(Path(source))
    videos = {path: inspect_video(path) for path in paths if path.suffix.lower() in VIDEO_EXTENSIONS}
    sizes = model_imgsz or {}
    out = new_run(Path(out))
    fields = ['model', 'weights_sha256', 'end2end', 'imgsz', 'conf_threshold', 'conf_source', 'iou_threshold',
              'iou_source', 'rect', 'frame', 'frame_index', 'n_boxes', 'max_conf', 'classes', 'infer_ms']
    summaries = []
    with (out / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in models:
            if name not in YOLO_MODELS:
                raise ValueError(f'{name} 请使用 video_baseline 视频入口')
            model = load_detector(name)
            settings = prediction_settings(name, model, imgsz=sizes.get(name), conf=conf, iou=iou)
            print(f'推理设置: {name}, {settings}', flush=True)
            summary = compare_model(name, model, settings, paths, videos, writer, out,
                                    conf=settings['conf'], device=device, stride=stride,
                                    save_images=save_images, show_images=show_images)
            summaries.append(summary)
            print(f"完成 {name}: {summary['frames']} 帧, 有检出 {summary['frames_with_detection']} 帧", flush=True)
            del model
    write_csv(out / 'summary.csv', summaries)
    print(f'结果: {out}\ndetect_rate 只反映有框帧占比，不等于召回率。')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=YOLO_MODELS, default=YOLO_MODELS)
    parser.add_argument('--conf', type=float, help='覆盖所有模型阈值；省略时按各模型上游示例')
    parser.add_argument('--iou', type=float, help='所有所选模型的 NMS 阈值；省略时使用各自默认')
    parser.add_argument('--device')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--out', type=Path, default=ROOT / 'samples/images_results')
    parser.add_argument('--save-images', '--save-results', dest='save_images', action='store_true')
    parser.add_argument('--show-images', action='store_true')
    args = vars(parser.parse_args(argv))
    return run(**args)
