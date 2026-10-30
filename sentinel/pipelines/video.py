"""视频基线：记录真实采样帧、模型配置、检测框和耗时，再执行离线告警评估。"""
import argparse
import csv
import json
import platform
import time
from contextlib import closing
from pathlib import Path

from sentinel.paths import ROOT
from sentinel.models.catalog import MODELS, DEFAULT_VIDEO_MODELS
from sentinel.models.runtime import load_detector, prediction_settings, predict, detections
from sentinel.media.inputs import source_paths, inspect_video, video_frames
from sentinel.media.output import ResultOutput
from sentinel.reports import new_run, write_csv, percentile
from sentinel.evaluation.temporal import evaluate

FIELDS = ['model', 'video_id', 'frame_index', 'timestamp_s', 'fps', 'stride',
          'decision_timestamp_s', 'available_frame_index', 'support_frame_indices',
          'smoke_boxes', 'fire_boxes', 'max_smoke_conf', 'detections_json', 'infer_ms', 'read_ms']


def timed_prediction(model, frame, settings, conf, device):
    import torch

    def synchronize():
        current_device = next(model.model.parameters()).device
        if current_device.type == 'cuda':
            torch.cuda.synchronize(current_device)

    synchronize()
    start = time.perf_counter()
    result = predict(model, frame.image, settings, conf, device,
                     frame.index if settings['temporal_frames'] > 1 else None)
    synchronize()
    return result, (time.perf_counter() - start) * 1000


def process_video(model, settings, path, video, writer, out, *, conf, device, stride, warmup, save_images, show_images):
    times, reads, cold_times = [], [], []
    hits = 0
    temporal = settings['temporal_frames'] > 1
    if temporal:
        model.start_video(path)
    caption = (f"{settings['model']} | imgsz={settings['imgsz']} | conf={conf:g} | "
               f"iou={settings['iou']:g} | sha={settings['weights_sha256'][:12]}")
    try:
        with closing(video_frames(path, stride, video)) as frames, \
             ResultOutput(out, settings['model'], path, save=save_images, show=show_images,
                          fps=video['fps'] / stride, caption=caption) as output:
            for frame in frames:
                if not times:
                    for _ in range(warmup):
                        _, ms = timed_prediction(model, frame, settings, conf, device)
                        cold_times.append(ms)
                result, ms = timed_prediction(model, frame, settings, conf, device)
                times.append(ms)
                reads.append(frame.read_ms)
                boxes = detections(result)
                smoke = [b for b in boxes if b['cls'] == 'smoke']
                hits += bool(smoke)
                available = model.available_frame_index if temporal else frame.index
                support = model.support_indices if temporal else [frame.index]
                writer.writerow(dict(model=settings['model'], video_id=path.name,
                                     frame_index=frame.index, timestamp_s=frame.index / video['fps'],
                                     decision_timestamp_s=available / video['fps'], available_frame_index=available,
                                     support_frame_indices=json.dumps(support), fps=video['fps'], stride=stride,
                                     smoke_boxes=len(smoke), fire_boxes=sum(b['cls'] == 'fire' for b in boxes),
                                     max_smoke_conf=max((b['conf'] for b in smoke), default=''),
                                     detections_json=json.dumps(boxes), infer_ms=ms, read_ms=frame.read_ms))
                output.write(result)
                if len(times) % 300 == 0:
                    print(f"{settings['model']} / {path.name}: {len(times)} 帧", flush=True)
    finally:
        if temporal:
            model.close_video()
    if not times:
        raise ValueError(f'视频没有推理帧: {path.name}')
    print(f"完成 {settings['model']} / {path.name}: {len(times)} 帧", flush=True)
    return dict(model=settings['model'], video_id=path.name, weights_sha256=settings['weights_sha256'],
                imgsz=settings['imgsz'], iou=settings['iou'], iou_source=settings['iou_source'],
                conf=conf, conf_source=settings['conf_source'], rect=settings['rect'], device=str(result.boxes.data.device), frames=len(times),
                smoke_hit_frames=hits, smoke_hit_rate=hits / len(times), warmup_ms_json=json.dumps(cold_times),
                mean_ms=sum(times) / len(times), p95_ms=percentile(times), mean_read_ms=sum(reads) / len(reads))


def run(source, models=DEFAULT_VIDEO_MODELS, *, conf=None, iou=None, device='0', stride=1, warmup=3,
        consecutive=(1, 3, 5), out=ROOT / 'samples/video_results', save_images=False,
        show_images=False, model_imgsz=None):
    if stride < 1 or warmup < 0 or (conf is not None and not 0 <= conf <= 1) or not models:
        raise ValueError('stride >=1、warmup >=0、conf 在 [0,1] 内，且至少选择一个模型')
    if not consecutive or any(n < 1 for n in consecutive):
        raise ValueError('连续帧数必须 >=1')
    if len(set(models)) != len(models):
        raise ValueError('模型不能重复')
    import cv2
    import torch
    import ultralytics

    paths = source_paths(Path(source), video_only=True)
    videos = [inspect_video(path) for path in paths]
    sizes = model_imgsz or {}
    out = new_run(Path(out))
    manifest = dict(status='running', conf=conf, stride=stride, warmup=warmup,
                    time_basis='frame_index / nominal_fps; CFR assumed; not wall-clock alarm latency',
                    evaluation_scope='local third-party video demonstration; no verified carport ground truth',
                    versions=dict(python=platform.python_version(), torch=torch.__version__,
                                  ultralytics=ultralytics.__version__, opencv=cv2.__version__),
                    device_requested=device, models=[], videos=videos, save_video=save_images, show_video=show_images)
    manifest_path = out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    summaries, counts = [], {}
    print(f'结果: {out}', flush=True)
    try:
        with (out / 'detections.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for name in models:
                model = load_detector(name)
                entry = MODELS[name]
                settings = prediction_settings(name, model, imgsz=sizes.get(name), conf=conf, iou=iou)
                settings.update(model=name, weights=entry['dest'], weights_sha256=entry['sha256'],
                                revision=entry['revision'], temporal_frames=entry.get('temporal_frames', 1),
                                classes=dict(model.names), end2end=bool(getattr(model.model, 'end2end', False)))
                if name == 'lwst_fsd':
                    from sentinel.models.lwst.sources import REVISION, SOURCE_HASHES
                    settings.update(source_revision=REVISION, source_sha256=SOURCE_HASHES, context_offsets=[-2, -1, 0, 1, 2])
                manifest['models'].append(settings)
                print(f"推理设置: {name}, imgsz={settings['imgsz']}, conf={settings['conf']} ({settings['conf_source']}), "
                      f"iou={settings['iou']} ({settings['iou_source']}), rect={settings['rect']}", flush=True)
                for path, video in zip(paths, videos):
                    summaries.append(process_video(model, settings, path, video, writer, out,
                                                   conf=settings['conf'], device=device, stride=stride, warmup=warmup,
                                                   save_images=save_images, show_images=show_images))
                    if path.name in counts and counts[path.name] != video['decoded_frames']:
                        raise ValueError(f'不同模型读取的帧数不一致: {path.name}')
                    counts[path.name] = video['decoded_frames']
                    stream.flush()
                del model
        write_csv(out / 'summary.csv', summaries)
        write_csv(out / 'annotations_template.csv', [dict(video_id=v['video_id'], start_frame=0,
                  end_frame=counts[v['video_id']] - 1, state='unknown', interference='', smoke_onset_frame='') for v in videos])
        evaluate(out / 'detections.csv', out / 'temporal', consecutive)
        manifest.update(status='complete', decoded_frames=counts)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('完成；未标注视频的触发时间不代表预警延迟，检出占比不代表召回率。')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=DEFAULT_VIDEO_MODELS)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, help='覆盖所有模型阈值；省略时按各模型上游示例')
    parser.add_argument('--iou', type=float, help='所有所选模型的 NMS 阈值；省略时使用各自默认')
    parser.add_argument('--model-imgsz', nargs='+', default=[], metavar='模型=尺寸')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--consecutive', nargs='+', type=int, default=[1, 3, 5])
    parser.add_argument('--out', type=Path, default=ROOT / 'samples/video_results')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--show-images', action='store_true')
    args = vars(parser.parse_args(argv))
    sizes = {}
    for item in args['model_imgsz']:
        name, _, value = item.partition('=')
        if name not in args['models'] or name in sizes or not value.isdigit() or int(value) < 32 or int(value) % 32:
            parser.error(f'尺寸格式应为所选模型=32的正整数倍，且不能重复: {item}')
        sizes[name] = int(value)
    args['model_imgsz'] = sizes
    return run(**args)
