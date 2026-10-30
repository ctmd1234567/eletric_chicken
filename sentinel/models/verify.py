"""校验固定权重，加载并可选运行一次单图推理。"""
import argparse
import json
import time
import traceback
from pathlib import Path

from sentinel.media.inputs import image_frame
from .catalog import MODELS
from .runtime import load_detector, prediction_settings, predict, detections


def inspect(name, image=None, device=None):
    entry = MODELS[name]
    start = time.perf_counter()
    model = load_detector(name)
    args = model.ckpt.get('train_args') or {}
    settings = prediction_settings(name, model)
    rec = dict(file=entry['dest'], load='ok', sha256=entry['sha256'], sha256_match=True,
               load_seconds=round(time.perf_counter() - start, 3), task=model.task,
               classes=dict(model.names), checkpoint_ultralytics=model.ckpt.get('version'),
               end2end=bool(getattr(model.model, 'end2end', False)),
               params=sum(p.numel() for p in model.model.parameters()), train_model=args.get('model'),
               train_imgsz=args.get('imgsz'), train_data=args.get('data'), native_imgsz=entry['imgsz'])
    if image is not None:
        frame = image_frame(image)
        start = time.perf_counter()
        result = predict(model, frame.image, settings, settings['conf'], device)
        rec.update(infer_seconds=round(time.perf_counter() - start, 3), infer_imgsz=settings['imgsz'],
                   infer_iou=settings['iou'], iou_source=settings['iou_source'],
                   infer_conf=settings['conf'], conf_source=settings['conf_source'],
                   infer_device=str(result.boxes.data.device), n_boxes=len(result.boxes),
                   detections=detections(result))
        if name == 'lwst_fsd':
            rec['temporal_test'] = 'single-image repeated five times; load/inference check only'
    return rec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--only', help='模型名、下载组名或文件名')
    parser.add_argument('--device')
    args = parser.parse_args(argv)
    selected = [name for name, entry in MODELS.items()
                if not args.only or args.only in {name, entry['group'], Path(entry['dest']).name}]
    if not selected:
        parser.error(f'没有匹配的模型: {args.only}')
    results = []
    for name in selected:
        try:
            rec = inspect(name, args.image, args.device)
            print(f"{name}: 加载成功，{rec['params']:,} 参数，类别={rec['classes']}", flush=True)
            if args.image:
                print(f"  推理: {rec['infer_device']}, imgsz={rec['infer_imgsz']}, "
                      f"conf={rec['infer_conf']}, iou={rec['infer_iou']} ({rec['iou_source']}), 框数={rec['n_boxes']}")
        except Exception as exc:
            # 批量验证保留其他模型的结果；失败退出码交给调用者。
            rec = dict(file=MODELS[name]['dest'], load='failed', error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        results.append(rec)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    failures = sum(r['load'] != 'ok' for r in results)
    print(f'汇总: {len(results) - failures}/{len(results)} 个权重校验、加载成功')
    return int(bool(failures))
