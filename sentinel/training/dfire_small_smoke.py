"""Fine-tune an isolated copy of D-Fire on a prepared YOLO dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from sentinel.models.catalog import MODELS
from sentinel.paths import ROOT


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(weights, data, output, imgsz, batch):
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split='test', imgsz=imgsz, batch=batch,
                        device=0, workers=0, plots=False, conf=0.001, iou=0.7,
                        project=str(output.parent), name=output.name, exist_ok=False,
                        half=False, verbose=False)
    per_class = {}
    for i, cls in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class[model.names[int(cls)]] = dict(precision=float(p), recall=float(r),
            AP50=float(ap50), AP50_95=float(ap))
    record = dict(weights_sha256=digest(weights), imgsz=imgsz,
                  metrics={k:float(v) for k,v in metrics.results_dict.items()},
                  classes=per_class, speed=metrics.speed,
                  note='P/R use the library-selected F1 operating point, not a fixed deployment confidence.')
    del model
    torch.cuda.empty_cache()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'datasets/fasdd_small_smoke_1000/data.yaml')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/dfire_fasdd_small_smoke/experiment_001')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--base-model', choices=['dfire_yolov8n', 'dfire_small_smoke', 'dfire_ludan'], default='dfire_yolov8n')
    parser.add_argument('--lr0', type=float, default=0.0003)
    parser.add_argument('--freeze', type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 < args.lr0 < 1 or args.freeze < 0:
        parser.error('lr0 must be in (0,1); freeze must be nonnegative')
    entry = MODELS[args.base_model]
    original = ROOT/entry['dest']
    if digest(original) != entry['sha256']:
        raise ValueError('Selected base model weight hash mismatch')
    config = yaml.safe_load(args.data.read_text(encoding='utf-8'))
    names = config['names']
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != {0:'smoke',1:'fire'}:
        raise ValueError('Dataset classes must match D-Fire: 0=smoke, 1=fire')
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the verified local CUDA device')
    # A fresh directory prevents accidental replacement of an earlier experiment.
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='baseline', data_sha256=digest(args.data),
                  base_model=args.base_model,
                  original_sha256=entry['sha256'], torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0),
                  limitations=['Source-event identities are not fully known; this is an internal experiment.',
                    'Possible overlap with upstream D-Fire training data; no independent generalization claim.',
                    'Small object annotations do not establish thin-smoke or early-warning capability.'])
    summary = args.out/'comparison.json'

    def save():
        summary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    try:
        save()
        # Evaluate both original resolution and training resolution, so the comparison
        # cannot attribute a resolution change to learning.
        report['baseline_640'] = evaluate(original,args.data,args.out/'baseline_640',640,args.batch)
        report['baseline'] = evaluate(original,args.data,args.out/'baseline',args.imgsz,args.batch)
        report['status']='training';save()
        model = YOLO(str(original))
        if model.names != {0:'smoke',1:'fire'}:
            raise ValueError('Unexpected checkpoint classes')
        model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=0, workers=0, optimizer='AdamW',
                    lr0=args.lr0, freeze=args.freeze, lrf=0.1, weight_decay=0.0005, patience=8,
                    warmup_epochs=2, seed=20261030, deterministic=True,
                    amp=False, cache=False, plots=True, resume=False,
                    mosaic=0.2, close_mosaic=5, scale=0.2, translate=0.1,
                    mixup=0.0, fliplr=0.5, hsv_s=0.3, hsv_v=0.3,
                    project=str(args.out), name='train', exist_ok=False,
                    save=True, save_period=5, pretrained=True)
        best=Path(model.trainer.best)
        report['best_weights']=str(best.relative_to(ROOT))
        report['epochs_completed']=int(model.trainer.epoch)+1
        report['best_epoch']=int(model.trainer.stopper.best_epoch)
        del model
        torch.cuda.empty_cache()
        report['status']='evaluation';save()
        report['finetuned']=evaluate(best,args.data,args.out/'finetuned',args.imgsz,args.batch)
        report['finetuned_640']=evaluate(best,args.data,args.out/'finetuned_640',640,args.batch)
        if digest(original) != entry['sha256']:
            raise RuntimeError('Original checkpoint changed')
        report['status']='complete';save()
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return 0
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
