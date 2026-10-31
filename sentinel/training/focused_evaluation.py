"""固定阈值对比本轮权重；训练图上的匹配结果仅用于检查拟合情况。"""
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

import cv2
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sentinel.models.catalog import MODELS
from sentinel.models.weights import sha256_of

DATA=ROOT/'datasets/ebike_focused_20261031_v2'
OUT=ROOT/'runs/dfire_ebike_focused/experiment_002'
# 两档阈值对所有候选保持一致，避免把调低阈值当成模型改善。
THRESHOLDS=(.15,.25)


def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]));h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def evaluate(weights, records, imgsz=960, skip_video=False):
    model=YOLO(str(weights))
    assert model.names=={0:'smoke',1:'fire'}
    stats={str(t):defaultdict(lambda:dict(images=0,negative_images=0,negative_images_with_detection=0,
                smoke_gt_boxes=0,smoke_tp=0,smoke_fp=0,fire_gt_boxes=0,fire_tp=0,fire_fp=0)) for t in THRESHOLDS}
    details=[]
    for i,r in enumerate(records,1):
        image=cv2.imread(str((DATA/r['image']).resolve()))
        if image is None:raise ValueError('Cannot decode image')
        H,W=image.shape[:2]
        truth=[(int(c),[(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H]) for c,x,y,w,h in r['boxes']]
        result=model.predict(image,imgsz=imgsz,conf=min(THRESHOLDS),iou=.7,rect=True,device=0,verbose=False)[0]
        predictions=result.boxes.data.cpu().tolist()
        for threshold in THRESHOLDS:
            selected=[p for p in predictions if p[4]>=threshold]
            rec=stats[str(threshold)][r['dataset']]
            rec['images']+=1
            if not truth:
                rec['negative_images']+=1;rec['negative_images_with_detection']+=bool(selected)
            for cls,name in ((0,'smoke'),(1,'fire')):
                gt=[box for c,box in truth if c==cls];used=set();tp=0
                det=sorted([p for p in selected if int(p[5])==cls],key=lambda x:x[4],reverse=True)
                for p in det:
                    possible=[(iou(p[:4],box),j) for j,box in enumerate(gt) if j not in used]
                    best=max(possible,default=(0,-1))
                    if best[0]>=.5:tp+=1;used.add(best[1])
                rec[name+'_gt_boxes']+=len(gt);rec[name+'_tp']+=tp;rec[name+'_fp']+=len(det)-tp
        details.append(dict(image=r['image'],dataset=r['dataset'],gt=r['boxes'],predictions=predictions))
        if i%50==0:print('LABELLED_CHECK',i,'/',len(records),flush=True)
    if skip_video:
        return dict(weights_sha256=sha256_of(weights),labelled_groups={t:dict(groups) for t,groups in stats.items()}),details
    cap=cv2.VideoCapture(str(ROOT/'video/jjs.mp4'))
    if not cap.isOpened():raise ValueError('Cannot decode jjs video')
    video={str(t):dict(frames=0,smoke_frames=0,fire_frames=0,any_detection_frames=0,
                      smoke_frame_indices=[],fire_frame_indices=[]) for t in THRESHOLDS}
    while True:
        ok,image=cap.read()
        if not ok:break
        result=model.predict(image,imgsz=imgsz,conf=min(THRESHOLDS),iou=.7,rect=True,device=0,verbose=False)[0]
        predictions=result.boxes.data.cpu().tolist()
        for threshold in THRESHOLDS:
            selected=[p for p in predictions if p[4]>=threshold]
            rec=video[str(threshold)];rec['frames']+=1
            rec['smoke_frames']+=any(int(p[5])==0 for p in selected)
            rec['fire_frames']+=any(int(p[5])==1 for p in selected)
            rec['any_detection_frames']+=bool(selected)
            if any(int(p[5])==0 for p in selected):rec['smoke_frame_indices'].append(rec['frames']-1)
            if any(int(p[5])==1 for p in selected):rec['fire_frame_indices'].append(rec['frames']-1)
    cap.release()
    return dict(weights_sha256=sha256_of(weights),labelled_groups={t:dict(groups) for t,groups in stats.items()},
                jjs_video_outputs=video),details


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--experiment',type=Path,default=OUT)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--tag',default='candidate')
    parser.add_argument('--skip-video',action='store_true')
    args=parser.parse_args();OUT=args.experiment
    records=[json.loads(line) for line in (DATA/'merged/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    # 人工组、jjs 是训练回放；网络参考背景也曾参与模型选择。
    records=[r for r in records if r['dataset'] in ('user_manual_previous','user_manual_current','replay_jjs','reference_web')]
    report=dict(imgsz=args.imgsz,iou=.7,box_match_iou=.5,thresholds=THRESHOLDS,
                jjs_video_sha256=sha256_of(ROOT/'video/jjs.mp4'),
                limitations=['Manual and jjs image groups are training data; scores measure fit.',
                             'Reference web negatives were used for model selection, not an independent test.',
                             'Full-video counts are output counts, not measured false-positive rates.'])
    models=[(args.tag,args.weights)] if args.weights else [('baseline',ROOT/MODELS['dfire_ludan']['dest']),('finetuned',OUT/'train/weights/best.pt')]
    report_file=f'target_regression_{args.tag}_{args.imgsz}.json' if args.weights else f'target_regression_{args.imgsz}.json'
    for name,weights in models:
        report[name],details=evaluate(weights,records,args.imgsz,args.skip_video)
        (OUT/(name+f'_target_details_{args.imgsz}.json')).write_text(json.dumps(details,ensure_ascii=False),encoding='utf-8')
        (OUT/report_file).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('MODEL_FINISHED',name,flush=True)
    print('TARGET_COMPARISON_READY',flush=True)
    return 0
