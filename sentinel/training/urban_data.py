"""Assemble an audited, weighted training list without changing source labels."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import yaml

from sentinel.paths import ROOT

DATA = ROOT/'datasets/urban_smoke_20261031'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def boxes_from_label(path):
    boxes = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid label: {path}')
        cls, x, y, w, h = values
        if cls not in (0, 1) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'Invalid class or coordinates: {path}')
        if min(x-w/2, y-h/2) < -.002 or max(x+w/2, y+h/2) > 1.002:
            raise ValueError(f'Box outside image: {path}')
        boxes.append(values)
    return boxes


def relative(path, directory=DATA):
    return os.path.relpath(path, directory).replace('\\', '/')


def record(image, label, source, split, *, priority=False, expected_sha=None):
    if not image.is_file() or not label.is_file():
        raise ValueError(f'Missing image/label pair: {image}')
    sha = digest(image)
    if expected_sha and expected_sha != sha:
        raise ValueError(f'Source image changed: {image}')
    boxes = boxes_from_label(label)
    return dict(image=relative(image), label=relative(label), sha256=sha,
                label_sha256=digest(label), dataset=source, split=split,
                boxes=boxes, negative=not boxes, priority=priority)


def preview(records, name):
    folder = DATA/'previews'
    folder.mkdir(parents=True, exist_ok=True)
    for page in range((len(records)+23)//24):
        chunk = records[page*24:(page+1)*24]
        sheet = Image.new('RGB', (4*360, 6*292), 'white')
        draw = ImageDraw.Draw(sheet)
        for idx, row in enumerate(chunk):
            image = Image.open(DATA/row['image']).convert('RGB')
            pen = ImageDraw.Draw(image)
            for cls, x,y,w,h in row['boxes']:
                W,H=image.size
                pen.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),
                              outline='lime' if cls == 0 else 'red', width=max(2,W//300))
            thumb = ImageOps.contain(image, (360,264))
            x,y=idx%4*360,idx//4*292
            sheet.paste(thumb,(x,y))
            draw.text((x+3,y+265),Path(row['image']).name[:46],fill='black')
            draw.text((x+3,y+279),row['split']+' / '+row['dataset'],fill='black')
        sheet.save(folder/f'{name}_{page+1:02d}.jpg')


def assemble(priority_share=.10):
    if not 0 < priority_share <= .25:
        raise ValueError('priority share must be in (0,.25]')
    merged = DATA/'merged'
    merged.mkdir(parents=True, exist_ok=True)
    if (merged/'audit.json').exists():
        raise ValueError('Dataset already assembled; preserve its audit before rebuilding')
    old = ROOT/'datasets/acquisition_20261030'
    audit = json.loads((old/'merged/audit.json').read_text())
    if audit['status'] != 'passed':
        raise ValueError('Original dataset audit has not passed')
    selected={split:set((old/'merged'/file).read_text().splitlines())
              for split,file in [('train','train_balanced.txt'),('val','val.txt'),('test','test.txt')]}
    records=[]
    for line in (old/'merged/manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        image=(old/r['image'].replace('\\','/')).resolve()
        split=next((s for s,paths in selected.items() if image.as_posix() in paths),None)
        if split:
            records.append(record(image,(old/r['label'].replace('\\','/')).resolve(),r['dataset'],split,expected_sha=r['sha256']))
    print('ORIGINAL_VERIFIED',len(records),flush=True)
    by_sha={r['sha256']:r for r in records}
    exclusions=[]
    # 只纳入用户已提供的 40 份标签；另外 16 张未标注帧仍不转为背景。
    local=ROOT/'datasets/cvat_ludan_review'
    confirmed=[(local/p).resolve() for p in (local/'annotated.txt').read_text().splitlines() if p.strip()]
    if len(confirmed) != 40:
        raise ValueError('Expected the existing 40 manually annotated images')
    local_records=[record(p,local/'labels'/p.with_suffix('.txt').name,'user_manual','train',priority=True) for p in confirmed]
    replay=json.loads((ROOT/'datasets/jjs_negative_finetune/manifest.json').read_text())['samples']
    replay_records=[record(ROOT/r['image'],ROOT/r['label'],'ludan_replay_'+r['origin'],'train')
                    for r in replay if r['split']=='train' and r['origin'] not in ('local_cvat','cvat')]
    review=json.loads((DATA/'review.json').read_text())
    additions=[json.loads(line) for line in (DATA/'new_manifest.jsonl').read_text().splitlines()]
    approved=set(review['approved'])
    additions=[r for r in additions if r['image'] in approved]
    if not additions:
        raise ValueError('No reviewed new images')
    for r in additions:
        image=(DATA/r['image']).resolve();label=(DATA/r['label']).resolve()
        checked=record(image,label,r['dataset'],r['split'],expected_sha=r['sha256'])
        if checked['label_sha256'] != r['label_sha256']:
            raise ValueError('New labels changed after review')
        r.update(checked)
    for r in local_records+replay_records+additions:
        existing=by_sha.get(r['sha256'])
        if existing:
            if r['priority']:
                if existing['split']!='train':
                    raise ValueError('Manual priority image overlaps holdout')
                # 同图以用户手工标签为准，保留原始来源文件和排除记录。
                exclusions.append(dict(image=existing['image'],reason='duplicate replaced by user manual label'))
                records.remove(existing);records.append(r);by_sha[r['sha256']]=r
            else:
                exclusions.append(dict(image=r['image'],reason='exact duplicate of '+existing['split']))
            continue
        records.append(r);by_sha[r['sha256']]=r
    priority=[r for r in records if r['priority']]
    scene_sources={r['dataset'] for r in additions}
    scenes=[r for r in records if r['split']=='train' and r['dataset'] in scene_sources]
    regular=[r for r in records if r['split']=='train' and not r['priority'] and r['dataset'] not in scene_sources]
    # 新场景约占采样的 5%，合成数据只作低占比补充；手工标注约占 10%。
    scene_share=.05
    denominator=1-priority_share-scene_share
    repeats=max(1,math.ceil(priority_share*len(regular)/(denominator*len(priority))))
    scene_repeats=max(1,math.ceil(scene_share*len(regular)/(denominator*len(scenes))))
    splits={s:[(DATA/r['image']).resolve().as_posix() for r in records if r['split']==s]
            for s in ('train','val','test')}
    splits['train'] += [(DATA/r['image']).resolve().as_posix() for r in priority for _ in range(repeats-1)]
    splits['train'] += [(DATA/r['image']).resolve().as_posix() for r in scenes for _ in range(scene_repeats-1)]
    for split,paths in splits.items():
        name='train_balanced.txt' if split=='train' else split+'.txt'
        (merged/name).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    cfg=dict(path=merged.as_posix(),train='train_balanced.txt',val='val.txt',test='test.txt',names={0:'smoke',1:'fire'})
    (merged/'data_balanced.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    (merged/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    (merged/'excluded.json').write_text(json.dumps(exclusions,indent=2),encoding='utf-8')
    summary=dict(status='passed',unique_images=len(records),splits={s:len(p) for s,p in splits.items()},
                 unique_by_split=dict(Counter(r['split'] for r in records)),
                 sources=dict(Counter(r['dataset'] for r in records)),manual_images=len(priority),
                 manual_repeats=repeats,manual_sampling_share=len(priority)*repeats/len(splits['train']),
                 scene_repeats=scene_repeats,scene_sampling_share=len(scenes)*scene_repeats/len(splits['train']),
                 all_image_sha256=True,all_labels_checked=True,cross_split_sha256_overlap=0,
                 limitations=['Manual images are from one previously trained event; not an independent test.',
                              'Base checkpoint may have seen source holdouts in earlier experiments.',
                              'New synthetic CCTV examples are not real battery fire evidence.',
                              'No proof of event independence across all upstream sources.'])
    (merged/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    preview(priority,'user_manual');preview(additions,'new_scenes')
    (DATA/'training_profile.json').write_text(json.dumps(dict(base_model='dfire_ludan',lr0=.0001,freeze=10,
         priority_share_target=priority_share,manual_repeats=repeats),indent=2),encoding='utf-8')
    print('URBAN_READY',json.dumps(summary),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--priority-share',type=float,default=.10,help='用户手工标注图片占每轮采样的目标比例，默认 10%%')
    args=parser.parse_args(argv)
    assemble(args.priority_share)
    return 0
