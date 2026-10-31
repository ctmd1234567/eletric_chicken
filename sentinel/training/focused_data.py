"""Build a compact fine-tuning set with explicit manual/background sampling."""
import argparse
from collections import Counter
import json
import math
import os
import shutil
from pathlib import Path

from PIL import Image
import yaml

from sentinel.paths import ROOT
from .urban_data import boxes_from_label, digest

DATA = ROOT/'datasets/ebike_focused_20261031_v2'


def verify_runtime_pairs(data_root):
    """Check the exact label paths YOLO will resolve, not only our manifest fields."""
    from ultralytics.data.utils import img2label_paths
    data_root=Path(data_root).resolve()
    rows=[json.loads(line) for line in (data_root/'merged/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in rows:
        image=(data_root/r['image']).resolve();label=(data_root/r['label']).resolve()
        resolved=Path(img2label_paths([str(image)])[0]).resolve()
        if resolved!=label or not resolved.is_file():
            raise ValueError('Actual YOLO label path does not match manifest: '+r['image'])
        if digest(label)!=r['label_sha256']:
            raise ValueError('Training label checksum mismatch: '+r['image'])
    return len(rows)


def sampling_repeats(counts, manual_share=.4, jjs_share=.15):
    if not 0 < manual_share < 1 or not 0 < jjs_share < 1 or manual_share+jjs_share>=1:
        raise ValueError('Sampling shares must be positive and sum to less than one')
    if any(counts.get(k,0)<=0 for k in ('manual','jjs','regular')):
        raise ValueError('Each sampling group must contain images')
    scale=counts['regular']/(1-manual_share-jjs_share)
    return dict(manual=max(1,math.ceil(scale*manual_share/counts['manual'])),
                jjs=max(1,math.ceil(scale*jjs_share/counts['jjs'])),regular=1)


def assemble(annotation_import, output=DATA):
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve existing datasets; choose a new output directory')
    annotation_import=Path(annotation_import).resolve()
    imported=json.loads((annotation_import/'manifest.json').read_text(encoding='utf-8'))
    if any(not r.get('training_eligible') for r in imported):
        raise ValueError('Training authorization has not been recorded for imported images')
    records=[];by_sha={};excluded=[]

    def add(image,label,source,split,group='regular',**meta):
        image=Path(image).resolve();label=Path(label).resolve()
        if not image.is_relative_to(ROOT) or not label.is_relative_to(ROOT) or not image.is_file() or not label.is_file():
            raise ValueError('Missing or unsafe training pair')
        boxes=boxes_from_label(label)
        sha=digest(image)
        previous=by_sha.get(sha)
        if previous:
            if group=='manual' and previous['split']!='train':
                raise ValueError('Manual training sample overlaps holdout')
            excluded.append(dict(image=image.relative_to(ROOT).as_posix(),reason='exact duplicate of '+previous['split']))
            return
        records.append(dict(image=os.path.relpath(image,output).replace('\\','/'),
                            label=os.path.relpath(label,output).replace('\\','/'),
                            sha256=sha,label_sha256=digest(label),boxes=boxes,negative=not boxes,
                            dataset=source,split=split,sampling_group=group,**meta))
        by_sha[sha]=records[-1]

    # 原留出集先登记，防止回放图片混入；它们只作历史参考，不称为新场景独立测试。
    replay_root=ROOT/'datasets/jjs_negative_finetune'
    replay=json.loads((replay_root/'manifest.json').read_text(encoding='utf-8'))['samples']
    for r in replay:
        if r['split'] in ('val','test'):
            add(ROOT/r['image'],ROOT/r['label'],'reference_'+r['origin'],r['split'])
    local=ROOT/'datasets/cvat_ludan_review'
    for line in (local/'annotated.txt').read_text(encoding='utf-8').splitlines():
        if line.strip():
            image=(local/line).resolve()
            add(image,local/'labels'/image.with_suffix('.txt').name,'user_manual_previous','train','manual',
                source_group='previous_local_event')
    new_root=ROOT/'datasets/ebike_early_smoke_review_20261031'
    for r in imported:
        image=new_root/r['image'];label=new_root/r['label']
        if digest(image)!=r['image_sha256'] or digest(label)!=r['label_sha256']:
            raise ValueError('Imported annotations changed')
        add(image,label,'user_manual_current','train','manual',source_group=r['source_id'],
            annotation_status=r['annotation_status'],training_authorization='user_confirmed_2026-10-31')
    for r in replay:
        if r['split']=='train' and r['origin'] in ('jjs','web'):
            add(ROOT/r['image'],ROOT/r['label'],'replay_'+r['origin'],'train',
                'jjs' if r['origin']=='jjs' else 'regular')
    # 仅保留旧专项微调的 160 张烟雾和 40 张背景回放，不直接导入三万多张混合清单。
    replay_small=ROOT/'datasets/ludan_finetune'
    for image in sorted((replay_small/'images/train').glob('*')):
        if image.suffix.lower() in ('.png','.jpg','.jpeg') and not image.name.startswith('cvat_'):
            add(image,replay_small/'labels/train'/image.with_suffix('.txt').name,'legacy_small_replay','train')
    urban=ROOT/'datasets/urban_smoke_20261031'
    approved=set(json.loads((urban/'review.json').read_text(encoding='utf-8'))['approved'])
    for line in (urban/'new_manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['image'] in approved:
            image=urban/r['image'];label=urban/r['label']
            if digest(image)!=r['sha256'] or digest(label)!=r['label_sha256']:
                raise ValueError('Reviewed scene pair changed')
            add(image,label,r['dataset'],'train')
    counts=Counter(r['sampling_group'] for r in records if r['split']=='train')
    repeats=sampling_repeats(counts)
    if counts['manual']!=140 or counts['jjs']!=32:
        raise ValueError(f'Unexpected reviewed sample counts: {counts}')
    output.mkdir(parents=True);merged=output/'merged';merged.mkdir()
    splits={s:[] for s in ('train','val','test')}
    from ultralytics.data.utils import img2label_paths
    for r in records:
        source_image=(output/r['image']).resolve()
        source_label=(output/r['label']).resolve()
        # YOLO 根据图片路径自动查找 labels/，不会读取 manifest 中自定义的 label 字段。
        # 所有来源统一复制到严格成对的 images/{split} 与 labels/{split} 目录。
        stem=r['dataset']+'_'+r['sha256'][:20]
        image=output/'images'/r['split']/(stem+source_image.suffix.lower())
        label=output/'labels'/r['split']/(stem+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_image,image);shutil.copyfile(source_label,label)
        if Path(img2label_paths([str(image)])[0]).resolve()!=label.resolve():
            raise ValueError('YOLO image/label resolver mismatch')
        if digest(image)!=r['sha256'] or digest(label)!=r['label_sha256']:
            raise ValueError('Copied pair checksum mismatch')
        r['source_image']=source_image.relative_to(ROOT).as_posix()
        r['source_label']=source_label.relative_to(ROOT).as_posix()
        r['image']=image.relative_to(output).as_posix();r['label']=label.relative_to(output).as_posix()
        # 完整解码本轮小训练集；不修改图片或用户标注。
        with Image.open(image) as im:im.load()
        times=repeats[r['sampling_group']] if r['split']=='train' else 1
        r['sampling_repeats']=times
        splits[r['split']].extend([image.as_posix()]*times)
    for split,paths in splits.items():
        name='train_balanced.txt' if split=='train' else split+'.txt'
        (merged/name).write_text('\n'.join(paths)+'\n',encoding='utf-8')
    cfg=dict(path=merged.as_posix(),train='train_balanced.txt',val='val.txt',test='test.txt',names={0:'smoke',1:'fire'})
    (merged/'data_balanced.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    (merged/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    shares={k:counts[k]*repeats[k]/len(splits['train']) for k in counts}
    audit=dict(status='passed',unique_images=len(records),unique_train_images=sum(counts.values()),
               splits={s:len(p) for s,p in splits.items()},groups=dict(counts),sampling_repeats=repeats,
               sampling_shares=shares,manual_user_reviewed_backgrounds=sum(r['dataset']=='user_manual_current' and r['negative'] for r in records),
               sources=dict(Counter(r['dataset'] for r in records)),cross_split_sha256_overlap=0,
               all_labels_checked=True,all_image_sha256=True,all_images_decoded=True,
               yolo_resolved_label_paths_match=True,
               limitations=['All user-reviewed examples are training data; fit is not unseen-event generalization.',
                            'The 14 empty labels mean no human-discernible target, not absence of a real hazard.',
                            'Validation/test are previously used reference sets, not independent carport tests.',
                            'No proof of upstream event independence; synthetic scenes retain CC BY-NC restrictions.'])
    (merged/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (merged/'excluded.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    profile=dict(base_model='dfire_ludan',dataset='focused',lr0=.0001,freeze=10,manual_share_target=.4,
                 jjs_share_target=.15,import_summary=json.loads((annotation_import/'summary.json').read_text(encoding='utf-8')))
    (output/'training_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    print('FOCUSED_READY',json.dumps(audit),flush=True)
    return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-import',type=Path,default=ROOT/'datasets/ebike_early_smoke_review_20261031/annotations/import_004')
    parser.add_argument('--out',type=Path,default=DATA)
    args=parser.parse_args(argv)
    assemble(args.annotation_import,args.out)
    return 0
