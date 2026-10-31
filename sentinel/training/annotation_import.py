"""Import annotation-only exports without treating missing labels as backgrounds."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from PIL import Image, ImageDraw, ImageOps
import yaml

from sentinel.paths import ROOT
from .urban_data import boxes_from_label, digest


def clip_labels(text, mapping):
    """按用户要求将框裁到图片边界；保留原 ZIP，不改动未越界的坐标。"""
    lines=[];changes=[]
    for index,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        values=list(map(float,line.split()))
        if len(values)!=5 or not all(math.isfinite(v) for v in values):
            raise ValueError('Malformed or nonfinite bounding box')
        cls,x,y,w,h=values
        if cls!=int(cls) or int(cls) not in mapping or w<=0 or h<=0:
            raise ValueError('Invalid class or empty bounding box')
        bounds=(x-w/2,y-h/2,x+w/2,y+h/2)
        x1,y1,x2,y2=max(0,bounds[0]),max(0,bounds[1]),min(1,bounds[2]),min(1,bounds[3])
        if x2<=x1 or y2<=y1:
            raise ValueError('Bounding box does not intersect image')
        clipped=[mapping[int(cls)],(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]
        if min(bounds[:2])<0 or max(bounds[2:])>1:
            changes.append(dict(line=index,original=values,clipped=clipped))
            lines.append(str(clipped[0])+' '+' '.join(f'{v:.8f}' for v in clipped[1:]))
        elif mapping[int(cls)]!=int(cls):
            lines.append(str(mapping[int(cls)])+' '+' '.join(line.split()[1:]))
        else:
            lines.append(line)
    return '\n'.join(lines)+ ('\n' if lines else ''),changes


def import_export(archive, dataset, output, *, review_missing_as_background=False, training_authorized=False):
    archive, dataset, output = map(lambda p: Path(p).resolve(), (archive,dataset,output))
    if not output.is_relative_to(dataset):
        raise ValueError('Import output must stay inside the source dataset')
    if output.exists():
        raise ValueError('Preserve existing imports; choose a new output directory')
    source_rows=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    originals={Path(r['image']).stem:r for r in source_rows}
    if len(originals)!=len(source_rows):
        raise ValueError('Duplicate image stems in source manifest')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError('Duplicate or corrupt archive members')
        cfg=yaml.safe_load(z.read('data.yaml').decode('utf-8-sig'))
        names=cfg['names'];names=dict(enumerate(names)) if isinstance(names,list) else names
        names={int(k):str(v).strip().lower() for k,v in names.items()}
        if set(names.values()) != {'smoke','fire'} or len(names)!=2:
            raise ValueError('Expected explicitly named smoke/fire classes')
        mapping={k:{'smoke':0,'fire':1}[v] for k,v in names.items()}
        labels={}
        for name in z.namelist():
            member=PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or '\\' in name:
                raise ValueError('Unsafe archive member')
            if member.suffix=='.txt' and 'labels' in member.parts:
                stem=member.stem
                if stem not in originals or stem in labels:
                    raise ValueError('Unknown or duplicate annotation stem: '+stem)
                labels[stem]=z.read(name)
        if not labels:
            raise ValueError('Export contains no labels')
    missing=sorted(set(originals)-set(labels))
    exported_count=len(labels)
    if review_missing_as_background:
        # 仅在用户明确判定这些帧没有可辨认目标后，才创建背景空标签。
        labels.update({stem:b'' for stem in missing})
    output.mkdir(parents=True)
    (output/'labels').mkdir()
    shutil.copyfile(archive,output/'source_export.zip')
    rows=[];quarantined=[];clipping=[];shared_boxes=defaultdict(set);counts=Counter()
    for stem,raw in sorted(labels.items()):
        source=originals[stem]
        image=(dataset/source['image']).resolve()
        if not image.is_relative_to(dataset) or digest(image)!=source['image_sha256']:
            raise ValueError('Local original image changed: '+stem)
        label=output/'labels'/(stem+'.txt')
        try:
            text,changes=clip_labels(raw.decode('utf-8-sig'),mapping)
            label.write_text(text,encoding='utf-8')
            boxes=boxes_from_label(label)
            if changes:clipping.append(dict(image=source['image'],changes=changes))
        except ValueError as exc:
            quarantined.append(dict(image=source['image'],label=label.relative_to(dataset).as_posix(),
                                    source_id=source['source_id'],reason=str(exc),training_eligible=False))
            continue
        for box in boxes:
            counts[['smoke','fire'][int(box[0])]]+=1
            shared_boxes[tuple(box)].add(source['source_id'])
        rows.append(dict(image=source['image'],label=label.relative_to(dataset).as_posix(),
                         image_sha256=source['image_sha256'],label_sha256=digest(label),
                         boxes=boxes,source_id=source['source_id'],
                         annotation_status='user_reviewed_no_visible_target' if stem in missing else 'user_labelled',
                         training_eligible=training_authorized,
                         rights='User confirmed training authorization' if training_authorized else source.get('rights','Not established')))
    (output/'missing_labels.txt').write_text('\n'.join(missing)+'\n',encoding='utf-8')
    (output/'annotated.txt').write_text('\n'.join((dataset/r['image']).as_posix() for r in rows)+'\n',encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'quarantined.json').write_text(json.dumps(quarantined,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'clipping.json').write_text(json.dumps(clipping,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=dict(status='imported',source_export_sha256=digest(archive),source_images=len(originals),
                 exported_label_files=exported_count,labelled_images=len(rows),quarantined_images=len(quarantined),
                 clipped_images=len(clipping),clipped_boxes=sum(len(r['changes']) for r in clipping),
                 explicit_empty_labels=sum(not r['boxes'] for r in rows),
                 missing_labels=len(missing),missing_labels_used_as_background=review_missing_as_background,
                 user_reviewed_empty_images=len(missing) if review_missing_as_background else 0,
                 boxes=dict(counts),source_clips=len({r['source_id'] for r in rows}),
                 source_class_names=names,output_class_names={0:'smoke',1:'fire'},
                 identical_boxes_across_scenes=[dict(box=list(box),scenes=sorted(scenes))
                    for box,scenes in shared_boxes.items() if len(scenes)>1],
                 rights_status='user_confirmed_training_authorization' if training_authorized else 'pending; imported for local review, not added to training',
                 limitations=['YOLO export does not retain track IDs or Outside keyframes.',
                              'Geometry checks cannot prove semantic label correctness.'])
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    previews=output/'previews';previews.mkdir()
    for page in range((len(rows)+23)//24):
        sheet=Image.new('RGB',(4*360,6*252),'white');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[page*24:(page+1)*24]):
            im=Image.open(dataset/row['image']).convert('RGB');W,H=im.size;pen=ImageDraw.Draw(im)
            for cls,x,y,w,h in row['boxes']:
                pen.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),
                              outline='lime' if cls==0 else 'red',width=max(2,W//250))
            x=j%4*360;y=j//4*252;sheet.paste(ImageOps.contain(im,(360,224)),(x,y))
            draw.text((x+2,y+227),Path(row['image']).stem,fill='black')
        sheet.save(previews/f'contact_{page+1:02d}.jpg')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/ebike_early_smoke_review_20261031')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--review-missing-as-background',action='store_true',help='仅在用户已确认缺标签帧无可辨认目标时使用')
    parser.add_argument('--training-authorized',action='store_true',help='记录用户已确认训练使用授权')
    args=parser.parse_args(argv)
    import_export(args.archive,args.dataset,args.out or args.dataset/'annotations/import_001',
                  review_missing_as_background=args.review_missing_as_background,
                  training_authorized=args.training_authorized)
    return 0
