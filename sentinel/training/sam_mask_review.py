"""Small, memory-bounded SAM 2.1 mask previews; never approve training data."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sentinel.models.catalog import SEGMENTATION_MODELS

MODEL = SEGMENTATION_MODELS['sam21_small']
CHECKPOINT_SHA256 = MODEL['sha256']


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_prompts(path, limit):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not 1 <= len(rows) <= limit <= 12:
        raise ValueError('Preview requires 1..12 instances; split larger requests after human review')
    names = set()
    for row in rows:
        name = row['id']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name) or name in names:
            raise ValueError('Each id must be unique and contain only letters, numbers, _ or -')
        names.add(name)
        with Image.open(resolve(path.parent, row['image'])) as image:
            w, h = image.size
        box = np.asarray(row['box_xyxy'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError(f'Invalid prompt box: {name}')
        points = np.asarray(row.get('points', []), dtype=float).reshape(-1, 2)
        labels = row.get('point_labels', [])
        if len(points) != len(labels) or not np.isfinite(points).all() or any(v not in (0, 1) for v in labels):
            raise ValueError(f'Invalid prompt points: {name}')
        if len(points) and ((points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any()):
            raise ValueError(f'Points outside image: {name}')
    return rows


def preview(image, mask, row, out):
    rgb = np.asarray(image)
    overlay = rgb.copy()
    overlay[mask] = (0.55 * rgb[mask] + 0.45 * np.array([30, 220, 125])).astype('uint8')
    left = image.copy()
    draw = ImageDraw.Draw(left)
    draw.rectangle(row['box_xyxy'], outline=(255, 185, 30), width=2)
    for (x, y), label in zip(row.get('points', []), row.get('point_labels', [])):
        draw.ellipse((x-4, y-4, x+4, y+4), fill=(30,220,125) if label else (255,40,40))
    panels = [left, Image.fromarray(overlay), Image.fromarray(mask.astype('uint8')*255).convert('RGB')]
    card = Image.new('RGB', (1080, 280), 'white')
    for index, panel in enumerate(panels):
        panel.thumbnail((355, 230))
        card.paste(panel, (index*360, 25))
    draw = ImageDraw.Draw(card)
    draw.text((5, 4), f"{row['id']} | PROMPT / SAM OVERLAY / BINARY MASK", fill='black')
    draw.text((5, 258), 'CANDIDATE - HUMAN REVIEW REQUIRED - NOT TRAINING GROUND TRUTH', fill='black')
    card.save(out)
    return card


def run(args):
    # Validate all inputs and the exact checkpoint before allocating GPU memory.
    rows = load_prompts(args.prompts.resolve(), args.limit)
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('SAM 2.1 small checkpoint SHA-256 mismatch')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory must be new/empty; retain previous reviews')
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch.set_num_threads(min(torch.get_num_threads(), 8))
    device = 'cpu'
    preflight = {}
    if args.device != 'cpu' and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        preflight = {'free_bytes': free, 'total_bytes': total}
        # Keep >=2 GiB of currently free memory; cap this process at 3 GiB.
        budget = min(3*1024**3, free-2*1024**3)
        if budget >= 2*1024**3:
            torch.cuda.set_per_process_memory_fraction(budget/total)
            device = 'cuda'
            preflight['allocator_budget_bytes'] = budget
    model = build_sam2(MODEL['config'], str(args.checkpoint), device='cpu', apply_postprocessing=False)
    fallbacks = []
    if device == 'cuda':
        try:
            model.to('cuda')
        except torch.cuda.OutOfMemoryError:
            model.cpu()
            torch.cuda.empty_cache()
            device = 'cpu'
            fallbacks.append('GPU allocator limit during model transfer; switched to CPU')
    predictor = SAM2ImagePredictor(model)
    for folder in ('images', 'masks', 'labels', 'previews'):
        (args.out/folder).mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    start = time.perf_counter()
    for row in rows:
        source = resolve(args.prompts.parent, row['image'])
        with Image.open(source) as opened:
            image = opened.convert('RGB')
        points = np.asarray(row.get('points', []), dtype=np.float32).reshape(-1, 2)
        labels = np.asarray(row.get('point_labels', []), dtype=np.int32)
        box = np.asarray(row['box_xyxy'], dtype=np.float32)
        if device == 'cuda' and torch.cuda.mem_get_info()[0] < 2*1024**3:
            predictor.reset_predictor()
            model.cpu()
            torch.cuda.empty_cache()
            device = 'cpu'
            fallbacks.append(f"{row['id']}: free VRAM below reserve; switched to CPU")
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        while True:
            try:
                dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
                autocast = torch.autocast('cuda', dtype=dtype) if device == 'cuda' else contextlib.nullcontext()
                with torch.inference_mode(), autocast:
                    predictor.set_image(image)
                    masks, scores, _ = predictor.predict(
                        point_coords=points if len(points) else None,
                        point_labels=labels if len(points) else None,
                        box=box, multimask_output=True)
                break
            except torch.cuda.OutOfMemoryError:
                predictor.reset_predictor()
                model.cpu()
                torch.cuda.empty_cache()
                device = 'cpu'
                fallbacks.append(f"{row['id']}: GPU allocator limit; retried on CPU")
        if not np.isfinite(scores).all() or not np.isfinite(masks).all():
            raise ValueError(f"Non-finite SAM output: {row['id']}")
        choice = int(np.argmax(scores))
        mask = masks[choice].astype(bool)
        name = row['id']
        shutil.copy2(source, args.out/'images'/f'{name}{source.suffix.lower()}')
        Image.fromarray(mask.astype('uint8')*255).save(args.out/'masks'/f'{name}.png')
        ys, xs = np.where(mask)
        w, h = image.size
        flags = []
        if not len(xs):
            flags.append('empty_mask')
        if len(xs) and mask.mean() > .7:
            flags.append('mask_covers_over_70_percent_of_image')
        # Original reviewed detection labels stay intact. New labels are proposals only.
        if row.get('label'):
            shutil.copy2(resolve(args.prompts.parent, row['label']), args.out/'labels'/f'{name}.txt')
            label_origin = 'existing_detection_label_copy'
        else:
            text = ''
            if len(xs):
                x1,y1,x2,y2 = xs.min(),ys.min(),xs.max()+1,ys.max()+1
                text = f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n'
            (args.out/'labels'/f'{name}.txt').write_text(text, encoding='utf-8')
            label_origin = 'sam_proposal_not_reviewed'
        cards.append(preview(image, mask, row, args.out/'previews'/f'{name}.jpg'))
        record = dict(row, image=f'images/{name}{source.suffix.lower()}', mask=f'masks/{name}.png',
                      label=f'labels/{name}.txt', target_label_index=row.get('target_label_index',0),
                      source_image=str(source), source_image_sha256=sha256(source),
                      mask_sha256=sha256(args.out/'masks'/f'{name}.png'), label_origin=label_origin,
                      annotation_status='sam_candidate_needs_human_review', training_eligible=False,
                      sam_predicted_iou=float(scores[choice]), score_is_measured_accuracy=False,
                      mask_area_pixels=int(mask.sum()), image_area_fraction=float(mask.mean()),
                      seconds=time.perf_counter()-t, device=device, flags=flags,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0)
        records.append(record)
        (args.out/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        predictor.reset_predictor()
        if device == 'cuda':
            torch.cuda.empty_cache()
        print(name, device, 'mask_pixels',int(mask.sum()),'peak_MiB',round(record['peak_allocated_bytes']/1024**2,1),flush=True)
    for offset in range(0,len(cards),4):
        group=cards[offset:offset+4]
        sheet=Image.new('RGB',(1080,280*len(group)),(230,230,230))
        for i,card in enumerate(group):sheet.paste(card,(0,i*280))
        sheet.save(args.out/f'contact_{offset//4+1:02d}.jpg')
    summary=dict(instances=len(records),human_approved=0,training_eligible=0,preflight=preflight,
                 checkpoint_sha256=CHECKPOINT_SHA256,device=device,fallbacks=fallbacks,
                 max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),
                 max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in records),
                 seconds=time.perf_counter()-start,batch_size=1,cuda_postprocessing=False)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=('auto','cpu'),default='auto')
    parser.add_argument('--limit',type=int,default=12)
    args=parser.parse_args()
    print(json.dumps(run(args),indent=2))


if __name__=='__main__':
    main()
