"""Export and relocate the verified balanced dataset for a private Linux GPU instance."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

from sentinel.paths import ROOT
from sentinel.models.catalog import MODELS
from sentinel.models.weights import verify as verify_weights

DATA = Path('datasets/acquisition_20261030')


def portable(value):
    return str(value).replace('\\', '/')


def pack(output, data_root=None, base_model='dfire_ludan'):
    data_root = (data_root or ROOT/DATA).resolve()
    if not data_root.is_relative_to(ROOT.resolve()):
        raise ValueError('Dataset must be inside project')
    data_relative = data_root.relative_to(ROOT)
    merged = data_root/'merged'
    audit = json.loads((merged/'audit.json').read_text())
    if audit['status'] != 'passed':
        raise ValueError('Dataset audit has not passed')
    records = [json.loads(line) for line in (merged/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    splits = {name: (merged/filename).read_text(encoding='utf-8').splitlines() for name,filename in
              [('train','train_balanced.txt'),('val','val.txt'),('test','test.txt')]}
    selected_paths = {p for paths in splits.values() for p in paths}
    selected = [r for r in records if (data_root/portable(r['image'])).resolve().as_posix() in selected_paths]
    by_path = {(data_root/portable(r['image'])).resolve().as_posix():r for r in selected}
    if len(by_path) != len(selected_paths) or len(selected) != len(by_path):
        raise ValueError('Split lists and unique manifest records do not match')
    # 重复采样只保留在训练清单中；图片在包内只存一份，验证和测试禁止重复。
    if any(len(paths) != len(set(paths)) for name, paths in splits.items() if name != 'train'):
        raise ValueError('Duplicate holdout images')
    if any(set(splits[a]) & set(splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Cross-split path overlap')
    cloud_splits = {name:[portable((data_root/portable(by_path[p]['image'])).resolve().relative_to(ROOT)) for p in paths]
                    for name,paths in splits.items()}
    entry = MODELS[base_model]
    weights = ROOT/entry['dest']
    if not verify_weights(weights, entry):
        raise ValueError('Base model checksum mismatch')
    output = output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    files = []

    def add_bytes(archive, relative, data):
        name = portable(relative)
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Unsafe bundle path')
        info = tarfile.TarInfo('eletric_chicken/'+name)
        info.size = len(data);info.mode = 0o644;info.mtime = 0
        archive.addfile(info,io.BytesIO(data))
        files.append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))

    with tarfile.open(output,mode='x',format=tarfile.GNU_FORMAT) as archive:
        code = sorted((ROOT/'sentinel').rglob('*.py'))
        code += [ROOT/'scripts'/name for name in ['_bootstrap.py','train_expanded_smoke.py',
                'prepare_autodl.py','prepare_urban_smoke.py','prepare_focused_smoke.py',
                'import_annotations.py','verify_cuda.py','verify_models.py']]
        code += [ROOT/name for name in ['LICENSE','NOTICE.md','requirements-cloud.txt','install_cloud.sh','train_and_shutdown.sh']]
        code += [ROOT/'docs'/name for name in ['autodl_3080ti.md','expanded_training_runbook.md','training_data_expansion.md']]
        code += [p for p in [ROOT/'docs/urban_smoke_autodl.md'] if p.is_file()]
        code += [p for p in [ROOT/'docs/ebike_focused_autodl.md'] if p.is_file()]
        code.append(weights)
        # 保留新来源的许可快照和抽查记录，供私有云端训练追溯。
        code += sorted((data_root/'research').glob('*.md'))
        code += sorted((data_root/'research').glob('*.txt'))
        code += [p for p in [data_root/'review.json', data_root/'training_profile.json',data_root/'permission_record.json'] if p.is_file()]
        for path in code:
            add_bytes(archive,path.relative_to(ROOT),path.read_bytes())
        provenance = []
        for i, record in enumerate(selected,1):
            image = (data_root/portable(record['image'])).resolve();label = (data_root/portable(record['label'])).resolve()
            if not image.is_relative_to(ROOT.resolve()) or not label.is_relative_to(ROOT.resolve()):
                raise ValueError('Dataset payload outside project')
            image_data = image.read_bytes()
            if hashlib.sha256(image_data).hexdigest() != record['sha256']:
                raise ValueError('Image changed since audit: '+str(image))
            add_bytes(archive,image.relative_to(ROOT),image_data)
            label_data = label.read_bytes()
            if record.get('label_sha256') and hashlib.sha256(label_data).hexdigest() != record['label_sha256']:
                raise ValueError('Label changed since audit: '+str(label))
            add_bytes(archive,label.relative_to(ROOT),label_data)
            provenance.append(dict(record,image=portable(record['image']),label=portable(record['label'])))
            if i%10000 == 0: print('PACKED',i,'/',len(selected),flush=True)
        add_bytes(archive,'cloud_splits.json',json.dumps(cloud_splits,indent=2).encode())
        add_bytes(archive,data_relative/'merged/manifest.jsonl',
                  ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in provenance).encode())
        if (merged/'jpeg_repairs.json').is_file():
            add_bytes(archive,data_relative/'merged/jpeg_repairs.json',(merged/'jpeg_repairs.json').read_bytes())
        profile=json.loads((data_root/'training_profile.json').read_text(encoding='utf-8')) if (data_root/'training_profile.json').is_file() else {}
        summary = dict(dataset=profile.get('dataset','balanced'),images=len(selected),splits={k:len(v) for k,v in cloud_splits.items()},
                       data_dir=portable(data_relative), base_model=base_model,
                       base_sha256=entry['sha256'], local_dataset_audit=audit,full_dataset_included=False,
                       note='Private training transfer only; preserve NOTICE and per-source obligations.')
        add_bytes(archive,'cloud_bundle.json',json.dumps(summary,indent=2).encode())
        # This manifest intentionally excludes itself and covers every preceding payload.
        add_bytes(archive,'cloud_files.json',json.dumps(files,indent=2).encode())
    with output.open('rb') as stream:
        sha = hashlib.file_digest(stream,'sha256').hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_bytes(f'{sha}  {output.name}\n'.encode('ascii'))
    result=dict(archive=str(output.relative_to(ROOT)),bytes=output.stat().st_size,sha256=sha,
                images=len(selected),splits={k:len(v) for k,v in cloud_splits.items()})
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('BUNDLE_READY',json.dumps(result),flush=True)


def prepare(verify=False):
    splits = json.loads((ROOT/'cloud_splits.json').read_text())
    bundle = json.loads((ROOT/'cloud_bundle.json').read_text())
    if verify:
        files = json.loads((ROOT/'cloud_files.json').read_text())
        for i, entry in enumerate(files,1):
            path = (ROOT/entry['path']).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                raise ValueError('Missing or unsafe payload: '+entry['path'])
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream,'sha256').hexdigest()
            if sha != entry['sha256']: raise ValueError('Payload checksum mismatch: '+entry['path'])
            if i%20000 == 0: print('VERIFIED',i,'/',len(files),flush=True)
    data_root = (ROOT/bundle.get('data_dir', portable(DATA))).resolve()
    if not data_root.is_relative_to(ROOT.resolve()):
        raise ValueError('Unsafe dataset destination')
    merged = data_root/'merged'
    merged.mkdir(parents=True,exist_ok=True)
    for split, paths in splits.items():
        resolved = [(ROOT/p).resolve() for p in paths]
        if any(not p.is_relative_to(ROOT.resolve()) or not p.is_file() for p in resolved):
            raise ValueError('Missing image or path outside bundle')
        name = 'train_balanced.txt' if split=='train' else split+'.txt'
        (merged/name).write_text('\n'.join(p.as_posix() for p in resolved)+'\n',encoding='utf-8')
    config = 'path: '+json.dumps(merged.as_posix())+'\ntrain: train_balanced.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: smoke\n  1: fire\n'
    (merged/'data_balanced.yaml').write_text(config,encoding='utf-8')
    print('CLOUD_DATA_READY',json.dumps({k:len(v) for k,v in splits.items()}),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    export=sub.add_parser('pack');export.add_argument('--out',type=Path,default=ROOT/'outputs/cloud/autodl_ebike_focused_20261031.tar')
    export.add_argument('--data-root',type=Path,default=ROOT/'datasets/ebike_focused_20261031_v2')
    export.add_argument('--base-model',choices=('dfire_ludan','dfire_small_smoke','dfire_yolov8n'),default='dfire_ludan')
    relocate=sub.add_parser('prepare');relocate.add_argument('--verify',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='pack':pack(args.out,args.data_root,args.base_model)
    else:prepare(args.verify)
    return 0
