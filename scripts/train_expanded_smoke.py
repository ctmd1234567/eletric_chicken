"""Manual entry for the expanded fire/smoke dataset; starts no job until run."""
import argparse
from pathlib import Path

if __package__:
    from . import _bootstrap
else:
    import _bootstrap

from sentinel.paths import ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认使用本轮场景补充清单和用户已验证的本地微调权重；旧清单仍可显式选择。
    parser.add_argument('--dataset', choices=('focused', 'urban', 'balanced', 'full'), default='focused')
    parser.add_argument('--base-model', choices=('dfire_ludan', 'dfire_small_smoke', 'dfire_yolov8n'), default='dfire_ludan')
    parser.add_argument('--lr0', type=float, default=0.0001)
    parser.add_argument('--freeze', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--out', type=Path, default=ROOT/'runs/dfire_ebike_focused/experiment_002')
    parser.add_argument('--dry-run', action='store_true', help='Show the configuration without training')
    args = parser.parse_args(argv)
    if min(args.epochs, args.imgsz, args.batch) < 1:
        parser.error('epochs, imgsz and batch must be positive')
    if not 0 < args.lr0 < 1 or args.freeze < 0:
        parser.error('lr0 must be in (0,1); freeze must be nonnegative')
    folders={'focused':'datasets/ebike_focused_20261031_v2/merged',
             'urban':'datasets/urban_smoke_20261031/merged',
             'balanced':'datasets/acquisition_20261030/merged',
             'full':'datasets/acquisition_20261030/merged'}
    folder = ROOT/folders[args.dataset]
    balanced = args.dataset != 'full'
    data = folder/('data_balanced.yaml' if balanced else 'data.yaml')
    train = folder/('train_balanced.txt' if balanced else 'train.txt')
    if not data.is_file() or not train.is_file():
        parser.error('Expanded dataset has not been prepared')
    if args.dataset=='focused':
        from sentinel.training.focused_data import verify_runtime_pairs
        print(f'YOLO image/label pairs verified: {verify_runtime_pairs(folder.parent)}',flush=True)
    for name, path in [('train', train), ('val', folder/'val.txt'), ('test', folder/'test.txt')]:
        print(f'{name}: {len(path.read_text(encoding="utf-8").splitlines())} images', flush=True)
    print(f'data: {data}\nout: {args.out}\nepochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}', flush=True)
    from sentinel.models.catalog import MODELS
    from sentinel.models.weights import verify
    entry = MODELS[args.base_model]
    if not verify(ROOT/entry['dest'], entry):
        parser.error('Base model missing or hash mismatch')
    print(f'base_model={args.base_model}, lr0={args.lr0}, freeze={args.freeze}', flush=True)
    print('The workflow evaluates the selected base model before training and compares the best model afterward.', flush=True)
    if args.dry_run:
        return 0
    from sentinel.training.dfire_small_smoke import main as train_main
    return train_main(['--data', str(data), '--out', str(args.out), '--epochs', str(args.epochs),
                       '--imgsz', str(args.imgsz), '--batch', str(args.batch),
                       '--base-model', args.base_model, '--lr0', str(args.lr0), '--freeze', str(args.freeze)])


if __name__ == '__main__':
    raise SystemExit(main())
