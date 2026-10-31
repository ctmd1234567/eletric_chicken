"""合并已核对的新场景标注和旧训练集，提高用户手工标注图片的采样占比。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.urban_data import main

if __name__ == '__main__':
    raise SystemExit(main())
