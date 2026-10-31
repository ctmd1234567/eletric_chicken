"""匹配本地原图并检查 CVAT 导出的 YOLO Detection 标注，不自动开训。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.annotation_import import main

if __name__ == '__main__':
    raise SystemExit(main())
