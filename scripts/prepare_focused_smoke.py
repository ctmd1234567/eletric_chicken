"""准备人工标注和 jjs 干扰重点采样清单，不启动训练。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.focused_data import main

if __name__ == '__main__':
    raise SystemExit(main())
