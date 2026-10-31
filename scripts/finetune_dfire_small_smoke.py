"""本地 D-Fire 小烟雾微调入口；不覆盖原始权重和默认推理配置。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.dfire_small_smoke import main

if __name__ == '__main__':
    raise SystemExit(main())
