"""命令行入口；实现见 sentinel/models/verify.py。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.models.verify import main

if __name__ == "__main__":
    raise SystemExit(main())
