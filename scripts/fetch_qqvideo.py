"""命令行入口；实现见 sentinel/media/fetch_qqvideo.py。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.media.fetch_qqvideo import main

if __name__ == "__main__":
    raise SystemExit(main())
