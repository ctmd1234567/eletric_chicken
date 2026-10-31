"""本轮人工标注与干扰回放对比；实现位于 sentinel/training/focused_evaluation.py。"""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.focused_evaluation import main

if __name__ == "__main__":
    raise SystemExit(main())
