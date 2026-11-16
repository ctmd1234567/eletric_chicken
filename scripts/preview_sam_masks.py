"""Run a small SAM 2.1 mask review without modifying training data."""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.sam_mask_review import main

if __name__ == '__main__':
    main()
