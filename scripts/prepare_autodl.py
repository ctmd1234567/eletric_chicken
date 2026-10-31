"""Build the local AutoDL upload bundle or relocate it after extraction."""
if __package__:
    from . import _bootstrap
else:
    import _bootstrap
from sentinel.training.cloud_bundle import main

if __name__ == '__main__':
    raise SystemExit(main())
