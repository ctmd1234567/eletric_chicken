"""Detection-specific augmentation operators."""

from .mls import WarpResult, mask_to_bbox, warp_local_instance

__all__ = ["WarpResult", "mask_to_bbox", "warp_local_instance"]
