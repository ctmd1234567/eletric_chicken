"""Local affine-MLS image warping for smoke-instance augmentation.

This is the B3 operator baseline, not the full MSCEA optimizer. It uses a
binary instance mask to define contour controls and fixed ROI-boundary anchors.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class WarpResult:
    image: np.ndarray
    mask: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    roi_xyxy: tuple[int, int, int, int]
    area_ratio: float
    bottom_shift_ratio: float
    control_points: int


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return a binary mask's tight bbox as (x1, y1, x2, y2), x2/y2 exclusive."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Instance mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _sample_contour(mask: np.ndarray, count: int) -> np.ndarray:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("No external contour found in instance mask")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < 4:
        raise ValueError("Instance contour has fewer than four points")
    closed = np.vstack((contour, contour[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    perimeter = float(cumulative[-1])
    if perimeter <= 0:
        raise ValueError("Instance contour has zero perimeter")
    targets = np.linspace(0.0, perimeter, num=max(4, count), endpoint=False)
    sampled = np.column_stack((
        np.interp(targets, cumulative, closed[:, 0]),
        np.interp(targets, cumulative, closed[:, 1]),
    ))
    rounded = np.round(sampled, 4)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    sampled = sampled[np.sort(indices)]
    return sampled.astype(np.float32)


def _boundary_anchors(width: int, height: int, per_edge: int) -> np.ndarray:
    """Fixed controls around the ROI perimeter prevent distant background drift."""
    if width < 2 or height < 2:
        raise ValueError("ROI must be at least 2x2")
    n = max(3, int(per_edge))
    xs = np.linspace(0, width - 1, n, dtype=np.float32)
    ys = np.linspace(0, height - 1, n, dtype=np.float32)
    points = np.concatenate((
        np.column_stack((xs, np.zeros_like(xs))),
        np.column_stack((xs, np.full_like(xs, height - 1))),
        np.column_stack((np.zeros_like(ys), ys)),
        np.column_stack((np.full_like(ys, width - 1), ys)),
    ))
    return np.unique(points, axis=0).astype(np.float32)


def _mls_affine_map(
    query_xy: np.ndarray,
    control_src_xy: np.ndarray,
    control_dst_xy: np.ndarray,
    *,
    alpha: float = 1.0,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Evaluate the affine moving-least-squares map at query points.

    All coordinates use (x, y). This routine maps control_src_xy to
    control_dst_xy. Call it with destination controls as the domain and
    original controls as the range to build an inverse remap for cv2.remap.
    """
    query = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    p = np.asarray(control_src_xy, dtype=np.float64).reshape(-1, 2)
    q = np.asarray(control_dst_xy, dtype=np.float64).reshape(-1, 2)
    if p.shape != q.shape or p.ndim != 2 or p.shape[0] < 3:
        raise ValueError("MLS requires at least three paired 2D control points")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(query).all():
        raise ValueError("MLS coordinates must be finite")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    mapped = np.empty_like(query)
    eps = 1e-10
    for start in range(0, len(query), chunk_size):
        x = query[start:start + chunk_size]
        delta = x[:, None, :] - p[None, :, :]
        dist2 = np.einsum("bni,bni->bn", delta, delta)
        exact = dist2 <= eps
        weights = 1.0 / np.maximum(dist2, eps) ** alpha
        weight_sum = weights.sum(axis=1, keepdims=True)
        p_star = weights @ p / weight_sum
        q_star = weights @ q / weight_sum
        p_hat = p[None, :, :] - p_star[:, None, :]
        q_hat = q[None, :, :] - q_star[:, None, :]
        covariance = np.einsum("bn,bni,bnj->bij", weights, q_hat, p_hat)
        scatter = np.einsum("bn,bni,bnj->bij", weights, p_hat, p_hat)
        affine = covariance @ np.linalg.pinv(scatter, rcond=1e-10)
        y = np.einsum("bij,bj->bi", affine, x - p_star) + q_star

        exact_rows = exact.any(axis=1)
        if exact_rows.any():
            nearest = np.argmax(exact[exact_rows], axis=1)
            y[exact_rows] = q[nearest]
        mapped[start:start + len(x)] = y
    return mapped.astype(np.float32)


def _bottom_anchor_scale(
    points: np.ndarray,
    mask_bbox: tuple[int, int, int, int],
    bottom_anchor_strength: float,
) -> np.ndarray:
    if not 0 <= bottom_anchor_strength <= 1:
        raise ValueError("bottom_anchor_strength must be in [0, 1]")
    _, y1, _, y2 = mask_bbox
    mask_height = max(1.0, float(y2 - y1))
    bottom_start = y2 - 0.2 * mask_height
    ramp = np.clip((points[:, 1] - bottom_start) / max(1.0, 0.2 * mask_height), 0, 1)
    return 1.0 - bottom_anchor_strength * ramp


def local_contour_controls(
    mask: np.ndarray,
    *,
    roi_scale: float = 2.0,
    contour_points: int = 32,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return deterministic local contour controls, local bbox, and global ROI."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be a single-channel image")
    x1, y1, x2, y2 = mask_to_bbox(binary)
    if roi_scale <= 1:
        raise ValueError("roi_scale must be greater than 1")
    height, width = binary.shape
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    roi_w = max(x2 - x1 + 2, int(math.ceil((x2 - x1) * roi_scale)))
    roi_h = max(y2 - y1 + 2, int(math.ceil((y2 - y1) * roi_scale)))
    rx1 = max(0, int(math.floor(cx - roi_w / 2)))
    ry1 = max(0, int(math.floor(cy - roi_h / 2)))
    rx2 = min(width, rx1 + roi_w)
    ry2 = min(height, ry1 + roi_h)
    rx1 = max(0, rx2 - roi_w)
    ry1 = max(0, ry2 - roi_h)
    roi_w, roi_h = rx2 - rx1, ry2 - ry1
    if roi_w < 8 or roi_h < 8:
        raise ValueError("Local ROI is too small for stable MLS")
    if x1 < rx1 or y1 < ry1 or x2 > rx2 or y2 > ry2:
        raise ValueError("Instance mask does not fit inside the selected ROI")
    local_bbox = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)
    contour = _sample_contour(binary[ry1:ry2, rx1:rx2], contour_points)
    return contour, local_bbox, (rx1, ry1, rx2, ry2)


def _random_contour_targets(
    points: np.ndarray,
    mask_bbox: tuple[int, int, int, int],
    roi_width: int,
    roi_height: int,
    rng: np.random.Generator,
    *,
    alpha: float,
    bottom_anchor_strength: float,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    if not 0 <= bottom_anchor_strength <= 1:
        raise ValueError("bottom_anchor_strength must be in [0, 1]")
    dx = rng.uniform(-alpha * roi_width, alpha * roi_width, len(points))
    dy = rng.uniform(-alpha * roi_height, alpha * roi_height, len(points))
    if len(points) >= 5 and smooth_sigma > 0:
        radius = max(1, int(math.ceil(2 * smooth_sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / smooth_sigma) ** 2)
        kernel /= kernel.sum()
        dx = sum(weight * np.roll(dx, int(offset)) for weight, offset in zip(kernel, offsets))
        dy = sum(weight * np.roll(dy, int(offset)) for weight, offset in zip(kernel, offsets))
    _, y1, _, y2 = mask_bbox
    mask_height = max(1.0, float(y2 - y1))
    bottom_start = y2 - 0.2 * mask_height
    ramp = np.clip((points[:, 1] - bottom_start) / max(1.0, 0.2 * mask_height), 0, 1)
    scale = 1.0 - bottom_anchor_strength * ramp
    moved = points.copy()
    moved[:, 0] += dx * scale
    moved[:, 1] += dy * scale
    return moved


def warp_local_instance(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
    *,
    roi_scale: float = 2.0,
    alpha: float = 0.07,
    contour_points: int = 32,
    boundary_points_per_edge: int = 12,
    bottom_anchor_strength: float = 0.75,
    area_ratio_range: tuple[float, float] = (0.7, 1.4),
    max_bottom_shift_ratio: float = 0.05,
    blend_radius: int = 8,
    blend_sigma: float = 3.0,
    contour_offsets: np.ndarray | None = None,
    validate_geometry: bool = True,
) -> WarpResult:
    """Warp one masked smoke instance inside a local ROI and preserve its bbox.

    The image outside the ROI is copied byte-for-byte. The returned bbox is
    recalculated from the warped mask; callers should replace the corresponding
    YOLO detection label with this bbox.
    """
    image = np.asarray(image)
    mask = np.asarray(mask)
    if image.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 pixel values")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError("mask must be a single-channel image matching image size")
    if not np.isfinite(mask).all():
        raise ValueError("mask must contain finite values")
    binary = (mask > 0).astype(np.uint8)
    x1, y1, x2, y2 = mask_to_bbox(binary)
    if roi_scale <= 1:
        raise ValueError("roi_scale must be greater than 1")
    if alpha <= 0 or alpha > 0.25:
        raise ValueError("alpha must be in (0, 0.25]")
    if not 0 < area_ratio_range[0] <= area_ratio_range[1]:
        raise ValueError("Invalid area_ratio_range")
    if max_bottom_shift_ratio < 0:
        raise ValueError("max_bottom_shift_ratio must be nonnegative")
    if blend_radius < 0 or blend_sigma <= 0:
        raise ValueError("blend_radius must be nonnegative and blend_sigma positive")

    contour, local_bbox, roi_xyxy = local_contour_controls(
        binary, roi_scale=roi_scale, contour_points=contour_points
    )
    rx1, ry1, rx2, ry2 = roi_xyxy
    roi_w, roi_h = rx2 - rx1, ry2 - ry1
    image_roi = image[ry1:ry2, rx1:rx2]
    mask_roi = binary[ry1:ry2, rx1:rx2]
    generator = rng if rng is not None else np.random.default_rng()
    if contour_offsets is None:
        moved = _random_contour_targets(
            contour, local_bbox, roi_w, roi_h, generator,
            alpha=alpha, bottom_anchor_strength=bottom_anchor_strength,
        )
    else:
        offsets = np.asarray(contour_offsets, dtype=np.float32).reshape(-1, 2)
        if offsets.shape != contour.shape or not np.isfinite(offsets).all():
            raise ValueError(
                f"contour_offsets must have shape {contour.shape}, got {offsets.shape}"
            )
        scale = _bottom_anchor_scale(contour, local_bbox, bottom_anchor_strength)
        moved = contour + offsets * scale[:, None]
    anchors = _boundary_anchors(roi_w, roi_h, boundary_points_per_edge)
    source_controls = np.vstack((contour, anchors))
    target_controls = np.vstack((moved, anchors))

    grid_x, grid_y = np.meshgrid(
        np.arange(roi_w, dtype=np.float32),
        np.arange(roi_h, dtype=np.float32),
    )
    output_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    source_points = _mls_affine_map(output_points, target_controls, source_controls)
    map_x = source_points[:, 0].reshape(roi_h, roi_w)
    map_y = source_points[:, 1].reshape(roi_h, roi_w)

    warped_image_roi = cv2.remap(
        image_roi, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask_roi = cv2.remap(
        mask_roi, map_x, map_y, interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_mask_roi = (warped_mask_roi > 0).astype(np.uint8)
    new_local_bbox = mask_to_bbox(warped_mask_roi)
    old_area = int(binary.sum())
    new_area = int(warped_mask_roi.sum())
    area_ratio = new_area / max(1, old_area)
    bottom_shift = abs(new_local_bbox[3] - local_bbox[3]) / roi_h
    if validate_geometry and not area_ratio_range[0] <= area_ratio <= area_ratio_range[1]:
        raise ValueError(f"Mask area ratio {area_ratio:.3f} is outside allowed range")
    if validate_geometry and bottom_shift > max_bottom_shift_ratio:
        raise ValueError(f"Bottom-edge shift {bottom_shift:.3f} exceeds allowed range")

    union_mask = np.maximum(mask_roi, warped_mask_roi)
    if blend_radius:
        kernel_size = 2 * blend_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        support = cv2.dilate(union_mask, kernel)
    else:
        support = union_mask
    alpha_mask = cv2.GaussianBlur(
        support.astype(np.float32), (0, 0), sigmaX=blend_sigma, sigmaY=blend_sigma
    )
    alpha_mask = np.clip(alpha_mask, 0.0, 1.0)
    if image_roi.ndim == 3:
        alpha_mask = alpha_mask[..., None]
    blended = (
        warped_image_roi.astype(np.float32) * alpha_mask
        + image_roi.astype(np.float32) * (1.0 - alpha_mask)
    )
    blended = np.clip(np.rint(blended), 0, 255).astype(image.dtype)

    result_image = image.copy()
    result_mask = binary.copy()
    result_image[ry1:ry2, rx1:rx2] = blended
    result_mask[ry1:ry2, rx1:rx2] = warped_mask_roi
    bx1, by1, bx2, by2 = new_local_bbox
    return WarpResult(
        image=result_image,
        mask=result_mask,
        bbox_xyxy=(bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1),
        roi_xyxy=(rx1, ry1, rx2, ry2),
        area_ratio=float(area_ratio),
        bottom_shift_ratio=float(bottom_shift),
        control_points=int(len(source_controls)),
    )
