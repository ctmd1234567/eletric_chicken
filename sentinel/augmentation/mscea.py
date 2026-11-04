"""Detection-oriented adaptation of MSCEA's constrained MLS search.

This module contains the optimizer-side problem definition. A detector adapter
must provide per-instance features, matched-class confidence, and the strongest
new background detection outside the local ROI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .das_de import DASDEResult, Evaluation, optimize_das_de
from .mls import WarpResult, _bottom_anchor_scale, local_contour_controls, warp_local_instance


@dataclass
class MicroscaleRegion:
    """Union of alpha-neighborhood boxes centered on the origin and feasible points."""
    origin: np.ndarray
    radius: np.ndarray
    max_boxes: int = 256
    hard_lower: np.ndarray | None = None
    hard_upper: np.ndarray | None = None

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        self.radius = np.asarray(self.radius, dtype=np.float64).reshape(-1)
        if self.origin.shape != self.radius.shape or self.origin.size == 0:
            raise ValueError("origin and radius must have the same non-empty shape")
        if not np.isfinite(self.origin).all() or not np.isfinite(self.radius).all():
            raise ValueError("region coordinates must be finite")
        if np.any(self.radius <= 0):
            raise ValueError("all region radii must be positive")
        if self.max_boxes < 1:
            raise ValueError("max_boxes must be positive")
        if (self.hard_lower is None) != (self.hard_upper is None):
            raise ValueError("both hard bounds must be provided")
        if self.hard_lower is not None:
            self.hard_lower = np.asarray(self.hard_lower, dtype=np.float64).reshape(-1)
            self.hard_upper = np.asarray(self.hard_upper, dtype=np.float64).reshape(-1)
            if (self.hard_lower.shape != self.origin.shape or self.hard_upper.shape != self.origin.shape
                    or not np.isfinite(self.hard_lower).all() or not np.isfinite(self.hard_upper).all()
                    or np.any(self.hard_lower > self.hard_upper)
                    or np.any(self.origin < self.hard_lower) or np.any(self.origin > self.hard_upper)):
                raise ValueError("invalid absolute search bounds")
        self.centers = [self.origin.copy()]

    def _box(self, center):
        lower, upper = center - self.radius, center + self.radius
        if self.hard_lower is not None:
            lower = np.maximum(lower, self.hard_lower)
            upper = np.minimum(upper, self.hard_upper)
        return lower, upper

    @property
    def bounds(self):
        lows, highs = zip(*(self._box(center) for center in self.centers))
        return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)

    def contains(self, vector):
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        if x.shape != self.origin.shape or not np.isfinite(x).all():
            return False
        if self.hard_lower is not None and (np.any(x < self.hard_lower) or np.any(x > self.hard_upper)):
            return False
        return any(np.all(np.abs(x - center) <= self.radius + 1e-10)
                   for center in self.centers)

    def sample(self, rng: np.random.Generator):
        center = self.centers[int(rng.integers(len(self.centers)))]
        return rng.uniform(*self._box(center))

    def repair(self, vector):
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        if x.shape != self.origin.shape or not np.isfinite(x).all():
            raise ValueError("candidate dimension does not match microscale region")
        if self.contains(x):
            return x.copy()
        # Project to the nearest box in normalized coordinates. This preserves
        # the union of neighborhoods rather than filling its entire bounding hull.
        best, best_distance = None, float("inf")
        scale = np.maximum(self.radius, 1e-12)
        for center in self.centers:
            projected = np.clip(x, *self._box(center))
            distance = float(np.sum(((x - projected) / scale) ** 2))
            if distance < best_distance:
                best, best_distance = projected, distance
        return np.asarray(best, dtype=np.float64)

    def expand(self, feasible_vector):
        center = np.asarray(feasible_vector, dtype=np.float64).reshape(-1)
        if center.shape != self.origin.shape or not np.isfinite(center).all():
            raise ValueError("feasible vector has invalid shape or values")
        if self.hard_lower is not None and (np.any(center < self.hard_lower) or np.any(center > self.hard_upper)):
            raise ValueError("feasible vector lies outside absolute search bounds")
        # Add a neighborhood around each distinct feasible point, even when
        # its center lies inside an older box: the new box can still extend the
        # union beyond that older box.
        if any(np.all(np.abs(center - old) <= 1e-6 * self.radius)
               for old in self.centers):
            return
        if len(self.centers) >= self.max_boxes:
            return
        self.centers.append(center.copy())


@dataclass
class AugmentedCandidate:
    offsets: np.ndarray
    warp: WarpResult
    diversity: float
    confidence: float
    background_confidence: float
    fire_confidence: float
    constraint_components: dict[str, float]


@dataclass
class MSCEAInstanceResult:
    candidates: list[AugmentedCandidate]
    optimizer: DASDEResult
    initial_control_points: int
    initial_roi_xyxy: tuple[int, int, int, int]
    expanded_region_boxes: int
    evaluation_history: list[dict[str, Any]]


def feature_diversity(candidate_feature, reference_features) -> float:
    """Sum of per-dimension population variances after adding the candidate."""
    feature = np.asarray(candidate_feature, dtype=np.float64).reshape(-1)
    bank = np.asarray(reference_features, dtype=np.float64)
    if bank.ndim != 2 or bank.shape[0] == 0:
        raise ValueError("reference_features must be a non-empty 2D feature bank")
    if bank.shape[1] != feature.size:
        raise ValueError(
            f"feature dimension mismatch: bank={bank.shape[1]}, candidate={feature.size}"
        )
    if not np.isfinite(bank).all() or not np.isfinite(feature).all():
        raise ValueError("feature vectors must be finite")
    joined = np.vstack((bank, feature[None, :]))
    return float(np.var(joined, axis=0, ddof=0).sum())


def detection_constraint_violation(
    *,
    original_confidence: float,
    candidate_confidence: float,
    background_confidence: float,
    fire_confidence: float,
    area_ratio: float,
    bottom_shift_ratio: float,
    background_threshold: float = 0.10,
    fire_threshold: float = 0.10,
    area_ratio_range: tuple[float, float] = (0.70, 1.40),
    max_bottom_shift_ratio: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """Return C1-C5 violations; C5 limits new fire detections anywhere."""
    values = (original_confidence, candidate_confidence, background_confidence, fire_confidence,
              area_ratio, bottom_shift_ratio)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("detection and geometry metrics must be finite")
    if not 0 <= original_confidence <= 1 or not 0 <= candidate_confidence <= 1:
        raise ValueError("class confidence must be in [0, 1]")
    if not 0 <= background_confidence <= 1:
        raise ValueError("background confidence must be in [0, 1]")
    if not 0 <= fire_confidence <= 1:
        raise ValueError("fire confidence must be in [0, 1]")
    if not 0 <= background_threshold <= 1 or not 0 <= fire_threshold <= 1:
        raise ValueError("false-positive thresholds must be in [0, 1]")
    area_min, area_max = area_ratio_range
    if not 0 < area_min <= area_max or max_bottom_shift_ratio <= 0:
        raise ValueError("invalid geometry constraint settings")

    c1 = max(0.0, original_confidence - candidate_confidence)
    c2 = max(0.0, background_confidence - background_threshold)
    c3 = (
        max(0.0, area_min - area_ratio) / area_min
        + max(0.0, area_ratio - area_max) / area_max
    )
    c4 = max(0.0, bottom_shift_ratio - max_bottom_shift_ratio) / max_bottom_shift_ratio
    c5 = max(0.0, fire_confidence - fire_threshold)
    components = {"class_confidence": c1, "background_false_positive": c2,
                  "mask_area": c3, "bottom_edge": c4, "fire_false_positive": c5}
    return float(sum(components.values())), components


def optimize_mls_instance(
    image: np.ndarray,
    mask: np.ndarray,
    reference_features: np.ndarray,
    evaluate_candidate: Callable[[np.ndarray, np.ndarray, tuple, tuple], dict[str, Any]],
    *,
    original_confidence: float,
    population_size: int = 30,
    max_evals: int = 300,
    correlation_evals: int | None = None,
    alpha: float = 0.07,
    roi_scale: float = 2.0,
    contour_points: int = 32,
    bottom_anchor_strength: float = 0.75,
    area_ratio_range: tuple[float, float] = (0.70, 1.40),
    max_bottom_shift_ratio: float = 0.05,
    background_threshold: float = 0.10,
    fire_threshold: float = 0.10,
    seed: int = 20261103,
    top_k: int = 3,
) -> MSCEAInstanceResult:
    """Search for up to top_k feasible local-MLS variants of one smoke instance.

    evaluate_candidate(warped_image, warped_mask, bbox_xyxy, roi_xyxy) must
    return: feature (1D), confidence (matched smoke detection), and
    background_confidence (strongest new smoke detection crossing/outside ROI),
    and fire_confidence (strongest new fire detection anywhere). All confidence
    fields are required. The feature bank must contain same-class, same-size-bucket
    reference instances extracted by the same frozen detector.
    """
    if not 0 < alpha <= 0.25:
        raise ValueError("alpha must be in (0, 0.25]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 <= original_confidence <= 1:
        raise ValueError("original_confidence must be in [0, 1]")

    contour, local_bbox, roi_xyxy = local_contour_controls(
        mask, roi_scale=roi_scale, contour_points=contour_points
    )
    rx1, ry1, rx2, ry2 = roi_xyxy
    roi_width, roi_height = rx2 - rx1, ry2 - ry1
    radius = np.tile(np.asarray([alpha * roi_width, alpha * roi_height]), len(contour))
    scale = _bottom_anchor_scale(contour, local_bbox, bottom_anchor_strength)[:, None]
    # Limit the encoded offsets, rather than allowing the expanded search to
    # grow arbitrarily while the image decoder silently clips all controls.
    hard_lower = np.divide(-contour, scale, out=np.zeros_like(contour), where=scale > 0)
    hard_upper = np.divide(np.array([roi_width - 1, roi_height - 1]) - contour, scale,
                           out=np.zeros_like(contour), where=scale > 0)
    region = MicroscaleRegion(np.zeros(radius.size, dtype=np.float64), radius,
                              max_boxes=max_evals + 1,
                              hard_lower=hard_lower.ravel(), hard_upper=hard_upper.ravel())

    # Keep only the best top_k feasible images in memory. Retaining every
    # feasible 960px candidate can consume several gigabytes per source image.
    feasible_archive: dict[bytes, AugmentedCandidate] = {}
    evaluation_history = []

    def evaluate(offset_vector):
        offsets = np.asarray(offset_vector, dtype=np.float64).reshape(-1, 2)
        try:
            warped = warp_local_instance(
                image, mask, roi_scale=roi_scale, alpha=alpha,
                contour_points=contour_points,
                bottom_anchor_strength=bottom_anchor_strength,
                area_ratio_range=area_ratio_range,
                max_bottom_shift_ratio=max_bottom_shift_ratio,
                contour_offsets=offsets,
                validate_geometry=False,
            )
            metrics = evaluate_candidate(
                warped.image, warped.mask, warped.bbox_xyxy, warped.roi_xyxy
            )
            feature = np.asarray(metrics["feature"], dtype=np.float64).reshape(-1)
            diversity = feature_diversity(feature, reference_features)
            violation, components = detection_constraint_violation(
                original_confidence=original_confidence,
                candidate_confidence=float(metrics["confidence"]),
                background_confidence=float(metrics["background_confidence"]),
                fire_confidence=float(metrics["fire_confidence"]),
                area_ratio=warped.bbox_area_ratio,
                bottom_shift_ratio=warped.bottom_shift_ratio,
                background_threshold=background_threshold,
                fire_threshold=fire_threshold,
                area_ratio_range=area_ratio_range,
                max_bottom_shift_ratio=max_bottom_shift_ratio,
            )
            payload = {
                "warp": warped,
                "feature": feature,
                "diversity": diversity,
                "confidence": float(metrics["confidence"]),
                "background_confidence": float(metrics["background_confidence"]),
                "fire_confidence": float(metrics["fire_confidence"]),
                "constraint_components": components,
                "mask_area_ratio": warped.area_ratio,
                "bbox_area_ratio": warped.bbox_area_ratio,
            }
            evaluation_history.append({
                "evaluation": len(evaluation_history) + 1,
                "objective": -diversity, "violation": violation,
                "feasible": violation == 0.0,
                "diversity": diversity,
                "confidence": float(metrics["confidence"]),
                "background_confidence": float(metrics["background_confidence"]),
                "fire_confidence": float(metrics["fire_confidence"]),
                "constraint_components": components,
                "mask_area_ratio": warped.area_ratio,
                "bbox_area_ratio": warped.bbox_area_ratio,
                "bottom_shift_ratio": warped.bottom_shift_ratio,
                "region_boxes_before_expansion": len(region.centers),
            })
            # Only the feasible callback needs pixels. Population members retain
            # scalar diagnostics, never full frames or feature tensors.
            if violation > 0.0:
                payload.pop("warp")
            payload.pop("feature")
            return Evaluation(objective=-diversity, violation=violation, payload=payload)
        except (ValueError, KeyError, FloatingPointError) as exc:
            # Invalid masks or detector metrics are infeasible, not silently
            # discarded as if they had never consumed a fitness evaluation.
            evaluation_history.append({"evaluation": len(evaluation_history) + 1,
                                       "objective": 1e12, "violation": 1e6,
                                       "feasible": False, "invalid_candidate": str(exc),
                                       "region_boxes_before_expansion": len(region.centers)})
            return Evaluation(objective=1e12, violation=1e6,
                              payload={"invalid_candidate": str(exc)})

    def remember_feasible(vector, evaluation):
        if not isinstance(evaluation.payload, dict) or "warp" not in evaluation.payload:
            return
        payload = evaluation.payload
        key = np.round(vector, 7).tobytes()
        feasible_archive[key] = AugmentedCandidate(
            offsets=vector.reshape(-1, 2).copy(),
            warp=payload.pop("warp"),
            diversity=float(payload["diversity"]),
            confidence=float(payload["confidence"]),
            background_confidence=float(payload["background_confidence"]),
            fire_confidence=float(payload["fire_confidence"]),
            constraint_components=dict(payload["constraint_components"]),
        )
        if len(feasible_archive) > top_k:
            worst_key = min(feasible_archive, key=lambda item: feasible_archive[item].diversity)
            del feasible_archive[worst_key]

    optimizer = optimize_das_de(
        evaluate,
        bounds=(-radius, radius),
        max_evals=max_evals,
        population_size=population_size,
        correlation_evals=correlation_evals,
        seed=seed,
        region=region,
        feasible_callback=remember_feasible,
    )
    correlation_end = max((h["evaluations"] for h in optimizer.history
                           if h["stage"] == "correlation"), default=population_size // 2)
    for entry in evaluation_history:
        entry["stage"] = ("initialization" if entry["evaluation"] <= population_size // 2
                          else "correlation" if entry["evaluation"] <= correlation_end
                          else "optimization")

    # The callback maintains a bounded elitist archive of feasible candidates.
    candidates = sorted(
        feasible_archive.values(), key=lambda item: item.diversity, reverse=True
    )[:top_k]
    return MSCEAInstanceResult(
        candidates=candidates,
        optimizer=optimizer,
        initial_control_points=len(contour),
        initial_roi_xyxy=roi_xyxy,
        expanded_region_boxes=len(region.centers),
        evaluation_history=evaluation_history,
    )
