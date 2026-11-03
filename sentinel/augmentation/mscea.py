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
from .mls import WarpResult, local_contour_controls, warp_local_instance


@dataclass
class MicroscaleRegion:
    """Union of alpha-neighborhood boxes centered on the origin and feasible points."""
    origin: np.ndarray
    radius: np.ndarray
    max_boxes: int = 256

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        self.radius = np.asarray(self.radius, dtype=np.float64).reshape(-1)
        if self.origin.shape != self.radius.shape or self.origin.size == 0:
            raise ValueError("origin and radius must have the same non-empty shape")
        if not np.isfinite(self.origin).all() or not np.isfinite(self.radius).all():
            raise ValueError("region coordinates must be finite")
        if np.any(self.radius <= 0):
            raise ValueError("all region radii must be positive")
        self.centers = [self.origin.copy()]

    @property
    def bounds(self):
        lows = [center - self.radius for center in self.centers]
        highs = [center + self.radius for center in self.centers]
        return np.min(np.stack(lows), axis=0), np.max(np.stack(highs), axis=0)

    def contains(self, vector):
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        if x.shape != self.origin.shape:
            return False
        return any(np.all(np.abs(x - center) <= self.radius + 1e-10)
                   for center in self.centers)

    def sample(self, rng: np.random.Generator):
        center = self.centers[int(rng.integers(len(self.centers)))]
        return rng.uniform(center - self.radius, center + self.radius)

    def repair(self, vector):
        x = np.asarray(vector, dtype=np.float64).reshape(-1)
        if x.shape != self.origin.shape:
            raise ValueError("candidate dimension does not match microscale region")
        if self.contains(x):
            return x.copy()
        # Project to the nearest box in normalized coordinates. This preserves
        # the union of neighborhoods rather than filling its entire bounding hull.
        best, best_distance = None, float("inf")
        scale = np.maximum(self.radius, 1e-12)
        for center in self.centers:
            projected = np.clip(x, center - self.radius, center + self.radius)
            distance = float(np.sum(((x - projected) / scale) ** 2))
            if distance < best_distance:
                best, best_distance = projected, distance
        return np.asarray(best, dtype=np.float64)

    def expand(self, feasible_vector):
        center = np.asarray(feasible_vector, dtype=np.float64).reshape(-1)
        if center.shape != self.origin.shape or not np.isfinite(center).all():
            raise ValueError("feasible vector has invalid shape or values")
        if self.contains(center):
            # A feasible point already inside an existing box does not enlarge
            # the union; it is still retained by the optimizer's feasible archive.
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
    constraint_components: dict[str, float]


@dataclass
class MSCEAInstanceResult:
    candidates: list[AugmentedCandidate]
    optimizer: DASDEResult
    initial_control_points: int
    initial_roi_xyxy: tuple[int, int, int, int]
    expanded_region_boxes: int


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
    area_ratio: float,
    bottom_shift_ratio: float,
    background_threshold: float = 0.10,
    area_ratio_range: tuple[float, float] = (0.70, 1.40),
    max_bottom_shift_ratio: float = 0.05,
) -> tuple[float, dict[str, float]]:
    """Return normalized C1-C4 violations; feasibility requires every Cj == 0."""
    values = (original_confidence, candidate_confidence, background_confidence,
              area_ratio, bottom_shift_ratio)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("detection and geometry metrics must be finite")
    if not 0 <= original_confidence <= 1 or not 0 <= candidate_confidence <= 1:
        raise ValueError("class confidence must be in [0, 1]")
    if not 0 <= background_confidence <= 1:
        raise ValueError("background confidence must be in [0, 1]")
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
    components = {"class_confidence": c1, "background_false_positive": c2,
                  "mask_area": c3, "bottom_edge": c4}
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
    seed: int = 20261103,
    top_k: int = 3,
) -> MSCEAInstanceResult:
    """Search for up to top_k feasible local-MLS variants of one smoke instance.

    evaluate_candidate(warped_image, warped_mask, bbox_xyxy, roi_xyxy) must
    return: feature (1D), confidence (matched smoke detection), and
    background_confidence (strongest newly introduced smoke detection outside
    the local ROI). The feature bank must contain same-class, same-size-bucket
    reference instances extracted by the same frozen detector.
    """
    if not 0 < alpha <= 0.25:
        raise ValueError("alpha must be in (0, 0.25]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 <= original_confidence <= 1:
        raise ValueError("original_confidence must be in [0, 1]")

    contour, _, roi_xyxy = local_contour_controls(
        mask, roi_scale=roi_scale, contour_points=contour_points
    )
    rx1, ry1, rx2, ry2 = roi_xyxy
    roi_width, roi_height = rx2 - rx1, ry2 - ry1
    radius = np.tile(np.asarray([alpha * roi_width, alpha * roi_height]), len(contour))
    region = MicroscaleRegion(np.zeros(radius.size, dtype=np.float64), radius)

    feasible_archive: list[tuple[np.ndarray, Evaluation]] = []

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
                area_ratio=warped.area_ratio,
                bottom_shift_ratio=warped.bottom_shift_ratio,
                background_threshold=background_threshold,
                area_ratio_range=area_ratio_range,
                max_bottom_shift_ratio=max_bottom_shift_ratio,
            )
            payload = {
                "warp": warped,
                "feature": feature,
                "diversity": diversity,
                "confidence": float(metrics["confidence"]),
                "background_confidence": float(metrics["background_confidence"]),
                "constraint_components": components,
            }
            return Evaluation(objective=-diversity, violation=violation, payload=payload)
        except (ValueError, KeyError, FloatingPointError) as exc:
            # Invalid masks or detector metrics are infeasible, not silently
            # discarded as if they had never consumed a fitness evaluation.
            return Evaluation(objective=1e12, violation=1e6,
                              payload={"invalid_candidate": str(exc)})

    def remember_feasible(vector, evaluation):
        if not isinstance(evaluation.payload, dict) or "warp" not in evaluation.payload:
            return
        feasible_archive.append((vector.copy(), evaluation))

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

    # The DAS-DE population is not an elitist archive of every feasible trial.
    # Preserve all feasible evaluations so top-k output cannot be lost later.
    unique: dict[bytes, AugmentedCandidate] = {}
    for vector, evaluation in feasible_archive:
        payload = evaluation.payload
        key = np.round(vector, 7).tobytes()
        unique[key] = AugmentedCandidate(
            offsets=vector.reshape(-1, 2).copy(),
            warp=payload["warp"],
            diversity=float(payload["diversity"]),
            confidence=float(payload["confidence"]),
            background_confidence=float(payload["background_confidence"]),
            constraint_components=dict(payload["constraint_components"]),
        )
    candidates = sorted(unique.values(), key=lambda item: item.diversity, reverse=True)[:top_k]
    return MSCEAInstanceResult(
        candidates=candidates,
        optimizer=optimizer,
        initial_control_points=len(contour),
        initial_roi_xyxy=roi_xyxy,
        expanded_region_boxes=len(region.centers),
    )
