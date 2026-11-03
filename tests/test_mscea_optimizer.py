"""Tests for the constrained MSCEA search core."""
import unittest

import cv2
import numpy as np

from sentinel.augmentation.das_de import (
    Evaluation, Individual, _is_better, optimize_das_de,
)
from sentinel.augmentation.mls import local_contour_controls, warp_local_instance
from sentinel.augmentation.mscea import (
    MicroscaleRegion,
    detection_constraint_violation,
    feature_diversity,
    optimize_mls_instance,
)


class MicroscaleRegionTests(unittest.TestCase):
    def test_feasible_centres_expand_union_of_neighbourhoods(self):
        region = MicroscaleRegion(np.zeros(2), np.ones(2))
        region.expand(np.array([0.8, 0.0]))
        self.assertEqual(len(region.centers), 2)
        lower, upper = region.bounds
        np.testing.assert_allclose(lower, [-1.0, -1.0])
        np.testing.assert_allclose(upper, [1.8, 1.0])
        self.assertTrue(region.contains(np.array([1.7, 0.0])))
        projected = region.repair(np.array([5.0, 0.0]))
        self.assertTrue(region.contains(projected))

    def test_feature_diversity_and_constraint_components(self):
        bank = np.array([[0.0, 0.0], [2.0, 2.0]])
        score = feature_diversity(np.array([1.0, 3.0]), bank)
        self.assertAlmostEqual(score, float(np.var([[0, 0], [2, 2], [1, 3]], axis=0).sum()))
        violation, parts = detection_constraint_violation(
            original_confidence=0.8,
            candidate_confidence=0.7,
            background_confidence=0.3,
            area_ratio=1.5,
            bottom_shift_ratio=0.10,
        )
        self.assertGreater(violation, 0)
        self.assertAlmostEqual(parts["class_confidence"], 0.1)
        self.assertAlmostEqual(parts["background_false_positive"], 0.2)
        self.assertAlmostEqual(parts["mask_area"], (1.5 - 1.4) / 1.4)
        self.assertAlmostEqual(parts["bottom_edge"], 1.0)

    def test_objective_and_constraint_priority_comparisons(self):
        objective_winner = Individual(np.zeros(1), Evaluation(1.0, 0.8))
        constraint_winner = Individual(np.zeros(1), Evaluation(2.0, 0.1))
        self.assertTrue(_is_better(objective_winner, constraint_winner, "OP-C"))
        self.assertTrue(_is_better(constraint_winner, objective_winner, "CP-C"))
        feasible = Individual(np.zeros(1), Evaluation(3.0, 0.0))
        infeasible = Individual(np.zeros(1), Evaluation(0.1, 0.01))
        self.assertTrue(_is_better(feasible, infeasible, "OP", epsilon=0.0))

    def test_das_de_respects_evaluation_budget_and_is_reproducible(self):
        def evaluate(x):
            objective = float((x[0] - 0.7) ** 2 + (x[1] + 0.2) ** 2)
            violation = max(0.0, 0.2 - float(x[0]))
            return Evaluation(objective, violation)

        kwargs = dict(
            bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            max_evals=120,
            population_size=12,
            correlation_evals=18,
            seed=19,
        )
        first = optimize_das_de(evaluate, **kwargs)
        second = optimize_das_de(evaluate, **kwargs)
        self.assertLessEqual(first.evaluations, 120)
        self.assertEqual(first.evaluations, second.evaluations)
        self.assertEqual(first.kappa, second.kappa)
        np.testing.assert_allclose(
            np.asarray([x.x for x in first.population]),
            np.asarray([x.x for x in second.population]),
        )
        self.assertTrue(all(x.g >= 0 for x in first.population))


class MLSOptimizationTests(unittest.TestCase):
    def _sample(self):
        height, width = 180, 240
        image = np.full((height, width, 3), (45, 65, 85), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (118, 82), (18, 34), -12, 0, 360, 1, -1)
        cv2.circle(mask, (128, 55), 9, 1, -1)
        image[mask > 0] = (215, 220, 225)
        return image, mask

    def test_deterministic_offsets_drive_mls_warp(self):
        image, mask = self._sample()
        contour, _, _ = local_contour_controls(mask)
        offsets = np.zeros_like(contour)
        offsets[:, 0] = 0.25
        result = warp_local_instance(
            image, mask, contour_offsets=offsets, validate_geometry=False
        )
        self.assertEqual(result.control_points, len(contour) + 4 * 12 - 4)
        self.assertEqual(result.mask.shape, mask.shape)
        self.assertGreater(result.area_ratio, 0)
        self.assertAlmostEqual(result.bbox_area_ratio, 1.0, places=2)

    def test_optimizer_returns_only_feasible_candidates(self):
        image, mask = self._sample()
        bank = np.array([
            [0.10, 0.20, 0.30],
            [0.20, 0.25, 0.35],
            [0.15, 0.30, 0.40],
        ])

        def evaluate_candidate(warped_image, warped_mask, bbox, roi):
            feature = warped_image[warped_mask > 0].mean(axis=0)
            return {
                "feature": feature,
                "confidence": 0.95,
                "background_confidence": 0.0,
            }

        result = optimize_mls_instance(
            image,
            mask,
            bank,
            evaluate_candidate,
            original_confidence=0.90,
            population_size=8,
            max_evals=60,
            correlation_evals=12,
            alpha=0.01,
            seed=7,
            top_k=2,
        )
        self.assertLessEqual(result.optimizer.evaluations, 60)
        self.assertLessEqual(len(result.candidates), 2)
        for candidate in result.candidates:
            self.assertGreaterEqual(candidate.confidence, 0.90)
            self.assertEqual(candidate.constraint_components["class_confidence"], 0.0)
            self.assertEqual(candidate.constraint_components["background_false_positive"], 0.0)


if __name__ == "__main__":
    unittest.main()
