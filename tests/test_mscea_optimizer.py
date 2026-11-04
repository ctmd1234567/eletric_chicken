"""Tests for the constrained MSCEA search core."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from sentinel.augmentation.das_de import (
    Evaluation, Individual, _is_better, optimize_das_de,
    _correlation, _compare_correlations, _frofi_generation, _frofi_trial,
)
from sentinel.augmentation.mls import local_contour_controls, warp_local_instance
from sentinel.augmentation.mscea import (
    MicroscaleRegion,
    detection_constraint_violation,
    feature_diversity,
    optimize_mls_instance,
)


class MicroscaleRegionTests(unittest.TestCase):
    def item(self, f, g):
        return Individual(np.zeros(2), Evaluation(f, g))

    def test_correlation_and_all_kappa_sign_pairs(self):
        self.assertEqual(_correlation([1, 2], [1, 1]), 1)
        self.assertEqual(_correlation([1, 2], [2, 3]), -1)
        self.assertEqual(_correlation([1, 2], [0, 3]), 0)
        initial = [self.item(10, 10)]
        for c1 in (-1, 0, 1):
            for c2 in (-1, 0, 1):
                with self.subTest(c1=c1, c2=c2):
                    expected = -1 if (c1, c2) == (-1, 1) else 1
                    self.assertEqual(_compare_correlations(c1, c2, initial,
                                                          [self.item(1, 2)], [self.item(3, 4)]), expected)
        cases = [(1, 2, 3, 4, 1), (3, 4, 1, 2, -1),
                 (1, 8, 6, 7, 1), (1, 8, 2, 3, -1), (1, 4, 3, 2, 0)]
        for f1, g1, f2, g2, expected in cases:
            self.assertEqual(_compare_correlations(0, 0, initial,
                                                  [self.item(f1, g1)], [self.item(f2, g2)]), expected)
        self.assertEqual(_compare_correlations(0, 1, initial,
                                              [self.item(1, 10)], [self.item(2, 3)]), 0)
        self.assertEqual(_compare_correlations(1, 0, initial,
                                              [self.item(1, 2)], [self.item(10, 3)]), 0)

    def test_selection_ties_and_epsilon_relaxation(self):
        a, b = self.item(1, .8), self.item(2, .1)
        self.assertTrue(_is_better(a, b, "OP", epsilon=1))
        self.assertFalse(_is_better(a, b, "OP", epsilon=.5))
        self.assertTrue(_is_better(self.item(1, .1), b, "CP-C"))
        self.assertTrue(_is_better(self.item(2, .01), b, "OP-C"))
        for rule in ("OP-C", "CP-C", "OP"):
            self.assertFalse(_is_better(a, a, rule))
        with self.assertRaises(ValueError):
            _is_better(a, b, "unknown")
        self.assertFalse(self.item(0, 1e-14).feasible)
        self.assertTrue(_is_better(self.item(1, 0), self.item(0, 1e-14), "CP-C"))

    def test_all_four_allocation_criteria(self):
        for kappa, found, expected in ((1, True, "OP-C"), (1, False, "OP"),
                                       (-1, True, "CP-C"), (-1, False, "FROFI")):
            calls = []
            def evaluate(x):
                calls.append(x)
                return Evaluation(-len(calls), 0 if found and len(calls) > 4 else 1)
            with patch("sentinel.augmentation.das_de._compare_correlations", return_value=kappa):
                result = optimize_das_de(evaluate, ([-1, -1], [1, 1]),
                                         population_size=8, max_evals=13)
            self.assertEqual(result.comparison, expected)
            self.assertEqual(result.evaluations, 13)
        allocations = set()
        for seed in range(10):
            with patch("sentinel.augmentation.das_de._compare_correlations", return_value=0):
                result = optimize_das_de(lambda x: (0, 1), ([-1], [1]),
                                         population_size=8, max_evals=12, seed=seed)
            allocations.add(result.allocation)
        self.assertEqual(allocations, {"objective", "constraint"})

    def test_region_union_not_hull_and_saturating_boundary(self):
        region = MicroscaleRegion(np.zeros(2), np.ones(2), max_boxes=3)
        region.expand([1, 1])
        region.expand([2, 2])
        region.expand([3, 3])
        self.assertEqual(len(region.centers), 3)
        self.assertFalse(region.contains([-1, 3]))
        self.assertTrue(region.contains(region.repair([-1, 3])))
        for _ in range(50):
            self.assertTrue(region.contains(region.sample(np.random.default_rng(_))))

    def test_expansion_respects_absolute_roi_domain(self):
        region = MicroscaleRegion(np.zeros(2), np.ones(2), hard_lower=[-.5, -.5], hard_upper=[1.5, 1.5])
        region.expand([1., 1.])
        np.testing.assert_allclose(region.bounds, ([-.5, -.5], [1.5, 1.5]))
        np.testing.assert_allclose(region.repair([100., 100.]), [1.5, 1.5])
        self.assertFalse(region.contains([1.6, 1.]))
        for seed in range(10):
            self.assertTrue(region.contains(region.sample(np.random.default_rng(seed))))
        with self.assertRaisesRegex(ValueError, "absolute"):
            region.expand([2., 2.])

    def test_budgets_count_every_callback_and_infeasible_never_expand(self):
        for budget in (12, 13, 15, 16, 29, 30, 31):
            calls, feasible_calls = [], []
            region = MicroscaleRegion(np.zeros(2), np.ones(2))
            def evaluate(x):
                calls.append(x)
                return Evaluation(float(x.sum()), 1)
            result = optimize_das_de(evaluate, ([-1, -1], [1, 1]),
                                     population_size=8, max_evals=budget, region=region,
                                     feasible_callback=lambda *args: feasible_calls.append(args))
            self.assertEqual(len(calls), budget)
            self.assertEqual(result.evaluations, budget)
            self.assertEqual(len(region.centers), 1)
            self.assertEqual(feasible_calls, [])
            self.assertEqual(result.feasible, [])
        with self.assertRaisesRegex(ValueError, "paired"):
            optimize_das_de(lambda x: (0, 0), ([-1], [1]), population_size=10, max_evals=12)

    def test_frofi_archive_replacement_runs_at_exact_budget(self):
        population = [self.item(f, 0) for f in (10, 20, 30, 40)]
        calls = []
        def evaluate(x):
            calls.append(x)
            return Evaluation(1, 1)
        with patch("sentinel.augmentation.das_de._frofi_trial", return_value=np.zeros(2)):
            updated, used = _frofi_generation(population, evaluate, np.random.default_rng(1),
                                              np.full(2, -1), np.ones(2), None, None, 4, 0)
        self.assertEqual(used, 4)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(p.f == 1 for p in updated))
        # This is the *search* archive, permitted to reintroduce infeasible items.
        self.assertTrue(all(not p.feasible for p in updated))

    def test_frofi_single_coordinate_escape_obeys_remaining_budget(self):
        population = [Individual(np.array([i * .1, i * .1]), Evaluation(10, 1)) for i in range(4)]
        calls = []
        def evaluate(x):
            calls.append(x.copy())
            return Evaluation(1, 1) if len(calls) == 5 else Evaluation(20, 1)
        with patch("sentinel.augmentation.das_de._frofi_trial", return_value=np.zeros(2)):
            updated, used = _frofi_generation(population, evaluate, np.random.default_rng(4),
                                              np.full(2, -1), np.ones(2), None, None, 5, 0)
        self.assertEqual(used, 5)
        self.assertEqual(min(p.f for p in updated), 1)
        self.assertTrue(any(np.count_nonzero(calls[-1] != p.x) == 1 for p in population))

    def test_frofi_trial_formulas_and_forced_crossover(self):
        class Rng:
            def __init__(self, branch):
                self.values = iter([branch, .25])
            def choice(self, values, size=None, replace=None):
                return np.array([1, 2, 3]) if size else values[0]
            def random(self, size=None):
                return np.full(size, .99) if size else next(self.values)
            def integers(self, size):
                return 1
        population = [Individual(np.array([i, i+1.]), Evaluation(i, 1)) for i in range(4)]
        lower, upper = np.full(2, -100.), np.full(2, 100.)
        current = _frofi_trial(population, 0, Rng(.1), lower, upper, None)
        np.testing.assert_allclose(current, population[0].x + .25*(population[1].x-population[0].x)
                                   + .6*(population[2].x-population[3].x))
        best = _frofi_trial(population, 0, Rng(.9), lower, upper, None)
        mutant = population[1].x + .25*(population[0].x-population[1].x) + .6*(population[2].x-population[3].x)
        np.testing.assert_allclose(best, [population[0].x[0], mutant[1]])

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
            fire_confidence=0.0,
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
    def test_fire_alone_prevents_final_archive_and_is_logged(self):
        image, mask = self._sample()
        result = optimize_mls_instance(
            image, mask, np.zeros((2, 3)),
            lambda *args: dict(feature=np.ones(3), confidence=.95,
                               background_confidence=0., fire_confidence=.9),
            original_confidence=.8, population_size=8, max_evals=12, contour_points=8, alpha=.001)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.expanded_region_boxes, 1)
        self.assertEqual(len(result.evaluation_history), 12)
        for row in result.evaluation_history:
            self.assertEqual(row["fire_confidence"], .9)
            self.assertAlmostEqual(row["constraint_components"]["fire_false_positive"], .8)
            self.assertEqual(row["constraint_components"]["class_confidence"], 0)
            self.assertEqual(row["constraint_components"]["background_false_positive"], 0)
            self.assertFalse(row["feasible"])

    def test_missing_or_nonfinite_fire_metric_fails_closed(self):
        image, mask = self._sample()
        for fire in (None, float("nan"), -1., 2.):
            metrics = dict(feature=np.ones(3), confidence=.95, background_confidence=0.)
            if fire is not None:
                metrics["fire_confidence"] = fire
            result = optimize_mls_instance(
                image, mask, np.zeros((2, 3)), lambda *args: metrics,
                original_confidence=.8, population_size=8, max_evals=12, contour_points=8, alpha=.001)
            self.assertEqual(result.candidates, [])
            self.assertEqual(result.optimizer.evaluations, 12)
            self.assertTrue(all("invalid_candidate" in row for row in result.evaluation_history))

    def test_geometry_rejects_bottom_shift_and_pixels_outside_roi_stay_fixed(self):
        image, mask = self._sample()
        contour, _, roi = local_contour_controls(mask)
        offsets = np.zeros_like(contour)
        offsets[:, 1] = 10
        result = warp_local_instance(image, mask, contour_offsets=offsets,
                                     bottom_anchor_strength=0, validate_geometry=False)
        self.assertGreater(result.bottom_shift_ratio, 0)
        with self.assertRaisesRegex(ValueError, "Bottom-edge"):
            warp_local_instance(image, mask, contour_offsets=offsets,
                                bottom_anchor_strength=0, max_bottom_shift_ratio=0,
                                area_ratio_range=(.1, 5))
        outside = np.ones(mask.shape, dtype=bool)
        x1, y1, x2, y2 = roi
        outside[y1:y2, x1:x2] = False
        np.testing.assert_array_equal(result.image[outside], image[outside])
        np.testing.assert_array_equal(result.mask[outside], mask[outside])

    def test_zero_offsets_preserve_pixels_mask_and_bottom(self):
        image, mask = self._sample()
        contour, _, _ = local_contour_controls(mask)
        warp = warp_local_instance(image, mask, contour_offsets=np.zeros_like(contour))
        np.testing.assert_array_equal(warp.image, image)
        np.testing.assert_array_equal(warp.mask, mask)
        self.assertEqual(warp.bottom_shift_ratio, 0)
        self.assertEqual(warp.area_ratio, 1)

    def test_infeasible_mock_never_produces_final_pixels(self):
        image, mask = self._sample()
        result = optimize_mls_instance(
            image, mask, np.zeros((2, 3)),
            lambda *args: dict(feature=np.ones(3), confidence=.1, background_confidence=.9, fire_confidence=0.),
            original_confidence=.8, population_size=8, max_evals=12, contour_points=8)
        self.assertEqual(result.candidates, [])
        self.assertEqual(len(result.evaluation_history), 12)
        self.assertTrue(all(not h["feasible"] for h in result.evaluation_history))
        self.assertTrue(all("warp" not in p.evaluation.payload for p in result.optimizer.population))
        self.assertEqual(result.expanded_region_boxes, 1)

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
        self.assertGreater(result.bbox_area_ratio, 0.8)
        self.assertLess(result.bbox_area_ratio, 1.2)

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
                "fire_confidence": 0.0,
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
        self.assertTrue(result.candidates, "mock integration must actually find a feasible output")
        self.assertEqual(len(result.evaluation_history), result.optimizer.evaluations)
        self.assertTrue(all("warp" not in p.evaluation.payload for p in result.optimizer.population))
        for candidate in result.candidates:
            self.assertGreaterEqual(candidate.confidence, 0.90)
            self.assertEqual(candidate.constraint_components["class_confidence"], 0.0)
            self.assertEqual(candidate.constraint_components["background_false_positive"], 0.0)


if __name__ == "__main__":
    unittest.main()
