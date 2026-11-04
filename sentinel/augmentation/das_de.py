"""Correlation-guided DAS-DE for constrained optimization.

The DAS stage follows Huang et al., IEEE TEVC 2024. The constraint-priority
branch uses the FROFI mechanism of Wang et al., IEEE TCyb 2016.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Evaluation:
    """Scalar minimization objective and nonnegative aggregate violation."""
    objective: float
    violation: float = 0.0
    payload: Any = None


@dataclass
class Individual:
    x: np.ndarray
    evaluation: Evaluation

    @property
    def f(self) -> float:
        return float(self.evaluation.objective)

    @property
    def g(self) -> float:
        return max(0.0, float(self.evaluation.violation))

    @property
    def feasible(self) -> bool:
        return self.g == 0.0

    def clone(self) -> "Individual":
        return Individual(self.x.copy(), self.evaluation)


@dataclass
class DASDEResult:
    population: list[Individual]
    feasible: list[Individual]
    evaluations: int
    kappa: int
    allocation: str
    comparison: str
    correlation_objective: int
    correlation_constraint: int
    history: list[dict[str, Any]] = field(default_factory=list)


def _coerce_evaluation(value: Any) -> Evaluation:
    if isinstance(value, Evaluation):
        result = value
    elif isinstance(value, tuple) and len(value) >= 2:
        result = Evaluation(float(value[0]), float(value[1]),
                            value[2] if len(value) > 2 else None)
    else:
        raise TypeError("evaluate(x) must return Evaluation or (objective, violation[, payload])")
    if not np.isfinite(result.objective) or not np.isfinite(result.violation):
        raise ValueError("objective and constraint violation must be finite")
    if result.violation < -1e-12:
        raise ValueError("constraint violation must be nonnegative")
    result.violation = max(0.0, float(result.violation))
    result.objective = float(result.objective)
    return result


def _is_better(a: Individual, b: Individual, rule: str, epsilon: float = 0.0) -> bool:
    """Compare a and b using OP-C, CP-C, or epsilon-OP."""
    if rule == "OP-C":
        if a.f < b.f:
            return True
        if a.f == b.f:
            return a.g < b.g
        return False
    if rule == "CP-C":
        if a.g < b.g:
            return True
        if a.g == b.g:
            return a.f < b.f
        return False
    if rule == "OP":
        a_inside, b_inside = a.g <= epsilon, b.g <= epsilon
        if a_inside and b_inside:
            return a.f < b.f
        if a.g == b.g:
            return a.f < b.f
        return a.g < b.g
    raise ValueError(f"Unknown comparison rule: {rule}")


def _bounds_arrays(bounds):
    lower = np.asarray(bounds[0], dtype=np.float64).reshape(-1)
    upper = np.asarray(bounds[1], dtype=np.float64).reshape(-1)
    if lower.shape != upper.shape or lower.size == 0:
        raise ValueError("bounds must contain equal non-empty lower/upper vectors")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("bounds must be finite")
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be smaller than its upper bound")
    return lower, upper


def _repair(x, lower, upper, region=None):
    x = np.asarray(x, dtype=np.float64)
    if region is not None and hasattr(region, "bounds"):
        dynamic_lower, dynamic_upper = region.bounds
        x = np.clip(x, dynamic_lower, dynamic_upper)
        x = np.asarray(region.repair(x), dtype=np.float64)
        dynamic_lower, dynamic_upper = region.bounds
        return np.clip(x, dynamic_lower, dynamic_upper)
    x = np.clip(x, lower, upper)
    if region is not None:
        x = np.asarray(region.repair(x), dtype=np.float64)
    return np.clip(x, lower, upper)


def _sample(rng, lower, upper, region=None):
    if region is not None:
        return _repair(region.sample(rng), lower, upper, region)
    return rng.uniform(lower, upper)


def _evaluate(x, evaluate, region, feasible_callback):
    vector = np.asarray(x, dtype=np.float64).copy()
    item = Individual(vector, _coerce_evaluation(evaluate(vector.copy())))
    if item.feasible:
        if region is not None:
            region.expand(item.x)
        if feasible_callback is not None:
            feasible_callback(item.x.copy(), item.evaluation)
    return item


def _draw_indices(rng, size: int, excluded: int, count: int = 3):
    choices = np.delete(np.arange(size), excluded)
    if len(choices) < count:
        raise ValueError("DE requires at least four individuals")
    return rng.choice(choices, size=count, replace=False)


def _standard_trial(population, index, rng, lower, upper, region, f_scale, cr):
    n, dim = len(population), len(population[0].x)
    r1, r2, r3 = _draw_indices(rng, n, index)
    x = population[index].x
    mutant = population[r1].x + f_scale * (population[r2].x - population[r3].x)
    mask = rng.random(dim) < cr
    mask[int(rng.integers(dim))] = True
    return _repair(np.where(mask, mutant, x), lower, upper, region)


def _standard_generation(population, evaluate, rule, rng, lower, upper, region,
                         feasible_callback, epsilon=0.0, f_scale=0.5, cr=0.9,
                         max_evals=None, current_evals=0):
    updated = [p.clone() for p in population]
    used = 0
    for i in range(len(population)):
        if max_evals is not None and current_evals + used >= max_evals:
            break
        x = _standard_trial(population, i, rng, lower, upper, region, f_scale, cr)
        child = _evaluate(x, evaluate, region, feasible_callback)
        used += 1
        if _is_better(child, population[i], rule, epsilon):
            updated[i] = child
    return updated, used


def _correlation(initial, final) -> int:
    """Algorithm 1: +1 harmonious, -1 conflicting, 0 independent."""
    count = sum(1 if before >= after else -1
                for before, after in zip(initial, final))
    if count == len(initial):
        return 1
    if count == -len(initial):
        return -1
    return 0


def _compare_correlations(c1, c2, initial, p1, p2):
    """Algorithm 3 of the correlation-based DAS paper."""
    q1, q2 = list(p1), list(p2)
    if c1 == 0 and c2 == 1:
        q1 = [x for i, x in enumerate(p1) if x.g < initial[i].g]
    elif c1 == 1 and c2 == 0:
        q2 = [x for i, x in enumerate(p2) if x.f < initial[i].f]
    if not q1 or not q2:
        return 0

    m1 = float(np.mean([x.f for x in q1]))
    m1g = float(np.mean([x.g for x in q1]))
    m2 = float(np.mean([x.f for x in q2]))
    m2g = float(np.mean([x.g for x in q2]))

    if c1 == 1 and c2 == -1:
        return 1
    if c1 == -1 and c2 == 1:
        return -1
    if m1g <= m2g and m2 > m1:
        return 1
    if m2 <= m1 and m1g > m2g:
        return -1
    df, dg = abs(m2 - m1), abs(m1g - m2g)
    return 1 if df > dg else (-1 if df < dg else 0)


def _frofi_trial(population, index, rng, lower, upper, region,
                 f_pool=(0.6, 0.8, 1.0), cr_pool=(0.1, 0.2, 1.0)):
    n = len(population)
    r1, r2, r3 = _draw_indices(rng, n, index)
    x = population[index].x
    f_scale = float(rng.choice(f_pool))
    if rng.random() < 0.5:
        # DE/current-to-rand/1; no binomial crossover.
        rand_scale = float(rng.random())
        mutant = x + rand_scale * (population[r1].x - x)
        mutant += f_scale * (population[r2].x - population[r3].x)
        trial = mutant
    else:
        # DE/rand-to-best/1 with binomial crossover.
        best = min(population, key=lambda item: item.f)
        rand_scale = float(rng.random())
        mutant = population[r1].x + rand_scale * (best.x - population[r1].x)
        mutant += f_scale * (population[r2].x - population[r3].x)
        cr = float(rng.choice(cr_pool))
        mask = rng.random(x.size) < cr
        mask[int(rng.integers(x.size))] = True
        trial = np.where(mask, mutant, x)
    return _repair(trial, lower, upper, region)


def _frofi_generation(population, evaluate, rng, lower, upper, region,
                      feasible_callback, max_evals, current_evals,
                      f_pool=(0.6, 0.8, 1.0), cr_pool=(0.1, 0.2, 1.0),
                      max_replacements=None):
    """FROFI: feasibility-rule selection, archive replacement, and mutation."""
    updated = [p.clone() for p in population]
    archive = []
    used = 0
    for i in range(len(population)):
        if current_evals + used >= max_evals:
            break
        x = _frofi_trial(population, i, rng, lower, upper, region, f_pool, cr_pool)
        child = _evaluate(x, evaluate, region, feasible_callback)
        used += 1
        if _is_better(child, population[i], "CP-C"):
            updated[i] = child
        elif child.f < population[i].f - 1e-12:
            archive.append(child)

    # Replacement uses already evaluated individuals, including at the budget edge.
    if archive:
        mrn = max_replacements or max(5, len(updated[0].x) // 2)
        mrn = min(max(1, mrn), len(updated))
        order = sorted(range(len(updated)), key=lambda i: updated[i].f, reverse=True)
        groups = np.array_split(np.asarray(order, dtype=int), mrn)
        for group in groups:
            if not archive:
                break
            target_idx = max(group.tolist(), key=lambda i: updated[i].g)
            archive_idx = min(range(len(archive)), key=lambda i: archive[i].g)
            candidate = archive[archive_idx]
            if candidate.f < updated[target_idx].f - 1e-12:
                updated[target_idx] = candidate
                archive.pop(archive_idx)

    # FROFI's one-coordinate mutation is used only while all individuals
    # remain infeasible. It consumes one additional evaluation.
    if updated and not any(item.feasible for item in updated) and used < max_evals - current_evals:
        source_idx = int(rng.integers(len(updated)))
        worst_idx = max(range(len(updated)), key=lambda i: updated[i].g)
        mutant = updated[source_idx].x.copy()
        dim = int(rng.integers(mutant.size))
        mutation_lower, mutation_upper = (
            region.bounds if region is not None and hasattr(region, "bounds")
            else (lower, upper)
        )
        mutant[dim] = rng.uniform(mutation_lower[dim], mutation_upper[dim])
        mutant = _repair(mutant, lower, upper, region)
        child = _evaluate(mutant, evaluate, region, feasible_callback)
        used += 1
        if child.f < updated[worst_idx].f - 1e-12:
            updated[worst_idx] = child
    return updated, used


def optimize_das_de(
    evaluate: Callable[[np.ndarray], Evaluation | tuple],
    bounds,
    *,
    max_evals: int = 300,
    population_size: int = 30,
    correlation_evals: int | None = None,
    f_scale: float = 0.5,
    crossover_rate: float = 0.9,
    seed: int = 20261103,
    region=None,
    feasible_callback: Callable[[np.ndarray, Evaluation], None] | None = None,
) -> DASDEResult:
    """Run correlation-guided DAS-DE for a constrained minimization problem.

    The optional region object may implement sample(rng), repair(x), and
    expand(x). This allows the feasible-solution neighborhood to expand online.
    """
    if max_evals < 12 or population_size < 8:
        raise ValueError("max_evals >= 12 and population_size >= 8 are required")
    if population_size % 2:
        raise ValueError("population_size must be even for the two-population DAS stage")
    if max_evals < population_size:
        raise ValueError("max_evals must be at least population_size")
    if not 0 < f_scale <= 2 or not 0 <= crossover_rate <= 1:
        raise ValueError("invalid DE parameters")

    lower, upper = _bounds_arrays(bounds)
    rng = np.random.default_rng(seed)
    half = population_size // 2
    # One complete correlation generation needs N/2 initial evaluations
    # and N offspring evaluations. Enforce this minimum.
    minimum_cor_fes = 3 * half
    if max_evals < minimum_cor_fes:
        raise ValueError("max_evals must allow one complete paired correlation generation")
    cor_budget = max(minimum_cor_fes,
                     int(np.ceil(0.05 * max_evals)) if correlation_evals is None
                     else int(correlation_evals))
    cor_budget = min(cor_budget, max_evals)

    # Sample the complete initial population inside the original microscale
    # neighborhood before any feasible point is allowed to expand the region.
    initial_vectors = [_sample(rng, lower, upper, region) for _ in range(half)]
    initial = [
        _evaluate(vector, evaluate, None, feasible_callback)
        for vector in initial_vectors
    ]
    if region is not None:
        for item in initial:
            if item.feasible:
                region.expand(item.x)
    p1 = [item.clone() for item in initial]
    p2 = [item.clone() for item in initial]
    initial_f = [item.f for item in initial]
    initial_g = [item.g for item in initial]
    fes = half
    history = []

    while fes + 2 * half <= cor_budget:
        p1, used1 = _standard_generation(
            p1, evaluate, "OP-C", rng, lower, upper, region, feasible_callback,
            f_scale=f_scale, cr=crossover_rate, max_evals=cor_budget,
            current_evals=fes)
        fes += used1
        p2, used2 = _standard_generation(
            p2, evaluate, "CP-C", rng, lower, upper, region, feasible_callback,
            f_scale=f_scale, cr=crossover_rate, max_evals=cor_budget,
            current_evals=fes)
        fes += used2
        history.append({"stage": "correlation", "evaluations": fes,
                        "objective_feasible": sum(x.feasible for x in p1),
                        "constraint_feasible": sum(x.feasible for x in p2)})
        if used1 != half or used2 != half:
            break

    c1 = _correlation(initial_g, [x.g for x in p1])
    c2 = _correlation(initial_f, [x.f for x in p2])
    kappa = _compare_correlations(c1, c2, initial, p1, p2)
    if kappa == 1:
        allocation = "objective"
    elif kappa == -1:
        allocation = "constraint"
    else:
        allocation = "objective" if rng.random() <= 0.5 else "constraint"

    population = [x.clone() for x in p1 + p2]
    objective_feasible = any(x.feasible for x in p1)
    constraint_feasible = any(x.feasible for x in p2)
    if allocation == "objective":
        comparison = "OP-C" if objective_feasible else "OP"
    else:
        comparison = "CP-C" if constraint_feasible else "FROFI"

    epsilon0 = max((x.g for x in population), default=0.0)
    while fes < max_evals:
        progress = min(1.0, fes / max_evals)
        epsilon = epsilon0 * (1.0 - progress) ** 2
        if comparison == "FROFI":
            population, used = _frofi_generation(
                population, evaluate, rng, lower, upper, region,
                feasible_callback, max_evals, fes)
        else:
            population, used = _standard_generation(
                population, evaluate, comparison, rng, lower, upper,
                region, feasible_callback, epsilon=epsilon,
                f_scale=f_scale, cr=crossover_rate,
                max_evals=max_evals, current_evals=fes)
        if used == 0:
            break
        fes += used
        history.append({"stage": "optimization", "evaluations": fes,
                        "comparison": comparison, "epsilon": epsilon,
                        "feasible": sum(x.feasible for x in population)})

    feasible = sorted((x.clone() for x in population if x.feasible),
                      key=lambda item: item.f)
    unique, seen = [], set()
    for item in feasible:
        key = np.round(item.x, 8).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return DASDEResult(
        population=population, feasible=unique, evaluations=fes,
        kappa=kappa, allocation=allocation, comparison=comparison,
        correlation_objective=c1, correlation_constraint=c2,
        history=history)
