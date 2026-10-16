# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Error budget: turn an :class:`~fp_arena.search.config.ErrorBudget` into numeric
per-array limits and judge a precision point's error against them.
"""

from __future__ import annotations

from fp_arena.search.config import ErrorBudget
from fp_arena.search.results import (
    METRICS,
    ConstraintResult,
    ErrorStats,
    PerturbationResult,
)


def noise_floor(
    results: list[PerturbationResult], metrics: list[str]
) -> dict[str, dict[str, float]]:
    """
    Worst-case output deviation induced by input noise, as ``{metric: {array: value}}``.
    """
    floor: dict[str, dict[str, float]] = {m: {} for m in metrics}
    for r in results:
        for array, stats in r.errors.items():
            for m in metrics:
                value = float(getattr(stats, m))
                current = floor[m].get(array)
                floor[m][array] = (
                    value if current is None else METRICS[m].worse(current, value)
                )
    return floor


def resolve_limits(
    budget: ErrorBudget,
    floor: dict[str, dict[str, float]],
    arrays: list[str],
) -> dict[str, dict[str, float]]:
    """
    The numeric thresholds a point must meet, as ``{metric: {array: limit}}``.
    """
    out: dict[str, dict[str, float]] = {}
    for metric, spec in budget.limits.items():
        per = out.setdefault(metric, {})
        if isinstance(spec, dict):
            for array, value in spec.items():
                if array not in arrays:
                    raise ValueError(
                        f"ErrorBudget.limits[{metric!r}] names array {array!r}, "
                        f"which is not a checked output array; known: {arrays}"
                    )
                per[array] = float(value)
        else:
            for array in arrays:
                per[array] = float(spec)
    for metric, factor in (budget.from_perturbation or {}).items():
        per = out.setdefault(metric, {})
        for array, value in floor.get(metric, {}).items():
            if array not in arrays:
                continue
            derived = METRICS[metric].scale(value, factor)
            current = per.get(array)
            per[array] = (
                derived if current is None else METRICS[metric].better(current, derived)
            )
    return out


def check(
    errors: dict[str, ErrorStats], limits: dict[str, dict[str, float]]
) -> list[ConstraintResult]:
    """Evaluate every budget constraint against one point's per-array error."""
    out: list[ConstraintResult] = []
    for metric in sorted(limits):
        for array in sorted(limits[metric]):
            if array not in errors:
                continue
            value = float(getattr(errors[array], metric))
            limit = limits[metric][array]
            out.append(
                ConstraintResult(
                    metric=metric,
                    array=array,
                    value=value,
                    limit=limit,
                    ok=METRICS[metric].satisfies(value, limit),
                )
            )
    return out
