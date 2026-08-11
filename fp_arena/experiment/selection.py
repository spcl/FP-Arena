# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Selection: the fastest per-array precision assignment that stays inside an error budget.

A meta-analysis -- it executes nothing itself, it drives the other three over an
enumerated search space and takes an argmin subject to a constraint:

* :func:`~fp_arena.experiment.runner.run_perturbation` at the baseline point
  measures how far input uncertainty alone moves each output. That *noise floor*
  is a property of the problem's conditioning, which is why it is measured once,
  at the highest-precision point, rather than per candidate -- measured at a
  candidate it would conflate the problem's sensitivity with that candidate's
  rounding, and a budget derived from it would vary with the point it is judging.
* :func:`~fp_arena.experiment.runner.run_error` measures every candidate against
  the reference; the budget turns those into a feasible set.
* :func:`~fp_arena.experiment.runner.run_performance` times only the survivors.

Error runs at every point because error *is* the constraint; performance runs
only at the feasible ones because timing a rejected point buys nothing.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from fp_arena.experiment.config import (
    CONSTANTS_KEY,
    ErrorAnalysisConfig,
    ErrorBudget,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PrecisionMap,
    SelectionAnalysisConfig,
)
from fp_arena.experiment.results import (
    HIGHER_IS_BETTER,
    ConstraintResult,
    ErrorStats,
    PerfResult,
    PerturbationResult,
    SelectionCandidate,
    SelectionResult,
)
from fp_arena.experiment.retarget import resolve_vectorize_config
from fp_arena.experiment.runner import run_error, run_performance, run_perturbation
from fp_arena.experiment.store import ResultStore

#: A precision point's identity, for matching it across the three analyses.
#: Positional matching would misalign -- the timed list is a subset.
PointKey = tuple[tuple[str, str], ...]


def _key(pin_map: PrecisionMap) -> PointKey:
    return tuple(sorted(pin_map.items()))


def _dedupe(points: list[PrecisionMap]) -> list[PrecisionMap]:
    """The points in order, first occurrence of each kept."""
    seen: set[PointKey] = set()
    out: list[PrecisionMap] = []
    for p in points:
        k = _key(p)
        if k not in seen:
            seen.add(k)
            out.append(dict(p))
    return out


def _better(metric: str, a: float, b: float) -> float:
    """The more accurate of two values of ``metric``."""
    return max(a, b) if metric in HIGHER_IS_BETTER else min(a, b)


def _worse(metric: str, a: float, b: float) -> float:
    """The less accurate of two values of ``metric``."""
    return min(a, b) if metric in HIGHER_IS_BETTER else max(a, b)


def _satisfies(metric: str, value: float, limit: float) -> bool:
    return value >= limit if metric in HIGHER_IS_BETTER else value <= limit


def _scale(metric: str, floor: float, factor: float) -> float:
    """
    The limit meaning "allow ``factor`` times the noise-floor error".
    """
    if metric in HIGHER_IS_BETTER:
        return floor - 20.0 * math.log10(factor)
    return floor * factor


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
                    value if current is None else _worse(m, current, value)
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
            derived = _scale(metric, value, factor)
            current = per.get(array)
            per[array] = (
                derived if current is None else _better(metric, current, derived)
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
                    ok=_satisfies(metric, value, limit),
                )
            )
    return out


def _objective_ms(perf: PerfResult, objective: str) -> float:
    """Median milliseconds of the objective phase over the timed repetitions."""
    series = getattr(perf, f"{objective}_times")
    if not series:
        raise ValueError(
            f"Objective phase {objective!r} was not timed at precision point "
            f"{perf.precision or '(baseline)'}; the phase does not exist for this "
            f"program (CPU runs have no h2d/d2h, an uncast point has no cast_in/"
            f"cast_out). Pick another objective."
        )
    return statistics.median(series)


def _label(key: str) -> str:
    """Column label for a precision-map key."""
    return "consts" if key == CONSTANTS_KEY else key


def _fmt_point(point: PrecisionMap) -> str:
    """One precision point on a line, e.g. ``A=fp16 B=fp32 consts=fp32``."""
    return " ".join(f"{_label(k)}={v}" for k, v in sorted(point.items())) or (
        "(unmodified program)"
    )


def _num(value: float | None) -> str:
    """A metric, limit or duration; ``-`` when it was never measured."""
    return "-" if value is None else f"{value:.4g}"


def _render(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: str,
    indent: str = "",
    groups: Sequence[tuple[str, int]] | None = None,
) -> list[str]:
    """
    A column-aligned table, widths taken from the content.
    """
    widths = [
        max([len(h), *(len(row[i]) for row in rows)]) for i, h in enumerate(headers)
    ]
    gap = "  "
    _MIN_RULE = 4

    if groups:
        if sum(count for _, count in groups) != len(headers):
            raise ValueError("groups must span exactly the table's columns")
        at = 0
        for label, count in groups:
            span = sum(widths[at : at + count]) + len(gap) * (count - 1)
            needed = len(label) + _MIN_RULE if label else 0
            if needed > span:
                widths[at + count - 1] += needed - span
            at += count

    def line(cells: Sequence[str]) -> str:
        return indent + gap.join(
            c.ljust(w) if a == "l" else c.rjust(w)
            for c, w, a in zip(cells, widths, align)
        )

    def banner(label: str, span: int) -> str:
        if not label:
            return " " * span
        pad = span - len(label) - 2
        left = pad // 2
        return f"{'-' * left} {label} {'-' * (pad - left)}"

    out = []
    if groups:
        cells, at = [], 0
        for label, count in groups:
            span = sum(widths[at : at + count]) + len(gap) * (count - 1)
            cells.append(banner(label, span))
            at += count
        out.append((indent + gap.join(cells)).rstrip())

    rule = indent + "-" * (sum(widths) + len(gap) * (len(widths) - 1))
    return [*out, line(headers), rule, *(line(r) for r in rows)]


def format_selection(
    result: SelectionResult,
    name: str | None = None,
    max_rows: int | None = 20,
) -> str:
    """
    A summary of a selection search: the budget that was applied,
    the candidate table, and the winner.

    :param result: the search to summarise.
    :param name: experiment name for the heading, if you have one.
    :param max_rows: candidates to tabulate, best first; ``None`` for all.
    """
    out: list[str] = [f"=== selection: {name} ===" if name else "=== selection ==="]
    out.append(
        f"objective={result.objective}  evaluated={result.n_evaluated}  "
        f"within budget={result.n_feasible}  n_samples={result.n_samples}  "
        f"seed={result.seed}"
    )
    out.append(f"speedup baseline: {_fmt_point(result.speedup_baseline)}")

    out.append("")
    out.append("--- budget ---")
    budget_rows = [
        [
            metric,
            array,
            _num(limit),
            _num(result.noise_floor.get(metric, {}).get(array)),
        ]
        for metric in sorted(result.limits)
        for array, limit in sorted(result.limits[metric].items())
    ]
    if budget_rows:
        out += _render(
            ["metric", "array", "limit", "noise floor"], budget_rows, "llrr", "  "
        )
    else:
        out.append("  no numeric limits -- the budget is a predicate alone")

    # --- candidates: the assigned precisions, then every graded array's error.
    metrics = sorted(result.limits)
    keys = sorted(
        {k for c in result.candidates for k in c.precision if k != CONSTANTS_KEY}
    )
    if any(CONSTANTS_KEY in c.precision for c in result.candidates):
        keys.append(CONSTANTS_KEY)
    point_headers = [_label(k) for k in keys] or ["point"]
    error_cols = [(m, array) for m in metrics for array in sorted(result.limits[m])]

    shown = result.candidates if max_rows is None else result.candidates[:max_rows]
    rows = []
    for c in shown:
        verdicts = {(k.metric, k.array): k for k in c.constraints}
        cells = []
        for col in error_cols:
            k = verdicts.get(col)
            # Trailing "!" is the over-budget mark.
            cells.append("-" if k is None else _num(k.value) + ("" if k.ok else "!"))
        rows.append(
            [
                "*" if c.precision == result.best else "",
                "ok" if c.feasible else "over",
                *([c.precision.get(k, "-") for k in keys] or ["(unmodified)"]),
                *cells,
                _num(c.objective_ms),
                f"{c.speedup:.2f}x" if c.speedup is not None else "-",
            ]
        )
    out.append("")
    out.append("--- candidates (feasible first, fastest first) ---")
    groups = [("", 2), ("precision", len(point_headers))]
    groups += [(m, len(result.limits[m])) for m in metrics]
    groups.append(("timing", 2))
    out += _render(
        [
            "",
            "budget",
            *point_headers,
            *(array for _, array in error_cols),
            f"{result.objective} ms",
            "speedup",
        ],
        rows,
        "ll" + "l" * len(point_headers) + "r" * (len(error_cols) + 2),
        groups=groups,
    )
    if len(shown) < len(result.candidates):
        out.append(
            f"... {len(result.candidates) - len(shown)} more in "
            f"SelectionResult.candidates"
        )
    out.append("* selected   ! over budget   '-' left at the program's own precision")

    out.append("")
    if result.best is not None:
        winner = next(c for c in result.candidates if c.precision == result.best)
        speedup = (
            f", {winner.speedup:.2f}x over the baseline"
            if winner.speedup is not None
            else ""
        )
        out.append(f"Fastest point within budget: {_fmt_point(result.best)}")
        out.append(f"  {_num(winner.objective_ms)} ms {result.objective}{speedup}")
        return "\n".join(out)

    points = "point" if result.n_evaluated == 1 else "points"
    out.append(
        f"Nothing met the budget ({result.n_evaluated} {points} evaluated) -- "
        f"loosen it or widen the search space."
    )
    misses: dict[tuple[str, str], list[float]] = {}
    for c in result.candidates:
        for k in c.constraints:
            if not k.ok:
                misses.setdefault((k.metric, k.array), []).append(k.value)
    if not misses:
        out.append("  every numeric constraint passed; the budget predicate rejected.")
        return "\n".join(out)
    miss_rows = []
    for (metric, array), values in sorted(misses.items()):
        closest = values[0]
        for v in values[1:]:
            closest = _better(metric, closest, v)
        miss_rows.append(
            [
                metric,
                array,
                f"{len(values)}/{result.n_evaluated}",
                _num(result.limits[metric][array]),
                _num(closest),
            ]
        )
    out.append("  rejected by:")
    out += _render(
        ["metric", "array", "points", "limit", "closest"], miss_rows, "llrrr", "  "
    )
    return "\n".join(out)


def run_selection(
    cfg: SelectionAnalysisConfig, store: ResultStore | None = None
) -> SelectionResult:
    """
    Search ``cfg.precisions`` for the fastest point meeting ``cfg.budget``.

    The sub-analyses append their own rows to ``store`` under the experiment's
    name, so the raw error, timing and sensitivity data behind the decision stay
    queryable alongside the selection result itself.

    Prints :func:`format_selection` of the outcome.
    """
    exp = cfg.experiment
    points = _dedupe([cfg.speedup_baseline, *cfg.precisions])
    base_key = _key(cfg.speedup_baseline)
    vectorization = resolve_vectorize_config(
        exp.target, exp.gpu_vectorize, exp.gpu_vectorize_config
    )

    # 1. Sensitivity to input noise.
    perturbations = run_perturbation(
        PerturbationAnalysisConfig(
            exp,
            noise=cfg.noise,
            precisions=points if cfg.perturb_all else [dict(cfg.speedup_baseline)],
            n_samples=cfg.n_samples,
        ),
        store=store,
    )
    sensitivity_of: dict[PointKey, dict[str, dict[str, ErrorStats]]] = {}
    for r in perturbations:
        sensitivity_of.setdefault(_key(r.precision), {})[r.perturbed] = r.errors

    metrics = sorted({*cfg.budget.limits, *(cfg.budget.from_perturbation or {})})
    floor = noise_floor(
        [r for r in perturbations if _key(r.precision) == base_key], metrics
    )

    # 2. Error at every point
    errors = run_error(
        ErrorAnalysisConfig(
            exp,
            precisions=points,
            reference=cfg.reference,
            n_samples=cfg.n_samples,
        ),
        store=store,
    )
    error_of = {_key(r.precision): r for r in errors}

    # 3. Feasibility.
    arrays = list(cfg.budget.arrays or sorted(errors[0].errors))
    limits = resolve_limits(cfg.budget, floor, arrays)
    constraints_of: dict[PointKey, list[ConstraintResult]] = {}
    feasible_of: dict[PointKey, bool] = {}
    for key, result in error_of.items():
        constraints = check(result.errors, limits)
        ok = all(c.ok for c in constraints)
        if ok and cfg.budget.predicate is not None:
            ok = bool(cfg.budget.predicate(result.errors))
        constraints_of[key] = constraints
        feasible_of[key] = ok

    # 4. Time the survivors (plus the baseline, which is the speedup denominator).
    to_time = [
        p for p in points if cfg.time_all or feasible_of[_key(p)] or _key(p) == base_key
    ]
    perfs = run_performance(
        PerformanceAnalysisConfig(
            exp,
            precisions=to_time,
            n_warmup=cfg.n_warmup,
            n_reps=cfg.n_reps,
        ),
        store=store,
    )
    perf_of = {_key(r.precision): r for r in perfs}

    # 5. Rank.
    base_ms = _objective_ms(perf_of[base_key], cfg.objective)
    candidates: list[SelectionCandidate] = []
    for point in points:
        key = _key(point)
        perf = perf_of.get(key)
        ms = _objective_ms(perf, cfg.objective) if perf is not None else None
        candidates.append(
            SelectionCandidate(
                precision=dict(point),
                errors=error_of[key].errors,
                constraints=constraints_of[key],
                feasible=feasible_of[key],
                objective_ms=ms,
                speedup=(base_ms / ms) if ms else None,
                sensitivity=sensitivity_of.get(key),
            )
        )
    candidates.sort(
        key=lambda c: (
            not c.feasible,
            c.objective_ms if c.objective_ms is not None else math.inf,
        )
    )
    best = next(
        (c.precision for c in candidates if c.feasible and c.objective_ms is not None),
        None,
    )

    result = SelectionResult(
        best=best,
        speedup_baseline=dict(cfg.speedup_baseline),
        objective=cfg.objective,
        limits=limits,
        noise_floor=floor,
        candidates=candidates,
        n_evaluated=len(points),
        n_feasible=sum(1 for c in candidates if c.feasible),
        n_samples=cfg.n_samples,
        seed=exp.seed,
    )
    if store is not None:
        store.add(
            cfg.name or exp.name,
            result,
            symbols=exp.symbols,
            scalars=exp.scalar_args,
            vectorization=vectorization,
        )
    print(format_selection(result, name=cfg.name or exp.name))
    return result
