# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Result records returned by the analysis drivers, one per precision point. Each
serialises to a plain dict (``to_dict``) for the results database.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from fp_arena.experiment.config import PrecisionMap


@dataclass
class ErrorStats:
    """
    Error metrics of an array versus a reference, reduced over elements and samples.

    Calculations use the error vector `e` (array - reference) and the reference vector `r`.

    Metrics
    -------
    abs_mean  : Mean(|e|)

    rel_mean  : Mean(|e| / |r|)
    rel_max   : Max(|e| / |r|)

    l1        : sum(|e|)
    l2        : sqrt(sum(e**2))
    linf      : max(|e|)

    l1_norm   : sum(|e|) / sum(|r|)
    l2_norm   : sqrt(sum(e**2)) / sqrt(sum(r**2))
    linf_norm : max(|e|) / max(|r|)
    snr       : 10 * log10(sum(r**2) / sum(e**2))
    """

    abs_mean: float
    rel_mean: float
    rel_max: float
    l1: float
    l2: float
    linf: float
    l1_norm: float
    l2_norm: float
    linf_norm: float
    snr: float


def proportional(floor: float, factor: float) -> float:
    """``factor`` times the error, for a metric that is linear in the error."""
    return floor * factor


def power_decibels(floor: float, factor: float) -> float:
    """
    ``factor`` times the error, for a power ratio in dB: scaling the error
    vector by ``factor`` scales its power by ``factor**2``, i.e. costs
    ``20*log10(factor)`` dB.
    """
    return floor - 20.0 * math.log10(factor)


@dataclass(frozen=True)
class Metric:
    """
    How one :class:`ErrorStats` field behaves as a budget constraint.

    :param higher_is_better: whether a *larger* value means the more accurate
        result. Sets the direction of every comparison against a limit.
    :param scale: turns a measured error into the limit meaning "allow
        ``factor`` times that much error", as ``(floor, factor) -> limit``.
    """

    higher_is_better: bool = False
    scale: Callable[[float, float], float] = proportional

    def better(self, a: float, b: float) -> float:
        """The more accurate of two values."""
        return max(a, b) if self.higher_is_better else min(a, b)

    def worse(self, a: float, b: float) -> float:
        """The less accurate of two values."""
        return min(a, b) if self.higher_is_better else max(a, b)

    def satisfies(self, value: float, limit: float) -> bool:
        """Whether ``value`` is inside ``limit``."""
        return value >= limit if self.higher_is_better else value <= limit


#: Every metric a budget can constrain, i.e. every :class:`ErrorStats` field.
#: A new field has to be entered here before it can be used as a constraint.
METRICS: dict[str, Metric] = {
    "abs_mean": Metric(),
    "rel_mean": Metric(),
    "rel_max": Metric(),
    "l1": Metric(),
    "l2": Metric(),
    "linf": Metric(),
    "l1_norm": Metric(),
    "l2_norm": Metric(),
    "linf_norm": Metric(),
    "snr": Metric(higher_is_better=True, scale=power_decibels),
}


@dataclass
class PerfResult:
    """
    Per-repetition phase timings (milliseconds) for one precision point, with
    ``total = h2d + cast_in + kernel + cast_out + d2h``. Only raw lists are kept;
    derive medians/etc. from them.
    """

    precision: PrecisionMap
    total_times: list[float] = field(default_factory=list)
    h2d_times: list[float] = field(
        default_factory=list
    )  # host->device transfer (GPU only)
    d2h_times: list[float] = field(
        default_factory=list
    )  # device->host transfer (GPU only)
    cast_in_times: list[float] = field(default_factory=list)  # input precision cast
    cast_out_times: list[float] = field(default_factory=list)  # output precision cast
    kernel_times: list[float] = field(default_factory=list)  # compute
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorResult:
    """Per-array error for one precision point."""

    precision: PrecisionMap
    errors: dict[str, ErrorStats]
    n_samples: int
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationResult:
    """
    Per-array output deviation induced by perturbing the single input
    ``perturbed``, versus the clean run at the same precision point.
    """

    precision: PrecisionMap
    perturbed: str
    errors: dict[str, ErrorStats]
    n_samples: int
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintResult:
    """
    One budget constraint checked on one output array of one precision point.

    ``ok`` is set if ``value`` stays inside ``limit``, in whichever direction
    the metric's :class:`Metric` calls accurate.
    """

    metric: str
    array: str
    value: float
    limit: float
    ok: bool


@dataclass
class SelectionCandidate:
    """
    One precision point as evaluated by the selection search.
    """

    precision: PrecisionMap
    errors: dict[str, ErrorStats]
    constraints: list[ConstraintResult]
    feasible: bool
    objective_ms: float | None = None
    speedup: float | None = None
    #: ``{perturbed_input: {output_array: stats}}``
    sensitivity: dict[str, dict[str, ErrorStats]] | None = None


@dataclass
class SelectionResult:
    """
    The outcome of a selection search: the fastest precision point that met the
    budget, plus every candidate that was evaluated.

    ``best`` is ``None`` if nothing met the budget.

    ``limits`` stores the resolved numeric thresholds.
    """

    best: PrecisionMap | None
    speedup_baseline: PrecisionMap
    objective: str
    #: ``{metric: {array: limit}}`` -- the thresholds actually applied.
    limits: dict[str, dict[str, float]]
    #: ``{metric: {array: value}}`` -- input-noise-induced deviation at the baseline.
    noise_floor: dict[str, dict[str, float]]
    #: Feasible points first, fastest first; then the rest.
    candidates: list[SelectionCandidate]
    n_evaluated: int
    n_feasible: int
    n_samples: int = 1
    seed: int = 0

    @property
    def precision(self) -> PrecisionMap:
        """The winning assignment, or ``{}`` if none -- the store's precision column."""
        return dict(self.best) if self.best is not None else {}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
