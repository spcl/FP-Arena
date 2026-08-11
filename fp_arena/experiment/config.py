# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Experiment config.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig

from fp_arena.experiment.inputs import DistributionLike, Noise

if TYPE_CHECKING:
    from fp_arena.experiment.results import ErrorStats

#: ``{array_name: precision_key}`` -- pins a subset of arrays to a target format.
PrecisionMap = dict[str, str]

#: Reserved :data:`PrecisionMap` key that targets float literals; absent means literals stay fp64.
CONSTANTS_KEY = "__constants__"


def precision_grid(arrays: dict[str, Sequence[str]]) -> list[PrecisionMap]:
    """
    Every combination of per-array precision choices, as a list of precision points.

    ::

        precision_grid({"A": ["fp16", "fp32"], CONSTANTS_KEY: ["fp32", "fp64"]})

        # [{"A": "fp16", CONSTANTS_KEY: "fp32"},
        #  {"A": "fp16", CONSTANTS_KEY: "fp64"},
        #  {"A": "fp32", CONSTANTS_KEY: "fp32"},
        #  {"A": "fp32", CONSTANTS_KEY: "fp64"}]

    Arrays keep the caller's order, so the last one varies fastest.

    :param arrays: ``{array: precision keys}`` -- the keys to try per array.
    :returns: one :data:`PrecisionMap` per combination (the full product).
    :raises ValueError: if ``arrays`` is empty or an array has no keys.
    """
    space = {name: list(choices) for name, choices in arrays.items()}
    if not space:
        raise ValueError("precision_grid needs at least one array")
    for name, choices in space.items():
        if not choices:
            raise ValueError(f"No precision keys given for array {name!r}")
    names = list(space)
    return [
        dict(zip(names, combo))
        for combo in itertools.product(*(space[name] for name in names))
    ]


@dataclass
class ExperimentConfig:
    """
    The program under test plus everything needed to run it -- shared by all analyses.

    :param name: identifier used to group results in the database.
    :param program: the ``dace.SDFG`` under test
    :param inputs: per-array input distributions for the *read* arrays.
    :param promotion_rules: promotion rules for resolving precision conflicts (e.g. ``{frozenset({fp16, fp32}): fp32}``); ``None`` uses ``DEFAULT_PROMOTION_RULES``.
    :param symbols: values for the SDFG's free symbols (e.g. ``{"N": 100}``);
    :param scalar_args: values for non-array scalar arguments.
    :param target: ``"cpu"`` (default) or ``"gpu"``;
    :param seed: base RNG seed for inputs and noise, shared across analyses.
    :param gpu_block_size: GPU thread-block size ``[x, y, z]`` (x = contiguous dim) set on every GPU_Device map; ``None`` uses DaCe's default.
    :param gpu_vectorize: if True, runs VectorizeGPU()
    :param gpu_vectorize_config: the ``VectorizeConfig`` used when ``gpu_vectorize`` is set; ``None`` uses ``DEFAULT_GPU_VECTORIZE_CONFIG``.
    """

    name: str
    program: dace.SDFG
    inputs: dict[str, DistributionLike] = field(default_factory=dict)
    promotion_rules: (
        dict[frozenset[dace.dtypes.typeclass], dace.dtypes.typeclass] | None
    ) = None
    symbols: dict[str, int] = field(default_factory=dict)
    scalar_args: dict[str, Any] = field(default_factory=dict)
    target: str = "cpu"
    seed: int = 0
    gpu_block_size: list[int] | None = None
    gpu_vectorize: bool = False
    gpu_vectorize_config: VectorizeConfig | None = None


@dataclass
class PerformanceAnalysisConfig:
    """
    Measure wall-clock runtime across precision points.
    """

    experiment: ExperimentConfig
    precisions: list[PrecisionMap] = field(default_factory=lambda: [{}])
    noise: dict[str, Noise] = field(default_factory=dict)
    n_warmup: int = 1
    n_reps: int = 10


@dataclass
class ErrorAnalysisConfig:
    """
    Measure per-array error of each precision point against a high-precision
    reference, aggregated over ``n_samples`` input realisations.
    ``reference`` is a single key (e.g. ``"mpfr128"``, ``"fp64"``) or a per-array ``{name: key}`` map.
    """

    experiment: ExperimentConfig
    precisions: list[PrecisionMap] = field(default_factory=lambda: [{}])
    noise: dict[str, Noise] = field(default_factory=dict)
    reference: str | dict[str, str] = "fp64"
    n_samples: int = 1


@dataclass
class PerturbationAnalysisConfig:
    """
    Measure input sensitivity: perturb one input array at a time and compare
    each written array against the clean run at the same precision point,
    aggregated over ``n_samples`` input realisations.
    ``noise`` names the inputs to perturb (each analysed separately);
    ``precisions`` lists the points to analyse at (default: the unmodified program).
    """

    experiment: ExperimentConfig
    noise: dict[str, Noise]
    precisions: list[PrecisionMap] = field(default_factory=lambda: [{}])
    n_samples: int = 1


@dataclass
class ErrorBudget:
    """
    How accurate a precision point has to be to count as feasible.

    Every constraint has to hold on every checked output array; one violation
    is enough to reject the point. There are two ways to state a constraint and
    both can be used at once -- if they cover the same metric and array, the
    tighter one wins.

    ``limits`` gives thresholds directly, per metric, either as one number for
    all arrays or as a ``{array: limit}`` map::

        ErrorBudget(limits={"rel_max": 1e-5, "l2_norm": {"A": 1e-7, "B": 1e-6}})

    ``from_perturbation`` takes them from the measured noise floor, i.e. how far
    input uncertainty alone already moves each output at the baseline point. The
    factor says how many times that error to allow, so 1.0 accepts every point
    whose numerical error stays under the error the inputs cause anyway::

        ErrorBudget(from_perturbation={"rel_max": 1.0})

    :param limits: ``{metric: limit}`` or ``{metric: {array: limit}}``; metrics
        are :class:`~fp_arena.experiment.results.ErrorStats` field names.
    :param from_perturbation: ``{metric: factor}`` scaling the noise floor.
    :param arrays: restrict the checked output arrays; ``None`` checks all.
    :param predicate: extra feasibility test on the point's per-array stats,
        run after the numeric constraints pass.
    """

    limits: dict[str, float | dict[str, float]] = field(default_factory=dict)
    from_perturbation: dict[str, float] | None = None
    arrays: list[str] | None = None
    predicate: Callable[[dict[str, ErrorStats]], bool] | None = None

    def __post_init__(self) -> None:
        if not self.limits and not self.from_perturbation and self.predicate is None:
            raise ValueError(
                "ErrorBudget constrains nothing: set limits, from_perturbation, "
                "and/or a predicate (otherwise every point is trivially feasible)"
            )
        # Local import: results imports PrecisionMap from this module.
        from fp_arena.experiment.results import ErrorStats

        known = {f.name for f in fields(ErrorStats)}
        for metric in (*self.limits, *(self.from_perturbation or {})):
            if metric not in known:
                raise ValueError(
                    f"Unknown error metric {metric!r}; known: {sorted(known)}"
                )
        for metric, factor in (self.from_perturbation or {}).items():
            if factor <= 0.0:
                raise ValueError(
                    f"from_perturbation[{metric!r}] must be a positive factor, "
                    f"got {factor}"
                )


#: The ``PerfResult`` timing phases usable as a selection objective.
OBJECTIVES = ("total", "kernel", "h2d", "d2h", "cast_in", "cast_out")


@dataclass
class SelectionAnalysisConfig:
    """
    Search precision assignments for the fastest one that stays inside an error budget.

    Composes the other three analyses over one shared :class:`ExperimentConfig`,
    so feasibility and speed are always measured on the same program at the same
    problem size: perturbation at ``speedup_baseline`` measures the noise floor,
    error measures every point in ``precisions``, and performance times only the
    points that met the budget.

    :param experiment: the program under test, shared by all three sub-analyses.
    :param precisions: the search space; see :func:`precision_grid`.
    :param budget: the accuracy a point must deliver to be admissible.
    :param noise: the inputs to perturb when measuring the noise floor (each
        analysed separately, as in :class:`PerturbationAnalysisConfig`).
    :param reference: the error analysis' high-precision reference.
    :param n_samples: input realisations, shared by error and perturbation.
    :param n_warmup: untimed warmup invocations per timed point.
    :param n_reps: timed repetitions per timed point.
    :param speedup_baseline: the point speedups are measured against, always
        error-measured and timed; ``{}`` is the unmodified program.
    :param objective: the timing phase to minimise, one of :data:`OBJECTIVES`.
    :param time_all: also time infeasible points, for a full speed/error table.
    :param perturb_all: also measure input sensitivity at every point; the
        budget still derives from ``speedup_baseline``.
    :param name: database name for the selection result; defaults to the
        experiment's (the sub-analyses always store under the experiment's).
    """

    experiment: ExperimentConfig
    precisions: list[PrecisionMap]
    budget: ErrorBudget
    noise: dict[str, Noise]
    reference: str | dict[str, str] = "fp64"
    n_samples: int = 1
    n_warmup: int = 1
    n_reps: int = 10
    speedup_baseline: PrecisionMap = field(default_factory=dict)
    objective: str = "total"
    time_all: bool = False
    perturb_all: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.noise:
            raise ValueError(
                "SelectionAnalysisConfig.noise must name at least one input "
                "array to perturb; the noise floor it measures is what the "
                "budget is judged against"
            )
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"Unknown objective {self.objective!r}; expected one of {list(OBJECTIVES)}"
            )
