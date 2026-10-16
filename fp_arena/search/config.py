# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Experiment config.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig

from fp_arena.search.inputs import DistributionLike, Noise

if TYPE_CHECKING:
    from fp_arena.search.results import ErrorStats

#: ``{array_name: precision_key}`` -- pins a subset of arrays to a target format.
PrecisionMap = dict[str, str]

#: Reserved :data:`PrecisionMap` key for the program's constants (float literals, constant
#: fills, read-only float scalars and symbols); absent leaves them unchanged.
CONSTANTS_KEY = "__constants__"

#: Reserved :data:`PrecisionMap` key that targets every single-element float temporary at once.
SCALARS_KEY = "__scalars__"


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
    :param gpu_offload: program-specific GPU offload ``(sdfg) -> None`` run in place of ``apply_gpu_transformations`` when ``target="gpu"``; ``None`` uses the generic offload. Must be a module-level function: the config is pickled to the search's workers.
    :param derived_outputs: extra outputs computed from a run's arguments after the call, ``{name: fn(args) -> ndarray}``.
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
    gpu_offload: Callable[[dace.SDFG], None] | None = None
    derived_outputs: dict[str, Callable[[dict[str, Any]], Any]] = field(
        default_factory=dict
    )


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
        are :class:`~fp_arena.search.results.ErrorStats` field names.
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
        from fp_arena.search.results import METRICS

        for metric in (*self.limits, *(self.from_perturbation or {})):
            if metric not in METRICS:
                raise ValueError(
                    f"Unknown error metric {metric!r}; known: {sorted(METRICS)}"
                )
        for metric, factor in (self.from_perturbation or {}).items():
            if factor <= 0.0:
                raise ValueError(
                    f"from_perturbation[{metric!r}] must be a positive factor, "
                    f"got {factor}"
                )


#: The ``PerfResult`` timing phases usable as a selection objective.
OBJECTIVES = ("total", "kernel", "h2d", "d2h", "cast_in", "cast_out")

