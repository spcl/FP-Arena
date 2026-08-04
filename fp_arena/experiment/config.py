# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Experiment config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig

from fp_arena.experiment.inputs import DistributionLike, Noise

#: ``{array_name: precision_key}`` -- pins a subset of arrays to a target format.
PrecisionMap = dict[str, str]

#: Reserved :data:`PrecisionMap` key that targets float literals; absent means literals stay fp64.
CONSTANTS_KEY = "__constants__"


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
    precisions: list[PrecisionMap]
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
    precisions: list[PrecisionMap]
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
