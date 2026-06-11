# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Experiment config.
TODO: Select and perturbation configs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Union

import dace

from fp_arena.experiment.inputs import DistributionLike, Noise

#: ``{array_name: precision_key}`` -- pins a subset of arrays to a target format.
PrecisionMap = Dict[str, str]


@dataclass
class ExperimentConfig:
    """
    The program under test plus everything needed to run it -- shared by all analyses.

    :param name: identifier used to group results in the database.
    :param program: the ``dace.SDFG`` under test
    :param inputs: per-array input distributions for the *read* arrays.
    :param promotion_rules: promotion rules for resolving precision conflicts (e.g. ``{frozenset({fp16, fp32}): fp32}``).
    :param symbols: values for the SDFG's free symbols (e.g. ``{"N": 100}``);
    :param scalar_args: values for non-array scalar arguments.
    :param target: ``"cpu"`` (default) or ``"gpu"``;
    :param seed: base RNG seed for inputs and noise, shared across analyses.
    """

    name: str
    program: dace.SDFG
    inputs: Dict[str, DistributionLike] = field(default_factory=dict)
    promotion_rules: Dict[FrozenSet[dace.dtypes.typeclass], dace.dtypes.typeclass] = (
        field(default_factory=dict)
    )
    symbols: Dict[str, int] = field(default_factory=dict)
    scalar_args: Dict[str, Any] = field(default_factory=dict)
    target: str = "cpu"
    seed: int = 0


@dataclass
class PerformanceAnalysisConfig:
    """
    Measure wall-clock runtime across precision points.
    """

    experiment: ExperimentConfig
    precisions: List[PrecisionMap]
    noise: Dict[str, Noise] = field(default_factory=dict)
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
    precisions: List[PrecisionMap]
    noise: Dict[str, Noise] = field(default_factory=dict)
    reference: Union[str, Dict[str, str]] = "fp64"
    n_samples: int = 1
