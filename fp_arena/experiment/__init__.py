# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena experiment framework: precision performance and error analysis.
"""

from fp_arena.experiment import registry
from fp_arena.experiment.config import (
    CONSTANTS_KEY,
    ErrorAnalysisConfig,
    ExperimentConfig,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PrecisionMap,
)
from fp_arena.experiment.inputs import Noise
from fp_arena.experiment.results import (
    ErrorResult,
    ErrorStats,
    PerfResult,
    PerturbationResult,
)
from fp_arena.experiment.runner import run_error, run_performance, run_perturbation
from fp_arena.experiment.store import ResultStore, StoredResult

__all__ = [
    "CONSTANTS_KEY",
    "ErrorAnalysisConfig",
    "ErrorResult",
    "ErrorStats",
    "ExperimentConfig",
    "Noise",
    "PerfResult",
    "PerformanceAnalysisConfig",
    "PerturbationAnalysisConfig",
    "PerturbationResult",
    "PrecisionMap",
    "ResultStore",
    "StoredResult",
    "registry",
    "run_error",
    "run_performance",
    "run_perturbation",
]
