# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena experiment framework: precision performance and error analysis.
"""

from fp_arena.experiment.config import (
    ErrorAnalysisConfig,
    ExperimentConfig,
    PerformanceAnalysisConfig,
    PrecisionMap,
)
from fp_arena.experiment.results import ErrorResult, ErrorStats, PerfResult
from fp_arena.experiment.inputs import Noise
from fp_arena.experiment import registry
from fp_arena.experiment.runner import run_error, run_performance
from fp_arena.experiment.store import ResultStore, StoredResult

__all__ = [
    "ExperimentConfig",
    "PerformanceAnalysisConfig",
    "ErrorAnalysisConfig",
    "PrecisionMap",
    "PerfResult",
    "ErrorResult",
    "ErrorStats",
    "Noise",
    "run_performance",
    "run_error",
    "registry",
    "ResultStore",
    "StoredResult",
]
