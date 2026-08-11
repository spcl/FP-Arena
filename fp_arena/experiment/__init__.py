# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena experiment framework: precision performance and error analysis.
"""

from fp_arena.experiment import registry
from fp_arena.experiment.config import (
    CONSTANTS_KEY,
    OBJECTIVES,
    ErrorAnalysisConfig,
    ErrorBudget,
    ExperimentConfig,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PrecisionMap,
    SelectionAnalysisConfig,
    precision_grid,
)
from fp_arena.experiment.inputs import Noise
from fp_arena.experiment.results import (
    METRICS,
    ConstraintResult,
    ErrorResult,
    ErrorStats,
    Metric,
    PerfResult,
    PerturbationResult,
    SelectionCandidate,
    SelectionResult,
)
from fp_arena.experiment.retarget import candidate_fp_arrays
from fp_arena.experiment.runner import run_error, run_performance, run_perturbation
from fp_arena.experiment.selection import format_selection, run_selection
from fp_arena.experiment.store import ResultStore, StoredResult

__all__ = [
    "CONSTANTS_KEY",
    "METRICS",
    "OBJECTIVES",
    "ConstraintResult",
    "ErrorAnalysisConfig",
    "ErrorBudget",
    "ErrorResult",
    "ErrorStats",
    "ExperimentConfig",
    "Metric",
    "Noise",
    "PerfResult",
    "PerformanceAnalysisConfig",
    "PerturbationAnalysisConfig",
    "PerturbationResult",
    "PrecisionMap",
    "ResultStore",
    "SelectionAnalysisConfig",
    "SelectionCandidate",
    "SelectionResult",
    "StoredResult",
    "candidate_fp_arrays",
    "format_selection",
    "precision_grid",
    "registry",
    "run_error",
    "run_performance",
    "run_perturbation",
    "run_selection",
]
