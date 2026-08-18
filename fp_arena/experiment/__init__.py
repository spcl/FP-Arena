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
from fp_arena.experiment.knobs import Knob, KnobSelection, find_knobs
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
from fp_arena.experiment.screening import Screening, screen
from fp_arena.experiment.search import (
    SearchCandidate,
    SearchResult,
    SelectionSearchConfig,
    format_search,
    run_search,
)
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
    "Knob",
    "KnobSelection",
    "Metric",
    "Noise",
    "PerfResult",
    "PerformanceAnalysisConfig",
    "PerturbationAnalysisConfig",
    "PerturbationResult",
    "PrecisionMap",
    "ResultStore",
    "Screening",
    "SearchCandidate",
    "SearchResult",
    "SelectionAnalysisConfig",
    "SelectionCandidate",
    "SelectionResult",
    "SelectionSearchConfig",
    "StoredResult",
    "candidate_fp_arrays",
    "find_knobs",
    "format_search",
    "format_selection",
    "precision_grid",
    "registry",
    "run_error",
    "run_performance",
    "run_perturbation",
    "run_search",
    "run_selection",
    "screen",
]
