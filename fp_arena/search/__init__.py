# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
FP-Arena experiment framework: precision performance and error analysis.
"""

from fp_arena.search import registry
from fp_arena.search.config import (
    CONSTANTS_KEY,
    OBJECTIVES,
    SCALARS_KEY,
    ErrorAnalysisConfig,
    ErrorBudget,
    ExperimentConfig,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PrecisionMap,
)
from fp_arena.search.inputs import Noise
from fp_arena.search.knobs import Knob, KnobSelection, find_knobs
from fp_arena.search.results import (
    METRICS,
    ConstraintResult,
    ErrorResult,
    ErrorStats,
    Metric,
    PerfResult,
    PerturbationResult,
)
from fp_arena.search.retarget import candidate_fp_arrays
from fp_arena.search.runner import run_error, run_performance, run_perturbation
from fp_arena.search.screening import Screening, screen
from fp_arena.search.search import (
    SearchCandidate,
    SearchEvaluation,
    SearchFailure,
    SearchResult,
    SelectionSearchConfig,
    format_search,
    run_search,
)
from fp_arena.search.store import ResultStore, StoredResult

__all__ = [
    "CONSTANTS_KEY",
    "METRICS",
    "OBJECTIVES",
    "SCALARS_KEY",
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
    "SearchEvaluation",
    "SearchFailure",
    "SearchResult",
    "SelectionSearchConfig",
    "StoredResult",
    "candidate_fp_arrays",
    "find_knobs",
    "format_search",
    "registry",
    "run_error",
    "run_performance",
    "run_perturbation",
    "run_search",
    "screen",
]
