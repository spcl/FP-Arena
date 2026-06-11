# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Drivers that execute the experiment kinds.

* :func:`run_performance` -- compile each precision point and time it.
* :func:`run_error` -- dual-execute each point against a high-precision reference (lockstep on identical inputs).

TODO: Select and perturbation runners
"""

import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import dace

from fp_arena.experiment.config import (
    ErrorAnalysisConfig,
    PerformanceAnalysisConfig,
    PrecisionMap,
)
from fp_arena.experiment.results import ErrorResult, ErrorStats, PerfResult
from fp_arena.experiment.inputs import make_call_args
from fp_arena.experiment.retarget import (
    apply_precision,
    apply_reference,
    apply_target,
    fresh_sdfg,
)
from fp_arena.experiment.store import ResultStore


def _copy_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Independent copy of the array arguments; scalars/symbols pass through."""
    return {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in args.items()
    }


def _sample_rngs(seed: int, n: int) -> List[np.random.Generator]:
    """
    ``n`` independent, reproducible per-sample generators from one base seed.
    """
    return [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)
    ]


def _new_acc() -> Dict[str, float]:
    return {
        "abs_sum": 0.0,
        "abs_cnt": 0,
        "abs_max": 0.0,
        "rel_sum": 0.0,
        "rel_cnt": 0,
        "rel_max": 0.0,
    }


def _accumulate(acc: Dict[str, float], ref: np.ndarray, cand: np.ndarray) -> None:
    """Fold one (reference, candidate) array pair into the running error stats."""
    r = np.asarray(ref, dtype=np.float64).ravel()
    c = np.asarray(cand, dtype=np.float64).ravel()
    diff = np.abs(c - r)
    diff = np.where(np.isnan(diff), np.inf, diff)
    if diff.size:
        acc["abs_sum"] += float(diff.sum())
        acc["abs_cnt"] += int(diff.size)
        acc["abs_max"] = max(acc["abs_max"], float(diff.max()))
    denom = np.abs(r)
    mask = denom > 0
    with np.errstate(invalid="ignore"):
        rel = diff[mask] / denom[mask]
    rel = np.where(np.isnan(rel), np.inf, rel)
    if rel.size:
        acc["rel_sum"] += float(rel.sum())
        acc["rel_cnt"] += int(rel.size)
        acc["rel_max"] = max(acc["rel_max"], float(rel.max()))


def _finalize(acc: Dict[str, float]) -> ErrorStats:
    abs_mean = acc["abs_sum"] / acc["abs_cnt"] if acc["abs_cnt"] else 0.0
    rel_mean = acc["rel_sum"] / acc["rel_cnt"] if acc["rel_cnt"] else 0.0
    return ErrorStats(
        abs_mean=abs_mean,
        abs_max=acc["abs_max"],
        rel_mean=rel_mean,
        rel_max=acc["rel_max"],
    )


Reference = Tuple[Any, dace.SDFG]


def compile_reference(experiment, reference) -> Reference:
    """Build and compile the high-precision reference SDFG once."""
    sdfg = fresh_sdfg(experiment)
    apply_reference(sdfg, reference, experiment.promotion_rules)
    apply_target(sdfg, experiment.target)
    return sdfg.compile(), sdfg


def measure_error(
    experiment,
    pin_map: PrecisionMap,
    reference,
    n_samples: int,
    seed: int,
    ref: Optional[Reference] = None,
    noise=None,
) -> ErrorResult:
    """
    Dual-execute one precision point against the reference and reduce per-array error over ``n_samples`` input realisations.
    """

    if ref is None:
        ref = compile_reference(experiment, reference)
    ref_csdfg, _ = ref

    cand_sdfg = fresh_sdfg(experiment)
    apply_precision(cand_sdfg, pin_map, experiment.promotion_rules)
    apply_target(cand_sdfg, experiment.target)
    cand_csdfg = cand_sdfg.compile()

    reads, writes = cand_sdfg.read_and_write_sets()
    acc = {
        name: _new_acc()
        for name in writes
        if name in cand_sdfg.arrays
        and isinstance(cand_sdfg.arrays[name], dace.data.Array)
        and not cand_sdfg.arrays[name].transient
    }

    for rng in _sample_rngs(seed, n_samples):
        ref_args = make_call_args(cand_sdfg, experiment, rng, noise, reads=reads)
        cand_args = _copy_args(ref_args)
        ref_csdfg(**ref_args)
        cand_csdfg(**cand_args)
        for name in acc:
            _accumulate(acc[name], ref_args[name], cand_args[name])

    errors = {name: _finalize(a) for name, a in acc.items()}
    return ErrorResult(
        precision=dict(pin_map), errors=errors, n_samples=n_samples, seed=seed
    )


def run_performance(
    cfg: PerformanceAnalysisConfig, store: Optional[ResultStore] = None
) -> List[PerfResult]:
    """Time each precision point, optionally appending to ``store``."""
    results: List[PerfResult] = []
    for pin_map in cfg.precisions:
        sdfg = fresh_sdfg(cfg.experiment)
        apply_precision(sdfg, pin_map, cfg.experiment.promotion_rules)
        apply_target(sdfg, cfg.experiment.target)
        csdfg = sdfg.compile()

        rng = _sample_rngs(cfg.experiment.seed, 1)[0]
        args = make_call_args(sdfg, cfg.experiment, rng, cfg.noise)
        for _ in range(cfg.n_warmup):
            csdfg(**_copy_args(args))

        times: List[float] = []
        for _ in range(cfg.n_reps):
            run_args = _copy_args(args)
            t0 = time.perf_counter()
            csdfg(**run_args)
            times.append(time.perf_counter() - t0)

        result = PerfResult(
            precision=dict(pin_map),
            times=times,
            time_median=statistics.median(times),
            time_std=statistics.pstdev(times) if len(times) > 1 else 0.0,
            time_min=min(times),
            seed=cfg.experiment.seed,
        )
        results.append(result)
        if store is not None:
            store.add_perf(cfg.experiment.name, result)
    return results


def run_error(
    cfg: ErrorAnalysisConfig, store: Optional[ResultStore] = None
) -> List[ErrorResult]:
    """Measure per-array error of each precision point, optionally appending to ``store`` database."""
    ref = compile_reference(cfg.experiment, cfg.reference)
    results: List[ErrorResult] = []
    for pin_map in cfg.precisions:
        result = measure_error(
            cfg.experiment,
            pin_map,
            cfg.reference,
            cfg.n_samples,
            cfg.experiment.seed,
            ref=ref,
            noise=cfg.noise,
        )
        results.append(result)
        if store is not None:
            store.add_error(cfg.experiment.name, result)
    return results
