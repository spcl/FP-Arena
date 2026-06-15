# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Drivers that execute the experiment kinds.

* :func:`run_performance` -- compile each precision point and time it.
* :func:`run_error` -- dual-execute each point against a high-precision reference (lockstep on identical inputs).

TODO: Select and perturbation runners
"""

import statistics
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


def _phase_series(report, name_pred, n: int) -> List[float]:
    """
    Per-repetition milliseconds for every Timer whose name matches ``name_pred`` in the given instrumentation report.
    The last ``n`` samples are returned.
    """
    matched: List[List[float]] = []
    if report is not None:
        for names in report.durations.values():
            for name, tid_map in names.items():
                if not name_pred(name):
                    continue
                for times_ms in tid_map.values():
                    matched.append(times_ms[-n:])
    if not matched:
        return []
    out = [0.0] * n
    for samples in matched:
        offset = n - len(samples)
        for i, ms in enumerate(samples):
            if offset + i >= 0:
                out[offset + i] += ms
    return out


def _phase_breakdown(sdfg: dace.SDFG, n_reps: int) -> Dict[str, Any]:
    """
    Split the latest instrumentation report into copy_in / copy_out / compute
    series (milliseconds), where ``compute = total - copy_in - copy_out`` per rep.
    All series are empty when the report carries no Timer data.
    """
    report = sdfg.get_latest_report()
    total = _phase_series(report, lambda nm: nm.startswith("SDFG "), n_reps)
    copy_in = _phase_series(report, lambda nm: nm == "State copy_in", n_reps)
    copy_out = _phase_series(
        report, lambda nm: nm.startswith("State copy_out_"), n_reps
    )

    if total:
        ci = copy_in or [0.0] * n_reps
        co = copy_out or [0.0] * n_reps
        compute = [max(0.0, total[i] - ci[i] - co[i]) for i in range(n_reps)]
    else:
        compute = []

    def _median(xs: List[float]) -> float:
        return statistics.median(xs) if xs else 0.0

    return {
        "total_times": total,
        "copy_in_times": copy_in,
        "copy_out_times": copy_out,
        "compute_times": compute,
        "total_median": _median(total),
        "copy_in_median": _median(copy_in),
        "copy_out_median": _median(copy_out),
        "compute_median": _median(compute),
    }


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
    prev_each = dace.Config.get("instrumentation", "report_each_invocation")
    dace.Config.set("instrumentation", "report_each_invocation", value=False)
    try:
        for pin_map in cfg.precisions:
            sdfg = fresh_sdfg(cfg.experiment)
            apply_precision(
                sdfg, pin_map, cfg.experiment.promotion_rules, instrument=True
            )
            apply_target(sdfg, cfg.experiment.target)
            sdfg.instrument = dace.InstrumentationType.Timer
            csdfg = sdfg.compile()
            sdfg.clear_instrumentation_reports()

            rng = _sample_rngs(cfg.experiment.seed, 1)[0]
            args = make_call_args(sdfg, cfg.experiment, rng, cfg.noise)
            for _ in range(cfg.n_warmup):
                csdfg(**_copy_args(args))
            for _ in range(cfg.n_reps):
                csdfg(**_copy_args(args))

            csdfg.finalize()

            result = PerfResult(
                precision=dict(pin_map),
                seed=cfg.experiment.seed,
                **_phase_breakdown(sdfg, cfg.n_reps),
            )
            results.append(result)
            if store is not None:
                store.add_perf(cfg.experiment.name, result)
    finally:
        dace.Config.set("instrumentation", "report_each_invocation", value=prev_each)
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
