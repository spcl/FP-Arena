# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Drivers that execute the experiment kinds.

* :func:`run_performance` -- compile each precision point and time it.
* :func:`run_error` -- dual-execute each point against a high-precision reference (lockstep on identical inputs).
* :func:`run_perturbation` -- perturb one input at a time and compare against the clean run at the same precision point.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

import dace

from fp_arena.experiment.config import (
    ErrorAnalysisConfig,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PrecisionMap,
)
from fp_arena.experiment.results import (
    ErrorResult,
    ErrorStats,
    PerfResult,
    PerturbationResult,
)
from fp_arena.experiment.inputs import make_call_args
from fp_arena.experiment.retarget import (
    _transfer_direction,
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


def _reset_arrays(working: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Refill ``working``'s array buffers in place from ``source``."""
    for k, v in source.items():
        if isinstance(v, np.ndarray):
            working[k][...] = v


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
        "sq_err_sum": 0.0,
        "sq_ref_sum": 0.0,
        "ref_abs_sum": 0.0,
        "ref_abs_max": 0.0,
    }


def _accumulate(acc: Dict[str, float], ref: np.ndarray, cand: np.ndarray) -> None:
    """Fold one (reference, candidate) array pair into the running error stats."""
    r = np.asarray(ref, dtype=np.float64).ravel()
    c = np.asarray(cand, dtype=np.float64).ravel()
    diff = np.abs(c - r)
    diff = np.where(np.isnan(diff), np.inf, diff)
    denom = np.abs(r)
    if diff.size:
        acc["abs_sum"] += float(diff.sum())
        acc["abs_cnt"] += int(diff.size)
        acc["abs_max"] = max(acc["abs_max"], float(diff.max()))
        acc["sq_err_sum"] += float(np.square(diff).sum())
        acc["sq_ref_sum"] += float(np.square(r).sum())
        acc["ref_abs_sum"] += float(denom.sum())
        acc["ref_abs_max"] = max(acc["ref_abs_max"], float(denom.max()))
    mask = denom > 0
    with np.errstate(invalid="ignore"):
        rel = diff[mask] / denom[mask]
    rel = np.where(np.isnan(rel), np.inf, rel)
    if rel.size:
        acc["rel_sum"] += float(rel.sum())
        acc["rel_cnt"] += int(rel.size)
        acc["rel_max"] = max(acc["rel_max"], float(rel.max()))


def _ratio(num: float, den: float) -> float:
    """``num/den`` with the zero-reference convention: ``0/0 -> 0``, ``x/0 -> inf``."""
    if den > 0.0:
        return num / den
    return 0.0 if num <= 0.0 else math.inf


def _finalize(acc: Dict[str, float]) -> ErrorStats:
    abs_mean = acc["abs_sum"] / acc["abs_cnt"] if acc["abs_cnt"] else 0.0
    rel_mean = acc["rel_sum"] / acc["rel_cnt"] if acc["rel_cnt"] else 0.0
    err_power = acc["sq_err_sum"]
    ref_power = acc["sq_ref_sum"]
    if err_power <= 0.0:
        snr = math.inf
    elif (
        ref_power <= 0.0 or not math.isfinite(ref_power) or not math.isfinite(err_power)
    ):
        snr = -math.inf
    else:
        snr = 10.0 * math.log10(ref_power / err_power)
    l1 = acc["abs_sum"]
    l2 = math.sqrt(err_power)
    linf = acc["abs_max"]
    return ErrorStats(
        abs_mean=abs_mean,
        rel_mean=rel_mean,
        rel_max=acc["rel_max"],
        l1=l1,
        l2=l2,
        linf=linf,
        l1_norm=_ratio(l1, acc["ref_abs_sum"]),
        l2_norm=_ratio(l2, math.sqrt(ref_power)),
        linf_norm=_ratio(linf, acc["ref_abs_max"]),
        snr=snr,
    )


def _has_cast_map(state) -> bool:
    """Whether ``state`` contains a precision-cast map."""
    return any(
        isinstance(n, dace.nodes.MapEntry) and n.map.label.startswith("cast_map_")
        for n in state.nodes()
    )


def _classify_state(sdfg: dace.SDFG, state) -> str:
    """Assign ``state`` to a timing phase: h2d/d2h, cast_in/cast_out, or kernel."""
    direction = _transfer_direction(sdfg, state)
    if direction is not None:
        return direction
    if _has_cast_map(state):
        return "cast_out" if state.label.startswith("copy_out") else "cast_in"
    return "kernel"


def _group_per_invocation(
    samples: List[float], n_invocations: int, name: str
) -> List[float]:
    """Sum a timer's per-execution samples into one value per invocation (loop bodies fire repeatedly)."""
    if not samples:
        return []
    if len(samples) % n_invocations != 0:
        raise ValueError(
            f"Timer {name!r} fired {len(samples)} times over {n_invocations} "
            f"invocations; per-rep phase attribution requires a static "
            f"per-invocation execution count"
        )
    k = len(samples) // n_invocations
    if k == 1:
        return list(samples)
    return [math.fsum(samples[i * k : (i + 1) * k]) for i in range(n_invocations)]


def _phase_series(report, name_pred, n_reps: int, n_invocations: int) -> List[float]:
    """Per-rep milliseconds summed over matching timers, warmup invocations dropped."""
    out = [0.0] * n_reps
    matched = False
    if report is not None:
        for names in report.durations.values():
            for name, tid_map in names.items():
                if not name_pred(name):
                    continue
                for times_ms in tid_map.values():
                    per_inv = _group_per_invocation(times_ms, n_invocations, name)
                    if not per_inv:
                        continue
                    for i, ms in enumerate(per_inv[-n_reps:]):
                        out[i] += ms
                    matched = True
    return out if matched else []


def _phase_breakdown(
    sdfg: dace.SDFG, n_reps: int, n_invocations: int
) -> Dict[str, Any]:
    """Reduce the latest report into per-rep phase series; ``total`` is their sum."""
    report = sdfg.get_latest_report()
    # Map each state's report name -> phase category.
    cat_of: Dict[str, str] = {}
    for state in sdfg.all_states():
        cat_of.setdefault(f"State {state.label}", _classify_state(sdfg, state))

    def series(category: str) -> List[float]:
        return _phase_series(
            report, lambda nm: cat_of.get(nm) == category, n_reps, n_invocations
        )

    phases = {
        "h2d_times": series("h2d"),
        "cast_in_times": series("cast_in"),
        "kernel_times": series("kernel"),
        "cast_out_times": series("cast_out"),
        "d2h_times": series("d2h"),
    }
    nonempty = [p for p in phases.values() if p]
    total = [math.fsum(col) for col in zip(*nonempty)] if nonempty else []
    return {"total_times": total, **phases}


def _fmt_pin(pin_map: PrecisionMap) -> str:
    """Compact one-line summary of a precision point for progress display."""
    return " ".join(f"{k}={v}" for k, v in pin_map.items())


def _output_arrays(sdfg: dace.SDFG) -> List[str]:
    """Non-transient written array names -- the arrays error metrics are reduced over."""
    _, writes = sdfg.read_and_write_sets()
    return sorted(
        name
        for name in writes
        if name in sdfg.arrays
        and isinstance(sdfg.arrays[name], dace.data.Array)
        and not sdfg.arrays[name].transient
    )


#: Per input sample: (pristine call args, reference outputs of the written arrays).
ReferenceSamples = List[Tuple[Dict[str, Any], Dict[str, Any]]]


def _pin_tag(pin_map: PrecisionMap) -> str:
    """Identifier-safe tag naming a precision point's build folder."""
    return "_".join(f"{k}_{v}" for k, v in sorted(pin_map.items())) or "baseline"


def _distinguish(sdfg: dace.SDFG, tag: str) -> None:
    """Unique build folder per point."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag)
    sdfg.name = f"{sdfg.name}_{safe}"


def compile_reference(experiment, reference):
    """Build and compile the high-precision reference SDFG once."""
    sdfg = fresh_sdfg(experiment)
    apply_reference(sdfg, reference, experiment.promotion_rules)
    apply_target(
        sdfg,
        experiment.target,
        gpu_block_size=experiment.gpu_block_size,
        gpu_vectorize=experiment.gpu_vectorize,
    )
    _distinguish(sdfg, "reference")
    return sdfg.compile()


def run_reference(
    experiment, reference, n_samples: int, seed: int, noise=None
) -> ReferenceSamples:
    """
    Execute the reference once per input sample.
    """
    ref_csdfg = compile_reference(experiment, reference)
    program = experiment.program
    reads, _ = program.read_and_write_sets()
    outputs = _output_arrays(program)
    samples: ReferenceSamples = []
    for rng in tqdm(
        _sample_rngs(seed, n_samples), desc="reference", unit="smp", leave=False
    ):
        args = make_call_args(program, experiment, rng, noise, reads=reads)
        ref_args = _copy_args(args)
        ref_csdfg(**ref_args)
        samples.append((args, {name: ref_args[name] for name in outputs}))
    return samples


def measure_error(
    experiment,
    pin_map: PrecisionMap,
    ref_samples: ReferenceSamples,
    seed: int,
) -> ErrorResult:
    """
    Execute one precision point on the reference's input samples and reduce the per-array error against the cached reference outputs.
    """
    cand_sdfg = fresh_sdfg(experiment)
    apply_precision(cand_sdfg, pin_map, experiment.promotion_rules)
    apply_target(
        cand_sdfg,
        experiment.target,
        gpu_block_size=experiment.gpu_block_size,
        gpu_vectorize=experiment.gpu_vectorize,
    )
    _distinguish(cand_sdfg, _pin_tag(pin_map))
    cand_csdfg = cand_sdfg.compile()

    acc = {name: _new_acc() for name in _output_arrays(cand_sdfg)}
    for args, ref_out in tqdm(ref_samples, desc="samples", unit="smp", leave=False):
        cand_args = _copy_args(args)
        cand_csdfg(**cand_args)
        for name in acc:
            _accumulate(acc[name], ref_out[name], cand_args[name])

    errors = {name: _finalize(a) for name, a in acc.items()}
    return ErrorResult(
        precision=dict(pin_map), errors=errors, n_samples=len(ref_samples), seed=seed
    )


def run_performance(
    cfg: PerformanceAnalysisConfig, store: Optional[ResultStore] = None
) -> List[PerfResult]:
    """Time each precision point, optionally appending to ``store``."""
    results: List[PerfResult] = []
    target = cfg.experiment.target
    # CPU: host std::chrono; GPU: CUDA events (on-device, not async-launch, time).
    provider = (
        dace.InstrumentationType.GPU_Events
        if target == "gpu"
        else dace.InstrumentationType.Timer
    )
    n_invocations = cfg.n_warmup + cfg.n_reps

    prev_each = dace.Config.get("instrumentation", "report_each_invocation")
    dace.Config.set("instrumentation", "report_each_invocation", value=False)
    prev_streams = dace.Config.get("compiler", "cuda", "max_concurrent_streams")
    if target == "gpu":
        dace.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)

    points = tqdm(cfg.precisions, desc="performance", unit="pt")
    try:
        for pin_map in points:
            points.set_postfix_str(_fmt_pin(pin_map))
            sdfg = fresh_sdfg(cfg.experiment)
            apply_precision(sdfg, pin_map, cfg.experiment.promotion_rules)
            apply_target(
                sdfg,
                target,
                gpu_block_size=cfg.experiment.gpu_block_size,
                gpu_vectorize=cfg.experiment.gpu_vectorize,
            )
            _distinguish(sdfg, _pin_tag(pin_map))
            # Time every state; classified into a phase at readout.
            for state in sdfg.all_states():
                state.instrument = provider
            csdfg = sdfg.compile()
            sdfg.clear_instrumentation_reports()

            rng = _sample_rngs(cfg.experiment.seed, 1)[0]
            initial_args = make_call_args(sdfg, cfg.experiment, rng, cfg.noise)
            args = _copy_args(initial_args)
            runs = tqdm(
                total=n_invocations,
                desc="warmup",
                unit="run",
                leave=False,
            )
            for _ in range(cfg.n_warmup):
                _reset_arrays(args, initial_args)
                csdfg(**args)
                runs.update(1)
            runs.set_description("reps")
            for _ in range(cfg.n_reps):
                _reset_arrays(args, initial_args)
                csdfg(**args)
                runs.update(1)
            runs.close()

            csdfg.finalize()

            result = PerfResult(
                precision=dict(pin_map),
                seed=cfg.experiment.seed,
                **_phase_breakdown(sdfg, cfg.n_reps, n_invocations),
            )
            results.append(result)
            if store is not None:
                store.add(
                    cfg.experiment.name,
                    result,
                    symbols=cfg.experiment.symbols,
                    scalars=cfg.experiment.scalar_args,
                )
    finally:
        dace.Config.set("instrumentation", "report_each_invocation", value=prev_each)
        dace.Config.set(
            "compiler", "cuda", "max_concurrent_streams", value=prev_streams
        )
    return results


def run_perturbation(
    cfg: PerturbationAnalysisConfig, store: Optional[ResultStore] = None
) -> List[PerturbationResult]:
    """
    Measure per-array output sensitivity to input noise: for each precision
    point, execute clean and perturbed inputs (one noisy array at a time) on the
    same compiled SDFG and reduce the output deviation over ``cfg.n_samples``
    input realisations. One result per (precision point, perturbed input).
    """
    if not cfg.noise:
        raise ValueError(
            "PerturbationAnalysisConfig.noise must name at least one input array"
        )
    exp = cfg.experiment
    results: List[PerturbationResult] = []
    points = tqdm(cfg.precisions, desc="perturbation", unit="pt")
    for pin_map in points:
        points.set_postfix_str(_fmt_pin(pin_map))
        sdfg = fresh_sdfg(exp)
        apply_precision(sdfg, pin_map, exp.promotion_rules)
        apply_target(
            sdfg,
            exp.target,
            gpu_block_size=exp.gpu_block_size,
            gpu_vectorize=exp.gpu_vectorize,
        )
        _distinguish(sdfg, _pin_tag(pin_map))
        csdfg = sdfg.compile()

        reads, _ = sdfg.read_and_write_sets()
        for name in cfg.noise:
            if name not in reads or name not in sdfg.arrays:
                raise ValueError(
                    f"Perturbed array {name!r} is not a read input of SDFG {sdfg.name!r}"
                )
        outputs = _output_arrays(sdfg)
        acc = {pert: {out: _new_acc() for out in outputs} for pert in cfg.noise}

        for rng in tqdm(
            _sample_rngs(exp.seed, cfg.n_samples),
            desc="samples",
            unit="smp",
            leave=False,
        ):
            clean_args = make_call_args(sdfg, exp, rng, reads=reads)
            base_args = _copy_args(clean_args)
            csdfg(**base_args)
            for pert_name, pert_noise in cfg.noise.items():
                pert_args = _copy_args(clean_args)
                pert_args[pert_name] = pert_noise.apply(clean_args[pert_name], rng)
                csdfg(**pert_args)
                for out in outputs:
                    _accumulate(acc[pert_name][out], base_args[out], pert_args[out])

        for pert_name in cfg.noise:
            result = PerturbationResult(
                precision=dict(pin_map),
                perturbed=pert_name,
                errors={out: _finalize(a) for out, a in acc[pert_name].items()},
                n_samples=cfg.n_samples,
                seed=exp.seed,
            )
            results.append(result)
            if store is not None:
                store.add(
                    exp.name, result, symbols=exp.symbols, scalars=exp.scalar_args
                )
    return results


def run_error(
    cfg: ErrorAnalysisConfig, store: Optional[ResultStore] = None
) -> List[ErrorResult]:
    """Measure per-array error of each precision point, optionally appending to ``store`` database."""
    exp = cfg.experiment
    ref_samples = run_reference(
        exp, cfg.reference, cfg.n_samples, exp.seed, noise=cfg.noise
    )
    results: List[ErrorResult] = []
    points = tqdm(cfg.precisions, desc="error", unit="pt")
    for pin_map in points:
        points.set_postfix_str(_fmt_pin(pin_map))
        result = measure_error(exp, pin_map, ref_samples, exp.seed)
        results.append(result)
        if store is not None:
            store.add(exp.name, result, symbols=exp.symbols, scalars=exp.scalar_args)
    return results
