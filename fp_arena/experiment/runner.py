# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Drivers that execute the experiment kinds.

* :func:`run_performance` -- compile each precision point and time it.
* :func:`run_error` -- dual-execute each point against a high-precision reference (lockstep on identical inputs).
* :func:`run_perturbation` -- perturb one input at a time and compare against the clean run at the same precision point.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any

import dace
import numpy as np
from tqdm.auto import tqdm

from fp_arena.experiment.config import (
    ErrorAnalysisConfig,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PrecisionMap,
)
from fp_arena.experiment.inputs import make_call_args
from fp_arena.experiment.results import (
    ErrorResult,
    ErrorStats,
    PerfResult,
    PerturbationResult,
)
from fp_arena.experiment.retarget import (
    _transfer_direction,
    apply_precision,
    apply_reference,
    apply_target,
    fresh_sdfg,
    resolve_vectorize_config,
)
from fp_arena.experiment.store import ResultStore


def _copy_args(args: dict[str, Any]) -> dict[str, Any]:
    """Independent copy of the array arguments; scalars/symbols pass through."""
    return {
        k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
        for k, v in args.items()
    }


def _reset_arrays(working: dict[str, Any], source: dict[str, Any]) -> None:
    """Refill ``working``'s array buffers in place from ``source``."""
    for k, v in source.items():
        if isinstance(v, np.ndarray):
            working[k][...] = v


def _sample_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """
    ``n`` independent, reproducible per-sample generators from one base seed.
    """
    return [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(n)
    ]


def _new_acc() -> dict[str, float]:
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


def _accumulate(acc: dict[str, float], ref: np.ndarray, cand: np.ndarray) -> None:
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


def _finalize(acc: dict[str, float]) -> ErrorStats:
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
    samples: list[float], n_invocations: int, name: str
) -> list[float]:
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


def _phase_series(report, name_pred, n_reps: int, n_invocations: int) -> list[float]:
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
) -> dict[str, Any]:
    """Reduce the latest report into per-rep phase series; ``total`` is their sum."""
    report = sdfg.get_latest_report()
    # Map each state's report name -> phase category.
    cat_of: dict[str, str] = {}
    for state in sdfg.all_states():
        cat_of.setdefault(f"State {state.label}", _classify_state(sdfg, state))

    def series(category: str) -> list[float]:
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


def _output_arrays(sdfg: dace.SDFG) -> list[str]:
    """Non-transient written array names -- the arrays error metrics are reduced over."""
    _, writes = sdfg.read_and_write_sets()
    return sorted(
        name
        for name in writes
        if name in sdfg.arrays
        and isinstance(sdfg.arrays[name], dace.data.Array)
        and not sdfg.arrays[name].transient
    )


#: Length past which a pin tag becomes a hash.
_TAG_BUDGET = 120


def _pin_tag(pin_map: PrecisionMap) -> str:
    """Identifier-safe tag naming a precision point's build folder."""

    if not pin_map:
        return "baseline"
    tag = "_".join(f"{k}_{v}" for k, v in sorted(pin_map.items()))
    if len(tag) > _TAG_BUDGET:
        tag = f"pins_{hashlib.blake2b(tag.encode(), digest_size=6).hexdigest()}"
    return tag


def _distinguish(sdfg: dace.SDFG, tag: str) -> None:
    """
    Give the SDFG its own build folder for this build.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag)
    unique = uuid.uuid4().hex[:8]
    sdfg.name = f"{sdfg.name}_{safe}_{unique}"


def compile_reference(experiment, reference):
    """Build and compile the high-precision reference SDFG once."""
    sdfg = fresh_sdfg(experiment)
    apply_reference(sdfg, reference, experiment.promotion_rules)
    apply_target(
        sdfg,
        experiment.target,
        gpu_block_size=experiment.gpu_block_size,
        gpu_vectorize=experiment.gpu_vectorize,
        gpu_vectorize_config=experiment.gpu_vectorize_config,
    )
    _distinguish(sdfg, "reference")
    return sdfg.compile()


def build_candidate_sdfg(
    experiment, pin_map: PrecisionMap, instrument: bool = False
) -> dace.SDFG:
    """
    Build one precision point's SDFG: retargeted, given its own (unique) build
    folder, and -- when ``instrument`` is set -- with every state timed.

    Returns the SDFG (not the compiled handle) so the caller can both
    ``.compile()`` it and, after running, read its instrumentation report.
    """
    sdfg = fresh_sdfg(experiment)
    apply_precision(sdfg, pin_map, experiment.promotion_rules)
    apply_target(
        sdfg,
        experiment.target,
        gpu_block_size=experiment.gpu_block_size,
        gpu_vectorize=experiment.gpu_vectorize,
        gpu_vectorize_config=experiment.gpu_vectorize_config,
    )
    if instrument:
        provider = (
            dace.InstrumentationType.GPU_Events
            if experiment.target == "gpu"
            else dace.InstrumentationType.Timer
        )
        for state in sdfg.all_states():
            state.instrument = provider
    _distinguish(sdfg, _pin_tag(pin_map))
    return sdfg


def compile_candidate(experiment, pin_map: PrecisionMap):
    """Build and compile one precision point."""
    return build_candidate_sdfg(experiment, pin_map).compile()


def measure(
    sdfg: dace.SDFG,
    csdfg,
    experiment,
    n_warmup: int,
    n_reps: int,
    rng: np.random.Generator,
    noise: dict[str, Any] | None = None,
    prior_invocations: int = 0,
) -> dict[str, Any]:
    """
    Run ``n_warmup`` untimed then ``n_reps`` timed invocations of an already
    compiled, state-instrumented ``sdfg``, finalize it (which flushes the
    instrumentation report to disk), and reduce that report into the per-rep
    phase breakdown -- the timing kwargs of a :class:`PerfResult`.

    ``csdfg`` is finalized here, so the caller must not run or finalize it again.
    """
    n_invocations = prior_invocations + n_warmup + n_reps
    initial_args = make_call_args(sdfg, experiment, rng, noise)
    args = _copy_args(initial_args)
    for _ in range(n_warmup):
        _reset_arrays(args, initial_args)
        csdfg(**args)
    for _ in range(n_reps):
        _reset_arrays(args, initial_args)
        csdfg(**args)
    csdfg.finalize()
    return _phase_breakdown(sdfg, n_reps, n_invocations)


def run_performance(
    cfg: PerformanceAnalysisConfig, store: ResultStore | None = None
) -> list[PerfResult]:
    """Time each precision point, optionally appending to ``store``."""
    results: list[PerfResult] = []
    target = cfg.experiment.target
    vectorization = resolve_vectorize_config(
        target, cfg.experiment.gpu_vectorize, cfg.experiment.gpu_vectorize_config
    )
    prev_each = dace.Config.get("instrumentation", "report_each_invocation")
    dace.Config.set("instrumentation", "report_each_invocation", value=False)
    prev_streams = dace.Config.get("compiler", "cuda", "max_concurrent_streams")
    if target == "gpu":
        dace.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)

    points = tqdm(cfg.precisions, desc="performance", unit="pt")
    try:
        for pin_map in points:
            points.set_postfix_str(_fmt_pin(pin_map))
            # Time every state; each is classified into a phase at readout.
            sdfg = build_candidate_sdfg(cfg.experiment, pin_map, instrument=True)
            csdfg = sdfg.compile()
            rng = _sample_rngs(cfg.experiment.seed, 1)[0]
            breakdown = measure(
                sdfg, csdfg, cfg.experiment, cfg.n_warmup, cfg.n_reps, rng, cfg.noise
            )

            result = PerfResult(
                precision=dict(pin_map),
                seed=cfg.experiment.seed,
                **breakdown,
            )
            results.append(result)
            if store is not None:
                store.add(
                    cfg.experiment.name,
                    result,
                    symbols=cfg.experiment.symbols,
                    scalars=cfg.experiment.scalar_args,
                    vectorization=vectorization,
                )
    finally:
        dace.Config.set("instrumentation", "report_each_invocation", value=prev_each)
        dace.Config.set(
            "compiler", "cuda", "max_concurrent_streams", value=prev_streams
        )
    return results


def run_perturbation(
    cfg: PerturbationAnalysisConfig, store: ResultStore | None = None
) -> list[PerturbationResult]:
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
    results: list[PerturbationResult] = []
    vectorization = resolve_vectorize_config(
        exp.target, exp.gpu_vectorize, exp.gpu_vectorize_config
    )
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
            gpu_vectorize_config=exp.gpu_vectorize_config,
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
                    exp.name,
                    result,
                    symbols=exp.symbols,
                    scalars=exp.scalar_args,
                    vectorization=vectorization,
                )
    return results


def run_error(
    cfg: ErrorAnalysisConfig, store: ResultStore | None = None
) -> list[ErrorResult]:
    """
    Measure per-array error of each precision point, optionally appending to
    ``store`` database.

    Samples are the outer loop: every point is compiled up front, then each
    input realisation is executed by the reference and by every candidate in
    lockstep and folded into that point's accumulator. Only one sample is ever
    resident, so memory is flat in ``n_samples``.
    """
    exp = cfg.experiment
    results: list[ErrorResult] = []
    vectorization = resolve_vectorize_config(
        exp.target, exp.gpu_vectorize, exp.gpu_vectorize_config
    )
    reads, _ = exp.program.read_and_write_sets()
    outputs = _output_arrays(exp.program)

    ref_csdfg = compile_reference(exp, cfg.reference)
    candidates = [
        compile_candidate(exp, pin_map)
        for pin_map in tqdm(cfg.precisions, desc="compiling", unit="pt")
    ]
    # One accumulator per (precision point, written array); the samples fold in.
    accs = [{name: _new_acc() for name in outputs} for _ in cfg.precisions]

    for rng in tqdm(
        _sample_rngs(exp.seed, cfg.n_samples),
        desc="error",
        unit="smp",
        total=cfg.n_samples,
    ):
        args = make_call_args(exp.program, exp, rng, cfg.noise, reads=reads)
        ref_args = _copy_args(args)
        ref_csdfg(**ref_args)
        # Only the reference's written arrays are compared against; dropping the
        # rest keeps the read-only inputs from being held a third time over.
        ref_out = {name: ref_args[name] for name in outputs}
        del ref_args
        cand_args = _copy_args(args)
        points = tqdm(cfg.precisions, desc="points", unit="pt", leave=False)
        for i, pin_map in enumerate(points):
            points.set_postfix_str(_fmt_pin(pin_map))
            _reset_arrays(cand_args, args)
            candidates[i](**cand_args)
            for name in outputs:
                _accumulate(accs[i][name], ref_out[name], cand_args[name])

    if cfg.n_samples > 0:
        # Release each point's persistent (on GPU, device-resident) state.
        ref_csdfg.finalize()
        for csdfg in candidates:
            csdfg.finalize()

    for pin_map, acc in zip(cfg.precisions, accs):
        result = ErrorResult(
            precision=dict(pin_map),
            errors={name: _finalize(a) for name, a in acc.items()},
            n_samples=cfg.n_samples,
            seed=exp.seed,
        )
        results.append(result)
        if store is not None:
            store.add(
                exp.name,
                result,
                symbols=exp.symbols,
                scalars=exp.scalar_args,
                vectorization=vectorization,
            )
    return results
