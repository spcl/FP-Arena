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
    apply_precision,
    apply_reference,
    apply_target,
    fresh_sdfg,
    resolve_vectorize_config,
)
from fp_arena.experiment.store import ResultStore
from fp_arena.experiment.timers import (
    insert_timers,
    phase_breakdown,
    read_timers,
    reset_timers,
)


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


def build_candidate_sdfg(experiment, pin_map: PrecisionMap) -> dace.SDFG:
    """
    Retarget one precision point into its own SDFG (unique build folder).

    Returns the SDFG, not the compiled handle, so the caller can insert timers
    (:func:`fp_arena.experiment.timers.insert_timers`) before ``.compile()``.
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
    _distinguish(sdfg, _pin_tag(pin_map))
    return sdfg


def compile_candidate(experiment, pin_map: PrecisionMap):
    """Build and compile one precision point."""
    return build_candidate_sdfg(experiment, pin_map).compile()


def measure(
    sdfg: dace.SDFG,
    csdfg,
    categories: list[str],
    experiment,
    n_warmup: int,
    n_reps: int,
    rng: np.random.Generator,
    noise: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run ``n_warmup`` untimed then ``n_reps`` timed invocations, then reduce the
    timer buffer into the per-rep :class:`PerfResult` timing kwargs.

    The buffer is reset up front, so earlier invocations of ``csdfg`` (e.g. the
    search's error grading) are discarded -- only the reps here are timed.
    ``csdfg`` is finalized here; the caller must not run or finalize it again.
    """
    reset_timers(csdfg)
    initial_args = make_call_args(sdfg, experiment, rng, noise)
    args = _copy_args(initial_args)
    for _ in range(n_warmup):
        _reset_arrays(args, initial_args)
        csdfg(**args)
    for _ in range(n_reps):
        _reset_arrays(args, initial_args)
        csdfg(**args)
    buf = read_timers(csdfg, len(categories))
    csdfg.finalize()
    return phase_breakdown(categories, buf, n_reps)


def run_performance(
    cfg: PerformanceAnalysisConfig, store: ResultStore | None = None
) -> list[PerfResult]:
    """Time each precision point, optionally appending to ``store``."""
    results: list[PerfResult] = []
    target = cfg.experiment.target
    vectorization = resolve_vectorize_config(
        target, cfg.experiment.gpu_vectorize, cfg.experiment.gpu_vectorize_config
    )
    # All work on the default stream, so the GPU timers' per-phase syncs isolate
    # real work rather than overlapping streams (see runtime/.../timers.h).
    prev_streams = dace.Config.get("compiler", "cuda", "max_concurrent_streams")
    if target == "gpu":
        dace.Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)

    points = tqdm(cfg.precisions, desc="performance", unit="pt")
    try:
        for pin_map in points:
            points.set_postfix_str(_fmt_pin(pin_map))
            sdfg = build_candidate_sdfg(cfg.experiment, pin_map)
            categories = insert_timers(sdfg, target)
            csdfg = sdfg.compile()
            rng = _sample_rngs(cfg.experiment.seed, 1)[0]
            breakdown = measure(
                sdfg,
                csdfg,
                categories,
                cfg.experiment,
                cfg.n_warmup,
                cfg.n_reps,
                rng,
                cfg.noise,
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
