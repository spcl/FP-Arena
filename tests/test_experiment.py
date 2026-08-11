# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.

import dataclasses

import dace
import numpy as np
import pytest
from scipy import stats

import fp_arena  # noqa: F401
from fp_arena.experiment import (
    CONSTANTS_KEY,
    METRICS,
    ConstraintResult,
    ErrorAnalysisConfig,
    ErrorBudget,
    ErrorStats,
    ExperimentConfig,
    Metric,
    Noise,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    PerturbationResult,
    ResultStore,
    SelectionAnalysisConfig,
    SelectionCandidate,
    SelectionResult,
    format_selection,
    precision_grid,
    registry,
    run_error,
    run_performance,
    run_perturbation,
    run_selection,
)
from fp_arena.experiment.inputs import make_call_args
from fp_arena.experiment.retarget import apply_target, fresh_sdfg
from fp_arena.experiment.runner import (
    _accumulate,
    _finalize,
    _group_per_invocation,
    _new_acc,
)
from fp_arena.experiment.selection import (
    check,
    noise_floor,
    resolve_limits,
)

N = dace.symbol("N")


@dace.program
def _axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i] + c[i]


_AXPY_SDFG = _axpy.to_sdfg(simplify=True)


def _exp(**kw):
    base = {
        "name": "axpy",
        "program": _AXPY_SDFG,
        "inputs": {
            "a": stats.uniform(0.5, 1.0),
            "b": stats.uniform(0.5, 1.0),
            "c": stats.uniform(0.5, 1.0),
        },
        "symbols": {"N": 64},
    }
    base.update(kw)
    return ExperimentConfig(**base)


def _has_gpu() -> bool:
    """Whether a runnable GPU device is present."""
    import logging
    import shutil
    import subprocess

    logger = logging.getLogger(__name__)

    for smi, args in (("nvidia-smi", ["-L"]), ("rocm-smi", ["--showid"])):
        if shutil.which(smi) is None:
            continue
        try:
            out = subprocess.run(
                [smi, *args], capture_output=True, text=True, timeout=15, check=False
            )
        except Exception:
            logger.exception("GPU detection failed for %s", smi)
            continue
        if out.returncode == 0 and out.stdout.strip():
            return True
    return False


requires_gpu = pytest.mark.skipif(not _has_gpu(), reason="no GPU device available")

#: Time-stepped axpy: the kernel runs T times per invocation (tests loop grouping).
T = dace.symbol("T")


@dace.program
def _axpy_loop(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for _ in range(T):
        for i in dace.map[0:N]:
            c[i] = a[i] * b[i] + c[i]


_AXPY_LOOP_SDFG = _axpy_loop.to_sdfg(simplify=True)


def test_precision_roundtrip():
    for key in ("fp16", "fp32", "fp64", "fp32sr", "fp64sr", "mpfr128"):
        assert registry.key_of(registry.to_typeclass(key)) == key


def test_store_roundtrip():
    db = ResultStore(":memory:")
    res = run_error(
        ErrorAnalysisConfig(_exp(), precisions=[{"a": "fp32"}], reference="fp64"),
        store=db,
    )
    rows = db.query(kind="error")
    assert len(rows) == 1
    assert rows[0].precision == {"a": "fp32"}
    assert rows[0].payload["errors"]["c"]["rel_mean"] == res[0].errors["c"].rel_mean


def test_error_decreases_with_precision():
    errs = run_error(
        ErrorAnalysisConfig(
            _exp(),
            precisions=[{"a": "fp16"}, {"a": "fp32"}],
            reference="fp64",
            n_samples=2,
        ),
    )
    e16, e32 = errs[0].errors["c"], errs[1].errors["c"]
    assert e32.rel_mean < e16.rel_mean
    assert e32.rel_mean < 1e-5
    assert e16.linf >= 0 and e16.rel_max >= 0


def test_error_zero_when_candidate_equals_reference():
    errs = run_error(
        ErrorAnalysisConfig(_exp(), precisions=[{"a": "fp64"}], reference="fp64")
    )
    assert errs[0].errors["c"].linf == 0.0


def test_performance_runs():
    perfs = run_performance(
        PerformanceAnalysisConfig(
            _exp(),
            precisions=[{"a": "fp32", "b": "fp32", "c": "fp32"}],
            n_warmup=1,
            n_reps=2,
            noise={"a": Noise(absolute=0.1, absolute_dist=stats.norm(0.0, 1.0))},
        ),
    )
    assert len(perfs) == 1
    assert len(perfs[0].total_times) == 2
    assert all(t > 0 for t in perfs[0].total_times)


def test_performance_phase_breakdown():
    perfs = run_performance(
        PerformanceAnalysisConfig(
            _exp(symbols={"N": 1 << 16}),
            precisions=[{"a": "fp32", "b": "fp32", "c": "fp32"}, {}],
            n_warmup=1,
            n_reps=3,
        ),
    )
    cast, baseline = perfs

    assert len(cast.total_times) == 3
    assert len(cast.cast_in_times) == 3
    assert len(cast.cast_out_times) == 3
    assert len(cast.kernel_times) == 3
    assert all(t > 0 for t in cast.cast_in_times)
    assert all(t >= 0 for t in cast.kernel_times)
    # No transfers on CPU.
    assert cast.h2d_times == []
    assert cast.d2h_times == []
    # total == sum of phases per rep.
    for i in range(3):
        expected = cast.cast_in_times[i] + cast.cast_out_times[i] + cast.kernel_times[i]
        assert cast.total_times[i] == pytest.approx(expected)

    # No cast: total is kernel.
    assert baseline.cast_in_times == []
    assert baseline.cast_out_times == []
    assert baseline.h2d_times == []
    assert baseline.d2h_times == []
    assert baseline.kernel_times == baseline.total_times
    assert all(t > 0 for t in baseline.total_times)


@requires_gpu
def test_performance_gpu_runs():
    perfs = run_performance(
        PerformanceAnalysisConfig(
            _exp(symbols={"N": 1 << 16}, target="gpu"),
            precisions=[{"a": "fp32", "b": "fp32", "c": "fp32"}],
            n_warmup=1,
            n_reps=3,
        ),
    )
    assert len(perfs) == 1
    assert len(perfs[0].total_times) == 3
    assert all(t > 0 for t in perfs[0].total_times)


@requires_gpu
def test_performance_gpu_phase_breakdown():
    perfs = run_performance(
        PerformanceAnalysisConfig(
            _exp(symbols={"N": 1 << 18}, target="gpu"),
            precisions=[{"a": "fp32", "b": "fp32", "c": "fp32"}, {}],
            n_warmup=1,
            n_reps=3,
        ),
    )
    cast, baseline = perfs

    # Every phase is populated; transfers/cast/kernel are strictly positive.
    for phase in (
        cast.h2d_times,
        cast.cast_in_times,
        cast.kernel_times,
        cast.cast_out_times,
        cast.d2h_times,
    ):
        assert len(phase) == 3
        assert all(t >= 0 for t in phase)
    assert all(t > 0 for t in cast.h2d_times)
    assert all(t > 0 for t in cast.d2h_times)
    assert all(t > 0 for t in cast.cast_in_times)
    assert all(t > 0 for t in cast.kernel_times)

    # total == sum of phases per rep.
    for i in range(3):
        expected = (
            cast.h2d_times[i]
            + cast.cast_in_times[i]
            + cast.kernel_times[i]
            + cast.cast_out_times[i]
            + cast.d2h_times[i]
        )
        assert cast.total_times[i] == pytest.approx(expected)

    # No cast: transfers + kernel only.
    assert baseline.cast_in_times == []
    assert baseline.cast_out_times == []
    assert all(t > 0 for t in baseline.h2d_times)
    assert all(t > 0 for t in baseline.d2h_times)
    assert all(t > 0 for t in baseline.kernel_times)
    for i in range(3):
        expected = (
            baseline.h2d_times[i] + baseline.kernel_times[i] + baseline.d2h_times[i]
        )
        assert baseline.total_times[i] == pytest.approx(expected)


@requires_gpu
def test_performance_gpu_loop_kernel_grouped_per_invocation():
    # Kernel fires T times per invocation; must collapse to one value per rep.
    perfs = run_performance(
        PerformanceAnalysisConfig(
            ExperimentConfig(
                name="axpy_loop",
                program=_AXPY_LOOP_SDFG,
                inputs={
                    "a": stats.uniform(0.5, 1.0),
                    "b": stats.uniform(0.5, 1.0),
                    "c": stats.uniform(0.5, 1.0),
                },
                symbols={"N": 1 << 14, "T": 8},
                target="gpu",
            ),
            precisions=[{"a": "fp32", "b": "fp32", "c": "fp32"}],
            n_warmup=1,
            n_reps=3,
        ),
    )
    (r,) = perfs
    assert len(r.kernel_times) == 3
    assert len(r.total_times) == 3
    assert all(t > 0 for t in r.kernel_times)


@requires_gpu
def test_error_gpu_zero_when_candidate_equals_reference():
    errs = run_error(
        ErrorAnalysisConfig(
            _exp(target="gpu"), precisions=[{"a": "fp64"}], reference="fp64"
        )
    )
    assert errs[0].errors["c"].linf == 0.0


def test_noise_half_specified_raises():
    # A term needs both its magnitude and its distribution; an inert Noise is an error.
    with pytest.raises(ValueError, match="half-specified"):
        Noise(absolute=1.0)
    with pytest.raises(ValueError, match="half-specified"):
        Noise(relative_dist=stats.norm(0.0, 1.0))
    with pytest.raises(ValueError, match="perturbs nothing"):
        Noise()


def test_perturbation_error_scales_with_noise():
    def run(mag):
        cfg = PerturbationAnalysisConfig(
            _exp(),
            noise={"a": Noise(relative=mag, relative_dist=stats.uniform(-1.0, 2.0))},
            n_samples=2,
        )
        return run_perturbation(cfg)[0].errors["c"]

    small, big = run(1e-6), run(1e-2)
    assert 0.0 < small.rel_mean < big.rel_mean
    # axpy: c = a*b + c with positive inputs, so a relative perturbation of `a`
    # bounded by mag moves c by at most mag relative.
    assert big.rel_max <= 1e-2 * (1.0 + 1e-9)


def test_perturbation_one_input_at_a_time():
    noise = Noise(relative=1e-3, relative_dist=stats.uniform(-1.0, 2.0))
    res = run_perturbation(
        PerturbationAnalysisConfig(_exp(), noise={"a": noise, "b": noise})
    )
    assert [r.perturbed for r in res] == ["a", "b"]
    assert all(r.precision == {} for r in res)
    assert all(r.errors["c"].linf > 0.0 for r in res)


def test_perturbation_store_roundtrip():
    db = ResultStore(":memory:")
    res = run_perturbation(
        PerturbationAnalysisConfig(
            _exp(),
            noise={"a": Noise(relative=1e-3, relative_dist=stats.norm(0.0, 1.0))},
            precisions=[{"a": "fp32"}],
        ),
        store=db,
    )
    rows = db.query(kind="perturbation")
    assert len(rows) == 1
    assert rows[0].precision == {"a": "fp32"}
    assert rows[0].payload["perturbed"] == "a"
    assert rows[0].payload["errors"]["c"]["rel_mean"] == res[0].errors["c"].rel_mean


def test_perturbation_unknown_input_raises():
    with pytest.raises(ValueError, match="not a read input"):
        run_perturbation(
            PerturbationAnalysisConfig(
                _exp(),
                noise={"z": Noise(absolute=1.0, absolute_dist=stats.norm(0.0, 1.0))},
            )
        )


def test_perturbation_empty_noise_raises():
    with pytest.raises(ValueError, match="at least one input"):
        run_perturbation(PerturbationAnalysisConfig(_exp(), noise={}))


def test_noise_perturbs_inputs():
    exp = _exp()
    sdfg = fresh_sdfg(exp)
    a_clean = make_call_args(sdfg, exp, np.random.default_rng(0))["a"]
    a_noisy = make_call_args(
        sdfg,
        exp,
        np.random.default_rng(0),
        {"a": Noise(absolute=10.0, absolute_dist=stats.norm(0.0, 1.0))},
    )["a"]
    assert not np.allclose(a_clean, a_noisy)


def test_noise_terms_take_distinct_distributions():
    exp = _exp()
    sdfg = fresh_sdfg(exp)
    a_clean = make_call_args(sdfg, exp, np.random.default_rng(0))["a"]
    noisy = make_call_args(
        sdfg,
        exp,
        np.random.default_rng(0),
        {"a": Noise(relative=0.1, relative_dist=stats.uniform(-1.0, 2.0))},
    )["a"]
    assert not np.allclose(a_clean, noisy)
    assert np.all(np.abs(noisy - a_clean) <= 0.1 * np.abs(a_clean) + 1e-6)


def test_error_noise_is_per_analysis():
    errs = run_error(
        ErrorAnalysisConfig(
            _exp(),
            precisions=[{"a": "fp32"}],
            reference="fp64",
            noise={"a": Noise(relative=1e-2, relative_dist=stats.norm(0.0, 1.0))},
        ),
    )
    s = errs[0].errors["c"]
    assert np.isfinite(s.rel_mean) and s.linf >= 0.0


def test_is_mpfr():
    assert registry.is_mpfr("mpfr128")
    assert not registry.is_mpfr("fp32")


def test_group_per_invocation():
    # k executions per invocation are summed into one value per invocation.
    assert _group_per_invocation([1.0, 2.0, 3.0, 4.0], 2, "t") == [3.0, 7.0]
    # One execution per invocation passes through.
    assert _group_per_invocation([1.0, 2.0], 2, "t") == [1.0, 2.0]
    assert _group_per_invocation([], 2, "t") == []
    # A data-dependent execution count cannot be attributed to reps: fail loudly.
    with pytest.raises(ValueError, match="static per-invocation"):
        _group_per_invocation([1.0] * 7, 11, "State s7")


def test_overflowing_reference_yields_inf_error_not_nan():
    acc = _new_acc()
    _accumulate(acc, np.array([np.inf, 2.0]), np.array([1.0, 2.0]))
    s = _finalize(acc)
    assert s.rel_max == np.inf
    assert not np.isnan(s.rel_mean)


def test_error_norms_match_hand_computed_values():
    # e = cand - ref = [0, -4]; r = [3, 4] -> ||r||_1 = 7, ||r||_2 = 5, max|r| = 4
    acc = _new_acc()
    _accumulate(acc, np.array([3.0, 4.0]), np.array([3.0, 0.0]))
    s = _finalize(acc)

    assert s.l1 == pytest.approx(4.0)
    assert s.l2 == pytest.approx(4.0)
    assert s.linf == pytest.approx(4.0)

    assert s.l1_norm == pytest.approx(4.0 / 7.0)
    assert s.l2_norm == pytest.approx(0.8)
    assert s.linf_norm == pytest.approx(1.0)

    assert s.snr == pytest.approx(10.0 * np.log10(25.0 / 16.0))
    # snr is the same quantity as the normalized L2 error, in decibels
    assert s.snr == pytest.approx(-20.0 * np.log10(s.l2_norm))


def test_error_norms_concatenate_across_accumulate_calls():
    # Folding two (ref, cand) pairs must equal one fold over their concatenation,
    # matching the "all elements over all samples" reduction semantics.
    ref1, cand1 = np.array([1.0, 2.0]), np.array([1.5, 2.0])
    ref2, cand2 = np.array([3.0, 4.0]), np.array([3.0, 5.0])

    split = _new_acc()
    _accumulate(split, ref1, cand1)
    _accumulate(split, ref2, cand2)
    whole = _new_acc()
    _accumulate(whole, np.concatenate([ref1, ref2]), np.concatenate([cand1, cand2]))

    a, b = _finalize(split), _finalize(whole)
    for f in ("l1", "l2", "linf", "l1_norm", "l2_norm", "linf_norm", "snr"):
        assert getattr(a, f) == pytest.approx(getattr(b, f))


def test_error_norms_zero_on_exact_match():
    acc = _new_acc()
    _accumulate(acc, np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    s = _finalize(acc)
    assert s.l1 == 0.0 and s.l2 == 0.0 and s.linf == 0.0
    assert s.l1_norm == 0.0 and s.l2_norm == 0.0 and s.linf_norm == 0.0
    assert s.snr == np.inf


def test_error_norms_infinite_when_reference_signal_is_zero():
    acc = _new_acc()
    _accumulate(acc, np.zeros(2), np.array([0.0, 1.0]))
    s = _finalize(acc)
    assert s.l1_norm == np.inf
    assert s.l2_norm == np.inf
    assert s.linf_norm == np.inf
    assert s.snr == -np.inf


def test_error_norms_non_finite_error_gives_minus_inf_snr():
    acc = _new_acc()
    _accumulate(acc, np.array([1.0, 2.0]), np.array([np.nan, 2.0]))
    s = _finalize(acc)
    assert s.l2 == np.inf
    assert s.snr == -np.inf


def test_error_metrics_persist_to_store():
    db = ResultStore(":memory:")
    run_error(
        ErrorAnalysisConfig(_exp(), precisions=[{"a": "fp16"}], reference="fp64"),
        store=db,
    )
    payload = db.query(kind="error")[0].payload["errors"]["c"]
    for f in ("l1", "l2", "linf", "l1_norm", "l2_norm", "linf_norm", "snr"):
        assert f in payload


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="Unknown target"):
        apply_target(fresh_sdfg(_exp()), "tpu")


def _stats(**kw) -> ErrorStats:
    """An ErrorStats with every metric zeroed except the ones named."""
    base = dict.fromkeys(
        (f.name for f in dataclasses.fields(ErrorStats)),
        0.0,
    )
    base.update(kw)
    return ErrorStats(**base)


def _noise(mag=1e-4):
    return {"a": Noise(relative=mag, relative_dist=stats.uniform(-1.0, 2.0))}


def _selection_cfg(**kw):
    base = {
        "experiment": _exp(symbols={"N": 1 << 12}),
        "precisions": precision_grid({"a": ["fp16", "fp64"], "b": ["fp16", "fp64"]}),
        "budget": ErrorBudget(from_perturbation={"rel_max": 1.0}),
        "noise": _noise(),
        "reference": "fp64",
        "n_samples": 1,
        "n_warmup": 1,
        "n_reps": 2,
    }
    base.update(kw)
    return SelectionAnalysisConfig(**base)


def test_precision_grid():
    # Full product, last array varying fastest.
    assert precision_grid({"A": ["fp16", "fp32"], "B": ["fp16", "fp32"]}) == [
        {"A": "fp16", "B": "fp16"},
        {"A": "fp16", "B": "fp32"},
        {"A": "fp32", "B": "fp16"},
        {"A": "fp32", "B": "fp32"},
    ]
    # Per-array choices, including the constants pseudo-array.
    assert precision_grid({"A": ["fp16", "fp32"], CONSTANTS_KEY: ["fp64"]}) == [
        {"A": "fp16", CONSTANTS_KEY: "fp64"},
        {"A": "fp32", CONSTANTS_KEY: "fp64"},
    ]
    with pytest.raises(ValueError, match="at least one array"):
        precision_grid({})
    with pytest.raises(ValueError, match="No precision keys"):
        precision_grid({"A": []})


def test_error_budget_validation():
    with pytest.raises(ValueError, match="constrains nothing"):
        ErrorBudget()
    with pytest.raises(ValueError, match="Unknown error metric"):
        ErrorBudget(limits={"nonsense": 1.0})
    with pytest.raises(ValueError, match="positive factor"):
        ErrorBudget(from_perturbation={"rel_max": 0.0})


def test_selection_config_validation():
    with pytest.raises(ValueError, match="at least one input"):
        _selection_cfg(noise={})
    with pytest.raises(ValueError, match="Unknown objective"):
        _selection_cfg(objective="wall_clock")


def test_metrics_registry_covers_every_error_stat():
    # A new ErrorStats field has to declare its direction and scale before a budget can name it.
    assert set(METRICS) == {f.name for f in dataclasses.fields(ErrorStats)}


def test_scale_treats_the_factor_as_an_error_multiple():
    # An error magnitude scales directly...
    assert METRICS["rel_max"].scale(1e-6, 2.0) == pytest.approx(2e-6)
    # ...but snr is a dB ratio, where "2x the error" is a 6.02 dB shift down.
    assert METRICS["snr"].scale(60.0, 2.0) == pytest.approx(60.0 - 20.0 * np.log10(2.0))
    assert METRICS["snr"].scale(60.0, 1.0) == pytest.approx(60.0)


def test_metric_direction_and_scale_are_independent():
    # The dB scaling belongs to snr, not to "higher is better" in general: a
    # linear higher-is-better metric keeps the proportional scale.
    linear_up = Metric(higher_is_better=True)
    assert linear_up.scale(1e-6, 2.0) == pytest.approx(2e-6)
    assert linear_up.better(1.0, 2.0) == 2.0
    assert linear_up.satisfies(2.0, 1.0)


def test_noise_floor_keeps_the_least_accurate_perturbation():
    results = [
        PerturbationResult(
            precision={},
            perturbed=name,
            errors={"c": _stats(rel_max=rel, snr=snr)},
            n_samples=1,
        )
        for name, rel, snr in (("a", 1e-3, 40.0), ("b", 1e-6, 90.0))
    ]
    floor = noise_floor(results, ["rel_max", "snr"])
    # Worst case: the largest error magnitude, but the *smallest* snr.
    assert floor["rel_max"]["c"] == 1e-3
    assert floor["snr"]["c"] == 40.0


def test_resolve_limits_tighter_constraint_binds():
    floor = {"rel_max": {"c": 1e-7}}
    # Derived limit is tighter than the explicit one.
    tight = resolve_limits(
        ErrorBudget(limits={"rel_max": 1e-5}, from_perturbation={"rel_max": 1.0}),
        floor,
        ["c"],
    )
    assert tight["rel_max"]["c"] == 1e-7
    # Explicit limit is tighter than the derived one.
    loose = resolve_limits(
        ErrorBudget(limits={"rel_max": 1e-9}, from_perturbation={"rel_max": 1.0}),
        floor,
        ["c"],
    )
    assert loose["rel_max"]["c"] == 1e-9


def test_resolve_limits_per_array_and_unknown_array():
    limits = resolve_limits(
        ErrorBudget(limits={"l2_norm": {"c": 1e-8}, "rel_max": 1e-5}),
        {},
        ["b", "c"],
    )
    assert limits["l2_norm"] == {"c": 1e-8}
    # A scalar limit fans out to every checked array.
    assert limits["rel_max"] == {"b": 1e-5, "c": 1e-5}
    with pytest.raises(ValueError, match="not a checked output array"):
        resolve_limits(ErrorBudget(limits={"rel_max": {"zz": 1.0}}), {}, ["c"])


def test_check_direction_depends_on_the_metric():
    errors = {"c": _stats(rel_max=1e-6, snr=50.0)}
    # rel_max: smaller is better.
    (passed,) = check(errors, {"rel_max": {"c": 1e-5}})
    assert passed.ok and passed.value == 1e-6 and passed.limit == 1e-5
    (failed,) = check(errors, {"rel_max": {"c": 1e-7}})
    assert not failed.ok
    # snr: larger is better, so the comparison flips.
    (passed,) = check(errors, {"snr": {"c": 40.0}})
    assert passed.ok
    (failed,) = check(errors, {"snr": {"c": 60.0}})
    assert not failed.ok


def test_check_treats_non_finite_error_as_infeasible():
    (c,) = check({"c": _stats(rel_max=np.inf)}, {"rel_max": {"c": 1e-3}})
    assert not c.ok


def test_selection_picks_a_feasible_point_and_records_every_candidate():
    db = ResultStore(":memory:")
    res = run_selection(_selection_cfg(), store=db)

    # The baseline is always evaluated on top of the grid, and always timed.
    assert (
        res.n_evaluated
        == len(precision_grid({"a": ["fp16", "fp64"], "b": ["fp16", "fp64"]})) + 1
    )
    assert len(res.candidates) == res.n_evaluated
    baseline = next(c for c in res.candidates if c.precision == {})
    assert baseline.objective_ms is not None
    assert baseline.speedup == pytest.approx(1.0)

    # The winner is feasible, timed, and no slower than any other feasible point.
    assert res.best is not None
    winner = next(c for c in res.candidates if c.precision == res.best)
    assert winner.feasible and winner.objective_ms is not None
    timed = [
        c.objective_ms
        for c in res.candidates
        if c.feasible and c.objective_ms is not None
    ]
    assert winner.objective_ms == min(timed)

    # Feasible points sort ahead of infeasible ones.
    flags = [c.feasible for c in res.candidates]
    assert flags == sorted(flags, reverse=True)

    # Every candidate carries a verdict per constraint, and feasibility follows it.
    for c in res.candidates:
        assert c.constraints
        assert c.feasible == all(k.ok for k in c.constraints)
    assert res.n_feasible == sum(c.feasible for c in res.candidates)

    # By default only the survivors are timed.
    assert all(c.objective_ms is None for c in res.candidates if not c.feasible)

    # The noise floor is measured at the baseline only.
    assert baseline.sensitivity is not None
    assert set(baseline.sensitivity) == {"a"}
    assert all(c.sensitivity is None for c in res.candidates if c.precision != {})
    assert res.limits["rel_max"]["c"] == pytest.approx(res.noise_floor["rel_max"]["c"])

    assert sorted({r.kind for r in db.query()}) == [
        "error",
        "performance",
        "perturbation",
        "selection",
    ]


def test_selection_store_roundtrip():
    db = ResultStore(":memory:")
    res = run_selection(_selection_cfg(), store=db)
    (row,) = db.query(kind="selection")
    # The store's precision column holds the winning assignment.
    assert row.precision == res.best
    assert row.payload["best"] == res.best
    assert len(row.payload["candidates"]) == res.n_evaluated
    assert row.payload["candidates"][0]["constraints"][0]["metric"] == "rel_max"


def test_selection_zero_budget_admits_exactly_the_bit_exact_points():
    # Error magnitudes are non-negative, so a limit of 0 is met only by points
    # that reproduce the fp64 reference exactly -- here, those that cast nothing.
    res = run_selection(_selection_cfg(budget=ErrorBudget(limits={"rel_max": 0.0})))
    assert res.best is not None
    for c in res.candidates:
        assert c.feasible == (c.errors["c"].rel_max == 0.0)


def test_selection_reports_no_winner_when_nothing_meets_the_budget():
    # No error can undercut a negative limit, so every point is rejected.
    res = run_selection(_selection_cfg(budget=ErrorBudget(limits={"rel_max": -1.0})))
    assert res.best is None
    assert res.n_feasible == 0
    # The full table still comes back, so you can see what you missed by.
    assert len(res.candidates) == res.n_evaluated
    assert all(not c.feasible for c in res.candidates)
    assert all(c.constraints[0].limit == -1.0 for c in res.candidates)


def test_selection_time_all_times_the_rejected_points_too():
    res = run_selection(
        _selection_cfg(budget=ErrorBudget(limits={"rel_max": -1.0}), time_all=True)
    )
    assert res.best is None  # still nothing feasible
    assert all(c.objective_ms is not None for c in res.candidates)
    assert all(c.speedup is not None for c in res.candidates)


def test_selection_perturb_all_measures_sensitivity_everywhere():
    res = run_selection(_selection_cfg(perturb_all=True))
    assert all(c.sensitivity is not None for c in res.candidates)
    assert all(set(c.sensitivity) == {"a"} for c in res.candidates)


def test_selection_budget_predicate_can_reject_everything():
    res = run_selection(
        _selection_cfg(
            budget=ErrorBudget(
                limits={"rel_max": np.inf}, predicate=lambda errors: False
            )
        )
    )
    assert res.best is None
    # The numeric constraints all passed; only the predicate rejected.
    assert all(all(k.ok for k in c.constraints) for c in res.candidates)
    assert all(not c.feasible for c in res.candidates)


def _candidate(precision, constraints, **kw):
    """A candidate from ``{(metric, array): (value, limit)}``; ok is derived."""
    return SelectionCandidate(
        precision=precision,
        errors={array: _stats() for _, array in constraints},
        constraints=[
            ConstraintResult(
                metric=metric,
                array=array,
                value=value,
                limit=limit,
                ok=METRICS[metric].satisfies(value, limit),
            )
            for (metric, array), (value, limit) in sorted(constraints.items())
        ],
        feasible=kw.pop("feasible"),
        **kw,
    )


def test_format_selection_adapts_to_whatever_the_search_varied():
    # Nothing heat3d-shaped here: other array names, a second metric, an
    # array that only some points pin, and a higher-is-better metric.
    winner = _candidate(
        {"u": "fp32", "flux": "fp16", CONSTANTS_KEY: "fp32"},
        {
            ("rel_max", "u_next"): (3e-7, 1e-6),
            ("rel_max", "flux"): (9e-6, 1e-5),
            ("snr", "u_next"): (91.0, 80.0),
        },
        feasible=True,
        objective_ms=1.25,
        speedup=3.2,
    )
    baseline = _candidate(
        {},
        {("rel_max", "u_next"): (0.0, 1e-6), ("snr", "u_next"): (99.0, 80.0)},
        feasible=True,
        objective_ms=4.0,
        speedup=1.0,
    )
    text = format_selection(
        SelectionResult(
            best=winner.precision,
            speedup_baseline={},
            objective="kernel",
            limits={"rel_max": {"u_next": 1e-6, "flux": 1e-5}, "snr": {"u_next": 80.0}},
            noise_floor={"rel_max": {"u_next": 1e-6}},
            candidates=[winner, baseline],
            n_evaluated=2,
            n_feasible=2,
        ),
        name="navier_stokes",
    )
    header, table = text.split("--- candidates")
    assert "navier_stokes" in header
    lines = table.splitlines()
    banner = next(line for line in lines if "precision" in line)
    columns = next(line for line in lines if line.split()[:1] == ["budget"]).split()
    # A column per pinned array and per (metric, graded array) -- none hardcoded.
    assert {"u", "flux", "consts"} <= set(columns)  # assigned
    assert {"u_next", "flux"} <= set(columns)  # graded
    assert {"kernel", "speedup"} <= set(columns)
    # "flux" is both assigned and graded, so only the banner tells its two
    # columns apart -- the metric names live there, not in the header row.
    assert columns.count("flux") == 2
    assert banner.index("precision") < banner.index("rel_max") < banner.index("snr")
    assert "rel_max" not in columns and "snr" not in columns
    assert "__constants__" not in text  # rendered under its readable label
    # Every graded array reports its own value rather than one collapsed worst.
    assert "3e-07" in text and "9e-06" in text
    # The baseline pins nothing, and unpinned reads as "left alone", not fp64.
    assert "fp64" not in text
    assert "3.20x" in text and "1.25 ms kernel" in text


def test_format_selection_marks_only_the_arrays_that_bust_the_budget():
    # One point, one metric, three graded arrays: the middle one is inside the
    # limit while its neighbours are not. A per-point verdict would hide that.
    point = _candidate(
        {"A": "fp32"},
        {
            ("rel_max", "A"): (0.804, 1e-6),
            ("rel_max", "Q"): (2.1e-7, 1e-6),
            ("rel_max", "R"): (1.913, 1e-6),
        },
        feasible=False,
        objective_ms=1.0,
        speedup=1.0,
    )
    text = format_selection(
        SelectionResult(
            best=None,
            speedup_baseline={},
            objective="kernel",
            limits={"rel_max": {"A": 1e-6, "Q": 1e-6, "R": 1e-6}},
            noise_floor={},
            candidates=[point],
            n_evaluated=1,
            n_feasible=0,
        )
    )
    row = next(line for line in text.splitlines() if "0.804" in line)
    assert "0.804!" in row and "1.913!" in row
    assert "2.1e-07" in row and "2.1e-07!" not in row
    assert "! over budget" in text


def test_format_selection_explains_a_search_with_no_winner():
    misses = [
        _candidate(
            {"a": key},
            {("rel_max", "c"): (value, 1e-9), ("snr", "c"): (snr, 120.0)},
            feasible=False,
        )
        for key, value, snr in (("fp16", 4e-3, 41.0), ("fp32", 7e-8, 88.0))
    ]
    text = format_selection(
        SelectionResult(
            best=None,
            speedup_baseline={"a": "fp64"},
            objective="total",
            limits={"rel_max": {"c": 1e-9}, "snr": {"c": 120.0}},
            noise_floor={},
            candidates=misses,
            n_evaluated=2,
            n_feasible=0,
        )
    )
    assert "Nothing met the budget" in text
    # Both constraints rejected everything; the closest miss is the value that
    # came nearest the limit -- the smallest error, but the *largest* snr.
    assert "2/2" in text
    assert "7e-08" in text and "88" in text
    # An unmeasured noise floor prints as absent rather than as a zero.
    floor_line = next(line for line in text.splitlines() if "1e-09" in line)
    assert floor_line.split()[-1] == "-"


def test_format_selection_handles_a_predicate_only_budget():
    # No numeric limits at all: no metric columns, and nothing to tabulate as
    # a rejection reason.
    text = format_selection(
        SelectionResult(
            best=None,
            speedup_baseline={},
            objective="total",
            limits={},
            noise_floor={},
            candidates=[
                SelectionCandidate(
                    precision={}, errors={}, constraints=[], feasible=False
                )
            ],
            n_evaluated=1,
            n_feasible=0,
        )
    )
    assert "predicate alone" in text
    assert "(unmodified)" in text
    assert "1 point evaluated" in text
    assert "predicate rejected" in text


def test_format_selection_truncates_a_large_search():
    candidates = [
        _candidate(
            {"a": key},
            {("rel_max", "c"): (1e-9, 1e-6)},
            feasible=True,
            objective_ms=ms,
            speedup=1.0,
        )
        for key, ms in (("fp16", 1.0), ("fp32", 2.0), ("fp64", 3.0))
    ]
    result = SelectionResult(
        best=candidates[0].precision,
        speedup_baseline={},
        objective="total",
        limits={"rel_max": {"c": 1e-6}},
        noise_floor={"rel_max": {"c": 1e-6}},
        candidates=candidates,
        n_evaluated=3,
        n_feasible=3,
    )
    assert "2 more" in format_selection(result, max_rows=1)
    assert "more in SelectionResult" not in format_selection(result, max_rows=None)


def test_run_selection_prints_the_summary(capsys):
    res = run_selection(_selection_cfg())
    printed = capsys.readouterr().out
    assert format_selection(res, name="axpy") in printed


def test_callable_init_function_covers_constant_and_fixed_data():
    exp = _exp(
        inputs={
            "a": lambda shape, rng: np.full(shape, 2.0),
            "b": stats.uniform(0.5, 1.0),
            "c": stats.uniform(0.5, 1.0),
        }
    )
    args = make_call_args(fresh_sdfg(exp), exp, np.random.default_rng(0))
    assert np.allclose(args["a"], 2.0)
