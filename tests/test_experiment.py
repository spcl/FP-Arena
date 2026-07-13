# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.

import dace
import numpy as np
import pytest
from scipy import stats
import fp_arena  # noqa: F401

from fp_arena.experiment import (
    ErrorAnalysisConfig,
    ExperimentConfig,
    Noise,
    PerformanceAnalysisConfig,
    PerturbationAnalysisConfig,
    ResultStore,
    registry,
    run_error,
    run_performance,
    run_perturbation,
)
from fp_arena.experiment.inputs import make_call_args
from fp_arena.experiment.retarget import apply_target, fresh_sdfg
from fp_arena.experiment.runner import (
    _accumulate,
    _finalize,
    _group_per_invocation,
    _new_acc,
)

N = dace.symbol("N")


@dace.program
def _axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i] + c[i]


_AXPY_SDFG = _axpy.to_sdfg(simplify=True)


def _exp(**kw):
    base = dict(
        name="axpy",
        program=_AXPY_SDFG,
        inputs={
            "a": stats.uniform(0.5, 1.0),
            "b": stats.uniform(0.5, 1.0),
            "c": stats.uniform(0.5, 1.0),
        },
        symbols={"N": 64},
    )
    base.update(kw)
    return ExperimentConfig(**base)


def _has_gpu() -> bool:
    """Whether a runnable GPU device is present."""
    import shutil
    import subprocess

    for smi, args in (("nvidia-smi", ["-L"]), ("rocm-smi", ["--showid"])):
        if shutil.which(smi) is None:
            continue
        try:
            out = subprocess.run(
                [smi, *args], capture_output=True, text=True, timeout=15
            )
        except Exception:
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


def test_needs_mpfr_link():
    assert registry.needs_mpfr_link("mpfr128")
    assert registry.needs_mpfr_link("fp23_46")  # elemental functions use MPFR
    assert not registry.needs_mpfr_link("fp32")


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
