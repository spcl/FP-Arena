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
    ResultStore,
    registry,
    run_error,
    run_performance,
)
from fp_arena.experiment.inputs import make_call_args
from fp_arena.experiment.retarget import apply_target, fresh_sdfg
from fp_arena.experiment.runner import _accumulate, _finalize, _new_acc

N = dace.symbol("N")


@dace.program
def _axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i] + c[i]


_AXPY_SDFG = _axpy.to_sdfg(simplify=True)

RULES = {
    frozenset({dace.float16, dace.float32}): dace.float32,
    frozenset({dace.float16, dace.float64}): dace.float64,
    frozenset({dace.float32, dace.float64}): dace.float64,
}


def _exp(**kw):
    base = dict(
        name="axpy",
        program=_AXPY_SDFG,
        inputs={
            "a": stats.uniform(0.5, 1.0),
            "b": stats.uniform(0.5, 1.0),
            "c": stats.uniform(0.5, 1.0),
        },
        promotion_rules=RULES,
        symbols={"N": 64},
    )
    base.update(kw)
    return ExperimentConfig(**base)


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
    assert e16.abs_max >= 0 and e16.rel_max >= 0


def test_error_zero_when_candidate_equals_reference():
    errs = run_error(
        ErrorAnalysisConfig(_exp(), precisions=[{"a": "fp64"}], reference="fp64")
    )
    assert errs[0].errors["c"].abs_max == 0.0


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
    assert len(perfs[0].times) == 2
    assert perfs[0].time_min > 0


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


def test_noise_without_distribution_is_noop():
    exp = _exp()
    sdfg = fresh_sdfg(exp)
    a_clean = make_call_args(sdfg, exp, np.random.default_rng(0))["a"]
    a_scaled = make_call_args(
        sdfg, exp, np.random.default_rng(0), {"a": Noise(absolute=10.0)}
    )["a"]
    assert np.array_equal(a_clean, a_scaled)


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
    assert np.isfinite(s.rel_mean) and s.abs_max >= 0.0


def test_is_mpfr():
    assert registry.is_mpfr("mpfr128")
    assert not registry.is_mpfr("fp32")


def test_overflowing_reference_yields_inf_error_not_nan():
    acc = _new_acc()
    _accumulate(acc, np.array([np.inf, 2.0]), np.array([1.0, 2.0]))
    s = _finalize(acc)
    assert s.rel_max == np.inf
    assert not np.isnan(s.rel_mean)


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
