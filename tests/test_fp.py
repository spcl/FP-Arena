"""SDFG-level tests for the emulated fp<Exp, Prec> type (dace.fp)."""

import numpy as np
import pytest

import dace
import fp_arena
from fp_arena.experiment import registry
from fp_arena.transformations.change_and_propagate_fp_types import (
    change_and_propagate_fp_types,
)

_native = fp_arena.native


def _single_tasklet_sdfg(name: str, body: str) -> dace.SDFG:
    """Build a single-tasklet SDFG that writes one double output."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array("out", [1], dace.float64)
    out_node = state.add_write("out")
    tasklet = state.add_tasklet("op", {}, {"o"}, body, language=dace.Language.CPP)
    state.add_edge(tasklet, "o", out_node, None, dace.Memlet("out[0]"))
    return sdfg


def _run_scalar(name: str, body: str) -> float:
    sdfg = _single_tasklet_sdfg(name, body)
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    return out[0]


def test_scalar_compute():
    """1/3 at fp<23,46> matches the native binding bit-for-bit."""
    got = _run_scalar(
        "fp_scalar",
        "fp_arena::fp<23, 46> a(1.0), b(3.0); o = (double)(a / b);",
    )
    expected = float(_native.fp23_46(1.0) / _native.fp23_46(3.0))
    assert got == expected, f"Expected {expected}, got {got}"


def test_binary_ops():
    """Each binary operator (+, -, *, /) against the native binding."""
    cases = [
        ("add", "lhs + rhs", 7.0, 0.3),
        ("sub", "lhs - rhs", 7.0, 0.3),
        ("mul", "lhs * rhs", 7.0, 0.3),
        ("div", "lhs / rhs", 9.0, 0.3),
    ]
    for op_name, expr, lv, rv in cases:
        got = _run_scalar(
            f"fp_{op_name}",
            f"fp_arena::fp<23, 46> lhs({lv}), rhs({rv}), res = {expr}; o = (double)res;",
        )
        l, r = _native.fp23_46(lv), _native.fp23_46(rv)
        expected = float(eval(expr, {}, {"lhs": l, "rhs": r}))
        assert got == expected, f"operator {op_name}: expected {expected}, got {got}"


def test_array_sum():
    """Fill an fp array with 1..N inside a tasklet and sum into a double."""
    N = 4
    sdfg = dace.SDFG("fp_array_sum")
    state = sdfg.add_state()
    sdfg.add_array("arr", [N], dace.fp(23, 46), transient=True)
    sdfg.add_array("out", [1], dace.float64)
    arr_node = state.add_access("arr")
    out_node = state.add_write("out")
    fill = state.add_tasklet(
        "fill",
        {},
        {"a"},
        "\n".join(f"a[{i}] = {i + 1};" for i in range(N)),
        language=dace.Language.CPP,
    )
    sumup = state.add_tasklet(
        "sum",
        {"a"},
        {"o"},
        f"fp_arena::fp<23, 46> acc(0); for (int i = 0; i < {N}; ++i) acc += a[i]; o = (double)acc;",
        language=dace.Language.CPP,
    )
    state.add_edge(fill, "a", arr_node, None, dace.Memlet(f"arr[0:{N}]"))
    state.add_edge(arr_node, None, sumup, "a", dace.Memlet(f"arr[0:{N}]"))
    state.add_edge(sumup, "o", out_node, None, dace.Memlet("out[0]"))
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert out[0] == 10.0, f"Expected 10.0, got {out[0]}"


def test_copy():
    """Copying preserves the value exactly (trivially copyable storage)."""
    got = _run_scalar(
        "fp_copy",
        "fp_arena::fp<23, 46> src(1.0), three(3.0); src /= three;\n"
        "fp_arena::fp<23, 46> dst = src; o = (double)dst;",
    )
    assert got == float(_native.fp23_46(1.0) / _native.fp23_46(3.0))


def test_overflow_to_inf():
    """fp<5,11> (IEEE half): 60000 * 2 overflows to +inf (emax=15)."""
    got = _run_scalar(
        "fp_overflow",
        "fp_arena::fp<5, 11> a(60000.0), b(2.0); o = (double)(a * b);",
    )
    assert np.isposinf(got), f"Expected +inf, got {got}"


def test_underflow_to_zero():
    """fp<5,11>: half of the smallest subnormal (2^-25) rounds to 0 (RNE)."""
    got = _run_scalar(
        "fp_underflow",
        "fp_arena::fp<5, 11> a(6.0e-8), b(0.5); o = (double)(a * b);",
    )
    assert got == 0.0, f"Expected 0.0, got {got}"


def test_subnormal_preserved():
    """fp<5,11>: the smallest subnormal (2^-24) is representable."""
    got = _run_scalar(
        "fp_subnormal",
        "fp_arena::fp<5, 11> a(5.9604644775390625e-8); o = (double)a;",
    )
    assert got == 2.0**-24, f"Expected 2^-24, got {got}"


def test_dace_math_dispatch():
    """dace::math::sin resolves to the fp overload (46-bit result)."""
    got = _run_scalar(
        "fp_math_sin",
        "fp_arena::fp<23, 46> x(1.0); o = (double)dace::math::sin(x);",
    )
    expected = float(_native.sin(_native.fp23_46(1.0)))
    assert got == expected
    assert got != np.sin(1.0)  # coarser than double: must differ from libm


def test_mpfr_cross_check():
    """fp<30,46> matches dace::mpfr<46> (unbounded exponent range) exactly."""
    body = (
        "dace::set_mpfr_exponent_bits(0);\n"
        "dace::mpfr<46> ma(1.0), mb(3.0), mc(7.0);\n"
        "fp_arena::fp<30, 46> fa(1.0), fb(3.0), fc(7.0);\n"
        "o = (double)(ma / mb * mc) - (double)(fa / fb * fc);"
    )
    got = _run_scalar("fp_vs_mpfr", body)
    assert got == 0.0, f"fp<30,46> and mpfr<46> disagree by {got}"


# -- change_and_propagate integration ---------------------------------------

_FP2346 = dace.fp(23, 46)
_FP824 = dace.fp(8, 24)
_FP1153 = dace.fp(11, 53)


def _rules(tc):
    return {frozenset({dace.float64, tc}): tc}


@dace.program
def _prog_double(x: dace.float64[1], y: dace.float64[1]):
    y[0] = x[0] * 2.0


@dace.program
def _prog_axpy(A: dace.float64[64], B: dace.float64[64], C: dace.float64[64]):
    for i in dace.map[0:64]:
        C[i] = A[i] * B[i] + 0.5


def test_cap_elementwise():
    """change_and_propagate: float64 doubled through fp23_46 internally."""
    sdfg = _prog_double.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"x": _FP2346}, _rules(_FP2346))
    assert sdfg.arrays["x"].dtype == dace.float64
    assert "fp_casted_x_fp23_46" in sdfg.arrays
    assert sdfg.arrays["fp_casted_x_fp23_46"].dtype == _FP2346
    csdfg = sdfg.compile()
    x = np.array([1.5], dtype=np.float64)
    y = np.zeros(1, dtype=np.float64)
    csdfg(x=x, y=y)
    assert y[0] == 3.0


def _random_inputs(rng, n):
    """Mixed magnitudes, including float32-subnormal and overflow ranges."""
    a = rng.uniform(-1.0, 1.0, n) * 10.0 ** rng.uniform(-42.0, 40.0, n)
    return a.astype(np.float64)


def test_cap_float32_bit_exact():
    """A map computed in fp<8,24> is bit-identical to numpy float32."""
    sdfg = _prog_axpy.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"A": _FP824, "B": _FP824}, _rules(_FP824))
    csdfg = sdfg.compile()
    rng = np.random.default_rng(7)
    A, B = _random_inputs(rng, 64), _random_inputs(rng, 64)
    C = np.zeros(64, dtype=np.float64)
    csdfg(A=A, B=B, C=C)
    with np.errstate(over="ignore"):  # overflow to inf is part of the test
        ref = (A.astype(np.float32) * B.astype(np.float32) + np.float32(0.5)).astype(
            np.float64
        )
    assert np.array_equal(C, ref), (C, ref)


def test_cap_float64_bit_exact():
    """A map computed in fp<11,53> is bit-identical to plain float64."""
    sdfg = _prog_axpy.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"A": _FP1153, "B": _FP1153}, _rules(_FP1153))
    csdfg = sdfg.compile()
    rng = np.random.default_rng(8)
    A, B = _random_inputs(rng, 64), _random_inputs(rng, 64)
    C = np.zeros(64, dtype=np.float64)
    csdfg(A=A, B=B, C=C)
    ref = A * B + 0.5
    assert np.array_equal(C, ref), (C, ref)


# -- fixed-point widths (exp_bits 0 and 1) -----------------------------------

_FP08 = dace.fp(0, 8)
_FP18 = dace.fp(1, 8)


def test_fixed_point_tasklet():
    """fp<0,8> arithmetic in generated code, on and past its finite range."""
    got = _run_scalar(
        "fp0_tasklet",
        "fp_arena::fp<0, 8> a(1.5), b(0.25); o = (double)(a - b);",
    )
    assert got == 1.25, f"Expected 1.25, got {got}"
    over = _run_scalar(
        "fp0_tasklet_overflow",
        "fp_arena::fp<0, 8> a(1.5), b(0.5); o = (double)(a + b);",
    )
    assert np.isposinf(over), f"Expected +inf, got {over}"


def test_fixed_point_matches_bindings():
    """Generated code and the native bindings agree bit-for-bit at Exp <= 1."""
    for exp_bits, cls in [(0, _native.fp0_8), (1, _native.fp1_8)]:
        got = _run_scalar(
            f"fp{exp_bits}_8_vs_binding",
            f"fp_arena::fp<{exp_bits}, 8> a(1.1), b(0.3); o = (double)(a * b);",
        )
        assert got == float(cls(1.1) * cls(0.3)), exp_bits


@dace.program
def _prog_scale(x: dace.float64[8], y: dace.float64[8]):
    for i in dace.map[0:8]:
        y[i] = x[i] * 0.5


def test_cap_fixed_point():
    """change_and_propagate through fp<0,8>: in-range values land on the grid,
    out-of-range ones overflow to inf instead of failing silently."""
    sdfg = _prog_scale.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"x": _FP08}, _rules(_FP08))
    assert sdfg.arrays["fp_casted_x_fp0_8"].dtype == _FP08
    csdfg = sdfg.compile()
    x = np.array([0.0, 0.5, 1.0, 1.5, -1.5, 0.25, 3.0, 100.0], dtype=np.float64)
    y = np.zeros(8, dtype=np.float64)
    csdfg(x=x, y=y)
    assert np.array_equal(y[:6], [0.0, 0.25, 0.5, 0.75, -0.75, 0.125]), y
    # 3.0 and 100.0 exceed the largest finite magnitude (1.953125) on the way
    # in, so they arrive as +inf and stay there.
    assert np.isposinf(y[6]) and np.isposinf(y[7]), y


def test_fixed_point_dtype_plumbing():
    assert _FP08.to_string() == "fp0_8"
    assert _FP08.ctype == "fp_arena::fp<0, 8>"
    assert _FP08.bytes == 1  # 0 + 8 bits
    assert _FP18.bytes == 2  # 1 + 8 bits
    assert dace.fp.from_json(_FP08.to_json()) == _FP08
    assert registry.to_typeclass("fp0_8") == _FP08
    assert registry.key_of(_FP08) == "fp0_8"
    assert registry.needs_mpfr_link("fp0_8")


def test_fixed_point_validation():
    """exp_bits=0 needs precision >= 3 to fit the reserved inf/NaN codes."""
    assert dace.fp(0, 3).bytes == 1  # smallest legal exp_bits=0 format
    assert dace.fp(1, 2).bytes == 1  # exp_bits=1 reserves a field, not a code
    with pytest.raises(ValueError, match="precision must be >= 3"):
        dace.fp(0, 2)
    with pytest.raises(ValueError, match=r"exp_bits must be in \[0, 30\]"):
        dace.fp(-1, 8)


# -- dtype plumbing ----------------------------------------------------------


def test_dtype_properties():
    tc = dace.fp(23, 46)
    assert tc.to_string() == "fp23_46"
    assert tc.ctype == "fp_arena::fp<23, 46>"
    assert tc.bytes == 9
    assert tc == dace.fp(23, 46)
    assert tc != dace.fp(8, 24)
    assert dace.fp23_46 == tc  # registered as an attribute on dace


def test_dtype_json_roundtrip():
    tc = dace.fp(23, 46)
    assert dace.fp.from_json(tc.to_json()) == tc

    sdfg = dace.SDFG("fp_json")
    sdfg.add_array("a", [4], tc, transient=True)
    restored = dace.SDFG.from_json(sdfg.to_json())
    assert restored.arrays["a"].dtype == tc


def test_registry_keys():
    tc = registry.to_typeclass("fp23_46")
    assert tc == dace.fp(23, 46)
    assert registry.key_of(tc) == "fp23_46"
    assert registry.needs_mpfr_link("fp23_46")
    assert not registry.needs_mpfr_link("fp32")


if __name__ == "__main__":
    test_scalar_compute()
    test_binary_ops()
    test_array_sum()
    test_copy()
    test_overflow_to_inf()
    test_underflow_to_zero()
    test_subnormal_preserved()
    test_dace_math_dispatch()
    test_mpfr_cross_check()
    test_cap_elementwise()
    test_cap_float32_bit_exact()
    test_cap_float64_bit_exact()
    test_fixed_point_tasklet()
    test_fixed_point_matches_bindings()
    test_cap_fixed_point()
    test_fixed_point_dtype_plumbing()
    test_fixed_point_validation()
    test_dtype_properties()
    test_dtype_json_roundtrip()
    test_registry_keys()
    print("All fp SDFG tests passed.")
