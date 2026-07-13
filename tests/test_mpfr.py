import numpy as np
import dace
import fp_arena
from fp_arena.transformations.change_and_propagate_fp_types import (
    change_and_propagate_fp_types,
)


def test_sdfg_scalar_compute():
    """Compute 1/3 at 128-bit precision and return as double."""
    sdfg = dace.SDFG("mpfr_scalar")
    state = sdfg.add_state()

    sdfg.add_scalar("tmp", dace.mpfr(128), transient=True)
    sdfg.add_array("out", [1], dace.float64, transient=False)

    tmp_node = state.add_access("tmp")
    out_node = state.add_write("out")

    init = state.add_tasklet(
        "init_mpfr",
        {},
        {"t"},
        "t = 1.0 / 3.0;",
        language=dace.Language.CPP,
    )
    conv = state.add_tasklet(
        "to_double",
        {"t"},
        {"o"},
        "o = (double)t;",
        language=dace.Language.CPP,
    )

    state.add_edge(init, "t", tmp_node, None, dace.Memlet("tmp"))
    state.add_edge(tmp_node, None, conv, "t", dace.Memlet("tmp"))
    state.add_edge(conv, "o", out_node, None, dace.Memlet("out[0]"))

    csdfg = sdfg.compile()

    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert abs(out[0] - 1.0 / 3.0) < 1e-15, f"Expected ~0.333…, got {out[0]}"


def test_sdfg_array_sum():
    """Fill an mpfr array with values 1..N, sum into a double output."""
    N = 4
    sdfg = dace.SDFG("mpfr_array_sum")
    state = sdfg.add_state()

    sdfg.add_array("arr", [N], dace.mpfr(128), transient=True)
    sdfg.add_array("out", [1], dace.float64, transient=False)

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
        f"""dace::mpfr<128> acc(0); for (int i = 0; i < {N}; ++i) acc += a[i]; o = (double)acc;""",
        language=dace.Language.CPP,
    )

    state.add_edge(fill, "a", arr_node, None, dace.Memlet(f"arr[0:{N}]"))
    state.add_edge(arr_node, None, sumup, "a", dace.Memlet(f"arr[0:{N}]"))
    state.add_edge(sumup, "o", out_node, None, dace.Memlet("out[0]"))

    csdfg = sdfg.compile()

    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    expected = N * (N + 1) / 2  # 1+2+3+4 = 10
    assert abs(out[0] - expected) < 1e-15, f"Expected {expected}, got {out[0]}"


def test_sdfg_binary_ops():
    """Test each binary operator (+, -, *, /) independently."""
    cases = [
        ("add", "lhs + rhs", 7.0, 3.0, 10.0),
        ("sub", "lhs - rhs", 7.0, 3.0, 4.0),
        ("mul", "lhs * rhs", 7.0, 3.0, 21.0),
        ("div", "lhs / rhs", 9.0, 3.0, 3.0),
    ]
    for op_name, expr, lv, rv, expected in cases:
        sdfg = dace.SDFG(f"mpfr_{op_name}")
        state = sdfg.add_state()

        sdfg.add_array("out", [1], dace.float64, transient=False)
        out_node = state.add_write("out")

        tasklet = state.add_tasklet(
            op_name,
            {},
            {"o"},
            f"dace::mpfr<128> lhs({lv}), rhs({rv}), res = {expr}; o = (double)res;",
            language=dace.Language.CPP,
        )
        state.add_edge(tasklet, "o", out_node, None, dace.Memlet("out[0]"))

        csdfg = sdfg.compile()

        out = np.zeros(1, dtype=np.float64)
        csdfg(out=out)
        assert abs(out[0] - expected) < 1e-14, (
            f"operator{op_name}: expected {expected}, got {out[0]}"
        )


def test_sdfg_copy():
    """Copy one mpfr scalar into another and verify the value is preserved."""
    sdfg = dace.SDFG("mpfr_copy")
    state = sdfg.add_state()

    sdfg.add_scalar("src", dace.mpfr(128), transient=True)
    sdfg.add_scalar("dst", dace.mpfr(128), transient=True)
    sdfg.add_array("out", [1], dace.float64, transient=False)

    src_node = state.add_access("src")
    dst_node = state.add_access("dst")
    out_node = state.add_write("out")

    init = state.add_tasklet(
        "init", {}, {"s"}, "s = 1.0 / 3.0;", language=dace.Language.CPP
    )
    copy = state.add_tasklet("copy", {"s"}, {"d"}, "d = s;", language=dace.Language.CPP)
    conv = state.add_tasklet(
        "conv", {"d"}, {"o"}, "o = (double)d;", language=dace.Language.CPP
    )

    state.add_edge(init, "s", src_node, None, dace.Memlet("src"))
    state.add_edge(src_node, None, copy, "s", dace.Memlet("src"))
    state.add_edge(copy, "d", dst_node, None, dace.Memlet("dst"))
    state.add_edge(dst_node, None, conv, "d", dace.Memlet("dst"))
    state.add_edge(conv, "o", out_node, None, dace.Memlet("out[0]"))

    csdfg = sdfg.compile()

    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert abs(out[0] - 1.0 / 3.0) < 1e-15, f"Copy changed value: got {out[0]}"


_MPFR128 = dace.mpfr(128)
_MPFR_RULES = {frozenset({dace.float64, _MPFR128}): _MPFR128}


@dace.program
def _prog_double(x: dace.float64[1], y: dace.float64[1]):
    y[0] = x[0] * 2.0


@dace.program
def _prog_add_one(A: dace.float64[4], B: dace.float64[4]):
    for i in dace.map[0:4]:
        B[i] = A[i] + 1.0


@dace.program
def _prog_chain(A: dace.float64[1], tmp: dace.float64[1], B: dace.float64[1]):
    tmp[0] = A[0] + 0.5
    B[0] = tmp[0] * 2.0


def test_cap_elementwise():
    """change_and_propagate: float64 scalar doubled via mpfr(128) internally."""
    sdfg = _prog_double.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"x": _MPFR128}, _MPFR_RULES)

    # External interface stays float64; internal casted arrays are mpfr(128).
    assert sdfg.arrays["x"].dtype == dace.float64
    assert "fp_casted_x_mpfr128" in sdfg.arrays
    assert sdfg.arrays["fp_casted_x_mpfr128"].dtype == _MPFR128

    csdfg = sdfg.compile()
    x = np.array([1.5], dtype=np.float64)
    y = np.zeros(1, dtype=np.float64)
    csdfg(x=x, y=y)
    assert abs(y[0] - 3.0) < 1e-15, f"Expected 3.0, got {y[0]}"


def test_cap_array_map():
    """change_and_propagate: float64 array map promoted to mpfr(128)."""
    sdfg = _prog_add_one.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"A": _MPFR128}, _MPFR_RULES)

    assert sdfg.arrays["A"].dtype == dace.float64
    assert sdfg.arrays["fp_casted_A_mpfr128"].dtype == _MPFR128

    csdfg = sdfg.compile()
    A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    B = np.zeros(4, dtype=np.float64)
    csdfg(A=A, B=B)
    np.testing.assert_allclose(B, A + 1.0, atol=1e-15)


def test_cap_chain_transient():
    """change_and_propagate: all arrays promoted to mpfr(128); float64 interface preserved."""
    sdfg = _prog_chain.to_sdfg()
    change_and_propagate_fp_types(sdfg, {"A": _MPFR128}, _MPFR_RULES)

    # All three non-transients are changed, so each gets a cast wrapper; external type stays float64.
    for name in ("A", "tmp", "B"):
        assert sdfg.arrays[name].dtype == dace.float64
        assert sdfg.arrays[f"fp_casted_{name}_mpfr128"].dtype == _MPFR128

    csdfg = sdfg.compile()
    A = np.array([2.0], dtype=np.float64)
    tmp = np.zeros(1, dtype=np.float64)
    B = np.zeros(1, dtype=np.float64)
    csdfg(A=A, tmp=tmp, B=B)
    # (2.0 + 0.5) * 2.0 = 5.0
    assert abs(B[0] - 5.0) < 1e-15, f"Expected 5.0, got {B[0]}"


# ── exponent-bits tests ──────────────────────────────────────────────────────
#
# All tests use 5 exponent bits, the IEEE-754 convention (bias 2^(5-1)-1 = 15):
#   largest finite  = (2 - 2^-(prec-1)) * 2^15  (just under 2^16 = 65536)
#   smallest normal = 2^(1-15) = 2^-14
#   smallest subnormal (prec = 128) = 2^(2 - 15 - 128) = 2^-141
#
# So 32768 = 2^15 is representable, 65536 = 2^16 overflows to +inf, and
# values below half the smallest subnormal round to 0.


def _make_exp_bits_sdfg(name: str, body: str) -> dace.SDFG:
    """Build a single-tasklet SDFG that writes one double output."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array("out", [1], dace.float64)
    out_node = state.add_write("out")
    tasklet = state.add_tasklet("op", {}, {"o"}, body, language=dace.Language.CPP)
    state.add_edge(tasklet, "o", out_node, None, dace.Memlet("out[0]"))
    return sdfg


def test_exponent_bits_value_in_range():
    """With emax=15, 1.0 + 1.0 = 2.0 is within range and unchanged."""
    sdfg = _make_exp_bits_sdfg(
        "mpfr_exp_in_range",
        "dace::set_mpfr_exponent_bits(5);\n"
        "dace::mpfr<128> a(1.0), b(1.0);\n"
        "o = (double)(a + b);",
    )
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert abs(out[0] - 2.0) < 1e-15, f"Expected 2.0, got {out[0]}"


def test_exponent_bits_largest_binade_representable():
    """With 5 exponent bits, 32768 = 2^15 is a valid normal (IEEE emax=15)."""
    sdfg = _make_exp_bits_sdfg(
        "mpfr_exp_top_binade",
        "dace::set_mpfr_exponent_bits(5);\ndace::mpfr<128> a(32768.0);\no = (double)a;",
    )
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert out[0] == 32768.0, f"Expected 32768.0, got {out[0]}"


def test_exponent_bits_overflow_construction():
    """With 5 exponent bits, constructing 65536 = 2^16 overflows to +inf."""
    sdfg = _make_exp_bits_sdfg(
        "mpfr_exp_overflow_ctor",
        "dace::set_mpfr_exponent_bits(5);\n"
        "dace::mpfr<128> a(65536.0);\n"  # 2^16 > largest finite
        "o = (double)a;",
    )
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert np.isposinf(out[0]), f"Expected +inf, got {out[0]}"


def test_exponent_bits_overflow_arithmetic():
    """With 5 exponent bits, 32768.0 * 2.0 = 65536.0 overflows to +inf."""
    sdfg = _make_exp_bits_sdfg(
        "mpfr_exp_overflow_arith",
        "dace::set_mpfr_exponent_bits(5);\n"
        "dace::mpfr<128> a(32768.0), b(2.0);\n"
        "o = (double)(a * b);",
    )
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert np.isposinf(out[0]), f"Expected +inf, got {out[0]}"


def test_exponent_bits_matches_float16():
    """With 5 exponent bits, mpfr<11> is bit-compatible with numpy float16."""
    cases = [
        ("mul_ovf", "a * b", 1000.0, 70.0),  # overflows half
        ("mul", "a * b", 3.1415, 0.1),
        ("add", "a + b", 0.1, 1000.0),  # alignment + rounding
        ("div", "a / b", 1.0, 3.0),
        ("tiny", "a * b", 6.0e-8, 1.0),  # smallest subnormal 2^-24
        ("half_tiny", "a * b", 2.9802322387695312e-08, 1.0),  # 2^-25 ties to 0
    ]
    for name, expr, av, bv in cases:
        sdfg = _make_exp_bits_sdfg(
            f"mpfr_f16_{name}",
            "dace::set_mpfr_exponent_bits(5);\n"
            f"dace::mpfr<11> a({av!r}), b({bv!r});\n"
            f"o = (double)({expr});",
        )
        csdfg = sdfg.compile()
        out = np.zeros(1, dtype=np.float64)
        csdfg(out=out)
        fa, fb = np.float16(av), np.float16(bv)
        with np.errstate(over="ignore"):
            ref = float(eval(expr, {}, {"a": fa, "b": fb}))
        assert out[0] == ref, f"{name}: expected {ref}, got {out[0]}"


def test_exponent_bits_underflow_to_zero():
    """With emin=-14 and precision=128, values below the min subnormal flush to 0.

    Dividing 1.0 by 2 repeatedly: after n steps the MPFR exponent is -(n-1).
    At n=143 the exponent is -142 < emin-(precision-1)=-141, so it rounds to 0.
    """
    sdfg = _make_exp_bits_sdfg(
        "mpfr_exp_underflow",
        "dace::set_mpfr_exponent_bits(5);\n"
        "dace::mpfr<128> a(1.0), two(2.0);\n"
        "for (int i = 0; i < 200; ++i) a /= two;\n"
        "o = (double)a;",
    )
    csdfg = sdfg.compile()
    out = np.zeros(1, dtype=np.float64)
    csdfg(out=out)
    assert out[0] == 0.0, f"Expected 0.0, got {out[0]}"


if __name__ == "__main__":
    test_sdfg_scalar_compute()
    test_sdfg_array_sum()
    test_sdfg_binary_ops()
    test_sdfg_copy()
    test_cap_elementwise()
    test_cap_array_map()
    test_cap_chain_transient()
    test_exponent_bits_value_in_range()
    test_exponent_bits_largest_binade_representable()
    test_exponent_bits_overflow_construction()
    test_exponent_bits_overflow_arithmetic()
    test_exponent_bits_matches_float16()
    test_exponent_bits_underflow_to_zero()
    print("All SDFG tests passed.")
