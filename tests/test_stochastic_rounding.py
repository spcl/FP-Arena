# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Statistical correctness tests for the FP-Arena stochastic-rounding types,
exercised end-to-end through DaCe (build -> compile -> run).
"""

import numpy as np
import dace

import fp_arena


def _cast_sdfg(name: str, in_dtype, out_dtype, n: int) -> dace.SDFG:
    """Build an SDFG that casts every element of ``A`` (``in_dtype``) into ``C`` (``out_dtype``)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [n], in_dtype)
    sdfg.add_array("C", [n], out_dtype)
    state = sdfg.add_state()
    a = state.add_read("A")
    c = state.add_write("C")
    me, mx = state.add_map("cast", dict(i=f"0:{n}"))
    tasklet = state.add_tasklet("cast", {"inp"}, {"out"}, f"out = static_cast<{out_dtype.ctype}>(inp);",
                                dace.Language.CPP)
    state.add_memlet_path(a, me, tasklet, dst_conn="inp", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(tasklet, mx, c, src_conn="out", memlet=dace.Memlet("C[i]"))
    fp_arena.enable_fp_arena_extensions(sdfg)
    return sdfg


def _add_sdfg(name: str, dtype, n: int) -> dace.SDFG:
    """Build an SDFG that computes ``C = A + B`` element-wise in ``dtype``."""
    sdfg = dace.SDFG(name)
    for nm in ("A", "B", "C"):
        sdfg.add_array(nm, [n], dtype)
    state = sdfg.add_state()
    a = state.add_read("A")
    b = state.add_read("B")
    c = state.add_write("C")
    me, mx = state.add_map("add", dict(i=f"0:{n}"))
    tasklet = state.add_tasklet("add", {"x", "y"}, {"z"}, "z = x + y;", dace.Language.CPP)
    state.add_memlet_path(a, me, tasklet, dst_conn="x", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(b, me, tasklet, dst_conn="y", memlet=dace.Memlet("B[i]"))
    state.add_memlet_path(tasklet, mx, c, src_conn="z", memlet=dace.Memlet("C[i]"))
    fp_arena.enable_fp_arena_extensions(sdfg)
    return sdfg


def test_fp32sr_probability_proportional_to_distance():
    n = 2_000_000
    base = 2.0**24  # exactly representable in float32; the float32 ulp here is 2.0
    upper = base + 2.0
    frac = 0.3
    value = base + frac * 2.0  # exact double lying 30% of the way to ``upper``

    sdfg = _cast_sdfg("f32sr_cast", dace.float64, fp_arena.float32sr, n)
    A = np.full(n, value, np.float64)
    C = np.zeros(n, np.float32)
    sdfg(A=A, C=C)

    # Only the two neighbouring floats may appear.
    assert np.isin(C, np.array([base, upper], np.float32)).all()
    p_up = float(np.mean(C == np.float32(upper)))
    assert abs(p_up - frac) < 0.01, p_up


def test_fp32sr_exact_value_is_deterministic():
    n = 4096
    value = 0.5  # exactly representable in float32 -> no rounding
    sdfg = _cast_sdfg("f32sr_exact", dace.float64, fp_arena.float32sr, n)
    A = np.full(n, value, np.float64)
    C = np.zeros(n, np.float32)
    sdfg(A=A, C=C)
    assert np.all(C == np.float32(0.5))


def test_fp64sr_half_ulp_rounds_half_the_time():
    n = 2_000_000
    a = 1.0
    b = 2.0**-53  # exactly half a ulp above 1.0
    upper = 1.0 + 2.0**-52  # the next double after 1.0

    sdfg = _add_sdfg("f64sr_add", fp_arena.float64sr, n)
    A = np.full(n, a, np.float64)
    B = np.full(n, b, np.float64)
    C = np.zeros(n, np.float64)
    sdfg(A=A, B=B, C=C)

    assert np.isin(C, np.array([1.0, upper])).all()
    p_up = float(np.mean(C == upper))
    assert abs(p_up - 0.5) < 0.01, p_up


def test_fp64sr_quarter_ulp_rounds_a_quarter_of_the_time():
    # Exercises the integer rounding threshold at a non-trivial probability.
    n = 2_000_000
    a = 1.0
    b = 2.0**-54  # a quarter of a ulp above 1.0
    upper = 1.0 + 2.0**-52  # the next double after 1.0

    sdfg = _add_sdfg("f64sr_quarter", fp_arena.float64sr, n)
    A = np.full(n, a, np.float64)
    B = np.full(n, b, np.float64)
    C = np.zeros(n, np.float64)
    sdfg(A=A, B=B, C=C)

    assert np.isin(C, np.array([1.0, upper])).all()
    p_up = float(np.mean(C == upper))
    assert abs(p_up - 0.25) < 0.01, p_up


def test_fp64sr_exact_sum_is_deterministic():
    n = 4096
    sdfg = _add_sdfg("f64sr_exact", fp_arena.float64sr, n)
    A = np.full(n, 1.0, np.float64)
    B = np.full(n, 0.25, np.float64)  # 1.25 is exactly representable -> no rounding
    C = np.zeros(n, np.float64)
    sdfg(A=A, B=B, C=C)
    assert np.all(C == 1.25)
