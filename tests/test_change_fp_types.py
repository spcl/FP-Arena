# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Tests for the FP typecast transformation (:func:`fp_arena.change_fptype`)."""

import numpy as np
import dace

import fp_arena
from fp_arena import change_fptype


@dace.program
def _axpy(A: dace.float64[64], B: dace.float64[64], C: dace.float64[64]):
    for i in dace.map[0:64]:
        C[i] = A[i] + B[i]


@dace.program
def _indirect(A: dace.float64[10, 10], B: dace.float64[10, 10], C: dace.float64[10, 10], idx: dace.int64[10, 10],
              idy: dace.int64[10, 10]):
    for i, j in dace.map[0:10, 0:10]:
        C[i, j] = A[i, j] + B[idy[i, j], idx[i, j]]


def test_change_to_float32_compiles():
    sdfg = _axpy.to_sdfg()
    sdfg.validate()
    change_fptype(sdfg, dace.float64, dace.float32, cast_in_and_out_data=True)
    sdfg.validate()
    sdfg.compile()


def test_change_to_float32_nested_compiles():
    sdfg = _indirect.to_sdfg()
    sdfg.validate()
    change_fptype(sdfg, dace.float64, dace.float32, cast_in_and_out_data=True)
    sdfg.validate()
    sdfg.compile()


def test_retarget_to_float32sr_matches_reference():
    # Reference result in plain float64.
    rng = np.random.default_rng(0)
    A = rng.random(64)
    B = rng.random(64)
    ref = np.zeros(64)
    _axpy.f(A, B, ref)

    # Retarget the SDFG to the stochastic-rounding float32 type and run it.
    sdfg = _axpy.to_sdfg()
    change_fptype(sdfg, dace.float64, fp_arena.float32sr)
    sdfg.enable_fp_arena_extensions()

    A32 = A.astype(np.float32)
    B32 = B.astype(np.float32)
    C32 = np.zeros(64, np.float32)
    sdfg(A=A32, B=B32, C=C32)

    # Stochastic rounding is unbiased, so single-add results stay within a
    # couple of float32 ULPs of the float64 reference.
    assert np.allclose(C32, ref, atol=1e-6)
