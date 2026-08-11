# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
Tests for automatic FP-Arena enablement, which is installed by ``import
fp_arena`` and wraps ``SDFG.compile`` so SDFGs using FP-Arena types are enabled
without an explicit call.
"""

import dace
import numpy as np
import pytest
from dace.codegen.exceptions import CompilationError

import fp_arena


def _cast_sdfg_without_enable(n: int) -> dace.SDFG:
    """A float64 -> float32sr cast SDFG that does NOT call enable_fp_arena_extensions."""
    sdfg = dace.SDFG("auto_cast")
    sdfg.add_array("A", [n], dace.float64)
    sdfg.add_array("C", [n], fp_arena.float32sr)
    state = sdfg.add_state()
    a = state.add_read("A")
    c = state.add_write("C")
    me, mx = state.add_map("cast", {"i": f"0:{n}"})
    tasklet = state.add_tasklet(
        "cast",
        {"inp"},
        {"out"},
        "out = static_cast<fp_arena::float32sr>(inp);",
        dace.Language.CPP,
    )
    state.add_memlet_path(a, me, tasklet, dst_conn="inp", memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(tasklet, mx, c, src_conn="out", memlet=dace.Memlet("C[i]"))
    return sdfg


def test_auto_enable_runs_without_explicit_enable():
    n = 4096
    sdfg = _cast_sdfg_without_enable(n)
    A = np.full(n, 0.5, np.float64)  # exactly representable in float32
    C = np.zeros(n, np.float32)
    sdfg(A=A, C=C)  # auto-enable injects headers at compile time
    assert np.all(C == np.float32(0.5))


def test_headers_included_at_codegen_not_compile():
    # Attaching the environment is a codegen concern: invoking codegen directly
    # (no compile) must already contain the FP-Arena includes, by a path
    # relative to the environment's include root rather than an absolute one.
    sdfg = _cast_sdfg_without_enable(8)
    generated = "\n".join(co.code for co in sdfg.generate_code())
    assert '#include "fp_arena/float32sr.h"' in generated
    assert fp_arena.INCLUDE_DIR not in generated


def test_environment_attached_to_code_nodes():
    sdfg = _cast_sdfg_without_enable(8)
    assert fp_arena.required_environments(sdfg) == {"fp_arena.environments.FPArenaSR"}

    fp_arena.attach_environments(sdfg)
    tasklets = [
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)
    ]
    assert tasklets
    assert all("fp_arena.environments.FPArenaSR" in n.environments for n in tasklets)


def test_mpfr_environment_carries_the_link_flag():
    sdfg = dace.SDFG("mpfr_env")
    sdfg.add_array("A", [8], dace.mpfr(128))
    assert fp_arena.required_environments(sdfg) == {"fp_arena.environments.MPFR"}
    assert "mpfr" in fp_arena.MPFR.cmake_libraries


def test_uses_fp_arena_types_detection():
    assert fp_arena.uses_fp_arena_types(_cast_sdfg_without_enable(8)) is True

    plain = dace.SDFG("plain")
    plain.add_array("A", [8], dace.float64)
    assert fp_arena.uses_fp_arena_types(plain) is False


def test_disabling_auto_skips_injection():
    # With auto disabled and no explicit enable, the headers are absent and
    # compilation of an FP-Arena SDFG must fail (unknown type).
    fp_arena.disable_auto_extensions()
    try:
        sdfg = _cast_sdfg_without_enable(64)
        with pytest.raises(CompilationError):
            sdfg.compile()
    finally:
        fp_arena.enable_auto_extensions()

    # After re-enabling, the same kind of SDFG compiles and runs again.
    n = 256
    sdfg = _cast_sdfg_without_enable(n)
    A = np.full(n, 0.25, np.float64)
    C = np.zeros(n, np.float32)
    sdfg(A=A, C=C)
    assert np.all(C == np.float32(0.25))
