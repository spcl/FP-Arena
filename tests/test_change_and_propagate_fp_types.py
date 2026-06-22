import dace
import pytest
from dace.libraries.standard.nodes.reduce import Reduce
from fp_arena.transformations.change_and_propagate_fp_types import (
    DEFAULT_PROMOTION_RULES,
    change_and_propagate_fp_types,
)


def _tasklet_chain(n_states: int, transient_intermediates: bool = True):
    """
    Build an SDFG with a linear chain of states:
      A -> [T0 -> B0] -> [T1 -> B1] -> ... -> [T_{n-1} -> B_{n-1}]
    Returns (sdfg, 'A', 'B0', ..., 'B_{n-1}').
    """
    sdfg = dace.SDFG("chain")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    arr_names = ["A"]
    states = []
    prev_name = "A"
    for i in range(n_states):
        out_name = f"B{i}"
        sdfg.add_array(
            out_name,
            [1],
            dace.float32,
            transient=(transient_intermediates and i < n_states - 1),
        )
        arr_names.append(out_name)
        s = sdfg.add_state(f"s{i}")
        states.append(s)
        if i > 0:
            sdfg.add_edge(states[-2], s, dace.InterstateEdge())
        t = s.add_tasklet(f"t{i}", {"x"}, {"y"}, "y = x")
        s.add_edge(s.add_read(prev_name), None, t, "x", dace.Memlet(f"{prev_name}[0]"))
        s.add_edge(t, "y", s.add_write(out_name), None, dace.Memlet(f"{out_name}[0]"))
        prev_name = out_name
    return sdfg, arr_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_transient_intermediate_propagates():
    """Type propagates from a non-transient input through a transient intermediate to the output."""
    sdfg = dace.SDFG("test")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    sdfg.add_array("B", [1], dace.float32, transient=True)
    sdfg.add_array("C", [1], dace.float32, transient=False)

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    t1 = s1.add_tasklet("t1", {"a"}, {"b"}, "b = a")
    s1.add_edge(s1.add_read("A"), None, t1, "a", dace.Memlet("A[0]"))
    s1.add_edge(t1, "b", s1.add_write("B"), None, dace.Memlet("B[0]"))

    t2 = s2.add_tasklet("t2", {"b"}, {"c"}, "c = b")
    s2.add_edge(s2.add_read("B"), None, t2, "b", dace.Memlet("B[0]"))
    s2.add_edge(t2, "c", s2.add_write("C"), None, dace.Memlet("C[0]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    # Transient intermediate should be promoted to f16.
    assert sdfg.arrays["B"].dtype == dace.float16, sdfg.arrays["B"].dtype

    # Non-transient interface: external type preserved (f32), internal casted array exists.
    assert sdfg.arrays["A"].dtype == dace.float32
    assert sdfg.arrays["C"].dtype == dace.float32
    assert "fp_casted_A_float16" in sdfg.arrays
    assert "fp_casted_C_float16" in sdfg.arrays
    assert sdfg.arrays["fp_casted_A_float16"].dtype == dace.float16
    assert sdfg.arrays["fp_casted_C_float16"].dtype == dace.float16

    sdfg.validate()
    # sdfg.compile() TO


def test_all_nontransient_interface_preserved():
    """All non-transient: all arrays get cast wrappers, none change dtype externally."""
    sdfg = dace.SDFG("test_state_order")
    sdfg.add_array("A", [1], dace.float32)
    sdfg.add_array("B", [1], dace.float32)
    sdfg.add_array("C", [1], dace.float32)

    # States added in reverse alphabetical order to exercise topological sort.
    state_C = sdfg.add_state("state_C")
    state_B = sdfg.add_state("state_B")
    state_A = sdfg.add_state("state_A")
    sdfg.add_edge(state_A, state_B, dace.InterstateEdge())
    sdfg.add_edge(state_B, state_C, dace.InterstateEdge())

    ta = state_A.add_tasklet("ta", {"a"}, {"b"}, "b = a")
    state_A.add_edge(state_A.add_read("A"), None, ta, "a", dace.Memlet("A[0]"))
    state_A.add_edge(ta, "b", state_A.add_write("B"), None, dace.Memlet("B[0]"))

    tc = state_C.add_tasklet("tc", {"b"}, {"c"}, "c = b")
    state_C.add_edge(state_C.add_read("B"), None, tc, "b", dace.Memlet("B[0]"))
    state_C.add_edge(tc, "c", state_C.add_write("C"), None, dace.Memlet("C[0]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    # All are non-transient: external types preserved.
    assert sdfg.arrays["A"].dtype == dace.float32
    assert sdfg.arrays["B"].dtype == dace.float32
    assert sdfg.arrays["C"].dtype == dace.float32
    # Casted versions exist for the changed arrays.
    assert "fp_casted_A_float16" in sdfg.arrays
    assert "fp_casted_C_float16" in sdfg.arrays

    sdfg.validate()
    sdfg.compile()


def test_mixed_precision_promotes():
    """When a f16 and f32 array feed the same tasklet, the output is f32"""
    sdfg = dace.SDFG("mixed")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    sdfg.add_array("D", [1], dace.float32, transient=False)
    sdfg.add_array("E", [1], dace.float32, transient=True)

    s = sdfg.add_state("s")
    t = s.add_tasklet("t", {"a", "d"}, {"e"}, "e = a + d")
    s.add_edge(s.add_read("A"), None, t, "a", dace.Memlet("A[0]"))
    s.add_edge(s.add_read("D"), None, t, "d", dace.Memlet("D[0]"))
    s.add_edge(t, "e", s.add_write("E"), None, dace.Memlet("E[0]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    # D stays f32, A is demoted to f16; their mix promotes E to f32.
    assert sdfg.arrays["E"].dtype == dace.float32, sdfg.arrays["E"].dtype

    sdfg.validate()
    sdfg.compile()


def test_map_passthrough():
    """Type flows through MapEntry and MapExit connectors."""
    sdfg = dace.SDFG("map_test")
    sdfg.add_array("A", [4], dace.float32, transient=False)
    sdfg.add_array("B", [4], dace.float32, transient=True)

    s = sdfg.add_state("s")
    me, mx = s.add_map("m", {"i": "0:4"})
    t = s.add_tasklet("t", {"a"}, {"b"}, "b = a * 2.0;", language=dace.Language.CPP)

    a_an = s.add_read("A")
    b_an = s.add_write("B")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")

    s.add_edge(a_an, None, me, "IN_A", dace.Memlet("A[0:4]"))
    s.add_edge(me, "OUT_A", t, "a", dace.Memlet("A[i]"))
    s.add_edge(t, "b", mx, "IN_B", dace.Memlet("B[i]"))
    s.add_edge(mx, "OUT_B", b_an, None, dace.Memlet("B[0:4]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    assert sdfg.arrays["B"].dtype == dace.float16, sdfg.arrays["B"].dtype

    sdfg.validate()
    sdfg.compile()


def test_reduce_node():
    """Type propagates through a Reduce library node."""
    sdfg = dace.SDFG("reduce_test")
    sdfg.add_array("A", [4], dace.float32, transient=False)
    sdfg.add_scalar("S", dace.float32, transient=True)

    s = sdfg.add_state("s")
    reduce_node = Reduce("sum", "lambda a, b: a + b", axes=[0], identity=0)
    s.add_node(reduce_node)
    s.add_edge(s.add_read("A"), None, reduce_node, None, dace.Memlet("A[0:4]"))
    s.add_edge(reduce_node, None, s.add_write("S"), None, dace.Memlet("S"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    assert sdfg.arrays["S"].dtype == dace.float16, sdfg.arrays["S"].dtype

    sdfg.validate()
    sdfg.compile()


def test_initial_type_pinned():
    """An array listed in initial_types keeps that type even if higher-precision data flows into it."""
    sdfg = dace.SDFG("pinned")
    sdfg.add_array("A", [1], dace.float64, transient=False)
    sdfg.add_array("B", [1], dace.float32, transient=True)

    s = sdfg.add_state("s")
    t = s.add_tasklet("t", {"a"}, {"b"}, "b = a")
    s.add_edge(s.add_read("A"), None, t, "a", dace.Memlet("A[0]"))
    s.add_edge(t, "b", s.add_write("B"), None, dace.Memlet("B[0]"))

    # Even though A is f64, B must stay f16.
    change_and_propagate_fp_types(sdfg, {"A": dace.float64, "B": dace.float16})

    assert sdfg.arrays["B"].dtype == dace.float16, sdfg.arrays["B"].dtype

    sdfg.validate()
    sdfg.compile()


def test_long_chain_convergence():
    """Fixpoint converges for a longer state chain without hitting the iteration cap."""
    sdfg, arr_names = _tasklet_chain(n_states=5, transient_intermediates=True)

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    for name in arr_names[1:-1]:  # B0..B3 are transient intermediates
        assert sdfg.arrays[name].dtype == dace.float16, (
            f"{name}: {sdfg.arrays[name].dtype}"
        )

    sdfg.validate()
    sdfg.compile()


def test_unconnected_array_unchanged():
    """Arrays with no data-flow path from initial_types are not modified."""
    sdfg = dace.SDFG("unconnected")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    sdfg.add_array("B", [1], dace.float32, transient=True)
    sdfg.add_array("X", [1], dace.float64, transient=True)  # not connected to A

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    t1 = s1.add_tasklet("t1", {"a"}, {"b"}, "b = a")
    s1.add_edge(s1.add_read("A"), None, t1, "a", dace.Memlet("A[0]"))
    s1.add_edge(t1, "b", s1.add_write("B"), None, dace.Memlet("B[0]"))

    # s2 works on X independently.
    t2 = s2.add_tasklet("t2", {"x"}, {"y"}, "y = x")
    s2.add_edge(s2.add_read("X"), None, t2, "x", dace.Memlet("X[0]"))
    s2.add_edge(t2, "y", s2.add_write("X"), None, dace.Memlet("X[0]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    assert sdfg.arrays["B"].dtype == dace.float16
    assert sdfg.arrays["X"].dtype == dace.float64  # unchanged

    sdfg.validate()
    sdfg.compile()


def test_interface_copy_in_only_for_inputs():
    """Arrays that are only written (output-only) do not get a copy-in state."""
    sdfg = dace.SDFG("output_only")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    sdfg.add_array("B", [1], dace.float32, transient=False)

    s = sdfg.add_state("s")
    t = s.add_tasklet("t", {}, {"b"}, "b = 1.0;", language=dace.Language.CPP)
    # B is purely written (no read from outside), A is not used.
    s.add_edge(t, "b", s.add_write("B"), None, dace.Memlet("B[0]"))

    change_and_propagate_fp_types(sdfg, {"B": dace.float16})

    # B is non-transient and changed: external B stays f32, casted version is f16.
    assert sdfg.arrays["B"].dtype == dace.float32
    casted_name = "fp_casted_B_float16"
    assert casted_name in sdfg.arrays

    # B is output-only and A is unused, so nothing needs casting on the way in:
    # the copy_in state is created lazily and should not exist at all here.
    copy_in = next((st for st in sdfg.states() if st.label == "copy_in"), None)
    assert copy_in is None, "Output-only arrays should not produce a copy_in state"

    sdfg.validate()
    sdfg.compile()


def test_requires_two_fixpoint_passes():
    """
    Verify that the fixpoint loop runs at least two passes. In the first pass, B is promoted to f32 due to the write from E; in the second pass, C is promoted to f32 due to the read from B.
    """
    sdfg = dace.SDFG("two_pass_required")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    sdfg.add_array("E", [1], dace.float32, transient=False)
    sdfg.add_array("B", [1], dace.float32, transient=True)
    sdfg.add_array("C", [1], dace.float32, transient=True)

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    s3 = sdfg.add_state("s3")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())
    sdfg.add_edge(s2, s3, dace.InterstateEdge())

    # S1: A -> B
    t1 = s1.add_tasklet("t1", {"a"}, {"b"}, "b = a")
    s1.add_edge(s1.add_read("A"), None, t1, "a", dace.Memlet("A[0]"))
    s1.add_edge(t1, "b", s1.add_write("B"), None, dace.Memlet("B[0]"))

    # S2: B -> C  (visited before S3 promotes B to f32)
    t2 = s2.add_tasklet("t2", {"b"}, {"c"}, "c = b")
    s2.add_edge(s2.add_read("B"), None, t2, "b", dace.Memlet("B[0]"))
    s2.add_edge(t2, "c", s2.add_write("C"), None, dace.Memlet("C[0]"))

    # S3: E(f32) -> B  (second write; promotes B from f16 to f32)
    t3 = s3.add_tasklet("t3", {"e"}, {"b"}, "b = e")
    s3.add_edge(s3.add_read("E"), None, t3, "e", dace.Memlet("E[0]"))
    s3.add_edge(t3, "b", s3.add_write("B"), None, dace.Memlet("B[0]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    # B is written by both S1(f16) and S3(f32) -> promoted to f32.
    assert sdfg.arrays["B"].dtype == dace.float32, (
        f"B should be f32 (promoted), got {sdfg.arrays['B'].dtype}"
    )
    # C reads from B; only correct after the second pass propagates B=f32 into S2.
    assert sdfg.arrays["C"].dtype == dace.float32, (
        f"C should be f32 (requires two passes), got {sdfg.arrays['C'].dtype} — "
        "this failure means the fixpoint loop ran only once"
    )

    sdfg.validate()
    sdfg.compile()


def test_three_level_lattice():
    """With three precision levels, each array takes the widest (join) of its producers."""
    sdfg = dace.SDFG("three_level")
    sdfg.add_array("A", [1], dace.float32, transient=False)  # pinned f16
    sdfg.add_array("B", [1], dace.float32, transient=False)  # source f32
    sdfg.add_array("C", [1], dace.float64, transient=False)  # source f64
    sdfg.add_array("AB", [1], dace.float32, transient=True)  # join(f16, f32) = f32
    sdfg.add_array("AC", [1], dace.float32, transient=True)  # join(f16, f64) = f64
    sdfg.add_array("ABC", [1], dace.float32, transient=True)  # join(f32, f64) = f64

    s = sdfg.add_state("s")
    ab = s.add_access("AB")

    t1 = s.add_tasklet("t1", {"a", "b"}, {"o"}, "o = a + b")
    s.add_edge(s.add_read("A"), None, t1, "a", dace.Memlet("A[0]"))
    s.add_edge(s.add_read("B"), None, t1, "b", dace.Memlet("B[0]"))
    s.add_edge(t1, "o", ab, None, dace.Memlet("AB[0]"))

    t2 = s.add_tasklet("t2", {"a", "c"}, {"o"}, "o = a + c")
    s.add_edge(s.add_read("A"), None, t2, "a", dace.Memlet("A[0]"))
    s.add_edge(s.add_read("C"), None, t2, "c", dace.Memlet("C[0]"))
    s.add_edge(t2, "o", s.add_write("AC"), None, dace.Memlet("AC[0]"))

    t3 = s.add_tasklet("t3", {"ab", "c"}, {"o"}, "o = ab + c")
    s.add_edge(ab, None, t3, "ab", dace.Memlet("AB[0]"))
    s.add_edge(s.add_read("C"), None, t3, "c", dace.Memlet("C[0]"))
    s.add_edge(t3, "o", s.add_write("ABC"), None, dace.Memlet("ABC[0]"))

    change_and_propagate_fp_types(sdfg, {"A": dace.float16})

    assert sdfg.arrays["AB"].dtype == dace.float32, sdfg.arrays["AB"].dtype
    assert sdfg.arrays["AC"].dtype == dace.float64, sdfg.arrays["AC"].dtype
    assert sdfg.arrays["ABC"].dtype == dace.float64, sdfg.arrays["ABC"].dtype

    sdfg.validate()
    sdfg.compile()


def test_cyclic_dependency_terminates():
    """A dependency cycle that no seed reaches keeps original precision and terminates."""
    sdfg = dace.SDFG("cycle")
    sdfg.add_array("P", [1], dace.float64, transient=True)
    sdfg.add_array("Q", [1], dace.float64, transient=True)

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    t1 = s1.add_tasklet("t1", {"q"}, {"p"}, "p = q")
    s1.add_edge(s1.add_read("Q"), None, t1, "q", dace.Memlet("Q[0]"))
    s1.add_edge(t1, "p", s1.add_write("P"), None, dace.Memlet("P[0]"))

    t2 = s2.add_tasklet("t2", {"p"}, {"q"}, "q = p")
    s2.add_edge(s2.add_read("P"), None, t2, "p", dace.Memlet("P[0]"))
    s2.add_edge(t2, "q", s2.add_write("Q"), None, dace.Memlet("Q[0]"))

    change_and_propagate_fp_types(sdfg, {})

    assert sdfg.arrays["P"].dtype == dace.float64, sdfg.arrays["P"].dtype
    assert sdfg.arrays["Q"].dtype == dace.float64, sdfg.arrays["Q"].dtype

    sdfg.validate()
    sdfg.compile()


def test_end_to_end_float32_runs():
    """Full pipeline: transform (f64->f32), compile, run, and check numerics."""
    import numpy as np

    n = 8
    sdfg = dace.SDFG("e2e_f32")
    sdfg.add_array("A", [n], dace.float64, transient=False)
    sdfg.add_array("B", [n], dace.float64, transient=True)
    sdfg.add_array("C", [n], dace.float64, transient=False)

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    for st, src, dst in [(s1, "A", "B"), (s2, "B", "C")]:
        me, mx = st.add_map("m", {"i": f"0:{n}"})
        t = st.add_tasklet(
            "t", {"x"}, {"y"}, "y = x * 2.0;", language=dace.Language.CPP
        )
        me.add_in_connector(f"IN_{src}")
        me.add_out_connector(f"OUT_{src}")
        mx.add_in_connector(f"IN_{dst}")
        mx.add_out_connector(f"OUT_{dst}")
        st.add_edge(
            st.add_read(src), None, me, f"IN_{src}", dace.Memlet(f"{src}[0:{n}]")
        )
        st.add_edge(me, f"OUT_{src}", t, "x", dace.Memlet(f"{src}[i]"))
        st.add_edge(t, "y", mx, f"IN_{dst}", dace.Memlet(f"{dst}[i]"))
        st.add_edge(
            mx, f"OUT_{dst}", st.add_write(dst), None, dace.Memlet(f"{dst}[0:{n}]")
        )

    change_and_propagate_fp_types(sdfg, {"A": dace.float32})
    sdfg.validate()

    assert sdfg.arrays["B"].dtype == dace.float32, sdfg.arrays["B"].dtype
    assert sdfg.arrays["A"].dtype == dace.float64
    assert sdfg.arrays["C"].dtype == dace.float64

    A = np.arange(n, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)
    sdfg(A=A, C=C)
    np.testing.assert_allclose(C, A * 4.0)


def test_default_rules_used_and_not_merged():
    """promotion_rules=None falls back to DEFAULT_PROMOTION_RULES; an explicit
    dict is used as-is, so omitting a needed pair still raises."""

    def _build():
        sdfg = dace.SDFG("default_rules")
        sdfg.add_array("A", [1], dace.float64, transient=False)  # pinned f16
        sdfg.add_array("B", [1], dace.float64, transient=False)  # source f64
        sdfg.add_array("C", [1], dace.float64, transient=True)  # join(f16, f64)

        s = sdfg.add_state("s")
        t = s.add_tasklet("t", {"a", "b"}, {"c"}, "c = a + b")
        s.add_edge(s.add_read("A"), None, t, "a", dace.Memlet("A[0]"))
        s.add_edge(s.add_read("B"), None, t, "b", dace.Memlet("B[0]"))
        s.add_edge(t, "c", s.add_write("C"), None, dace.Memlet("C[0]"))
        return sdfg

    # No rules -> defaults apply -> join(f16, f64) = f64.
    sdfg = _build()
    change_and_propagate_fp_types(sdfg, {"A": dace.float16})
    assert sdfg.arrays["C"].dtype == dace.float64, sdfg.arrays["C"].dtype
    assert frozenset({dace.float16, dace.float64}) in DEFAULT_PROMOTION_RULES
    sdfg.validate()

    # An explicit dict missing {f16, f64}
    sdfg = _build()
    incomplete = {frozenset({dace.float16, dace.float32}): dace.float32}
    with pytest.raises(ValueError):
        change_and_propagate_fp_types(sdfg, {"A": dace.float16}, incomplete)


def test_end_to_end_demoted_written_array_runs():
    """A *written* array is demoted below the precision of the f64 computation that produces it.

    Chain A(f64) -> B -> C with B pinned to f32. The producer of B reads A (f64),
    so its tasklet computes in f64 but must store into the f32 array B.
    """
    import numpy as np

    n = 8
    sdfg = dace.SDFG("e2e_demoted_write")
    sdfg.add_array("A", [n], dace.float64, transient=False)
    sdfg.add_array("B", [n], dace.float64, transient=True)
    sdfg.add_array("C", [n], dace.float64, transient=False)

    s1 = sdfg.add_state("s1")
    s2 = sdfg.add_state("s2")
    sdfg.add_edge(s1, s2, dace.InterstateEdge())

    for st, src, dst in [(s1, "A", "B"), (s2, "B", "C")]:
        me, mx = st.add_map("m", {"i": f"0:{n}"})
        t = st.add_tasklet(
            "t", {"x"}, {"y"}, "y = x * 2.0;", language=dace.Language.CPP
        )
        me.add_in_connector(f"IN_{src}")
        me.add_out_connector(f"OUT_{src}")
        mx.add_in_connector(f"IN_{dst}")
        mx.add_out_connector(f"OUT_{dst}")
        st.add_edge(
            st.add_read(src), None, me, f"IN_{src}", dace.Memlet(f"{src}[0:{n}]")
        )
        st.add_edge(me, f"OUT_{src}", t, "x", dace.Memlet(f"{src}[i]"))
        st.add_edge(t, "y", mx, f"IN_{dst}", dace.Memlet(f"{dst}[i]"))
        st.add_edge(
            mx, f"OUT_{dst}", st.add_write(dst), None, dace.Memlet(f"{dst}[0:{n}]")
        )

    # Demote only the WRITTEN intermediate B; A stays f64, so the producer of B
    # computes in f64 and must store into an f32 array (the heat3d crash pattern).
    change_and_propagate_fp_types(sdfg, {"B": dace.float32})
    sdfg.validate()

    assert sdfg.arrays["B"].dtype == dace.float32, sdfg.arrays["B"].dtype
    assert sdfg.arrays["A"].dtype == dace.float64

    A = np.arange(1, n + 1, dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)
    sdfg(A=A, C=C)

    # B = (f32)(A*2); C = (f32)(B*2). Small integers are exact in f32, so C == A*4.
    b_ref = (A * 2.0).astype(np.float32)
    c_ref = (b_ref.astype(np.float64) * 2.0).astype(np.float32).astype(np.float64)
    np.testing.assert_allclose(C, c_ref, rtol=1e-6)
    np.testing.assert_allclose(C, A * 4.0, rtol=1e-6)


if __name__ == "__main__":
    test_transient_intermediate_propagates()
    test_all_nontransient_interface_preserved()
    test_mixed_precision_promotes()
    test_map_passthrough()
    test_reduce_node()
    test_initial_type_pinned()
    test_long_chain_convergence()
    test_unconnected_array_unchanged()
    test_interface_copy_in_only_for_inputs()
    test_requires_two_fixpoint_passes()
    test_three_level_lattice()
    test_cyclic_dependency_terminates()
    test_end_to_end_float32_runs()
    test_default_rules_used_and_not_merged()
    test_end_to_end_demoted_written_array_runs()
    print("All tests passed.")
