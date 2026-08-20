import dace
import pytest
from corpus.heat3d import heat3d_kernel
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


def _const_sdfg(tasklet_code: str, dtype=dace.float32, language=dace.Language.Python):
    """A -> t('x' -> 'y', tasklet_code) -> B, both non-transient, same dtype."""
    sdfg = dace.SDFG("const_test")
    sdfg.add_array("A", [1], dtype, transient=False)
    sdfg.add_array("B", [1], dtype, transient=False)
    s = sdfg.add_state("s")
    t = s.add_tasklet("t", {"x"}, {"y"}, tasklet_code, language=language)
    s.add_edge(s.add_read("A"), None, t, "x", dace.Memlet("A[0]"))
    s.add_edge(t, "y", s.add_write("B"), None, dace.Memlet("B[0]"))
    return sdfg


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


def _conditional_write_sdfg(n=8, dtype=dace.float64):
    """
    ``B[i] = 3*A[i] where A[i] < 0.5``, in the shape the DaCe frontend lowers a
    boolean-masked assignment (``B[I] = ...``) to: the tasklet performs the
    write itself, so its output memlet is dynamic.
    """
    sdfg = dace.SDFG("cond_write")
    sdfg.add_array("A", [n], dtype, transient=False)
    sdfg.add_array("B", [n], dtype, transient=False)

    s = sdfg.add_state("s")
    me, mx = s.add_map("m", {"i": f"0:{n}"})
    t = s.add_tasklet("cond", {"a"}, {"b"}, "if a < 0.5:\n    b = 3.0 * a")
    me.add_in_connector("IN_A")
    me.add_out_connector("OUT_A")
    mx.add_in_connector("IN_B")
    mx.add_out_connector("OUT_B")

    s.add_edge(s.add_read("A"), None, me, "IN_A", dace.Memlet(f"A[0:{n}]"))
    s.add_edge(me, "OUT_A", t, "a", dace.Memlet("A[i]"))
    s.add_edge(t, "b", mx, "IN_B", dace.Memlet("B[i]", dynamic=True))
    s.add_edge(mx, "OUT_B", s.add_write("B"), None, dace.Memlet(f"B[0:{n}]"))
    return sdfg


def test_conditional_write_connector_stays_a_pointer():
    """
    A dynamic output memlet means the tasklet writes the container itself, so
    codegen emits no copy-out; the connector must stay a pointer or the write
    is silently dropped.
    """
    sdfg = _conditional_write_sdfg()
    change_and_propagate_fp_types(sdfg, {"A": dace.float32})

    tasklet = next(
        n
        for s in sdfg.all_states()
        for n in s.nodes()
        if isinstance(n, dace.nodes.Tasklet) and n.label == "cond"
    )
    assert isinstance(tasklet.out_connectors["b"], dace.dtypes.pointer), (
        tasklet.out_connectors["b"]
    )
    assert tasklet.out_connectors["b"].base_type == dace.float32

    sdfg.validate()


def test_conditional_write_survives_retyping():
    """End to end: the masked write still happens after a precision change."""
    import numpy as np

    n = 8
    sdfg = _conditional_write_sdfg(n)
    change_and_propagate_fp_types(sdfg, {"A": dace.float32})
    # B is non-transient, so the fp32 computation happens on an internal
    # transient and the fp64 interface is preserved.
    assert sdfg.arrays["fp_casted_B_float32"].dtype == dace.float32
    assert sdfg.arrays["B"].dtype == dace.float64

    A = np.arange(n, dtype=np.float64) / (n - 1)
    B = np.zeros(n, dtype=np.float64)
    sdfg(A=A, B=B)
    np.testing.assert_allclose(B, np.where(A < 0.5, 3.0 * A, 0.0), rtol=1e-6)


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


def test_interface_copy_in_also_for_outputs():
    """Output-only arrays get a copy-in too: the kernel may overwrite them
    only partially, and untouched regions must round-trip the caller's
    pre-call contents (the pass cannot prove a full overwrite)."""
    import numpy as np

    sdfg = dace.SDFG("output_only")
    sdfg.add_array("A", [1], dace.float32, transient=False)
    sdfg.add_array("B", [2], dace.float32, transient=False)

    s = sdfg.add_state("s")
    t = s.add_tasklet("t", {}, {"b"}, "b = 1.0;", language=dace.Language.CPP)
    # B[0] is written; B[1] is never touched. A is not used.
    s.add_edge(t, "b", s.add_write("B"), None, dace.Memlet("B[0]"))

    change_and_propagate_fp_types(sdfg, {"B": dace.float16})

    # B is non-transient and changed: external B stays f32, casted version is f16.
    assert sdfg.arrays["B"].dtype == dace.float32
    casted_name = "fp_casted_B_float16"
    assert casted_name in sdfg.arrays

    copy_in = next((st for st in sdfg.states() if st.label == "copy_in"), None)
    assert copy_in is not None, "changed interface arrays must be copied in"

    sdfg.validate()

    # The unwritten element survives (round-tripped through f16).
    B = np.array([-3.0, 0.5], dtype=np.float32)
    sdfg(A=np.zeros(1, dtype=np.float32), B=B)
    np.testing.assert_allclose(B, [1.0, 0.5])


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


def test_boundary_cast_inserted_for_fusion():
    """Regression: a fused map with mixed precision compiles via a map-boundary cast."""
    import numpy as np
    from dace.sdfg import nodes
    from dace.transformation.dataflow import MapFusion

    M = dace.symbol("M")

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        B[:] = A * 2.0
        A[:] = B * 3.0

    # Fuse the two statements through a transient holding B's value.
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MapFusion)

    # fp16 transient written into fp32 B -> a cast must be inserted.
    change_and_propagate_fp_types(
        sdfg, {"A": dace.float16, "B": dace.float32}, DEFAULT_PROMOTION_RULES
    )
    sdfg.validate()
    casts = [
        n.label
        for s in sdfg.all_states()
        for n in s.nodes()
        if isinstance(n, nodes.Tasklet) and "map_fusion_B_to_B" in n.label
    ]
    assert casts, "expected a cast on the fused transient's write into B"

    csdfg = sdfg.compile()
    A = np.full(4, 3.0, dtype=np.float64)
    B = np.zeros(4, dtype=np.float64)
    csdfg(A=A, B=B, M=4)
    # B = A*2 = 6, A = B*3 = 18 (exact in fp16/fp32).
    np.testing.assert_allclose(B, 6.0)
    np.testing.assert_allclose(A, 18.0)


def test_constant_type_default_leaves_literals_untouched():
    """Without constant_type (the default), float literals are emitted as-is."""
    sdfg = _const_sdfg("y = x + 0.5")
    change_and_propagate_fp_types(sdfg, {})
    code = sdfg.generate_code()[0].clean_code
    assert "0.5" in code
    assert "float(0.5)" not in code, code


def test_constant_type_casts_float_literals():
    """constant_type wraps every float literal in a functional cast to that type."""
    sdfg = _const_sdfg("y = x + 0.5")
    change_and_propagate_fp_types(sdfg, {}, constant_type=dace.float32)
    code = sdfg.generate_code()[0].clean_code
    assert "float(0.5)" in code, code

    sdfg.validate()
    sdfg.compile()


def test_constant_type_leaves_int_literals():
    """Only float literals are cast; integer literals are left alone."""
    sdfg = _const_sdfg("y = x * 2 + 0.5")
    change_and_propagate_fp_types(sdfg, {}, constant_type=dace.float32)
    code = sdfg.generate_code()[0].clean_code
    assert "float(0.5)" in code, code
    assert "float(2)" not in code, code


def test_constant_type_only_touches_python_tasklets():
    """A C++ tasklet's constants are left untouched (only Python bodies are rewritten)."""
    sdfg = _const_sdfg("y = x + 0.5;", language=dace.Language.CPP)
    change_and_propagate_fp_types(sdfg, {}, constant_type=dace.float32)
    code = sdfg.generate_code()[0].clean_code
    assert "0.5" in code
    assert "float(0.5)" not in code, code


def test_constant_type_end_to_end_float32_precision():
    """The wrapped constant is evaluated in the requested precision at runtime."""
    import numpy as np

    n = 64

    @dace.program
    def prog(A: dace.float64[n], B: dace.float64[n]):
        B[:] = A + 0.1

    sdfg = prog.to_sdfg(simplify=True)
    change_and_propagate_fp_types(
        sdfg, {"A": dace.float32, "B": dace.float32}, constant_type=dace.float32
    )
    code = sdfg.generate_code()[0].clean_code
    assert "float(0.1)" in code, code

    rng = np.random.default_rng(0)
    A = rng.random(n).astype(np.float64)
    B = np.zeros(n, dtype=np.float64)
    sdfg(A=A, B=B)

    # Generated code adds the f32 constant to the f32 input in f32 arithmetic.
    ref = (A.astype(np.float32) + np.float32(0.1)).astype(np.float64)
    np.testing.assert_array_equal(B, ref)


def test_heat3d_no_fp64_in_generated_code():
    """heat3d (corpus/heat3d.py) lowered from fp64 to fp16: the generated
    C++ contains fp64 only at the preserved A/B interface."""
    import re

    sdfg = heat3d_kernel.to_sdfg(simplify=True)
    change_and_propagate_fp_types(
        sdfg,
        {"A": dace.float16, "B": dace.float16},
        constant_type=dace.float16,
    )
    sdfg.validate()

    code_objects = sdfg.generate_code()

    # The computation is in fp16.
    assert any("dace::float16" in co.clean_code for co in code_objects), (
        "expected dace::float16 in the generated code"
    )

    # fp64 only on interface lines: A/B references and interface cast tasklets.
    allowed = re.compile(r"\b[AB]\b|\bdouble\s+_out;|=\s*double\(_in\)")
    leaks = [
        f"[{co.title}] {line.strip()}"
        for co in code_objects
        for line in co.clean_code.splitlines()
        if "double" in line and not allowed.search(line)
    ]
    assert not leaks, "fp64 leaked into the computation:\n" + "\n".join(leaks)


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


def test_boundary_cast_inserted_for_fusion():
    """Regression: a fused map with mixed precision compiles via a map-boundary cast."""
    import numpy as np
    from dace.sdfg import nodes
    from dace.transformation.dataflow import MapFusion

    M = dace.symbol("M")

    @dace.program
    def prog(A: dace.float64[M], B: dace.float64[M]):
        B[:] = A * 2.0
        A[:] = B * 3.0

    # Fuse the two statements through a transient holding B's value.
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(MapFusion)

    # fp16 transient written into fp32 B -> a cast must be inserted.
    change_and_propagate_fp_types(
        sdfg, {"A": dace.float16, "B": dace.float32}, DEFAULT_PROMOTION_RULES
    )
    sdfg.validate()
    casts = [
        n.label
        for s in sdfg.all_states()
        for n in s.nodes()
        if isinstance(n, nodes.Tasklet) and "map_fusion_B_to_B" in n.label
    ]
    assert casts, "expected a cast on the fused transient's write into B"

    csdfg = sdfg.compile()
    A = np.full(4, 3.0, dtype=np.float64)
    B = np.zeros(4, dtype=np.float64)
    csdfg(A=A, B=B, M=4)
    # B = A*2 = 6, A = B*3 = 18 (exact in fp16/fp32).
    np.testing.assert_allclose(B, 6.0)
    np.testing.assert_allclose(A, 18.0)


def test_direct_copy_cast_inserted():
    """A direct AccessNode->AccessNode copy between differently-typed arrays
    is replaced by an elementwise cast map (DaCe cannot lower a mixed-dtype
    copy), including subset copies with different src/dst offsets."""
    import numpy as np
    from dace.sdfg import nodes

    n = 6
    sdfg = dace.SDFG("copy_cast")
    sdfg.add_array("A", [n], dace.float64, transient=False)
    sdfg.add_array("B", [n], dace.float64, transient=False)

    s = sdfg.add_state("s")
    # Copy A[1:4] into B[2:5] (offsets differ on the two sides).
    mem = dace.Memlet(
        data="A",
        subset=dace.subsets.Range([(1, 3, 1)]),
        other_subset=dace.subsets.Range([(2, 4, 1)]),
    )
    s.add_edge(s.add_read("A"), None, s.add_write("B"), None, mem)

    change_and_propagate_fp_types(sdfg, {"A": dace.float64, "B": dace.float32})
    sdfg.validate()

    casts = [
        node.label
        for st in sdfg.all_states()
        for node in st.nodes()
        if isinstance(node, nodes.Tasklet) and node.label.startswith("cast_copy_")
    ]
    assert casts, "expected a cast map on the mixed-dtype direct copy"

    A = np.arange(1, n + 1, dtype=np.float64) * 1.1
    B = np.zeros(n, dtype=np.float64)
    sdfg(A=A, B=B)

    ref = np.zeros(n)
    ref[2:5] = A[1:4].astype(np.float32)
    np.testing.assert_allclose(B, ref, rtol=1e-7)


def test_direct_copy_cast_degenerate_dims():
    """Copies whose degenerate (size-1) dimensions do not line up positionally
    -- a row copied into a column, and a rank-changing single element -- pair
    the non-degenerate dimensions instead of zipping dims positionally."""
    import numpy as np

    n = 4
    sdfg = dace.SDFG("copy_cast_degenerate")
    sdfg.add_array("A", [n, n], dace.float64, transient=False)
    sdfg.add_array("B", [n, n], dace.float64, transient=False)
    sdfg.add_array("C", [n], dace.float64, transient=False)

    s = sdfg.add_state("s")
    # Row A[1, 0:n] into column B[0:n, 2]: degenerate dims transposed.
    s.add_edge(
        s.add_read("A"),
        None,
        s.add_write("B"),
        None,
        dace.Memlet(
            data="A",
            subset=dace.subsets.Range([(1, 1, 1), (0, n - 1, 1)]),
            other_subset=dace.subsets.Range([(0, n - 1, 1), (2, 2, 1)]),
        ),
    )
    # Rank-changing single element A[3, 3] -> C[1].
    s2 = sdfg.add_state_after(s, "s2")
    s2.add_edge(
        s2.add_read("A"),
        None,
        s2.add_write("C"),
        None,
        dace.Memlet(
            data="A",
            subset=dace.subsets.Range([(3, 3, 1), (3, 3, 1)]),
            other_subset=dace.subsets.Range([(1, 1, 1)]),
        ),
    )

    change_and_propagate_fp_types(
        sdfg, {"A": dace.float64, "B": dace.float32, "C": dace.float32}
    )
    sdfg.validate()

    A = (np.arange(n * n, dtype=np.float64) * 1.1).reshape(n, n).copy()
    B = np.zeros((n, n), dtype=np.float64)
    C = np.zeros(n, dtype=np.float64)
    sdfg(A=A, B=B, C=C)

    ref_b = np.zeros((n, n))
    ref_b[0:n, 2] = A[1, 0:n].astype(np.float32)
    np.testing.assert_allclose(B, ref_b, rtol=1e-7)
    ref_c = np.zeros(n)
    ref_c[1] = np.float32(A[3, 3])
    np.testing.assert_allclose(C, ref_c, rtol=1e-7)


# ---------------------------------------------------------------------------
# NestedSDFG tests
# ---------------------------------------------------------------------------


def test_nested_uniform_pin_no_boundary_casts():
    """Pinning both outer arrays retypes the nested level through the boundary,
    without inserting any cast at the NestedSDFG boundary (unification), and
    the result is numerically correct through the preserved fp64 interface."""
    import numpy as np
    from dace.sdfg import nodes

    @dace.program
    def _inner_double(x: dace.float64[8], y: dace.float64[8]):
        for i in dace.map[0:8]:
            y[i] = x[i] * 2.0

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8]):
        _inner_double(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    nsdfg = next(
        n
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.NestedSDFG)
    )

    change_and_propagate_fp_types(sdfg, {"a": dace.float32, "b": dace.float32})
    sdfg.validate()

    # Inner arrays (named x/y, not a/b) follow the outer pins via unification.
    assert nsdfg.sdfg.arrays["x"].dtype == dace.float32
    assert nsdfg.sdfg.arrays["y"].dtype == dace.float32
    # Interface preserved at the root.
    assert sdfg.arrays["a"].dtype == dace.float64
    assert sdfg.arrays["b"].dtype == dace.float64

    # No casts at any nested level; root casts only in copy_in/copy_out.
    inner_casts = [
        n.label
        for sd in sdfg.all_sdfgs_recursive()
        if sd is not sdfg
        for st in sd.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.label.startswith("cast_")
    ]
    assert not inner_casts, inner_casts
    outer_casts_elsewhere = [
        n.label
        for st in sdfg.all_states()
        if not st.label.startswith(("copy_in", "copy_out"))
        for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.label.startswith("cast_")
    ]
    assert not outer_casts_elsewhere, outer_casts_elsewhere

    a = np.arange(1, 9, dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    sdfg(a=a, b=b)
    np.testing.assert_allclose(b, a * 2.0)


def test_nested_pin_propagates_through_boundary():
    """A single outer pin flows into the nested computation and back out to
    the (unpinned) outer output through the out-connector."""
    import numpy as np
    from dace.sdfg import nodes

    @dace.program
    def _inner_double(x: dace.float64[8], y: dace.float64[8]):
        for i in dace.map[0:8]:
            y[i] = x[i] * 2.0

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8]):
        _inner_double(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    nsdfg = next(
        n
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.NestedSDFG)
    )

    change_and_propagate_fp_types(sdfg, {"a": dace.float16})
    sdfg.validate()

    # x is unified with a (pinned); y derives f16 from x; b is unified with y.
    assert nsdfg.sdfg.arrays["x"].dtype == dace.float16
    assert nsdfg.sdfg.arrays["y"].dtype == dace.float16
    assert sdfg.arrays["fp_casted_b_float16"].dtype == dace.float16

    a = np.arange(1, 9, dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    sdfg(a=a, b=b)
    np.testing.assert_allclose(b, a * 2.0)  # small ints: exact in fp16


def test_nested_inout_connector():
    """An in-place update (inout connector) unifies the inner array with both
    the outer input and output; the update runs at the pinned precision."""
    import numpy as np
    from dace.sdfg import nodes

    @dace.program
    def _inner_acc(x: dace.float64[8]):
        for i in dace.map[0:8]:
            x[i] = x[i] + 1.0

    @dace.program
    def prog(a: dace.float64[8]):
        _inner_acc(a)

    sdfg = prog.to_sdfg(simplify=False)
    nsdfg = next(
        n
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.NestedSDFG)
    )
    assert nsdfg.in_connectors.keys() & nsdfg.out_connectors.keys() == {"x"}

    change_and_propagate_fp_types(sdfg, {"a": dace.float32})
    sdfg.validate()

    assert nsdfg.sdfg.arrays["x"].dtype == dace.float32
    assert sdfg.arrays["a"].dtype == dace.float64  # interface preserved

    a = np.arange(1, 9, dtype=np.float64)
    sdfg(a=a)
    np.testing.assert_allclose(a, np.arange(2, 10, dtype=np.float64))


def test_nested_mixed_precision_promotes():
    """f16 and f32 outer pins meeting inside a nested SDFG promote its output
    (and the unified outer output) to f32."""
    from dace.sdfg import nodes

    @dace.program
    def _inner_add(x: dace.float64[8], y: dace.float64[8], z: dace.float64[8]):
        for i in dace.map[0:8]:
            z[i] = x[i] + y[i]

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8], c: dace.float64[8]):
        _inner_add(a, b, c)

    sdfg = prog.to_sdfg(simplify=False)
    nsdfg = next(
        n
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.NestedSDFG)
    )

    change_and_propagate_fp_types(sdfg, {"a": dace.float16, "b": dace.float32})
    sdfg.validate()

    assert nsdfg.sdfg.arrays["x"].dtype == dace.float16
    assert nsdfg.sdfg.arrays["y"].dtype == dace.float32
    assert nsdfg.sdfg.arrays["z"].dtype == dace.float32, nsdfg.sdfg.arrays["z"].dtype
    assert sdfg.arrays["fp_casted_c_float32"].dtype == dace.float32


def test_nested_two_levels():
    """Pins propagate through two levels of nesting."""
    import numpy as np
    from dace.sdfg import nodes

    @dace.program
    def _lvl2(x: dace.float64[8]):
        for i in dace.map[0:8]:
            x[i] = x[i] * 2.0

    @dace.program
    def _lvl1(x: dace.float64[8]):
        _lvl2(x)

    @dace.program
    def prog(a: dace.float64[8]):
        _lvl1(a)

    sdfg = prog.to_sdfg(simplify=False)

    change_and_propagate_fp_types(sdfg, {"a": dace.float32})
    sdfg.validate()

    # The frontend adds a wrapper level per call; regardless of depth, no
    # fp64 array may survive anywhere below the root.
    inner_sdfgs = [sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg]
    assert len(inner_sdfgs) >= 2
    leaks = [
        (sd.name, name)
        for sd in inner_sdfgs
        for name, desc in sd.arrays.items()
        if desc.dtype == dace.float64
    ]
    assert not leaks, leaks

    a = np.arange(1, 9, dtype=np.float64)
    sdfg(a=a)
    np.testing.assert_allclose(a, np.arange(1, 9, dtype=np.float64) * 2.0)


def test_nested_constant_type_reaches_inner_tasklets():
    """constant_type rewrites float literals inside nested Python tasklets."""
    from dace.sdfg import nodes

    @dace.program
    def _inner_double(x: dace.float64[8], y: dace.float64[8]):
        for i in dace.map[0:8]:
            y[i] = x[i] * 2.0

    @dace.program
    def prog(a: dace.float64[8], b: dace.float64[8]):
        _inner_double(a, b)

    sdfg = prog.to_sdfg(simplify=False)
    nsdfg = next(
        n
        for st in sdfg.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.NestedSDFG)
    )

    change_and_propagate_fp_types(
        sdfg, {"a": dace.float32, "b": dace.float32}, constant_type=dace.float32
    )

    # The literal lives in the deepest nesting level (the frontend adds a
    # wrapper level per call), so search every level below the root.
    inner_code = "\n".join(
        n.code.as_string
        for sd in sdfg.all_sdfgs_recursive()
        if sd is not sdfg
        for st in sd.all_states()
        for n in st.nodes()
        if isinstance(n, nodes.Tasklet) and n.language == dace.Language.Python
    )
    assert "dace.float32(2.0)" in inner_code, inner_code


def test_shared_nested_sdfg_rejected():
    """One inner SDFG object referenced by two NestedSDFG nodes must be
    rejected, not silently retyped at both call sites."""
    inner = dace.SDFG("shared_inner")
    inner.add_array("x", [1], dace.float64, transient=False)
    inner.add_array("y", [1], dace.float64, transient=False)
    ist = inner.add_state("s")
    it = ist.add_tasklet("t", {"xi"}, {"yo"}, "yo = xi")
    ist.add_edge(ist.add_read("x"), None, it, "xi", dace.Memlet("x[0]"))
    ist.add_edge(it, "yo", ist.add_write("y"), None, dace.Memlet("y[0]"))

    outer = dace.SDFG("shared_outer")
    for name in ("a", "b", "c"):
        outer.add_array(name, [1], dace.float64, transient=False)
    st = outer.add_state("s")
    for src, dst in (("a", "b"), ("b", "c")):
        node = st.add_nested_sdfg(inner, {"x"}, {"y"})
        st.add_edge(st.add_read(src), None, node, "x", dace.Memlet(f"{src}[0]"))
        st.add_edge(node, "y", st.add_write(dst), None, dace.Memlet(f"{dst}[0]"))

    with pytest.raises(NotImplementedError, match="referenced by more than one"):
        change_and_propagate_fp_types(outer, {"a": dace.float32})


def _doubling_maps(names, n=8, dtype=dace.float64):
    """A chain of one-map states, each ``dst[i] = src[i] * 2.0``, over *names*.

    The first and last array are non-transient, the rest transient.
    """
    sdfg = dace.SDFG("doubling")
    for i, name in enumerate(names):
        sdfg.add_array(name, [n], dtype, transient=0 < i < len(names) - 1)
    prev_state = None
    for src, dst in zip(names, names[1:]):
        st = sdfg.add_state(f"s_{src}_{dst}")
        if prev_state is not None:
            sdfg.add_edge(prev_state, st, dace.InterstateEdge())
        prev_state = st
        me, mx = st.add_map("m", {"i": f"0:{n}"})
        t = st.add_tasklet("t", {"x"}, {"y"}, "y = x * 2.0")
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
    return sdfg


def test_narrowing_write_becomes_an_explicit_cast():
    """A tasklet that computes in f64 but stores into an f32 array keeps the
    narrowing in a cast tasklet of its own, not inside the body."""
    import numpy as np
    from dace.sdfg import nodes

    n = 8
    sdfg = _doubling_maps(["A", "B", "C"], n=n)
    change_and_propagate_fp_types(sdfg, {"B": dace.float32})
    sdfg.validate()

    # The body now writes an f64 transient; a cast tasklet stores it into B.
    assert sdfg.arrays["B"].dtype == dace.float32
    assert "B_wide" in sdfg.arrays, sorted(sdfg.arrays)
    assert sdfg.arrays["B_wide"].dtype == dace.float64
    assert sdfg.arrays["B_wide"].transient

    tasklets = [(s, n_) for s in sdfg.all_states() for n_ in s.nodes()
                if isinstance(n_, nodes.Tasklet)]
    bodies = [n_ for s, n_ in tasklets
              if any(e.data.data == "B_wide" for e in s.out_edges(n_))]
    assert len(bodies) == 1, [n_.label for n_ in bodies]
    assert bodies[0].out_connectors["y"] == dace.float64

    casts = [n_ for _, n_ in tasklets if n_.label == "cast_B_wide_to_B"]
    assert len(casts) == 1, [n_.label for _, n_ in tasklets]
    assert "float32" in casts[0].code.as_string, casts[0].code.as_string

    # Rounding happens exactly where it did before: once, on the store into B.
    A = np.arange(1, n + 1, dtype=np.float64) / 3.0
    C = np.zeros(n, dtype=np.float64)
    sdfg(A=A, C=C)
    b_ref = (A * 2.0).astype(np.float32)
    c_ref = (b_ref.astype(np.float64) * 2.0).astype(np.float32)
    np.testing.assert_array_equal(C, c_ref.astype(np.float64))


def test_widening_write_is_left_alone():
    """A pin *above* the computation's precision needs no cast node: storing an
    f16 result into an f64 array is a widening the backend does implicitly."""
    sdfg = _doubling_maps(["A", "B"])
    change_and_propagate_fp_types(sdfg, {"A": dace.float16, "B": dace.float64})
    sdfg.validate()

    assert not [name for name in sdfg.arrays if name.endswith("_wide")], (
        f"no cast transient expected for a widening store: {sorted(sdfg.arrays)}"
    )


def test_narrowing_write_survives_gpu_vectorization():
    """Regression (heat3d 'overrides' search): DaCe's GPU vectorizer turns a
    tasklet into a typed tile op, whose operands may only widen to the output
    dtype. An implicit narrowing inside the body fails its validation."""
    from fp_arena.experiment.retarget import apply_target

    sdfg = _doubling_maps(["A", "B", "C"])
    change_and_propagate_fp_types(sdfg, {"B": dace.float32})
    apply_target(sdfg, "gpu", gpu_block_size=(256, 1, 1), gpu_vectorize=True)
    sdfg.validate()


if __name__ == "__main__":
    test_transient_intermediate_propagates()
    test_all_nontransient_interface_preserved()
    test_mixed_precision_promotes()
    test_map_passthrough()
    test_conditional_write_connector_stays_a_pointer()
    test_conditional_write_survives_retyping()
    test_reduce_node()
    test_initial_type_pinned()
    test_long_chain_convergence()
    test_unconnected_array_unchanged()
    test_interface_copy_in_also_for_outputs()
    test_requires_two_fixpoint_passes()
    test_three_level_lattice()
    test_cyclic_dependency_terminates()
    test_end_to_end_float32_runs()
    test_default_rules_used_and_not_merged()
    test_end_to_end_demoted_written_array_runs()
    test_boundary_cast_inserted_for_fusion()
    test_constant_type_default_leaves_literals_untouched()
    test_constant_type_casts_float_literals()
    test_constant_type_leaves_int_literals()
    test_constant_type_only_touches_python_tasklets()
    test_constant_type_end_to_end_float32_precision()
    test_heat3d_no_fp64_in_generated_code()
    test_end_to_end_demoted_written_array_runs()
    test_boundary_cast_inserted_for_fusion()
    test_direct_copy_cast_inserted()
    test_direct_copy_cast_degenerate_dims()
    test_nested_uniform_pin_no_boundary_casts()
    test_nested_pin_propagates_through_boundary()
    test_nested_inout_connector()
    test_nested_mixed_precision_promotes()
    test_nested_two_levels()
    test_nested_constant_type_reaches_inner_tasklets()
    test_shared_nested_sdfg_rejected()
    test_narrowing_write_becomes_an_explicit_cast()
    test_widening_write_is_left_alone()
    test_narrowing_write_survives_gpu_vectorization()
    print("All tests passed.")
