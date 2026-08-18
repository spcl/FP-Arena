# Copyright 2019-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""Tests for the selection search (knobs, canonical typing, run_search)."""

import dace
import pytest

import fp_arena  # noqa: F401
from fp_arena.experiment import (
    CONSTANTS_KEY,
    ErrorBudget,
    ExperimentConfig,
    KnobSelection,
    ResultStore,
    SelectionSearchConfig,
    find_knobs,
    run_search,
)
from fp_arena.experiment.knobs import canonical_typing

N = dace.symbol("N")


@dace.program
def _axpy(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * b[i] + c[i]


@dace.program
def _scaled(a: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * 2.5


_AXPY = _axpy.to_sdfg(simplify=True)
_SCALED = _scaled.to_sdfg(simplify=True)


def _experiment(sdfg, name="prog", n=64, **inputs):
    return ExperimentConfig(name=name, program=sdfg, symbols={"N": n}, inputs=inputs)


# --------------------------------------------------------------------------- knobs


def test_find_knobs_classifies_inputs_as_sources():
    knobs = {k.name: k for k in find_knobs(_AXPY)}
    # a, b, c are all read inputs (c is read-and-written): sources.
    for name in ("a", "b", "c"):
        assert knobs[name].kind == "source"
        assert knobs[name].original == "fp64"
        assert knobs[name].domain == ("fp16", "fp32", "fp64")
    # No float literals -> no constants knob.
    assert CONSTANTS_KEY not in knobs


def test_find_knobs_detects_constants():
    names = {k.name: k for k in find_knobs(_SCALED)}
    assert CONSTANTS_KEY in names
    assert names[CONSTANTS_KEY].kind == "constants"


def test_knob_selection_toggles_and_overrides():
    default = {k.name for k in find_knobs(_SCALED, KnobSelection())}
    assert CONSTANTS_KEY not in default  # constants off by default
    with_consts = {k.name for k in find_knobs(_SCALED, KnobSelection(constants=True))}
    assert CONSTANTS_KEY in with_consts
    excluded = {
        k.name for k in find_knobs(_SCALED, KnobSelection(exclude=frozenset({"a"})))
    }
    assert "a" not in excluded


def test_domain_capped_at_original():
    @dace.program
    def half(a: dace.float32[N], c: dace.float32[N]):
        for i in dace.map[0:N]:
            c[i] = a[i]

    knobs = {k.name: k for k in find_knobs(half.to_sdfg(simplify=True))}
    assert knobs["a"].domain == ("fp16", "fp32")  # no fp64 rung above fp32


# ------------------------------------------------------------------- canonical key


def test_canonical_typing_dedups_redundant_pins():
    exp = _experiment(_AXPY)
    root = canonical_typing(exp, {})
    # Pinning to the type propagation already gives is redundant -> same key.
    assert canonical_typing(exp, {"a": "fp64"}) == root
    # A genuine lowering is a different key.
    assert canonical_typing(exp, {"a": "fp16"}) != root
    # Constants are part of the key.
    assert canonical_typing(exp, {CONSTANTS_KEY: "fp32"}) != root


# --------------------------------------------------------------------- run_search


def test_search_tight_budget_keeps_root():
    """A budget only fp64 can meet -> the search returns the all-highest root."""
    exp = _experiment(_AXPY)
    # rel_max well below fp32's rounding but above fp64-vs-fp64 (which is exact).
    budget = ErrorBudget(limits={"rel_max": 1e-10})
    result = run_search(
        SelectionSearchConfig(
            experiment=exp,
            budget=budget,
            reference="fp64",
            n_samples=2,
            n_warmup=1,
            n_reps=3,
            objective="total",
        )
    )
    assert not result.unsatisfiable
    # Nothing lowered: every knob at its highest rung.
    highest = {k.name: k.highest for k in result.knobs}
    assert result.best == highest


def test_search_reports_unsatisfiable_when_root_fails():
    """A predicate no config can satisfy -> even the root fails -> unsatisfiable."""
    exp = _experiment(_AXPY)
    budget = ErrorBudget(limits={"rel_max": 1.0}, predicate=lambda errs: False)
    result = run_search(
        SelectionSearchConfig(
            experiment=exp,
            budget=budget,
            reference="fp64",
            n_samples=1,
            n_warmup=1,
            n_reps=2,
        )
    )
    assert result.unsatisfiable
    assert result.best is None


def test_search_worker_count_does_not_change_result():
    """The result is independent of n_workers (1 pool worker vs several)."""
    exp = _experiment(_AXPY)
    budget = ErrorBudget(limits={"rel_max": 1e-10})  # only fp64 passes
    kw = {
        "experiment": exp,
        "budget": budget,
        "reference": "fp64",
        "n_samples": 1,
        "n_warmup": 1,
        "n_reps": 2,
        "objective": "total",
    }
    one = run_search(SelectionSearchConfig(n_workers=1, **kw))
    many = run_search(SelectionSearchConfig(n_workers=4, **kw))
    assert not many.unsatisfiable
    assert many.best == one.best  # nothing lowered, both
    assert many.n_pruned == one.n_pruned


def test_search_records_candidates_and_persists(tmp_path):
    """Feasible configs are recorded (root included) and the result is stored."""
    exp = _experiment(_AXPY)
    budget = ErrorBudget(limits={"rel_max": 1e-10})  # only the root passes
    store = ResultStore(str(tmp_path / "r.db"))
    try:
        result = run_search(
            SelectionSearchConfig(
                experiment=exp,
                budget=budget,
                reference="fp64",
                n_samples=1,
                n_warmup=1,
                n_reps=2,
            ),
            store=store,
        )
        # root is feasible -> at least one candidate, the root (nothing lowered, 1x).
        assert result.candidates
        assert result.candidates[0].precision == {}
        assert result.candidates[0].speedup == 1.0
        rows = store.query(kind="search")
        assert len(rows) == 1
        assert rows[0].payload["objective"] == "total"
        assert len(rows[0].payload["candidates"]) == len(result.candidates)
    finally:
        store.close()


def test_search_no_knobs_raises():
    exp = _experiment(_AXPY)
    with pytest.raises(ValueError, match="No knobs in scope"):
        run_search(
            SelectionSearchConfig(
                experiment=exp,
                budget=ErrorBudget(limits={"rel_max": 1.0}),
                knobs=KnobSelection(sources=False, constants=False, overrides=False),
            )
        )
